% Disaggregation

大语言模型在线推理通常包含两个计算阶段：

- **Prefill**：并行处理用户输入的 prompt，建立上下文状态；
- **Decode**：逐 token 生成输出，反复读取上下文状态。

两者的硬件特征不同。Prefill 的有效 token 数较多，更接近大矩阵乘；Decode 每次通常只处理每个请求的一个 token，更容易受显存带宽、KV Cache 访问和 batch 规模影响。因此，SGLang 的 PD disaggregation 可以把两种阶段放到不同的 worker 上，使 Prefill 和 Decode 分别扩容、分别调度。

这里的 PD 指 `Prefill-Decode`，不是把一次请求简单地切成两个互不相关的模型副本。Prefill 必须把 Decode 继续推理所需的状态交给 Decode。

### 首 token 的归属

设 prompt 为

\[
X = (x_0, x_1, \ldots, x_{L-1})
\]

Target 模型对 prompt 做一次并行 forward，得到各位置 hidden state：

\[
H_0, H_1, \ldots, H_{L-1}
\]

最后一个位置经过 LM Head 得到 logits：

\[
\operatorname{logits}_{L-1} = H_{L-1}W_{\mathrm{lm}}^{\mathsf T}
\]

再依据 temperature、top-\(p\)、top-\(k\) 等 sampling 参数采样出首个生成 token：

\[
y_0 \sim p_\theta(\cdot \mid x_0,\ldots,x_{L-1})
\]

因此，在 SGLang 典型的 PD 路径中，需要区分两个层次：

| 事项 | 归属 |
| --- | --- |
| Prompt forward、最后位置 logits 和首 token sampling | Prefill |
| 首 token 的请求状态接管、提交和输出流归属 | Decode |
| 首 token 之后的增量生成 | Decode |

所以“首 token 在 Decode 生成”这个说法容易产生歧义。更准确的说法是：**首 token 通常在 Prefill 侧已经完成 logits 计算和 sampling，Decode 侧接收并提交它，然后继续生成后续 token。**

### 为什么 Decode 需要首 token

假设 prompt 为

```text
[A, B, C]
```

Prefill 计算 `C` 位置的 logits 并采样出 `D`：

```text
Prefill:
    KV[A, B, C]
    logits(C) -> sample D
```

此时模型还没有执行 `D` 位置的 forward。Decode 接收 `D` 后，会把它作为下一次 forward 的输入，用于生成 `E`：

```text
Decode:
    input = D
    KV = KV[A, B, C]
    forward(D) -> sample E
```

因此，PD 的边界状态可以表示为：

```text
已传输的状态：KV[A, B, C]
已采样的输出：D
Decode 下一次输入：D
Decode 下一次新生成：E
```

通常，Prefill 产生的 handoff token `D` 尚未拥有由 `D` 位置 forward 产生的 KV；`D` 的 KV 会在 Decode 消费它时写入 Decode 本地的 KV Cache。

### P 到 D 需要传输什么

#### Target KV Cache

最核心的数据是 prompt 对应的 Target KV Cache：

\[
\mathcal{K}_{\mathrm{P}} = \{K_\ell(X),V_\ell(X)\}_{\ell=0}^{N-1}
\]

Decode 后续每一步需要访问：

\[
\mathcal{K}_{t} = [\mathcal{K}_{\mathrm{prompt}},
\mathcal{K}_{y_0},\ldots,\mathcal{K}_{y_{t-1}}]
\]

SGLang 实际传输的通常是 paged KV Cache 的 page 或 block，而不是抽象意义上的一个连续 tensor。除数据本身外，还需要处理：

- source 和 destination 的 page 映射；
- page size、KV dtype 和层布局；
- TP、PP、DCP 等并行维度；
- bootstrap room、transfer status 和完成信号；
- 特定模型的 MLA、SWA、Mamba 或其他 state。

代码入口主要位于 `python/sglang/srt/disaggregation/prefill.py`、`python/sglang/srt/disaggregation/decode.py` 和 `python/sglang/srt/disaggregation/utils.py`。

#### Handoff token

Prefill 侧从 `result.next_token_ids` 得到首 token，并通过 disaggregation metadata 交给 Decode。当前代码中，Prefill 侧会把它写入 metadata buffer：

```python
self.output_ids[req.metadata_buffer_index][0] = req.output_ids[0]
```

Decode 侧读取后，将其提交到 Decode request：

```python
decode_req.req.output_ids.append(committed_output_id)
```

这也是“sampling 在 Prefill、output state 在 Decode”的具体体现。

#### Logprob、sampling 和统计 metadata

如果请求 `return_logprob=True`，首 token 的 logprob 也需要从 Prefill 传给 Decode，因为 Decode 没有重新计算该 token 的 prompt-last-position logits。相关 metadata 可能包括：

- `output_token_logprobs_val` 和 `output_token_logprobs_idx`；
- top-logprobs；
- sampling mask 和对应 logprob；
- `cached_tokens` 及 device、host、storage 分项；
- bootstrap room 和 transfer 校验信息。

多模态请求还需要传递 image、audio、video token 数量。SGLang 当前复用 `cached_tokens` buffer 的后续槽位保存这些计数。

#### 非 Transformer 状态

对于混合或非标准 attention 模型，PD 传输的概念不应局限于 KV。还可能包括：

- Mamba recurrent state；
- convolution state；
- Gated Linear Attention 的 recurrent state；
- sliding-window attention state；
- DeepSeek V4 等模型的压缩 attention state。

更一般地说，P 需要传输的是让 D 能够从边界继续增量推理的全部上下文状态。

### 开启 MTP 后的边界

MTP 或 speculative decoding 的目标不是直接生成并提交多个最终 token，而是让 draft 模型提出候选，再由 Target 一次性验证。

设一次 speculative iteration 最多提出四个候选：

\[
(d_1,d_2,d_3,d_4)
\]

它们来自 draft 分布：

\[
d_i \sim q_\phi(\cdot \mid X,y_0,d_1,\ldots,d_{i-1})
\]

Target 随后计算验证 logits，并决定接受长度 \(a\)：

\[
0 \leq a \leq 4
\]

最终可能提交前 \(a\) 个 draft token 和一个 bonus token，而不是固定提交四个 token。

因此，`draft_tokens = 4` 的含义是“一轮最多准备四个候选 token”，不是“Prefill 必须提前生成四个已经确定且全部有效的输出 token”。

### MTP Prefill 做什么

SGLang 的 MTP/EAGLE 类路径在 Target Prefill 后，会调用类似下面的 draft extend：

```python
batch_output.next_draft_input = draft_worker._draft_extend_for_prefill(
    batch,
    batch_output.logits_output.hidden_states,
    batch_output.next_token_ids,
)
```

以 `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py` 为例，MTP prefill 会完成以下工作：

1. 使用 Target 对 prompt 的 hidden states 初始化 draft/MTP 边界；
2. 使用 Target 的 `next_token_ids` 作为 handoff 或 bonus token；
3. 按 MTP depth 执行 draft runner；
4. 产生第一轮候选的 `topk_p` 和 `topk_index`；
5. 根据模型结构保存后续 draft 所需的 hidden state；
6. 如果该实现拥有独立 draft KV pool，则初始化 prompt 对应的 draft KV/state。

多层 MTP 的抽象流程是：

```text
Target prompt forward
        │
        ├── target KV[A ... prompt_last]
        ├── target handoff token y0
        └── target hidden state
                │
                ▼
        MTP/draft extend step 0 -> d1 candidate
                │
                ▼
        MTP/draft extend step 1 -> d2 candidate
                │
                ▼
        MTP/draft extend step 2 -> d3 candidate
                │
                ▼
        MTP/draft extend step 3 -> d4 candidate
```

对于 chain-style MTP，上一个 MTP step 的 hidden state 会作为下一个 step 的输入；非 chain-style MTP 可能只需要 Target 的最后 hidden state。SGLang 代码中分别使用 `CaptureHiddenMode.FULL` 或 `CaptureHiddenMode.LAST` 等模式处理这种差异。

### MTP draft KV 的准确理解

#### Prompt 部分的 draft KV

如果 MTP/draft 模型拥有独立的 draft KV Cache，那么 Prefill 可以建立 prompt 对应的 draft context：

\[
\mathcal{K}_{\mathrm{D,prompt}}
= \{K^{\mathrm{draft}}_\ell(X),V^{\mathrm{draft}}_\ell(X)\}
\]

这部分对应已经确定的 prompt token，因此可以在 Prefill 阶段计算并缓存。

#### 候选 token 部分的 draft KV

对于未来的候选 token，Prefill 只能在 draft loop 中逐步提出。候选是否被 Target 接受尚未确定，因此它们对应的 draft KV 通常随着 Decode 阶段的 draft forward 动态追加：

\[
\mathcal{K}_{\mathrm{draft},t}
= [\mathcal{K}_{\mathrm{draft,prompt}},
\mathcal{K}_{d_1},\ldots,\mathcal{K}_{d_t}]
\]

验证后只保留 accepted prefix，并回收 rejected suffix。这是 speculative decoding 能够回滚的基础。

因此，对“`draft_tokens = 4` 时，Prefill 是否只生成 prompt tokens 对应的 draft KV？”的回答是：

> 对于有独立 draft KV 的实现，Prefill 的 draft KV 主要是 prompt/context 部分；但 Prefill 还会执行 MTP/draft extend，产生第一轮候选的 token、top-k 概率、hidden state 等。未来候选 token 的增量 KV 通常由 Decode 侧动态建立、验证和回滚。

### 两种 PD 加 MTP 部署方式

#### MTP 只在 Decode 侧启用

一种常见部署方式是 Prefill worker 不运行完整 MTP draft worker：

```text
Prefill:
    Target prompt forward
    传 Target KV
    传 handoff token
    传 MTP bootstrap 所需的 hidden state 或 metadata

Decode:
    接收 Target KV
    在本地运行 MTP draft
    产生 d1, d2, d3, d4
    Target verify 并提交 accepted tokens
```

这种方式下 P 到 D 通常不传 draft KV。SGLang 的 `init_disaggregation` 附近有注释指出，某些配置只在 Decode 节点启用 MTP，因此不在 P/D 之间传 draft KV。

#### P/D 两侧都启用 MTP

如果两侧都有 draft worker，并且 transfer 配置包含 draft pool，则可以传输 draft KV：

```text
Prefill:
    Target KV[prompt]
    Draft KV[prompt]
    handoff token
    draft proposal metadata
    hidden state

Decode:
    接收并绑定两类 KV
    继续 draft forward
    Target verify
    接受 prefix，回滚拒绝 suffix
```

SGLang 的 `PrefillBootstrapQueue` 和 Decode 侧的 KV manager 会在存在 `draft_token_to_kv_pool` 时，把 draft KV buffer 追加到 transfer 参数中。相关代码包括：

- `python/sglang/srt/disaggregation/prefill.py`；
- `python/sglang/srt/disaggregation/decode.py`；
- `python/sglang/srt/disaggregation/utils.py` 中的 `get_transfer_draft_kv_layer_ids`；
- `python/sglang/srt/managers/scheduler.py` 中的 draft KV pool 初始化。

是否真的传输 draft KV，取决于 P/D 两侧是否启用对应 draft worker、speculative 配置是否一致，以及 KV layout 是否兼容，不能仅根据“开启了 MTP”这一条件判断。

### 一次完整的 PD 加 MTP 流程

设 prompt 为 \(X\)，Target 首 token 为 \(y_0\)，一轮最多四个 draft token。

#### Prefill 侧

\[
H^T(X),\quad
\mathcal{K}_{T}(X),\quad
y_0 \sim p_\theta(\cdot\mid X)
\]

随后执行 draft extend：

\[
(d_1,d_2,d_3,d_4),
\quad \text{top-}k\text{ scores},
\quad \text{draft hidden states}
\]

如果启用 draft KV，则还建立：

\[
\mathcal{K}_{D}(X)
\]

#### PD 传输

最小逻辑 payload 可以表示为：

\[
\mathcal{P}_{P\rightarrow D} =
\left(
\mathcal{K}_{T}(X),
y_0,
\text{transfer metadata}
\right)
\]

MTP bootstrap 或两侧都启用 draft 时，payload 还可能扩展为：

\[
\mathcal{P}_{P\rightarrow D} =
\left(
\mathcal{K}_{T}(X),
\mathcal{K}_{D}(X),
 y_0,
\text{draft candidates},
\text{hidden states},
\text{sampling/logprob metadata}
\right)
\]

#### Decode 侧

Decode 首先提交 `y0`，然后用它作为输入继续运行：

\[
\mathcal{K}_{T}(X) + y_0
\xrightarrow{\text{Target decode}}
 y_1
\]

同时，MTP draft 产生候选并由 Target 验证：

\[
(d_1,d_2,d_3,d_4)
\xrightarrow{\text{Target verify}}
(a,\text{bonus token})
\]

最终只提交验证通过的 token 序列。

### 常见误区

#### 把首 token 和 draft token 混为一谈

Target 的首个 handoff token 是正式生成边界的一部分；MTP 的 draft token 是等待 Target 验证的候选。前者可以直接成为用户输出，后者必须经过 verify。

#### 认为 Prefill 只传 KV

只传 KV 不足以让 Decode 无额外计算地接管请求。至少还需要 handoff token，以及 transfer metadata；logprob、多模态统计、hidden state 和非 Transformer state 则取决于请求和模型。

#### 认为四个 draft token 都会被提交

`draft_tokens = 4` 只是候选窗口上限。实际接受长度由 Target verify 决定：

\[
0 \leq a \leq 4
\]

被拒绝的后缀需要回滚其 cache/state。

#### 认为 MTP 必然需要传 draft KV

MTP 可以只在 Decode 侧运行，此时 P 到 D 主要传 Target KV 和 MTP bootstrap 信息。只有在 Prefill 侧也建立了 draft pool，并且 transfer backend 配置包含它时，才会传输 draft KV。

### 总结

```text
Prefill = 计算 prompt 上下文 + 建立状态 + 采样 handoff token
Decode  = 接管 handoff token + 增量生成 + 管理后续输出
```

开启 MTP 后：

```text
Prefill = Target prompt forward
        + 可选的 MTP/draft extend
        + 第一轮 draft proposal/bootstrap

Decode  = 继续 draft
        + Target verify
        + 接受或回滚候选
        + 输出最终 token
```

对于 `draft_tokens = 4`，Prefill 的核心不是提前生成四个确定的最终 token，而是初始化 prompt 上下文、生成第一轮 draft proposal，并根据具体实现决定是否建立或传输 prompt 对应的 draft KV。候选 token 的动态 KV、验证、接受和回滚主要发生在 Decode 侧。


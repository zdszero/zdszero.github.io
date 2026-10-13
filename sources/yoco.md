% YOCO

YOCO（You Only Cache Once）是 Microsoft Research 在 2024 年提出的 Decoder-Decoder 架构。它没有把 Decoder-only Transformer 的每一层都设计成独立的全局 Self-Attention，而是将模型拆成两个阶段：前半部分是 `Self-Decoder`，负责编码历史并生成一份全局 KV 表示；后半部分是 `Cross-Decoder`，只生成各层自己的 Query，并通过 Cross-Attention 反复读取这份共享的 KV 表示。

这一区分很重要：YOCO 不是简单地对 KV Cache 做量化或压缩，也不是把普通层的 KV 张量复制给其他层。它改变的是 KV 的生成和所有权关系，从“每层一份”改成“Self-Decoder 生成一份、Cross-Decoder 共享一份”。因此它同时影响 KV Cache 显存、Prefill 路径和 Decode 阶段的访存模式。

### 为什么需要 YOCO

对于标准的 Decoder-only Transformer，序列中的每个 token 会依次经过全部层。第 \(l\) 层产生的 Key 和 Value 会在后续自回归步骤中被缓存：

\[
K_l = X_l W_K^{(l)}, \qquad V_l = X_l W_V^{(l)}
\]

假设序列长度为 \(N\)，层数为 \(L\)，每层缓存的 KV 表示宽度为 \(D\)，则所有层的 KV Cache 规模近似为：

\[
M_{\mathrm{Transformer}} = O(LND)
\]

实际显存还会受到 batch size、KV head 数、head dimension、数据类型和 padding 的影响，但“随层数和上下文长度同时增长”是核心问题。长上下文服务中，KV Cache 可能比模型权重更先成为显存瓶颈；即使使用 GQA 或 KV 量化，也只是减小每层 KV 的常数，不能消除层数 \(L\) 这一维。

YOCO 的目标可以概括为：

- 让全局、随序列长度增长的 KV 只保存一份；
- 让 Self-Decoder 使用滑动窗口注意力或 Retention 等有界状态，避免它重新引入 \(O(N)\) 的逐层 Cache；
- 利用计算依赖关系，在 Prefill 时提前结束；
- 在 Decode 时保留自回归能力，并让每个 Cross-Decoder 层用独立的 Query 读取全局信息。

### 架构总览

YOCO 将总层数 \(L\) 分成两部分。论文的基本配置是前 \(L/2\) 层为 Self-Decoder，后 \(L/2\) 层为 Cross-Decoder。设输入 embedding 为 \(X^0\)，则前半部分先计算：

\[
X^l = \operatorname{SelfDecoder}_l(X^{l-1}),
\qquad l \in \left[1, \frac{L}{2}\right]
\]

Self-Decoder 的输出经过投影后生成共享的全局 KV：

\[
\hat{K} = \operatorname{LN}\left(X^{L/2}\right) W_K,
\qquad
\hat{V} = \operatorname{LN}\left(X^{L/2}\right) W_V
\]

后半部分的每一层不再从自己的隐藏状态生成供未来步骤缓存的全局 \(K_l,V_l\)，而是生成自己的 Query 并读取 \(\hat{K},\hat{V}\)：

\[
\hat{Q}^{l} = \operatorname{LN}(X^l) W_Q^{(l)}
\]

\[
X^{l+1} = \operatorname{CrossAttention}_l
\left(\hat{Q}^{l}, \hat{K}, \hat{V}\right) + X^l
\]

随后仍然接 FFN、残差和归一化等标准 Transformer 组件。Cross-Attention 使用因果 Mask，因此位置 \(i\) 只能访问不晚于 \(i\) 的全局 KV；这保证了模型仍然可以用于自回归语言建模，而不是退化成双向编码器。

可以用下面的依赖关系理解它：

```text
输入 token
    │
    ▼
Self-Decoder（局部/有界状态）
    │
    ├── 产生中间表示 X
    └── 产生一次全局 KV：K̂、V̂
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Q^(L/2+1)    Q^(L/2+2)      Q^L
          │            │            │
          └──── Cross-Attention ───┘
                       │
                       ▼
                    输出 logits
```

这里的“共享”只针对全局 KV，不代表 Cross-Decoder 的计算完全相同。每层都有自己的参数、隐藏状态和 Query：

\[
\hat{Q}^{l} \neq \hat{Q}^{l+1}
\]

因此不同层仍然可以从不同角度查询同一份全局记忆。共享 KV 不等于共享注意力权重，也不等于各层表示被强行绑定。

### Self-Decoder：把历史编码成可复用的全局表示

Self-Decoder 的任务不是保存每一层的完整历史，而是将输入序列逐步编码为 Cross-Decoder 可查询的表示。它使用高效的因果 Self-Attention，例如滑动窗口注意力（Sliding-Window Attention）或 Gated Retention。

滑动窗口注意力只允许当前位置访问固定窗口内的 token。若窗口大小为 \(C\)，其单层 Decode Cache 规模从随序列长度增长的 \(O(N)\) 降为 \(O(C)\)。Gated Retention 则可以在训练时使用并行或 Chunkwise 形式，在推理时维护递归状态。例如，一个简化的递归状态可以写成：

\[
S_t = \gamma_t S_{t-1} + K_t^{\mathsf T} V_t
\]

\[
y_t = Q_t S_t
\]

其中 \(S_t\) 是有界的中间状态， \(\gamma_t\) 是由输入控制的衰减因子。具体采用哪种高效模块是架构选择；YOCO 的关键约束是 Self-Decoder 不能为每个层保留完整的、随 \(N\) 增长的历史 KV。

### Cross-Decoder：共享 KV，但不共享 Query

Cross-Decoder 使用 Self-Decoder 的输出生成一次全局 \(\hat{K},\hat{V}\)，之后各层只保留所需的共享 KV 和当前解码状态。第 \(l\) 层的注意力可抽象为：

\[
\operatorname{Attn}_l =
\operatorname{softmax}\left(
\frac{\hat{Q}^{l}\hat{K}^{\mathsf T}}{\sqrt{d_k}} + M_{\mathrm{causal}}
\right)\hat{V}
\]

其中 \(M_{\mathrm{causal}}\) 屏蔽未来位置。由于每个 \(\hat{Q}^{l}\) 都来自不同层的隐藏状态和投影矩阵，Cross-Decoder 仍然具有层间差异，同时获得对整个历史序列的全局访问能力。

这也是 YOCO 与“只保留最后一层 KV”这类朴素方案的区别。朴素方案会丢掉大量层级表示；YOCO 则训练一个专门的 Self-Decoder，使其输出成为后半部分共同使用的记忆空间。换句话说，Self-Decoder 承担了“构造可查询记忆”的职责，Cross-Decoder 承担了“基于当前层语义查询记忆”的职责。

### Prefill 与 Decode 的不同路径

YOCO 最容易被误解的地方，是 Prefill 和 Decode 并不使用完全相同的执行路径。

#### Prefill：在 Self-Decoder 处提前结束

Prefill 一次处理整段 Prompt。标准 Transformer 必须让整段序列经过全部 \(L\) 层，才能得到最终输出和各层 KV。YOCO 则先让 Prompt 经过 Self-Decoder，并生成共享的全局 KV。由于 Cross-Decoder 的输入依赖于这份已经生成的全局表示，YOCO 的推理计算流允许在进入 Cross-Decoder 前 Early Exit：

```text
Prompt
  │
  ▼
Self-Decoder
  │
  ├── 生成全局 K̂、V̂
  └── Prefill 路径提前结束
```

该 Early Exit 是 YOCO 架构和训练目标共同保证的推理性质，不应简单理解为“把普通 Transformer 的后半层删掉”。在具体实现中，需要使用论文定义的 Prefill 输出路径和对应的输出头；如果直接在任意普通 Decoder 上跳过后半层，通常会改变模型结果。

从注意力计算的渐进复杂度看，标准 Transformer 的全序列 Prefill 注意力成本近似为：

\[
O(LN^2D)
\]

YOCO 的 Self-Decoder 使用高效注意力，且 Prefill 不进入 Cross-Decoder，论文将其注意力部分概括为：

\[
O(LND)
\]

实际速度仍取决于实现的 Retention、窗口大小、Chunk 划分、Kernel 融合、并行策略和硬件利用率；因此不能仅凭复杂度直接推出固定倍数的端到端加速。

#### Decode：新 token 仍需经过两部分

自回归 Decode 时，每次只输入一个或一小段新 token。新 token 必须先经过 Self-Decoder，以更新其有界状态并得到新的中间表示；随后还要经过 Cross-Decoder，生成最终 logits：

```text
新 token
  │
  ▼
Self-Decoder
  │
  ├── 更新有界状态
  └── 得到当前 token 的中间表示
          │
          ▼
Cross-Decoder
  │
  ├── Q^(L/2+1) 读取 K̂、V̂
  ├── Q^(L/2+2) 读取 K̂、V̂
  └── Q^L        读取 K̂、V̂
          │
          ▼
      next-token logits
```

因此，YOCO 并不是在 Decode 时只运行 Self-Decoder。它减少的是需要长期保存的 KV 副本，以及 Prefill 的计算路径；Cross-Decoder 对全局 KV 的读取仍然存在。

### KV Cache 与计算量分析

假设共享全局 KV 的长度为 \(N\)，宽度为 \(D\)，Self-Decoder 每层只需要大小与窗口或递归状态相关的有界 Cache。若把该有界规模记为常数 \(C\)，则 YOCO 的 Cache 规模可写为：

\[
M_{\mathrm{YOCO}} = O(ND + CLD)
\]

当长上下文满足 \(N \gg CL\) 时，可近似为：

\[
M_{\mathrm{YOCO}} = O(ND)
\]

与标准 Transformer 的 \(O(LND)\) 相比，长序列下可以获得接近 \(L\) 倍的全局 KV 存储量级收益。但“接近 \(L\) 倍”只是理想化比较：GQA、不同 KV 宽度、量化、窗口 Cache、临时 workspace 和 batch 布局都会影响实际显存。

需要将三类成本分开：

| 成本 | 标准 Decoder-only Transformer | YOCO |
| --- | --- | --- |
| 运行期间的全局 KV 常驻显存 | 每层一份，约为 \(O(LND)\) | 共享一份，约为 \(O(ND)\)，另加有界状态 |
| Prefill 注意力路径 | 通常经过全部层 | 可在 Self-Decoder 后 Early Exit |
| Decode 的全局 KV 访存 | 各层读取各自的 KV | 多个 Cross-Decoder 层重复读取共享 KV |
| 主要收益 | 通用且实现成熟 | 长上下文容量和 Prefill 延迟更有优势 |
| 主要代价 | KV Cache 随层数增长 | 共享 KV 的带宽和读取调度成为关键路径 |

需要特别注意：YOCO 的 KV 显存收益不等于 Decode 计算量按相同比例下降。Decode 阶段每个 Cross-Decoder 层仍然要计算 Query、注意力分数以及加权 Value；它只是避免了为这些层分别保存全局 KV。共享一份 KV 还可能增加同一数据被多层反复读取的带宽压力，因此实际部署需要关注 HBM 带宽、Cache 复用、Kernel 融合和批处理调度。

### 与其他 KV 优化方法的区别

| 方法 | 主要做法 | 是否改变层级结构 | 典型收益 | 主要限制 |
| --- | --- | --- | --- | --- |
| MHA | 每个 Query Head 都有独立 KV Head | 否 | 表达能力完整 | KV Cache 最大 |
| MQA/GQA | 多个 Query Head 共享 KV Head | 否 | 减少每层 KV 的 head 数 | 共享粒度可能影响模型质量 |
| MLA | 缓存低维 latent，并在计算中吸收投影 | 通常否 | 压缩每层 KV 的表示 | 需要特殊的投影与 Kernel |
| Sliding Window | 每层只看固定窗口 | 否 | Cache 与窗口大小相关 | 单层无法直接访问完整历史 |
| YOCO | Self-Decoder 生成一次全局 KV，Cross-Decoder 跨层复用 | 是 | 跨层减少全局 KV，并支持 Prefill Early Exit | Decode 反复读取共享全局 KV |

YOCO 可以与 GQA 等机制组合：共享的全局 KV 本身仍可使用较少的 KV Head。它与 DCP、量化或 KV offload 的关系也不是互斥的，但这些系统优化分别解决并行、精度、设备容量或传输问题，不能把它们的收益混为一谈。

### 优势、限制与适用场景

YOCO 更适合以下场景：

- 上下文长度很大，KV Cache 已经成为主要显存占用；
- Prompt 很长，降低 Prefill 延迟比单 token Decode 延迟更重要；
- 服务系统希望在同一 GPU 上容纳更多并发序列；
- 可以为新架构重新训练或进行充分的架构适配，而不是直接替换已有 Transformer 层。

它的限制也很明确：

- YOCO 不是对现有模型的无损推理插件，需要专门的模型结构和训练；
- Self-Decoder 的局部注意力或 Retention 必须保留足够的历史信息，否则共享 KV 的质量会受影响；
- Decode 仍需执行 Cross-Decoder，并多次访问长序列的共享 KV，因此内存容量问题缓解后，带宽和调度可能成为新的瓶颈；
- 全局 KV 的物理布局、并行切分、Paged Cache 和算子实现会显著影响真实吞吐；
- 模型加载显存峰值、运行期间常驻显存和注意力临时 workspace 是不同指标，不能只看 KV Cache 复杂度评估部署成本。

标准 Transformer 是“每层都缓存自己的历史 KV”；YOCO 是“Self-Decoder 将历史编码成一份全局 KV，Cross-Decoder 的各层只用自己的 Query 共享读取它”。因此 YOCO 的核心收益来自跨层复用和 Prefill Early Exit，而不是简单的 KV 量化；它把显存容量压力显著降低，同时把 Decode 阶段的关键问题转移为共享全局 KV 的带宽、布局和调度问题。

--

- [You Only Cache Once: Decoder-Decoder Architectures for Language Models](https://arxiv.org/abs/2405.05254)
- [NeurIPS 2024 Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/file/0df38cd13520747e1e64e5b123a78ef8-Paper-Conference.pdf)


% Chunked Prefill

### 背景

在大模型推理中，**prefill 阶段** 是计算密集型，尤其是在真实业务中，prompt 往往是 **1K～4K tokens** 甚至更长。一个直觉上的做法是：

> 既然是 prefill，就一次性把所有 prompt token 喂给模型。

但实践中会遇到一个问题：**一次性塞更多 token，并不会无限提升吞吐率**。

#### Prefill 阶段

**关键观察 1**：Prefill 吞吐存在“边际收益递减”

在固定模型和 GPU 的情况下，prefill 阶段的吞吐率会在某个 token 数附近达到上限。比如下图是 FlashAttention 3 在 H200 上的性能测试结果（这里采用定长序列测试，行是序列长度，列是 batch size）：

可以观察到，在某个序列长度下，随着 batch size 的提升，attention kernel 的 FLOPS 不进一步提升，反而会逐步下降。所以这个时候进一步提升输入的 batch size 是没有意义的，反而会增加请求的 TTFT。

而且一个很重要的现象是：

> **模型越大（hidden size 越大），达到算力饱和所需的 token 数反而越小**

也就是说，只要 chunk 大小选得合理，就已经可以充分利用 GPU 计算能力。

---

#### Decode 阶段

decode 阶段（逐 token 生成）也有类似现象：

* 随着 batch size / decode token 总数增加，吞吐率持续上升
* 当总 token 数接近 **512** 左右时，也会进入 **compute-bound**
* 再继续加 batch，对性能帮助不大

这意味着：
**prefill 和 decode 本质上都不需要“越大越好”的一次性计算单元。**

---

### 目标

综合来看，chunked prefill 的目标是解决两个现实问题：

1. **超长 prompt 的计算调度问题**

真实业务中：

* 一个请求可能带 **2K / 4K token prompt**
* 如果一次性 prefill：

  * 显存占用大
  * kernel 执行时间长
  * 调度不灵活，容易阻塞其他请求

而 **chunked prefill** 允许我们把一个长 prompt 拆成多个**计算友好的小单元**。

---

2. **提高系统层面的并发与公平性**

在 serving 场景中：

* 很多请求是「长 prompt + 短输出」
* 如果长 prompt 一次性 prefill：

  * GPU 会被单个请求“霸占”较长时间
* 拆成 chunk 后：

  * 每个 chunk 都是一个可调度的计算单元
  * 可以和其他请求交错执行
  * 系统整体 latency 和 tail latency 更可控

---

### 长序列分治

**Chunked Prefill 的核心思想**就是“化整为零”：将长的输入序列（`K, V`）切分成若干个较小的块（Chunks），然后让查询（`Q`）逐个与这些块进行计算，最后像“拼图”一样，将每个块的结果合并成完整的注意力输出。这就像你无法一口吃完一个大蛋糕，但可以把它切成小块慢慢吃。

将长序列分成多个 chunk（块），逐块处理：

```
输入序列: [chunk1, chunk2, chunk3, ...]
处理流程:
1. 处理 chunk1 → 得到部分结果
2. 处理 chunk2 → 与 chunk1 的结果合并
3. 处理 chunk3 → 与前两个 chunk 的结果合并
...
```

### Chunk Attn

设请求总长 \(N\) tokens，chunked_prefill_size = \(C\)，分成 \(\lceil N/C \rceil\) 轮。

第 \(t\) 轮（\(t = 0, 1, ...\)）：

- 前缀长度：\(P_t = \min(tC, N)\)
- Extend 长度：\(E_t = \min(C, N - P_t)\)
- 总 seq_len：\(S_t = P_t + E_t\)
- Query 只包含位置 \([P_t, P_t + E_t)\) 的 token。对于 extend 中的第 \(i\) 个 query token（\(0 \le i < E_t\)），其全局位置为 \(P_t + i\)，attention 输出为：

\[o_i = \frac{\sum_{j=0}^{P_t + i} \exp(q_i \cdot k_j / \sqrt{d}) \cdot v_j}{\sum_{j=0}^{P_t + i} \exp(q_i \cdot k_j / \sqrt{d})}\]

其中 \(k_j, v_j\) 来源分两部分：

\(j \in [0, P_t)\)：从 KV cache 中读取（前几轮已算好）
\(j \in [P_t, P_t + i]\)：从本轮新计算的 KV 中读取

---

Triton 两阶段公式更明确：

\[o_i^{(1)} = \text{Attn}(q_i, K_{\text{prefix}}, V_{\text{prefix}}) \quad \text{(Stage 1: 全部 prefix，无 mask)}\]

\[o_i^{(2)} = \text{CausalAttn}(q_i, K_{\text{extend}}[:i+1], V_{\text{extend}}[:i+1]) \quad \text{(Stage 2: causal)}\]

\[o_i = \text{OnlineSoftmaxMerge}(o_i^{(1)}, o_i^{(2)})\]

Online softmax merge：

\[m = \max(m_1, m_2)\] \[o_i = \frac{e^{m_1 - m} \cdot l_1 \cdot o_i^{(1)} + e^{m_2 - m} \cdot l_2 \cdot o_i^{(2)}}{e^{m_1 - m} \cdot l_1 + e^{m_2 - m} \cdot l_2}\]

其中 \(m_1, l_1\) 和 \(m_2, l_2\) 分别是两阶段的 log-sum-exp 统计量。

### Workflow

#### 核心状态变量

Scheduler 维护两个关键状态：

- **`chunked_prefill_size`**：每轮 prefill 最多处理的 token 数，根据 GPU 显存自动设置（如 H100 默认 8192）
- **`chunked_req`**：当前正在分块处理的请求，**全局至多一个**

#### 每轮调度循环

每一轮调度大致经过以下步骤：

**Step 1：暂存未完成的 chunked request**

```python
if self.chunked_req is not None:
    self.stash_chunked_request(self.chunked_req)
    # 将已计算的 KV 写入 radix tree cache，下一轮可作为 prefix 复用
```

**Step 2：处理上一轮 prefill batch 中已完成的请求**

上一轮 prefill batch 中，除了 `chunked_req` 以外的请求都已完成 prefill，合并进 `running_batch` 进入 decode 阶段。`chunked_req` 被过滤掉，不进入 decode。

**Step 3：尝试构建新的 prefill batch**

```python
new_batch = self.get_new_batch_prefill()
```

**Step 4：prefill 优先于 decode**

```
if new_batch:     → 执行 prefill
elif running_batch: → 执行 decode
else:             → 空闲
```

**Prefill 始终优先。** 只有没有任何 prefill 工作时才 decode。

#### 构建 Prefill Batch

这是调度的核心，通过 `PrefillAdder` 管理三个独立的 token 预算：

| 预算 | 含义 |
|------|------|
| `rem_total_tokens` | KV cache pool 剩余可分配的 slot 数 |
| `rem_input_tokens` | 本轮 prefill batch 的总 input token 上限（`max_prefill_tokens`） |
| `rem_chunk_tokens` | chunked prefill 大小限制，即本轮 prefill 最多处理多少 token |

任何一个预算耗尽，就停止添加请求。

##### 3a. 优先继续未完成的 chunked request

```python
if self.chunked_req is not None:
    self.chunked_req.init_next_round_input()  # 重新计算 prefix/extend
    self.chunked_req = adder.add_chunked_req(self.chunked_req)
```

`init_next_round_input` 会查 radix cache，把上一轮已写入 cache 的 token 识别为 `prefix_indices`（不需要重算），剩余的作为 `extend_input_len`（本轮需要计算的 Q token）。

`add_chunked_req` 的核心逻辑：

```python
_rem_tokens = min(rem_chunk_tokens, rem_total_tokens)
truncated = req.extend_input_len > _rem_tokens
req.set_extend_input_len(min(req.extend_input_len, _rem_tokens))
req.fill_ids = req.fill_ids[:len(req.prefix_indices) + req.extend_input_len]
return req if truncated else None  # 还没完就返回 req，完了返回 None
```

关键细节：**只有请求完全 prefill 完毕时，才预留 `max_new_tokens` 的 decode 空间**。中间 chunk 不预留，避免过早占用 KV cache。

##### 3b. 从等待队列添加新请求

遍历 `waiting_queue`，对每个请求：

1. 查 radix cache 计算可复用的 prefix 长度
2. 计算需要新算的 `extend_input_len`
3. 如果 extend 长度在预算内 → 完整加入
4. 如果超出 `rem_chunk_tokens` → **截断**，成为新的 `chunked_req`

```python
if input_tokens > self.rem_chunk_tokens:
    trunc_len = self.rem_chunk_tokens // page_size * page_size  # 对齐到 page
    req.set_extend_input_len(trunc_len)
    self.new_chunked_req = req  # 标记为需要后续继续
```

截断长度必须对齐 `page_size`，且**一轮最多产生一个新的 chunked request**。

#### Mixed Chunked Prefill

当 `enable_mixed_chunk=True` 时，prefill batch 会和当前 decode batch **合并成一个 forward pass**：

```python
if self.is_mixed_chunk and not self.running_batch.is_empty():
    self.running_batch.prepare_for_decode()
    new_batch.mix_with_running(self.running_batch)
```

此时 `rem_chunk_tokens` 和 `rem_input_tokens` 会预先扣除 decode token 数，给 decode 留出空间。这样 decode 请求不会因为长 prefill 被饿死。

#### 完整生命周期举例

一个 12000 token 的请求，`chunked_prefill_size=4096`，`page_size=16`：

```
Round 1:
  - waiting_queue 取出请求
  - prefix_len=0, extend_len=12000 > 4096 → 截断为 4096
  - Q=[0:4096], KV=[0:4096]
  - scheduler.chunked_req = req
  - 执行 forward，KV 写入 cache

Round 2:
  - stash_chunked_request: 把 4096 个 KV 写入 radix cache
  - init_next_round_input: prefix_len=4096, extend_len=7904 > 4096 → 截断为 4096
  - Q=[4096:8192], KV=[0:8192]（Q attend 到完整 KV）
  - scheduler.chunked_req = req（还没完）
  - 可能与 decode batch 混合执行

Round 3:
  - stash: 8192 个 KV 在 cache 中
  - init_next_round: prefix_len=8192, extend_len=3808 ≤ 4096 → 不截断
  - Q=[8192:12000], KV=[0:12000]
  - add_chunked_req 返回 None → 此时预留 max_new_tokens 空间
  - scheduler.chunked_req = None（完成）

Round 4:
  - 该请求进入 running_batch，开始 decode
```


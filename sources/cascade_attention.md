% Cascade Attention

### 标准 Attention 的可拆分性

标准 attention 对一个 query token \(q\) 的计算是：

\[o = \text{softmax}\left(\frac{q K^T}{\sqrt{d}}\right) V\]

**关键性质**：如果 KV 可以分成两段不相交的子集 \(K = [K_A; K_B]\)，则：

\[\text{LSE}_A = \log \sum_{j \in A} \exp\left(\frac{q k_j}{\sqrt{d}}\right), \quad o_A = \text{softmax}(q K_A^T) V_A\]

\[\text{LSE}_{total} = \log\left(e^{\text{LSE}_A} + e^{\text{LSE}_B}\right)\]

\[o = e^{\text{LSE}_A - \text{LSE}_{total}} \cdot o_A + e^{\text{LSE}_B - \text{LSE}_{total}} \cdot o_B\]

这就是 `merge_state_v2` 做的事，**两段分开算再合并，数学上完全等价于一次完整 attention**。

---

### Topk = 1 时没有问题

设前缀 KV 长度为 \(L\)，draft tokens 长度为 \(S\)：

```
Request i:  [prefix_KV (L tokens)] + [draft_KV (S tokens)]
```

一次 `flash_attn_with_kvcache` 搞定，`cache_seqlens = L + S`。

---

### Topk > 1 引入了什么问题

以 `topk=4`、`S=3` 为例，Target Verify 阶段：

```
Request i 有 4 条候选树，每条 3 个 draft tokens：
  Branch 0: [d_00, d_01, d_02]  各自需要 attend: prefix_KV_i + 本分支祖先
  Branch 1: [d_10, d_11, d_12]  各自需要 attend: prefix_KV_i + 本分支祖先
  Branch 2: [d_20, d_21, d_22]  ──────────────────────────────────────────
  Branch 3: [d_30, d_31, d_32]        共享同一段 prefix_KV_i !!
```

**朴素展开（Naive Expand）方案**：将 batch 展成 `B × topk` 个虚拟请求：

| 虚拟请求 (i, k) | Q | KV |
|---|---|---|
| (i, 0) | 3 个 query | prefix_KV_i (L) + branch0_KV (3) |
| (i, 1) | 3 个 query | prefix_KV_i (L) + branch1_KV (3) |
| (i, 2) | 3 个 query | prefix_KV_i (L) + branch2_KV (3) |
| (i, 3) | 3 个 query | prefix_KV_i (L) + branch3_KV (3) |

prefix_KV_i 被**重复读取 topk 次**，总读 KV 量：

\[\text{Naive} = B \times topk \times (L + S) = B \times 4 \times (L + 3)\]

当 \(L = 1000\) 时，主要开销都在反复读那 1000 token 的 prefix。

---

### Cascade 方案的数学结构

将 KV 拆成两段，**利用 prefix 被所有分支共享**：

**第一段 attention**（共享 prefix，`metadata.page_table`）：

\[B \text{ 个请求，每个请求的 Q} = \text{所有 topk 分支的全部 query tokens}\]
\[\text{KV} = \text{prefix\_KV}_i \quad (\text{每个请求只读一次！})\]

输出：\(o^{(A)}\)，\(\text{LSE}^{(A)}\)

**第二段 attention**（各自独立的 draft KV，`forward_metadata_spec_decode_expand.page_table`）：

\[B \times topk \text{ 个虚拟请求，每个请求的 Q} = \text{本分支 query tokens}\]
\[\text{KV} = \text{各自分支的 draft tokens (很短，只有 } S \text{ 个)}\]

输出：\(o^{(B)}\)，\(\text{LSE}^{(B)}\)

**合并**：`merge_state_v2(o_A, LSE_A, o_B, LSE_B)` → 精确等价于原始完整 attention。

总读 KV 量：

\[\text{Cascade} = \underbrace{B \times L}_{\text{prefix 只读一次}} + \underbrace{B \times topk \times S}_{\text{draft 各自读}}\]

---

### 多条候选 token（topk）影响的核心

| 方案 | prefix KV 读取次数 | draft KV 读取次数 |
|------|------------------|-----------------|
| Naive Expand | \(B \times topk \times L\) | \(B \times topk \times S\) |
| Cascade | \(B \times L\) | \(B \times topk \times S\) |
| **节省** | **topk 倍** | 无节省 |

当 \(topk = 4\)，\(L = 2000\)，\(S = 5\)：
- Naive 每请求读 8020 个 KV token
- Cascade 每请求读 2020 个 KV token
- **节省约 4 倍**，且 \(L\) 越长、\(topk\) 越大，优势越明显

这就是为什么 `use_cascade_attn` 的触发条件是 `topk > 1`——**topk = 1 时没有共享 prefix 可利用，cascade 无意义**；topk 越大，prefix KV 的重复读取浪费越严重，cascade 收益越大。


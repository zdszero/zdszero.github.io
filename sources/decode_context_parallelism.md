% DCP

Decode Context Parallelism（DCP，解码上下文并行）是一种面向大模型推理 decode 阶段的并行策略。它沿着上下文长度或 KV cache 的 token 维度切分工作，而不是主要沿着 attention head、请求或网络层切分。

在自回归生成中，每一步通常只有一个或少量 query token，却需要访问很长的历史 KV cache。设当前 query 为矩阵

\[
Q \in \mathbb{R}^{B \times H \times D}
\]

历史缓存为

\[
K,V \in \mathbb{R}^{B \times S \times H \times D},
\]

其中 \(B\) 是 batch size，\(H\) 是 attention head 数，\(D\) 是 head dimension，\(S\) 是上下文长度。decode 阶段的主要问题不是生成 query 的数量，而是每次都要从显存中读取长度为 \(S\) 的 KV。

如果使用 \(N\) 个 DCP rank，可以让每个 rank 只处理约 \(S/N\) 个历史 token，从而降低单卡的 KV 读取量和局部 attention 计算量。DCP 的目标可以概括为：

- 降低单卡每一步需要读取的 KV cache 规模；
- 将长上下文 attention 的计算和显存带宽压力分摊到多个设备；
- 在保持全局 attention 语义的前提下，利用设备间高速互联完成结果合并。

DCP 并不会把一次完整 attention 变成彼此独立的多个 attention。每个 rank 只计算一个局部 KV 分片，最后仍然必须通过通信恢复完整 softmax 的结果。

### 基本原理

#### 按上下文维度切分 KV

设上下文中的 token 位置为 \(p\)，DCP world size 为 \(N\)，一种常见的循环切分规则是

\[
\operatorname{owner}(p) = p \bmod N.
\]

于是 rank \(r\) 持有的位置集合为

\[
\mathcal{P}_r = \{p \mid p \bmod N = r\}.
\]

例如，当 \(N=4\) 时，四个 rank 的 token 归属可以表示为：

```text
rank 0: 0, 4, 8, 12, ...
rank 1: 1, 5, 9, 13, ...
rank 2: 2, 6, 10, 14, ...
rank 3: 3, 7, 11, 15, ...
```

这种切分方式与连续 block 切分不同。连续切分会使每个 rank 持有一个连续的上下文区间；循环切分则让每个 rank 以固定步长交错持有 token。循环切分有利于平衡不同位置的工作量，也能避免某些 rank 长期只负责序列前部或后部。

当上下文长度为 \(S\) 时，rank \(r\) 持有的 token 数量为

\[
S_r = \left\lfloor \frac{S}{N} \right\rfloor + \mathbf{1}\left[r < S \bmod N\right].
\]

因此各 rank 的长度差最多为一个 token。对于从全局位置 \(a\) 开始、长度为 \(L\) 的片段，需要统计区间 \([a,a+L)\) 中满足 \(p \bmod N=r\) 的位置数量，而不能简单地使用 \(L/N\)。

#### 局部 attention

完整 attention 可以写成

\[
O = \operatorname{softmax}\left(\frac{QK^{\mathsf T}}{\sqrt{D}}\right)V.
\]

将 \(K,V\) 按 DCP rank 切分为 \((K_r,V_r)\) 后，rank \(r\) 先计算局部结果

\[
O_r = \operatorname{softmax}_{\text{local}}\left(\frac{QK_r^{\mathsf T}}{\sqrt{D}}\right)V_r.
\]

同时，rank \(r\) 必须保留局部 softmax 的归一化统计量。使用自然对数时，可以记为

\[
L_r = \log\sum_{p\in\mathcal{P}_r} \exp\left(\frac{qk_p}{\sqrt{D}}\right).
\]

这里的 \(L_r\) 不是可有可无的辅助输出，而是跨分片恢复全局 softmax 所必需的信息。只传输或平均局部 output，无法得到正确的全局 attention。

### LSE 加权合并

#### 为什么不能直接平均

局部 attention 的 softmax 只在本 rank 的 KV 集合上归一化。不同 rank 的局部归一化因子通常不同，因此下面的做法一般不正确：

\[
O \ne \frac{1}{N}\sum_{r=0}^{N-1}O_r.
\]

正确的合并方式是根据每个 rank 的局部 LSE 重新计算权重：

\[
O = \frac{\sum_{r=0}^{N-1}\exp(L_r)O_r}{\sum_{r=0}^{N-1}\exp(L_r)}.
\]

如果 attention 使用的是 base-2 的 softmax 统计量，则对应形式为

\[
O = \frac{\sum_{r=0}^{N-1}2^{L_r}O_r}{\sum_{r=0}^{N-1}2^{L_r}}.
\]

因此，DCP 通信通常需要交换两类数据：

- 局部 attention output；
- 局部 LSE 或等价的 softmax 统计量。

#### 数值稳定的合并

直接计算 \(\exp(L_r)\) 可能溢出。令

\[
m = \max_{0\le r<N}L_r,
\]

则可以改写为

\[
O = \frac{\sum_{r=0}^{N-1}\exp(L_r-m)O_r}{\sum_{r=0}^{N-1}\exp(L_r-m)}.
\]

此时所有指数的输入都不大于零，数值稳定性更好。全局 LSE 也可以同时恢复：

\[
L = m + \log\sum_{r=0}^{N-1}\exp(L_r-m).
\]

对于 base-2 统计量，将自然对数和指数替换为 \(\log_2\) 与 \(2^x\)：

\[
L = m + \log_2\sum_{r=0}^{N-1}2^{L_r-m}.
\]

#### 合并的语义

DCP 的结果合并本质上是一个分布式的 online softmax。每个 rank 不需要拿到其他 rank 的全部 KV，也不需要恢复完整的 attention score 矩阵；它只需要交换足以描述局部 softmax 分布的统计量和局部加权结果。

这正是 DCP 能够节省 KV 读取的关键：KV 留在拥有它的 rank 上，跨 rank 传输的是 attention 的中间结果，而不是完整 KV cache。

### DCP 的理论基础

前面的公式已经给出了 DCP 的基本机制。本节把它们补成完整的推导，回答三个问题：合并公式为什么成立；局部结果里最少需要传什么；这套合并对哪些 attention 变体仍然成立。

#### 合并恒等式的推导

把局部结果写成未归一化的形式，推导会更清楚。设当前 query 可见的 KV 位置集合为 \(\mathcal{I}\)，DCP 把它划分成 \(N\) 个互不相交的子集

\[
\mathcal{I} = \mathcal{P}_0 \cup \mathcal{P}_1 \cup \cdots \cup \mathcal{P}_{N-1},
\qquad
\mathcal{P}_r \cap \mathcal{P}_s = \varnothing \quad (r \ne s).
\]

对位置 \(p\) 记 score

\[
s_p = \frac{q^{\mathsf T} k_p}{\sqrt{D}},
\]

则全局 attention 可以拆成分子与分母两部分：

\[
O = \frac{\sum_{p \in \mathcal{I}} \exp(s_p) v_p}{\sum_{p \in \mathcal{I}} \exp(s_p)}
= \frac{\mathcal{N}}{\mathcal{Z}}.
\]

rank \(r\) 在本地对同一个公式求值，只是把求和范围换成自己的分片 \(\mathcal{P}_r\)：

\[
\mathcal{Z}_r = \sum_{p \in \mathcal{P}_r} \exp(s_p),
\qquad
\mathcal{N}_r = \sum_{p \in \mathcal{P}_r} \exp(s_p) v_p,
\qquad
O_r = \frac{\mathcal{N}_r}{\mathcal{Z}_r}.
\]

因为 \(\mathcal{I}\) 是这些分片的并集，求和可以直接拆分：

\[
\mathcal{Z} = \sum_{p \in \mathcal{I}} \exp(s_p) = \sum_{r=0}^{N-1} \mathcal{Z}_r,
\qquad
\mathcal{N} = \sum_{p \in \mathcal{I}} \exp(s_p) v_p = \sum_{r=0}^{N-1} \mathcal{N}_r.
\]

把 \(\mathcal{N}_r = \mathcal{Z}_r O_r\) 与 \(\mathcal{Z}_r = \exp(L_r)\) 代回去，就得到前面用到的合并公式：

\[
O = \frac{\mathcal{N}}{\mathcal{Z}}
= \frac{\sum_{r=0}^{N-1} \exp(L_r) O_r}{\sum_{r=0}^{N-1} \exp(L_r)}.
\]

因此 LSE 加权合并不是经验公式，而是 softmax 的分子和分母都可以按 KV 分片可加的直接结果。这个推导同时点明了理论前提：只有当每个位置对 score 的贡献只依赖它自己时，这种可加性才成立。

#### 只传局部 output 为什么不够

上面的推导里，一个 rank 提供给全局的信息是“归一化的 \(O_r\)”加上“归一化因子 \(L_r\)”这一对。如果只传 \(O_r\) 而丢掉 \(L_r\)，全局结果在数学上就不可辨识。一个最小的反例：

```text
N = 2, D = 1
rank 0: 一个 token, v = 1, s = 0
rank 1: 一个 token, v = 0, s = log 3
```

| 量 | rank 0 | rank 1 |
| --- | --- | --- |
| \(\mathcal{Z}_r = \sum \exp(s_p)\) | 1 | 3 |
| \(\mathcal{N}_r = \sum \exp(s_p) v_p\) | 1 | 0 |
| \(O_r = \mathcal{N}_r / \mathcal{Z}_r\) | 1 | 0 |

正确的全局结果是

\[
O = \frac{1 + 0}{1 + 3} = 0.25,
\]

而对两个局部 output 直接取平均只会得到 \(0.5\)。更糟的是，这个 \(0.5\) 完全不依赖 score：只要两个 token 的 \(v\) 分别是 1 和 0，无论 \(s_1\) 取什么值，局部 output 永远是 \((1,0)\)，平均值永远是 \(0.5\)，而真实结果可以从 0 连续变化到 1。

局部 output 丢失的正是“这个分片在全局 softmax 中占多大的权重”。\(\mathcal{Z}_r\) 或等价的 \(L_r\) 补上的就是这个权重信息，所以它必须和 \(O_r\) 一起参与归约。

#### 合并算子与归约顺序

把每个 rank 的局部结果记为一个二元组 \((\mathcal{Z}_r, \mathcal{N}_r)\)，定义合并算子

\[
(\mathcal{Z}_a, \mathcal{N}_a) \oplus (\mathcal{Z}_b, \mathcal{N}_b)
= (\mathcal{Z}_a + \mathcal{Z}_b,\; \mathcal{N}_a + \mathcal{N}_b).
\]

它满足结合律与交换律，单位元是 \((0, 0)\)。这意味着 \(N\) 个分片可以按任意归约树合并：顺序累加、二叉树、ring、`AllReduce` 都得到同一个精确结果，差异只来自浮点数的舍入顺序。

这一点在工程上很重要：DCP 的归约在语义上是可交换的，全局同步不是数学上的必要条件，而只是某个 collective 实现的调度选择。前文提到的数值稳定写法，本质上也是在这个算子上先减去公共的 \(m\)，让指数输入不超过零，而不是改变运算的数学含义。

\(\mathcal{Z}_r\) 与 \(\mathcal{N}_r\) 也可以换成等价的参数化形式 \((L_r, O_r)\)，两者携带同样的信息。对一个 batch、一个 head 来说，每个 rank 至少要贡献“一个权重 + 一个 \(D\) 维的加权向量”，即约 \(D+1\) 个量。这就是 \(D\) 维 output 之外还必须额外传一个 LSE 的原因。

#### 与 online softmax 的统一

FlashAttention 在单卡上把长序列切成块，维护 running max 与 running sum（等价于 LSE），并逐块合并局部结果。DCP 做的事情是把同样的分块放到不同设备上并行，最后按同一个公式汇聚：

- FlashAttention 沿时间维度分块，块之间顺序合并；
- DCP 沿设备维度分片，分片之间按归约合并。

两者是同一个代数结构在不同并行轴上的展开。这也解释了为什么 DCP 可以直接复用 attention kernel 已有的 partial output 与 LSE：需要的前提只是 kernel 能把局部归一化统计量暴露出来，而不是在返回前就把结果提前归一化掉。

#### 算术强度：decode 为什么天然访存受限

一次 decode 中 attention 的主要计算是两次矩阵乘，FLOPs 约为

\[
4BHSD,
\]

而需要读取的 KV 数据量为 \(2BSHD\) 个元素。若每个元素占 \(b\) 字节，则算术强度

\[
I = \frac{4BHSD}{2BSHD \cdot b} = \frac{2}{b}\ \text{FLOP/byte}
\]

与上下文长度 \(S\)、分片数 \(N\) 都无关。对 fp16（\(b=2\)）来说 \(I = 1\)，而现代 GPU 的 ridge point 在 \(10^2 \sim 10^3\) FLOP/byte 量级，因此 decode attention 天然是访存受限的。原因是每个 KV 元素在一步里只被 \(L_q = 1\) 个 query 使用，增加上下文长度只会线性增加要读的数据，并不会提高复用率。

缓解这一点只有两条路：提高 query 侧的复用（增大 batch、prefill 或多个 query token），或者把 KV 读取量分摊到更多设备上——后者就是 DCP。注意 DCP 并不改变每个 rank 局部的算术强度，它改变的是“一个 rank 需要读多少字节”，相当于把一部分显存带宽压力换成了互联上的归约通信。因此收益是否成立，仍取决于本地 KV 读取节省的时间是否大于归约通信时间；更细的通信量估计与收益判据见后文「计算量、显存与通信分析」。

#### 合并成立的条件与失效边界

上面的推导只依赖两件事：可见位置集合被划分为互不相交的分片；每个位置对 score 的贡献只由它自己决定。据此可以判断哪些 attention 变体仍然适用：

| 变体 | 是否保持可合并 | 原因 |
| --- | --- | --- |
| causal mask、滑动窗口 | 是 | mask 只是缩小可见集合，分片仍是可见集合的划分 |
| attention sink、位置偏置 | 是 | 等价于在某个分片里增加一个额外的 logit 项 |
| softcap 等逐元素 logit 变换 | 是 | 变换作用在每个 score 上，不引入位置之间的耦合 |
| 逐元素缩放、量化与反量化 | 是 | 只要所有 rank 使用同一套变换即可 |
| 线性注意力、状态空间模型（如 KDA、Gated DeltaNet） | 否，需单独设计 | 输出由状态递推决定，合并更接近带衰减修正的状态累加，而不是 LSE 加权 |

失效的共同点是“某个位置的权重依赖其他位置的 score”。只要模型在 softmax 之前引入了跨位置的归一化或耦合，局部分片的简单合并就不再等价于全局结果，必须重新推导合并算子，而不能照搬 LSE 公式。表中线性注意力一行属于按结构推断的结论，具体合并方式应以对应实现为准。

### 一次 decode 的执行流程

一次典型的 DCP decode 可以抽象为以下步骤：

1. 每个 rank 根据全局上下文位置和自己的 rank，确定本地 KV token 集合；
2. 每个 rank 读取本地 KV shard，并构造局部 KV 的索引或页表；
3. 根据并行布局决定是否需要交换 query，或者在每个 rank 上复制完整 query 投影；
4. 每个 rank 对本地 KV 执行 attention，得到 \(O_r\) 与 \(L_r\)；
5. 在 DCP group 内交换局部 output 和 softmax 统计量；
6. 使用 LSE 加权公式合并所有局部结果；
7. 将合并后的结果交给后续的输出投影和采样流程。

其数据流可以概括为：

```text
完整上下文
   ↓  按 token 位置分片
本地 KV shard
   ↓
局部 attention
   ↓
局部 output + 局部 LSE
   ↓  跨设备通信
全局 attention output
```

DCP 只改变 attention 的上下文访问方式，不改变模型的数学结构。对于相同的 query、相同的 KV 和相同的 mask，理想情况下 DCP 的输出应与单设备完整 attention 一致，差别主要来自浮点计算顺序和通信归约顺序。

### 通信原语与实现策略

#### AllGather 与 ReduceScatter

一种直接的策略是先 `AllGather` 所有 rank 的 LSE，再对局部 output 进行校正，最后通过 `AllReduce` 或 `ReduceScatter` 汇总结果。

其逻辑可以写成：

\[
\{L_0,L_1,\ldots,L_{N-1}\}
\xrightarrow{\text{AllGather}}
\text{global LSE}.
\]

随后每个 rank 计算

\[
\widetilde{O}_r = \exp(L_r-L)O_r,
\]

再执行

\[
O = \sum_{r=0}^{N-1}\widetilde{O}_r.
\]

这种方案概念清晰、适用范围较广，但可能需要多个 collective，并且中间 output 的通信量较大。

#### All-to-All

另一种策略是使用 `All-to-All`：将不同 head 的局部 output 和 LSE 直接发送给最终负责该 head 的 rank，然后在目标 rank 本地完成合并。

这类策略通常可以减少 collective 的组织次数，并将通信结果直接路由到目标 head。但它并不意味着总通信字节数必然按相同比例下降。实际收益取决于：

- head 数量和 DCP size 是否匹配；
- output、LSE 的数据类型；
- All-to-All 的实现效率；
- 节点内互联和跨节点网络拓扑；
- 是否能够把 output 与 LSE 打包到同一次通信中。

在高带宽、低延迟的 GPU 互联环境中，All-to-All 可能比多次 `AllGather`、`AllReduce` 更有优势；在普通网络环境中，通信模式的复杂性也可能抵消计算收益。

#### Query 的处理

DCP 需要所有参与本地 KV attention 的 rank 拥有与其 partial output 对应的 query 信息。常见有两种方式：

- **Query AllGather**：每个 rank 先产生本地 query head，再交换 query，使各 rank 获得所需的 query；
- **Query projection replication**：复制 query projection 权重，让每个 rank 直接计算完整 query，减少 decode 时的 query 通信，但会增加重复计算和权重占用。

两者的选择是计算与通信之间的权衡。不能笼统地说复制权重一定更省显存或更快：它减少的是运行时通信，增加的是模型权重常驻空间和 query projection 计算量。

### 计算量、显存与通信分析

#### 局部 attention 计算量

忽略 softmax 和 kernel 融合细节，完整 attention 的主要矩阵乘计算量近似为

\[
\operatorname{FLOPs}_{\text{full}}
\approx 4BHS D.
\]

其中前一个二表示 \(QK^{\mathsf T}\) 和概率乘 \(V\) 两次矩阵乘，每次乘加按两个 FLOPs 计数。

DCP 后，每个 rank 处理约 \(S/N\) 个 KV token，因此局部计算量近似为

\[
\operatorname{FLOPs}_{\text{local}}
\approx \frac{4BHSD}{N}.
\]

这说明 DCP 的局部 attention 计算具有接近 \(N\) 倍的理论分摊效果。但总系统性能还要加上 query 通信、LSE 合并和 collective 调度开销，因此端到端加速通常小于 \(N\) 倍。

#### KV cache 读取量

若每个 KV token 的存储大小为 \(M_{KV}\) 字节，完整 attention 每个 rank 需要读取的 KV 数据近似为

\[
\operatorname{Bytes}_{\text{full}} \approx BSM_{KV}.
\]

DCP 后，单个 rank 的本地读取量近似为

\[
\operatorname{Bytes}_{\text{local}} \approx \frac{BSM_{KV}}{N}.
\]

因此 DCP 主要减少的是**单卡 KV cache 的读取压力和局部工作集**。它不一定减少整个系统的 KV 总容量：如果每个 rank 都保存一份独立的 KV shard，所有 rank 合计仍然需要保存完整上下文的 KV。

#### 归约通信量

假设每个 query 需要传输 output 和 LSE，output 的元素数量为 \(BHD\)，output 每个元素占 \(b_O\) 字节，LSE 每个元素占 \(b_L\) 字节，则一次逻辑上的归约数据规模大致为

\[
\operatorname{Bytes}_{\text{reduce}}
\propto B H (D b_O + b_L).
\]

注意它与上下文长度 \(S\) 基本无关，而 KV 读取量与 \(S\) 成正比。这解释了 DCP 更适合长上下文 decode：当 \(S\) 足够大时，节省的 KV 读取量可以覆盖 output 归约的固定成本。

如果使用 Query AllGather，还需要额外传输 query：

\[
\operatorname{Bytes}_{Q\text{-gather}}
\propto B H D_Q b_Q.
\]

因此 DCP 是否有收益，取决于本地 KV 计算节省与 query、output、LSE 通信之间的平衡。

#### 一个简化的收益判据

用 \(T_{\text{attn}}(S/N)\) 表示每个 rank 处理本地 KV 的时间，用 \(T_{\text{comm}}\) 表示 query 交换和结果归约时间，则 DCP 的单步时间可以粗略写成

\[
T_{\text{DCP}} \approx T_{\text{attn}}(S/N) + T_{\text{comm}}.
\]

非 DCP 的单步时间近似为

\[
T_{\text{base}} \approx T_{\text{attn}}(S).
\]

只有当

\[
T_{\text{attn}}(S) - T_{\text{attn}}(S/N) > T_{\text{comm}}
\]

时，DCP 才有明确的端到端收益。若通信可以与 attention 计算充分 overlap，则有效通信成本会降低；但 overlap 需要足够大的计算块、独立的通信与计算资源，以及高效的调度实现。

### DCP 与其他并行策略的区别

| 策略 | 主要切分维度 | 主要目标 | KV cache 视角 | 典型通信 |
| --- | --- | --- | --- | --- |
| TP | head、权重或 hidden | 分摊模型计算和权重 | 通常按 head 或模型布局分片 | `AllReduce`、`AllGather` |
| DP | 请求或 batch | 提高请求吞吐 | 请求之间相互独立 | batch 或参数同步相关通信 |
| PP | 网络层 | 降低单卡模型驻留 | 不同 stage 持有不同层 | activation 传递 |
| SP/CP | sequence/context | 分摊长序列表示或 attention | 可能分摊 prefill 的序列 | `AllGather`、Ring、`All-to-All` |
| DCP | decode 上下文中的 KV token | 降低单步 KV 读取和 attention 压力 | 每个 rank 持有 KV 的 token 子集 | query 交换、LSE/output 归约 |

DCP 与 sequence parallelism 或 context parallelism 在思想上相近，都沿上下文相关维度切分工作；但 DCP 更专门针对自回归 decode 的 KV cache 访问。它的关键不是把输入序列本身切开，而是让历史 KV 的读取和局部 attention 分布到多个 rank，再合并 partial attention。

DCP 也不同于 MQA、GQA 和 MLA。后者改变 KV head 数量或 KV 表示方式，以减少每个 token 的 KV 存储量；DCP 不改变 attention 的 head 语义，而是把 KV token 分布到多个设备。两者可以组合，但优化方向不同：

- MQA/GQA/MLA 主要减少单个 token 的 KV 表示大小；
- DCP 主要减少单个 rank 每一步需要访问的 KV token 数量。

### 优势与代价

#### 优势

- **适合长上下文 decode**：上下文越长，KV 读取和 attention 的分摊空间越大；
- **单卡局部工作集更小**：每个 rank 只需要访问自己的 KV shard；
- **token 级负载较均衡**：循环切分可以使各 rank 的 KV 数量差异很小；
- **可与 TP 组合**：TP 负责模型或 head 维度，DCP 负责上下文 KV 维度；
- **不需要传输完整 KV**：通常只交换 query、partial output 和 softmax 统计量。

#### 代价

- **增加跨设备通信**：query、output 和 LSE 都可能需要交换；
- **需要数值稳定的归约**：局部 softmax 不能直接平均；
- **缓存管理复杂**：逻辑 token 位置、rank-local 物理位置和页表必须保持一致；
- **短上下文可能不划算**：固定通信成本可能超过节省的 attention 计算；
- **负载不一定完全均衡**：请求长度、padding、稀疏模式和 mask 会造成实际工作量差异；
- **对高级 attention 特性有约束**：自定义 mask、滑动窗口、特殊 score 变换等需要额外设计；
- **容错和调度更复杂**：DCP group 中任一 rank 的异常都可能影响该请求的 attention 结果。

### 总结

DCP 的核心思想是沿上下文 token 维度分布 KV cache，让每个设备只对自己的 KV 子集执行 attention，再通过 LSE-aware reduction 恢复全局 softmax 结果。

其关键数学关系是

\[
O = \frac{\sum_r \exp(L_r)O_r}{\sum_r \exp(L_r)},
\]

而不是对各 rank 的局部 output 做简单平均。

从系统角度看，DCP 的收益主要来自：

- 单卡 KV 读取量从约 \(S\) 降到约 \(S/N\)；
- 局部 attention 计算量随 KV 分片缩小；
- KV 本身通常不需要在每一步跨设备搬运。

它的代价主要来自：

- query 和 partial attention 结果的通信；
- LSE 合并带来的实现复杂度；
- 分布式 KV cache 的索引、分配和一致性管理。

因此，DCP 不是一个无条件的加速开关，而是一种针对“长上下文 decode + 高速互联 + attention/KV 读取受限”场景的专门优化。最终是否有效，应以实际的端到端 decode 延迟、吞吐、通信占比和显存带宽利用率为准。


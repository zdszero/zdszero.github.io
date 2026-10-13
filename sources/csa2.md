% CSA2

**CSA2 = 会跨层复用的 CSA + 两级稀疏检索。**

它把长上下文成本沿三个**相乘**的维度一起压：

1. **entry size**：一条 KV 有多宽（GQA / MLA 已经做过）；
2. **sequence**：隔多少个 token 记一条 KV（V4 的 CSA / HCA 已经做过）；
3. **layer**：哪些层可以有自己独立的一份（CSA2 新增）。

并且把第三维上的两件事**解耦**：

- **cache 共享**（main KV 与 indexer K 跨层共享）——省的是显存与存储；
- **索引复用**（Top-K 结果跨层复用）——省的是 indexer 计算。

解耦之后就有了三种静态模式：Full / Reindex / Reuse。

### 为什么要往层维度压

三个维度的关系是乘法而不是加法：总 KV 成本大致是

\[
\text{cost} \;\propto\; \underbrace{(\text{entry size})}_{\text{每条 KV 多大}} \times \underbrace{\frac{N}{m}}_{\text{序列方向条目数}} \times \underbrace{\#\text{layers keeping their own}}_{\text{层方向份数}} \times \text{bits}
\]

前两个维度 V4 已经压过了，第三维是剩下的空间。论文列举的前人工作在层维度上各有缺口：

| 工作 | 做法 | 缺口 |
| --- | --- | --- |
| IndexCache | 跨层复用 Top-K 索引 | 省的是 indexer 计算，**省不了 main KV 存储** |
| YOIO | 一次算完 sparse routing，全层共享 | 全网共享一份 routing，**限制性能** |
| HySparse | sparse 层复用 dense 层 KV | **仍保留全量 attention 层** |

论文的结论很直接：*index reuse alone saves no main KV storage, network-wide routing sharing limits performance, and hybrid designs still retain full attention layers*——**没有一种方法同时覆盖三个维度**。

CSA2 的答案是把三个维度乘起来用，同时把 cache 共享与索引复用拆开——拆开之后，"共享了 cache 但想让每层选出不同的 KV"这种中间形态（Reindex）才存在。

### 相比 V4 的 CSA，改了什么

| 维度 | CSA（DeepSeek-V4） | CSA2（DeepSeek-V4.1） |
| --- | --- | --- |
| 压缩窗口 | 一条压缩条目由 \(2m\) 个原始 entry 产生，相邻条目**重叠** | **去掉 overlap**，一条由 \(m\) 个 entry 产生 |
| 位置编码 | 压缩时用 absolute positional embedding 标注这 \(2m\) 个位置 | **去掉** |
| indexer K 的来源 | 从 hidden state 走**独立的压缩路径** | 由 main KV **投影**得到 |
| 全局分支结构 | CSA + HCA 混合 | **纯 CSA2**（HCA 取消），\(m=1\) 作为不压缩特例 |
| 与其它层的关系 | 每层各自持有自己的 cache 与索引 | Full / Reindex / Reuse 三模式跨层复用 |
| 深层 indexer 的打分域 | 每层都在全量可见范围打分 | Hierarchical Sparse Indexer：只在前层给出的候选池内打分（仅 decoder） |

前三行都是**减法**，论文给的理由是 *simplify the implementation and increase the training efficiency*。从结构上看：

- **去掉 overlap 是顺理成章的**。V4 需要 overlap 是因为 \(m=4\) 时窗口太短、边界 token 被切在块缝上的损失明显（见同目录《压缩数学原理》中关于覆盖完备性的讨论）。V4.1 里 encoder 用 \(m=2\)、decoder 直接用 \(m=1\)，重叠带来的收益下降，而它的代价（约 2 倍的投影参数与 kernel 内存带宽、双流 state buffer）一点没少——所以砍掉；
- **indexer K 改为从 main KV 投影**省掉了一整条独立压缩路径：V4 里 indexer 有自己的一套 `Compressor`（`head_dim=128`、独立的 gate 与 state），现在复用 main KV 即可，既有参数收益，也让 indexer K 与 main KV **天然对齐**——同一份 cache 可以同时服务 attention 与索引；
- **去掉 APE** 与上面两条同源：压缩窗口变成恰好划分之后，块内位置先验这类显式的形状参数收益变薄。（去 APE 后位置信息仍由压缩之后统一施加的 RoPE 承担，这一句是笔者的推断，报告只写了"remove"。）

### 三种模式

**三个模式共享的部分**：每层都自己计算 main Q 和 SWA KV，都用「选中的 main KV 条目 + 本层 SWA KV」拼接后做 core attention。**模式之间的差别只在三件事上**：main KV、indexer K、Top-K 索引从哪来。

| 模式 | main KV | indexer K | indexer Q | Top-K 索引 |
| --- | --- | --- | --- | --- |
| **Full** | 本层算 | 由本层 main KV 投影 | 本层算 | 本层全量打分后选出 |
| **Reindex** | 复用最近可用的 | 随 main KV 一起复用 | 本层算 | 本层**在候选池内**重新选出 |
| **Reuse** | 复用最近可用的 | 随 main KV 一起复用 | 不算 | 直接复用（针对同一份 main KV 的最近一次选择） |

```text
Full    :  本层 hidden ──► main KV ──► indexer K ──┐
                                                     ├─► indexer Q ──► 全量打分 ──► Top-K
Reindex :  main KV ◄──复用── (前层) ──► indexer K ──┘   └─► 本层 indexer Q ──► 池内打分 ──► Top-K

Reuse   :  main KV ◄──复用── (前层)，Top-K ◄──复用── (前层)，不算 indexer Q、不打分
```

两点值得单独说：

- **Full 是唯一"完整路径"**，所以它既是 cache 的产出者，也是候选池与 Top-K 的产出者；
- **Reindex 是这套设计里最有意思的一格**：cache 是共享的，但选择可以变。也就是说"哪些位置重要"这件事仍然允许逐层特化，而"这些位置长什么样"被省掉了。这一点是单纯复用索引（IndexCache / YOIO）做不到的。

**与 CED 结合时**：decoder 里被分配为 Full 模式的层，它的 global KV 不是从本层 hidden state 算出来的，而是从第 \(L/2\) 层——也就是 causal encoder 最后一层——的 hidden state 投影出来的：

\[
C_l = H_{L/2}\,W^{KV}_l, \qquad Z_l = H_{L/2}\,W^{Z}_l, \qquad l > \frac{L}{2}
\]

其中 \(C, Z\) 分别是 KV 条目及其压缩权重，\(W^{KV}_l, W^{Z}_l\) 是逐层不同的投影矩阵。Reindex 与 Reuse 模式不受这条改动影响——它们本来就只负责"复用"。

**成本上的意义**：论文给了一个很具体的数字——一个处于 Reuse 模式的 CSA2 层，prefill 只需要 **15 个 kernel**、decode 只需要 **11 个**。层层都算 indexer 的开销被摊薄了。

### Hierarchical Sparse Indexer

**问题**：跨层复用减少了 indexer 被执行的**次数**，但每次执行时它仍然要打**整段**因果可见 context 的分。在百万级 context 下，剩下的那几次全量打分本身就成了瓶颈。

**做法分成两级**：

1. **第一个 Full 模式层做全量检索**：对所有因果可见的 main KV 位置打分、选出自己的 Top-K；**同时**做一次块级选择——每个 block 取块内最大 index score 作为该块的分数，选出分数最高的若干 block，把这些 block 覆盖的位置收集成一个**候选池**。报告给的例子是选 2,048 个 block、每块 8 个位置，得到 16,384 个候选位置；
2. **后续 Reindex 层只做池内精排**：只在候选池里打分，并在这个池内选自己的 Top-K；Reuse 层则完全不再索引。

候选池在索引层之间**共享**，但每层最终的 Top-K 可以不同。

**复杂度上的关键结论**：候选池大小固定之后，**每个后续 indexer 每 query 需要打分的位置数与 context 长度无关**；只有第一个 Full 层仍然扫描全量可见范围。也就是说层次化索引把"每一次都是 \(O(N)\)"变成了"**一次 \(O(N)\) + 若干次 \(O(1)\)**"。

**训练一致性**：这个机制是 *training-aware* 的，在 post-training 阶段引入，训练与推理使用完全相同的候选域。否则深层 indexer 会在训练时见过全量位置、推理时只看得到池子，产生明显的 domain shift。

### V4.1-Flash 的实际配置

| 项 | 取值 |
| --- | --- |
| 总层数 / 结构 | 40 层 = 20 层 causal encoder + 20 层 decoder，\(d = 5120\) |
| encoder 前 2 层 | 纯 SWA（无 global attention） |
| encoder 其余 18 层 | CSA2，压缩率 \(m = 2\)；分 3 组、每组 6 层：**1 Full + 5 Reuse** |
| decoder 20 层 | CSA2，压缩率 \(m = 1\)；分 5 组、每组 4 层：第 1 组 **1 Full + 3 Reuse**，其余 4 组 **1 Reindex + 3 Reuse** |
| indexer | 32 个 indexer query head，head dim 128，attention top-k = 512 |
| main attention | 64 个 query head，head dim 512，query 压缩维度 1280 |
| Hierarchical Indexer | 最多 2,048 个 block × 8 个位置 = 最多 16,384 个候选位置 |
| 输出投影 | 8 组，每组中间维度 1024 |
| SWA | 窗口 \(n_{\text{win}} = 128\) |
| MoE / mHC | 1 shared + 384 routed experts，每 token 激活 6 个；mHC 扩展因子 4、Sinkhorn 20 次迭代 |

这张表里有两件事值得单独指出来。

**decoder 的压缩率是 1，也就是不压缩。** 换言之 CSA2 在 decoder 里其实只做"跨层复用 + Top-K 稀疏"，序列方向上的压缩（\(m=2\)）只发生在 encoder。真正让 decoder 变便宜的是：top-k 固定为 512（decode 时每 token 只读 512 个 main KV 条目）加上大部分层复用前层的 cache 与索引。这也解释了为什么 CSA2 要费力气做层次化 indexer——在 \(m=1\) 的 decoder 里，"候选数"没有被压缩稀释过，indexer 的打分成本必须靠搜索域约束来降。

**模式分配的比例很激进**：encoder 是 1:5、decoder 是 1:3。按层数算，40 层里只有 4 层（1 个 decoder Full + 3 个 encoder Full）产出全局 cache 和候选池，4 层做 Reindex，其余 30 层都是 Reuse。

### 收益与代价

**收益**（报告给出的数字）：

| 指标 | 结果 |
| --- | --- |
| global KV cache（常驻 HBM） | **890 bytes / token**，约为 V4-Flash 的 \(1/4\)；相对 V1 是 437 倍 |
| persistent KV cache（SSD / host memory） | 约为 V4-Flash 的 \(1/8\) |
| 单 token decode FLOPs | context 从 4K 扩到 1M（256 倍）只增加约 \(1/4\)（论文用词是 "nearly constant"） |

\(1/8\) 是两件事相乘的结果：

\[
\underbrace{\frac{1}{2}}_{\text{不再持久化 SWA KV}} \;\times\; \underbrace{\frac{1}{4}}_{\text{global KV 架构 + 精度优化}} \;=\; \frac{1}{8}
\]

- **SWA KV 移出持久化缓存**：V4 的持久化缓存里 SWA KV 占了近一半，但它的访问是"会话内分钟级"的，和持久缓存的 72 小时保留策略根本不匹配。V4.1 把它挪到 host DRAM 上的分布式内存池（每台机器 10% 的 DRAM），TTL 只有分钟级，靠高周转覆盖并发会话；
- **global KV 再压到 \(1/4\)**：架构上是 CSA2 的跨层复用，精度上是 FP4 global KV cache（报告称训练时就用 FP4，性能损失 marginal）。

**代价**：

- **miss 的恢复变成近似**。SWA KV 被逐出后，精确重建需要 replay \(L \times n_{\text{win}}\) 个 token 的前向；V4.1 改成 **SWA Bounded Replay**——只 replay 最近的 \(n_{\text{win}}\) 个 token，并截断 SWA 到 replay 段，接受一个近似状态。论文认为这种"把灾难性 miss 换成有界的轻微退化"是这条路线成立的前提；
- **KV cache 的量测口径要分清**：890 bytes/token 说的是常驻 HBM 的 global KV，\(1/8\) 说的是持久化部分，而 SWA KV 是另一条账（host DRAM 池 + replay 重算）。这三者混在一起谈会得出错误结论；
- **选择错误无法挽回**。候选池是后续所有 Reindex 层的搜索上界，一旦第一个 Full 层漏掉真正重要的位置，后面的层再精排也找不回来。报告在 limitations 里明确写了这一点：*Potential selection errors in CSA2 and approximate state reconstruction in SWA Bounded Replay may still cause capability degradation in untested boundary cases*。

### 训练侧怎么支撑 CSA2

共享组件的层可能被切在**不同的 pipeline stage** 上，这让"直接复用模块对象"在传统的 stage-local 执行模型下不成立。报告给了三个设计：

- **Shadow indexer**：在每个参与 stage 上放一份轻量的**可执行副本**，但共享参数保留唯一逻辑 owner——owner 负责优化器与 checkpoint，副本之间通过参数同步与梯度聚合保持一致。这样就不用让 pipeline scheduler 把"共享层"当成特殊执行单元；
- **Pipeline payload extension**：当产出方与消费方跨 pipeline 边界时，把下游需要的中间表示和 sparse routing 信息塞进已有的 p2p 通信路径，并**按 context parallel 的方式分区**，避免重复传输，同时保持梯度流；
- **Micro-batch 级共享状态管理**：跟踪并发 micro-batch 的共享状态生命周期，跨越前向、activation 重算与反向传播，直到最后一个消费者用完才释放，以免额外占用显存。

再加上 optimizer、checkpointing、warm-up、计算图追踪的适配，CSA2 才能在原有分布式训练接口和 pipeline 调度下"透明"运行。

### 笔者的分析与开放问题

以下不属于报告的结论：

- **层次化索引本质是"召回 + 精排"两阶段检索**。第一个 Full 层做完一次全量召回，产出候选池；后续层在池内精排。这和搜索/推荐系统的两阶段结构同构——差别在于这里的"召回"用的是下采样后的 main KV 自身的 indexer 打分，而不是独立的向量索引。这个类比直接指出了它的失效模式：**召回率决定上限**，精排再准也补不回漏召；
- **压缩的重心从序列轴搬到了层轴**。V4 的主线是"每 \(m\) 个 token 压成一条"，V4.1 的主线变成"每 4~6 层共用一份"。序列轴的压缩率从 4 降到 2 甚至 1，层轴的复用率却做到 3/4 ~ 5/6。这是一个相当明确的取舍方向：**复用已有计算结果，比重新设计一个更激进的压缩算子更容易训练和维护**；
- **去掉 overlap 与 APE 是一次"用表达力换工程简洁"的下注**。V4 的技术报告花了篇幅论证 overlap 的覆盖完备性，V4.1 直接砍掉；在 \(m \le 2\) 的设定下这个取舍看起来合理，但如果后续把 \(m\) 调大，重新引入 overlap 的收益可能会回来；
- **静态模式分配是个便宜且稳的设计**。Full / Reindex / Reuse 是**静态**分配而不是学出来的路由——省掉了可学路由的不稳定性和负载均衡问题，也让 kernel 形状、显存规划和 pipeline 切分都能在编译期定下来。代价是失去了逐样本自适应：无论输入多难，都是同几层做全量检索；
- **值得追问的问题**：
    - 30 层 Reuse 共享同一份 top-512 选择，会不会削弱层的特化能力？报告用"选择错误"承认了风险，但没有给出 Reuse 层与 Full 层的表示差异分析；
    - 候选池（2048 × 8 = 16384）对 1M context 而言只有约 1.6% 的覆盖率，这个比例如何随 context 长度和任务类型伸缩？
    - encoder 用 \(m=2\)、decoder 用 \(m=1\) 的分工，是否意味着"压缩只在 prefill 阶段划算"——即 prefill 是吞吐问题、decode 是延迟问题，两者的最优压缩率本来就不同？

### 与 CSA/HCA 的关系

| | V4 的 CSA / HCA | V4.1 的 CSA2 |
| --- | --- | --- |
| 压缩的轴 | 序列轴（每 \(m\) 个 token 一条） | 层轴（跨层复用）+ 序列轴（\(m=1\) 或 \(2\)） |
| 核心算子 | 逐通道 softmax 门控的凸组合（软池化） | **没变**，压缩器本身被简化（去 overlap、去 APE） |
| 稀疏选择 | Lightning Indexer + TopK | TopK 保留，新增层次化候选池把后续 indexer 的打分域压到 \(O(1)\) |
| 省的是 | KV cache 条目数 | KV cache 的**份数**（层维度）与 indexer 计算 |

一句话：**CSA2 没有换掉压缩算子，而是改变了"谁来压缩、压缩结果给谁用"**。压缩器本身仍然是那套逐通道 softmax 加权平均（见《压缩数学原理》），变化发生在算子的**复用拓扑**上——这才是 CSA2 这个名字里那个 "2" 实际指的东西。


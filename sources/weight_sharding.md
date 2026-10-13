% Weight Sharding

权重 sharding 是一种以通信换显存的推理优化：多个 GPU 不再永久保存同一份完整权重，而是各自保存一片权重；真正执行计算前，再通过集合通信临时恢复所需的数据。

它解决的主要问题是 **HBM 容量**，而不是直接减少模型的 FLOPs。是否能够提升整体性能，取决于节省的显存能否转化为更大的 batch、更多 KV cache 或更高并发，以及新增通信能否被计算隐藏。

### 最基本的形式

设完整权重为 \(W\)，通信组中有 \(N\) 个 rank：

完整权重：\(W\)

rank 0：\(W_0\)
rank 1：\(W_1\)
...
rank \(N-1\)：\(W_{N-1}\)

\[
W = \operatorname{concat}(W_0, W_1, \ldots, W_{N-1})
\]

模型初始化后，每个 rank 只保留约 \(1/N\) 的权重 shard。某个算子即将执行时，各 rank 通过通信收集所需数据：

常驻阶段：每个 rank 保存约 \(W/N\)。

执行阶段：

\[
\operatorname{AllGather}(W_0, W_1, \ldots, W_{N-1})
\longrightarrow W
\]

随后每个 rank 得到完整 \(W\)，沿用原有 GEMM / Linear。

因此，最保守的 weight sharding 实现并不会改变算子的数学计算路径，只是把完整权重从“永久保存”改成“按需恢复”。

### Shard 的维度与计算语义

线性层通常计算：

\[
Y = X W^{\mathsf T}
\]

若权重形状为：

\[
W \in \mathbb{R}^{\text{out\_features} \times \text{in\_features}}
\]

一种常见做法是沿第 0 维，也就是 `out_features` 维切分：

\[
W = \begin{bmatrix}
W_0 \\
W_1 \\
\vdots \\
W_{N-1}
\end{bmatrix},
\qquad
W_i \in \mathbb{R}^{(\text{out\_features}/N) \times \text{in\_features}}
\]

此时每个 rank 只使用自己的 shard，可以计算出部分输出：

\[
Y_i = X W_i^{\mathsf T}
\]

但如果后续算子需要完整输出，就还需要拼接这些局部输出。另一种更保守的方案是先 AllGather 权重：

\[
W_i \xrightarrow{\operatorname{AllGather}} W,
\qquad
Y = X W^{\mathsf T}
\]

两种方案的差别如下：

| 方案 | 常驻权重 | 计算时权重 | 计算量 | 主要通信 |
| --- | --- | --- | --- | --- |
| Replicated weight | 完整 \(W\) | 完整 \(W\) | 完整 GEMM | 无额外权重通信 |
| Weight shard + AllGather | 约 \(W/N\) | 恢复完整 \(W\) | 基本不变 | AllGather 权重 |
| Output-sharded GEMM | 约 \(W/N\) | 只使用 \(W_i\) | 每个 rank 的局部计算 | 输出 Gather / Reduce |

所以要明确区分：

> **权重 sharding 不一定等于 Tensor Parallel。**

如果 shard 后又把权重完整 AllGather 回来，它首先是存储优化；只有算子直接使用本地 shard 并相应处理局部输出时，才同时改变了计算并行方式。

### AllGather 权重还是 AllReduce 结果？

这取决于具体实现。

#### Weight AllGather

weight-shard 方案中，每个 rank 持有不同的数据片段：

\[
\text{rank }0: W_0,\quad
\text{rank }1: W_1,\quad
\ldots,\quad
\text{rank }N-1: W_{N-1}
\]

执行：

\[
\operatorname{AllGather}(W_0, W_1, \ldots, W_{N-1})
\]

完成后，每个 rank 都得到：

\[
\left[W_0, W_1, \ldots, W_{N-1}\right]
\]

这是“收集并拼接”，不涉及求和。

#### Output AllReduce / ReduceScatter

如果每个 rank 直接使用本地权重 shard 计算：

\[
Y_i = X W_i^{\mathsf T}
\]

那么后续可能需要：

- `AllGather`：拼接不同输出维度的结果；
- `AllReduce`：累加不同 rank 的部分和；
- `ReduceScatter`：累加后只保留输出的一片。

因此，不能笼统地说“sharding 的代价是 AllReduce”。对于“存储 shard、计算前恢复完整权重”的方案，核心通信是 **AllGather 权重**；对于真正的 output-sharded GEMM，通信可能发生在输出侧。

### Flat AllGather

一个层通常不只需要一个 replicated weight。例如可能需要 \(W_{\text{down}}\)、\(W_{\text{up}}\)、\(W_{\text{gate-up}}\) 和 \(W_{\text{shared-down}}\)。

最直接的实现是分别通信：

\[
\begin{aligned}
&\operatorname{AllGather}(W_{\text{down}}^{\text{shard}}) \\
&\operatorname{AllGather}(W_{\text{up}}^{\text{shard}}) \\
&\operatorname{AllGather}(W_{\text{gate-up}}^{\text{shard}}) \\
&\operatorname{AllGather}(W_{\text{shared-down}}^{\text{shard}})
\end{aligned}
\]

这样会产生多次 collective launch、多个通信 buffer 和多组 stream/event 依赖。

#### 把多个权重序列化

Flat AllGather 先定义一个连续的 flat layout。设完整 flat buffer 为 \(T\)：

\[
T = \operatorname{concat}\left(
W_{\text{down}},
W_{\text{up}},
W_{\text{gate-up}},
W_{\text{shared-down}},
\text{padding}
\right)
\]

每个 rank 的本地输入是完整 flat buffer 的一个等大小 chunk。若 chunk 大小为 \(C = |T|/N\)，则 rank \(i\) 保存：

\[
T_i = T[iC:(i+1)C],
\qquad 0 \leq i < N
\]

然后只执行一次：

```python
dist.all_gather_into_tensor(
    output_tensor=flat_full,
    input_tensor=flat_shard,
    group=group,
)
```

AllGather 完成后，每个 rank 都有完整的 \(T\)，再通过预先记录的 byte offset 创建各个权重的 view。对任意权重 \(W_j\)，有：

\[
W_j = \operatorname{view}\left(T[o_j:o_j+s_j]\right)
\]

其中 \(o_j\) 是该权重在 flat buffer 中的 byte offset，\(s_j\) 是它占用的字节数。

这些 view 只是重新解释同一块内存，不需要再次拷贝权重。

#### 为什么“一次”可以恢复多个权重？

因为通信库看到的只是一个连续的 byte buffer，它不需要知道 flat buffer 内部的逻辑边界：

```text
多个逻辑 weight
    ↓ serialize / concatenate
一个 flat shard
    ↓ 一次 AllGather
一个 flat full buffer
    ↓ offset + view
多个逻辑 full weight
```

“一次”指的是**同一层的一组权重发起一个 collective**，并不是整个模型所有层只 AllGather 一次。不同层通常仍然需要在各自执行前准备各自的 flat weight。

Flat AllGather 主要减少：

- collective launch 次数；
- NCCL 调度和同步的固定开销；
- stream/event 依赖；
- 多个临时 buffer 的管理成本。

它通常不会显著减少总传输字节数。多个 weight 的总大小仍然是通信 payload 的主要决定因素。

### 通信量为什么可能很大？

是的，AllGather 权重的通信量可能很大。假设某层需要恢复的权重总大小为 \(S\)，通信组大小为 \(N\)：

\[
\text{每个 rank 常驻：约 } \frac{S}{N},
\qquad
\text{AllGather 后：每个 rank 得到 } S
\]

以 ring AllGather 的理想情况估算，每个 rank 的传输量通常约为：

\[
\frac{N-1}{N}S
\]

具体读写量会依赖 NCCL 采用的算法、协议和硬件拓扑。Flat 只把多个通信合并成一个操作，并不会把通信成本从 \(S\) 变成 \(S/N\)。

因此这种优化的本质是：

\[
\text{HBM 容量} \longleftrightarrow \text{GPU 互联带宽}
\]

如果模型本来就能完整放入显存，且推理性能受通信限制，那么新增权重通信可能不划算；如果模型受 HBM 容量限制，sharding 释放的空间可以用于 KV cache、batch 或并发，整体收益可能更大。

### 如何隐藏权重通信？

常见方法是 lookahead 和 side stream：

```text
Layer i：
    主 stream：计算 Layer i
    side stream：AllGather Layer i+1 的权重

Layer i+1：
    等待 AllGather 完成
    使用已经准备好的 full weight
```

理想情况下：

\[
T_{\text{allgather}} \leq T_{\text{compute}}
\]

那么通信可以被前一层计算覆盖，额外暴露的延迟较小。实际效果取决于：

- batch size 和 token 数；
- 权重大小；
- NVLink / PCIe / 跨节点带宽；
- 当前 TP、EP、DP 通信压力；
- NCCL stream 和 Copy Engine 调度；
- full buffer 是否可以安全复用。

Prefill 的矩阵计算通常更重，比较容易覆盖通信；小 batch decode 的 GEMM 较瘦，AllGather 更容易暴露在关键路径上。

### 显存模型

设模型中被 sharding 的 replicated weight 总大小为 \(R\)，shard group size 为 \(N\)。

不使用 sharding 时，每个 rank 的长期权重显存约为：

\[
M_{\text{replicated}} \approx R
\]

使用 weight shard 后，长期保存的 shard 约为：

\[
M_{\text{sharded}} \approx \frac{R}{N}
\]

但运行时还需要：

```text
各层的 persistent shard / flat shard
+ 一个或少量可复用的 full-weight workspace
+ 通信和 event 管理所需的额外资源
```

因此实际显存收益不是简单地把所有权重变成 \(R/N\)，而是释放了原本长期复制的 full weight。一般需要通过 staging pool 复用 full buffer，避免所有层同时恢复完整权重。

还需区分加载峰值：如果实现先加载完整 checkpoint，再完成 requant、切片和释放，那么它主要降低的是**运行期间常驻显存**，不一定降低模型加载瞬间的显存峰值。

### 与 TP、EP 的关系

#### 与 Tensor Parallel 的区别

Tensor Parallel 通常从计算定义出发：权重和输入按某个维度切分，每个 rank 直接执行局部 GEMM，再通过集合通信得到后续需要的布局。

Weight sharding 可以只从存储角度出发：权重加载完成后按 rank 保存 shard，执行时再恢复 full weight。后者不一定减少计算量。

#### 与 Expert Parallel 的区别

Expert Parallel 是把不同 expert 放到不同 rank，token 通过 dispatch/combine 到负责的 expert。它解决的是 expert 计算和权重的分布问题，通常伴随 All-to-All 通信。

Weight sharding 关注的是某个 replicated weight 是否需要在每个 rank 永久保存。它可以和 TP、EP、DP 共存，但 shard group、权重布局和通信时机必须明确，不能把几种并行策略的通信组混为一谈。

### 主要收益

1. **降低常驻 HBM。** 尤其适合大模型中的 replicated projection、shared expert 或其他复制参数。
2. **释放 KV cache 空间。** 同样显存下可以提高上下文长度、batch size 或并发数。
3. **不必立刻重写算子。** 采用“按需恢复 full weight”时，可以继续复用已有 GEMM 和模型前向路径。
4. **可以结合通信重叠。** side stream、lookahead、NCCL Copy Engine 等机制能够降低通信暴露时间。
5. **可按层启用。** 对显存占用最大或收益最明显的层进行选择性 sharding，便于在显存和延迟之间折中。

### 主要代价

1. **权重通信量大。** 每层恢复的权重可能达到数百 MB，尤其在 decode 中容易成为瓶颈。
2. **不一定减少 FLOPs。** 计算前恢复 full weight 的方案仍然执行原来的完整 GEMM。
3. **不一定减少计算阶段的权重读取。** full weight 恢复后，每个 rank 仍可能读取完整权重。
4. **需要 full-weight workspace。** 运行时并非完全没有完整权重，只是通过 pool 复用并缩短其生命周期。
5. **增加通信竞争。** 可能与 TP AllReduce、EP All-to-All、DP Gather/Scatter 等通信争用带宽。
6. **实现复杂。** 需要处理 stream、event、buffer 生命周期、CUDA Graph capture 和多请求并发。
7. **硬件和软件有要求。** 高速 NVLink、合适的 NCCL 版本、symmetric memory 或 Copy Engine 能力会显著影响效果。
8. **量化权重更复杂。** packed weight、scale、zero point 和 layout 必须一起设计，不能只切原始 tensor。

### 什么时候值得使用？

适合考虑 weight sharding 的情况：

```text
模型受 HBM 容量限制
+ GPU 间互联足够快
+ 有较大的 batch 或 prefill 计算可覆盖通信
+ 释放的显存能转化为更多 KV cache / 并发
```

需要谨慎的情况：

```text
模型本来就能完整驻留
+ 小 batch decode
+ 跨节点通信占主导
+ TP / EP 通信已经拥塞
+ 目标权重很大但没有足够计算用于 overlap
```

实际判断不能只看显存节省，还要测量：

- AllGather 的实际带宽和耗时；
- 通信暴露在关键路径上的比例；
- prefill 吞吐和 decode ITL；
- batch size 变化后的整体吞吐；
- 释放显存带来的 KV cache / 并发收益；
- 与不开启 sharding 的端到端对比。

### 总结

权重 sharding 可以概括为：

```text
完整权重长期复制
        ↓
每个 rank 只保存一个 shard
        ↓
按层、按需恢复 full weight
        ↓
复用已有计算路径
```

针对“计算前恢复完整权重”的实现，最准确的描述是：

> **用权重 AllGather 换取 replicated weight 的常驻显存。**

Flat AllGather 只是把同一层的多个逻辑 weight 序列化到一个 flat buffer，用一次 collective 完成收集，再通过 offset 和 view 拆回多个 weight；它减少的是 collective 调度开销，而不是总通信字节数。

最终收益取决于：

\[
\text{显存节省带来的系统收益}
\mathop{\gtrless}
\text{新增权重通信和同步带来的性能损失}
\]


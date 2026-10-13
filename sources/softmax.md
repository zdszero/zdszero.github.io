% Softmax

softmax 把一行 logits 映射为概率分布：

\[
\text{softmax}(x)_j = \frac{e^{x_j}}{\sum_k e^{x_k}}
\]

直接算会溢出，所以用**数值稳定（safe）softmax**：先减去行最大值，保证指数非正、不会上溢：

\[
\text{softmax}(x)_j = \frac{e^{x_j - m}}{\sum_k e^{x_k - m}}, \quad m = \max_k x_k
\]

这是经典的**三遍**结构：求 max → 求 \(\sum e^{x-m}\) → 归一化。CPU 参考 `softmax_forward_cpu` 用 `double` 累加求和以获得高精度基准。

#### Online softmax

[Online softmax](http://arxiv.org/abs/1805.02867) 把「求 max」和「求 sum」融进**一遍**：维护运行最大值 \(m\) 与运行归一化项 \(d\)，每见到一个新元素就更新，并在 max 变大时用**修正因子** \(e^{m_{\text{old}} - m_{\text{new}}}\) 追溯地缩放已累加的和：

```c
float maxval_prev = maxval;
if (inp_row[j] > maxval) {
    maxval = inp_row[j];
    sum = sum * expf(maxval_prev - maxval) + expf(inp_row[j] - maxval);
} else {
    sum += expf(inp_row[j] - maxval);
}
```

两个在线部分的合并满足结合律，可以并行规约。llm.c 用 `SumMax` 结构承载 \((m, d)\) 对：

```c
struct __align__(8) SumMax { float maxval; float sum; };

__device__ SumMax reduce_sum_max_op(SumMax a, SumMax b) {
    bool a_bigger = (a.maxval > b.maxval);
    SumMax bigger  = a_bigger ? a : b;
    SumMax smaller = a_bigger ? b : a;
    SumMax res;
    res.maxval = bigger.maxval;
    res.sum = bigger.sum + smaller.sum * expf(smaller.maxval - bigger.maxval);
    return res;
}
```

即「取较大的 max，把较小那份的和按 \(e^{\Delta m}\) 缩放后相加」。这正是 Flash Attention 的核心运算，[Attention](../attention/) 一章会再用到它。

### version 1：一线程一行（朴素三遍）

`i = blockIdx.x*blockDim.x + threadIdx.x` 映射一行，单线程串行跑三遍。正确但没有行内并行，且三遍扫 global memory，慢。

### version 2：一 block 一行 + 共享内存规约

`idx = blockIdx.x` 一 block 处理一行，block 内线程粗化分摊 \(C\) 列，max 与 sum 都用共享内存树形规约：

```c
for (int i = tid; i < C; i += block_size) maxval = fmaxf(maxval, x[i]);
shared[tid] = maxval;
for (int stride = block_size / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (tid < stride) shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
}
```

改进：整个 block 协作处理一行，max/sum 并行完成；访存也合并了。

### version 3：warp 内规约（限定 block_size = 32）

把共享内存树规约换成 **warp shuffle**（`warpReduceMax`/`warpReduceSum`），完全在寄存器里做，无需共享内存、无需 `__syncthreads()`：

```c
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}
...
maxval = warpReduceMax(maxval);
float offset = __shfl_sync(0xFFFFFFFF, maxval, 0);  // 广播给全 warp
```

**为什么必须 block_size = 32**：`__shfl_*` 只能在**同一个 32 线程 warp 内**交换寄存器。kernel 没有 warp 间的汇总步骤，所以整个 block 必须恰好是一个 warp，launcher 硬编码 `block_size = 32`。否则各 warp 的局部结果无法合并。

### version 4：warp + 共享内存两级规约（任意 block_size % 32 == 0）

在 v3 的 warp 规约之上，补一层 **warp 间的共享内存规约**：每个 warp 的 lane 0 把局部结果写进 `max_or_sum_storage[warpId]`，`__syncthreads()` 后由 thread 0 汇总所有 warp：

```c
maxval = warpReduceMax(maxval);
if (laneId == 0) max_or_sum_storage[warpId] = maxval;
__syncthreads();
if (tid == 0) {
    float val = max_or_sum_storage[0];
    for (int i = 1; i < warpsPerBlock; i++) val = fmaxf(val, max_or_sum_storage[i]);
    max_or_sum_storage[0] = val;
}
```

改进：解除了 block_size = 32 的限制，**任意 `block_size % 32 == 0`（64/128/…/1024）都能一 block 处理一行**，只需 `block_size/32` 个共享内存槽。注释称其为「hopefully the most efficient version」。这套「warp shuffle 打底 + 共享内存做 warp 间」是规约 kernel 的标准写法。

### version 5 / 6：online softmax（两遍 → 并行两遍）

- **version 5**：online 算法的朴素移植，一线程一行。只需**两遍**（融合的 max+sum，再归一化），但仍单线程，慢。
- **version 6**：online softmax 并行版。一个 warp 一行，行内用 `reduce_sum_max_op` 累加 `SumMax`，再用 cooperative groups 的 `cg::reduce` 跨 warp 合并：

```c
SumMax sm_partial = {-INFINITY, 0.0f};
for (int i = warp.thread_rank(); i < C; i += warp.size())
    sm_partial = reduce_sum_max_op(sm_partial, { x[i], 1.0f });
SumMax sm_total = cg::reduce(warp, sm_partial, reduce_sum_max_op);
for (int i = warp.thread_rank(); i < C; i += warp.size())
    __stcs(out + idx*C + i, expf(x[i] - sm_total.maxval) / sm_total.sum);  // 流式写
```

改进：把 max 遍和 sum 遍**融合成一遍**规约，对 global memory 的扫描从三遍降到两遍；输出用 `__stcs` 流式写（结果不会马上被读回）。

### version 7：面向超大 C 的展开

结构同 v4，但加了 `UNROLL_FACTOR = 8` 的**寄存器数组展开**，每轮循环发起 8 笔独立 load，用**内存级并行（MLP）**掩盖访存延迟；越界索引用 `min(C-1, ...)` 夹住做到无分支、始终发射；并把 `exp` 与 `sumval` 累加融进同一循环：

```c
for (int i = tid; i < C; i += blockDim.x * UNROLL_FACTOR) {
    float reg[UNROLL_FACTOR];
    #pragma unroll
    for (int u = 0; u < UNROLL_FACTOR; u++)
        reg[u] = __ldcs(&x[min(C - 1, i + u*blockDim.x)]);
    #pragma unroll
    for (int u = 0; u < UNROLL_FACTOR; u++)
        if (i + u*blockDim.x < C) {
            float o = expf(reg[u] - offset);
            y[min(C-1, i + u*blockDim.x)] = o;
            sumval += o;  // 与 exp 合并进同一循环
        }
}
```

改进：对很长的行（如词表维度 `V=50257`），一次发起多笔在途 load 比逐个 load 更能打满带宽。注释也点明这对小 C 反而不划算（有冗余读取）。

### 小结

| version | 手段 | 遍数 / 约束 |
|---|---|---|
| 1 | 一线程一行 | 3 遍，无并行 |
| 2 | 一 block 一行 + 共享内存树规约 | 3 遍，合并访问 |
| 3 | warp shuffle 规约 | 3 遍，须 block_size=32 |
| 4 | warp + 共享内存两级规约 | 3 遍，任意 32 倍数 block |
| 5 | online 朴素移植 | 2 遍，无并行 |
| 6 | online + cg 并行规约 | 2 遍并行 |
| 7 | v4 + 寄存器展开（大 C） | 3 遍，MLP 掩盖延迟 |

主线：**并行规约 → warp shuffle → warp+共享内存突破 block 限制 → online 算法减少遍数 → 针对数据规模（大 C）做访存展开**。


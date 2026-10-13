% LayerNorm

对每一行 \(x \in \mathbb{R}^{C}\)（即 `inp[b,t,:]`），LayerNorm 计算：

\[
\mu = \frac{1}{C}\sum_{i} x_i, \qquad
\sigma^2 = \frac{1}{C}\sum_{i}(x_i - \mu)^2
\]

\[
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad
y_i = \hat{x}_i \cdot w_i + b_i
\]

其中 `float eps = 1e-5f`。反向传播需要用到 \(\mu\) 与 \(1/\sqrt{\sigma^2+\epsilon}\)（记作 `rstd`），前向会把它们缓存下来。

CPU 参考 `layernorm_forward_cpu` 是经典的**两遍法**：第一遍累加求 \(\mu\)，第二遍用 \(\mu\) 累加 \((x_i-\mu)^2\) 求方差，再做归一化与仿射变换。「求方差必须先知道均值」正是两遍法的由来。

### 优化主线

LayerNorm 的核心是**行内规约**（对 \(C\) 个元素求和）。整章的优化都围绕「怎么让一堆线程协作把这个求和做快、并保持访存合并」展开。

### version 1：一线程一行（朴素移植）

`idx` 直接映射到一个 `(b,t)` 行，单线程串行跑完两遍求和加归一化——就是 CPU 代码套了个 `if (idx < N)`：

```c
int idx = blockIdx.x * blockDim.x + threadIdx.x;
const float* x = inp + idx * C;
float m = 0.0f;
for (int i = 0; i < C; i++) { m += x[i]; }
m = m / C;
float v = 0.0f;
for (int i = 0; i < C; i++) { float xshift = x[i] - m; v += xshift * xshift; }
v = v / C;
float s = 1.0f / sqrtf(v + eps);
```

问题：**访存不合并**。相邻线程处理的行相隔 `C` 个 float，一个 warp 内的读取地址是跨步 `C` 的，带宽利用极差；而且每个线程要串行扫 3 遍自己的行。

### version 2：并行到 B,T,C（三个 kernel + 共享内存规约）

拆成 `mean_kernel`、`rstd_kernel`、`normalization_kernel` 三个 kernel。前两个改成**一个 block 处理一行**，block 内线程用「线程粗化」分摊 \(C\) 个元素，再做**共享内存树形规约**：

```c
float sum = 0.0f;
for (int i = tid; i < C; i += block_size) { sum += x[i]; }  // 粗化 + 合并访问
shared[tid] = sum;
__syncthreads();
for (int stride = block_size / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (tid < stride) { shared[tid] += shared[tid + stride]; }
}
if (tid == 0) { mean[idx] = shared[0] / C; }
```

改进：`for (i = tid; i < C; i += block_size)` 让同一 warp 访问连续地址，**修复了合并**；求和从 \(O(C)\) 串行变成 \(O(\log \text{block\_size})\) 的并行规约。代价：**三次 kernel launch**，且 `mean`/`rstd` 要写回 global memory 再读回。

### version 3：cooperative groups，一个 warp 处理一行

把三个 kernel 重新**融合成一个**，用 cooperative groups 以「一个 warp 一行」的粒度做规约，配合 `__restrict__` 与流式访存提示 `__ldcs`/`__stcs`：

```c
namespace cg = cooperative_groups;
cg::thread_block block = cg::this_thread_block();
cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();  // 该 warp 负责哪一行
...
float sum = 0.0f;
for (int i = warp.thread_rank(); i < C; i += warp.size()) { sum += x[i]; }
sum = cg::reduce(warp, sum, cg::plus<float>{});   // warp 内规约，无需共享内存/同步
float m = sum / C;
```

改进：单次 launch，中间结果不再往返 global memory；`cg::reduce` 在寄存器层面做 warp 规约，省掉了 v2 的共享内存和 `__syncthreads()`。`meta_group_size()`/`meta_group_rank()` 分别是「一个 block 里有几个 warp」「本 warp 的序号」，用来把 warp 映射到行。

### version 4：单遍方差 `Var(x) = E[x²] − E[x]²`

v3 仍是两遍（先算 \(\mu\)，再回头算方差）。version 4 用恒等式

\[
\sigma^2 = \mathbb{E}[x^2] - (\mathbb{E}[x])^2
\]

在**同一个循环里**同时累加 `sum` 和 `sum2`，只需**读一遍 x**：

```c
float sum = 0.0f;   // Σx
float sum2 = 0.0f;  // Σx²
for (int i = warp.thread_rank(); i < C; i += warp.size()) {
    float xi = x[i];
    sum += xi;
    sum2 += xi * xi;
}
sum  = cg::reduce(warp, sum,  cg::plus<float>{}) / C;
sum2 = cg::reduce(warp, sum2, cg::plus<float>{}) / C;
float m = sum;
float var = sum2 - sum * sum;
float s = rsqrtf(var + 1e-5f);
```

改进：把对 `x` 的读取从两遍减到**一遍**。对 memory-bound 的规约算子，读取次数几乎直接决定耗时，这一步收益明显。（数值上 `E[x²]-E[x]²` 在极端分布下不如两遍法稳，但对 LayerNorm 的输入范围足够。）

### version 5：block-per-row 代替 warp-per-row

算法同 v4，但把粒度从「一个 warp 一行」放大到「一个 block 一行」（`idx = blockIdx.x`，最多 1024 线程一行）。这需要**两级规约**：先每个 warp 用 `cg::reduce`，把局部结果写进 `__shared__ float shared_sum[32]`，`__syncthreads()` 后再由第一个 warp 汇总各 warp 的结果：

```c
float warp_sum  = cg::reduce(warp, thread_sum,  cg::plus<float>{});
shared_sum[warp_id] = warp_sum;
__syncthreads();
warp_sum = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;  // 越界补 0
float block_sum = cg::reduce(warp, warp_sum, cg::plus<float>{});
```

改进：warp（32 线程）一行时，每行最多 32 线程协作，粗化循环较长；block 一行时最多 1024 线程协作，**每个线程的串行段更短、占用率更高**，对大 \(C\) 更友好。代价是多了一层 warp 间的共享内存规约。

> 文件里还有一个 **kernel 6**（顶部注释没列出），借鉴 `fused_residual_forward` 的思路：warp-per-row + `x128` 向量化 128-bit 访存 + 把 `weight`/`bias`/输入行缓存进动态共享内存复用，并在需要 >48KiB 共享内存时调用 `cudaFuncSetAttribute`，失败则回退到 v5。它把前面所有手段叠加到了一起。

### 小结

| version | 关键手段 | 解决的问题 |
|---|---|---|
| 1 | 一线程一行 | —（基线，访存不合并） |
| 2 | 一 block 一行 + 共享内存树规约（三 kernel） | 合并访问、并行规约 |
| 3 | 融合成单 kernel + cooperative groups warp 规约 | 去掉中间落地与共享内存往返 |
| 4 | `Var=E[x²]−E[x]²` 单遍 | 对 x 只读一遍 |
| 5 | block-per-row 两级规约 | 大 C 下更高并行度/占用率 |
| 6 | x128 向量化 + 共享内存缓存复用 | 逼近带宽上限 |

规约类算子的优化范式：**修复合并 → 并行规约（warp shuffle 优先，共享内存做 warp 间）→ 融合减少落地 → 单遍算法减少扫描 → 选对 warp/block 粒度 → 向量化访存**。


% Cuda Graph


### 原理

#### 它解决的是什么问题

GPU kernel 是异步提交的：CPU 把命令写进 stream 队列后立刻返回，GPU 再按顺序消费。单次提交的 CPU 侧成本（参数打包、驱动调用、写命令缓冲区）通常在几微秒量级，而一次 kernel 的 GPU 执行时间可能只有一两微秒。

设第 \(i\) 个 kernel 的 CPU 提交时间是 \(t^{\text{cpu}}_i\)，GPU 执行时间是 \(t^{\text{gpu}}_i\)，一共有 \(N\) 个 kernel。在 CPU 能持续提前排队的理想情况下：

\[
T_{\text{wall}} \approx \max\left(\sum_{i=1}^{N} t^{\text{cpu}}_i,\ \sum_{i=1}^{N} t^{\text{gpu}}_i\right)
\]

当 \(\sum_i t^{\text{cpu}}_i > \sum_i t^{\text{gpu}}_i\) 时，GPU 会在两次 kernel 之间空转等待命令，流水线出现断档（bubble），这称为 **launch-bound**。decode 阶段每个 step 要跑几十到上百个小 kernel，正是这种典型场景。

CUDA Graph 的做法是：把一整段固定的命令序列**录制**成一张 DAG，之后每次执行只需提交一次：

\[
\Delta T \approx \sum_{i=1}^{N} t^{\text{cpu}}_i - t^{\text{cpu}}_{\text{graph}}
\]

其中 \(t^{\text{cpu}}_{\text{graph}}\) 是单次 graph launch 的成本，量级接近一次普通 kernel launch，与 \(N\) 基本无关。这就是它把 \(N\) 次提交压缩成 \(1\) 次的意义。

#### 它不改变什么

CUDA Graph 只优化「CPU 提交命令」这一段：

- 不减少 GPU 的计算量，不减少 kernel 数量；
- 不做算子融合，图里本来就是那些 kernel；
- 不改变 stream/event 的同步语义；
- kernel 总时间远大于 CPU 提交时间时（大 GEMM、大 batch、prefill 的 compute-bound 区间），收益接近零。

#### capture 与 replay 的生命周期

```cpp
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
launch_kernel_a<<<grid_a, block, 0, stream>>>(d_a, d_b);
launch_kernel_b<<<grid_b, block, 0, stream>>>(d_b, d_c);
cudaMemcpyAsync(d_out, d_c, bytes, cudaMemcpyDeviceToDevice, stream);
cudaStreamEndCapture(stream, &graph);

cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);

for (int step = 0; step < steps; ++step) {
    update_static_inputs();          // 往 capture 时的同一块 buffer 写入新数据
    cudaGraphLaunch(graph_exec, stream);   // 一次提交，内部 N 个 kernel
}
```

三个阶段各自的语义：

| 阶段 | 做什么 | 关键点 |
| --- | --- | --- |
| capture | 把 stream 上的命令记录成 DAG，**不真正执行** | 期间禁止 CPU 侧阻塞同步和依赖 CPU 值的控制流 |
| instantiate | 校验拓扑、规划内存、生成可执行的 `cudaGraphExec_t` | 一次性成本，不要放进热路径 |
| replay | 一次提交，GPU 按 DAG 执行 | 图的参数被固化，输入必须写回 capture 时的同一地址 |

PyTorch 里的对应写法是 `torch.cuda.CUDAGraph` 配合 `torch.cuda.graph` 上下文：

```python
g = torch.cuda.CUDAGraph()
static_in = torch.empty_like(x)
static_out = None

# 先在 side stream 上 warmup，触发 cudaMalloc、cuBLAS 初始化、JIT 等一次性开销
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        static_out = model(static_in)
torch.cuda.current_stream().wait_stream(s)

with torch.cuda.graph(g):          # capture
    static_out = model(static_in)

for step in range(steps):
    static_in.copy_(next_input())  # 写回同一块 buffer
    g.replay()                     # replay
    consume(static_out)
```

warmup 不可省略：如果第一次调用里包含显存分配或库初始化，这些操作会被录进图里，replay 时行为可能不正确。`torch.cuda.graph` 会为捕获到的张量分配一个私有内存池，保证多次 replay 之间地址稳定。

### 图能复用的条件

一段代码能否被同一张图反复 replay，取决于四个「一致」：

1. **kernel 数量一致**；
2. **执行顺序（DAG 拓扑）一致**；
3. **grid / block 一致**；
4. **所有 tensor 地址一致**。

逐条对照 decode 阶段的 attention：

| 条件 | decode 的情况 | 结果 |
| --- | --- | --- |
| kernel 数量 | 每个 step 固定 | 满足 |
| 执行顺序 | 固定 | 满足 |
| grid / block | \(q\_len\) 固定，batch padding 到固定档位 | 满足 |
| tensor 地址 | KV cache、Q buffer、workspace 全部预分配 | 满足 |

真正决定图能不能复用的，不是「shape 变不变」，而是「**launch topology 变不变**」。只要 block 数量、每个 block 处理多少 query、kernel 数量保持一致，kernel 内部循环次数变化是无所谓的。

### Attention 为什么通常不把 kv_len 放进 grid

以 FlashAttention / paged attention 为例，decode 阶段：

```text
Q: [bs, 1, n_heads, head_dim]
K: [bs, kv_len, n_heads, head_dim]
```

attention 的核心计算是

\[
\mathrm{softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt{d}}\right)V
\]

每个 query token 都要算一整行 attention。大多数实现的并行策略是

\[
\text{grid.x} \approx \frac{bs \times n\_heads \times q\_len}{\text{queries per block}}
\]

也就是「一个 thread block 负责一个 query block」。decode 时 \(q\_len = 1\)，所以 grid 规模只和 \(bs \times n\_heads\) 有关，与 kv_len 无关：

- \(q\_len\) 从 1 变成 8，grid 会扩大约 8 倍，launch 参数改变，图失效；
- \(kv\_len\) 从 1k 涨到 32k，grid 不变。

为什么 grid 不依赖 kv_len？因为 kv_len 可以到几千甚至几万，如果写成 `grid.x = kv_len`，grid 会直接爆炸。更常见的做法是每个 block 处理一个 query，在 block 内部循环 kv_len 并分 tile 读 K/V：

```cpp
for (int i = 0; i < kv_len; i += TILE) {
    load K tile;
    load V tile;
    accumulate;
}
```

这里 kv_len 是循环上界，不是 grid 维度。于是只要 KV cache 的物理大小不变，grid 就固定，图就可以 replay。

需要强调的是：**GPU kernel 的并行维度由「怎么分配工作」决定，不由数学维度决定**。attention 在数学上是四维张量 \((bs, q\_len, kv\_len, hidden)\)，工程上可以选择并行在 q 上、并行在 kv 上或混合。decode 实现不把 kv_len 放进 grid，是为了图友好与 occupancy 稳定，这是设计选择，不是理论必然。

### Prefill 为什么难 graph

prefill 阶段：

```text
Q: [bs, seq_len, hidden]
```

此时 \(q\_len = seq\_len\)，而 prompt 长度是用户输入决定的、天然可变的：

- \(q\_len\) 变 \(\rightarrow\) grid 变 \(\rightarrow\) launch 参数变 \(\rightarrow\) 图失效。

所以 prefill 的图复用问题出在 \(q\_len\)，不是 kv_len。实践中处理不定长输入有几条路线：

| 方案 | 做法 | 代价 |
| --- | --- | --- |
| 不做 graph | prefill 走 eager，只对 decode 用图 | 放弃 prefill 的 launch 收益 |
| padding 到固定档位 | 把 \(q\_len\) pad 到预先定义的 bucket | 浪费算力，短 prompt 被拉长 |
| 分段 graph | 只把与序列长度无关的子图 capture | 图覆盖不完整，边界仍有 launch 开销 |
| shape 特化 | 对每个 bucket 各 capture 一张图 | 图数量随桶数增长，显存与管理成本上升 |

prefill 本身是 compute-bound（大 GEMM 占主导），CPU 提交时间相对总时间占比低，因此「不做 graph」往往就是合理选择。这也解释了为什么主流推理框架默认只对 decode 启用 CUDA Graph。

### Batch size 分档

decode 能 graph 的前提是 grid 固定，但线上的 batch size 是动态的。解决办法是预先 capture 若干 batch size 档位（bucket），运行时把真实 batch pad 到不小于它的最小档位：

\[
b_k = \min\left\{\, b_m \in \mathcal{B} \;\middle|\; b_m \ge b \,\right\}
\]

其中 \(\mathcal{B}\) 是预先 capture 的档位集合，例如 \(1, 2, 4, 8, 16, \ldots\)。padding 带来的算力浪费比例为

\[
\eta = 1 - \frac{b}{b_k}
\]

档位越密，\(\eta\) 越小，但要维护的图越多，capture 阶段的显存与时间成本也越高。常见策略是在小 batch 区间用密集档位（1、2、4、8），大 batch 区间用固定步长或几何增长。

一个容易被忽略的副作用是 MoE：pad 出来的 dummy token 如果参与路由统计，会污染 expert 的负载均衡。实现上通常要保证 dummy token 只占位、不参与统计与通信。

### Piecewise CUDA Graph

整图 graph 化有一个硬前提：从入口到出口，中间不能有需要 CPU 参与决策的地方。真实推理图往往不满足：

- MoE 的动态路由结果可能需要在 host 侧做后续调度；
- attention backend 的选择依赖运行时状态；
- 动态 batch、动态并行策略调整需要 CPU 判断；
- PagedAttention 的 block table、序列长度等是动态的。

**piecewise CUDA Graph** 的思路是把计算图切成若干段，每段单独 capture 成一张子图，段与段之间保留普通的 kernel launch 和 CPU 逻辑：

```text
eager:  [op A]  [op B]  [attention]  [op C]  [op D]
graph : [--- graph 1 ---]  eager  [--- graph 2 ---]
                          ^ CPU 在这里做决策、选 backend、改 block table
```

- 收益来自段内的批量提交，段数越少、段越长，收益越大；
- 段边界仍然是普通 launch，存在开销，段太碎会抵消收益；
- 每张子图仍然要求段内 launch topology 固定，所以通常还要叠加 batch 分档。

vLLM V1 的 `CUDAGraphMode` 提供了 `NONE` / `PIECEWISE` / `FULL` 等模式，其中 piecewise 模式就是按 attention 这类动态算子把图切开，让 attention 走 eager、其余 shape 稳定的部分 replay。SGLang 则通过 `--cuda-graph-max-bs`、`--cuda-graph-bs` 这类参数控制 capture 的 batch 档位，用 `--disable-cuda-graph` 整体关闭。PyTorch 侧对应的是 `torch.compile(mode="reduce-overhead")`，它用 CUDA Graph 加静态输入 buffer 来自动做这件事。

### 与其它机制的关系

| 机制 | 交互点 |
| --- | --- |
| stream / event | capture 支持多 stream 的 fork/join，图内部会保留这些依赖边 |
| CPU-GPU 同步 | capture 期间禁止 CPU 阻塞同步；replay 时 CPU 只需提交一次 |
| 通信库 | NCCL 的 collective 可以被捕获；CustomAllreduce 这类基于 IPC 共享内存的实现通常需要切换到可捕获的 out-of-place 路径 |
| 显存管理 | 需要专用内存池（PyTorch 的 graph pool）保证地址稳定 |
| 计算图 / 编译器 | CUDA Graph 是执行层的「静态化」，与编译器做算子融合是两个正交维度 |

### 常见坑

- **warmup 不足**：第一次调用中的 `cudaMalloc`、cuBLAS/cuDNN 初始化、JIT 编译会被录进图里，导致 replay 行为异常。
- **输入地址漂移**：图记录的是指针。replay 前必须把新输入写回 capture 时用的那块静态 buffer，不能换 tensor。
- **显存池不匹配**：跨 capture 复用同一块显存、或让 caching allocator 在图外重新分配，会让地址失效或触发保护机制。
- **图内出现同步**：`cudaStreamSynchronize`、`.item()` 这类操作无法被正确捕获，capture 会直接失败或录出错误结果。
- **档位与图数量爆炸**：batch 档位乘以 seq_len 档位再乘以 piecewise 段数，图的数量会迅速失控，需要限制组合规模。
- **参数更新**：权重指针、LoRA、并行度变化后，旧图不再有效，需要用 `cudaGraphExecUpdate` 更新或重新 capture。
- **报错难定位**：graph 内部 kernel 的错误堆栈不友好，调试时往往要先关掉 graph 跑 eager 复现。

### 总结

CUDA Graph 把一段固定命令序列录制成 DAG，用一次提交替代 \(N\) 次提交，消除 launch-bound 场景下 GPU 的等待断档。它只优化 CPU 提交成本，不改变 GPU 计算量和同步语义。

图能否复用的判据是 **launch topology 是否稳定**（kernel 数量、执行顺序、grid/block、tensor 地址四者一致），而不是 shape 是否变化。decode 的 \(q\_len = 1\) 固定、kv_len 只作为循环上界，因此天然适合 graph；prefill 的 \(q\_len\) 随 prompt 变化，通常选择不做 graph 或走 piecewise。动态 batch 通过分档加 padding 解决，动态决策点则用 piecewise graph 把图切开，在段边界保留 CPU 参与。

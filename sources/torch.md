% Torch

在 LLM Inference 工程中，PyTorch 有大量文档里有，但实际需要踩坑才能建立直觉的默会知识（Tacit Knowledge）。

如果从 LLM Inference 工程师的角度建立知识体系，我建议分成以下 10 个 Topic。重点不是学习 PyTorch API，而是理解 PyTorch 的底层执行语义、性能代价以及容易产生的隐式行为。

## 一、PyTorch 工程默会知识地图

| Topic | 核心知识 | LLM Inference 典型场景 | 详细页 |
| --- | --- | --- | --- |
| 1. Tensor 内存模型                | Storage、stride、offset、contiguous、view/copy、alias | QKV reshape、KV Cache、Tensor 切片 | [Tensor Memory](./torch_tensor_memory.md) |
| 2. Tensor 数据类型                | FP32、BF16、FP16、FP8、量化、类型提升                       | KV Cache 精度、MXFP4/NVFP4        | [Data Types](./torch_dtype.md) |
| 3. GPU 异步执行模型                 | CUDA Stream、Event、同步点、异步错误                       | Prefill/Decode overlap         | [CUDA Execution](./torch_cuda_execution.md) |
| 4. 内存分配与生命周期                  | CUDA Caching Allocator、显存碎片、OOM、引用计数             | KV Cache、CUDA Graph            | [Memory Management](./torch_memory_management.md) |
| 5. PyTorch 算子调度               | Dispatcher、ATen、Kernel、Backend                   | 算子性能分析、自定义算子                   | [Dispatcher](./torch_dispatcher.md) |
| 6. Broadcasting 与 Tensor 操作语义 | 广播、in-place、隐式拷贝、类型转换                            | Attention Mask、RoPE            | [Broadcast & Indexing](./torch_broadcast_indexing.md) |
| 7. CUDA Graph                 | Capture、Replay、静态地址、Graph Pool                   | Decode CUDA Graph              | [Cuda Graph](./cuda_graph.md) |
| 8. 数值计算与精度                    | 累加精度、归约顺序、数值稳定性、非确定性                             | Softmax、RMSNorm、MoE            | [Numerical Precision](./torch_numerical_precision.md) |
| 9. Profiling 与性能模型            | GPU 时间测量、Profiler、带宽、算力、Launch overhead          | Kernel 优化、TTFT/TPOT            | [Kernel & Compile](./torch_kernel_compile.md)、[Roofline](./roofline.md) |
| 10. 多 GPU 通信与执行               | NCCL、Collective、Stream ordering、同步               | TP、EP、DP、All-to-All            | [Multi-GPU](./torch_multi_gpu.md) |

其中 1–6 是 PyTorch 基础运行语义，7–10 更偏向 LLM Inference 的工程实践。

## 二、每个 Topic 具体需要掌握什么？

### 1. Tensor 内存模型

你已经提到了其中三个核心概念，可以进一步扩展为：

- Tensor、Storage、Storage Offset 之间的关系
- Shape、Stride 如何决定逻辑索引到物理地址的映射
- Contiguous 与非 Contiguous
- `view()`、`reshape()`、`flatten()`、`permute()`、`transpose()` 的内存行为
- `expand()` 的 zero-stride 机制
- `as_strided()` 与重叠内存
- Tensor aliasing（不同 Tensor 共享 Storage）
- `clone()`、`contiguous()`、`detach()` 的区别

关键默会知识：`reshape()` 不保证零拷贝，`contiguous()` 也不保证发生拷贝。

例如：

```python
x = torch.randn(4, 8, device="cuda")
y = x.T
print(y.is_contiguous())  # False
z = y.reshape(32)          # 通常需要 copy
```

这在 Attention 的 QKV 维度变换中非常常见。

### 2. Tensor 数据类型与转换

- `dtype` 与 `element_size()`
- FP16 和 BF16 的动态范围、尾数精度
- `torch.promote_types` 与算子类型提升规则
- `torch.autocast` 的作用范围
- `to(dtype)` 是否分配新内存
- `view(dtype)` 的位重解释与数值转换的区别
- FP8 的 scale 与量化表示
- FP32 accumulation 与低精度输入
- FP4 的 packed representation

关键是区分 存储精度、计算精度和累加精度，三者不一定相同。

### 3. CUDA 异步执行模型

这是我认为 LLM Inference 中最重要、也最容易被忽略的 Topic 之一。

- CPU 发射 CUDA Kernel 通常是异步的
- GPU Stream 内的执行顺序
- 不同 Stream 之间没有天然的全序保证
- CUDA Event 的 record / wait
- `torch.cuda.synchronize()` 的作用
- `.item()`、`.tolist()`、CPU 数据读取可能引入同步
- GPU → CPU Copy 的同步行为
- Pinned Memory 与 `non_blocking=True`
- CUDA 异步错误可能在后续 API 才暴露
- Tensor 生命周期与异步 Stream 使用：`record_stream()`

例如：

```python
start = time.perf_counter()
y = torch.matmul(a, b)
end = time.perf_counter()
print(end - start)
```

这里主要测量 CPU 发射操作的耗时，而不是 GPU 完成矩阵乘法的时间。

实际意义： Prefill/Decode overlap、通信计算重叠、Profiler 时间分析都依赖对异步模型的理解。

### 4. CUDA 内存管理

- CUDA Caching Allocator 的工作方式
- `memory_allocated()` vs `memory_reserved()`
- 为什么 `del tensor` 不一定让 `nvidia-smi` 中显存下降
- `torch.cuda.empty_cache()` 的真实作用
- 显存碎片化
- Tensor 引用与 Python GC
- 跨 Stream 的 Tensor 生命周期
- Pinned Host Memory
- CUDA Graph Private Memory Pool
- OOM 不一定意味着所有显存都被有效 Tensor 占用

典型问题：为什么显存看起来还有剩余，却无法分配一个新的大 Tensor？

### 5. PyTorch 算子执行与 Dispatch

这一块适合有 CUDA/Triton Kernel 开发需求的工程师。

- Python API → ATen → Dispatcher → Backend Kernel
- Tensor device / dtype / layout 如何影响 dispatch
- 一个 PyTorch Op 不一定对应一个 CUDA Kernel
- 算子可能调用 cuBLAS、cuDNN 或自定义 CUDA Kernel
- TensorIterator 的作用
- 算子内部可能存在隐式 contiguous/copy
- Kernel Fusion 为什么能减少 launch 和中间 Tensor
- `torch.compile`、TorchInductor、Triton 的基本关系
- Custom Op 的注册与调用

关键默会知识：Python 层只有一行代码，不意味着 GPU 上只有一个 Kernel。

### 6. Broadcasting 与 In-place 语义

- Broadcasting 的维度匹配规则
- Broadcasting 通常通过 zero-stride view 实现
- Broadcasting 后的计算可能产生大规模输出
- In-place Op 的 aliasing 副作用
- `x += y` 与 `x = x + y` 的区别
- Advanced Indexing 与 Basic Slicing 的区别
- Boolean Masking 可能创建新 Tensor
- `index_select`、`gather`、`scatter` 的访存特点
- `torch.cat()` 通常需要分配输出
- `torch.stack()` 与 `cat()` 的关系

例如：

```python
x = torch.randn(1024, 1, device="cuda")
y = x.expand(1024, 4096)
```

`y` 逻辑上有 400 多万个元素，但不会因此实际分配同等规模的新 Storage。

这类机制在 Attention Mask、Batch Broadcasting 中很重要。

### 7. CUDA Graph

LLM Decode 中尤其重要。

- Capture / Instantiate / Replay
- 为什么 CUDA Graph 需要稳定的内存地址
- Capture 期间的内存分配限制
- Graph Private Pool
- Static Input Buffer
- Graph Replay 与输入更新
- Dynamic Shape 和 Padding/Bucketing
- Graph Capture 与 CUDA Stream 的关系
- Graph 内部跨 Stream 依赖
- 为什么某些 CPU 同步操作无法 Capture
- Graph 生命周期与显存占用

这里有一个典型的认知差异：

CUDA Graph 不是简单地把 Python 函数编译成 GPU 程序，而是记录并重放 GPU 工作及其依赖关系。

### 8. 数值计算与稳定性

- Floating-point non-associativity
- Reduction 顺序对结果的影响
- FP16/BF16 accumulation
- Softmax 的数值稳定性
- `exp`、`log`、`rsqrt` 的精度
- TF32 的计算行为
- Deterministic vs Non-deterministic Kernel
- Atomic 操作与结果差异
- Fused Kernel 与非 Fused Kernel 的数值差异
- `torch.allclose()` 的容差语义

尤其是在比较 Triton Kernel 与 PyTorch Reference 时，不能简单地要求逐位相等。

### 9. Profiling 与性能模型

- CUDA Event 与 CPU Wall Time
- Warmup 为什么必要
- CUDA Graph Replay 的测量方式
- `torch.profiler` 的 CPU/CUDA Timeline
- `nsys` 与 `ncu` 的区别
- Kernel Launch Overhead
- Memory-bound vs Compute-bound
- Arithmetic Intensity
- Roofline Model
- GPU Occupancy
- Register Pressure
- Memory Coalescing
- Kernel Fusion 的收益与代价

对于 LLM Inference，特别需要建立一个直觉：

小 Batch Decode 与大 Batch Prefill 的性能瓶颈可能完全不同。

### 10. 多 GPU 通信与分布式执行

详细说明见 [Torch 多 GPU 通信与执行](./torch_multi_gpu.md)。

- NCCL Collective 的基本语义
- AllReduce、AllGather、ReduceScatter、AllToAll
- NCCL 操作与 CUDA Stream 的关系
- Async Collective 与 Completion
- 通信 Buffer 的生命周期
- 不同 Rank 的 Collective 调用顺序
- TP / EP 的通信模式
- Compute-Communication Overlap
- 通信等待与 GPU Idle 的区别
- 分布式 Deadlock 的常见来源

例如，`async_op=True` 不等于可以立即在任意 Stream 上安全使用通信结果，仍需理解 completion 和 stream dependency。

## 三、学习优先级

如果目标是 LLM Inference Runtime 开发与性能优化，可以这样排序：

优先级与知识依赖

P0 · 基础语义

必须掌握

Tensor 内存模型 → Broadcasting / In-place → CUDA 异步执行 → 显存管理

P1 · Runtime 核心

重点掌握

CUDA Graph → Profiling → 多 GPU 通信

P2 · Kernel 深入

按需深入

Dispatcher / ATen → 数值精度 → Kernel 实现与优化

## 四、跨 Topic 的经典默会知识

这些问题特别适合做成工程师的知识卡片：

| 常见直觉                                    | 实际情况                            |
| --------------------------------------- | ------------------------------- |
| `reshape()` 只是修改 Shape                  | 可能发生真实内存拷贝                      |
| `flatten()` 一定零拷贝                       | 取决于 Stride                      |
| `x.to("cuda")` 一定发生拷贝                   | 如果已经在目标 Device，默认可能直接返回原 Tensor |
| `non_blocking=True` 就一定完全异步             | 还取决于传输方向、内存类型和同步依赖              |
| `del x` 就释放 GPU 显存                      | 可能还有引用，释放的显存也可能被缓存              |
| `torch.cuda.empty_cache()` 能释放活跃 Tensor | 不能                              |
| `torch.matmul()` 只启动一个 Kernel           | 不保证                             |
| GPU Kernel 返回就代表计算完成                    | CUDA 调用通常异步                     |
| CUDA Graph 可以随意更改 Shape                 | 通常需要固定形状或其他专门机制                 |
| In-place 一定更快                           | 可能带来依赖、aliasing 或其他性能问题         |
| `torch.cat()` 只是拼接 Tensor 元数据           | 通常需要创建新 Tensor 并复制数据            |
| PyTorch 算子输出一定是 Contiguous              | 不保证                             |
| 两个数学等价的 Kernel 结果一定完全相同                 | 浮点运算顺序不同可能产生差异                  |

## 五、还可以单独增加一个 Topic：Python 与 PyTorch 的交互边界

这一类问题在 LLM Runtime 中非常常见，却容易被忽视：

- Python GIL 与 GPU 异步执行的关系
- Python Loop 的 CPU Dispatch Overhead
- Tensor `.item()` 为什么容易成为性能陷阱
- Python 引用计数如何影响 Tensor 生命周期
- `torch.inference_mode()` 与 `torch.no_grad()` 的区别
- Python 控制流与 CUDA Graph Capture 的关系
- 为什么 GPU 利用率低不一定是 GPU Kernel 慢
- 多线程调用 PyTorch 时的线程安全和 Stream 语义

我会把它列为第 11 个 Topic，因为很多 LLM Serving 性能问题，实际上发生在 Python Runtime、PyTorch Dispatcher 与 CUDA Driver 的边界，而不是 CUDA Kernel 内部。

整体而言，最值得形成系统性直觉的是四种模型：Tensor 内存模型、CUDA 异步执行模型、GPU 内存生命周期模型，以及 PyTorch 算子执行模型。 理解它们之后，再看 SGLang、vLLM 中的 CUDA Graph、KV Cache、Overlap Scheduling、Custom Kernel，会更容易判断代码背后的隐式成本和正确性约束。

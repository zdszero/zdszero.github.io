% CPU & GPU

CPU 和 GPU 是两个独立的执行体。CPU 上跑的 Python/C++ 代码不会「直接执行」kernel，它只是把命令写进一个队列；GPU 在另一端按自己的节奏取命令执行。这个模型和生产者-消费者队列完全同构：

```text
CPU (producer)                        GPU (consumer)
---------------                       ----------------
launch kernel A     --->  [cmd queue] --->  SM 执行 A
launch kernel B     --->  [cmd queue] --->  SM 执行 B
memcpy D2H          --->  [cmd queue] --->  copy engine 执行
```

由此得到三条基本事实：

1. `cudaLaunchKernel` 这类调用是**异步**的：CPU 把命令入队后立刻返回，不等待 GPU；
2. 同一个队列内的命令按入队顺序执行，跨队列默认没有任何顺序保证；
3. CPU 不阻塞，就意味着它可以在 GPU 还在算 A 的时候，提前把 B、C、D 都排好队——这正是性能的来源。

反过来，一旦 CPU 需要「看到」GPU 的结果，或者需要复用 GPU 仍在使用的内存，就必须插入同步点，让两个异步执行体重新对齐。

### 一次 kernel launch 的开销

一次 launch 的 CPU 侧成本（参数检查、驱动调用、写入命令缓冲区）通常在几微秒量级，而一个 GPU kernel 的执行时间可以小到 \(\mu s\) 以下（例如 elementwise 加、LayerNorm），也可以大到 \(\text{ms}\) 级（大 GEMM、attention）。

设第 \(i\) 个 kernel 的 CPU 提交时间为 \(t^{\text{cpu}}_i\)，GPU 执行时间为 \(t^{\text{gpu}}_i\)，一共 \(N\) 个 kernel。在 CPU 能持续提前排队的理想流水线下，端到端墙钟时间近似是「谁慢谁说话」：

\[
T_{\text{wall}} \approx \max\left(\sum_{i=1}^{N} t^{\text{cpu}}_i,\ \sum_{i=1}^{N} t^{\text{gpu}}_i\right)
\]

（严格地说还要加上首个 kernel 的启动延迟 \(\mathcal{O}(1)\) 量级的填充时间，以及 CPU 未及时跟上的断档。）

小 kernel 很多时，\(\sum_i t^{\text{cpu}}_i\) 反而成为瓶颈，GPU 会在两次 kernel 之间空转，这称为 **launch-bound**；大 kernel 时 GPU 满负荷，CPU 有充足的提前量，称为 **compute-bound**。CUDA Graph 的核心动机就是把 \(\sum_i t^{\text{cpu}}_i\) 压缩成一次提交，从而消除断档，详见《Cuda Graph》。

### Stream：命令队列与顺序边界

#### stream 内保序，stream 间无序

CUDA stream 就是上面那条命令队列。它的语义只有两条：

- **stream 内严格保序**：同一 stream 里先入队的命令一定先完成，后入队的才能开始；
- **stream 间默认无顺序**：不同 stream 的命令由 GPU 自行调度，先后不确定。

第一点是关键：**stream 内保序是「免费」的依赖保证，不需要任何 CPU 参与**。只要把有数据依赖的 kernel 放进同一个 stream，正确性就自动成立，CPU 不需要停下来等。

```text
stream 1:  A -> B -> C        (三者严格按序)
stream 2:  X -> Y             (两者严格按序，但与 stream 1 无关系)
```

如果 A 的输出要给 X 用，就必须在两个 stream 之间显式建立依赖——这正是 event 的用途。

#### 默认 stream

不指定 stream 时命令进入默认 stream。默认 stream 有两种模式：

- **legacy 默认 stream**：它会与所有其它阻塞 stream 隐式同步，行为近似于「全局串行」，容易在不知情的情况下破坏并行；
- **per-thread 默认 stream**：每个线程一个默认 stream，不与其它 stream 隐式同步。

PyTorch 使用默认 stream，并且 `torch.cuda.Stream()` 创建的流带 `cudaStreamNonBlocking` 语义，因此与默认流之间**不会**隐式同步。自己写 CUDA/C++ 扩展时要显式确认编译选项与运行库对默认 stream 语义的处理。

#### 典型的 stream 划分

工程上一般按「资源类型」划分 stream，因为只有分开才能让不同硬件单元同时工作：

| stream | 承载的命令 | 主要占用资源 |
| --- | --- | --- |
| compute stream | kernel 计算 | SM |
| copy stream | H2D / D2H 拷贝 | Copy Engine |
| comm stream | NCCL AllReduce 等 | NVLink / NIC |

### 同步点的三种动机

同步不是为了「让代码看起来顺序正确」，它只解决三类问题：

1. **正确性：CPU 要读 GPU 写的数据。** GPU 写显存是异步的，CPU 直接读到的可能是旧值；任何 D2H 拷贝之后的 CPU 读取都必须等拷贝完成。
2. **资源生命周期：CPU 要释放或复用 GPU 仍在使用的内存。** `cudaFree`、显存池回收、复用 output buffer，都必须保证相关 kernel 已经执行完。
3. **背压：防止 CPU 无限制入队。** 如果 CPU 完全不等待，队列会无限增长，CPU 侧内存与端到端延迟都会失控；同步点是把队列长度控制在有限范围内的闸门。

第 3 点最容易被忽略：即使正确性上不需要等待，长期运行的流水线也需要周期性同步来做背压和错误检查。

### 同步原语的粒度与代价

CUDA 提供的同步手段可以按粒度分层：

| 原语 | 粒度 | CPU 是否阻塞 | 典型用途 | 代价 |
| --- | --- | --- | --- | --- |
| `cudaDeviceSynchronize` | 整个 device，所有 stream | 是 | 调试、benchmark、异常兜底 | 最高 |
| `cudaStreamSynchronize` | 单个 stream 的全部命令 | 是 | 等一个阶段结束 | 中 |
| `cudaEventSynchronize` | event 之前记录的工作 | 是 | 等一个里程碑 | 较低 |
| `cudaStreamWaitEvent` | 跨 stream 建立依赖 | 否 | 重叠 copy / compute / comm | 最低 |
| `cudaEventQuery` / `cudaStreamQuery` | 查询完成状态 | 否（轮询） | 非阻塞检查 | 低 |

#### CPU 阻塞同步与 GPU 内部依赖

这两种同步经常被混为一谈，但语义完全不同：

- **CPU 阻塞同步**（`cudaDeviceSynchronize`、`cudaStreamSynchronize`、`cudaEventSynchronize`）：CPU 停下来等，GPU 继续跑。CPU 在这段时间无法提交新命令，流水线会断。
- **GPU 内部依赖**（`cudaStreamWaitEvent`，以及同一 stream 内的顺序）：CPU 不阻塞，只是给 GPU 的命令队列插入一条「依赖边」。GPU 执行到这里时自己等待，CPU 可以继续往前排队。

要在流水线里做 overlap，几乎只能用第二种。第一种只在 CPU 确实需要数据时才使用。

### cudaEvent：跨 stream 依赖与计时

#### 记录语义

`cudaEventRecord(event, stream)` 的语义是：在 stream 的命令流中打一个标记，当 stream 中**此前所有命令**都执行完毕时，该 event 变为完成态。

于是可以这样建立跨 stream 依赖：

```cpp
cudaEventRecord(ev, stream_a);             // 标记 stream_a 的完成点
cudaStreamWaitEvent(stream_b, ev, 0);      // stream_b 后续工作都等这个点
```

`cudaStreamWaitEvent` 不阻塞 CPU，只往 `stream_b` 队列里插入一条等待边。这是 copy/compute 重叠、计算/通信重叠的标准写法。

#### 计时正确姿势

用 event 计时，而不是 `time.time()`：

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
# ... 待测工作 ...
end.record()
torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end)
```

`elapsed_time` 得到的是

\[
\Delta t = t_{\text{end}} - t_{\text{start}}
\]

其中 \(t_{\text{start}}\) 与 \(t_{\text{end}}\) 都取自 GPU 时间线，因此**不包含 CPU 提交命令的间隙**。这也是 event 计时往往小于 `time.time()` 的原因：后者会把 CPU 排队等待的时间一起算进去。

只用 event 做同步、不做计时时，创建时指定 `enable_timing=False`（对应 C API 的 `cudaEventDisableTiming`）可以降低开销。另外 event 默认不支持跨进程复用，多进程场景需要 `cudaEventInterprocess`。

### PyTorch 中的对应接口

| CUDA 原语 | PyTorch 接口 |
| --- | --- |
| `cudaDeviceSynchronize` | `torch.cuda.synchronize()` |
| `cudaStreamSynchronize` | `stream.synchronize()` |
| `cudaEventRecord` | `event.record(stream)` |
| `cudaEventSynchronize` | `event.synchronize()` |
| `cudaStreamWaitEvent` | `stream.wait_event(event)` |
| 新建 stream | `torch.cuda.Stream()` |
| 当前 stream | `torch.cuda.current_stream()` |

一个典型的跨 stream 依赖在 PyTorch 里写成：

```python
ev = torch.cuda.Event()
with torch.cuda.stream(compute_stream):
    y = model(x)
    ev.record()
with torch.cuda.stream(copy_stream):
    copy_stream.wait_event(ev)
    z = y.to("cpu", non_blocking=True)
```

注意 `non_blocking=True` 只有在源 tensor 位于 pinned memory 时才是真异步；否则底层仍会退化成同步拷贝。`torch.cuda.Stream()` 本身也有创建开销，通常 2 到 4 条 stream 就足够覆盖 copy、compute、通信三类需求，不要按层创建。

### 隐式同步点

比显式 `synchronize()` 更常见的，是那些**你没有写同步、但框架替你同步了**的操作。典型触发条件：

- **D2H 拷贝**：`.item()`、`.cpu()`、`.tolist()`、`float(tensor)`、`print(tensor)`、写日志时取标量值；
- **依赖张量值的控制流**：`if tensor > 0:`、`while tensor.item() < ...`、`bool(tensor)`；
- **动态 shape 或索引相关算子**：`nonzero`、`unique`、布尔 mask 之后的 shape 计算；
- **显存分配**：caching allocator 缺块时调用 `cudaMalloc` / `cudaFree`，会触发设备同步；
- **调试开关**：`CUDA_LAUNCH_BLOCKING=1` 把每次 launch 变成同步，只用于定位问题，绝不能用于生产。

其中每步调用 `loss.item()` 是训练代码里最经典的性能杀手：它强制一次全设备或全 stream 的等待，把 CPU 与 GPU 之间原本可以重叠的部分全部串行化。正确做法是累积若干步再一次性取出，或者干脆在 GPU 上完成数值判断。

PyTorch 提供了定位工具：`torch.cuda.set_sync_debug_mode("warn")` 会对隐式同步操作发出警告，适合排查「莫名其妙变慢」的代码路径。

### 如何设置同步点

1. **能用 stream 顺序表达的依赖，不要用 CPU 同步。** 同一 stream 内保序是免费的。
2. **跨 stream 的依赖一律用 event。** 用 `cudaStreamWaitEvent` / `stream.wait_event`，而不是 `cudaDeviceSynchronize`。
3. **把同步点尽量往后推、尽量合并。** 与其每个 micro-step 取一次结果，不如积累到一个阶段边界再取一次。
4. **同步点落在阶段边界，而不是 kernel 边界。** 前向结束、反向结束、optimizer 结束、一个 batch 结束，这些才是自然的对齐点。
5. **区分「必须等」和「只是想知道」。** 只有 CPU 真的要用数据时才必须等；如果只是想判断是否完成，用 `query` 轮询或改成异步回调。
6. **给等待留出重叠空间。** 训练里常见做法是等 D2H 结果的同时启动下一步 H2D 拷贝，用双缓冲把等待藏起来。
7. **测量时先 warmup 再同步。** 第一次 kernel 包含 JIT、`cudaMalloc`、库初始化等一次性开销；用 event 计时并配合 `torch.cuda.synchronize()` 才能测到真实 GPU 时间。

### 典型流水线的同步点设计

#### 朴素单流

```text
CPU:  H2D  ->  fwd  ->  bwd  ->  update
GPU:         [====]   [====]   [==]
```

如果 CPU 完全不等，这个流水线是正确的，因为 stream 内保序已经保证了数据依赖；但队列会无限增长。工程上会定期同步做背压，例如每 \(K\) 步或每个 epoch 同步一次。

#### copy / compute 双缓冲重叠

```text
stream copy   :  H2D(batch 0) -> H2D(batch 1) -> H2D(batch 2)
stream compute:                  fwd/bwd(0)   -> fwd/bwd(1)
                        ^ ev0            ^ ev1
```

- `H2D(batch 1)` 与 `fwd/bwd(0)` 并行执行；
- `fwd/bwd(1)` 通过 `wait_event(ev1)` 等 `H2D(batch 1)` 完成；
- CPU 只负责提交，不需要在两步之间停下来等待。

这是数据加载与计算重叠的标准模板，也是 `pin_memory` 配合 `non_blocking=True` 想达到的效果。

#### 推理服务：为什么采样通常放在 GPU 上

一个自回归推理 step 的时间线大致是：

```text
GPU:  forward  ->  logits  ->  (sample)  ->  next token
CPU:    调度请求 / 组 batch             ^ 若在此处把 logits 拉回 CPU 采样，就是一次强制同步
```

如果每步都用 `.cpu()` 把 logits 取回、在 CPU 上用 numpy 做 top-k/top-p 采样，再用 `.cuda()` 送回去，那么每步都会产生两次 H2D/D2H 拷贝加至少一次同步，GPU 会在采样期间完全空转。把采样（temperature、top-k、top-p、multinomial）整体实现在 GPU 上，或者用异步采样与下一步 forward 重叠，是 vLLM、SGLang 这类框架的标准做法。采样本身的数学原理见《采样》。

同样的逻辑也适用于每步记录 metrics：把统计量累积在 GPU 上、定期同步取出，而不是每步 `.item()`。

### 与 CUDA Graph 的关系

CUDA Graph 把一段固定的命令序列录制下来，之后每次 replay 只需一次提交。它改变的是 CPU 提交命令的成本，不改变 GPU 执行命令的成本，也不改变同步语义：

- 录制期间可以使用 event 与多 stream，但 replay 时这些依赖关系被固化在 graph 内部；
- graph 内部不能有 CPU 侧阻塞同步，否则无法正确录制或行为异常；
- 需要 CPU 介入的点（例如根据结果决定下一步）必须放在 graph 外面，或使用 conditional node 这类受限机制。

因此「先想清楚同步点在哪里，再决定哪些片段可以 graph 化」是更合理的顺序。

### 总结

CPU 与 GPU 是生产者与消费者：CPU 只负责向 stream 队列提交命令，GPU 按序消费。同一 stream 内的顺序是免费的正确性保证；跨 stream 的依赖要用 event 建立，而不是让 CPU 阻塞等待。同步点只服务于三件事——CPU 要读 GPU 的数据、内存要被释放或复用、需要背压控制队列长度。

实践中最容易出问题的地方不是显式同步，而是 `.item()`、`.cpu()`、`print` 这类隐式同步，以及把同步点放错位置导致的流水线断档。设置原则可以浓缩成一句：**把 CPU 阻塞同步集中到阶段边界，把跨 stream 依赖交给 event，剩下的交给 stream 顺序。**


% Stream

可以大致理解为“默认使用一个 stream，也可以创建多个 stream”，但有个关键点：**stream 是操作的有序队列，不对应固定的 SM 或 warp。**

**1. Stream 定义的是任务之间的顺序**

往 stream 中提交的操作可以包括 kernel、内存拷贝、event 等。对通常的 kernel 启动：

- **同一 stream 内**：按提交顺序执行，后面的操作等待前面的操作完成。
- **不同 stream 间**：没有默认的先后关系，可以重叠执行，但不保证一定并发。
- kernel 启动通常对 CPU 异步：CPU 提交之后可以继续运行。

例如：

```cpp
cudaStream_t s1, s2;
cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);

A<<<grid, block, 0, s1>>>();
B<<<grid, block, 0, s1>>>();
C<<<grid, block, 0, s2>>>();
```

这里保证 `A 完成 → B 开始`；`C` 与 `A/B` 可以重叠，也可能因为资源不足而串行执行。[CUDA 异步执行文档](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)

**2. 默认 stream 有两种模式**

不指定 stream 的写法：

```cpp
A<<<grid, block>>>();
```

会使用默认 stream，但默认 stream 的语义取决于模式：

| 模式 | 默认 stream 的作用域 | 与其他 stream 的关系 |
|---|---|---|
| Legacy，未配置时的默认模式 | 同一 CUDA context 中，各 CPU 线程共享 | 与普通 blocking stream 有隐式同步 |
| Per-thread，使用 `--default-stream per-thread` | 每个 CPU 线程各有一个 | 像普通 stream 一样，不带 legacy 的全局同步语义 |

`cudaStreamCreate()` 创建的是 blocking stream；`cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)` 可以避免与 legacy 默认 stream 的隐式同步。这里的 **blocking 指 stream 之间的同步关系，不代表 kernel 启动会阻塞 CPU**。[默认 stream 的同步语义](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html)

**3. 不同 stream 如何落到 SM、warp 上？**

可以按下面的层次理解：

```text
Stream：决定哪些操作已经满足执行依赖
   ↓
Kernel / Grid：包含许多 thread block
   ↓
Block 调度：分配到有可用资源的 SM
   ↓
SM 内：block 的线程组成 warp，每个 warp 通常有 32 个线程
   ↓
Warp scheduler：从就绪的驻留 warp 中选择并发射指令
```

**调度到 SM 的基本单位是 block；SM 内发射指令的单位是 warp。** 普通 thread block 整体放到一个 SM 上执行，一个 SM 可以同时驻留多个 block。Stream 不会给 kernel 预留某组 SM，也不会分配专属 warp scheduler。[CUDA 编程模型](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)、[NVIDIA 编程模型说明](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model)

来自不同 stream 的两个 kernel，如果支持并发且资源允许：

- 可以在不同 SM 上同时执行；
- 也可能让两个 kernel 的 block 同时驻留在同一个 SM 上；
- 在 SM 内，各自的 warp 由硬件调度器选择执行，不能理解为“轮流执行一个 stream”。

具体的 block 分配、warp 选择策略和公平性，CUDA 不提供可依赖的固定保证。[CUDA 执行与硬件调度说明](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html)

**4. 为什么创建多个 stream 仍可能没有并发？**

假设 kernel A 的 block 占满了所有 SM 的寄存器、shared memory 或驻留 block 名额，那么 kernel B 即使在另一条 stream 中，也得等资源释放。

反过来，如果 A 只启动了少量 block，B 就可能利用剩余 SM；如果同一 SM 的资源还有余量，也可能容纳 B 的 block。即使两者成功重叠，若都争用显存带宽，也未必更快。并发能力与资源限制可参见 [CUDA Programming Guide](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html)。

另外，**只有一个 stream，也能让一个大 kernel 使用整个 GPU**：kernel 内的 block 和 warp 本身就有并行性。多个 stream 增加的是不同任务之间重叠执行的机会。

**5. 跨 stream 的依赖需要显式表达**

如果 `s2` 中的 B 要读取 `s1` 中 A 的结果，用 event 建立依赖：

```cpp
A<<<grid, block, 0, s1>>>(data);
cudaEventRecord(done, s1);

cudaStreamWaitEvent(s2, done, 0);
B<<<grid, block, 0, s2>>>(data);
```

这样 B 会等待 A 完成，但 CPU 不需要停下来等待。CUDA 不会因为两个 kernel 使用了同一个指针，就自动推断这种依赖。[CUDA Events 与同步](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)

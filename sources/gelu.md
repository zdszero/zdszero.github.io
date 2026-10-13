% GELU

GELU（Gaussian Error Linear Unit）在 llm.c 中用 **tanh 近似**版本：

\[
\text{GELU}(x) = 0.5\,x\left(1 + \tanh\!\left[\sqrt{\tfrac{2}{\pi}}\left(x + 0.044715\,x^3\right)\right]\right)
\]

对应代码里的常量与表达式：

```c
#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

float cube = 0.044715f * x * x * x;
out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
```

CPU 参考实现 `gelu_forward_cpu` 就是对 `N` 个元素串行地套用这个公式，作为数值正确性的基准。

### 它是一个 memory-bound 算子

GELU 对每个元素只做「1 次读 + 1 次写」，中间是个位数的浮点运算，算术强度 \(I\) 极低。用[概述](../)里的 Roofline 判据，它必然是 **memory-bound**：kernel 的快慢几乎完全取决于**能不能把访存带宽吃满**，而不是算得多快。`main` 里也是按带宽评估的：

```c
// 每个输出元素 1 读 1 写；memory_ops = N * 2 * sizeof(floatX)
// 除以耗时即实测带宽，对标 A100 的 ~1,555 GB/s
```

所以这个算子的两个 version 之间的差别，不在于「算法」，而完全在于「怎么访存」。

### version 1：朴素移植

一个线程处理一个输出元素，直接标量 load/store：

```c
__global__ void gelu_forward_kernel1(floatX* out, const floatX* inp, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float xi = inp[i];
        float cube = 0.044715f * xi * xi * xi;
        out[i] = 0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube)));
    }
}
```

要点：
- I/O 指针是 `floatX`（本文件 `#define ENABLE_BF16`，即 bf16），但**计算提升到 `float`** 做（`float xi = inp[i]`），再写回 bf16——降精度省带宽，但算得准。
- 每个线程只碰 `sizeof(floatX)` = 2 字节。虽然相邻线程访问相邻地址、访存是合并的，但**单线程的事务宽度太窄**，没有把内存事务的宽度用满，带宽利用率有上限。

### version 2：bf16 + Packed128 向量化访存

优化手段只有一个字：**宽**。让每个线程一次搬运 128 bit，用 `LDG.128`/`STS.128` 宽事务代替一堆窄事务：

```c
__global__ void gelu_forward_kernel2(floatX* out, const floatX* inp, int N) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (i < N) {
        x128 packed_out;
        x128 packed_inp = load128cs(inp + i); // 流式载入，不驻留 cache
        for(int k = 0; k < packed_inp.size; ++k) {
            float xi = (float)packed_inp[k];
            float cube = 0.044715f * xi * xi * xi;
            packed_out[k] = (floatX)(0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube))));
        }
        store128(out + i, packed_out);
    }
}
```

关键变化：
- 索引按 `x128::size` **跨步**，每个线程负责一段连续元素；
- 输入用 `load128cs`（streaming，`__ldcs`，「读一次就走、别占 cache」），输出用 `store128`（**不**用 streaming 版本，注释解释是「留在 cache 里，方便下一个算子复用」）。

#### Packed128 / x128 是什么

`Packed128<ElementType>` 是一个 `alignas(16)`（16 字节 = 128 bit）的结构体，内部把这块数据重解释为一个 `int4`，从而**强制编译器发出 128-bit 的 load/store 指令**（类似 `float4` 之于 fp32，但支持任意精度）。每次打包的元素个数是：

```c
static constexpr int size = sizeof(int4) / sizeof(ElementType);
// bf16: 16/2 = 8 个；fp32: 16/4 = 4 个
```

配套的 `load128`/`load128cs`/`store128`/`store128cs` 分别通过 `reinterpret_cast<const int4*>` 一次读写 128 bit（`cs` 变体用 `__ldcs`/`__stcs` 加流式缓存提示）。

**为什么更快**：GPU 不再为每个标量元素发一条访存指令，而是一条宽事务一次搬 8 个 bf16。指令数更少、事务完全合并、每笔事务搬运的有效字节更多——对于带宽受限的 GELU 来说，这直接把实测带宽推向峰值。

### 小结

| version | 手段 | 瓶颈视角 |
|---|---|---|
| 1 | 一线程一元素，标量访存 | 事务窄，带宽利用率不足 |
| 2 | Packed128 一线程 8 个 bf16，128-bit 宽事务 | 逼近峰值带宽 |

GELU 的教训适用于**所有逐元素算子**：既然算得快没意义、访存才是瓶颈，那就把每笔访存做到最宽（向量化）、把精度降到够用（bf16 省一半带宽）、别让数据在 cache 里做无谓停留（streaming 提示）。


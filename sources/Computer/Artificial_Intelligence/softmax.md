% Softmax

### Principle

$$m = \max(x)$$

$$\text{Softmax}(x_i) = \frac{e^{x_i - m}}{\sum_{j} e^{x_i - m}}$$


简单的实现就是通过三层 for 循环，第一次循环计算最大值 $m$，第二次循环计算 $sum$，第三次循环计算每个 $x_i$ 的 softmax。

$$
\begin{aligned}
&\text{for } i \gets 1, N \text{ do} \\
&\quad m_i \gets \max(m_{i-1}, x_i) \\
&\text{end} \\
&\text{for } i \gets 1, N \text{ do} \\
&\quad sum_i \gets sum_{i-1} + e^{x_i - m_N} \\
&\text{end} \\
&\text{for } i \gets 1, N \text{ do} \\
&\quad a_i \gets \frac{e^{x_i - m_N}}{sum_N} \\
&\text{end}
\end{aligned}
$$

但这种实现没有效率不高，著名的 FlashAttention 也是因为把 softmax 改造成了可以分段迭代的形式，即 online softmax，而显著减少 IO 次数，以大幅提高了性能。

在 safe softmax 中计算最大值和求和需要 global 的信息，如果在没有 global 信息的情况下，如何通过迭代的方式完成等效计算呢？事实上，可以通过两个循环完成，即

$$
\begin{aligned}
&\text{for } i \gets 1, N \text{ do} \\
&\quad m_i \gets \max(m_{i-1}, x_i) \\
&\text{end} \\
&\text{for } i \gets 1, N \text{ do} \\
&\quad sum'_i \gets sum'_{i-1} e^{m_{i-1} - m_i} + e^{x_i - m_i} \\
&\text{end} \\
&\text{for } i \gets 1, N \text{ do} \\
&\quad a_i \gets \frac{e^{x_i - m_N}}{sum'_N} \\
&\text{end}
\end{aligned}
$$

这样就减少一次循环过程完成了等效计算。其中关键的步骤是 $sum_{i}^{'}$ 的计算，在此再稍加推导一下

$$
\begin{aligned}
\text{sum}_i &= \sum_{j=1}^{i} e^{x_j - m_i} \\
             &= \left( \sum_{j=1}^{i-1} e^{x_j - m_i} \right) + e^{x_i - m_i} \\
             &= \left( \sum_{j=1}^{i-1} e^{x_j - m_{i-1}} \right) e^{m_{i-1} - m_i} + e^{x_i - m_i} \\
             &= \text{sum}_{i-1} e^{m_{i-1} - m_i} + e^{x_i - m_i}
\end{aligned}
$$

### CPU

#### safe softmax

三次循环

```cpp
void softmax_forward_cpu(float* out, const float* inp, int N, int C) {
    // inp is (N, C)
    // out is (N, C), each row of inp will get softmaxed
    for (int i = 0; i < N; i++) {
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        float maxval = -INFINITY;
        for (int j = 0; j < C; j++) {
            if (inp_row[j] > maxval) {
                maxval = inp_row[j];
            }
        }
        // Note: since we want to ensure that the CUDA-kernels are accurate,
        // we do this accumulation in higher precision, so we can be assured
        // that our ground-truth is of high quality.
        double sum = 0.0;
        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval);
            sum += out_row[j];
        }
        float norm = 1.f / (float)sum;
        for (int j = 0; j < C; j++) {
            out_row[j] *= norm;
        }
    }
}
```

#### online softmax

两次循环

```cpp
// online version of softmax on CPU from the paper "Online normalizer calculation for softmax"
void softmax_forward_online_cpu(float* out, const float* inp, int N, int C) {
    // inp is (N, C)
    // out is (N, C), each row of inp will get softmaxed
    for (int i = 0; i < N; i++) {
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        float maxval = -INFINITY;
        float sum = 0.0f;
		for (int j = 0; j < C; j++) {
			float maxval_prev = maxval;
			if (inp_row[j] > maxval) {
				maxval = inp_row[j];
				sum = sum * expf(maxval_prev - maxval) + expf(inp_row[j] - maxval);
			} else {
				sum += expf(inp_row[j] - maxval);
			}
		}

        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval) / sum;
        }
    }
}
```

### GPU

#### safe softmax

在 N 的维度上并行计算，每个线程负责计算一行。

```cpp
__global__ void softmax_forward_kernel1(float* out, const float* inp, int N, int C) {
    // inp is (N, C)
    // out is (N, C), each row of inp will get softmaxed
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        float maxval = -INFINITY;
        for (int j = 0; j < C; j++) {
            if (inp_row[j] > maxval) {
                maxval = inp_row[j];
            }
        }
        double sum = 0.0;
        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval);
            sum += out_row[j];
        }
        for (int j = 0; j < C; j++) {
            out_row[j] /= (float)sum;
        }
    }
}
```

#### shared memory

#### shuffle

`int __shfl_down_sync(unsigned mask, int var, unsigned delta, int width=warpSize);`

返回值含义（比如 delta = 1）：

```
线程 0：获取线程 1 的值
线程 1：获取线程 2 的值
...
线程 30：获取线程 31 的值
```

`__shfl_down_sync` 是 CUDA warp 级别的 shuffle 指令，用于在 一个 warp 内 实现数据交换。它常用于 高效的归约（reduction）运算，比使用共享内存 `__shared__` 更快，因为 warp 内的线程可以直接通信，而不需要 `__syncthreads()`。

```cpp
// warp-level reduction for finding the maximum value
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// warp-level reduction for summing values
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}
```


### Ref

[SoftMax算子的 CUDA 实现](https://zhuanlan.zhihu.com/p/695307283)

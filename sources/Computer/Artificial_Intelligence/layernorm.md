% Layer Norm

$$y = w \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + b$$

$$
\mu = \frac{1}{m} \sum_{i=1}^{m} x_i, \quad
\sigma = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (x_i - \mu)^2}
$$

$\mu$ 是均值，$\sigma^2$ 是方差。

Layer Norm 主要用于改进神经网络的训练稳定性和收敛速度。

### Def

```c
// 输入输出 shape
int B, int T, int C
// 输入
const float *inp
// 均值和标准差的倒数
// 该值随机初始化后传入，前向完成后得到结果并在反向过程中重复使用
float *mean, float *rstd
// 供训练的权重矩阵
const float *weight, const float *bias
// 输出
float *out
```

### CPU

```c
void layernorm_forward_cpu(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C) {
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            const float* x = inp + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalized output
                float o = n * weight[i] + bias[i]; // scale and shift it
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}
```

### GPU

#### v1

由于 LayerNorm 的计算都是在维度 C 上进行，因此在 [B, T] 维度上可以通过 CUDA 实现并行计算，其实现的关键如下所示。其中 block_size 为可以设定的参数，grid_size 为 N/block_size 向上取整，因此总的线程grid_size * block_size >= N ，layernorm_forward_kernel1 的实现与 CPU 的内层循环的实现相同。

```cpp
__global__ void layernorm_forward_kernel1(float* out, float* mean, float* rstd,
                                       const float* inp, const float* weight, const float* bias,
                                       int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Seek to the input position inp[idx, :]
    const float* x = inp + idx * C;
    float* out_x = out + idx * C;

    // Compute mean
    float m = 0.0f;
    for (int i = 0; i < C; i++) {
        m += x[i];
    }
    m /= C;

    // Compute variance
    float v = 0.0f;
    for (int i = 0; i < C; i++) {
        float xshift = x[i] - m;
        v += xshift * xshift;
    }
    v /= C;

    // Compute rstd
    float s = rsqrtf(v + 1e-5f);

    // Normalize, scale, and shift
    for (int i = 0; i < C; i++) {
        float n = (x[i] - m) * s;
        out_x[i] = n * weight[i] + bias[i];
    }

    // Store mean and rstd for backward pass
    mean[idx] = m;
    rstd[idx] = s;
}

void layernorm_forward1(float* out, float* mean, float* rstd,
                           const float* inp, const float* weight, const float* bias,
                           int B, int T, int C,
                           const int block_size) {
    const int N = B * T;
    const int grid_size = ceil_div(N, block_size);
    layernorm_forward_kernel1<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}
```

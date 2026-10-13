% Activation

如果神经网络只堆叠线性层，那么无论堆叠多少层，整体仍然是线性变换。以两个无偏置线性层为例：

\[
y = W_2(W_1x) = (W_2W_1)x.
\]

矩阵 \(W_2W_1\) 仍然可以合并成一个矩阵，因此增加层数并没有增加函数的非线性表达能力。插入激活函数 \(\phi\) 后，计算变成

\[
y = W_2\phi(W_1x),
\]

一般无法再合并为一个线性变换。

激活函数通常承担以下作用：

- 引入非线性，使网络能够拟合复杂函数；
- 控制不同数值区域的响应强度；
- 影响梯度传播、训练稳定性和激活值分布；
- 在门控 MLP 中决定信息通过门的方式；
- 影响推理 kernel 的计算成本、融合方式和量化范围。

在现代 Transformer 中，常见名称可以分成两类：

1. **标量激活函数**：`ReLU`、`Sigmoid`、`Tanh`、`GELU`、`SiLU`，逐元素地把 \(x\) 映射为 \(\phi(x)\)；
2. **门控 MLP 激活结构**：`GLU`、`SwiGLU`、`GeGLU`、`SiTU`，同时接收 gate 和 up/value 两路张量并逐元素相乘。

理解这一区别很重要：`SiLU` 和 `GELU` 本身是标量函数，而 `SwiGLU` 和 Kimi K3 的 `SiTU` 是包含两路投影及逐元素乘法的门控结构。

### 基础函数

#### Sigmoid

Sigmoid 定义为

\[
\sigma(x)=\frac{1}{1+e^{-x}}.
\]

其输出范围为

\[
0 < \sigma(x) < 1,
\]

导数为

\[
\sigma'(x)=\sigma(x)(1-\sigma(x)).
\]

Sigmoid 可以把任意实数压缩到零和一之间，因此很适合作为概率或门控系数。输入很大时门接近打开，输入很小时门接近关闭。

但当 \(|x|\) 很大时，导数趋近于零，容易出现梯度饱和：

\[
\lim_{x\to\pm\infty}\sigma'(x)=0.
\]

现代 Transformer 通常不会把 Sigmoid 单独作为整个 MLP 的主激活函数，但它是 `SiLU` 和许多门控结构的重要组成部分。

#### Tanh

双曲正切定义为

\[
\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}.
\]

其输出范围为

\[
-1 < \tanh(x) < 1,
\]

导数为

\[
\frac{d}{dx}\tanh(x)=1-\tanh^2(x).
\]

与 Sigmoid 相比，Tanh 的输出以零为中心，但在绝对值较大的区域同样会饱和。Tanh 还可以用来构造平滑限幅函数：

\[
\operatorname{softcap}_{c}(x)=c\tanh\left(\frac{x}{c}\right).
\]

当 \(|x|\ll c\) 时，

\[
\operatorname{softcap}_{c}(x)\approx x;
\]

当 \(|x|\to\infty\) 时，输出平滑地趋近于 \(\pm c\)。Kimi K3 的 SiTU 正是使用这种方式限制 gate 和 up 分支的幅度。

#### ReLU

ReLU 定义为

\[
\operatorname{ReLU}(x)=\max(0,x).
\]

其分段导数为

\[
\operatorname{ReLU}'(x)=
\begin{cases}
0, & x<0,\\
1, & x>0.
\end{cases}
\]

在 \(x=0\) 处不可导，实际框架会约定一个次梯度，通常取零。

ReLU 的优点是计算简单，正半轴不会饱和，并且能产生精确的零值。缺点是负半轴全部被截断；如果神经元长期落在负区间，其梯度可能持续为零，这通常称为 dying ReLU。

ReLU 在经典神经网络中非常常见，但现代大语言模型的 MLP 更多使用平滑的 `GELU`、`SiLU` 或相应的门控变体。

### GELU

#### 精确定义

GELU 是 Gaussian Error Linear Unit。它可以写成

\[
\operatorname{GELU}(x)=x\Phi(x),
\]

其中 \(\Phi(x)\) 是标准正态分布的累积分布函数：

\[
\Phi(x)=\frac{1}{2}\left(1+\operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right).
\]

因此

\[
\operatorname{GELU}(x) =\frac{x}{2}\left(1+\operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right).
\]

它的直观含义不是像 ReLU 那样确定地保留所有正数、丢弃所有负数，而是根据输入在标准高斯分布中的位置进行平滑加权。

当 \(x\) 很大时，\(\Phi(x)\to 1\)，因此

\[
\operatorname{GELU}(x)\approx x.
\]

当 \(x\) 很小时，\(\Phi(x)\to 0\)，因此输出趋近于零。GELU 在负半轴并不恒为零，会保留一小段平滑的负输出。

其导数为

\[
\operatorname{GELU}'(x)=\Phi(x)+x\varphi(x),
\]

其中标准正态概率密度为

\[
\varphi(x)=\frac{1}{\sqrt{2\pi}}e^{-x^2/2}.
\]

#### Tanh 近似

精确 GELU 需要计算 `erf`。工程实现经常采用 Tanh 近似：

\[
\operatorname{GELU}_{\tanh}(x)
\approx
\frac{x}{2}
\left[
1+\tanh\left(
\sqrt{\frac{2}{\pi}}
\left(x+0.044715x^3\right)
\right)
\right].
\]

在 PyTorch 中，两种形式分别对应：

```python
import torch.nn.functional as F

y_exact = F.gelu(x, approximate="none")
y_tanh = F.gelu(x, approximate="tanh")
```

模型配置和 checkpoint 通常隐含了具体变体。推理时不能仅因为近似版本更快就任意替换，否则可能产生精度偏差。

#### QuickGELU

另一种近似是 QuickGELU：

\[
\operatorname{QuickGELU}(x)=x\sigma(1.702x).
\]

它在形式上与 SiLU 很相似，区别主要是 Sigmoid 输入前的缩放系数。QuickGELU 常见于部分视觉和多模态模型，但不能和精确 GELU、Tanh-GELU 或 SiLU 视为完全相同的函数。

### SiLU 与 Swish

SiLU 是 Sigmoid Linear Unit，定义为

\[
\operatorname{SiLU}(x)=x\sigma(x)=\frac{x}{1+e^{-x}}.
\]

在多数大模型语境中，SiLU 与参数为一的 Swish 是同一个函数：

\[
\operatorname{Swish}_{\beta}(x)=x\sigma(\beta x),
\qquad
\operatorname{SiLU}(x)=\operatorname{Swish}_{1}(x).
\]

其导数为

\[
\operatorname{SiLU}'(x) =\sigma(x)+x\sigma(x)(1-\sigma(x)).
\]

SiLU 有几个重要性质：

- 平滑且处处可导；
- 正半轴较大时近似恒等映射；
- 负半轴不会像 ReLU 那样全部变为零；
- 不是单调函数，在负半轴存在一个较浅的最低点；
- 输出没有上界，但负方向的输出幅度有限并趋近于零。

极限行为为

\[
\lim_{x\to+\infty}\operatorname{SiLU}(x)=x,
\qquad
\lim_{x\to-\infty}\operatorname{SiLU}(x)=0^{-}.
\]

PyTorch 中可以直接调用：

```python
import torch.nn.functional as F

y = F.silu(x)
```

### GELU、SiLU 与 ReLU 的直观对比

| 函数 | 定义 | 平滑性 | 负输入 | 正向大输入 | 常见用途 |
| --- | --- | --- | --- | --- | --- |
| ReLU | \(\max(0,x)\) | 在零点不光滑 | 全部截为零 | 近似 \(x\) | 经典 CNN、MLP |
| GELU | \(x\Phi(x)\) | 平滑 | 保留小幅负值 | 近似 \(x\) | BERT、GPT 系模型 |
| SiLU | \(x\sigma(x)\) | 平滑 | 保留小幅负值 | 近似 \(x\) | LLaMA、现代门控 MLP |
| Tanh | \(\tanh(x)\) | 平滑 | 压缩到负一和零之间 | 饱和到一 | 状态更新、平滑限幅 |
| Sigmoid | \(\sigma(x)\) | 平滑 | 接近零 | 饱和到一 | 概率和门控 |

GELU 与 SiLU 的曲线和用途相近，但定义并不相同：

\[
\operatorname{GELU}(x)=x\Phi(x),
\qquad
\operatorname{SiLU}(x)=x\sigma(x).
\]

两者都可以理解为“输入乘以一个随输入变化的软门”，但 GELU 的门来自高斯累积分布，SiLU 的门来自 Logistic Sigmoid。对于已经训练好的模型，不能把二者任意互换。

### GLU 与门控 MLP

#### 一般形式

GLU 是 Gated Linear Unit。设输入为 \(x\)，门控投影和数值投影分别为

\[
g=xW_g,
\qquad
u=xW_u.
\]

一般化的门控激活可以写成

\[
h=\phi(g)\odot u,
\]

再经过降维投影：

\[
y=hW_d.
\]

其中 \(\odot\) 表示逐元素乘法。门控分支 \(\phi(g)\) 决定数值分支 \(u\) 的哪些通道应该增强、衰减或改变符号。

#### 常见命名

GLU 家族通常根据门控分支使用的激活函数命名：

| 名称 | 门控计算 |
| --- | --- |
| GLU | \(\sigma(g)\odot u\) |
| ReGLU | \(\operatorname{ReLU}(g)\odot u\) |
| GeGLU | \(\operatorname{GELU}(g)\odot u\) |
| SwiGLU | \(\operatorname{SiLU}(g)\odot u\) |

因此 SwiGLU 不是 SiLU 的另一个名称。二者关系是：SwiGLU 在 gate 分支应用 SiLU，再乘以独立的 up/value 分支。

SwiGLU 的完整形式为

\[
\operatorname{SwiGLU}(x) =
\operatorname{SiLU}(xW_g)\odot(xW_u),
\]

也就是

\[
\operatorname{SwiGLU}(x) =
\left[(xW_g)\sigma(xW_g)\right]\odot(xW_u).
\]

对应的 PyTorch 伪代码是：

```python
gate = gate_proj(x)
up = up_proj(x)
hidden = torch.nn.functional.silu(gate) * up
output = down_proj(hidden)
```

#### 为什么 kernel 常叫 `silu_and_mul`

推理框架经常让一次 GEMM 同时产生 gate 和 up 两路：

\[
[g,u]=x[W_g,W_u].
\]

如果拼接结果的最后一维为 \(2D_{ff}\)，激活 kernel 将其切成两个宽度为 \(D_{ff}\) 的张量，然后计算

\[
h=\operatorname{SiLU}(g)\odot u.
\]

所以性能分析中常见 `silu_and_mul`、`act_and_mul` 或 fused SwiGLU kernel。它融合了以下操作：

1. 读取 gate 和 up；
2. 计算 SiLU；
3. 逐元素相乘；
4. 写回宽度减半的中间结果。

融合通常减少 kernel launch、中间张量读写和显存流量，但不会改变 SwiGLU 的数学语义。

### Kimi K3 的 SiTU

#### SiTU 不是 SiLU 的拼写变体

Kimi K3 使用的 SiTU，也称 SiTuGLU，是一种带平滑限幅的门控激活。它不是常见的通用标量激活函数，而是作用在 gate 和 up 两个分支上的 GLU 结构。

设 gate 分支为 \(g\)，up 分支为 \(u\)，gate softcap 参数为 \(\beta\)，up 分支的 softcap 参数为 \(\lambda\)，则 SiTU 定义为

\[
\widetilde{g} =
\beta\tanh\left(\frac{g}{\beta}\right)\sigma(g),
\]

\[
\widetilde{u} =
\lambda\tanh\left(\frac{u}{\lambda}\right),
\]

\[
\operatorname{SiTU}(g,u)=\widetilde{g}\odot\widetilde{u}.
\]

如果不对 up 分支启用 softcap，则可以令

\[
\widetilde{u}=u.
\]

Kimi K3 的一种配置使用

\[
\beta=4,
\qquad
\lambda=25.
\]

具体参数属于模型配置，不能把上述数值当成所有 SiTU 模型的固定常量。

#### 与 SwiGLU 的关系

SwiGLU 为

\[
\operatorname{SwiGLU}(g,u) =
g\sigma(g)\odot u.
\]

SiTU 则把两路输入先做平滑限幅：

\[
\eta\tanh\left(\frac{g}{\beta}\right)
\quad\text{替代}\quad g,
\]

并可选地使用

\[
\lambda\tanh\left(\frac{u}{\lambda}\right)
\quad\text{替代}\quad u.
\]

当 \(|g|\ll\beta\) 且 \(|u|\ll\lambda\) 时，由于 \(\tanh(z)\approx z\)，有

\[
\widetilde{g}\approx g\sigma(g),
\qquad
\widetilde{u}\approx u.
\]

因此在输入幅度较小时，SiTU 近似 SwiGLU：

\[
\operatorname{SiTU}(g,u)
\approx
\operatorname{SwiGLU}(g,u).
\]

当输入幅度很大时，SiTU 的 softcap 开始发挥作用：

\[
\left|\beta\tanh\left(\frac{g}{\beta}\right)\right|<\beta,
\qquad
\left|\lambda\tanh\left(\frac{u}{\lambda}\right)\right|<\lambda.
\]

因此 SiTU 可以抑制 gate 和 up 分支中的极端值。若两路都启用 softcap，则输出幅度满足一个粗略上界：

\[
|\operatorname{SiTU}(g,u)|<\beta\lambda,
\]

因为 \(0<\sigma(g)<1\)。这个有界性质对低精度训练、量化和异常值控制可能更友好，但最终效果取决于模型训练方式，不能把激活函数替换视为纯推理优化。

#### SiTU 的 PyTorch 参考实现

```python
import torch


def situ_and_mul(x, beta: float, linear_beta: float | None):
    gate, up = x.chunk(2, dim=-1)

    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)

    return gate * up
```

这里输入最后一维为 \(2D_{ff}\)，输出最后一维为 \(D_{ff}\)。它和 `silu_and_mul` 一样适合融合成一个 elementwise kernel，也可以与第一层 GEMM、量化或 MoE 路由后的计算进一步融合。

### 参数量与中间维度

标准两层 MLP 为

\[
y=W_d\phi(W_ux),
\]

忽略 bias 时，参数量约为

\[
P_{\text{standard}}=2D_{model}D_{ff}.
\]

门控 MLP 有 gate、up 和 down 三个矩阵：

\[
y=W_d\left[\phi(W_gx)\odot(W_ux)\right],
\]

参数量约为

\[
P_{\text{gated}}=3D_{model}D_{ff}.
\]

如果直接使用相同的 \(D_{ff}\)，门控 MLP 的参数和 GEMM 计算量都会更大。为了与扩展倍率为四的标准 MLP 保持接近的参数量，可以令

\[
3D_{model}D_{ff}^{\text{gated}}
\approx
2D_{model}(4D_{model}),
\]

得到

\[
D_{ff}^{\text{gated}}
\approx
\frac{8}{3}D_{model}.
\]

这就是许多门控 Transformer 的 intermediate size 接近 \(8D_{model}/3\)，而不是简单使用 \(4D_{model}\) 的原因之一。实际维度还会对齐到适合 TP 切分和硬件 GEMM 的整数倍。

### 推理与量化注意事项

#### 不要擅自替换激活函数

激活函数是模型训练参数的一部分。即使 GELU、SiLU 和 SiTU 的曲线看起来相似，替换后也会改变每层输出：

\[
\phi_1(x)\ne\phi_2(x).
\]

误差经过多层 residual block 累积，可能影响困惑度、生成结果和数值稳定性。推理实现必须遵循模型配置中的激活类型、近似方式和 SiTU 参数。

#### 激活函数与量化

ReLU 的负半轴为零，而 GELU 和 SiLU 会产生小幅负值；SwiGLU 的 up 分支没有天然上界；SiTU 则可以通过 Tanh softcap 控制两路范围。这些差异会影响：

- activation 的最小值和最大值；
- per-token 或 per-group quantization scale；
- outlier 的数量；
- FP8、INT8、MXFP 等低精度格式的有效利用率；
- fused activation-and-quantization kernel 的设计。

但 softcap 不代表量化一定无损。量化误差还取决于数据分布、scale 粒度、舍入方式以及后续 down projection 对通道的敏感性。

#### 精度与快速数学

Sigmoid、Tanh、`erf` 和指数函数可能使用硬件近似或 fast-math 实现。不同 kernel 的结果可能出现细小差异。实现精度验证时，应区分：

- 数学定义是否一致；
- GELU 使用精确形式还是 Tanh 近似；
- 中间计算使用 FP32、BF16、FP16 还是 FP8；
- 激活与乘法是否融合；
- 是否启用 fast-math；
- 比较误差来自函数近似还是低精度舍入。

### 快速记忆

可以用下面几句话区分这些名称：

- **Sigmoid**：把数压到零和一之间，常用作门；
- **Tanh**：把数压到负一和一之间，可用于平滑限幅；
- **ReLU**：负数归零，正数原样通过；
- **GELU**：输入乘以高斯 CDF 形成的软门；
- **SiLU/Swish**：输入乘以 Sigmoid 形成的软门；
- **SwiGLU**：对 gate 做 SiLU，再乘独立的 up 分支；
- **GeGLU**：对 gate 做 GELU，再乘独立的 up 分支；
- **SiTU**：类似 SwiGLU，但使用 Tanh softcap 平滑限制 gate，并可同时限制 up。

最容易混淆的关系是

\[
\operatorname{SiLU}(x)=x\sigma(x),
\]

\[
\operatorname{SwiGLU}(g,u)=\operatorname{SiLU}(g)\odot u,
\]

\[
\operatorname{SiTU}(g,u) =
\left[
\beta\tanh\left(\frac{g}{\beta}\right)\sigma(g)
\right]
\odot
\left[
\lambda\tanh\left(\frac{u}{\lambda}\right)
\right].
\]

前者是单输入标量函数，后两者是双分支门控 MLP 激活结构。


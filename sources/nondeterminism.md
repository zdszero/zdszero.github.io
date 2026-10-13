% Nonderterminsim

`temperature=0` 时采样退化为贪心解码，也就是每步都选概率最高的 token，理论上应该是确定的。但实践中并非如此：

- 商业 LLM API 用同一个 prompt 反复请求，返回的内容会不一样；
- 自建推理服务（vLLM、SGLang 等）用相同输入重复推理，logprob 也会有差异。

一个流传很广的解释是「并发 + 浮点」假设：GPU 高度并行，浮点又不满足结合律，并发线程的完成顺序不确定，导致累加顺序不确定，于是输出不确定。

这个假设并不完全错，但它解释不了下面这个反例：

```python
A = torch.randn(2048, 2048, device='cuda', dtype=torch.bfloat16)
B = torch.randn(2048, 2048, device='cuda', dtype=torch.bfloat16)
ref = torch.mm(A, B)
for _ in range(1000):
    assert (torch.mm(A, B) - ref).abs().max().item() == 0
```

同一个 matmul、同一份数据，跑 1000 次，按位完全相同。这里既用了浮点，也有大量并发，却没有任何 run-to-run 不确定性。所以真正的原因要更细一层。

下面四句话看起来互相矛盾，但它们同时成立：

| 陈述 | 是否成立 |
| --- | --- |
| GPU 上确实存在不确定的 kernel | 成立 |
| LLM 前向用到的 kernel 全都是确定的 | 成立 |
| 因此推理服务的前向过程也可以说是确定的 | 成立 |
| 但从使用服务的用户视角看，结果就是不确定的 | 成立 |

前两条不冲突，是因为 LLM 前向里几乎不含不确定的 kernel（下文解释）；第三条和第四条不冲突，是因为「给定完全相同的输入，服务是确定的」并不足以让系统整体确定——用户的请求还要和「别人同时在请求什么」这个不确定因素组合。真正缺失的性质叫 **batch invariance**。

### 浮点非结合性：差异从哪里来

在真实硬件上，浮点计算不像数学那样严格满足结合律：

\[
(a+b)+c \ne a+(b+c)
\]

```python
(0.1 + 1e20) - 1e20
>>> 0
0.1 + (1e20 - 1e20)
>>> 0.1
```

本质原因是浮点数采用 **有限精度 + 舍入（rounding）**：

\[
\text{float} = (-1)^{s} \times m \times 2^{e}
\]

其中 \(s\) 是符号位，\(m\) 是尾数（mantissa），\(e\) 是指数（exponent）。为了便于说明，下面用 10 进制、3 位尾数、1 位指数来举例，结论与二进制完全一致。

浮点的价值在于 **有效数字位数恒定、量级可以任意伸缩**：\(3450\) 写成 \(3.45\times10^{3}\)，\(0.486\) 写成 \(4.86\times10^{-1}\)，都只用 3 位有效数字。代价是 **绝对精度随指数变化**。

看一次加法 \(1230 + 23.4\)：

- 精确结果是 \(1253.4\)，需要 5 位有效数字；
- 但格式只能保留 3 位，对齐指数时 \(23.4\) 实际上被舍入成了 \(20.0\)；
- 结果只能得到 \(1.25\times10^{3}\)。

信息在加法中被永久破坏了。这就是所谓的 **大数吃小数**（swamping）：只要两个数的指数相差足够大，小的那个就会被整个吞掉。也正因为如此，**每一次改变加法顺序，都可能得到完全不同的结果**。

#### 用单位舍入误差量化

尾数精度为 \(p\) 位时（IEEE-754 的 round-to-nearest），单位舍入误差（unit roundoff）为：

\[
u = 2^{-p}
\]

| 类型 | 尾数精度 \(p\) | \(u = 2^{-p}\) |
| --- | --- | --- |
| fp64 | 53 | \(1.11\times10^{-16}\) |
| fp32 | 24 | \(5.96\times10^{-8}\) |
| fp16 | 11 | \(4.88\times10^{-4}\) |
| bf16 | 8 | \(3.91\times10^{-3}\) |

单次加法的相对误差可以写成：

\[
fl(a+b) = (a+b)(1+\delta), \qquad |\delta| \le u
\]

于是顺序求和的误差界是：

\[
\left|\hat{s} - \sum_{i=1}^{n} x_i\right| \le \gamma_{n-1}\sum_{i=1}^{n}|x_i|,
\qquad
\gamma_{n} = \frac{n u}{1 - n u}
\]

如果换成树形（pairwise / 分块）求和，误差界变成：

\[
\left|\hat{s} - s\right| \le \lceil \log_2 n \rceil\, u \sum_{i=1}^{n}|x_i|
\]

两者相差一个 \(n\) 与 \(\log_2 n\) 的因子，这就是「归约顺序会实质影响数值」的定量来源。注意 \(u\) 越小并不代表差异越小：误差界同时正比于 \(\sum_i|x_i|\)，而 LLM 里的累加规模（\(K\) 维、序列长度）很大，所以即便 fp32 累加也会累积出可观偏差。

一个极端例子：取 `[1e-10, 1e-5, 1e-2, 1]` 以及它们各自的相反数，共 8 个数，随机打乱后求和：

```python
import random

vals = [1e-10, 1e-5, 1e-2, 1]
vals = vals + [-v for v in vals]

results = []
random.seed(42)
for _ in range(10000):
    random.shuffle(vals)
    results.append(sum(vals))

print(len(set(results)))
# There are 102 unique results: [-8.326672684688674e-17, ..., 8.326672684688674e-17]
```

数学上这 8 个数的和恒为 0，实际却出现了 102 个不同的结果。

{{< region note >}}
浮点非结合性只解释了「为什么会有数值差异」，**没有**解释「为什么会有不确定性」。只要 kernel 每次都用同一个顺序累加，结果就是确定的——前面那个跑 1000 次都按位相同的 matmul 正是如此。所以真正要问的是：**什么情况下累加顺序会变？**
{{< /region >}}

### 三种必须区分开的「确定性」

讨论确定性时，至少有三个不同的性质容易混为一谈：

| 性质 | 含义 | 典型的破坏者 |
| --- | --- | --- |
| run-to-run 确定性 | 相同输入、相同软件栈，重复执行结果按位相同 | atomic add、并发归约 |
| batch invariance | 某条样本的结果与其所在 batch 的大小、以及同 batch 中其他样本无关 | 随形状变化的 tile / split 策略 |
| 硬件与版本不变性 | 换 GPU、换 PyTorch / CUDA / 库版本后结果仍按位相同 | 不同指令集、不同 kernel 实现 |

前两者的区别是本主题的关键。

#### atomic add 会造成 run-to-run 不确定性

GPU 会把程序并发地铺到许多 SM 上，SM 之间没有天然同步。如果多个 core 需要累加到同一个元素，就要用 **atomic add**（fetch-and-add）：硬件保证每个 core 的贡献都会被加进去，但**不保证顺序**，顺序完全取决于谁先算完。

比如用 100 个 core 归约一个 100 维向量（类似 `torch.sum()`）：可以并行读入 100 个元素，但最终必须归约成一个标量，如果直接用 atomic add 累加，结果就依赖 core 的完成顺序。这就是通常意义上说的不确定 kernel，重复执行同一个程序会得到不同输出。

但关键在于：**LLM 前向里通常一次 atomic add 都没有**。原因是：

1. batch 维往往已经提供了足够的并行度，不需要再沿归约维切分。例如与其归约 1 个 100 维向量，不如让每个 core 各归约一个向量，500 个向量并行处理；
2. 主流神经网络库为了可复现性，长期采用确定性的归约策略（固定 tile、固定归约树、确定性 split reduction 等）。

所以「并发 + 浮点」假设指出的机制存在于 GPU 上，但它不是 LLM 推理不确定性的来源。

{{< region note >}}
一个相关的实用事实：并行本身引入的归约（TP / EP / DP 里的 AllReduce 等）是否确定，取决于通信实现。文章提到 NVLink 的 SHARP in-switch reduction 在 Blackwell，以及 Hopper + CUDA 12.8+ 上是确定的。这部分不在本文讨论范围内，但原则和单卡 kernel 相同。
{{< /region >}}

### batch invariance：batch 大小为什么会污染单条请求

从数学上看，矩阵乘法沿 batch 维是可分解的：

\[
\begin{bmatrix}
A \\
B
\end{bmatrix}
C = \begin{bmatrix}
AC \\
BC
\end{bmatrix}
\]

也就是说：无论怎么切 batch（哪怕 batch size = 1），每个样本的结果都应该只由它自己决定。

```python
# 一次性算
Y = X @ W

# 分 batch 算
Y = torch.cat([X[:k] @ W, X[k:] @ W], dim=0)
```

但工程实现上并非如此。GPU kernel 把工作切成 tile 分配给 SM，而 **tile 划分、split 数量、wave 数量都会随形状变化**，从而改变累加顺序。矩阵乘法的 tile 划分通常在 \(M\) 和 \(N\) 两个维度上进行，因此 batch 大小（\(M\)）会直接改变数值。

```python
import torch
torch.set_default_device('cuda')

B = 2048
D = 4096
a = torch.linspace(-1000, 1000, B * D).reshape(B, D)
b = torch.linspace(-1000, 1000, D * D).reshape(D, D)

# 取 batch 的第 0 个元素做矩阵-向量乘
out1 = torch.mm(a[:1], b)
# 做完整的矩阵-矩阵乘，再取第 0 个元素
out2 = torch.mm(a, b)[:1]
print((out1 - out2).abs().max())  # tensor(1669.2500, device='cuda:0')
```

同一行输入、同一个权重，两次结果的差异高达 \(1669\)。注意这个 kernel 仍然是 **run-to-run 确定的**：脚本重复跑结果一致。它不满足的是 batch invariance，也不满足跨硬件/版本的不变性。

#### 组合出端到端的不确定性

把「kernel 对某个属性不具不变性」和「这个属性本身是随机的」组合起来，就得到了不确定的系统：

\[
\text{非 batch-invariant 的 kernel} \;\circ\; \text{不确定的 batch size}
\;\Rightarrow\; \text{不确定的输出}
\]

用户发请求时，服务端负载是未知的，负载决定 batch size，batch size 决定数值，于是每条请求的结果都受其他并发请求影响。单个请求无法知道别人会同时请求什么，所以从它的视角看，整个推理系统就是不确定的。

这也不是 GPU 独有的：CPU 或 TPU 上的推理服务有完全相同的这一层不确定性来源。**只要 kernel 的结果依赖 batch 大小，服务端负载就会变成用户可见的随机性。**

### 需要处理的三个算子

要让整个 Transformer 实现 batch-invariant，就要让每个 kernel 都 batch-invariant。

- **逐元素（pointwise）算子天然满足。** 输出第 \(i\) 个元素只依赖输入第 \(i\) 个元素，没有跨元素交互。不过这不是无条件的：某些 CPU 实现会对同一数组的不同区段使用不同的（向量化 / 非向量化）指令，二者不保证按位一致。
- **真正需要额外处理的是三个含 reduction 的算子**：RMSNorm、矩阵乘法、attention，难度依次递增。

核心判据是：

> **只有当 batch size 改变了 reduction 策略时，才破坏 batch invariance。**

这里有个容易误读的地方：它不等于「永远只能用同一个 reduction 策略」。如果被归约的元素个数本身就变了（例如序列长度不同），换策略仍然是 batch-invariant 的。问题出在 **被归约的元素个数相同，但因为 batch 变大或变小而换了策略**。

另一个实用的观察对象是 **线程块数量与 SM 数量的比值**：工程实现通常在并行度不足以喂满 GPU 时才切换策略，而这个阈值点正是 batch invariance 断裂的地方。

#### Batch-invariant RMSNorm

RMSNorm 的定义是：

\[
\operatorname{RMSNorm}(x_i)
=
\frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} x_{ij}^{2} + \epsilon}}
\odot \gamma
\]

实现上就是一次沿特征维的归约：

```python
# x: [batch_size, hidden_dim]
# weight: [hidden_dim]
def rms_norm(x, weight):
    return x * torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True)) * weight
```

最自然的策略是 **数据并行**：一个 core 负责一行 batch，整行的归约完全留在核内，不需要跨核通信。

- **batch 变大**：让每个 core 依次处理多行即可，每行的归约顺序不变，仍然 batch-invariant。batch size 从 200 涨到 2000，并行度只会更充足。
- **batch 变小**：一旦 batch 小于 core 数量，就会有 core 空闲。这时 kernel 工程师的本能是 split reduction（把一行的归约切给多个 core 再合并）或 atomic add——两者都改变归约顺序，破坏 batch invariance。

面对小 batch，有两个选择：

1. **不为这种情况优化。** batch 很小时 kernel 本身执行很快，性能损失通常可以接受。
2. **固定使用一种「即使 batch = 1 也有足够并行度」的归约策略。** 代价是大 batch 时并行度过剩，只能达到次优但尚可的性能，换来了全形状范围内的一致数值。

#### Batch-invariant 矩阵乘法

矩阵乘法的标准策略同样是数据并行：把输出切成二维 tile，一个 tile 交给一个 core，整个 \(K\) 维归约留在核内：

\[
C_{ij} = \sum_{k} A_{ik} B_{kj}
\]

与 RMSNorm 的区别是：出于算术强度和 tensorcore 利用率的要求，矩阵乘法必须按二维 tile 切分，而不能按单个输出元素切分。

破坏不变性的情况有三种：

1. **Split-K matmul。** \(M\) 和 \(N\) 同时很小时并行度不足。例如 \([1024, K] \times [K, 1024]\) 配 \([128, 128]\) 的 tile，只能切出 64 个 tile，喂不饱 GPU。于是改为沿 \(K\) 切分，每个 core 各算一部分、最后再合并——归约顺序被改变。
2. **Stream-K。** 它为了更好的负载均衡，会对不同的输出 tile 采用不同的 \(K\) 切分方式。普通的 matmul 库至少还满足更弱的 **batch-position invariance**（同 batch 内元素换位置不影响数值），而 stream-k 连这一点都不满足。
3. **tensorcore 指令的粒度。** 一条 tensorcore 指令（例如 `wgmma.mma_async.sync.aligned.m64n128k16`）内部有自己的归约顺序。batch 很小时，即便一个 \([128, 128]\) 的 tile 也放不下，最快的内核会退化成更小的指令，甚至完全不用 tensorcore，数值随之改变。

解决方式很直接：**只编译一种 kernel 配置，所有形状都用它**。性能损失通常可以接受，因为 split-K 主要用在 \(M\)、\(N\) 同时很小的场合，而 LLM 里 \(N\)（模型维度）通常很大。文章中的实现比 cuBLAS 慢约 20%，而且它是未优化的 Triton kernel（例如没有使用 TMA）。

性能曲线上的两个特征值得记下来：

- 极小 batch 时明显变慢，因为 instruction 过大而并行度不足；
- 中等 batch 区间出现「锯齿」（jigsaw）形状，来自 tile 量化与 wave 量化效应。常规库会通过切换 tile size 来消除这种锯齿，而固定配置意味着必须接受它。

#### Batch-invariant attention

FlashAttention2 的策略是沿 Q 并行、沿 K/V 归约，本质上也是数据并行：一次归约完整地留在单个 core 内。

Attention 比前两者多两个麻烦：

1. **归约维度有两个**，既有特征维 \(d\)，也有序列维：

\[
O_i = \sum_{j \le i} \operatorname{softmax}\!\left(\frac{q_i^{\mathsf T} k_j}{\sqrt{d}}\right) v_j
\]

其中 \(j \le i\) 体现 causal mask。decode 阶段 query 长度通常只有 1，而 \(L\) 是整个上下文长度（包含 KV Cache）。

2. **推理引擎会用各种方式切分序列**：chunked prefill、prefix caching 等。因此数值必须同时对「一次处理多少请求」和「一条请求被切成几段」保持不变。

**KV Cache 的边界条件问题。** 假设 block size 是 32，KV Cache 里已有 80 个元素，本次又要处理 48 个新元素：

- 计算 cache 部分的注意力，需要 3 个 block（2 个满块 + 1 个 masked 块）；
- 计算本次新 token 部分，需要 2 个 block（1 个满块 + 1 个 masked 块）；
- 合计 5 个 block 的归约。

如果这 128 个元素是一次性处理的（KV Cache 为空），只需要 4 个满块。两种情况归约顺序不同，因而不满足 batch invariance。按文章的说法，处理某条序列的第 1000 个 token 时，无论前面 999 个 token 是已在 KV Cache 里还是本次一起计算，结果都必须完全一致。

解法是把 KV Cache 与 page table 的更新放到 attention kernel **之前**执行，使 K/V 的排布与「本次处理多少 token」无关。

**固定 split size 的 Split-KV。** 与 matmul 不同，decode 阶段的 attention 经常**必须**切分归约（Split-KV，也叫 FlashDecoding）：如果不沿归约维切分，就只能沿 batch、head 和 query 长度并行，而 decode 时 query 长度很小，batch 也不大时就无法打满 GPU。而且这种情况不能像 RMSNorm 那样「直接忽略」——KV Cache 很长时，attention kernel 即便只处理一条请求也可能非常慢。

常用的 split 策略是「按需要多少并行度把 KV 维平均切分」：KV 长度 1000、需要 4 份，每份就是 250。这个切法与当前处理多少 query token 相关，因此破坏 batch invariance。FlashInfer 的 balanced scheduling 也是类似思路：选取仍能打满所有 core 的最大 split size。

正确做法是反过来——**固定每个 split 的大小，而不是固定 split 的数量**：

- 破坏不变性：KV 长度 1000 切成 4 份，每份 250；
- 保持不变性：KV 长度 1000 切成 3 个长度 256 的定长块 + 1 个长度 232 的块。

这样无论一次处理多少 token，归约顺序都完全相同。（文章指出这需要在 FlexAttention 内部做修改，未包含在其开源代码中。）

### 实现方式与可用开关

- 该工作基于 vLLM 的 FlexAttention backend 演示确定性推理，并通过 `torch.Library` 在不侵入上游代码的前提下替换相关 PyTorch 算子。
- batch-invariant kernel 集合开源在 `thinking-machines-lab/batch-invariant-ops`。
- vLLM 上游也提供了开关 `VLLM_BATCH_INVARIANT`：默认关闭，开启后要求 NVIDIA GPU 且 compute capability \(\ge 9.0\)（Hopper / Blackwell），对应上文中 SHARP in-switch reduction 的确定性条件。
- PyTorch 自带的 `torch.use_deterministic_algorithms(True)` 与 `CUBLAS_WORKSPACE_CONFIG` 解决的是 **run-to-run** 不确定性（禁用 atomic add 等），**不**保证 batch invariance。两者不能互相替代：前者开启后，同一个 batch 形状重复跑会一致，但换一个 batch 形状仍可能得到不同数值。

### 实验：不确定性到底有多大

设置：`Qwen/Qwen3-235B-A22B-Instruct-2507`，`temperature=0`，prompt 为 “Tell me about Richard Feynman”，每条输出 1000 token，共采样 1000 次。

| 现象 | 数值 |
| --- | --- |
| 不同的 completion 数量 | 80 |
| 最常见 completion 的出现次数 | 78 |
| 首次出现分歧的位置 | 第 103 个 token |
| 生成 “Queens, New York” | 992 次 |
| 生成 “New York City” | 8 次 |

前 102 个 token 完全一致，都生成了 “Feynman was born on May 11, 1918, in”，到第 103 个 token 才分成两支。开启 batch-invariant kernel 后，1000 次采样完全相同——这正是 `temperature=0` 在数学上应该给出的结果。

性能方面，单卡 Qwen3-8B，请求 1000 条、输出长度 90 到 110：

| 配置 | 耗时（秒） |
| --- | --- |
| vLLM 默认 | 26 |
| 未优化的确定性 vLLM | 55 |
| 上一行 + 改进的 attention kernel | 42 |

作者强调这部分 kernel 并未认真优化（vLLM 里的 FlexAttention 集成也没怎么调），所以结论是「性能没有灾难性下降」，而不是「确定性几乎免费」。

### True on-policy RL

训练与推理之间的数值差异，会把名义上的 on-policy RL 变成隐式的 off-policy RL。既然连两次相同的推理请求都无法按位一致，就更不可能让采样与训练按位一致。

实验设置：RLVR，数据集 Bigmath，策略初始化自 Qwen 2.5-VL instruct 8B，max rollout length 为 4096。三种配置的对比：

| 配置 | 行为 |
| --- | --- |
| 不加 off-policy correction（importance weighting） | reward 在训练中途崩溃，约 step 318 出现 loss spike，同时 logprob 的 KL 散度出现对应尖峰 |
| 加 importance weighting | KL 维持在约 \(0.001\)，偶有尖峰，训练平稳 |
| sampler 与 trainer 按位一致（True On-Policy） | KL 恒为 \(0\)，训练平稳 |

也就是说，只要推理与训练的数值完全对齐（\(0\) KL 散度），不加任何 off-policy 修正也能稳定训练。

### 如何度量不确定性

最直接的指标是逐 token 的 logprob 差的平方均值：

\[
\operatorname{MSE} = \frac{1}{n} \sum_{i=1}^{n} (x_i - y_i)^2
\]

其中：

- \(x_i\)：单条推理（batch size = 1）第 \(i\) 个 token 的 logprob；
- \(y_i\)：batch 推理时同一 token 的 logprob。

实际排查时通常配合另外两个指标：

- **top-1 分歧率**：两条推理中 argmax token 不同的比例。它最贴近用户可见的行为，因为贪心解码只看 argmax，MSE 很小也可能导致 top-1 翻转；
- **KL 散度** \(\operatorname{KL}(p_{\text{sampler}} \| p_{\text{trainer}})\)：直接对应 on-policy 与 off-policy 的偏离程度，是 RL 场景里最有说服力的指标。

### 工程上的取舍

- **可以不管的场景**：本来就依赖采样随机性工作的流程，例如 `temperature > 0` 的多样本生成、取了平均的评测。此时 batch invariance 带来的差异会被采样噪声淹没。
- **必须管的场景**：`temperature = 0` 的可复现评测、on-policy RL、需要精确复现 bug 的调试，以及任何「同一请求两次结果不同」会被当成故障的线上系统。
- **另一种规避方式**是把 batch 固定填充到同一个大小（或让服务在某一时刻只处理一条请求）。代价是吞吐下降：这与 batch-invariant kernel 的目标相同（消除形状依赖），只是把成本从 kernel 性能转移到了并行度上。
- **注意结论的边界**：batch-invariant kernel 保证的是「同一软件栈、同一硬件上的可复现」，它不保证跨 GPU 架构或跨库版本按位一致。

### 总结

- 浮点非结合性 \((a+b)+c \ne a+(b+c)\) 解释的是「为什么会有数值差异」，不是「为什么会有不确定性」；
- atomic add 会破坏 run-to-run 确定性，但 LLM 前向里基本不出现，因此「并发 + 浮点」不是主因；
- 端到端不确定性的真正来源是 kernel 缺少 batch invariance，与「服务负载即 batch size 本身随机」组合而成；
- 需要处理的是三个含归约的算子：RMSNorm（数据并行）、矩阵乘法（固定 tile 配置）、attention（固定 split size，并统一 KV Cache 排布）；
- 代价是性能（文章实现约比 cuBLAS 慢 20%），收益是可复现的贪心解码与真正的 on-policy RL。

---

- Horace He, Thinking Machines Lab, [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/), 2025.
- batch-invariant kernel 实现：`thinking-machines-lab/batch-invariant-ops`。
- 浮点误差界与求和分析：Higham, *Accuracy and Stability of Numerical Algorithms*。


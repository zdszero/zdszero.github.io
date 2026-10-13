% Norm

## 数据处理层面

Normalization 的英文本身含义就是让数据变得 Normal，但是根据使用场景常常拥有不同的语义。

一个是在数据处理层面，另外一个是处理模型内部的数值。

### Min-Max

(最小-最大归一化)

*   **公式**：\(x' = \frac{x - \min(x)}{\max(x) - \min(x)}\)
*   **结果**：数据被压缩到 **\([0, 1]\)** 区间。
*   **用途**：图像处理（像素值 0-255 转 0-1）、某些对范围敏感的算法。

### Standardization

*   **公式**：\(x' = \frac{x - \mu}{\sigma}\) （减去均值，除以标准差）
*   **结果**：数据变为**均值为 0，方差为 1**的标准正态分布。**注意：数值范围不再是 0-1，可能是 -3 到 +3 甚至更大。**
*   **用途**：大多数深度学习模型的首选预处理方式（如 ImageNet 的均值减法）。

{{< details "Standardization 和 LayerNorm 的语义区别" >}}
Standardization 是：

> 在数据进入模型之前做的

LayerNorm 是：

> 在每一层之间做的

- 前者是数据工程
- 后者是数值工程
{{< /details >}}

## 模型层面

### LayerNorm

**Layer Normalization（LayerNorm）** 是一种对神经网络中间特征进行归一化的方法。与 Batch Normalization 不同，LayerNorm **不依赖 batch 中其他样本的数据**，而是对**单个样本的特征维度**进行归一化，因此特别适合 Transformer、RNN 等序列模型，也天然适用于变长序列。

对于一个特征向量

\[
\mathbf{x}=(x_1,x_2,\cdots,x_d)
\]

LayerNorm 首先计算该向量在特征维度上的均值和方差，然后进行标准化：

\[
y_i=
\frac{x_i-\mu}
{\sqrt{\sigma^2+\epsilon}}
\cdot\gamma_i+\beta_i
\]

其中：

* \(y_i\)：LayerNorm 输出的第 \(i\) 个特征。
* \(x_i\)：输入向量的第 \(i\) 个特征。
* \(d\)：进行归一化的特征维度大小。例如 Transformer 的 hidden size 为 \(7168\) 时，通常就是对这 \(7168\) 个特征进行归一化。
* \(\mu\)：输入特征的均值。
* \(\sigma^2\)：输入特征的方差。
* \(\epsilon\)：一个很小的正数，用于提高数值稳定性，避免方差过小时出现除零问题。
* \(\gamma\)：可学习的缩放参数（scale）。
* \(\beta\)：可学习的偏移参数（bias）。

#### 1. 计算均值

\[
\mu=\frac{1}{d}\sum_{i=1}^{d}x_i
\]

即计算当前样本在被归一化特征维度上的平均值。

#### 2. 计算方差

\[
\sigma^2=
\frac{1}{d}
\sum_{i=1}^{d}(x_i-\mu)^2
\]

方差描述各个特征相对于均值的离散程度。

#### 3. 标准化

\[
\hat{x}_i=
\frac{x_i-\mu}
{\sqrt{\sigma^2+\epsilon}}
\]

经过这一步后，该组特征的均值接近 \(0\)，方差接近 \(1\)。

#### 4. 仿射变换

\[
y_i=\gamma_i\hat{x}_i+\beta_i
\]

其中 \(\gamma_i\) 和 \(\beta_i\) 都是模型训练过程中学习得到的参数。

因此，LayerNorm 可以理解为：

> LayerNorm = Normalize + Scale + Bias

归一化负责消除特征尺度和均值的影响，而 \(\gamma\) 和 \(\beta\) 则允许模型在训练过程中重新调整归一化后的特征。

---

**LayerNorm 到底对哪些数据进行归一化？**

这是理解 LayerNorm 最重要的一点。

假设输入张量为：

\[
X\in\mathbb{R}^{B\times S\times D}
\]

其中：

* \(B\)：batch size
* \(S\)：序列长度
* \(D\)：hidden size

例如：

\[
X\in\mathbb{R}^{8\times2048\times7168}
\]

对于每一个 token，LayerNorm **只在最后的 hidden dimension \(D=7168\) 上计算均值和方差**。

也就是说，对于：

\[
X[b,s,:]\in\mathbb{R}^{7168}
\]

单独计算：

\[
\mu_{b,s} =
\frac{1}{7168}
\sum_{i=1}^{7168}X[b,s,i]
\]

以及：

\[
\sigma^2_{b,s} =
\frac{1}{7168}
\sum_{i=1}^{7168}
(X[b,s,i]-\mu_{b,s})^2
\]

然后对这个 token 的 7168 个特征进行归一化。

因此：

> **LayerNorm 不会把整个 batch 或整个 sequence 混在一起计算均值和方差，而是对每个 token 独立地在 hidden dimension 上进行归一化。**

这也是 LayerNorm 与 BatchNorm 的一个重要区别。

---

**\(\gamma\) 和 \(\beta\) 的作用**

归一化之后：

\[
\hat{x}_i =
\frac{x_i-\mu}
{\sqrt{\sigma^2+\epsilon}}
\]

特征的均值和尺度被统一了。

但如果始终固定为这种标准化结果，模型的表达能力可能受到限制。因此 LayerNorm 引入可学习参数：

\[
y_i=\gamma_i\hat{x}_i+\beta_i
\]

其中：

* \(\gamma_i\) 控制第 \(i\) 个特征的**缩放幅度**；
* \(\beta_i\) 控制第 \(i\) 个特征的**偏移量**。

例如，如果某个特征经过归一化后：

\[
\hat{x}_i=0.5
\]

而模型学习到：

\[
\gamma_i=2,\qquad\beta_i=-1
\]

那么：

\[
y_i=2\times0.5-1=0
\]

因此，LayerNorm 并不是简单地把所有特征强行固定到某个分布，而是：

> **先进行标准化，再通过可学习的 \(\gamma\) 和 \(\beta\) 恢复模型认为有用的尺度和偏移。**

---

**LayerNorm 的主要作用**

LayerNorm 的核心作用并不是简单地“让每一层输入具有稳定分布”，更准确地说，是对每个 token 的特征进行中心化和尺度归一化，从而改善深层网络训练过程中的数值尺度和优化稳定性。

在 Transformer 中，hidden state 会经过大量层的重复变换：

\[
X
\rightarrow
\text{Attention}
\rightarrow
X
\rightarrow
\text{MLP}
\rightarrow
X
\rightarrow\cdots
\]

如果不同层之间的特征尺度发生剧烈变化，可能使训练和优化变得更加困难。

LayerNorm 通过：

\[
x
\rightarrow
\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
\]

将特征重新调整到相对稳定的尺度，再通过 \(\gamma,\beta\) 保留模型所需要的可学习自由度。

因此，它通常有助于：

* 稳定深层网络中的数值尺度；
* 改善梯度传播和优化过程；
* 降低不同 token 的特征尺度差异；
* 提高 Transformer 等深层序列模型的训练稳定性。

---

**注意：现代 LLM 不一定同时存在 \(\gamma\) 和 \(\beta\)**

上述公式描述的是**完整的 LayerNorm**：

\[
y_i=\gamma_i\hat{x}_i+\beta_i
\]

其参数通常为：

```text
weight [D]   ← γ
bias   [D]   ← β
```

但是在实际的大语言模型中，Normalization 的实现可能有所不同。例如某些实现会使用 **bias-free LayerNorm**，此时只有：

\[
y_i=\gamma_i\hat{x}_i
\]

对应参数只有：

```text
weight [D]
```

而没有：

```text
bias [D]
```

此外，现代 LLM 中还大量使用 **RMSNorm**。RMSNorm 不计算均值，而是只利用均方根对特征进行缩放，其核心形式为：

$$
y_i=
\frac{x_i}
{\sqrt{\frac{1}{d}\sum_{j=1}^{d}x_j^2+\epsilon}}
\cdot\gamma_i
$$

因此，在阅读具体模型的 checkpoint 时，**不能仅根据参数名 `xxx.weight` 就假定它一定是标准的 LayerNorm**，还需要结合模型实现判断它究竟是：

* 标准 LayerNorm；
* Bias-free LayerNorm；
* RMSNorm；
* 或者模型作者自定义的 normalization。

例如，如果 checkpoint 中只有：

```text
input_layernorm.weight [7168]
```

而没有：

```text
input_layernorm.bias [7168]
```

那么这里至少可以确定它不是带有 \(\gamma+\beta\) 两组参数的标准 LayerNorm，而更可能是 scale-only 的 normalization 实现。


### RMSNorm

RMSNorm 是 LayerNorm 的一个简化变体。它认为 LayerNorm 的成功主要来自于其重新缩放不变性，而减去均值的操作不是必须的。因此，它只对输入的均方根进行归一化。

\[y = \frac{x}{\sqrt{\underbrace{\frac{1}{n} \sum_{i=1}^n x_i^2}_{\text{均方值}} + \epsilon}} \cdot \gamma\]

- \(x\)：输入张量，表示模型中的某一层的输出或激活。
- \(x_i\)：\(x\) 中的第 \(i\) 个元素。
- \(n\)：\(x\) 的元素数量（即维度大小）。
- \(\sum_{i=1}^n x_i^2\)：对 \(x\) 中所有元素的平方求和。
- \(\frac{1}{n} \sum_{i=1}^n x_i^2\)：计算 \(x\) 中所有元素平方后的平均值，这被称为均方值。
- \(\epsilon\)：一个小的正数（通常设置为 \(10^{-5}\) 或 \(10^{-6}\)），用于防止分母为零的情况，确保数值稳定性。
- \(\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2 + \epsilon}\)：计算均方根（Root Mean Square, RMS），这是归一化因子。
- \(\gamma\)：缩放因子，是学习得到的参数，允许模型调整归一化后数据的尺度。

通过这种方式，RMSNorm 对输入张量 \(x\) 进行了归一化处理，而不需要像 LayerNorm 那样先减去均值，从而简化了运算过程并减少了需要训练的参数数量（因为它没有偏置项 \(\beta\)）。这种设计使得 RMSNorm 在某些情况下比 LayerNorm 更高效。


## 归一语义

很好，我们把三种“归一”的**数学约束条件**直接写出来，对比它们到底在“归一”什么。

我只写最核心的“归一结果满足什么条件”。

---

### 概率归一

Softmax function

定义：

\[
y_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
\]

它满足两个数学约束：

\[
\sum_i y_i = 1
\]

\[
y_i > 0
\]

**语义**

* 把任意实数向量映射到 **概率单纯形**
* 归一对象：**L1 范数（总和）**
* 作用空间：概率空间

👉 它控制的是“总质量”

### 方差归一

Layer Normalization

定义（忽略 γ、β）：

\[
y_i = \frac{x_i - \mu}{\sigma}
\]

其中：

\[
\mu = \frac{1}{d}\sum_i x_i
\]

\[
\sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2
\]

它满足：

\[
\frac{1}{d}\sum_i y_i = 0
\]

\[
\frac{1}{d}\sum_i y_i^2 = 1
\]

也就是说：

\[
\mathrm{Var}(y) = 1
\]

**语义**

* 均值归零
* 方差归一
* 控制分布形状

👉 它控制的是“中心 + 波动幅度”

### 二阶矩归一

RMSNorm

定义（忽略 γ）：

\[
y_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_j x_j^2}}
\]

它满足：

\[
\frac{1}{d}\sum_i y_i^2 = 1
\]

注意：

\[
\frac{1}{d}\sum_i y_i \neq 0
\]

**语义**

* 不减均值
* 只控制平均平方值
* 控制“能量规模”

👉 它控制的是“二阶矩”

---

**三者统一对比**

| 方法        | 归一对象    | 数学约束            | 几何意义      |
| --------- | ------- | --------------- | --------- |
| Softmax   | L1 和    | \( \sum y_i =1 \) | 投影到概率单纯形  |
| LayerNorm | 均值 + 方差 | 均值=0，Var=1      | 投影到零均值单位球 |
| RMSNorm   | 二阶矩     | \( E[y^2]=1 \)    | 投影到固定半径球面 |

#### 统一表达

可以写成：

\[
\text{Normalize}(x) = \frac{x}{|x|_p}
\]

不同方法只是：

* Softmax：本质类似 L1 归一（但带指数）
* RMSNorm：L2 归一
* LayerNorm：中心化后 L2 归一

---

Softmax：

> 归一“概率结构”

LayerNorm：

> 归一“统计分布”

RMSNorm：

> 归一“能量尺度”

---

* Softmax 让向量“像概率”
* LayerNorm 让向量“像标准正态”
* RMSNorm 让向量“长度固定”


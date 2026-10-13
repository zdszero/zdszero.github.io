% 线性代数基础

### 引言

神经网络中单个计算单元的激活值通常通过边权重向量 \(\mathbf{w}\) 与输入向量 \(\mathbf{x}\) 的点积加上标量偏置 \(b\) 来计算：

\[z(x) = \sum_{i=1}^{n} w_i x_i + b = \textbf{w} \cdot \textbf{x} + b\]

函数 \(z(x)\) 称为该单元的仿射函数，其后接一个修正线性单元（ReLU），该单元会将负值截断为零。

**训练** 该神经网络意味着选择权重 \(\mathbf{w}\) 和偏置 \(b\)，使得对于所有 \(N\) 个输入 \(\mathbf{x}\)，我们都能获得期望的输出。

为此，我们最小化损失函数，该函数比较网络对所有输入向量 \(\mathbf{x}\) 的最终 \(\text{activation}(\mathbf{x})\) 与目标值 \(\text{target}(\mathbf{x})\)。

为了最小化损失，我们使用梯度下降的某种变体，例如普通的随机梯度下降（SGD）。所有这些方法都需要计算 \(\text{activation}(\mathbf{x})\) 关于模型参数 \(\mathbf{w}\) 和 \(b\) 的偏导数。
我们的目标是逐步调整 \(\mathbf{w}\) 和 \(b\)，使得对于所有输入 \(\mathbf{x}\)，总体损失函数不断减小。

#### 向量规范

##### 列向量写法

列向量写法一般在数学课本上使用

设

* \(q_i, k_j \in \mathbb{R}^{d_k \times 1}\) （列向量），
* \(v_j \in \mathbb{R}^{d_v \times 1}\) （列向量）。

那么打分：

\[
s_{ij} = \frac{q_i^\top k_j}{\sqrt{d_k}} \quad (\text{标量})
\]

Softmax 权重：

\[
\alpha_{ij} = \frac{\exp(s_{ij})}{\sum_{t=1}^n \exp(s_{it})}
\]

输出：

\[
o_i = \sum_{j=1}^n \alpha_{ij} v_j \quad \in \mathbb{R}^{d_v \times 1}
\]

矩阵式：

\[
O = \operatorname{Softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

其中

* \(Q = [q_1^\top; q_2^\top; \dots; q_n^\top] \in \mathbb{R}^{n \times d_k}\)，
* \(K = [k_1^\top; k_2^\top; \dots; k_n^\top] \in \mathbb{R}^{n \times d_k}\)，
* \(V = [v_1^\top; v_2^\top; \dots; v_n^\top] \in \mathbb{R}^{n \times d_v}\)。

#####  行向量写法

行向量写法在工程上更常见

设

* \(q_i, k_j \in \mathbb{R}^{1 \times d_k}\) （行向量），
* \(v_j \in \mathbb{R}^{1 \times d_v}\) （行向量）。

那么打分：

\[
s_{ij} = \frac{q_i k_j^\top}{\sqrt{d_k}} \quad (\text{标量})
\]

Softmax 权重：

\[
\alpha_{ij} = \frac{\exp(s_{ij})}{\sum_{t=1}^n \exp(s_{it})}
\]

输出：

\[
o_i = \sum_{j=1}^n \alpha_{ij} v_j \quad \in \mathbb{R}^{1 \times d_v}
\]

矩阵式同样成立：

\[
O = \operatorname{Softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

只是这里的

* \(Q = \begin{bmatrix} q_1 \\ q_2 \\ \vdots \\ q_n \end{bmatrix} \in \mathbb{R}^{n \times d_k}\)，
  每一行是一个 query（行向量）。

* **列向量写法**更贴近数学教科书，点积写作 \(q_i^\top k_j\)。
* **行向量写法**更贴近实际工程框架（NumPy/PyTorch），点积写作 \(q_i k_j^\top\)。
* 矩阵公式 \(O = \text{Softmax}(QK^\top / \sqrt{d_k})V\) 在两种习惯下都完全一致。


### 矩阵微积分

#### 偏导数

导数与偏导数的区别：

- 普通导数用于单变量函数。
- 偏导数用于多变量函数。当我们试图求某个变量的偏导数时，其他变量被视为常数。

#### 梯度

当我们计算多个变量的偏导数时，与其让它们零散地分布且无组织，不如将它们组织成一个水平向量：

\[\nabla f(x, y) = [ \frac{\partial f(x, y)}{\partial x}, \frac{\partial f(x, y)}{\partial y}]\]

因此，\(f(x, y)\) 的梯度就是其偏导数构成的向量。

如果我们有两个函数，则得到**雅可比矩阵**，其中梯度作为行：

\[
J = \begin{bmatrix}
\nabla f(x, y) \\
\nabla g(x, y)
\end{bmatrix} = \begin{bmatrix}
\frac{\partial f(x, y)}{\partial x} & \frac{\partial f(x, y)}{\partial y} \\
\frac{\partial g(x, y)}{\partial x} & \frac{\partial g(x, y)}{\partial y}
\end{bmatrix}
\]

注意，表示这个雅可比矩阵的方式有很多种。我们使用的是所谓的分子布局，但许多论文会使用分母布局，它只是分子布局的转置。

#### 雅可比矩阵的一般定义

为了更一般地定义雅可比矩阵，让我们将多个参数组合成一个向量参数：\(f(x, y, z) \rightarrow f(\mathbf{x})\)

让我们明确一下格式：

- \(\mathbf{x}\) 与 \(\vec{x}\) 相同，是一个向量。
- \(x\) 是一个标量。
- \(x_i\) 是向量中的一个元素。

我们还必须定义向量 \(\mathbf{x}\) 的方向：

\[x = \begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{bmatrix}\]

对于多个标量值函数，我们可以像处理参数那样将它们全部组合成一个向量。

令 \(\mathbf{y} = \mathbf{f(x)}\) 是一个由 \(m\) 个标量值函数组成的向量，每个函数都接受一个向量 \(\mathbf{x}\)。并且 \(\mathbf{f}\) 中的每个 \(f_i\) 返回一个标量值。

\[y_1 = f_1(\mathbf{x})\]
\[y_2 = f_2(\mathbf{x})\]
\[\vdots\]
\[y_m = f_m(\mathbf{x})\]

一般来说，雅可比矩阵是所有 \(m \times n\) 个可能的偏导数的集合，即关于 \(\mathbf{x}\) 的 \(m\) 个梯度的堆叠。

\[
\frac{\partial y}{\partial x} =
\begin{bmatrix}
\nabla f_1(\mathbf{x}) \\
\nabla f_2(\mathbf{x}) \\
\vdots \\
\nabla f_m(\mathbf{x}) \\
\end{bmatrix} =
\begin{bmatrix}
\frac{\partial f_1(\mathbf{x})}{\partial \mathbf{x}} \\
\frac{\partial f_2(\mathbf{x})}{\partial \mathbf{x}} \\
\vdots \\
\frac{\partial f_m(\mathbf{x})}{\partial \mathbf{x}} \\
\end{bmatrix} =
\begin{bmatrix}
\frac{\partial f_1(\mathbf{x})}{\partial x_1} & \frac{\partial f_1(\mathbf{x})}{\partial x_2} & \cdots & \frac{\partial f_1(\mathbf{x})}{\partial x_n} \\
\frac{\partial f_2(\mathbf{x})}{\partial x_1} & \frac{\partial f_2(\mathbf{x})}{\partial x_2} & \cdots & \frac{\partial f_2(\mathbf{x})}{\partial x_n} \\
\vdots \\
\frac{\partial f_m(\mathbf{x})}{\partial x_1} & \frac{\partial f_m(\mathbf{x})}{\partial x_2} & \cdots & \frac{\partial f_m(\mathbf{x})}{\partial x_n} \\
\end{bmatrix}
\]

雅可比函数 \(\mathbf{f(x)} = \mathbf{x}\)，其中 \(f_i(\mathbf{x}) = x_i\)：

\[
\frac{\partial y}{\partial x} =
\begin{bmatrix}
1 & 0 & \cdots & 0 \\
0 & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & 1
\end{bmatrix}
\]

#### 逐元素二元运算符的导数

所谓“逐元素二元运算”，我们指的是对每个向量的第一个元素应用运算符得到输出的第一个元素，然后对输入的第二个元素应用运算符得到输出的第二个元素，依此类推。

我们可以用符号 \(\mathbf{y} = \mathbf{f(w)} \bigcirc \mathbf{g(x)}\) 来概括逐元素二元运算，其中 \(m = n = |y| = |w| = |x|\)

\[
\begin{bmatrix}
y_1 \\
y_2 \\
\vdots \\
y_n
\end{bmatrix} = \begin{bmatrix}
f_1(\mathbf{w}) \bigcirc g_1(\mathbf{x}) \\
f_2(\mathbf{w}) \bigcirc g_2(\mathbf{x}) \\
\vdots \\
f_n(\mathbf{w}) \bigcirc g_n(\mathbf{x}) \\
\end{bmatrix}
\]

\[
\mathbf{J_{W}} = \frac{\partial \mathbf{y}}{\partial \mathbf{w}} = \begin{bmatrix}
\frac{\partial }{\partial w_1}(f_1 (\mathbf{w}) \bigcirc g_1(\mathbf{x})) & \frac{\partial }{\partial w_2}(f_1 (\mathbf{w}) \bigcirc g_1(\mathbf{x})) & \cdots & \frac{\partial }{\partial w_n}(f_1 (\mathbf{w}) \bigcirc g_1(\mathbf{x})) \\
\frac{\partial }{\partial w_1}(f_2 (\mathbf{w}) \bigcirc g_2(\mathbf{x})) & \frac{\partial }{\partial w_2}(f_2 (\mathbf{w}) \bigcirc g_2(\mathbf{x})) & \cdots & \frac{\partial }{\partial w_n}(f_2 (\mathbf{w}) \bigcirc g_2(\mathbf{x})) \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial }{\partial w_1}(f_n (\mathbf{w}) \bigcirc g_n(\mathbf{x})) & \frac{\partial }{\partial w_2}(f_n (\mathbf{w}) \bigcirc g_n(\mathbf{x})) & \cdots & \frac{\partial }{\partial w_n}(f_n (\mathbf{w}) \bigcirc g_n(\mathbf{x})) \\
\end{bmatrix}
\]

考虑到当 \(j \ne i\) 时，\(\frac{\partial}{\partial w_j}(f_i(\mathbf{w}) \bigcirc g_i(\mathbf{x})) = 0\)，所以

\[
\frac{\partial \mathbf{y}}{\partial \mathbf{w}} = diag \big(\frac{\partial}{\partial w_1}(f_1(w_1) \bigcirc g_1(x_1)), \frac{\partial}{\partial w_2}(f_2(w_2) \bigcirc g_2(x_2)), \cdots, \frac{\partial}{\partial w_n}(f_n(w_n) \bigcirc g_n(x_n)) \big)
\]

对于 \(\mathbf{x}\)，我们可以得到类似的结果：

\[
\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = diag \big(\frac{\partial}{\partial x_1}(f_1(w_1) \bigcirc g_1(x_1)), \frac{\partial}{\partial x_2}(f_2(w_2) \bigcirc g_2(x_2)), \cdots, \frac{\partial}{\partial x_n}(f_n(w_n) \bigcirc g_n(x_n)) \big)
\]

#### 向量链式法则

单变量链式法则：

\[\frac{d}{dx} f(g(x)) = \frac{df}{dg} \frac{dg}{dx}\]

多变量链式法则：

\[
\frac{\partial}{\partial x} {\mathbf{f}(\mathbf{g}(\mathbf{x}))} =
\begin{bmatrix}
\frac{\partial f_1}{\partial g_1} & \frac{\partial f_1}{\partial g_2} & \cdots & \frac{\partial f_1}{\partial g_k} \\
\frac{\partial f_2}{\partial g_1} & \frac{\partial f_2}{\partial g_2} & \cdots & \frac{\partial f_2}{\partial g_k} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial g_1} & \frac{\partial f_m}{\partial g_2} & \cdots & \frac{\partial f_m}{\partial g_k} \\
\end{bmatrix}
\begin{bmatrix}
\frac{\partial g_1}{\partial x_1} & \frac{\partial g_1}{\partial x_2} & \cdots & \frac{\partial g_1}{\partial x_n} \\
\frac{\partial g_2}{\partial x_1} & \frac{\partial g_2}{\partial x_2} & \cdots & \frac{\partial g_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial g_k}{\partial x_1} & \frac{\partial g_k}{\partial x_2} & \cdots & \frac{\partial g_k}{\partial x_n} \\
\end{bmatrix}
\]

其中 \(m = |f|\)，\(n = |x|\)，\(k = |g|\)。
得到的雅可比矩阵是 \(m \times n\)（一个 \(m \times k\) 矩阵乘以 \(k \times n\) 矩阵）。

有时，对向量 \(\mathbf{w}\) 和 \(\mathbf{x}\) 的逐元素运算会产生对角矩阵，前面的等式可以简化为：

\[\frac{\partial \mathbf{f}}{\partial \mathbf{g}} = diag( \frac{\partial f_i}{\partial g_i} )\]

\[\frac{\partial \mathbf{g}}{\partial \mathbf{x}} = diag( \frac{\partial g_i}{\partial x_i} )\]

\[\frac{\partial}{\partial x} {\mathbf{f}(\mathbf{g}(\mathbf{x}))} = diag( \frac{\partial f_i}{\partial g_i} \frac{\partial g_i}{\partial x_i} )\]

#### 矩阵扩展运算

- 点积：\(A \cdot B\)，矩阵乘法
- 克罗内克积：\(A \otimes B\)，逐元素乘法

### 神经元激活的梯度

\(X = [x_1, x_2, \cdots, x_{N}]^{T}\)

\(y = [target(x_1), target(x_2), \cdots, target(x_N)]^T\)

其中 \(y_i\) 是标量，则成本函数变为

\(C(w, b, X, y) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \sigma(x_i))^2 = \frac{1}{N} \sum_{i=1}^{N} (y_i - max(0, w \cdot x_i + b))^2\)

### 训练

大语言模型的训练本质上是**最大似然估计**。其目标是找到一组模型参数 \(\theta\)，使得在给定输入 \(x\) 的条件下，观测到真实输出 \(y\) 的概率最大。

其数学形式为：

\[
\theta^* = \arg\max_\theta \prod_{i=1}^N P_\theta(y_i | x_i)
\]

为便于优化，通常将其转化为最小化负对数似然：

\[
\theta^* = \arg\min_\theta -\sum_{i=1}^N \log P_\theta(y_i | x_i)
\]

这个目标函数就是**交叉熵损失**。因此，大模型训练的核心是建模并优化条件概率 \(P(y|x)\)。

对于自回归语言模型，其具体任务是**预测下一个词元的概率**：

\[
P(x_t | x_1, x_2, ..., x_{t-1})
\]

#### Softmax 与交叉熵梯度

Softmax 函数将模型输出的 logits 向量 \(\mathbf{z}\) 转换为概率分布：

\[
P_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
\]

在训练中，Softmax 与交叉熵损失结合使用。其梯度形式简洁且至关重要：

\[
\frac{\partial L}{\partial z_i} = P_i - y_i
\]

其中，\(y_i\) 是目标词元的 one-hot 编码。**这个梯度公式是 Transformer 模型训练中最核心的反向传播信号。**

####  KL 散度

KL 散度衡量两个概率分布 \(P\) 和 \(Q\) 之间的差异：

\[
D_{KL}(P || Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}
\]

它在高级训练技术中广泛应用：
*   **知识蒸馏**：让学生模型分布逼近教师模型。
*   **RLHF / PPO**：在奖励最大化中约束策略模型，防止其偏离原始参考模型 \(\pi_{ref}\)，例如：
    \[
    L = L_{policy} + \beta \cdot D_{KL}(\pi || \pi_{ref})
    \]
*   **DPO**：直接基于偏好数据优化策略。

#### 熵

熵衡量一个概率分布 \(P\) 的不确定性或随机性：

\[
H(P) = -\sum_x P(x) \log P(x)
\]

在强化学习和采样策略中，熵项常用于鼓励探索，防止策略过早收敛到单一模式。

### 优化理论

训练过程在数学上是一个**最小化损失函数**的优化问题。

#### 梯度下降

基础的参数更新方式：
\[
\theta_{t} = \theta_{t-1} - \eta \cdot \nabla_\theta L(\theta)
\]
其中 \(\eta\) 是学习率。

#### 随机梯度下降

在实际训练中，使用小批量数据 \(B\) 来近似计算梯度，以提升效率：
\[
\nabla_\theta L(\theta) \approx \frac{1}{|B|} \sum_{i \in B} \nabla_\theta L_i(\theta)
\]

#### Adam / AdamW 优化器

这是当前训练大模型最主流的优化算法。它引入了动量和自适应学习率。

**更新步骤：**
1.  计算梯度的一阶矩估计（动量）和二阶矩估计：
    \[
    m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
    \]
    \[
    v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
    \]
2.  进行偏差校正：
    \[
    \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
    \]
3.  更新参数：
    \[
    \theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
    \]

**AdamW** 是 Adam 的一个变体，它将权重衰减与梯度更新解耦，通常能获得更好的泛化性能，**几乎所有现代大语言模型都使用 AdamW 进行训练**。


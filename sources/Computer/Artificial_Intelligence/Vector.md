% Vector

### 列向量写法

列向量写法一般在数学课本上使用

设

* $q_i, k_j \in \mathbb{R}^{d_k \times 1}$ （列向量），
* $v_j \in \mathbb{R}^{d_v \times 1}$ （列向量）。

那么打分：

$$
s_{ij} = \frac{q_i^\top k_j}{\sqrt{d_k}} \quad (\text{标量})
$$

Softmax 权重：

$$
\alpha_{ij} = \frac{\exp(s_{ij})}{\sum_{t=1}^n \exp(s_{it})}
$$

输出：

$$
o_i = \sum_{j=1}^n \alpha_{ij} v_j \quad \in \mathbb{R}^{d_v \times 1}
$$

矩阵式：

$$
O = \operatorname{Softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

其中

* $Q = [q_1^\top; q_2^\top; \dots; q_n^\top] \in \mathbb{R}^{n \times d_k}$，
* $K = [k_1^\top; k_2^\top; \dots; k_n^\top] \in \mathbb{R}^{n \times d_k}$，
* $V = [v_1^\top; v_2^\top; \dots; v_n^\top] \in \mathbb{R}^{n \times d_v}$。

###  行向量写法

行向量写法在工程上更常见

设

* $q_i, k_j \in \mathbb{R}^{1 \times d_k}$ （行向量），
* $v_j \in \mathbb{R}^{1 \times d_v}$ （行向量）。

那么打分：

$$
s_{ij} = \frac{q_i k_j^\top}{\sqrt{d_k}} \quad (\text{标量})
$$

Softmax 权重：

$$
\alpha_{ij} = \frac{\exp(s_{ij})}{\sum_{t=1}^n \exp(s_{it})}
$$

输出：

$$
o_i = \sum_{j=1}^n \alpha_{ij} v_j \quad \in \mathbb{R}^{1 \times d_v}
$$

矩阵式同样成立：

$$
O = \operatorname{Softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

只是这里的

* $Q = \begin{bmatrix} q_1 \\ q_2 \\ \vdots \\ q_n \end{bmatrix} \in \mathbb{R}^{n \times d_k}$，
  每一行是一个 query（行向量）。


* **列向量写法**更贴近数学教科书，点积写作 $q_i^\top k_j$。
* **行向量写法**更贴近实际工程框架（NumPy/PyTorch），点积写作 $q_i k_j^\top$。
* 矩阵公式 $O = \text{Softmax}(QK^\top / \sqrt{d_k})V$ 在两种习惯下都完全一致。

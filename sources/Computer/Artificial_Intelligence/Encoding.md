% Encoding

### 绝对位置编码

对于位置编码，常规的做法是在计算 query、key 和 value 向量之前，会计算一个位置编码向量 $p_i$ 加到词嵌入 $x_i$ 上，位置编码向量 $p_i$ 同样也是 $d$ 维向量，然后再乘以对应的变换矩阵 $W$：

$$
f_{t; t \in \{q,k,v\}}(x_i, i) := W_{t; t \in \{q,k,v\}} (x_i + p_i)
$$

而经典的位置编码向量 $p_i$ 的计算方式是使用 Sinusoidal 函数：

$$
p_{i,2t} = \sin \left( \frac{k}{10000^{2t/d}} \right) \\
p_{i,2t+1} = \cos \left( \frac{k}{10000^{2t/d}} \right)
$$

其中，$p_{i,2t}$ 表示位置 $i$ 维度向量 $p_i$ 中的第 $2t$ 位置分量，也就是偶数索引位置的计算公式，而 $p_{i,2t+1}$ 就对应第 $2t+1$ 位置分量，也就是奇数索引位置的计算公式。


### 旋转位置编码

旋转位置编码的核心就是用某种方式直接在 $\mathbf{q}_{t} \mathbf{k}_{j}^{T}$ 计算的过程中直接插入两者的位置信息。

所以需要找到一个公式满足以下条件：

$f(\mathbf{q}, t) f(\mathbf{k}^\top, j) = f(\mathbf{q} \mathbf{k}^\top, t-j)$

Rope 就是找了这样一个基于旋转的公式：

$$
\text{RoPE}(x_m)[2k-1:2k] =
\begin{pmatrix}
x_{2k-1}\cos(m\theta_k) - x_{2k}\sin(m\theta_k) \\
x_{2k-1}\sin(m\theta_k) + x_{2k}\cos(m\theta_k)
\end{pmatrix}
$$


假设 $R_a$ 表示角度为 $a$ 的旋转矩阵，那么 $R$ 具有如下性质：  

1. $R_a^T = R(-a)$  
2. $R_a R_b = R(a+b)$  

回到旋转位置编码，我们可以去证明  

$$
\langle R_a X, R_b Y \rangle = \langle X, R(b-a) Y \rangle
$$  

证明如下：  

$$
\begin{aligned}
\langle R_a X, R_b Y \rangle  
&= (R_a X)^T R_b Y \\  
&= X^T R_a^T R_b Y \\  
&= X^T R(-a) R_b Y \\  
&= X^T R(b-a) Y \\  
&= \langle X, R(b-a) Y \rangle  
\end{aligned}
$$

---

__那么使用 RoPE 的 attention 公式呢？__


使用 Rope 后，自注意力计算公式在形式上保持不变，但 $Q$ 和 $K$ 的计算方式发生了变化：

$$\text{Attention}(Q_{R}, K_{R}, V) = \text{softmax}\left(\frac{Q_{R} K_{R}^T}{\sqrt{d_k}}\right) V$$

其中：

* $Q_{R}$ 和 $K_{R}$ 分别是通过旋转变换后的 $Q$ 和 $K$ 向量。

具体来说，对于序列中的第 $m$ 个 token 和第 $n$ 个 token，其对应的 $Q$ 和 $K$ 向量分别是 $q_m$ 和 $k_n$。经过 Rope 旋转变换后，它们变为 $q'_m$ 和 $k'_n$。

$q'_m$ 和 $k'_n$ 的点积为：

$$(q'_m)^T k'_n = \left(\mathbf{R}_{\Theta, m} q_m\right)^T \left(\mathbf{R}_{\Theta, n} k_n\right) = q_m^T \mathbf{R}_{\Theta, m}^T \mathbf{R}_{\Theta, n} k_n = q_m^T \mathbf{R}_{\Theta, n-m} k_n$$

其中：

* $\mathbf{R}_{\Theta, m}$ 和 $\mathbf{R}_{\Theta, n}$ 是旋转矩阵，它们将位置信息 $m$ 和 $n$ 编码到向量中。
* $\mathbf{R}_{\Theta, m}^T \mathbf{R}_{\Theta, n} = \mathbf{R}_{\Theta, n-m}$ 是一个只依赖于**相对位置**差 $n-m$ 的旋转矩阵。

这个公式表明，Rope 将自注意力机制中的 **点积** 与 **相对位置** 关联了起来。

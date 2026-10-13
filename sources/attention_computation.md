% Attention 计算量

Attention 阶段核心包含以下几个数学公式：

- \(Q = X W_{Q}\)
- \(K = X W_{K}\)
- \(V = X W_{V}\)
- \(Y = \text{Attention}(Q, K, V) = \text{Softmax}(\frac{Q K^{T}}{\sqrt{d_k}})V\)
- \(Z = YW_{O}\)
- \(\text{Output} = \text{LayerNorm}(X+Z)\)

Attention 计算可以分为三大部分：

- 线性投影/矩阵乘法（linear projection/gemm）：\(X\) 和 \(W_{Q}, W_{K}, W_{V}, W_{O}\) 相乘
- 注意力得分计算（attention score）
- 其他运算：layernorm

### QKV

其中 __gemm__ 的计算量和访存量如下表所示：

| operation | inference FLOPs | params | output shape |
| :-: | :-: | :-: | :-: |
| \(A[B,T,\textcolor{red}{D}] \cdot W_Q[\textcolor{red}{D},N,H]\) | \(2BTDNH\) | \(DNH\) | \(Q[B,T,D,H]\) |
| \(A[B,T,\textcolor{red}{D}] \cdot W_K[\textcolor{red}{D},K,H]\) | \(2BTDKH\) | \(DKH\) | \(K[B,T,K,H]\) |
| \(A[B,T,\textcolor{red}{D}] \cdot W_V[\textcolor{red}{D},K,H]\) | \(2BTDKH\) | \(DKH\) | \(V[B,T,K,H]\) |
| \(A[B,T,\textcolor{red}{N,H}] \cdot W_O[\textcolor{red}{N,H},D]\) | \(2BTDNH\) | \(DNH\) | \(\text{Z}[B,T,D]\)|

### Attention Score

其中 __attention score__ 的计算量如下表所示：

| operation | inference FLOPs | output shape |
|-----------|-----------------|--|
| \(Q[\textcolor{blue}{B},T,\textcolor{blue}{K},G,\textcolor{red}{H}] \cdot K[\textcolor{blue}{B},S,\textcolor{blue}{K},\textcolor{red}{H}]\) | \(2BTSKGH=2BTSNH\) | \(\text{score}[B,T,S,K,G]=[B,T,S,N]\) |
| \(\text{softmax}_{S}\ L[B,T,S,K,G]\) | \(O(BTSKG)=O(BTSN)\) | |
| \(S[\textcolor{blue}{B},T,\textcolor{red}{S},\textcolor{blue}{K},G] \cdot V[\textcolor{blue}{B},\textcolor{red}{S},\textcolor{blue}{K},H]\) | \(2BTSKGH=2BTSNH\) | \(Y[B,T,K,G,H]=[B,T,N,H]\) |

根据以上推导，可以得到以下结论：

- Z 的 shape 和输入相同，都是 \([B,T,D]\)，所以可以将两者直接相加，再进行 LayerNorm 得到 Output。
- Self Attention 的计算量取决于 q 和 k/v length。
    - 如果忽略 softmax 的话，总共的计算量为 \(O(BTSNH)\)。

为了方便理解，我们考虑以下 \(B=1\)、\(N=1\) 的情况，计算过程为如下公式：

\[
\begin{aligned}
Y&=\underbrace{\begin{bmatrix}
\alpha_{11} & \alpha_{12} & \cdots & \alpha_{1s}\\
\alpha_{21} & \alpha_{22} & \cdots & \alpha_{2s}\\
\vdots & \vdots & \ddots & \vdots \\
\alpha_{t1} & \alpha_{t2} & \cdots & \alpha_{ts}
\end{bmatrix}}_{\displaystyle \alpha=\operatorname{Softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)}
\begin{bmatrix}
v_1\\ v_2\\ \vdots\\ v_s
\end{bmatrix}
\\[4pt]
&=\begin{bmatrix}
\alpha_{11}v_1+\alpha_{12}v_2+\cdots+\alpha_{1s}v_s\\
\alpha_{21}v_1+\alpha_{22}v_2+\cdots+\alpha_{2s}v_s\\
\vdots\\
\alpha_{t1}v_1+\alpha_{t2}v_2+\cdots+\alpha_{ts}v_s
\end{bmatrix}.
\end{aligned}
\]

其中 \(\alpha_{ij}\) 是当前 token 和先前每一个 token 的注意力得分，通过以下方式计算出来，注意 Softmax 是按照行作用的：

\[
s_{ij}=\frac{q_i\cdot k_j}{\sqrt{d_k}},\qquad
\alpha_{ij}=\frac{e^{s_{ij}}}{\sum_{t=1}^{n} e^{s_{it}}}
\]

\[
\begin{align*}
QK^{T} &= \begin{bmatrix}
q_1 \\ q_2 \\ \vdots \\ q_t
\end{bmatrix}
\begin{bmatrix}
k_1^T & k_2^T & \cdots & k_s^T
\end{bmatrix} \\
&= \begin{bmatrix}
q_1 \cdot k_1^T & q_1 \cdot k_2^T & q_1 \cdot k_3^T & \cdots & q_1 \cdot k_s^T \\
q_2 \cdot k_1^T & q_2 \cdot k_2^T & q_2 \cdot k_3^T & \cdots & q_2 \cdot k_s^T \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
q_t \cdot k_1^T & q_t \cdot k_2^T & q_t \cdot k_3^T & \cdots & q_t \cdot k_s^T
\end{bmatrix} \\
&\xrightarrow{\text{softmax 逐行归一化}}
\begin{bmatrix}
\alpha_{11} & \alpha_{12} & \cdots & \alpha_{1s}\\
\alpha_{21} & \alpha_{22} & \cdots & \alpha_{2s}\\
\vdots & \vdots & \ddots & \vdots \\
\alpha_{t1} & \alpha_{t2} & \cdots & \alpha_{ts}
\end{bmatrix}
\end{align*}
\]

注意力得分矩阵的 shape 为 \([T, S]\)，行长度就是 query length，列长度就是 kv length，每行就是一个 token 和之前 token 的注意力打分，还需要乘上对应的 \(v\) 向量。再乘以 \(V\) 的时候收缩的维度是在行上，所以 contracting dimension 是 \(S\)。


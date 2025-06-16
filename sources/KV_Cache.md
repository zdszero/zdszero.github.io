% KV Cache

__Why no Q Cache?__

$Q = \begin{bmatrix} q_1 \\q_2 \\ \vdots \\ q_n \end{bmatrix}$
$K = \begin{bmatrix} k_1 \\k_2 \\ \vdots \\ k_n \end{bmatrix}$
$V = \begin{bmatrix} v_1 \\v_2 \\ \vdots \\ v_n \end{bmatrix}$

$k_i, q_i \in R^{1 \times d_k}, v_{i} \in R^{1 \times d_v}$

$Q \cdot K^{T} =
\begin{bmatrix} q_1 \\q_2 \\ \vdots \\ q_n \end{bmatrix}
\begin{bmatrix} k_1^{T} & k_2^{T} & \cdots & k_n^{T} \end{bmatrix} =
\begin{bmatrix}
q_1 \cdot k_1^{T} & q_1 \cdot k_2^{T} & \cdots & q_1 \cdot k_n^{T} \\
q_2 \cdot k_1^{T} & q_2 \cdot k_2^{T} & \cdots & q_2 \cdot k_n^{T} \\
\vdots & \vdots & \ddots & \vdots \\
q_n \cdot k_1^{T} & q_n \cdot k_2^{T} & \cdots & q_n \cdot k_n^{T}
\end{bmatrix}$

Without casual mask:

$S(Q K^{T})V = \begin{bmatrix}
S(q_1 \cdot k_1^{T}) & S(q_1 \cdot k_2^{T}) & \cdots & S(q_1 \cdot k_n^{T}) \\
S(q_2 \cdot k_1^{T}) & S(q_2 \cdot k_2^{T}) & \cdots & S(q_2 \cdot k_n^{T}) \\
\vdots & \vdots & \ddots & \vdots \\
S(q_n \cdot k_1^{T}) & S(q_n \cdot k_2^{T}) & \cdots & S(q_n \cdot k_n^{T})
\end{bmatrix} \begin{bmatrix}
v1 \\ v2 \\ \vdots \\ v_n
\end{bmatrix} = \begin{bmatrix}
S(q_1 \cdot k_1^{T})v_1 + S(q_1 \cdot k_2^{T})v_2 + \cdots + S(q_1 \cdot k_n^{T})v_n \\
S(q_2 \cdot k_1^{T})v_1 + S(q_2 \cdot k_2^{T})v_2 + \cdots + S(q_2 \cdot k_n^{T})v_n \\
\vdots \\
S(q_n \cdot k_1^{T})v_1 + S(q_n \cdot k_2^{T})v_2 + \cdots + S(q_n \cdot k_n^{T})v_n
\end{bmatrix}$

在没有 Causal Mask 时，计算 t 位置的 Attention 需要未来的 KV，这在实际进行自回归推理时无法得到。

With casual mask:

$S(Q K^{T})V = \begin{bmatrix}
S(q_1 \cdot k_1^{T}) & 0 & \cdots & 0 \\
S(q_2 \cdot k_1^{T}) & S(q_2 \cdot k_2^{T}) & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
S(q_n \cdot k_1^{T}) & S(q_n \cdot k_2^{T}) & \cdots & S(q_n \cdot k_n^{T})
\end{bmatrix} \begin{bmatrix}
v1 \\ v2 \\ \vdots \\ v_n
\end{bmatrix} = \begin{bmatrix}
S(q_1 \cdot k_1^{T})v_1 \\
S(q_2 \cdot k_1^{T})v_1 + S(q_2 \cdot k_2^{T})v_2 \\
\vdots \\
S(q_n \cdot k_1^{T})v_1 + S(q_n \cdot k_2^{T})v_2 + \cdots + S(q_n \cdot k_n^{T})v_n
\end{bmatrix}$

在序列的 t 位置，Q 只有当前位置的 $q_t$ 参与了计算，而 K 和 V 多个位置参与了计算，所以需要 KV Cache，而不需要 Q Cache。

---

How to save flops in code? The actual implementation with the code?

vllm:

- `run_worker("execute_model")`
- `model(input_ids, positions,)`

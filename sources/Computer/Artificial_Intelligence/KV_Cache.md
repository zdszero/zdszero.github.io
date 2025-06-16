% KV Cache

__Why no Q Cache?__

Q, K, V are high dimension vector used to encode each token in vocabulary.

$Q = \begin{bmatrix} q_1 \\q_2 \\ \vdots \\ q_n \end{bmatrix}$
$K = \begin{bmatrix} k_1 \\k_2 \\ \vdots \\ k_n \end{bmatrix}$
$V = \begin{bmatrix} v_1 \\v_2 \\ \vdots \\ v_n \end{bmatrix}$

$k_i, q_i \in R^{1 \times d_k}, v_{i} \in R^{1 \times d_v}$

---

The core of transoformer mechanism is attention, where we calcuate the attention of current token with all its previous tokens.

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

So this matrix is quietly verbose, because the attention computation is incremental and auto-regressive.
Which means everytime we only care about the last token.

So we could optimize this process with causal mask.

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

For auto-regressive process, to calculate the last token's attention:

$\text{attn} = S(q_n \cdot k_1^{T})v_1 + S(q_n \cdot k_2^{T})v_2 + \cdots + S(q_n \cdot k_n^{T})v_n$

---

VLLM

- `execute_model(input_ids, input_positions, kv_caches, input_metadata)`
    - `input_ids`: the last token of each request in a batch
    - `input_positions`: index of input_ids
    - `input_metadata`: other information

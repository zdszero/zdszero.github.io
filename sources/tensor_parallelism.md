% TP

## Gemm

考虑矩阵 \(A B\) 相乘，我们可以考虑三种切分方式：

1. 将 \(B\) 按照列切（列拆分）
2. 将 \(A\) 按照行切（行拆分）
3. 将 \(A\) 按照列切，将 \(B\) 按照行切分（行列拆分）

### Outer Dim

**适用场景**：当权重矩阵 \(B\) 的列数（即输出特征维度）非常大时，适合沿列维度进行拆分。

**拆分方式**：

- 将权重矩阵 \(B\) 沿**列**维度均匀拆分为 \(N\) 块，其中 \(N\) 为并行设备数。
- 输入矩阵 \(A\) **保持完整**地分发到所有设备。

\[AB = A [B_1, B_2, \cdots, B_{N-1}] = [C_1, C_2, \cdots, C_{N-1}] = C\]

矩阵 \(B\) 的每个子矩阵可以分别与矩阵 \(A\) 相乘，在不同的设备上进行计算，计算完之后之后做一个 all gather 进行结果聚合：

\[
\begin{aligned}
B &= [B_0, B_1, \dots, B_{N-1}] \\
&\Downarrow \quad \text{每个设备 } k \text{ 计算} \\
C_k &= A \times B_k \\
&\Downarrow \quad \text{All-Gather 聚合} \\
C &= [C_0, C_1, \dots, C_{N-1}]
\end{aligned}
\]

其中 \(A \in \mathbb{R}^{L \times K}\)， \(B \in \mathbb{R}^{K \times M}\)， \(B_k \in \mathbb{R}^{K \times \frac{M}{N}}\)，\(C_k \in \mathbb{R}^{L \times \frac{M}{N}}\)，\(C \in \mathbb{R}^{L \times M}\)。

### Inner Dim

**适用场景**：当权重矩阵 \(B\) 的行数（即输入特征维度）非常大，或当输入 \(A\) 的批次/序列维度很大时，适合沿行维度进行拆分。

**拆分方式**：
- 将权重矩阵 \(B\) 沿**行**维度均匀拆分为 \(N\) 块。
- 同时，将输入矩阵 \(A\) 沿**列**维度进行对应的拆分，以匹配 \(B\) 的行分割。

用数学公式表示为：
\[
A = [A_0, A_1, ..., A_{N-1}], \quad B = \begin{bmatrix} B_0 \\ B_1 \\ \vdots \\ B_{N-1} \end{bmatrix}
\]
其中，\(A_k\) 的形状为 \((L, \frac{K}{N})\)，\(B_k\) 的形状为 \((\frac{K}{N}, M)\)。

**计算过程**：
每个设备 \(k\) 计算局部结果：
\[
C_k = A_k \times B_k
\]
这里，每个 \(C_k\) 的形状与最终输出 \(C\) 相同，均为 \((L, M)\)，但每个 \(C_k\) 只是基于局部输入和权重计算的**部分和**。

**通信与聚合**：
为了得到最终结果，需要对所有设备的局部结果 \(C_k\) 进行求和。这通过 **All-Reduce**（或更优化的 **Reduce-Scatter**）操作实现。
\[
C = \sum_{k=0}^{N-1} C_k \quad \text{（在每个设备上通过 All-Reduce 后得到）}
\]


## Attention 切分

### QKV

qkv_proj：`MergedColumnParallelLinear([hidden_size, q_dim + k_dim + v_dim])`

进行 TP 切分时，`MergedColumnParallelLinear` 内部处理了 KV 切分的一些细节：

- Q
    - 每个 rank 有 `num_heads / tp_size` 个 q head
- KV
    - 如果 `tp_size <= num_kv_heads`，那么每个 rank 有 `num_kv_heads / tp_size` 个 KV head
    - 如果 `tp_size > num_kv_heads`，那么每个 rank 有 1 个 KV head

```
self.num_heads = self.total_num_heads // attn_tp_size
self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
```

### Attention

上一步 QKV Projection 已经为每个 rank 切分好 head 了，这一步直接喂到 Attention kernel 中就可以了。

```
self.attn = RadixAttention(
    self.num_heads,
    self.head_dim,
    num_kv_heads=self.num_kv_heads,
    v_head_dim=self.v_head_dim,
)
```

% 数学计算符号

论文、代码和笔记里用到的数学符号是高度**重载**的：同一个 \(\odot\) 在不同资料里既可能是逐元素乘法，也可能是别的运算；同一个 \(\otimes\) 至少有三种完全不同的含义；同一个 \(\sigma\) 在门控里是 sigmoid，在 SVD 里却是奇异值。写公式的人不觉得有歧义，读公式的人经常要猜。

本文把 LLM 相关笔记里最常出现的矩阵/张量运算和对应 KaTeX 写法集中列一遍，并给出 PyTorch 中的实际实现，作为读写公式时的对照表。

### 形状与约定

本文统一用方括号写张量形状，与代码里的 shape 保持一致，例如：

- 矩阵 \(A\in\mathbb{R}^{m\times n}\)；
- token 状态 \(h\in\mathbb{R}^{d}\)，一批序列 \(X\in\mathbb{R}^{B\times S\times d}\)；
- 多头注意力中间结果 \([B,H,S,d_v]\)。

约定要点：

- 无特殊说明时，向量一律视为**列向量**，\(x^{\mathsf T}\) 表示其转置（行向量）；
- 下标 \(i,j\) 表示元素位置，\(B,S,H,d\) 这类大写字母表示该维的长度；
- 矩阵乘法既不交换也不逐元素，凡是逐元素运算都会显式写出符号。

### 逐元素运算

逐元素（elementwise / pointwise）运算的特点是：输出第 \((i,j)\) 个元素只依赖输入的第 \((i,j)\) 个元素，不同位置之间没有交互。

#### Hadamard 积

Hadamard 积（也叫 Schur product、逐元素积）写作 \(\odot\)，KaTeX 命令是 `\odot`：

\[
(A\odot B)_{ij}=A_{ij}B_{ij}.
\]

要求 \(A\) 与 \(B\) 形状相同，或满足广播规则。它的性质与矩阵乘法差别很大：

| 性质 | Hadamard 积 \(A\odot B\) | 矩阵乘法 \(AB\) |
| --- | --- | --- |
| 运算方向 | 逐位置 | 行 × 列缩并 |
| 交换律 | \(A\odot B=B\odot A\) | 一般不成立 |
| 结合律 | \((A\odot B)\odot C=A\odot(B\odot C)\) | 成立但结果不同 |
| 对加法分配 | \(A\odot(B+C)=A\odot B+A\odot C\) | 成立 |
| 形状要求 | 相同或可广播 | 中间维度匹配 |
| 计算量 | \(mn\) 次乘法 | \(2mnp\) 次浮点运算 |

注意一个常见错误：

\[
(A\odot B)x\neq (Ax)\odot(Bx).
\]

前者是先逐元素相乘再乘向量，后者是先做两次矩阵乘法再逐元素相乘，两者既不等值也不等形状。另外，对向量有

\[
a\odot b=\operatorname{Diag}(a)\,b,
\]

所以逐元素缩放可以看成乘一个对角矩阵，这也是门控可以写成线性算子的原因。

PyTorch 中对应关系：

```python
a * b                     # torch.mul，逐元素
torch.diag_embed(a) @ b   # 等价的矩阵形式
a[:, None] * b[None, :]   # 广播成外积
```

在 LLM 里 Hadamard 积最常见的三种用法：

- **门控**：\(x\odot\sigma(Wx)\)，例如 SwiGLU 的 MLP、Gated Attention 的输出门控；
- **掩码**：对已经算好的权重做乘法掩码，例如 \(A\odot M\)；
- **逐通道缩放**：\(\gamma\odot\operatorname{Norm}(x)\)，Norm 之后的可学习缩放。

它的反向传播很直接：

\[
\frac{\partial\,(a\odot b)}{\partial a}=\operatorname{Diag}(b),
\qquad
\frac{\partial\,(a\odot b)}{\partial b}=\operatorname{Diag}(a).
\]

从性能角度看，Hadamard 积只做 \(mn\) 次乘法，是典型的 memory-bound 算子：算力需求很低，但需要把两个输入读出、把结果写回，因此在融合 kernel 中常与相邻的归一化、激活或投影合并，避免额外的显存往返。

#### 逐元素除法、幂与函数

同一族的写法还有：

| 运算 | KaTeX | 公式 | PyTorch |
| --- | --- | --- | --- |
| 逐元素除法 | `\oslash` | \((A\oslash B)_{ij}=A_{ij}/B_{ij}\) | `a / b` |
| 逐元素幂 | `^{\odot p}` | \((A^{\odot p})_{ij}=A_{ij}^{p}\) | `a ** p` |
| 逐元素开方 | `\sqrt{A}` | \(\sqrt{A}_{ij}=\sqrt{A_{ij}}\) | `torch.sqrt(a)` |
| 逐元素指数 | `\exp` | \(\exp(A)_{ij}=e^{A_{ij}}\) | `torch.exp(a)` |
| 逐元素对数 | `\log` | \(\log(A)_{ij}=\ln A_{ij}\) | `torch.log(a)` |
| 逐元素取最大 | `\max(A,B)` | \(\max(A,B)_{ij}=\max(A_{ij},B_{ij})\) | `torch.maximum(a, b)` |

这些函数作用在矩阵上时一律理解为逐元素，只有在明确写 \(\operatorname{tr}\)、\(\det\)、\(\log\det\) 这类矩阵函数时才表示矩阵级运算。\(\exp\) 尤其容易混淆：Transformer 的 softmax 里 \(\exp\) 作用在标量分数上，而矩阵指数 \(e^{A}\) 是幂级数 \(\sum_k A^k/k!\)，两者完全不同。

#### 广播

广播让形状不同的张量也能做逐元素运算，规则是从最后一维开始对齐，长度为 1 或缺失的维度被扩展。注意力里最常见的几种写法：

| 写法 | 含义 | 代码 |
| --- | --- | --- |
| \([B,S,H,d]\odot[B,S,H,1]\) | 每个 head 一个标量门控 | `out * gate.unsqueeze(-1)` |
| \([B,S,H,d]\odot[B,S,1,d]\) | 每个位置、每个通道一个门控 | `out * gate.unsqueeze(2)` |
| \([B,S,H,d]\odot[B,S,1,1]\) | 每个 token 一个整体缩放 | `out * scale.unsqueeze(-1)` |
| \([B,S,H,d]\odot[H,d]\) | 与序列长度无关的静态通道缩放 | `out * weight` |
| \([B,S,H,S]+[1,1,1,S]\) | 广播出完整的 mask 矩阵 | `scores + mask` |

广播能把公式写得很短，但也会掩盖 shape bug：例如把 \([B,S,H,d]\) 与 \([B,S,d]\) 相乘时，广播方向未必是想要的。写笔记和写代码时都建议**显式写出 unsqueeze 的位置**，让被广播的维度一目了然。

### 矩阵乘法、内积与外积

这三种运算都涉及元素之间的缩并或组合，但结果类型完全不同：

| 运算 | KaTeX 写法 | 公式 | 结果 | PyTorch |
| --- | --- | --- | --- | --- |
| 矩阵乘法 | `AB` | \(\sum_k A_{ik}B_{kj}\) | 矩阵 | `a @ b` |
| 内积 | `\langle x,y\rangle`、`x^{\mathsf T}y` | \(\sum_i x_iy_i\) | 标量 | `torch.dot`、`x @ y` |
| 外积 | `xy^{\mathsf T}`、`x\otimes y` | \(x_iy_j\) | 矩阵 | `torch.outer`、`x[:, None] * y[None, :]` |
| 批量缩并 | `\operatorname{einsum}` | 按字母表约定缩并 | 张量 | `torch.einsum` |

一个实用的书写建议：`\cdot` 在文献里有时表示标量乘法、有时表示点积，建议只在标量乘标量时使用，向量内积统一写成 \(\langle x,y\rangle\) 或 \(x^{\mathsf T}y\)。

矩阵乘法按共享维度展开，就是一组外积之和：

\[
AB=\sum_{k=1}^{p}A_{:,k}B_{k,:},
\]

每一项都是 rank-1 的列向量乘行向量，这也是低秩结构、LoRA、线性注意力状态更新里反复出现的写法。

### \(\otimes\) 的三种含义

\(\otimes\) 是歧义最严重的符号，看到它必须先判断上下文：

| 含义 | 数学形式 | 典型场景 | PyTorch |
| --- | --- | --- | --- |
| 外积 / rank-1 张量积 | \((u\otimes v)_{ij}=u_iv_j\) | 线性注意力的状态写入 \(k_t\otimes v_t\) | `torch.outer`、`torch.einsum('i,j->ij')` |
| Kronecker 积 | \((A\otimes B)_{(i-1)p+k,(j-1)q+l}=A_{ij}B_{kl}\) | 把小块矩阵铺成大矩阵，分块结构推导 | `torch.kron` |
| 一般的张量积 | 按维度配对相乘 | 概率论、表示论、部分论文的算子记号 | 无直接对应 |

好消息是：当两个操作数都是**向量**时，外积、Kronecker 积与张量积是同一个东西，所以在 Gated DeltaNet、KDA 这类笔记里写到 \(k_t\otimes v_t\) 时，三种解释一致，不必纠结。但只要出现矩阵，就必须区分是外积还是 Kronecker 积——两者的形状分别是 \(m\times n\) 与 \(mp\times nq\)，差别是数量级的。

### 转置、逆、范数与函数记号

这类符号一般不会引起歧义，但要注意书写规范：

| 记号 | KaTeX | 含义 | PyTorch |
| --- | --- | --- | --- |
| \(A^{\mathsf T}\) | `^{\mathsf T}` 或 `^\top` | 转置 | `a.T`、`a.mT` |
| \(A^{\mathsf H}\) | `^{\mathsf H}` | 共轭转置 | `a.mH` |
| \(A^{-1}\) | `^{-1}` | 逆矩阵 | `torch.linalg.inv` |
| \(A^{+}\) | `^{+}` | 伪逆（Moore-Penrose） | `torch.linalg.pinv` |
| \(\|x\|_2\) | `\|x\|_2` | 欧氏范数 | `torch.linalg.norm` |
| \(\|A\|_F\) | `\|A\|_F` | Frobenius 范数 | `torch.linalg.norm` |
| \(\|A\|_2\) | `\|A\|_2` | 谱范数（最大奇异值） | `torch.linalg.matrix_norm` |
| \(\operatorname{tr}(A)\) | `\operatorname{tr}` | 迹 | `torch.trace` |
| \(\operatorname{diag}(A)\) | `\operatorname{diag}` | 取对角元素，结果是向量 | `torch.diagonal` |
| \(\operatorname{Diag}(x)\) | `\operatorname{Diag}` | 由向量构造对角矩阵 | `torch.diag_embed` |
| \(\operatorname{Concat}(A,B)\) | `\operatorname{Concat}` | 拼接 | `torch.cat` |
| \(\operatorname{softmax}(x)\) | `\operatorname{softmax}` | 归一化到概率单纯形 | `torch.softmax` |

需要区分两件事：

- \(\operatorname{diag}\)（小写）把矩阵压成向量，\(\operatorname{Diag}\)（大写）把向量展成对角矩阵，很多论文不加区分；
- Frobenius 范数可以写成内积形式 \(\|A\|_F^2=\operatorname{tr}(A^{\mathsf T}A)=\sum_{ij}A_{ij}^2\)，这正好把上一节的 Hadamard 积和迹联系起来了：

\[
\operatorname{tr}\!\left(A^{\mathsf T}(B\odot C)\right)=\sum_{ij}A_{ij}B_{ij}C_{ij}.
\]

### 字母重载：\(\sigma\)、\(\alpha\)、\(\gamma\)

读写公式时最容易踩的其实不是符号，而是**同一个字母在不同上下文有不同含义**：

| 字母 | 含义一 | 含义二 | 含义三 |
| --- | --- | --- | --- |
| \(\sigma\) | sigmoid 激活（门控中） | 奇异值（SVD 中） | 谱范数、标准差 |
| \(\alpha\) | Gated DeltaNet / KDA 的衰减门 | attention 权重（部分教材） | 学习率 |
| \(\beta\) | DeltaNet 的写入强度 | 动量系数（优化器） | softmax 温度 |
| \(\gamma\) | Norm 的可学习缩放 | 折扣因子 | 线性注意力的逐通道衰减 |
| \(\lambda\) | 特征值 | 权重衰减系数 | Lagrange 乘子 |
| \(\epsilon\) | 数值稳定小量 | 探索率 | 扰动项 |

因此笔记里第一次出现这类字母时，最好显式写一句“本节 \(\sigma\) 指 sigmoid”，或者在符号旁标注上下标（如 \(\sigma_{\text{gate}}\)、\(\sigma_i(A)\)），避免读者按自己习惯理解。

### 对照速查表

| 运算 | KaTeX | 公式示例 | PyTorch |
| --- | --- | --- | --- |
| 矩阵乘法 | `AB` | \(QK^{\mathsf T}\) | `q @ k.transpose(-1, -2)` |
| Hadamard 积 | `\odot` | \(x\odot\sigma(Wx)\) | `x * torch.sigmoid(wx)` |
| 逐元素除法 | `\oslash` | \(A\oslash B\) | `a / b` |
| 逐元素幂 | `^{\odot 2}` | \(A^{\odot 2}\) | `a ** 2` |
| 外积 | `\otimes`、`^{\mathsf T}` | \(k_t\otimes v_t\) | `torch.outer(k, v)` |
| Kronecker 积 | `\otimes` | \(A\otimes B\) | `torch.kron(a, b)` |
| 内积 | `\langle\cdot,\cdot\rangle` | \(\langle x,y\rangle\) | `torch.dot(x, y)` |
| 转置 | `^{\mathsf T}` | \(A^{\mathsf T}B\) | `a.T @ b` |
| 范数 | `\|\cdot\|_F` | \(\|A\|_F\) | `torch.linalg.norm(a)` |
| 对角 | `\operatorname{Diag}` | \(\operatorname{Diag}(x)\) | `torch.diag_embed(x)` |
| 掩码加法 | `+M` | \(\operatorname{softmax}(S+M)\) | `scores + mask` |
| 掩码乘法 | `\odot M` | \(A\odot M\) | `a * mask` |
| 缩并 | `\sum_k`、`\operatorname{einsum}` | \(\sum_k A_{ik}B_{kj}\) | `torch.einsum('ik,kj->ij', a, b)` |
| 广播 | 无专用符号 | \([B,S,H,d]\odot[B,S,H,1]\) | `out * gate.unsqueeze(-1)` |
| 拼接 | `[A;B]` | \([K,V]\) | `torch.cat([k, v], dim=-1)` |

### 写笔记时的几条约定

1. Hadamard 积统一写 \(\odot\)，不要用 `*` 或 `\times`；\(\circ\) 虽然也常表示 Hadamard 积，但它同时表示函数复合，容易误读；
2. 外积优先写 \(uv^{\mathsf T}\)，只有在强调 rank-1 或 Kronecker 语义时才用 \(\otimes\)，并在首次出现时说明是哪种；
3. 掩码要区分加法和乘法：softmax 之前统一写 \(S+M\)，已经归一化后的权重才写 \(A\odot M\)；
4. 门控写成 \(x\odot\sigma(\cdot)\) 时，明确 \(\sigma\) 是 sigmoid，而不是 SVD 的奇异值；
5. 张量形状统一写 \([B,S,H,d]\)，广播维度用 \(\operatorname{unsqueeze}\) 显式说明；
6. 变量用 KaTeX（如 \(H\)、\(d_v\)），只有代码标识符、路径和 API 用反引号（如 `o_proj`、`g_proj`）。

### 常见面试问题

- Hadamard 积与矩阵乘法的区别是什么？各自的计算量和复杂度如何？
- \(\odot\) 的反向传播为什么是 \(\operatorname{Diag}(b)\)？它和乘一个对角矩阵是什么关系？
- 为什么注意力掩码常用加法而不是乘法？加法掩码配合 \(-\infty\) 可以让 softmax 权重严格为 0，而乘法掩码需要先 softmax 再缩放，会改变权重之和。
- 广播按什么规则进行？把 \([B,S,H,d]\) 与 \([B,S,d]\) 相乘会发生什么？
- \(\otimes\) 有哪几种含义？在 \(k_t\otimes v_t\) 这种写法里为什么不会产生歧义？
- 一句话说清 \(\sigma\) 在 softmax、门控和 SVD 中的三种含义。

### 小结

LLM 笔记里真正高频的矩阵运算只有几类：逐元素的 Hadamard 积与各种逐元素函数、按行乘列的矩阵乘法、内积、外积，以及少量范数和求迹的写法。符号层面的主要风险不是数学难，而是**重载**：\(\odot\) 与 \(\otimes\) 在不同资料里含义不同，\(\sigma\)、\(\alpha\)、\(\gamma\) 在不同章节指代不同对象。把这份对照表固定下来，再在每处首次出现时写明约定，公式的歧义基本就可以消除了。


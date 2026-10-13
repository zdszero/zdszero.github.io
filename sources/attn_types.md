% Types

### Encoder 与 Decoder 到底是什么

Encoder 和 Decoder 最初描述的是两个不同的功能：

- **Encoder** 读取源输入，生成源序列的表示；
- **Decoder** 根据已有目标前缀生成下一个目标 token。

以翻译为例，Encoder 读取源语言句子，Decoder 则逐 token 生成目标语言句子。这里的“编码”和“解码”是模型中的功能角色，不是数据压缩领域中“压缩文件、还原文件”的含义。

#### 从 RNN Seq2Seq 到 Transformer

Encoder–Decoder 范式早于 Transformer：

1. 早期 RNN Seq2Seq 使用一个 RNN 编码源序列，再由另一个 RNN 生成目标序列。最初通常把整个源序列压缩进一个固定长度向量。
2. Attention 机制出现后，Decoder 不再只依赖一个固定向量，而是在每个生成步骤动态读取 Encoder 的各位置状态。今天可以把这种读取关系理解为 cross-attention 的前身。
3. 2017 年的 Transformer 用 self-attention 替代循环结构，但保留了 Encoder–Decoder 的功能分工：Encoder 建模源序列，Decoder 通过 cross-attention 读取源表示并自回归生成目标序列。
4. 后来模型沿这套结构分化出三条主要路线：Encoder-only、Encoder–Decoder 和 Decoder-only。

#### 三种主流架构

| 架构 | 典型模型 | 主要职责 | 常见注意力结构 |
| --- | --- | --- | --- |
| Encoder-only | BERT | 理解、分类、抽取、生成上下文表示 | 双向的 vanilla self-attention |
| Encoder–Decoder | 原始 Transformer、T5、Whisper | 把一条源序列转换成另一条目标序列 | Encoder 使用 self-attention；Decoder 使用 causal self-attention 和 cross-attention |
| Decoder-only | GPT、LLaMA、Qwen | 把所有任务统一为前缀续写 | causal self-attention |

“LLM”并不等于“Decoder-only”。T5 一类 Encoder–Decoder 模型也可以是大语言模型。只是当前通用对话和文本生成模型大多采用 Decoder-only，因此日常讨论中经常把两者近似地联系起来。

#### Transformer 中的 Decoder 有两种语境

“Decoder”一词还存在一个重要的历史语义变化：

- 在原始 Transformer 的 **Encoder–Decoder** 架构中，一个 Decoder 层通常同时包含 causal self-attention、cross-attention 和 FFN；
- 在 **Decoder-only** 模型中，人们仍把每个因果 Transformer block 称为 Decoder block，但其中通常没有 cross-attention，因为根本不存在独立的 Encoder 输出可供读取。

所以，“Decoder-only 中有 Decoder”不意味着它拥有原始 Transformer Decoder 的全部子层。它保留的是自回归生成这一角色和 causal self-attention，而不是完整复制 Encoder–Decoder 中的 Decoder 层。

### 用两个维度区分 Attention

所有常见 Attention 都可以写成同一个读取过程。设 query 序列为 \(X_q\)，key/value 序列为 \(X_{kv}\)：

\[
Q=X_qW_Q,\qquad K=X_{kv}W_K,\qquad V=X_{kv}W_V.
\]

对 query 位置 \(i\)，只在它的可见集合 \(\mathcal V(i)\) 内计算注意力：

\[
a_{ij} =
\frac{\exp\!\left(q_i^{\mathsf T}k_j/\sqrt d\right)}
{\sum_{r\in\mathcal V(i)}\exp\!\left(q_i^{\mathsf T}k_r/\sqrt d\right)},
\qquad
z_i=\sum_{j\in\mathcal V(i)}a_{ij}v_j.
\]

分类时只需要回答两个问题：

1. \(Q\) 与 \(K,V\) 是否来自同一条序列？
2. query 位置 \(i\) 可以读取哪些 key/value 位置？

| 名称 | \(Q\) 与 \(K,V\) 的来源 | 可见范围 | 解决的问题 |
| --- | --- | --- | --- |
| Vanilla self-attention | 同一条序列 | 通常可见整段序列 | 在完整输入内部双向交换信息 |
| Causal self-attention | 同一条序列 | 只能看当前位置及其之前 | 保证自回归生成不泄露未来信息 |
| Cross-attention | 两条不同序列 | 由任务决定，通常可见全部源表示 | 让一条序列读取另一条序列 |

这里最关键的结论是：

> `self` 与 `cross` 区分信息来源，`vanilla` 与 `causal` 区分可见范围。

#### Vanilla self-attention

本文所说的 vanilla self-attention 使用同一条序列产生 \(Q,K,V\)，并允许每个位置读取整段输入：

\[
X_q=X_{kv}=X,
\qquad
\mathcal V(i)=\{1,2,\ldots,T\}.
\]

因此，位置 \(i\) 的表示既可以吸收左侧信息，也可以吸收右侧信息，常被称为 **双向 self-attention** 或 **full self-attention**。

这种结构适合输入在计算开始前已经完整可用的场景。例如 BERT 和标准 Transformer Encoder 可以让源序列的每个位置看到整个源序列。

“Vanilla”不表示没有任何 mask。实际模型仍然可能使用 padding mask 排除补齐位置；这里说的“全可见”只是指没有因果方向限制。

#### Causal self-attention

Causal self-attention 同样从一条序列产生 \(Q,K,V\)，但位置 \(i\) 只能读取前缀：

\[
X_q=X_{kv}=X,
\qquad
\mathcal V(i)=\{1,2,\ldots,i\}.
\]

实现时通常使用因果 mask：

\[
M_{ij}=
\begin{cases}
0, & j\leq i,\\
-\infty, & j>i.
\end{cases}
\]

这样，位置 \(i\) 的表示不会依赖未来 token，模型才能学习自回归分解：

\[
P(x_1,\ldots,x_T) =
\prod_{i=1}^{T}P(x_i\mid x_{<i}).
\]

`causal` 描述的是信息依赖，而不是执行方式。训练时整段序列仍然可以用一张带下三角 mask 的矩阵并行计算；只有推理生成时，下一个 token 必须等待前一个 token 产生。

#### Cross-attention

Cross-attention 的 query 与 key/value 来自不同序列。设 Encoder 输出为 \(H\)，Decoder 当前隐状态为 \(Y\)：

\[
Q=YW_Q,\qquad K=HW_K,\qquad V=HW_V.
\]

这时每个目标位置都用自己的 query 去读取源序列表示。若目标长度为 \(T_y\)，源长度为 \(T_x\)，注意力分数矩阵的形状是 \(T_y\times T_x\)，并不天然是方阵或下三角矩阵。

标准 Encoder–Decoder 中通常存在两种不同的注意力：

1. Decoder 的 **causal self-attention** 读取目标前缀，防止看到未来目标 token；
2. Decoder 的 **cross-attention** 读取 Encoder 输出，让生成过程获得源输入信息。

因此 cross-attention 本身不负责目标序列的因果性。目标因果约束由 Decoder 的 self-attention 保证。

还要注意参数归属：Encoder 产生的是源表示 \(H\)，而把 \(H\) 投影成 \(K,V\) 的 \(W_K,W_V\) 通常属于 Decoder 的 cross-attention 子层。简写成“Encoder 提供 \(K,V\)”时，说的是数据来源，不是参数位于 Encoder 中。

### Attention 类型如何组成三种架构

把前面的概念放回模型结构，就会得到非常清晰的组合关系。

#### Encoder-only

```text
输入序列
   │
   ▼
双向 vanilla self-attention
   │
   ▼
每个位置的上下文表示
```

输入通常在开始计算前已经完整可见，因此每个位置可以同时利用左、右上下文。模型本身不要求按 next-token 方式生成。

#### 标准 Encoder–Decoder

```text
源序列 ──> Encoder 双向 self-attention ──> 源表示 H
                                                │
                                                │ cross-attention
                                                ▼
目标前缀 ──> Decoder causal self-attention ──> 逐 token 生成
```

两条流各司其职：

- Encoder 对完整源序列进行双向建模；
- Decoder 对目标序列保持因果性；
- cross-attention 把源信息传给 Decoder。

#### Decoder-only

```text
[prompt / 源信息] [已经生成的目标 token]
                    │
                    ▼
          同一条 causal self-attention 流
                    │
                    ▼
                下一个 token
```

Decoder-only 不建立独立源流。prompt、指令、示例和已生成 token 全部拼在同一条序列中，通过 causal self-attention 处理。

### Causal Encoder–Decoder 是什么

**Causal Encoder–Decoder，简称 CED**，通常指下面的组合：

- Encoder 使用 causal self-attention，只读取已经到达的源前缀；
- Decoder 使用 causal self-attention，只读取已经生成的目标前缀；
- Decoder 通过 cross-attention 读取 Encoder 已经产生或提交的源表示。

| 结构 | Encoder self-attention | Decoder self-attention | Cross-attention |
| --- | --- | --- | --- |
| 标准 Encoder–Decoder | 双向 | 因果 | 有 |
| Causal Encoder–Decoder | 因果 | 因果 | 有 |
| Decoder-only | 无独立 Encoder | 因果 | 无 |

“Causal Encoder”并不矛盾。`Encoder` 描述它负责表示源输入，`causal` 描述它不能读取尚未到达的未来输入。两者分别回答“做什么”和“能看哪里”。

CED 常见于流式语音识别、实时翻译等在线场景。输入不是一次性全部到达，而是不断产生新帧或新 token。若 Encoder 位置 \(i\) 只能看到源前缀，Decoder 位置 \(u\) 只能看到目标前缀，并且 cross-attention 只能读取已提交到位置 \(c\) 的 Encoder 状态，则：

\[
\mathcal V_{enc}(i)=\{1,\ldots,i\},
\qquad
\mathcal V_{dec}(u)=\{1,\ldots,u\},
\qquad
\mathcal V_{cross}(u)=\{1,\ldots,c\}.
\]

其中前两个集合表达源流和目标流各自的因果性，最后一个集合表达当前有哪些源状态已经可供 Decoder 使用。这使编码和解码可以部分重叠，从而降低端到端延迟。

需要说明的是，CED 并不是像 `Encoder-only`、`Decoder-only` 那样在所有文献中都严格统一的架构名称。不同项目还可能加入 chunk、look-ahead、滑动窗口或状态提交策略。阅读具体实现时，应检查“Encoder 能看到多少未来输入”以及“cross-attention 能读取哪些 Encoder 状态”。

### CED 与 Decoder-only 的本质区别

两者都使用 causal self-attention，但不能因此把它们视为同一种架构。

| 维度 | Causal Encoder–Decoder | Decoder-only |
| --- | --- | --- |
| 序列组织 | 源序列和目标序列是两条流 | prompt 与输出拼成一条流 |
| 模块结构 | 独立 Encoder 和 Decoder | 单个因果 Transformer 栈 |
| 源侧处理 | 由 causal Encoder 专门建模 | 由同一套 causal block 处理 prompt |
| 源到目标的信息通道 | 显式 cross-attention | causal self-attention |
| 参数 | Decoder 层通常有独立 cross-attention 参数 | 通常没有 cross-attention 参数 |
| 长度关系 | 源长度与目标长度分别管理 | 两者共同占用同一个上下文窗口 |
| 运行方式 | 可以让在线编码与解码流水化 | 先处理前缀，再在同一条序列上续写 |
| 常见场景 | 流式 ASR、实时翻译 | 通用文本生成、对话、代码补全 |

#### 两条流与一条流

CED 始终保留源流和目标流的边界：Encoder 产生源表示，Decoder 产生目标 token。即使两侧都使用 causal mask，源 token 也不会直接变成目标序列的一部分。

Decoder-only 则把任务改写成“给定前缀，继续生成”。源信息只是前缀中的一部分，与目标 token 一起经过同一个 Transformer 栈。

#### Cross-attention 与前缀读取

CED 的 Decoder 每层可以用 cross-attention 直接查询 Encoder 状态。读取源信息和建模目标历史是两个不同子层，通常使用不同参数。

Decoder-only 没有这条显式通道。生成位置通过 self-attention 读取前面的 prompt KV，源信息检索与目标历史建模共享同一套注意力参数。

这并不意味着 Decoder-only 每生成一步都要重新计算 prompt。实际推理会缓存 prompt 的 key/value；区别在于这些 key/value 属于同一条 causal self-attention 流，而不是来自独立 Encoder 的 cross-attention memory。

#### 因果 Encoder 不等于 Decoder

因果 Encoder 和 Decoder 都可能使用相同形状的 causal mask，但职责仍然不同：

- 因果 Encoder 把不断到达的源输入转换成可读取的表示；
- Decoder 根据目标前缀预测下一个目标 token；
- 是否生成 token、是否承担 next-token 目标，不能仅从 attention mask 判断。

因此，**模块名称不能只由 mask 决定**。一个 block 使用 causal self-attention，并不自动意味着它就是 Decoder。

### 为什么通用 LLM 多采用 Decoder-only

Decoder-only 的主要优势不是表达能力上天然“更高级”，而是统一和简单：

- 预训练、问答、翻译、代码生成都可以写成“前缀续写”；
- 整个模型只有一种主干 block 和一种主要注意力数据流；
- in-context learning 可以自然地表示为 prompt 中的示例与指令；
- KV cache、推理调度和训练目标相对统一。

Encoder–Decoder 或 CED 仍然适合源流和目标流天然不同的任务。例如语音帧与文本 token 的时间尺度差异很大，使用独立 Encoder 可以先把连续信号转换成更合适的表示；流式任务还可以精确控制 Encoder 提交状态与 Decoder 输出之间的延迟。

### 总结

可以用三句话记住全文：

1. **Encoder 与 Decoder 是功能角色。** Encoder 表示源输入，Decoder 根据目标前缀生成输出；Transformer 只是改变了实现这些角色的基本计算单元。
2. **Attention 类型由两个维度决定。** `self/cross` 看 \(Q\) 与 \(K,V\) 是否同源，`vanilla/causal` 看可见范围是否受到因果约束。
3. **CED 是两条因果流加 cross-attention，Decoder-only 是一条因果流。** 它们都可以流式处理前缀，但前者保留独立源表示和显式跨序列读取，后者把所有信息统一放进同一条前缀续写序列。


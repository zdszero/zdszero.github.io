% Latent 压缩

MLA 中的 \(c^{KV}\)、LatentMoE 中的 latent token，以及 Kimi K3 里的 LatentMoE，都围绕同一个词：latent。它不是某个模型特有的技巧，而是一个可以单独定义的数学对象。本节先给出 latent 的定义，再解释为什么要在 GEMM 两侧插入 latent，以及它在数学层面改变了什么。

### Latent 表示的定义

给定输入 \(X \in \mathbb{R}^{T \times H}\)，其中 \(T\) 是 token 数、\(H\) 是 hidden size。latent 表示指的是一个维度更低的中间变量 \(Z \in \mathbb{R}^{T \times L}\)（\(L < H\)），以及一对映射 \(\operatorname{Enc} : \mathbb{R}^{H} \to \mathbb{R}^{L}\) 和 \(\operatorname{Dec} : \mathbb{R}^{L} \to \mathbb{R}^{H}\)：

\[
Z = \operatorname{Enc}(X), \qquad \hat{X} = \operatorname{Dec}(Z).
\]

这个定义本身很空泛，真正决定含义的是三个问题：

1. **压缩的是什么？** 是整个 token 的 hidden state，还是 attention 的 K/V，还是专家计算使用的表示？
2. **latent 在哪里被物化？** 是写进 KV cache 长期保存，还是只在一次前向中短暂存在？
3. **谁在读取 latent？** 后续计算直接在这个低维空间中完成，还是立刻展开回 \(H\) 维？

"用了 latent"这句话本身并不蕴含任何显存或计算收益。收益来自 \(L\) 的大小、压缩与展开的次数，以及低维表示被复用的范围。后面的公式会用统一的语言量化这三点。

#### 与低维流形假设的区别

latent 只说明中间经过了一个更窄的变量，不蕴含输入数据一定位于某个低维流形上。数学上的硬约束来自映射的结构，而不是来自 \(L < H\) 这个事实本身：线性 latent 约束的是秩，带非线性的 latent 约束的是每层线性算子的内部宽度。

### GEMM 两侧的 latent 与低秩分解

先看最简单的形式。一个标准线性层写作

\[
Y = X W^{\mathsf T},
\]

其中 \(X \in \mathbb{R}^{T \times H}\)、\(W \in \mathbb{R}^{N \times H}\)、\(Y \in \mathbb{R}^{T \times N}\)。在它的两侧插入 latent，就是把一次 GEMM 拆成两次：

\[
Z = X W_{\mathrm{down}}^{\mathsf T}, \qquad
\hat{Y} = Z W_{\mathrm{up}}^{\mathsf T},
\]

其中

\[
W_{\mathrm{down}} \in \mathbb{R}^{L \times H}, \qquad
W_{\mathrm{up}} \in \mathbb{R}^{N \times L}.
\]

由于矩阵乘法满足结合律，

\[
X W^{\mathsf T} = X \left( W_{\mathrm{up}} W_{\mathrm{down}} \right)^{\mathsf T},
\]

所以这条路径计算的仍然是一个普通线性层，只是它的权重被限制为

\[
W_{\mathrm{eff}} = W_{\mathrm{up}} W_{\mathrm{down}} \in \mathbb{R}^{N \times H},
\qquad
\operatorname{rank}(W_{\mathrm{eff}}) \le L.
\]

这就回答了"一个 GEMM 为什么要加一个 latent"：**它把一个原本有 \(NH\) 个自由度的矩阵，换成了秩被 \(L\) 卡住的矩阵**，用表达能力换取参数量和计算量。latent 没有引入新的运算类型，它引入的是一个参数化约束。

#### 给定权重时的最优 latent

如果问题不是训练一个新矩阵，而是近似一个已经给定的 \(W\)，那么最优的秩 \(L\) 近似由 SVD 截断给出（Eckart–Young 定理）。设

\[
W = U \Sigma V^{\mathsf T}, \qquad
W_L = U_L \Sigma_L V_L^{\mathsf T},
\]

则

\[
\lVert W - W_L \rVert_F^2 = \sum_{i=L+1}^{r} \sigma_i^2,
\]

且最优分解可以取

\[
W_{\mathrm{up}} = U_L \Sigma_L^{1/2}, \qquad
W_{\mathrm{down}} = \Sigma_L^{1/2} V_L^{\mathsf T}.
\]

这给出了一个可直接执行的判据：查看 \(W\) 的奇异值谱。如果

\[
\frac{\sum_{i > L} \sigma_i^2}{\sum_{i} \sigma_i^2}
\]

已经很小，那么秩 \(L\) 的 latent 路径几乎不损失信息；如果谱尾部能量很大，瓶颈 \(L\) 就会成为真实的性能上限。**能否压缩是谱的性质，不是维度的性质。**

需要注意，真实训练中并不存在一个给定的 \(W\)，\(W_{\mathrm{up}}\) 与 \(W_{\mathrm{down}}\) 是和整个网络一起端到端学习的。SVD 只能给出"给定矩阵"意义下的最优近似，不能保证是网络层面的最优解。

#### 中间插入非线性之后

如果 latent 两侧之间还夹着非线性，例如

\[
f(X) = \sigma\!\left( X W_{\mathrm{down}}^{\mathsf T} \right) W_{\mathrm{up}}^{\mathsf T},
\]

那么复合映射不再是一个低秩线性映射，上面的秩结论对整体函数不再成立。此时保留下来的是一条局部结论：在任意点 \(x\)，该函数的 Jacobian 为

\[
J_f(x) = W_{\mathrm{up}} \operatorname{Diag}\!\left( \sigma'\!\left( W_{\mathrm{down}} x \right) \right) W_{\mathrm{down}}
\in \mathbb{R}^{N \times H},
\qquad
\operatorname{rank}(J_f(x)) \le L.
\]

也就是说，瓶颈限制的不是网络的整体表达能力，而是**每个线性算子可以使用的内部方向数**。这正是 LatentMoE 的形态：down projection、专家内部的非线性、up projection 三层结构，\(L\) 约束的是专家每一层线性权重的内部宽度。

### 参数量与计算量

#### 单层线性层

原路径的参数量与每个 token 的 FLOPs 分别为

\[
P_{\text{base}} = NH, \qquad F_{\text{base}} = 2THN.
\]

latent 路径为

\[
P_{\text{latent}} = LH + LN = L(H + N),
\qquad
F_{\text{latent}} = 2THL + 2TLN = 2TL(H + N).
\]

两者给出同一个压缩比：

\[
\frac{P_{\text{latent}}}{P_{\text{base}}}
= \frac{F_{\text{latent}}}{F_{\text{base}}}
= \frac{L}{H} + \frac{L}{N}.
\]

这个式子值得记住：压缩比由两项组成，一项来自 \(H \to L\) 的压缩，另一项来自 \(L \to N\) 的展开，两者都正比于 \(L\)。令比值小于 1，得到盈亏平衡条件

\[
L < \frac{HN}{H + N},
\]

即 \(L\) 必须小于 \(H\) 与 \(N\) 的调和平均数的一半。当 \(H = N\) 时，条件退化为

\[
L < \frac{H}{2}.
\]

换句话说，latent 只比 hidden size 小一点时，先压缩再展开反而比直接算更贵。

#### MoE 中的摊销

设每个专家是一个两层 FFN，输入输出宽度为 \(H\)、中间宽度为 \(N\)，共有 \(E\) 个专家、每个 token 激活 \(k\) 个。普通 MoE 的全部参数为

\[
P_{\text{base}} = 2ENH,
\]

而 LatentMoE 把共享的 down/up projection 加在两侧、让专家在 latent space 中工作之后，

\[
P_{\text{latent}} = 2LH + 2ENL,
\]

因此

\[
\frac{P_{\text{latent}}}{P_{\text{base}}}
= \frac{2LH + 2ENL}{2ENH}
= \frac{L}{EN} + \frac{L}{H}.
\]

每个 token 的激活计算量（只算被选中的 \(k\) 个专家）有同样的结构：

\[
F_{\text{base}} = 4kTHN, \qquad
F_{\text{latent}} = 4THL + 4kTLN,
\]

\[
\frac{F_{\text{latent}}}{F_{\text{base}}}
= \frac{L}{kN} + \frac{L}{H}.
\]

把三种场景放在一起对比：

| 场景 | 压缩比 | 摊销效果 |
| --- | --- | --- |
| 单层线性层 | \(\frac{L}{H} + \frac{L}{N}\) | 没有可摊销的对象 |
| MoE 全部专家参数 | \(\frac{L}{H} + \frac{L}{EN}\) | 共享投影被 \(E\) 个专家分摊 |
| MoE 每 token 激活计算 | \(\frac{L}{H} + \frac{L}{kN}\) | 只有 top-\(k\) 个专家参与 |

三行的第一项都是 \(L/H\)，第二项的分母被复用次数放大。这就是为什么 latent 在 MoE 里比在单层 dense 层里更有吸引力：同一个 down/up projection 被所有专家共享，\(E\) 和 \(k\) 越大，\(\frac{L}{EN}\) 与 \(\frac{L}{kN}\) 越接近零，压缩比趋近理想值 \(L/H\)。反过来，单个矩阵没有摊销对象，只能靠 \(N\) 本身足够大来压低 \(L/N\) 这一项。

也要注意反例：当 \(L\) 接近 \(H\) 时，三次 GEMM 的总量加上中间张量的读写会超过原来的一次 GEMM。所以问题通常不是"要不要加 latent"，而是"\(L\) 取多少"。

#### 显存与带宽

latent 同时改变了几种不同性质的资源占用，需要分开看：

- **常驻权重**：\(L(H + N)\) 替代 \(NH\)，减少的是参数存储；
- **每次前向的权重读取**：与常驻权重同阶，减少的是权重带宽；
- **中间张量**：latent 会引入 \(T \times L\) 的临时 workspace，通常远小于原来的 KV cache 或专家输出；
- **长期保存的表示**：如果 latent 需要缓存（例如 MLA 的 \(c^{KV}\)），节省体现在缓存容量与 decode 阶段的读取带宽上，而不是当次的 GEMM 计算量；
- **跨设备通信**：如果 latent 是 EP dispatch 的载荷（例如 LatentMoE），节省体现在 All-to-All 的字节数上。

一个容易混淆的点是：**latent 不自动减少 FLOPs**。它减少的是被压缩那个 GEMM 的 FLOPs，但会新增两次投影 GEMM，只有 \(\frac{L}{H} + \frac{L}{N} < 1\) 时净计算量才下降。

### 表达能力与信息瓶颈

#### 秩上限

纯线性 latent 路径的输出只能落在 \(W_{\mathrm{down}}\) 行空间张成的 \(L\) 维子空间中：

\[
W_{\mathrm{eff}} = W_{\mathrm{up}} W_{\mathrm{down}},
\qquad
\operatorname{rank}(W_{\mathrm{eff}}) \le L.
\]

这意味着下游计算只能读取 hidden space 中 \(L\) 个方向的信息，剩下的 \(H - L\) 个方向对这一分支不可见。是否可接受，取决于被压缩对象的信息是否真的集中在这 \(L\) 个方向上，这又回到前面的谱判据。

#### 信息瓶颈的下界

对固定目标 \(W\)，误差下界是谱尾部能量

\[
\sum_{i > L} \sigma_i^2.
\]

这是一个**与训练过程无关**的下界：无论怎样优化 \(W_{\mathrm{up}}\) 和 \(W_{\mathrm{down}}\)，只要 \(L\) 固定，都无法突破。这个下界在 \(L\) 很小时增长很快，所以瓶颈最直接的影响是模型容量。

在带非线性的结构里，通用逼近性仍然成立，表达能力的损失更多表现为**优化难度上升**和**有效宽度下降**，而不是某类函数完全无法表示。

#### 尺度对称与非凸性

参数化 \((W_{\mathrm{up}}, W_{\mathrm{down}}) \mapsto W_{\mathrm{up}} W_{\mathrm{down}}\) 不是单射。对任意可逆矩阵 \(S \in \mathbb{R}^{L \times L}\)，

\[
(W_{\mathrm{up}} S)\,(S^{-1} W_{\mathrm{down}}) = W_{\mathrm{up}} W_{\mathrm{down}}.
\]

也就是说，同一组有效权重对应无穷多组因子参数，多出的自由度有 \(L^2\) 个。这带来几个后果：

- 目标函数在因子参数空间中是非凸的，会出现鞍点和平台；
- 两个因子的尺度可以相互补偿（一个变大、另一个变小），权重衰减作用在因子上而不是作用在 \(W_{\mathrm{eff}}\) 上，因此通常需要额外的缩放约定或初始化策略；
- 有效映射的条件数 \(\sigma_1 / \sigma_L\) 直接影响数值稳定性，\(\sigma_L\) 很小时训练更困难。

#### 梯度也停留在低秩流形上

记 \(\mathcal{L}\) 为损失，\(G = \partial \mathcal{L} / \partial W_{\mathrm{eff}}\)，由链式法则

\[
\frac{\partial \mathcal{L}}{\partial W_{\mathrm{up}}} = G W_{\mathrm{down}}^{\mathsf T},
\qquad
\frac{\partial \mathcal{L}}{\partial W_{\mathrm{down}}} = W_{\mathrm{up}}^{\mathsf T} G.
\]

两个梯度矩阵的秩都不超过 \(L\)，因此一次梯度更新给出的

\[
\Delta W_{\mathrm{eff}} = \Delta W_{\mathrm{up}} W_{\mathrm{down}} + W_{\mathrm{up}} \Delta W_{\mathrm{down}}
\]

是两个秩不超过 \(L\) 的矩阵之和，秩不超过 \(2L\)。这正好是秩 \(L\) 矩阵流形在 \(W_{\mathrm{eff}}\) 处的切空间。换句话说，低秩约束在优化过程中是被解析地保持的：网络不会溢出到 \(L\) 维之外的子空间去。

### 两个典型应用

#### MLA：latent 作为可缓存的 KV 表示

沿用 `content/theory/attention/mla/index.md` 的记号，

\[
c^{KV} = X W_{DKV}, \qquad
K_{nope} = c^{KV} W_{UK}, \qquad
V = c^{KV} W_{UV}.
\]

把 K 与 V 的投影写在一起：

\[
\begin{bmatrix} K_{nope} & V \end{bmatrix}
= c^{KV} \begin{bmatrix} W_{UK} & W_{UV} \end{bmatrix},
\qquad \operatorname{rank} \le d_c.
\]

它的数学含义是：所有 head 的 key 和 value 都来自同一个 \(d_c\) 维 latent 子空间，head 之间共享的是坐标，而不是输出。

缓存量的差别可以直接算出来。以 DeepSeek-V2 的量级为例，\(d_c = 512\)、rope key 为 64，每 token 每层只需缓存 \(512 + 64 = 576\) 个元素；而 MHA 需要缓存 \(2 n_h d_h = 2 \times 128 \times 128 = 32768\) 个元素，两者相差约 \(57\) 倍。

MLA 与 MQA/GQA 的数学区别也在这里：

- MQA/GQA 减少的是 head 数量，让多个 query head 共用同一个 key/value head，相当于把 K/V 的表示限制在少数几个 head 中；
- MLA 保持 head 数量不变，让所有 head 共享一个 \(d_c\) 维 latent，再各自上投影。head 结构完整保留，被压掉的是 latent 子空间之外的方向。

decode 阶段的权重吸收只是结合律与转置恒等式。设 \(W_K \in \mathbb{R}^{d_c \times d_h}\) 是某个 head 的 key 上投影矩阵，\(c\) 是 latent，则

\[
q^{\mathsf T} \left( W_K c \right) = \left( W_K^{\mathsf T} q \right)^{\mathsf T} c,
\]

即 \(\langle q, W_K c \rangle = \langle W_K^{\mathsf T} q, c \rangle\)。query 被先吸收到 latent 空间后，attention 的 key 侧只剩 \(c^{KV}\) 一个"头"，所以 decode 阶段在形式上等价于 MQA；但 value 路径满足

\[
\sum_s a_s W_V c_s = W_V \left( \sum_s a_s c_s \right),
\]

说明 head 的上投影可以推迟到最后再做，因此效果上仍然接近 MHA。

有一点必须区分：**MLA 的压缩收益是缓存容量与 decode 阶段的读取带宽**。prefill 阶段 q_len 与 kv_len 同阶，MLA 走的就是 MHA 路径，压缩并不带来计算量收益；只有在 decode 阶段缓存被反复读取时，latent 才真正体现价值。

#### LatentMoE：latent 作为共享的专家工作空间

LatentMoE 的前向可以抽象为

\[
Z = X W_{\mathrm{down}}^{\mathsf T}, \qquad
Z' = \sum_{e \in \mathcal{S}(X)} \alpha_e(X) f_e(Z), \qquad
Y = Z' W_{\mathrm{up}}^{\mathsf T}.
\]

路由仍在 hidden space 中完成，被压缩的是专家计算使用的表示：

- \(W_{\mathrm{down}}\) 与 \(W_{\mathrm{up}}\) 由所有专家共享，专家特有的参数位于 latent space 内；
- 专家主体的内部宽度由原来的 \(H\) 变成 \(L\)，参数量与 FLOPs 按上一节的公式下降；
- latent 是一次前向中的临时量，不写回缓存，这一点与 MLA 相反。

两者共享同一副骨架：

```text
宽空间输入 X
    │  down projection
    ▼
窄空间表示 Z          （维度 L）
    │  在窄空间中重复主要计算
    ▼
窄空间结果 Z'
    │  up projection
    ▼
宽空间输出 Y
```

差异可以整理成一张表：

| 维度 | MLA | LatentMoE |
| --- | --- | --- |
| 压缩对象 | K/V 表示 | routed token 表示 |
| 窄空间中的计算 | attention 的匹配与加权 | 专家 FFN |
| latent 是否长期保存 | 是，写入 KV cache | 否，单次前向的临时量 |
| 主要收益 | KV cache 容量与读取带宽 | 专家参数、激活计算与 EP 通信载荷 |
| 是否改变 head 或专家数量 | 不改变 head 数 | 不改变 top-\(k\) |
| 数学约束 | K/V 投影的秩不超过 \(d_c\) | 专家线性层的内部宽度为 \(L\) |

LatentMoE 不改变路由的稀疏性，它改变的是被选中的专家在什么坐标空间里工作。

### 如何判断某个位置该不该加 latent

1. **先看谱。** 对目标权重做 SVD，计算前 \(L\) 个奇异值的能量占比。谱尾部能量大，说明 \(L\) 会成为硬瓶颈。
2. **再算盈亏平衡。** 单层 dense 用 \(L < \frac{HN}{H+N}\)；MoE 用 \(\frac{L}{H} + \frac{L}{kN} < 1\)（激活计算）或 \(\frac{L}{H} + \frac{L}{EN} < 1\)（全部参数）。
3. **确认有摊销对象。** 同一个 latent 被多少下游计算复用？MLA 里是所有 head 与所有历史位置，LatentMoE 里是所有专家。复用次数越多，收益越接近理想的 \(L/H\)。
4. **确认收益类型。** 是省缓存（MLA）、省参数与计算（LatentMoE），还是省通信？三者不能混为一谈。
5. **检查实现代价。** latent 引入的 GEMM 维度小，在 decode 场景可能受 kernel launch 和带宽限制，实际是否更快取决于融合程度与 batch size。
6. **检查精度敏感度。** attention 的 K/V 与专家中间表示通常能承受压缩，输出 logits 或需要精确复现的路径通常不能。

一个经验判据是：只有当 \(L\) 足够小、下游计算能在窄空间中完整完成，并且低维表示被大量复用时，latent 才值得引入。仅仅"中间过一层窄表示"并不构成优化。

### 与 LoRA、SVD 的关系

| 方法 | 数学形式 | 低秩结构扮演的角色 |
| --- | --- | --- |
| SVD 截断 | \(W_L = U_L \Sigma_L V_L^{\mathsf T}\) | 对给定 \(W\) 的最优秩 \(L\) 近似 |
| LoRA | \(W + BA\)，\(\operatorname{rank}(BA) \le r\) | 在满秩原矩阵上增加低秩增量 |
| latent 路径 | \(W_{\mathrm{up}} W_{\mathrm{down}}\)（中间可带非线性） | 用低秩形式替代原计算 |

三者都用到"两个瘦矩阵相乘"，差别在于低秩结构是替代、修正还是近似：

- LoRA 的总矩阵秩不低于原矩阵，只是少学了增量部分；
- latent 路径的秩上界就是 \(L\)，压缩更彻底，代价也更直接；
- SVD 是事后近似，latent 是事前设计，\(W_{\mathrm{up}}\) 与 \(W_{\mathrm{down}}\) 与网络其余部分一起训练。

### 常见面试问题

- 为什么在 GEMM 两侧插入 latent 会同时降低参数量和 FLOPs？压缩比是多少，什么时候反而更贵？
- 一个秩 \(L\) 的 latent 路径在数学上被限制在哪一类函数里？如果中间插入非线性，这个限制还在吗？
- latent 的 \(W_{\mathrm{up}} W_{\mathrm{down}}\) 参数化存在什么对称性？它对优化和权重衰减有什么影响？
- MLA 与 MQA 都减少了 K/V 信息，在数学上的区别是什么？
- MLA 的 decode 阶段为什么可以等价成 MQA？用到了哪条矩阵性质？
- LatentMoE 的收益为什么会随专家数 \(E\) 和 top-\(k\) 变化？
- 给定一个训练好的权重矩阵，如何估计需要多大维度的 latent？

### 小结

- latent 的数学本质是"低维中间变量 + 一对压缩/展开映射"；在纯线性情形下，它等价于秩不超过 \(L\) 的矩阵分解。
- 参数量与计算量的压缩比是 \(\frac{L}{H} + \frac{L}{N}\)，在 MoE 中被 \(E\) 或 \(k\) 进一步摊销；盈亏平衡点约为 \(L < \frac{HN}{H+N}\)。
- 数学代价是秩上限与信息瓶颈，误差下界由奇异值谱尾部给出；非线性保留了通用逼近性，但局部 Jacobian 的秩仍被 \(L\) 限制。
- 因子参数化是非凸的，存在 \(L^2\) 维的尺度冗余，梯度更新停留在秩 \(2L\) 的切空间内。
- MLA 用它压缩 KV cache 并用结合律做权重吸收，LatentMoE 用它构造共享的专家工作空间，两者共享同一个"先压缩、在窄空间计算、再展开"的骨架。


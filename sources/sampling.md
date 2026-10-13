% 采样

### 问题起点：下一步 token 是一个随机变量

自回归生成把序列概率分解为条件概率的乘积：

\[
p(y_{1:T}) = \prod_{t=1}^{T} p\!\left(y_t \mid y_{1:t-1}\right)
\]

给定前缀 \(y_{1:t-1}\)，模型只做一件事：对词表 \(V\) 中每个 token \(v\) 输出一个打分 \(z_v \in \mathbb{R}\)，记

\[
z = f_\theta\!\left(y_{1:t-1}\right) \in \mathbb{R}^{\lvert V \rvert}
\]

这组 \(z\) 叫 logits。它是实数打分，不是概率。所谓「采样」，就是先用某种规则把 \(z\) 变成一个概率分布 \(p\)，再从这个分布里抽出一个 token。不同解码策略的差别，几乎全部集中在「怎样把 \(z\) 变成 \(p\)」这一步。

- 贪心解码（greedy）取 \(\arg\max_v z_v\)，等价于极端的确定性选择；
- beam search 保留多条候选路径，按累积对数概率排序；
- 随机采样从分布中抽取，每次结果可能不同。

本文只讨论随机采样这条线，也就是 `temperature`、`top_k`、`top_p` 这些参数真正作用的地方。

### 从 logits 到概率分布

#### softmax

标准做法是把 logits 过一遍 softmax：

\[
p_v = \mathrm{softmax}(z)_v = \frac{\exp(z_v)}{\sum_{u \in V} \exp(z_u)}
\]

它满足 \(p_v > 0\) 且 \(\sum_v p_v = 1\)。softmax 保序：\(z_v > z_u \iff p_v > p_u\)，所以概率最高的 token 就是 logits 最大的 token。

#### 数值稳定性

直接算 \(\exp(z_v)\) 在 logits 较大时会溢出。工程实现统一先减去最大值：

\[
p_v = \frac{\exp\!\left(z_v - m\right)}{\sum_{u \in V} \exp\!\left(z_u - m\right)}, \qquad m = \max_u z_u
\]

分子分母同乘 \(e^{-m}\)，数学上等价，但指数里最大只剩 \(0\)，不会再溢出。这是所有推理框架里 softmax kernel 的默认写法。

#### logits、logprob 与概率

三者是同一份信息的不同刻度：

\[
\log p_v = z_v - \log \sum_{u \in V} \exp(z_u)
\]

返回给用户的 `logprobs` 就是 \(\log p_v\)。它和概率的关系是指数的，所以 \(0.1\) 与 \(0.01\) 的差距在 log 空间里只差约 \(2.3\)。阅读 logprob 时把它当成「负的惊讶程度」更直观：越接近 \(0\) 越确定。

### 统一框架：截断 = 选子集 + 重归一化

`top_k`、`top_p`、`min_p` 这些参数都属于**截断类**（truncation / warping）。它们共享同一个数学形式：先根据分布 \(p\) 选出一个候选子集 \(S \subseteq V\)，把 \(S\) 之外的 token 概率置零，再在 \(S\) 内重新归一化：

\[
\tilde{p}_v = \frac{p_v \cdot \mathbf{1}\!\left[v \in S\right]}{\sum_{u \in S} p_u}
\]

等价地，可以理解为先把 logits 改成

\[
z_v' = \begin{cases} z_v, & v \in S \\ -\infty, & v \notin S \end{cases}
\]

再对 \(z'\) 做一次 softmax，因为 \(\exp(-\infty) = 0\)，结果自动就是上面的重归一化分布。

这个视角把「限制」说清楚了：**截断参数不改变 \(S\) 内部 token 的相对大小，它只决定哪些 token 被彻底排除**。被排除的 token 概率直接归零，且不会把概率分给被排除者——原来的概率质量在 \(S\) 内按比例放大。

于是所有截断参数的差异，只剩一个核心问题：\(S\) 怎么选。

### 温度：调整分布的锐度

温度 \( \tau > 0 \) 先作用于 logits：

\[
p_v^{(\tau)} = \frac{\exp\!\left(z_v / \tau\right)}{\sum_{u \in V} \exp\!\left(z_u / \tau\right)}
\]

它等价于把 logits 整体缩放 \(1/\tau\) 后再 softmax。

- \(\tau = 1\)：保持模型原始分布；
- \(\tau < 1\)：logits 差距被放大，分布变尖，高概率 token 更占优，\( \tau \to 0^+ \) 时退化为贪心；
- \(\tau > 1\)：logits 差距被压缩，分布变平，低概率 token 也有机会被抽中。

温度的一个关键性质是**它不改变排序**：对 \(z_v > z_u\)，当 \(\tau > 0\) 时 \(z_v/\tau > z_u/\tau\)。因此 top-k 看到的名次不变，候选集合与温度无关；top-p 则不同，它的截断位置由概率的绝对刻度决定，温度一变，累积概率到达 \(p\) 的位置就可能移动，候选集合随之改变。

温度作用在 logits 上，也意味着它会放大或缩小后面所有截断操作的绝对概率刻度，实践中最常见的困惑就是「同一个 `top_p`，调小温度后候选数明显变少」，原因就在这里。

### Top-k：按排名截断

#### 数学定义

设把 \(p\) 降序排列为 \(p_{(1)} \ge p_{(2)} \ge \cdots \ge p_{(\lvert V \rvert)}\)，对应的 token 记为 \((1), (2), \ldots\)。top-k 的候选集是固定大小的排名前缀：

\[
S_k = \left\{\, (1), (2), \ldots, (k) \,\right\}
\]

等价的阈值形式是：取第 \(k\) 大的概率作为阈值

\[
\theta_k = p_{(k)}
\]

则 \(S_k = \left\{\, v \;\middle|\; p_v \ge \theta_k \,\right\}\)。实现上通常是 `sort` 后取前 \(k\)，或 `topk` 后把其余置为 \(-\infty\)。

#### 性质与局限

| 性质 | 说明 |
| --- | --- |
| 候选数恒定 | 恒为 \(k\)，与分布形状无关 |
| 计算简单 | 一次 top-k，\(O(\lvert V \rvert + k \log k)\) 或近似 |
| 不随置信度自适应 | 分布很尖时，会保留一堆概率接近 \(0\) 的 token |
| 分布很平时 | 仍然只保留 \(k\) 个，可能过早截掉了合理候选 |
| \(k = 1\) | 退化为贪心解码 |
| \(k = \lvert V \rvert\) | 等价于不截断 |

「候选数恒定」既是 top-k 的优点，也是它的局限：模型对下一个 token 非常确定时，本该只留一两个候选，top-k 却仍塞进 \(k-1\) 个噪声；模型不确定、分布很平时，本该留更多候选，top-k 又强行砍到 \(k\) 个。top-p 正是为了修补这一点。

### Top-p：nucleus 采样，按概率质量截断

#### 数学定义

top-p 又叫 nucleus sampling。它不再固定 token 个数，而是固定**累积概率质量**。给定 \(p \in (0, 1]\)，定义 \(S_p\) 为降序前缀中累积概率首次达到 \(p\) 的最小集合：

\[
m = \min \left\{\, r \;\middle|\; \sum_{i=1}^{r} p_{(i)} \ge p \,\right\}, \qquad S_p = \left\{\, (1), (2), \ldots, (m) \,\right\}
\]

例如 \(p = 0.9\) 表示「保留累计占 \(90\%\) 概率质量的、概率最高的那批 token」。这个集合叫 nucleus（核）。

#### 实现细节：边界、重归一化与非空

数学定义里「首次达到 \(p\)」是清晰的，但实现有三处需要对齐：

- **边界 token 必须保留**。按累积和判断时，第 \(m\) 个 token 正是把累积概率推过 \(p\) 的那一个，不能因为「超过了 \(p\)」就丢掉，否则实际保留质量会小于 \(p\)。Hugging Face 的实现用了一个「右移一位」的技巧：先把 `cumulative_probs > p` 标成待删除，再整体右移一格，并把第 \(0\) 位强制设回「不删除」，恰好实现「保留跨越阈值的那一项」。
- **至少保留一个 token**。若阈值过小或数值误差导致没有 token 满足条件，实现会保底留下概率最高的 token，避免空集导致除零。
- **重归一化后的概率才是采样分布**。\(S_p\) 内部原概率之和通常严格大于 \(p\)（因为最后一项是跨过阈值的），除以这个和才是采样用的 \(\tilde{p}\)。

#### 性质与局限

| 性质 | 说明 |
| --- | --- |
| 候选数自适应 | 分布尖时候选少，分布平时候选多，这正是 nucleus 的核心动机 |
| 无固定候选上限 | 极端情况下可能保留大量 token，解码开销随之上升 |
| 阈值语义是相对质量 | 它保证「覆盖的概率质量」，不保证「每个候选都足够好」 |
| 双峰分布 | 若两个峰都很窄，累计质量可能在一个峰内就达到 \(p\)，于是另一个峰被整体删掉 |
| \(p = 1\) | 等价于不截断（仅有一次不必要的重归一化） |
| \(p \to 0^+\) | 退化为取 \(\arg\max\)，即贪心 |

一句话总结两者差别：**top-k 约束的是候选个数，top-p 约束的是候选覆盖的概率质量**。前者用绝对计数换取确定性，后者用绝对质量换取自适应性。

### Top-k 与 top-p 的组合顺序

两个参数同时开启时，**执行顺序会影响最终候选集**，这不是可以随便交换的算子。

#### 一个数值例子

设词表只有 6 个 token，温度为 \(1\)，softmax 后概率为

| token | A | B | C | D | E | F |
| --- | --- | --- | --- | --- | --- | --- |
| \(p\) | 0.5573 | 0.2050 | 0.1243 | 0.0754 | 0.0277 | 0.0102 |

单独使用：

- `top_k = 3`：保留 \(\{A, B, C\}\)，概率和为 \(0.8866\)，重归一化后约为 \(0.6286, 0.2312, 0.1402\)；
- `top_p = 0.9`：累积概率依次为 \(0.5573, 0.7623, 0.8866, 0.9620\)，首次达到 \(0.9\) 的位置是第 \(4\) 项，因此保留 \(\{A, B, C, D\}\)，概率和为 \(0.9620\)，重归一化后约为 \(0.5793, 0.2131, 0.1293, 0.0784\)。

现在同时取 `top_k = 2` 与 `top_p = 0.6`：

- **先 top-k 后 top-p**：先保留 \(\{A, B\}\)，概率和为 \(0.7623\)，重归一化为 \(0.7311, 0.2689\)；此时累积概率在第 \(1\) 项就达到 \(0.7311 \ge 0.6\)，于是 top-p 再砍成 \(\{A\}\)。
- **先 top-p 后 top-k**：先在原分布上取 top-p，累积概率 \(0.5573 < 0.6\)、\(0.7623 \ge 0.6\)，保留 \(\{A, B\}\)；再用 top-k 取前 \(2\) 个，仍为 \(\{A, B\}\)。

同一个参数组合，两种顺序得到 \(\{A\}\) 与 \(\{A, B\}\) 两个不同候选集。

#### 主流框架的实际顺序

Hugging Face `generate` 的 warper 链（核心部分）是固定的：

```text
temperature -> top-k -> top-p -> typical_p -> min_p
```

先做温度缩放，再做 top-k，再做 top-p。由于 top-p 的累积概率是在 top-k 之后的分布上计算的，两者组合时的实际语义是「先按排名砍到 \(k\) 个，再在其中按概率质量二次截断」。

vLLM 的 `_apply_top_k_top_p` 同样是先用 `logits.topk` 做 top-k，再做 top-p 的排序与累积，因此顺序一致。自己在推理框架上加自定义采样器时，需要显式确认顺序，否则同一个 `top_p` 会给出不同结果。

### 其他常见的 logits 变换

#### min-p

min-p 用一个相对阈值代替固定排名或固定质量：

\[
S_{\min p} = \left\{\, v \;\middle|\; p_v \ge \alpha \cdot \max_u p_u \,\right\}, \qquad \alpha \in (0, 1]
\]

只有概率达到「最大概率的 \(\alpha\) 倍」的 token 才留下。它的自适应粒度比 top-p 更细：模型越确定，阈值越高，留下的 token 越少；模型越犹豫，阈值越低。\( \alpha \to 0^+ \) 时不截断，\( \alpha = 1 \) 时退化为贪心。

#### typical sampling

typical sampling 从信息量角度筛选「典型」token。设分布的熵为

\[
H = -\sum_{u} p_u \log p_u
\]

每个 token 的惊讶程度为 \(-\log p_v\)，取偏离熵最小的那些 token：

\[
d_v = \left| -\log p_v - H \right|
\]

按 \(d_v\) 升序累积概率，保留累积达到阈值 \(\gamma\) 的最小集合。直观上，它偏好「信息量接近分布平均惊讶程度」的 token，同时抑制过于平淡和过于意外的两端。

#### 重复惩罚与存在、频率惩罚

这些不是截断，而是直接改写 logits，作用在温度之前：

\[
z_v \leftarrow z_v - \alpha \cdot \mathbf{1}\!\left[v \in \text{context}\right] \quad (\text{presence penalty})
\]

\[
z_v \leftarrow z_v - \alpha n_v \quad (\text{frequency penalty})
\]

其中 \(n_v\) 是 token \(v\) 在已生成文本中出现的次数。CTRL 风格的 repetition penalty 则按符号分段缩放：

\[
z_v \leftarrow \begin{cases} z_v / \rho, & z_v > 0 \\ z_v \cdot \rho, & z_v \le 0 \end{cases} \qquad \rho > 1
\]

这类操作会改变排序，因此必须放在 top-k、top-p 之前执行；顺序颠倒了，截断集合就不是模型意图的那个。

### 如何从截断分布中真正抽出一个 token

有了 \(\tilde{p}\)，还需要一个随机数机制把它变成具体的 token。

#### 逆变换采样

对 \(\tilde{p}\) 做前缀和，得到累积分布函数 \(F_v = \sum_{u \le v} \tilde{p}_u\)。抽 \(u \sim \mathrm{Uniform}(0, 1)\)，取

\[
v^{\star} = \min \left\{\, v \;\middle|\; F_v \ge u \,\right\}
\]

这正是一维分布的通用采样方法，也是 `torch.multinomial` 一类接口的底层语义。它的复杂度主要来自累积和，通常和一次排序同量级。

#### Gumbel-max 技巧

另一种等价做法是给每个 \(\log \tilde{p}_v\) 加一个独立的标准 Gumbel 噪声 \(g_v\)，然后取最大值：

\[
v^{\star} = \arg\max_v \left(\log \tilde{p}_v + g_v\right), \qquad g_v \sim \mathrm{Gumbel}(0, 1)
\]

正确性可以这样看。标准 Gumbel 的累积分布函数是

\[
\Pr(g_v \le x) = \exp\!\left(-e^{-x}\right)
\]

令 \(G_v = \log \tilde{p}_v + g_v\)，则

\[
\Pr\!\left(G_v \le x\right) = \exp\!\left(-\tilde{p}_v e^{-x}\right)
\]

于是对所有 \(u\) 取交：

\[
\Pr\!\left(\max_u G_u \le x\right) = \prod_u \exp\!\left(-\tilde{p}_u e^{-x}\right) = \exp\!\left(-e^{-x} \sum_u \tilde{p}_u\right) = \exp\!\left(-e^{-x}\right)
\]

即最大值本身仍服从标准 Gumbel 分布。再对 \(G_v\) 的条件密度积分：

\[
\Pr\!\left(v = \arg\max_u G_u\right) = \int_{-\infty}^{\infty} \tilde{p}_v e^{-x} \exp\!\left(-\tilde{p}_v e^{-x}\right) \prod_{u \ne v} \exp\!\left(-\tilde{p}_u e^{-x}\right) dx = \tilde{p}_v
\]

所以 \(\arg\max\) 的分布恰好是 \(\tilde{p}\)。Gumbel-max 把「采样」变成了「加噪 + 取最大」，在需要批量、并行或与贪心 kernel 复用的场景里很好用。

### 采样如何影响 logprob 与可复现性

- **返回的 logprob 是截断后的对数概率**。框架报告的是 \(\log \tilde{p}_v\)，而不是原始 \(\log p_v\)。一旦开了 top-k 或 top-p，同一个 token 的 logprob 会因为候选集合变化而变化，跨配置比较 logprob 没有意义。
- **截断会系统性抬高保留 token 的概率**。原本 \(p_v\) 很小的 token，如果落在 nucleus 内，重归一化后可能变大；落在 nucleus 外则直接变成 \(0\)，对应 \(\log \tilde{p}_v = -\infty\)。
- **温度与 logprob 一致缩放**。\(\log p_v^{(\tau)} = z_v / \tau - \log \sum_u \exp(z_u / \tau)\)，温度越低，分布越尖，最优 token 的 logprob 越接近 \(0\)。
- **随机采样天然不可复现，除非固定种子**。即使固定种子，batch 组成、kernel 实现和浮点归约顺序仍可能引入差异；greedy 在 `temperature = 0` 时看似确定，但也受 batch-invariant 问题影响，详见《Nondeterminism》一文。

### 与约束解码的关系

约束解码（constrained decoding）也是截断，只是候选集合不是由概率决定，而是由语法决定：只保留「接上去仍然能走到合法输出」的 token。两者可以复合：

\[
z_v' = \begin{cases} z_v, & v \in S_{\text{top-p}} \cap A_t \\ -\infty, & \text{otherwise} \end{cases}
\]

其中 \(A_t\) 是第 \(t\) 步的语法合法 token 集合。需要注意实现顺序：先在 logits 上做重复惩罚、温度、top-k、top-p，再叠加语法 mask，还是反过来，结果并不相同。更细致的讨论见《约束解码》一节。

### 总结

| 参数 | 作用对象 | 候选集合 | 核心公式 |
| --- | --- | --- | --- |
| temperature \(\tau\) | logits 缩放 | 改变分布形状，不改变排序 | \(p_v^{(\tau)} \propto \exp(z_v/\tau)\) |
| top-k | 排名 | 固定 \(k\) 个 | \(S_k = \{(1), \ldots, (k)\}\) |
| top-p | 概率质量 | 累积达到 \(p\) 的最小集合 | \(m = \min\{r : \sum_{i \le r} p_{(i)} \ge p\}\) |
| min-p | 相对概率 | 概率不低于最大值的 \(\alpha\) 倍 | \(p_v \ge \alpha \max_u p_u\) |

把这一节压缩成一句话：**采样参数的全部作用就是选定一个子集 \(S\)，把 \(S\) 之外的 token 概率清零，再在 \(S\) 内重归一化后抽样**。top-k 用「固定个数」限制 \(S\)，top-p 用「固定概率质量」限制 \(S\)，min-p 用「相对最大概率」限制 \(S\)；温度则在此之前调整分布本身的锐度。理解了这两层，就能解释绝大多数采样相关的调参现象。


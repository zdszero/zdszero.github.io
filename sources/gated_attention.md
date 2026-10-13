% Gated Attention

## Gated MLA

MLA 和 Gated Attention 解决的是两个不同的问题：

- MLA 改变的是**状态**：用共享 latent \(c_{KV}\) 压缩每个 token 的 K/V，让 KV cache 与读取成本大幅下降；
- Gated Attention 改变的是**输出**：在注意力结果上乘一个由输入动态生成的门控，让每个 head（或每个 channel）有随 token 变化的输出强度。

Gated MLA 就是把后者的门控接到前者的输出上。**压缩路径、KV cache 结构、MQA absorb 等 MLA 特性全部保持不变，只在注意力输出之后多一条门控支路。** 所以它不是一种新的压缩方案，而是对 MLA 表达能力与训练稳定性的一次低成本补强。

### 公式

把 MLA 的压缩写开，并用 \(d_h\) 表示 QK 的 head 维度：

\[
c_{KV}=XW^{DKV},\qquad
K=c_{KV}W^{UK},\qquad
V=c_{KV}W^{UV}
\]

在标准 MLA 输出之后加入门控：

\[
\tilde{\boldsymbol{O}} =
\Big(\operatorname{softmax}\!\Big(\frac{\boldsymbol{Q}\boldsymbol{K}^{\mathsf T}}{\sqrt{d_h}}+M\Big)\boldsymbol{V}\Big)
\odot
\sigma\!\left(\boldsymbol{X}\boldsymbol{W}_g\right)
\]

其中 \(\boldsymbol{W}_g\) 直接从当前层的 hidden state \(\boldsymbol{X}\in\mathbb{R}^{T\times d}\) 生成，与 Q/K/V 的四条投影支路并列，因此它的输入永远是当前的 \(X\)，**不依赖历史 token，也不进入 KV cache**。

### 门控的形状与粒度

关键在 \(\boldsymbol{W}_g\) 的输出宽度：

| \(\boldsymbol{W}_g\) 输出宽度 | 门控形状 | 粒度 |
| --- | --- | --- |
| \(H\) | \([B,S,H]\) | 每个 head 一个标量 |
| \(H d_v\) | \([B,S,H,d_v]\) | 每个 head 的每个 value 通道一个门控 |
| \(d\) | \([B,S,d]\) | 与 head 结构无关的通道级门控 |

现代 MLA 实现通常取 \(H d_v\)，也就是让门控宽度与 `o_proj` 的输入宽度一致。这样做有两个直接好处：

1. 门控可以 reshape 成 \([H,d_v]\) 后与注意力输出逐元素相乘，**不需要广播**，kernel 融合最简单；
2. 粒度为 (head, value channel)，比 head 级标量更细，同时保留了 head 之间的区分度。

以 Kimi K3 的 MLA 层为例，`g_proj.weight` 的形状是 \(12288\times 7168\)，而 `o_proj.weight` 是 \(7168\times 12288\)，两者恰好互为转置。\(12288=H d_v\) 正是全部 head 拼起来的注意力输出宽度，这说明 `g_proj` 产生的门控作用在 `o_proj` 之前、带 head 结构的 value 空间上，而不是 hidden 通道上。

### 门控加在哪里

同一个门控放在不同位置，语义和成本都不一样：

| 位置 | 门控结构 | 说明 |
| --- | --- | --- |
| \(W^{UV}\) 展开之后、`o_proj` 之前 | \([B,S,H,d_v]\) | 最常见，能区分 head，与 `o_proj` 输入对齐 |
| `o_proj` 之后 | \([B,S,d]\) | 退化为 hidden 通道级，失去了 head 维度 |
| 直接乘在 latent \(c_{KV}\) 上 | \([B,S,d_c]\) | 同时改动所有 head 的 K 与 V，语义完全不同，属于另一类设计 |

需要特别强调的是 decode 阶段。MLA 在 decode 时走 **MQA absorb 路径**，先把 \(W^{UK}\) 吸收进 Q、把输出的 \(W^{UV}\) 展开推迟到最后：

\[
\Big(\operatorname{softmax}\!\Big(\frac{(c_qW^{UQ}W^{UK\,\mathsf T})c_{KV}^{\mathsf T}}{\sqrt{d_h}}\Big)c_{KV}\Big)W^{UV}
\]

门控乘在这一步**之后**。它只依赖 \(X\)，既不影响权重吸收的数学形式，也不影响 latent 的缓存方式，所以 MLA 的 decode 优化可以原样保留。

### 为什么 MLA 特别需要门控

除了 Gated Attention 的通用收益（缓解 softmax 凸组合带来的输出幅度约束、attention sink 与 massive activation），MLA 还有两个自身原因：

- **value 方向的自由度本来就被压缩过**：所有 head 的 \(V\) 都由同一个 \(d_c\) 维 latent 经 \(W^{UV}\) 展开，\(d_c\ll H d_v\)。门控虽然不能改变 \(W^{UV}\) 的秩，但它提供了随 token 变化的乘性调制，这种形式不是一个固定的低秩线性映射能表达的，相当于在不增加 cache 的前提下补回一部分区分度。
- **MLA 常与线性注意力层混用**：在 KDA + MLA 这类混合架构里，线性/循环层自带衰减门控，而 full attention 层原本没有任何门控，两层的写入强度控制能力不对等。给 MLA 加输出门控后，两类层都能决定自己往 residual stream 里写多少，层间分工更容易平衡。

### 代价

门控不是“零成本的一行乘法”，它的开销与一张独立投影矩阵相当：

| 项目 | 代价 |
| --- | --- |
| 参数量 | \(d\times Hd_v\)。K3 中为 \(12288\times 7168\approx 88\text{M}\)，BF16 约 168 MB |
| Prefill 计算 | 每 token \(d\cdot Hd_v\) 次乘加，与一次 `o_proj` 的 GEMM 同量级 |
| Decode 成本 | 每个 step 要完整读取这张权重，约 168 MB/层，是**显存带宽**开销而非 cache 开销 |
| KV cache | 完全不受影响，门控只依赖当前 hidden state |
| 精度 | sigmoid 需要在较高精度下计算，否则饱和区梯度会消失 |

因此在显存与通信预算里，`g_proj` 必须和 `q_proj`、`o_proj` 一样被当作主矩阵统计，不能因为它是“门控”就按小参数处理。推理实现上，门控通常被融合进 attention kernel 的 epilogue 或 `o_proj` 的前置步骤，避免额外的 HBM 往返。

### 与 KDA 的 `g_proj` 的区别

K3 的 KDA 层里同样有一张 \(12288\times7168\) 的 `g_proj`，名字和形状都一样，但含义完全不同：

| 归属 | 门控位置 | 门控作用对象 | 数学形式 |
| --- | --- | --- | --- |
| MLA 层 | 注意力输出之后 | 单次 attention 的输出 | \(\boldsymbol{O}\odot\sigma(XW_g)\) |
| KDA 层 | 递归状态更新中 | 历史状态 \(\boldsymbol{S}\) 的遗忘与新值写入 | \(\alpha_t,\beta_t\) 作用在 \(\boldsymbol{S}_t=\alpha_t\boldsymbol{S}_{t-1}+\dots\) |

读权重表或做显存分析时，这两者不能互相印证，也不能用 KDA 的 gate 语义去解释 MLA 的 `g_proj`。

### 最小实现

下面省略 RoPE、KV cache 与 MQA absorb，只保留 latent 压缩与输出门控的位置：

```python
class GatedMLA(nn.Module):
    def __init__(self, d_model, n_heads, d_qk, d_v, d_c):
        super().__init__()
        self.n_heads, self.d_qk, self.d_v = n_heads, d_qk, d_v

        # MLA 的 latent 压缩：低秩 Q 与共享的 KV latent
        self.q_a_proj = nn.Linear(d_model, d_c, bias=False)
        self.q_b_proj = nn.Linear(d_c, n_heads * d_qk, bias=False)
        self.kv_a_proj = nn.Linear(d_model, d_c, bias=False)
        self.kv_b_proj = nn.Linear(d_c, n_heads * (d_qk + d_v), bias=False)

        # 门控宽度取 H * d_v，与 o_proj 的输入宽度一致
        self.g_proj = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.o_proj = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, x):
        B, S, _ = x.shape

        q = self.q_b_proj(self.q_a_proj(x)).view(B, S, self.n_heads, self.d_qk)
        kv = self.kv_b_proj(self.kv_a_proj(x)).view(
            B, S, self.n_heads, self.d_qk + self.d_v
        )
        k, v = kv.split([self.d_qk, self.d_v], dim=-1)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B, H, S, *]

        scores = (q @ k.transpose(-1, -2)) * self.d_qk ** -0.5
        out = scores.softmax(dim=-1) @ v  # [B, H, S, d_v]，真实实现需先加 causal mask

        # [B, S, H, d_v]，每个 head 的每个 value 通道单独门控
        gate = torch.sigmoid(self.g_proj(x)).view(B, S, self.n_heads, self.d_v)
        out = out.transpose(1, 2) * gate

        return self.o_proj(out.reshape(B, S, -1))
```

### 常见面试问题

- Gated MLA 与 Gated Attention 的区别是什么？门控的粒度从 head 级标量扩展到 (head, value channel)，并且要与 MLA 的 latent 压缩、absorb 路径对齐。
- 门控能不能提高 MLA 的压缩率、减少 KV cache？不能。压缩发生在 token 状态维度，门控作用在输出维度，两者互不影响。
- 为什么门控要加在 `o_proj` 之前？因为那里还保留 head 与 value channel 的结构，能按 head 区分；过了 `o_proj` 就只剩 hidden 通道。
- 为什么 decode 阶段的门控不破坏权重吸收？门控只依赖当前 hidden state，不参与 \(W^{UK}\)、\(W^{UV}\) 的吸收变换。
- Gated MLA 的代价是什么？一张与 `o_proj` 同量级的 \(d\times Hd_v\) 矩阵，prefill 多一次 GEMM，decode 多一份权重读取带宽；收益在表达能力和稳定性上。
- 如何和 KDA 的 `g_proj` 区分？前者门控 attention 输出，后者门控递归状态的遗忘与写入。

### 小结

Gated MLA 是在 MLA 的注意力输出上乘一个由 hidden state 动态生成的门控，门控宽度通常取 \(H d_v\)，与 `o_proj` 的输入对齐，作用位置在 \(W^{UV}\) 展开之后、`o_proj` 之前。它不改变 MLA 的压缩语义、不影响 KV cache 与权重吸收，代价是一张与 `o_proj` 同量级的主矩阵及其 decode 带宽，换来的是更细粒度的输出控制、对低秩 value 的补偿，以及与线性注意力层更对等的门控能力。


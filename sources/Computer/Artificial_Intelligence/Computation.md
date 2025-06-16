% Computation

### Prerequisite

- number of layers: `L`
- sequence tokens: `t`
- batch size: `b`
- dimension/hidden size: `D`
- intermidiate size: `I`
- vocab size: `V`
- heads: `h`
- head dimension: `d = D / h`
    - $d_k = d_q$
    - $d_v$
    - generally, $d_k = d_v = \frac{D}{h}$
- model parameters: `N`

----

- $A \cdot B$
- shape(A) = (x, y)
- shape(B) = (y, z)
- computation is $2xyz$
    - x2: mul and add

### Weight

$Q, K, V = X W = X (W_{Q}, W_{K}, W_{V})$

| Matrix                | Shape    |
|-----------------------|----------|
| $W_{Q}, W_{K}, W_{V}$ | `(D D)`  |
| $W_{qkv}$             | `(D 3D)` |
| $W_{O}$               | `(D D)`  |
| $X$                   | `(t D)`  |
| $W_{up}$              | `(D I)`  |
| $W_{down}$            | `(I D)`  |

### Attention

$S(Q K^{T} / \sqrt{k}) * V$

| Matrix      | Shape                 |
|-------------|-----------------------|
| $Q$         | `(b t D) → (b h t d)` |
| $K^{T}$     | `(b D t) → (b h d t)` |
| $Q K^{T}$   | `(b t t), (b h t t)`  |
| $V$         | `(b t D) → (b h t d)` |
| $Q K^{T} V$ | `(b t D) ← (b h t d)` |

shape: single head → multi head

### Transformer

Matrix and Mul binding:

- gemm
    - $W_{Q}, W_{K}, W_{V}, W_{O}$: q/k/v/o_proj
    - $W_{up}, W_{down}$: MLP
    - $W_{unembed}$: logits
- attention:
    - $S(Q K^{T} / \sqrt{d_k}) V$

---

- gemm
    - q/k/v/o_proj
        - shape = $(b, t, D) \cdot (D, D) \cdot 4$
        - flops = $L \cdot 2 \cdot b \cdot t \cdot D^2 \cdot 4$
    - MLP
        - shape = $(b, t, D) \cdot (D, I)$
        - flops = $L \cdot 2 \cdot b \cdot t \cdot D \cdot 3 \cdot I \cdot 3$
    - logits
        - shape = $(b, t, D) \cdot (D, V)$
        - flops = $2 \cdot b \cdot t \cdot D \cdot V$
    - Total
        - flops = $8 \cdot L \cdot b \cdot t \cdot D^2 + 6 \cdot L \cdot b \cdot t \cdot D \cdot I + 2 \cdot b \cdot t \cdot D \cdot V = b \cdot t \cdot (8 \cdot L \cdot D^2 + 6 \cdot L \cdot D \cdot I + 2 \cdot D \cdot V)$
    - Each token
        - flops = $8L D^2 + 6 L D I + 2 D V = Lhd(8hd + 6I) + 2hdV \approx 2N$
- attention
    - $Q K^{T}$
        - shape = $(b, h, t, d) \cdot (b, h, d, t)$
        - flops = $L \cdot 2 \cdot b \cdot h \cdot d \cdot t^2$
    - $\text{softmax}(Q K^{T})$
        - shape = $(b, h, t, t)$
        - flops = $L \cdot b \cdot h \cdot t^2 \cdot \text{factor}$
            - factor = 1 + exp + division
    - $\text{softmax}(Q K^{T}) \cdot V$
        - shape = $(b, h, t, t) \cdot (b, h, t, d)$
        - flops = $L \cdot 2 \cdot b \cdot h \cdot d \cdot t^2$
    - Total
        - flops = $4 \cdot L \cdot b \cdot h \cdot d \cdot t^2 + L \cdot b \cdot h \cdot t^2 \cdot \text{factor}$
        - factor is less than $4d$ which we ignore first
    - Each token
        - flops = $4 \cdot L \cdot h \cdot d \cdot t$

---

Compare attention and gemm

attention flops / gemm flops = $\frac{t}{2hd + 1.5I} = \frac{t}{2D + 1.5I}$

For llama2-7b, $2D +1.5I = 24704$, in most cases, $t$ is much smaller thant $2D + 1.5T$

### Model Size

Model Parameters are saved in following martix:

- $W_{embed}$
- $W_{Q}, W_{K}, W_{V}, W_{O}$
- $W_{up}, W_{down}$
- $W_{unembed}$

---

- GPT3:
    - $\text{d_embed} = D = 12,288$
        - $d_k = d_v = \frac{D}{h} = 128$
    - $\text{n_vocab} = V = 50,257$
    - $\text{n_intermidiate} = I = 49,152$
    - $\text{n_heads} = h = 96$
    - $\text{n_layers} = L = 96$

| Matrix                       | Equation            | Parameters     |
|------------------------------|---------------------|----------------|
| $W_{embed}, W_{unembed}$     | $D \cdot V$         | 617,558,016    |
| $W_{Q}, W_{K}, W_{V}, W_{O}$ | $L \cdot D^2$       | 14,495,514,624 |
| $W_{up}, W_{down}$           | $L \cdot D \cdot I$ | 57,982,058,496 |

$N = 2DV + 4LD^2 + 2LDI$

updown * 2 + qkvo * 4 + embed * 2 = 175,181,291,520 ≈ 175B

### Autoregressive Process

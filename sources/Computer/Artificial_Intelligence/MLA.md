% MLA

### MHA

- $[\mathbf{q}_{t,1}; \mathbf{q}_{t,2}; ...; \mathbf{q}_{t,n_h}] = \mathbf{q}_t = W^{Q} \mathbf{h}_{t}$
- $[\mathbf{k}_{t,1}; \mathbf{k}_{t,2}; ...; \mathbf{k}_{t,n_h}] = \mathbf{k}_t = W^{K} \mathbf{h}_{t}$
- $[\mathbf{v}_{t,1}; \mathbf{v}_{t,2}; ...; \mathbf{v}_{t,n_h}] = \mathbf{v}_t = W^{V} \mathbf{h}_{t}$
- $\mathbf{o}_{t,i} = \sum_{j=1}^{t} \text{Softmax}_j\left(\frac{\mathbf{q}_{t,i} \mathbf{k}_{j,i}^\top}{\sqrt{d_h}}\right) \mathbf{v}_{j,i}$
    - $i$ 表示第 $i$ 个 head
    - $t$ 当时计算第 $t$ 个 token 的 attention
- $\mathbf{u}_t = W^O[\mathbf{o}_{t,1}; \mathbf{o}_{t,2}; ...; \mathbf{o}_{t,n_h}]$

> $\mathbf{o}_{t,i}$ 中的 $t$ 表示当前总共有 $t$ 个 token，

### MLA

- $ \mathbf{c}_t^{KV} = W^{DKV} \mathbf{h}_t $
  - $ \mathbf{c}_t^{KV} \in \mathbb{R}^{d_c} $：是 key 和 value 压缩的 latent vector，且 KV 的压缩维度 $ d_c \ll d_h n_h $
  - $ W^{DKV} \in \mathbb{R}^{d_c \times d} $：down-projection 矩阵
- $ \mathbf{k}_t^C = W^{UK} \mathbf{c}_t^{KV} $
- $ \mathbf{v}_t^C = W^{UV} \mathbf{c}_t^{KV} $
  - $ W^{UK}, W^{UV} \in \mathbb{R}^{d_h n_h \times d_c} $：up-projection 矩阵

__为什么 rope 和 nope 分离？__

首先 rope 的旋转矩阵无法被融合进入 W_q 和 k_b 矩阵之中（该矩阵并不是一个常量）：


$$
q_t^{(s)} {k_i^{(s)}}^\top 
= \left( x_t W_q^{(s)} \mathcal{R}_t \right) 
  \left( c_i W_k^{(s)} \mathcal{R}_i \right)^\top 
= x_t \left( W_q^{(s)} \mathcal{R}_{t-i} {W_k^{(s)}}^\top \right) c_i^\top
$$

MLA 在推理的时候不想缓存完整的 kv cache，只想缓存压缩的 ckv，但这样的话需要在升维后再重新计算 rope，它也不想，所以干脆把 q 和 k 拆成两部分，一部分是位置编码的，一部分是无关位置编码的。带位置编码的 k 可以缓存，比缓存全部维度都带位置的 k 强。

__normal 和 absorb 的区别？__

absorb 就是把 kv_b 拆分成 k_b 和 v_b，然后分别合入到 W_q 和 O_proj 中。

$$\mathbf{q}_t^\top \mathbf{k}_j^C = (W^Q \mathbf{h}_t)^\top W^{UK} \mathbf{c}_j^{KV} = \mathbf{h}_t^\top (W^{Q^\top}) W^{UK} \mathbf{c}_j^{KV} = \mathbf{h}_t^\top W^{Q^\top} W^{UK} \mathbf{c}_j^{KV} = \mathbf{h}_t^\top W^{Q^\top UK} \mathbf{c}_j^{KV}$$

$$\mathbf{u}_t = W^O \mathbf{o}_t = W^{O} W^{UV} \sum_{j=1}^t \text{Softmax}_j \left( \frac{\mathbf{q}_t^\top \mathbf{k}_j}{\sqrt{d_h}} \right) \mathbf{c}_j^{KV} = W^{OUV} \sum_{j=1}^t \text{Softmax} \left( \frac{\mathbf{q}_t^\top \mathbf{k}_j^{KV}}{\sqrt{d_h}} \right) \mathbf{c}_j^{KV}$$


% MHA

- $[\mathbf{q}_{t,1}; \mathbf{q}_{t,2}; ...; \mathbf{q}_{t,n_h}] = \mathbf{q}_t = W^{Q} \mathbf{h}_{t}$
- $[\mathbf{k}_{t,1}; \mathbf{k}_{t,2}; ...; \mathbf{k}_{t,n_h}] = \mathbf{k}_t = W^{K} \mathbf{h}_{t}$
- $[\mathbf{v}_{t,1}; \mathbf{v}_{t,2}; ...; \mathbf{v}_{t,n_h}] = \mathbf{v}_t = W^{V} \mathbf{h}_{t}$
- $\mathbf{o}_{t,i} = \sum_{j=1}^{t} \text{Softmax}_j\left(\frac{\mathbf{q}_{t,i} \mathbf{k}_{j,i}^\top}{\sqrt{d_h}}\right) \mathbf{v}_{j,i}$
    - $i$ 表示第 $i$ 个 head
    - $t$ 当时计算第 $t$ 个 token 的 attention
- $\mathbf{u}_t = W^O[\mathbf{o}_{t,1}; \mathbf{o}_{t,2}; ...; \mathbf{o}_{t,n_h}]$

上述公式是 q_len = 1 的场景，并且考察了多头，head 的下标用变量 $i$ 表示。

更简单的方式是去除多头的符号，因为 head，q_len 是可以并行的扩展。

- $\mathbf{o}_{t} = \sum_{j=1}^{t} \text{Softmax}_j\left(\frac{\mathbf{q}_{t} \mathbf{k}_{j}^\top}{\sqrt{d_h}}\right) \mathbf{v}_{j}$

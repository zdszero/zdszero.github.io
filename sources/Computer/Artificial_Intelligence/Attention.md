% Attention

### Attention

$Q, K, V = XW$ 

$Attention(Q, K, V) = \text{Softmax}(\frac{Q K^{T}}{\sqrt{d_k}}) V$

$Output = Attention(Q, K, V) W_{O}$

```python
import numpy as np
import torch
from einops import rearrange
from torch import nn

class SelfAttentionAISummer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_qvk = nn.Linear(dim, dim * 3, bias=False)
        self.scale_factor = dim ** -0.5  # 1/np.sqrt(dim)

    def forward(self, x, mask=None):
        assert x.dim() == 3, '3D tensor must be provided'

        qkv = self.to_qvk(x)  # [batch, tokens, dim*3 ]

        # decomposition to q,v,k
        # rearrange tensor to [3, batch, tokens, dim] and cast to tuple
        q, k, v = tuple(rearrange(qkv, 'b t (d k) -> k b t d ', k=3))

        # Resulting shape: [batch, tokens, tokens]
        scaled_dot_prod = torch.einsum('b i d , b j d -> b i j', q, k) * self.scale_factor

        if mask is not None:
            assert mask.shape == scaled_dot_prod.shape[1:]
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

        attention = torch.softmax(scaled_dot_prod, dim=-1)
        return torch.einsum('b i j , b j d -> b i d', attention, v)
```

### MultiHead Attention

![MultiHead Attention](/WikiImage/image_2024-11-04-19-49-23.png){ width=600px }

$MultiHead(Q, K, V) = Concat(head_1, \cdots, head_h) W_{o}$

$\text{ where } head_{i} = Attention(X W_{i}^{Q}, X W_{i}^{K}, X W_{i}^{V}) = Attention(Q_i, K_i, V_i)$

在多头注意力中，训练时会计算出更多的参数，用于表示不同的投影子空间。
但是在非多头注意力中，W 中只有一个投影子空间，所以捕获的特征更少。

```python
class MultiHeadSelfAttentionAISummer(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None):
        """
        Implementation of multi-head attention layer of the original transformer model.
        einsum and einops.rearrange is used whenever possible
        Args:
            dim: token's dimension, i.e. word embedding vector size
            heads: the number of distinct representations to learn
            dim_head: the dim of the head. In general dim_head<dim.
            However, it may not necessary be (dim/heads)
        """
        super().__init__()
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.heads = heads
        self.to_qvk = nn.Linear(dim, _dim * 3, bias=False)
        self.W_0 = nn.Linear( _dim, dim, bias=False)
        self.scale_factor = self.dim_head ** -0.5

    def forward(self, x, mask=None):
        assert x.dim() == 3
        qkv = self.to_qvk(x)  # [batch, tokens, dim*3*heads ]

        # decomposition to q,v,k and cast to tuple
        # the resulted shape before casting to tuple will be:
        # [3, batch, heads, tokens, dim_head]
        q, k, v = tuple(rearrange(qkv, 'b t (d k h) -> k b h t d ', k=3, h=self.heads))

        # resulted shape will be: [batch, heads, tokens, tokens]
        scaled_dot_prod = torch.einsum('b h i d , b h j d -> b h i j', q, k) * self.scale_factor

        if mask is not None:
            assert mask.shape == scaled_dot_prod.shape[2:]
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

        attention = torch.softmax(scaled_dot_prod, dim=-1)
        out = torch.einsum('b h i j , b h j d -> b h i d', attention, v)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.W_0(out)
```

### Intuition

![intuition](../../../docs/WikiImage/image_2025-02-08-17-06-30.png)

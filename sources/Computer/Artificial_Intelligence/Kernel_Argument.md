% Kernel Argument

### Sliding Window

In standard Transformer attention:

$$\text{Attn}(Q, K, V) = \text{softmax}!\left(\frac{QK^T}{\sqrt{d}}\right)V$$


__Window truncation = limited attention range__

$$\text{Attn}(q_t, K, V) = \text{softmax}!\left(\frac{q_t K_{[t-w:t]}^T}{\sqrt{d}}\right)V_{[t-w:t]}$$

where `w` is the **window size**.

> Q does *not* multiply with *all* K and V, only with a *subset* (recent ones).

Let’s say you have 10,000 tokens, but your **attention window = 512**.

Then when decoding token 10,001:

```
Q[10001] × K[9500:10000]^T
```

only involves the **last 512 keys**.

Older keys (`K[0:9499]`) are ignored — this is **window truncation**.

### Sink

Window truncation + sink tokens are used together in **long-context kernels**:
Then your attention pattern is:

$$Q_t \cdot [K_{\text{sink}}; K_{t-w:t}]$$

This allows **long-term memory (sink)** + **local context (window)** —
a hybrid of global and local attention.

### Combination

From an implementation view (e.g., in FlashInfer / DeepSeek MLA / Triton kernels):

* They **compute a masked QKᵀ** product using only valid indices.
* The **KV buffer** only holds `[sink_size + window_size]` entries.
* Older entries are either dropped or overwritten when the window slides.

So if you see something like this:

```python
launch_attention(q, k, v, sink=64, window=512)
```

The kernel will internally:

```cpp
effective_K = concat(K[0:64], K[prev-512:prev])
effective_V = concat(V[0:64], V[prev-512:prev])
```

and compute:

```cpp
scores = Q @ effective_K.T
```

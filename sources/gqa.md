% GQA

### 连续分组

GQA 按 **“连续块（contiguous block）”分组，而不是“交错（strided）分组”**

假设 q heads = 8，kv heads = 2，那么：

```
Q0,1,2,3 → KV0
Q4,5,6,7 → KV1
```

这种 连续分组设计，本质上是为 TP / 分布式推理服务的

```
KV0 ← Q0~3
KV1 ← Q4~7
```

这样可以直接 GPU0 计算 KV0，GPU1 计算 KV1 的部分。

### QKVParallelLinear

**核心逻辑**：在 weight load 的过程中就处理好每个 rank 需要的那一部分 qkv_proj 权重。

> QKVParallelLinear 在 weight_loader 中，将 fused QKV weight 按 [Q | K | V] 分段，对每一段分别根据 TP rank 和 KV replication 规则计算 shard_id，通过 narrow 截取对应 slice，并写入本 rank 的参数，从而构造出只包含本 rank 所需 Q/K/V 的局部 weight。

假设 q_heads=8，kv_heads=2，tp_size=2，那么：

```python
def weight_loader(...):
    # shard 逻辑：定义本 rank 应该拿多少 Q/K/V
    # 例如：
    # rank0:
    #     Q: 4 heads
    #     K: 1 head
    #     V: 1 head
    if loaded_shard_id == "q":
        shard_offset = 0
        shard_size = self.num_heads * self.head_size
    elif loaded_shard_id == "k":
        shard_offset = self.num_heads * self.head_size
        shard_size = self.num_kv_heads * self.head_size
    elif loaded_shard_id == "v":
        shard_offset = (self.num_heads + self.num_kv_heads) * self.head_size
        shard_size = self.num_kv_heads * self.v_head_size

    # 决定“拿哪一块”
    # 例如：rank 0 [ Q0 Q1 Q2 Q3 | K0 | V0 ]
    #       rank 1 [ Q4 Q5 Q6 Q7 | K1 | V1 ]
    if loaded_shard_id == "q":
        shard_id = self.tp_rank
    else:
        shard_id = self.tp_rank // self.num_kv_head_replicas

    # 切分 tensor
    start_idx = shard_id * shard_size
    loaded_weight = loaded_weight.narrow(
        output_dim, start_idx, shard_size
    )
    # 写入本 rank 的参数
    param_data.copy_(loaded_weight)
```

### 面试题：手写一个 GQA

#### 题目

请使用 PyTorch 实现一个简化版的 **Grouped-Query Attention（GQA）**。给定输入 `x: [B, S, D]`，其中 `B` 是 batch size，`S` 是序列长度，`D` 是隐藏维度；给定 query head 数 `h`、KV head 数 `h_kv` 以及每个 head 的维度 `d_qk`、`d_v`，完成一次 GQA 前向计算并返回 `out: [B, S, D]`。

要求：

1. 从一个 fused QKV 投影结果中按 `[Q | K | V]` 拆分出三部分，并 reshape 成多头张量。
2. `Q` 使用 `h` 个 head，`K/V` 只使用 `h_kv` 个 head；要求 `h` 能被 `h_kv` 整除。
3. 计算缩放点积注意力：`softmax(QK^T / sqrt(d_qk))V`，并支持可选的 attention mask。
4. 合并 head 后经过输出投影，返回与输入序列维度一致的结果。

#### 面试时应先说清楚的思路

GQA 的核心是：**Q head 比 K/V head 多，多个 Q head 共享同一个 K/V head**。为了让矩阵乘法的 head 维度对齐，可以把每个 K/V head 沿 head 维复制 `h // h_kv` 次。以 `h=8, h_kv=2` 为例，每个 KV head 被 4 个 Q head 共享；这能在尽量保留多头注意力表达能力的同时，减少 KV cache 的显存占用。

答题时建议按下面的张量形状逐步说明，而不是直接默写代码：

```text
Q: [B, S, h,    d_qk]
K: [B, S, h_kv, d_qk] -> repeat -> [B, S, h, d_qk]
V: [B, S, h_kv, d_v ] -> repeat -> [B, S, h, d_v ]
scores = Q @ K^T: [B, h, S, S]
out    = scores @ V: [B, S, h, d_v]
```

这里还要注意两个容易出错的点：`QK^T` 的最后两个序列维度要对齐，mask 必须在 softmax 前施加；最终输出维度应为 `h * d_v`，而不是 `h_kv * d_v`。

#### 常见追问

- MHA、MQA 和 GQA 的区别是什么？它们对 KV cache 的大小有什么影响？
- 为什么不能直接把 `K/V` 的 head 数改成 `h`？复制 K/V 会不会增加推理显存？如何避免物理复制？
- 自回归解码时，`Q` 的长度通常与 `K/V` 的长度有什么不同？代码中的两个序列维度应如何命名？
- 如果 `h` 不能被 `h_kv` 整除，应该在接口层限制、补齐，还是重新设计分组方式？

下面的代码是一个用于面试速写的简化参考实现。记忆几个点即可：

- 用 s 代表 q_len
- 用 t 代表 kv_len
- Q @ K 的 einsum 写作 `b s h d, b t h d -> b h s t`
- attn_scores @ V 的 einsum 写作 `b h s t, b t h d -> b s h d`
- attn_scores 计算记得 `/ (self.d_qk ** 0.5)`

完成前：

```python
class GQA(nn.Module):
    def __init__(self):
        super().__init__()

        # h: Query Head 数
        # h_kv: Key/Value Head 数
        # d: 每个 Head 的维度
        self.q_size = h * d
        self.k_size = h_kv * d
        self.v_size = h_kv * d

    def forward(self, x):
        # x: [B, S, D]
        # B: batch size
        # S: 当前 Q 的序列长度
        # D: hidden size
        B, S, D = x.shape

        # TODO: 完成剩下的步骤，返回 GQA 的 attention output
        # QKV projection 的权重可以通过 self.qkv_proj 访问
```



完成后：

```python
class GQA(nn.Module):
    def __init__(self):
        super().__init__()

        # h: Query Head 数
        # h_kv: Key/Value Head 数
        # d: 每个 Head 的维度
        self.q_size = h * d
        self.k_size = h_kv * d
        self.v_size = h_kv * d

    def forward(self, x):
        # x: [B, S, D]
        # B: batch size
        # S: 当前 Q 的序列长度
        # D: hidden size
        B, S, D = x.shape

        # 1. QKV 投影
        # Q: h 个 heads
        # K/V: h_kv 个 heads
        qkv, _ = self.qkv_proj(x)

        q, k, v = qkv.split(
            [self.q_size, self.k_size, self.v_size],
            dim=-1
        )

        # 2. 将 hidden dimension 拆成 [head, head_dim]
        q = q.reshape(B, S, h, d)
        k = k.reshape(B, S, h_kv, d)
        v = v.reshape(B, S, h_kv, d)

        # 3. GQA：每个 KV Head 被多个 Q Head 共享
        repeat_factor = h // h_kv

        # 注意：这里是逻辑上的 expand
        # 实际高性能实现通常不会真的复制 KV Cache
        k = k.repeat_interleave(repeat_factor, dim=2)
        v = v.repeat_interleave(repeat_factor, dim=2)

        # 4. Q @ K^T
        #
        # q: [B, S, h, d]
        # k: [B, S, h, d]
        #
        # 输出:
        # attn_scores: [B, h, S, S]
        attn_scores = torch.einsum(
            "b s h d, b t h d -> b h s t",
            q, k
        ) / (d ** 0.5)

        # 5. Softmax
        attn_scores = torch.softmax(attn_scores, dim=-1)

        # 6. Attention @ V
        #
        # attn_scores: [B, h, S, S]
        # v:           [B, S, h, d]
        #
        # 输出:
        # out: [B, S, h, d]
        out = torch.einsum(
            "b h s t, b t h d -> b s h d",
            attn_scores, v
        )

        # 7. 合并 Head，并进行输出投影
        out = out.reshape(B, S, h * d)
        out = self.o_proj(out)

        return out
```


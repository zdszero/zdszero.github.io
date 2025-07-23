% Naming

<b><u>nope/rope</u></b>

deepseek 中使用的简写

- `k_nope`：no positional encoding，不进行位置编码
- `k_pe`：positional encoding，进行位置编码

<b><u>Feature</u></b>


$$ \mathbf{y} = \mathbf{x}W^T + \mathbf{b} $$

* `in_features`: 输入向量的维度（如词向量、隐藏状态）
* `out_features`: 输出向量的维度（如变换后的隐藏状态）

举例：

```python
linear = nn.Linear(in_features=768, out_features=3072)
```

表示你输入一个长度为 768 的向量，它会变成一个长度为 3072 的向量。

<b><u>World</u></b>

`world_size` 是 **分布式训练中的术语**，它指的是：

> **训练过程中参与计算的总进程数（或 GPU 数）**

在 PyTorch 的分布式并行框架中，“世界”（`world`）表示所有设备组成的集合。

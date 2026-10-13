% MLP

在深度学习（特别是 Transformer 架构，如 LLaMA、Mistral 等现代大模型）中，MLP（多层感知机）模块通常有两种主要变体。

如果对 `ReLU`、`GELU`、`SiLU`、`SwiGLU` 或 Kimi K3 使用的 `SiTU` 不熟悉，可以先阅读[激活函数](../activation/)章节。需要特别区分：`SiLU` 是单输入标量函数，而 `SwiGLU`、`GeGLU` 和 `SiTU` 是包含 gate/up 两路和逐元素乘法的门控 MLP 结构。

### MLP 结构变体

1.  **标准 MLP (Standard MLP / Vanilla MLP)**：即你描述的 "up 然后 down"。
2.  **门控 MLP (Gated MLP / SwiGLU / GeGLU / SiTU)**：即你描述的 "up and gate 然后 down"。这种结构通常包含两个并行的升维投影，其中一个通过激活函数或平滑限幅作为“门”来控制另一个。

假设输入向量为 \(x\)，两种模式的数学公式如下：

### 标准 MLP

这是最传统的结构，包含一个线性升维层（Up），一个非线性激活函数，和一个线性降维层（Down）。

**公式：**
\[ \text{Output} = W_{\text{down}} \cdot \sigma(W_{\text{up}} \cdot x + b_{\text{up}}) + b_{\text{down}} \]

**其中：**
*   \(x \in \mathbb{R}^{d_{model}}\)：输入向量。
*   \(W_{\text{up}} \in \mathbb{R}^{d_{ff} \times d_{model}}\)：升维权重矩阵（Up 投影）。
*   \(b_{\text{up}}\)：升维偏置（有时省略）。
*   \(\sigma(\cdot)\)：激活函数（传统上用 ReLU，现代模型常用 GeLU 或 SiLU/Swish）。
*   \(W_{\text{down}} \in \mathbb{R}^{d_{model} \times d_{ff}}\)：降维权重矩阵（Down 投影）。
*   \(b_{\text{down}}\)：降维偏置。
*   \(d_{ff}\)：隐藏层维度（通常是 \(d_{model}\) 的 4 倍）。

---

### 门控 MLP

这种结构（如 SwiGLU）引入了门控机制。它有两个并行的升维投影：一个用于计算值（Value），另一个用于计算门（Gate）。门的输出经过激活函数后，与值的输出进行**逐元素相乘（Hadamard product）**，然后再通过 Down 投影。

**公式：**
\[ \text{Output} = W_{\text{down}} \cdot \left( \sigma(W_{\text{gate}} \cdot x + b_{\text{gate}}) \odot (W_{\text{up}} \cdot x + b_{\text{up}}) \right) + b_{\text{down}} \]

*(注：在某些实现如 LLaMA 中，可能没有偏置项 \(b\))*

**其中：**
*   \(W_{\text{up}}\) 和 \(W_{\text{gate}}\)：两个不同的升维权重矩阵，形状均为 \(\mathbb{R}^{d_{ff} \times d_{model}}\)（或者为了参数平衡，每个矩阵的宽度可能是标准 MLP 的一半，但在逻辑上它们共同构成了扩展空间）。
*   \(W_{\text{gate}} \cdot x\)：门控路径的线性变换。
*   \(W_{\text{up}} \cdot x\)：数值路径的线性变换。
*   \(\sigma(\cdot)\)：激活函数（在 SwiGLU 中通常是 **SiLU** 或 **Swish**，即 \(\sigma(z) = z \cdot \text{sigmoid}(z)\)；在 GeGLU 中是 GeLU）。
*   \(\odot\)：逐元素乘法（Hadamard product）。
*   \(W_{\text{down}}\)：降维权重矩阵。

---

在代码实现（如 PyTorch）中，门控 MLP 的核心部分通常长这样：

```python
# 标准 MLP
hidden = silu(self.up_proj(x)) # 或者 relu/gelu
output = self.down_proj(hidden)

# 门控 MLP (SwiGLU)
gate = self.gate_proj(x)
value = self.up_proj(x)
hidden = silu(gate) * value  # 门控机制核心
output = self.down_proj(hidden)
```


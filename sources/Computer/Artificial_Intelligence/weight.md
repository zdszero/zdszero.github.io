% weight

### w1, w2, w3

在 MLP（前馈网络，FFN）里，通常有 2 或 3 个权重矩阵：

* **两路 (GPT-2 风格)**

  $$
  \text{FFN}(x) = \text{GeLU}(x W_1 + b_1) W_2 + b_2
  $$

* **三路 (Gated FFN, 也叫 SwiGLU/GeGLU，LLaMA / DeepSeek 等用)**

  $$
  \text{FFN}(x) = (\text{GeLU}(x W_1) \;\; \odot \;\; x W_3) W_2
  $$

  * `W1`：生成激活（通常过 GeLU/SiLU）
  * `W3`：生成门控向量（gating）
  * `W2`：down projection，把大维度投影回隐藏层

在 checkpoint 里，就会看到：

* `w1_weight`
* `w2_weight`
* `w3_weight`

### w13

* 在一些实现里，作者把 `w1` 和 `w3` 合并存成一个张量：

  $$
  W_{13} = \text{concat}(W_1, W_3, \text{dim}=\text{output})
  $$
* 在 forward 里，再一次性算：

  $$
  [h_1, h_3] = x \cdot W_{13}
  $$
* 然后拆开 `h1` 和 `h3` 分别进入 GeLU 和 gate。
* 这样比单独调用两次 matmul 更高效。

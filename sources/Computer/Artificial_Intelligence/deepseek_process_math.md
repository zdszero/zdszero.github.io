% deepseek process math

### Attention

**Pre-Norm**

$$x_1 = \text{Norm}(x)$$

__QKV Projection__

$$Q = x_1 W_Q,\quad K = x_1 W_K,\quad V = x_1 W_V$$

__Attention Softmax__

$$A = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right)$$

__Attention Output__

$$\text{attn_out} = A V$$

__Residual Add__

$$x_2 = x + \text{attn_out}$$


### MOE

__Pre NORM__

$$x_3 = \text{Norm}(x_2)$$

---

__Gate, Softmax, Top-K__

对每个 token：


$$p_i = \text{Softmax}(x_3 W_g)$$

再做 top-k：

$${(e_{i,j}, \alpha_{i,j})}_{j=1}^k$$

__Dispatch__

$$x_3 \rightarrow x_3^{(e)}$$

__Group GEMM__

每个 expert：

$$\text{FFN}_e(h) = W_2^{(e)} , \sigma(W_1^{(e)} h)$$

$$
\text{moe_out}*i = \sum*{j=1}^{k}
\alpha_{i,j} \cdot \text{FFN}*{e*{i,j}}(x_{3,i})
$$

__Residual Add__

$$
\boxed{
x_4 = x_2 + \text{moe_out}
}
$$


### Next

下一层一开始又是：


$$x_{\text{next}} = \text{Norm}(x_4)$$

—— 周而复始。

### Summary

在 **DeepSeek 的 MoE Transformer 中：

**Norm 只出现在：**

1. Attention 之前（Pre-Norm）
2. MoE 之前（Pre-Norm）
3. 下一层开始（也是 Pre-Norm）

**Softmax 只出现在：**

1. Attention 的 `QK^T` 上
2. MoE 的 gating 路由上

**Residual Add 只出现在：**

1. Attention 输出后：`x + attn_out`
2. MoE combine 后：`x₂ + moe_out`

---

```text
x
→ Norm → Attention → Residual
→ Norm → Gate Softmax → Dispatch → Expert GEMM → Combine → Residual
→ (进入下一层)
```


> Norm 永远在“子层之前”，
> 
> Softmax 只在“Attn 和 Gate”，
>
> Residual 永远在“子层之后”。

哪些地方有 **Norm / Softmax / Residual**

| 阶段                   | 是否做 **Norm** | 是否做 **Softmax** | 是否做 **Residual Add** | 说明             |
| -------------------- | ------------ | --------------- | -------------------- | -------------- |
| 进入 Attention 之前      | ✅            | ❌               | ❌                    | Pre-Norm       |
| Attention 内部（QK）     | ❌            | ✅               | ❌                    | 只在 `QK^T` 后    |
| Attention 输出后        | ❌            | ❌               | ✅                    | `x + attn_out` |
| 进入 MoE 之前            | ✅            | ❌               | ❌                    | 第二个 Pre-Norm   |
| MoE gating           | ❌            | ✅               | ❌                    | top-k 归一化      |
| MoE FFN (group_gemm) | ❌            | ❌               | ❌                    | 纯 GEMM + SiLU  |
| MoE combine 后        | ❌            | ❌               | ✅                    | `x + moe_out`  |
| 下一层开始                | ✅            | ❌               | ❌                    | 下一层 Pre-Norm   |



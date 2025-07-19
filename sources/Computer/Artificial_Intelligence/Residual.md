% Residual

### 什么是 Residual

$$
\mathbf{h}_t^l = \underbrace{\mathbf{u}_t^l}_{\text{input}} + \underbrace{\mathrm{FFN}(\mathrm{LayerNorm}(\mathbf{u}_t^l))}_{\text{残差（residual）路径}}
$$

* **$\mathbf{u}_t^l$** 是 residual 的主路径（skip connection）；
* **$\mathrm{FFN}(\mathrm{LayerNorm}(\cdot))$** 是变换路径（learned delta）；
* **整个式子形成了 residual connection（残差连接）**。

---

✅ 重点澄清

| 项目                                                                  | 是否算 residual？ | 说明                            |
| ------------------------------------------------------------------- | ------------- | ----------------------------- |
| $\mathrm{LayerNorm}(\mathbf{u}_t^l)$                                | ❌             | 只是做标准化，属于变换的一部分               |
| $\mathrm{FFN}(\cdot)$                                               | ❌             | 是学习的变换部分                      |
| $\mathbf{u}_t^l + \mathrm{FFN}(\mathrm{LayerNorm}(\mathbf{u}_t^l))$ | ✅             | 整体构成 residual block           |
| “加号右边”整个部分                                                          | ✅（广义上）        | 它是 residual block 中的“残差项”或变化量 |

---

📐 __残差连接的标准定义是：__

> 给定一个输入 $\mathbf{x}$，网络只学习一个变换 $F(\cdot)$，最终输出为：
>
> $$
> \text{output} = \mathbf{x} + F(\mathbf{x})
> $$
>
> 这个 $F(\cdot)$ 就叫做 **残差函数（residual function）**

在你的例子中，残差函数就是：

$$
F(\mathbf{u}_t^l) = \mathrm{FFN}(\mathrm{LayerNorm}(\mathbf{u}_t^l))
$$

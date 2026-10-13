% Linear Attention

### Softmax Attention

\[
\begin{aligned}
\boldsymbol{q}_i, \boldsymbol{k}_i, \boldsymbol{v}_i, \boldsymbol{o}_i &\in \mathbb{R}^{d \times 1} \\
\boldsymbol{Q} &= [\boldsymbol{q}_1, \boldsymbol{q}_2, \cdots, \boldsymbol{q}_n]^\top \in \mathbb{R}^{n \times d} \\
\boldsymbol{K} &= [\boldsymbol{k}_1, \boldsymbol{k}_2, \cdots, \boldsymbol{k}_n]^\top \in \mathbb{R}^{n \times d} \\
\boldsymbol{V} &= [\boldsymbol{v}_1, \boldsymbol{v}_2, \cdots, \boldsymbol{v}_n]^\top \in \mathbb{R}^{n \times d} \\
\boldsymbol{O} &= [\boldsymbol{o}_1, \boldsymbol{o}_2, \cdots, \boldsymbol{o}_n]^\top \in \mathbb{R}^{n \times d} \\[6pt]
\text{注意力分数矩阵：} & \quad \boldsymbol{S} = \frac{\boldsymbol{Q} \boldsymbol{K}^\top}{\sqrt{d}} \in \mathbb{R}^{n \times n} \\[2pt]
\text{注意力权重矩阵：} & \quad \boldsymbol{A} = \mathrm{softmax}(\boldsymbol{S}) \in \mathbb{R}^{n \times n} \\[2pt]
\text{输出矩阵：} & \quad \boldsymbol{O} = \boldsymbol{A} \boldsymbol{V} \in \mathbb{R}^{n \times d}
\end{aligned}
\]

计算量为 \(BT^{2}HD\)，主要在于计算注意力得分矩阵的过程中 \(QK\) 时 q_len == kv_len = T，计算结果的 shape 为 \([B, H, T, T]\)

### Linear Attention

#### insight

根据矩阵乘法结合率，如果去掉 Softmax 操作的话，可以使用 \(QK^{T}V = Q(K^{T}V)\)，这样就降低了计算复杂度。

考虑完整的 MHA，\(Q, K, V\) 的 shape 都是 \([B, T, H, D]\)，当我们计算 \(K^{T} V\) 时：

- \(B, H\) 是 batching dimension
- \(D\) 是 free dimension
- \(T\) 是 contracting dimension

\(K^{T} V\) 计算结果的 shape 为 \([B, H, D, D]\)，计算量为 \(BTHD^2\)

所以如果使用这种方案的话，计算复杂度从 \(T^2\) 变成了 \(D^2\)

#### recursion

\[o_{t}=\sum_{j=1}^{t} v_{j}\left(k_{j}^{\top} q_{t}\right)=\sum_{j=1}^{t}\left(v_{j} k_{j}^{\top}\right) q_{t}=\left(\sum_{j=1}^{t} v_{j} k_{j}^{\top}\right) q_{t}\]

如果我们记括号部分为 \(S_t\)，那么有

\[o_{t}=S_{t} q_{t}, \quad S_{t}=S_{t-1}+v_{t} k_{t}^{\top}\]

使用这种方式后，Causal 格式的 Attention 体现为一个以 \(S_{t}\) 为 State 的 RNN。

但是这种最简单的递归方式本质上就是一个 cumsum，所有历史信息都等权地叠加，不难想象当叠加的 token 足够多时，每个 token 的信息占比都会变得极小，无法重建信息。

所以问题在于如何在递归的过程可以动态地赋予 \(v_{j} k_{j}^{\top}\)

#### evolution

| 模型/方法             | 公式                                                                 |
|-----------------------|----------------------------------------------------------------------|
| Softmax Attention     | \((\exp(QK^\top) \odot M)V\)                                          |
| 最早的线性Attention   | \((QK^\top \odot M)V\)                                                |
| 加入遗忘门后          | \((QK^\top \odot \Gamma)V\)                                           |
| DeltaNet              | \((QK^\top \odot M)(I + KK^\top \odot M^-)^{-1}V\)                    |
| Gated DeltaNet        | \(((QK^\top \odot M)(I + KK^\top \odot M^-)^{-1} \odot \Gamma)V= (QK^\top \odot \Gamma)(I + KK^\top \odot \Gamma^-)^{-1}V\) |

\[
\Gamma_{i,j} = 
\begin{cases}
\displaystyle \prod_{\tau=j+1}^{i} \gamma_\tau, & i > j \\
1, & i = j \\
0, & i < j
\end{cases}
\]


---

🧩 统一递归主公式（涵盖所有变体）

\[
\boxed{
\begin{aligned}
\hat{v}_t &= k_t \, S_{t-1} &&\text{（根据当前键 }k_t\text{ 从记忆中读取预测值）}\\
\Delta_t &= v_t - \hat{v}_t &&\text{（计算当前真实值与预测值的误差）}\\
S_t &= \gamma_t S_{t-1} + \alpha_t \, k_t^\top v_t + \beta_t \, k_t^\top \Delta_t &&\text{（统一状态更新）}\\
o_t &= q_t \, S_t &&\text{（用查询 }q_t\text{ 读取最终输出）}
\end{aligned}
}
\]

其中：

- \(\gamma_t\)：**遗忘门标量**（对旧状态做衰减）；
- \(\alpha_t\)：**直接写入系数**（控制是否直接把 \(k^\top v\) 塞进记忆）；
- \(\beta_t\)：**误差写入系数**（控制把“预测误差”写回记忆的强度，通常是归一化因子如 \(\frac{1}{1 + k_t k_t^\top}\)）。

📊 各变体如何通过调节 \((\gamma, \alpha, \beta)\) 退化得到

| 模型/方法 | \(\gamma_t\) | \(\alpha_t\) | \(\beta_t\) | 代入主公式后的实际更新（退化结果） |
| :--- | :--- | :--- | :--- | :--- |
| **最早的线性Attention** | \(1\) | \(1\) | \(0\) | \(S_t = S_{t-1} + k_t^\top v_t\)（纯累加，无纠错） |
| **加入遗忘门后** | \(\gamma_t\) | \(1\) | \(0\) | \(S_t = \gamma_t S_{t-1} + k_t^\top v_t\)（带衰减的累加） |
| **DeltaNet** | \(1\) | \(0\) | \(\beta_t\) | \(S_t = S_{t-1} + \beta_t k_t^\top (v_t - k_t S_{t-1})\)（纯误差驱动，不直接写原值） |
| **Gated DeltaNet** | \(\gamma_t\) | \(0\) | \(\beta_t\) | \(S_t = \gamma_t S_{t-1} + \beta_t k_t^\top (v_t - k_t S_{t-1})\)（遗忘门 + 误差驱动，完美融合） |

💡 这个“整体公式”的灵魂解读

把主公式拆开看，其实就干了三件事：

1. **读（Read）**：\( \hat{v}_t = k_t S_{t-1} \) —— 先看看记忆里对当前 token 怎么看；
2. **算误差（Error）**：\(\Delta_t = v_t - \hat{v}_t\) —— 看看记忆猜得准不准；
3. **写（Write）**：  
   - **直接写** \(\alpha_t k_t^\top v_t\)：不管三七二十一，把原始信息硬写进去（早期的笨办法）；  
   - **纠错写** \(\beta_t k_t^\top \Delta_t\)：只把没猜对的那部分“补”进去（DeltaNet 的精髓）；  
   - **遗忘** \(\gamma_t S_{t-1}\)：写之前先把旧记忆打个折。

> **演化史的数学本质**：从 \(\alpha=1\)（只信当前输入）进化到 \(\alpha=0, \beta>0\)（只信预测误差），再引入 \(\gamma\)（学会遗忘），这就是这条技术路线的全部秘密。而你之前看到的复杂逆矩阵 \((I + KK^\top \odot M)^{-1}\)，其实就是这个递归公式在**多步展开**后的闭合矩阵解（为了保证数值稳定性而做的归一化）。

### DeltaNet

这段代码想实现的是 **DeltaNet / delta-rule linear attention**：

\[
S_t =
S_{t-1} +
\beta_t k_t^\top
\left(
v_t-k_tS_{t-1}
\right)
\]

然后用 query 从状态矩阵中读取：

\[
o_t=q_tS_t
\]

它的核心思想不是简单地把 \(k_t^\top v_t\) 累加进记忆，而是先读取旧值，再只写入“误差”：

\[
\Delta v_t=v_t-\hat v_t,
\qquad
\hat v_t=k_tS_{t-1}
\]

\[
S_t=S_{t-1}+\beta_t k_t^\top\Delta v_t
\]


```py
def forward(self, x, mask=None, cache=None):
    # x: [b, t, d]
    b, t, d = x.shape
    h = self.num_heads
    d_head = d // h

    qkv = self.qkv_proj(x)

    q = qkv[:, :, :d].view(b, t, h, d_head).transpose(1, 2)
    k = qkv[:, :, d:2 * d].view(b, t, h, d_head).transpose(1, 2)
    v = qkv[:, :, 2 * d:].view(b, t, h, d_head).transpose(1, 2)

    # [b, h, t, d_head]
    q = F.normalize(F.silu(q), dim=-1)
    k = F.normalize(F.silu(k), dim=-1)

    # Shared across heads: [b, 1, t, 1]
    beta = torch.sigmoid(self.w_beta(x)).view(b, 1, t, 1)

    if cache is None:
        S = x.new_zeros(b, h, d_head, d_head)
    else:
        S = cache

    outputs = []

    for i in range(t):
        # [b, h, 1, d_head]
        q_i = q[:, :, i:i + 1, :]
        k_i = k[:, :, i:i + 1, :]
        v_i = v[:, :, i:i + 1, :]
        beta_i = beta[:, :, i:i + 1, :]

        # Read current prediction:
        # [b, h, 1, d_head]
        v_old = k_i @ S

        # Delta / prediction error:
        # [b, h, 1, d_head]
        delta = beta_i * (v_i - v_old)

        # Outer-product state update:
        # [b, h, d_head, 1] @ [b, h, 1, d_head]
        # -> [b, h, d_head, d_head]
        S = S + k_i.transpose(-1, -2) @ delta

        # Read from the updated state:
        # [b, h, 1, d_head]
        o_i = q_i @ S
        outputs.append(o_i)

    # [b, h, t, d_head]
    o = torch.cat(outputs, dim=2)

    # [b, t, d]
    o = o.transpose(1, 2).contiguous().view(b, t, d)

    return self.o_proj(o), S
```


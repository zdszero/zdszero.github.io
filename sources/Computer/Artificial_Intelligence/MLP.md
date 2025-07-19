% MLP

### 两种 Norm

__Pre-Norm 变体（常用）：__

$$
\mathbf{h}_t^l = \mathbf{u}_t^l + \mathrm{FFN}(\mathrm{LayerNorm}(\mathbf{u}_t^l))
$$

__Post-Norm 变体（原始 Transformer）：__

$$
\mathbf{h}_t^l = \mathrm{LayerNorm}\left( \mathbf{u}_t^l + \mathrm{FFN}(\mathbf{u}_t^l) \right)
$$

### 对比 MoE

对一个 token 的表示 $\mathbf{u}_t^l \in \mathbb{R}^d$，标准的前馈网络（Feed-Forward Network，FFN）结构如下：

$$
\mathbf{h}_t^l = \mathrm{FFN}\left( \mathbf{u}_t^l \right) + \mathbf{u}_t^l
$$

- $\mathbf{u}_t^l$：第 $l$ 层中 token $t$ 的输入向量，它是该层 attention 子层的输出。$\boxed{\mathbf{u}_t^l = \mathrm{LayerNorm}\left(\mathbf{x}_t^l + \mathrm{Attention}(\mathbf{x}_t^l)\right)}$
- $\mathbf{h}_t^l$：第 $l$ 层的最后输出。

其中 FFN 通常包含两层线性变换 + 非线性激活，例如：

$$
\mathrm{FFN}(\mathbf{x}) = W_2 \, \sigma(W_1 \mathbf{x} + b_1) + b_2
$$

* 所有 token 都用**相同的** FFN 参数（$W_1, W_2, b_1, b_2$）
* 所有 token 的完整 hidden vector 都被处理
* 没有「专家选择」的概念

$$
\mathbf{h}_t^l = \sum_{i=1}^{N} \left( g_{i,t} \, \mathrm{FFN}_i\left( \mathbf{u}_t^l \right) \right) + \mathbf{u}_t^l
$$

这个公式表示：

* 每个 token 的 $\mathbf{u}_t^l$ 会被送到多个专家（Top-K）
* 每个专家都有**独立的 FFN 参数**
* $g_{i,t}$ 是门控权重，只保留 top-k 个专家，其余为 0（稀疏）
* 最终将这些专家的输出加权求和，并加上 residual

🆚 __MoE vs MLP：逐点对比__

| 项目   | 普通 MLP             | MoE                       |
| ---- | ------------------ | ------------------------- |
| 参数共享 | 所有 token 共享一个 FFN  | 每个专家有独立 FFN，token 动态选择    |
| 计算开销 | 所有 token 都计算一次 FFN | 仅计算 Top-K 个专家，节省计算        |
| 表达能力 | 有限（全局统一）           | 更强（不同 token 用不同专家）        |
| 路由策略 | 固定                 | 动态（token-level routing）   |
| 稀疏性  | 无稀疏性               | 引入稀疏性（仅部分专家激活）            |
| 通信   | 无                  | 多专家调度可能涉及跨设备通信（尤其在分布式训练中） |

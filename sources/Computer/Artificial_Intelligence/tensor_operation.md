% Tensor Operation


| 函数                              | 用途                |
| ------------------------------- | ----------------- |
| `torch.tensor()`                | 从 Python 数据构建张量   |
| `torch.arange()`                | 类似 `range()`，连续整数 |
| `torch.linspace()`              | 从 a 到 b 等间隔划分     |
| `x.view()` / `.reshape()`       | 改变形状              |
| `x.permute()`                   | 维度换位              |
| `x.unsqueeze()` / `.squeeze()`  | 增减维               |
| `x.transpose()`                 | 转置两个维度            |
| `torch.cat()` / `torch.stack()` | 拼接张量              |
| `x.expand()` / `x.repeat()`     | 广播 vs 复制          |
| `torch.where()`                 | 条件选择              |
| `x.mean() / x.sum()`            | 求均值 / 和           |

### dtype

- FP8: `e4m3fn`
    - `e4`: Exponent（指数）4 位
    - `m3`: Mantissa（尾数/有效位）3 位
    - `fn`：finite + no subnormals，只有有限值，不包含非规格化数（subnormal）或 NaN

### einsum

- 矩阵乘法：`torch.einsum("ij,jk->ik", A, B)`
- 向量内积：`torch.einsum("i,i->", a, b)`
- Batch 矩阵乘法：`torch.einsum(bij,bjk->bik, A, B)`

### tensor operation

- reinterpret dimension
    - `reshape(dim1, dim2, ...)`
    - `view(dim1, dim2, ...)`
- shift dimensions
    - `permute(dim1, dim2, ...)`：任意维度重排
    - `transpose(dim1, dim2)`：交换两个维度
- split dimension
    - `split([split1, split2, ...], dim)`
- add/remove dimension
    - `unsqueeze(dim)`：在指定维度插入一个维度
    - `squeeze(dim)`：删除某个维度，只有在 dim 对应的维度大小为 1 时才会生效

### special syntax

- `-1`: 自动推导
- `:`：当前维度
- `...`：表示所有其他维度
- `None`：新增一个维度，类似于 `squeeze()`
- `[[index1], [index2]]`：高级索引

% deepseek model


### DeepSeek

#### config

- `q_lora_rank`: 1536
    - Q 的 MLA 的压缩维度
    - 开发者在实现各种参数高效微调方法时，可能会沿用 Q-LoRA 的命名规范，以表明这是一个与“低秩”、“量化”和“参数高效”相关的维度。
- `kv_lora_rank`：512
    - KV 的压缩维度和 Q 不同，MLA 主要应用在这里
- `qk_nope_head_dim`：128
- `v_head_dim`：128
    - q 和 k 中的每个头中不施加 rope 的 dim
    - v 的 dim 与 qk 不施加 rope 的保持相同
- `qk_rope_head_dim`：64
- `num_attention_heads`：128
- `num_key_value_heads`：128
    - 采用 128 个头的设计
    - Q, K, V 的头数都相同
- `n_routed_experts`: 8
    - moe 每层选 8 个专家
- `n_shared_experts`: 1
    - 每层有一个共享专家，所以每层专家总数为 257
- `intermediate_size`: 18432 = 2048 * 9
    - 前三层（9*`moe_intermediate_size`）
- `moe_intermediate_size`: 2048
    - 路由专家 MLP 中间维度

#### weights

可以通过如下代码打印权重：

```python
from safetensors import safe_open

# path = "model-00001-of-000163.safetensors"
path = "model-00100-of-000163.safetensors"

with safe_open(path, framework="pt", device="cpu") as f:
    for k in f.keys():
        tensor = f.get_tensor(k)
        print(f"{k}: shape={tensor.shape}, dtype={tensor.dtype}")
```

需要注意下表的权重行列是反过来的：

```python
input_layernorm.weight torch.Size([7168]) bfloat16
# out [ mlp_intermidiate_size, hidden_size ]
mlp.down_proj.weight torch.Size([7168, 18432]) fp8
mlp.down_proj.weight_scale_inv torch.Size([56, 144])

# in1, in2 [ hidden_size, mlp_intermidiate_size ]
mlp.gate_proj.weight torch.Size([18432, 7168])
mlp.gate_proj.weight_scale_inv torch.Size([144, 56])
mlp.up_proj.weight torch.Size([18432, 7168])
mlp.up_proj.weight_scale_inv torch.Size([144, 56])
post_attention_layernorm.weight torch.Size([7168])
self_attn.kv_a_layernorm.weight torch.Size([512])

# [ hidden_size, kv_lora_rank + qk_rope_head_dim ]
self_attn.kv_a_proj_with_mqa.weight torch.Size([576, 7168])
self_attn.kv_a_proj_with_mqa.weight_scale_inv torch.Size([5, 56])
# [ kv_lora_rank, 2 * num_key_value_heads * (kv)]
self_attn.kv_b_proj.weight torch.Size([32768, 512])
self_attn.kv_b_proj.weight_scale_inv torch.Size([256, 4])
self_attn.o_proj.weight torch.Size([7168, 16384])
self_attn.o_proj.weight_scale_inv torch.Size([56, 128])
self_attn.q_a_layernorm.weight torch.Size([1536])

# [ hidden_size, q_lora_rank ]
self_attn.q_a_proj.weight torch.Size([1536, 7168])
self_attn.q_a_proj.weight_scale_inv torch.Size([12, 56])
# [ q_lora_rank, num_attention_heads * (qk_nope_head_dim + qk_rope_head_dim) ]
self_attn.q_b_proj.weight torch.Size([24576, 1536])
self_attn.q_b_proj.weight_scale_inv torch.Size([192, 12])
```

专家权重如下：

```python
layers.3-60.mlp.(experts.0-255|shared_experts).down_proj.weight: shape=torch.Size([7168, 2048]), dtype=torch.float8_e4m3fn
layers.3-60.mlp.(experts.0-255|shared_experts).down_proj.weight_scale_inv: shape=torch.Size([56, 16]), dtype=torch.float32
layers.3-60.mlp.(experts.0-255|shared_experts).gate_proj.weight: shape=torch.Size([2048, 7168]), dtype=torch.float8_e4m3fn
layers.3-60.mlp.(experts.0-255|shared_experts).gate_proj.weight_scale_inv: shape=torch.Size([16, 56]), dtype=torch.float32
layers.3-60.mlp.(experts.0-255|shared_experts).up_proj.weight: shape=torch.Size([2048, 7168]), dtype=torch.float8_e4m3fn
layers.3-60.mlp.(experts.0-255|shared_experts).up_proj.weight_scale_inv: shape=torch.Size([16, 56]), dtype=torch.float32
```

### mla

### experts weights

前 3 层为 dense 层，后 58 层为稀疏 moe 层。

__对于 dense 层__

dense 层固定选择 8 个专家

权重大小为 7168 * 2048 * 3 * 9 * 3 = 1,189,085,184

__对于稀疏 moe__

256 专家，hidden size 为 7168，每个 token 从 256 专家中选择 8 个

每个专家为 3 路 MLP，7168 → 2048 → 7168

- 每个专家大小为：2048 * 7168 * 3
- router 大小为：7168 * 256 + 256

所以后面 58 层 MOE 所占的权重大小为：(2048 * 7168 * 3 * 257 + 7168 * 256 + 256) * 58 = 656,569,547,264

总共的专家大小为 656569547264 + 1189085184 = 657,758,632,448

差不多是 658 B

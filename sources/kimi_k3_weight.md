% Kimi K3 参数量与权重存储精算

### Config

| 配置项 | 数值 | 计数意义 |
| :--- | ---: | :--- |
| `hidden_size` | 7168 | 主干维度 H |
| `num_hidden_layers` | 93 | 层列表为 1-based；权重名为 0-based |
| KDA / MLA 层数 | 69 / 24 | MLA 为 4、8、…、92 及 93 层 |
| `first_k_dense_replace` / `moe_layer_freq` | 1 / 1 | 第 1 层 dense，其余 92 层 MoE |
| `intermediate_size` | 33792 | Dense FFN 中间维度 |
| `num_experts` / `num_experts_per_token` | 896 / 16 | 每层路由专家总数 / 每 token 选中数 |
| `num_shared_experts` | 2 | 合并中间维度 2 × 3072 = 6144 |
| `routed_expert_hidden_size` | 3584 | 路由专家的 latent 维度 D |
| `moe_intermediate_size` | 3072 | 每个专家中间维度 M |
| KDA heads × head dim | 96 × 128 = 12288 | Q/K/V/G/O 投影宽度 |
| MLA Q / KV rank | 1536 / 512 | Q、KV 低秩维度 |
| MLA nope / rope / V dim | 128 / 64 / 128 | `mla_use_nope=true` 不代表删除 checkpoint 的 rope 维权重 |
| `vocab_size` / `tie_word_embeddings` | 163840 / false | Embedding 和 LM head 分别计数 |
| `num_nextn_predict_layers` | 0 | 不计 MTP 层 |

### 精算总览

| 模块 | 逻辑参数量 |
| :--- | ---: |
| 69 层 KDA 注意力（含其内部 norm / conv） | 30,618,088,704 |
| 24 层 MLA 注意力（含 Q/KV norm） | 5,572,706,304 |
| 93 层公共 norm 与 Attention Residual 投影 | 3,999,744 |
| 1 层 Dense FFN | 726,663,168 |
| 92 层共享专家 | 12,155,092,992 |
| 92 层路由器（含 correction bias） | 590,955,008 |
| 92 层 latent 上下投影与 norm | 4,727,310,336 |
| 92 × 896 个路由专家 | 2,722,740,830,208 |
| Embedding、LM head 与输出 norm / residual | 2,348,831,744 |
| 语言模型合计 | 2,779,484,478,208 |
| Vision tower | 401,214,464 |
| 多模态 projector | 46,144,512 |
| 完整多模态模型合计 | 2,779,931,837,184 |

完整模型为 **2.779931837 T**（2779.931837 B）；语言模型为 **2.779484478 T**。

单 token 的语言模型激活参数口径为：所有公共权重加每层 16 个路由专家（包含完整 embedding / LM head 表的常见架构统计口径）：

`P_active = P_text − 92 × (896 − 16) × (3 × 3584 × 3072) = 105,364,019,968 ≈ 105.364020 B`。

这不是实际逐 token 读取量或 FLOPs：embedding 只查一行，注意力还受序列长度影响；视觉参数是否执行取决于输入。

| 存储类别 | 逻辑参数量 | 权重 Bytes（含 MXFP4 scale） |
| :--- | ---: | ---: |
| 路由专家 MXFP4 | 2,722,740,830,208 | 1,446,456,066,048 |
| 语言模型其他 BF16 | 56,732,525,568 | 113,465,051,136 |
| 语言模型 FP32 | 11,122,432 | 44,489,728 |
| 语言模型合计 | 2,779,484,478,208 | 1,559,965,606,912 |
| 视觉与 projector（按 BF16） | 447,358,976 | 894,717,952 |
| 完整多模态模型合计 | 2,779,931,837,184 | 1,560,860,324,864 |

混合精度完整权重约 **1560.860 GB / 1453.664 GiB**。若统一按 BF16 理论折算，则为 `5,559,863,674,368` Bytes；不能用 4 bit 直接乘完整模型参数量。

### 分模块张量明细

MLA、KDA 两表均包含每层的 6 个公共向量（合计 43,008 参数 / 86,016 Bytes）；总览中已单独提取，不能重复相加。FFN 表另计。

### Layer 0 MLP

| 权重名称 | 形状 | 数据类型 | 参数量 | 字节数 (Bytes) |
| :--- | :--- | :--- | :--- | :--- |
| layers.0.mlp.down_proj.weight | [7168, 33792] | BF16 | 242,221,056 | 484,442,112 |
| layers.0.mlp.gate_proj.weight | [33792, 7168] | BF16 | 242,221,056 | 484,442,112 |
| layers.0.mlp.up_proj.weight | [33792, 7168] | BF16 | 242,221,056 | 484,442,112 |

### MLA


| 参数字段 | 张量形状 | 数据类型 | 参数量 | 总字节数 |
| :--- | :--- | :--- | :--- | :--- |
| input_layernorm.weight | [7168] | BF16 | 7,168 | 14,336 |
| mlp_res_norm.weight | [7168] | BF16 | 7,168 | 14,336 |
| mlp_res_proj.weight | [1, 7168] | BF16 | 7,168 | 14,336 |
| post_attention_layernorm.weight | [7168] | BF16 | 7,168 | 14,336 |
| self_attention_res_norm.weight | [7168] | BF16 | 7,168 | 14,336 |
| self_attention_res_proj.weight | [1, 7168] | BF16 | 7,168 | 14,336 |
| self_attn.g_proj.weight | [12288, 7168] | BF16 | 88,080,384 | 176,160,768 |
| self_attn.kv_a_layernorm.weight | [512] | BF16 | 512 | 1,024 |
| self_attn.kv_a_proj_with_mqa.weight | [576, 7168] | BF16 | 4,128,768 | 8,257,536 |
| self_attn.kv_b_proj.weight | [24576, 512] | BF16 | 12,582,912 | 25,165,824 |
| self_attn.o_proj.weight | [7168, 12288] | BF16 | 88,080,384 | 176,160,768 |
| self_attn.q_a_layernorm.weight | [1536] | BF16 | 1,536 | 3,072 |
| self_attn.q_a_proj.weight | [1536, 7168] | BF16 | 11,010,048 | 22,020,096 |
| self_attn.q_b_proj.weight | [18432, 1536] | BF16 | 28,311,552 | 56,623,104 |
|                                 |               |          |            | 464,478,208 |

### KDA

每个张量在 KDA 计算流程里的位置和作用（`b_proj` 是写入强度 β，`f_a/f_b_proj`、`dt_bias`、`A_log` 合成逐通道遗忘门 α，`g_proj` 是输出门），见 [Linear Attention 的 Kimi K3 一节](./linear_attention.md)。

| 参数字段                        | 张量形状      | 数据类型 | 参数量     | 总字节数    |
| :---                            | :---          | :---     | :---       | :---        |
| input_layernorm.weight          | [7168]        | BF16     | 7,168      | 14,336      |
| mlp_res_norm.weight             | [7168]        | BF16     | 7,168      | 14,336      |
| mlp_res_proj.weight             | [1, 7168]     | BF16     | 7,168      | 14,336      |
| post_attention_layernorm.weight | [7168]        | BF16     | 7,168      | 14,336      |
| self_attention_res_norm.weight  | [7168]        | BF16     | 7,168      | 14,336      |
| self_attention_res_proj.weight  | [1, 7168]     | BF16     | 7,168      | 14,336      |
| self_attn.A_log                 | [128]         | F32      | 128        | 512         |
| self_attn.b_proj.weight         | [96, 7168]    | BF16     | 688,128    | 1,376,256   |
| self_attn.dt_bias               | [12288]       | F32      | 12,288     | 49,152      |
| self_attn.f_a_proj.weight       | [128, 7168]   | BF16     | 917,504    | 1,835,008   |
| self_attn.f_b_proj.weight       | [12288, 128]  | BF16     | 1,572,864  | 3,145,728   |
| self_attn.g_proj.weight         | [12288, 7168] | BF16     | 88,080,384 | 176,160,768 |
| self_attn.k_conv1d.weight       | [12288, 1, 4] | F32      | 49,152     | 196,608     |
| self_attn.k_proj.weight         | [12288, 7168] | BF16     | 88,080,384 | 176,160,768 |
| self_attn.o_norm.weight         | [128]         | F32      | 128        | 512         |
| self_attn.o_proj.weight         | [7168, 12288] | BF16     | 88,080,384 | 176,160,768 |
| self_attn.q_conv1d.weight       | [12288, 1, 4] | F32      | 49,152     | 196,608     |
| self_attn.q_proj.weight         | [12288, 7168] | BF16     | 88,080,384 | 176,160,768 |
| self_attn.v_conv1d.weight       | [12288, 1, 4] | F32      | 49,152     | 196,608     |
| self_attn.v_proj.weight         | [12288, 7168] | BF16     | 88,080,384 | 176,160,768 |
|                                 |               |          |            | 887,886,848 |

### MoE

每个 MoE 层的共享专家仍在 7168 维主干空间运行；896 个路由专家均在 3584 维 latent 空间运行。上下投影和 latent norm 每层各一份，不随专家数重复。

- 单个路由专家：`3 × 3584 × 3072 = 33,030,144` 逻辑参数。
- 每层全部路由专家：`896 × 33,030,144 = 29,595,009,024` 参数。
- 每层共享专家：`3 × 7168 × 6144 = 132,120,576` 参数。
- 每层 latent 投影及 norm：`2 × 7168 × 3584 + 3584 = 51,383,808` 参数。
- 每层路由器：`896 × 7168 + 896 = 6,423,424` 参数。
- 每层 MoE 合计：**29,784,936,832 参数**。

下表 `weight_packed` 的“参数量”为 **U8 存储元素数**，并非逻辑参数量；每个 U8 存两个 FP4 权重。scale 不计入上述逻辑参数量。


| 参数字段 | 张量形状 | 数据类型 | 参数量 | 总字节数 |
| :--- | :--- | :--- | :--- | :--- |
| shared_experts.down_proj.weight | [7168, 6144] | BF16 | 44,040,192 | 88,080,384 |
| shared_experts.gate_proj.weight | [6144, 7168] | BF16 | 44,040,192 | 88,080,384 |
| shared_experts.up_proj.weight | [6144, 7168] | BF16 | 44,040,192 | 88,080,384 |
| | | | | 264,241,152 |
| gate.e_score_correction_bias | [896] | F32 | 896 | 3,584 |
| gate.weight | [896, 7168] | BF16 | 6,422,528 | 12,845,056 |
| routed_expert_down_proj.weight | [3584, 7168] | BF16 | 25,690,112 | 51,380,224 |
| routed_expert_norm.weight | [3584] | BF16 | 3,584 | 7,168 |
| routed_expert_up_proj.weight | [7168, 3584] | BF16 | 25,690,112 | 51,380,224 |
| | | | | 115,616,256 |
| experts.0.w1.weight_packed | [3072, 1792] | U8 | 5,505,024 | 5,505,024 |
| experts.0.w1.weight_scale | [3072, 112] | U8 | 344,064 | 344,064 |
| experts.0.w2.weight_packed | [3584, 1536] | U8 | 5,505,024 | 5,505,024 |
| experts.0.w2.weight_scale | [3584, 96] | U8 | 344,064 | 344,064 |
| experts.0.w3.weight_packed | [3072, 1792] | U8 | 5,505,024 | 5,505,024 |
| experts.0.w3.weight_scale | [3072, 112] | U8 | 344,064 | 344,064 |
| | | | | 17,547,264 |

单个专家的 packed 数据为 16,515,072 Bytes，scale 为 1,032,192 Bytes，合计 17,547,264 Bytes。每层 896 个专家为 15,722,348,544 Bytes；加共享专家、路由器和 latent 投影后，每层 MoE 为 **16,102,205,952 Bytes**。

### ETC

| 权重名称 | 形状 | 数据类型 | 参数量 | 字节数 (Bytes) |
| :--- | :--- | :--- | :--- | :--- |
| embed_tokens.weight | [163840, 7168] | BF16 | 1,174,405,120 | 2,348,810,240 |
| model.norm.weight | [7168] | BF16 | 7,168 | 14,336 |
| model.output_attn_res_norm.weight | [7168] | BF16 | 7,168 | 14,336 |
| model.output_attn_res_proj.weight | [1, 7168] | BF16 | 7,168 | 14,336 |
| lm_head.weight | [163840, 7168] | BF16 | 1,174,405,120 | 2,348,810,240 |


### 视觉模块与多模态投影

依照本地 `kimi_k3_vl.py` 的构造：视觉注意力内部宽度是 `qkv_hidden_size=1536`，不是 `vt_hidden_size=1024`；视觉 MLP 为两矩阵 GELU MLP，不按三矩阵 gated FFN 计算。配置关闭 attention、linear、patch projection bias。

| 模块 | 公式 | 参数量 |
| :--- | :--- | ---: |
| Patch embed | 1024 × 3 × 14 × 14 | 602,112 |
| 空间位置嵌入 | 64 × 64 × 1024 | 4,194,304 |
| 每层视觉 QKV + O | 3 × 1536 × 1024 + 1024 × 1536 | 6,291,456 |
| 每层视觉 MLP | 2 × 1024 × 4096 | 8,388,608 |
| 每层两组 RMSNorm | 2 × 1024 | 2,048 |
| 27 层视觉 encoder | 27 × 14,682,112 | 396,417,024 |
| 视觉 final RMSNorm | 1024 | 1,024 |
| Vision tower 合计 | 上述非重复项之和 | 401,214,464 |
| Projector 第一矩阵 | 4096 × 4096 | 16,777,216 |
| Projector 第二矩阵 | 7168 × 4096 | 29,360,128 |
| Projector post RMSNorm | 7168 | 7,168 |
| Projector 合计 | 16,777,216 + 29,360,128 + 7,168 | 46,144,512 |

2 × 2 patch 合并得到 4096 维输入。时间位置编码为固定 sin/cos、非持久 buffer，不计参数；时间池化也不增加权重。视觉与 projector 合计 **447,358,976 参数**。

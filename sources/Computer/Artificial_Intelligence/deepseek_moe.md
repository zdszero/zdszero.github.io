% deepseek moe

$$
\begin{aligned}
U &= X W_1 \\
G &= X W_3 \\
H &= \text{SiLU}(U) \odot G \\
Y &= H W_2
\end{aligned}
$$

其中

- $W_1$: up
- $W_3$: gate
- $W_2$: down

### Process Optimization

1. Concat $W_1$ and $W_3$, put it in one gemm:

$$[U | G] = X \cdot [W_1 | W_3]$$

```py
deep_gemm.m_grouped_fp8_gemm_nt_masked(
    global_input_tokens_tuple, 
    (self.w13_weight, scale_b), 
    fake_tensors.fake_gateup_output, 
    recv_count,
    expected_m,
)
```

2. kernel fusion: SiLU + elementwise mul + Quantization

$$H = \text{SiLU}(U) \odot G$$

$$\boxed{\text{SiLU}(x) = x \cdot \sigma(x) = x \cdot \frac{1}{1 + e^{-x}}}$$

```py
speedgate.silu_and_mul_ep_index_quant_3d(
    fake_tensors.fake_gateup_output, 
    self.w2_input_scale.to(torch.bfloat16),
    recv_count, 
    fake_tensors.fake_per_token_row_fp8_input, 
    fake_tensors.fake_per_token_row_fp8_scale,
    0
)
```


### Weight

| 变量         | 数学含义 | 物理含义  | 规模               |
| ------------ | ----     | -------   | ------------------ |
| `w13_weight` | `W1`     | `W3`      |  **2 × d_ff × d_model** |
| `w2_weight`  | `W2`     | down 投影 | **d_model × d_ff** |

- `w_13`: 2 × 7168 × (9 × 2048)
    - ``
- `w_2`: 7168 × (9 × 2048)
    - `hidden_size * num_experts * moe_intermediate_size`
- `w_qa`: 7168 × 1536
    - `hidden_size * qk_lora_rank`
- `w_kva`: 7168 × 512
    - `hidden_size * v_lora_rank`
- `w_(qa, kva)`: 7168 × 2048 
- `w_(qb)`: 1536 × 128 × 192
- `absorb q_nope with W_(kb)`: 128 × 128 × 512

### Computation

- O Proj: `bs * num_attention_heads * v_head_dim * hidden_size`
    - W_13: `bs * 8 `

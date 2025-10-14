n_heads = 128
q_lora_rank = 1536
kv_lora_rank = 512
qk_nope_head_dim = 128
qk_rope_head_dim = 64
v_head_dim = 128
qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
hidden_size = 7168

def flops_vanilla(q_len, kv_len):
    return (
        q_len * hidden_size * q_lora_rank + # from Q to c_q
        q_len * q_lora_rank * n_heads * qk_head_dim + # from c_q to q_nope and q_pe
        kv_len * kv_lora_rank * n_heads * qk_head_dim + # from c_k to k_nope and k_pe
        n_heads * (q_len * kv_len * qk_head_dim + q_len * kv_len * v_head_dim) + # MHA
        q_len * n_heads * v_head_dim * hidden_size # o_proj
    )

def flops_absorb(q_len, kv_len):
    return (
        q_len * hidden_size * q_lora_rank + # from Q to c_q
        q_len * q_lora_rank * n_heads * qk_rope_head_dim + # from c_q to q_pe, using wq_b_rope
        q_len * q_lora_rank * kv_lora_rank * n_heads + # from c_q to q_nope, using wq_b_nope * wk_b_nope^T
        n_heads * (q_len * kv_len * (qk_rope_head_dim + kv_lora_rank) + q_len * kv_len * kv_lora_rank) + # MQA
        q_len * n_heads * kv_lora_rank * hidden_size # o_proj wv_b_nope^T
    )

# Abosrb martix:
# [q_lora_rank, n_heads, qk_nope_head_dim] * [kv_lora_rank, n_heads, qk_nope_head_dim]
# [q_lora_rank, kv_lora_rank, n_heads]

kv_len = 10000
print(f"prefill: absorb vs vanilla ratio  ~ {flops_absorb(kv_len, kv_len) / flops_vanilla(kv_len, kv_len)}")
print(f"decode: absorb vs vanilla ratio  ~ {flops_absorb(1, kv_len) / flops_vanilla(1, kv_len)}")

n_heads = 128
q_lora_rank = 1536
kv_lora_rank = 512
qk_nope_head_dim = 128
qk_rope_head_dim = 64
v_head_dim = 128
qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
hidden_size = 7168

def flops_vanilla(q_len, kv_len, verbose=False):
    all_ops = {
        'flops_q_a': q_len * hidden_size * q_lora_rank, # from Q to c_q
        'flops_kv_a': kv_len * hidden_size * (kv_lora_rank + qk_rope_head_dim), # from KV to c_kv and k_pe
        'flops_q_b': q_len * q_lora_rank * n_heads * qk_head_dim, # from c_q to q_nope and q_pe
        'flops_k_b': kv_len * kv_lora_rank * n_heads * qk_nope_head_dim, # from c_kv to k_nope
        'flops_v_b': kv_len * kv_lora_rank * n_heads * v_head_dim, # from c_kv to v_dim
        'flops_mha': n_heads * (q_len * kv_len * qk_head_dim + q_len * kv_len * v_head_dim), # MHA
        'flops_oproj': q_len * n_heads * v_head_dim * hidden_size # o_proj
    }
    total_flops = sum(v for v in all_ops.values())
    if verbose is True:
        print('-' * 20)
        print('Vanilla FLOPS for each step:')
        for k, v in all_ops.items():
            print(f'{k}: {v} (%{v/total_flops*100:.2f})')
        print('Total:', total_flops)
    return total_flops


def flops_absorb(q_len, kv_len, verbose=False):
    all_ops = {
        'flops_q_a': q_len * hidden_size * q_lora_rank, # from Q to c_q
        'flops_kv_a': kv_len * hidden_size * (kv_lora_rank + qk_rope_head_dim), # from KV to c_kv and k_pe
        'flops_q_b_rope': q_len * q_lora_rank * qk_rope_head_dim * n_heads, # from c_q to q_pe
        'flops_q_b_nope_k_b': q_len * q_lora_rank * kv_lora_rank * n_heads, # from c_q to q_nope, using wq_b_nope * wk_b^T
        'flops_mqa': n_heads * (q_len * kv_len * (qk_rope_head_dim + kv_lora_rank) + q_len * kv_len * kv_lora_rank), # MQA
        'flops_oproj_v_b': q_len * n_heads * kv_lora_rank * hidden_size # o_proj wv_b^T
    }
    total_flops = sum(v for v in all_ops.values())
    if verbose is True:
        print('-' * 20)
        print('Absorb FLOPS for each step:')
        for k, v in all_ops.items():
            print(f'{k}: {v} (%{v/total_flops*100:.2f})')
        print('Total:', total_flops)
    return total_flops

# Absorb wk_b in wq_b (nope part):
# [q_lora_rank, n_heads, qk_nope_head_dim] * [kv_lora_rank, n_heads, qk_nope_head_dim]
# [q_lora_rank, kv_lora_rank, n_heads]
#
# Absorb wv_b in o_proj:
# [v_head_dim, n_heads, hidden_size] * [kv_lora_rank, n_heads, v_head_dim]
# [n_heads, hidden_size, kv_lora_rank]

kv_len=10000
verbose=True
print(f"prefill: absorb vs vanilla ratio  ~ {flops_absorb(kv_len, kv_len, verbose) / flops_vanilla(kv_len, kv_len, verbose)}")
print(f"decode: absorb vs vanilla ratio  ~ {flops_absorb(1, kv_len, verbose) / flops_vanilla(1, kv_len, verbose)}")

% deepseek profile

### H800

D4 Profile

| 层级                  | 模块                     | Kernel                          | h800 ds prof | b 200 d1 mtp on |
|-----------------------|--------------------------|---------------------------------|--------------|-----------------|
|                       | 端到端                   |                                 | 96ms         |                 |
|                       | 61层执行耗时             |                                 | 87.32ms      |                 |
| attention层           | self.input_layernorm     | flashinfer::norm::FusedAddRM    | 4            |                 |
| attention层           | q_a_kv_a                 | deep_gemm::sm100_fp8_gem        | 13           |                 |
| attention层           | self.input_layernorm     | rmsnorm_split_col_kernel        | 2            |                 |
| attention层           | q_proj_b                 | per_token_group_quant_fp8_      | 2            |                 |
| attention层           | q_proj_b                 | deep_gemm::sm100_fp8_gem        | 19           |                 |
| attention层           | torch.bmm transpose(0,1) | nvjet_tst_256x128_64x5_2x1      | 16           |                 |
| attention层           | self.rotary_emb          | elementwise_kernel              |              |                 |
| attention层           | self.rotary_emb          | rotary_embedding_kernel         | 7            |                 |
| attention层           | self.rotary_emb          | unrolled_elementwise_kernel     |              |                 |
| attention层           | self.rotary_emb          | index_elementwise_kernel        |              |                 |
| attention层           | self.attn_mqa            | mhaSm100fKernel_Qkv             | 255          | 225.767         |
| attention层           | torch.bmm transpose(0,1) | nvjet_tst_64x128_64x13_2x1      | 15           |                 |
| attention层           | self.o_proj              | elementwise_kernel              |              |                 |
| attention层           | self.o_proj              | per_token_group_quant_fp8_      | 5            |                 |
| attention层           | self.o_proj              | deep_gemm::sm100_fp8_gem        | 50           | 43.512          |
| attention层           | sum                      |                                 | 388          |                 |
| gate层                | self.input_layernorm     | flashinfer::norm::FusedAddRM    | 3            |                 |
| gate层                |                          | nvjet_tst_64x32_64x16_2x4_2     | 6            |                 |
| gate层                |                          | cublasLt::splitKreduce_kernel   | 2            |                 |
| gate层                |                          | distribution_elementwise_grid   |              |                 |
| gate层                |                          | topk                            | 8            |                 |
| gate层                | sum                      |                                 | 19           |                 |
| moe层                 | dispatch                 | deep_ep::internode_ll::dispatch | 187          | 48.212          |
| moe层                 | up_gate_gemm             | deep_gemm::sm100_fp8_gem        | 81           | 195.136         |
| moe层                 | up_gate_gemm             | vectorized_elementwise_kernel   |              |                 |
| moe层                 | silu                     | silu_and_mul_kernel_ep_index    | 8            | 40.491          |
| moe层                 | down_gemm                | deep_gemm::sm100_fp8_gem        | 39           | 104.671         |
| moe层                 | combine                  |                                 | 394          |                 |
| moe层                 | sum                      |                                 |              |                 |
| combine/share overlap |                          | deep_ep::internode_ll::combine  |              |                 |
| combine/share overlap |                          | per_token_group_quant_fp8_      | 3            |                 |
| combine/share overlap |                          | deep_gemm::sm100_fp8_gem        | 17           |                 |
| combine/share overlap |                          | act_and_mul_kernel              | 4            |                 |
| combine/share overlap |                          | deep_gemm::sm100_fp8_gem        | 10           |                 |
| combine/share overlap |                          | deep_ep::internode_ll::combine  |              |                 |

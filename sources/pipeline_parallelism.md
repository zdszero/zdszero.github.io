% PP

## Workflow

### Example

PP4 async depth = 0

```
PP0  mb0  mb1  mb2  mb3
PP1       mb0  mb1  mb2  mb3
PP2            mb0  mb1  mb2  mb3
PP3                 mb0  mb1  mb2  mb3
```


PP4 async depth = 2

```
PP0  mb0  mb1  mb2  mb3  mb4  mb5
PP1       mb0  mb1  mb2  mb3  mb4  mb5
PP2            mb0  mb1  mb2  mb3  mb4  mb5
PP3                 mb0  mb1  mb2  mb3  mb4  mb5
```

以 PP1 处理 mb1 和 mb2 的流程为例：

| prev mb            | current mb                     | next mb                | next_first_rank_mb                          |
| ------------------- | ---------------------------- | ----------------------- | -------------------------------------------- |
|                     | `recv_meta(mb1)`             |                         |                                              |
| `commit_meta(mb0)`  |                              |                         |                                              |
|                     | `prefill_chunk(mb1)`         |                         |                                              |
|                     | `get_new_prefill_batch(mb1)` |                         |                                              |
|                     | `recv_hs(mb1)`               |                         |                                              |
|                     |                              |                         | `commit_output(next_first) if async_depth > 0`  |
| `commit_hs(mb0)` |                              |                         |                                              |
|                     | `launch(mb1)`                |                         |                                              |
|                     |                              |                         | `commit_output(next_first) if async_depth == 0` |
|                     |                              |                         | `send_consensus(next_first)`                    |
|                     |                              |                         | `commit_consensus(next_first)`                  |
|                     |                              |                         | `commit_release(next_first)`                    |
|                     |                              | `harvest_wait(mb2)`     |                                              |
|                     |                              | `harvest(mb2)`          |                                              |
|                     |                              | `process_inflight(mb2)` |                                              |
|                     | `send_req(mb1)`              |                         |                                              |
|                     | `send_hs(mb1)`               |                         |                                              |

### async depth

如果 async depth == 0，那么在 launch 下一个 loop 的 current_mb 时，必须同步等待上一个 loop 的 current_mb 完成。


## PP 和 TP 对比

讨论 PP 和 TP 方案在 Prefill 阶段的优劣

- PP 相比 TP
    - 优势：
        - 可以减少通信操作
            - AllReduce/Dispatch/Combine
        - 某些 kernel 计算效率更高
            - fused_moe_w2
        - 不同卡不存在重复计算
            - q_a_and_kv_a_proj
            - fused_moe
                - moe_gate
                - expert_up_proj
                - expert_down_proj
        - 可以做多机部署
    - 劣势
        - 不同卡不存在对于单一 batch 的并行计算，TTFT 天然大
        - 不同卡是弱同步状态，稳定性建设难度更大
            - 不同 rank 的 prefix cache 命中同步
            - 请求被 abort 的时候如何维持稳定性

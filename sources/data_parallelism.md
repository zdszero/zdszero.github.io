% DP

## DP 对比 TP

- TP
    - 优势
        - weight 可以切分
            - QKVProj, OProj
    - 劣势
        - 面对 MLA, GQA 可能存储 replicate KV cache，浪费显存
        - 可能存在重复计算
            - MLA q_a_and_kv_a_proj
- DP

### TP Size 选取

在 Prefill 阶段使用不同 TP Size 最后对极限性能会有怎样的影响？

#### Normal MoE

**那么 dp attention 何时有收益呢？**

首先，这个需要看你是否使用的是正常 MoE，如果是正常的 MoE，损耗会比较大，因为在 TP4, TP8 的 attention 之后需要增加一个 ReduceScatter，然后再运行 dispatch/combine 的 EP 模式的 MoE，这一步是额外的损耗。

如果使用 FusedMoE，那个差别其实不那么大，因为 TP1, TP4, TP8 在 attention 过后，fused moe 之前，都需要做一个 AllReduce。

> 注意在 sglang 的 TP8 DP8 的时候是通过 AllReduce 操作来模拟 AllGather，这样可以统一 DP1, DP2, DP4, DP8 的代码逻辑。

#### Fused MoE

这一节来讨论 dp attention 对于 prefill 阶段极限吞吐的影响。

首先控制变量，保证单卡 attention / moe  计算量一致的情况下 TP1 / TP4 进行对比。

后文给出三种 TP 情况下 Kimi k25 一层的时间（使用 Fused MoE + 不同 tp size 的 attention）进行性能对比。

- TP1: chunk size = 8k
- TP4: chunk size = 32k
- TP8: chunk size = 64k

三种情况下都是用 8k 的请求打满，保证每张卡的 attention 和 moe 计算量一致。

在该场景下，TP1 极限吞吐相比 TP4 提升约 7%，相比 TP8 提升约 12%

---

观察数据分析可知：

* attention MHA 和 moe 的 w13/w2 算子时间基本一致
* 通信时间基本一致
    * 由于使用 fused moe，TP1 的 attention 之后也要做 all gather，TP4/TP8 的 attention 之后直接做 8 卡的 all reduce，这个步骤两者通信量一致。当然 sglang 的实现逻辑，在 dp_gather 中使用 all_reduce 模拟了 all_gather。
* 差异在于小算子，TP4 的一些小算子需要额外计算：
    * q_a_and_kv_a_proj：在 MHA 的 qkv head 切分之前，每个 rank 重复计算，4 倍差距
    * norm
        * q_norm和 k_norm：4 倍差距
        * attention 之后的 fused_add_norm：8 倍差距。
        * MoE 之后的 fused_add_norm：4 倍差距。
    * 一些其他小算子也存在额外计算
        * set_mla_kv_buffer
        * dp_scatter 中的 trtion_mem_cpy 和 local_tokens.fill_(0)

### 不同 Attention 冗余显存分析

在 decode 阶段 dp attention 需要根据场景分析，如果不开启 dp attention，那么你不同卡的显存可能没有办法充分利用，导致 bs 打不上去，从而导致吞吐不高。

#### MLA & MQA

首先，MLA 在 decode 阶段使用 mqa，kv heads 数量为 1，在只有 1 个 dp 的情况下，相当于每张卡都要保存相同的 kv cache。

所以显存实际利用率不高，你的实际 bs 打不上去。

#### GQA

所以如果想要提升 decode 阶段的吞吐，需要保证不同卡能存储不同 kv head 的 cache，这样不重复存储相同的 kv cache，显存利用率才高。

考虑到现在一般大模型的 kv heads 数量都为 2 或 4，所以在这种情况下设置 decode attn 阶段的 dp_size = 8 // kv_heads，吞吐相比于 dp8 的模式不会明显下滑。

#### 其他损耗

其实损耗就是跟 prefill 阶段分析一致了：

- 小算子是否有重复计算
- 是否会额外 ReduceScatter 的通信（后面是正常 MoE 还是 FusedMoE）


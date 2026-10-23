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

## Fake-DP：一个入口下的多组独立模型副本

SGLang 当前仓库中的 `--fake-dp N`（字段名 `fake_dp_size`，默认 1）表示：在同一个 DataParallelController / HTTP 服务入口后面，启动 N 组独立模型实例。**`--fake-dp 2` 就是两组完整模型副本，每组可以再用 TP / CP / EP 组织自己的 GPU。** “完整”指每组 GPU 合起来包含完整模型，而非每张 GPU 都保存完整权重。

这里的 fake 指外层用 controller 做类似 DP 的请求分发，内部保持独立实例的执行方式；模型副本和 GPU 资源分配都是真实的。

### 部署示例：8 卡运行两套 TP4

单机、PP=1、默认连续 GPU 编号时：

```bash
--fake-dp 2 --tp-size 4
```

```text
                    一个 HTTP 服务入口
                            │
                   DataParallelController
                      请求分发 / 路由
                      ┌─────┴─────┐
                      ▼           ▼
                 Replica 0    Replica 1
                 GPU 0～3     GPU 4～7
                 完整模型 A    完整模型 B
                 独立 TP4     独立 TP4
                 独立 KV      独立 KV
```

两组会分别加载模型权重，分别维护 scheduler、batch、GPU KV cache 和通信组。两组之间不进行模型 forward 的 collective。一个请求通常路由到其中一组，由该组执行；该请求不会因为 fake-DP=2 自动使用全部 8 张 GPU。PD 分离部署时，Prefill 与 Decode 各自仍按其部署拓扑执行和传输 KV。

在上述 PP=1 场景，总 GPU 数为 `fake_dp_size × tp_size`。每个 TP4 副本都必须独立容纳完整模型及运行资源，不能依靠另一个副本补足权重或显存。若每组同时配置 CP4 / EP4，这些通常是在同一组 4 张卡上组织不同算子的并行维度，不是再将 GPU 数乘以 4 和 4。

### 与 DP-attention 的区别

| 项目 | Fake-DP，例如 fake-DP2 × TP4 | DP-attention，例如全局 TP8 / attention DP2 |
|---|---|---|
| 模型组织 | 两组完整且独立的模型副本 | 同一全局模型通信世界中划分 attention DP 组 |
| 内部 DP | 预期每个副本使用实际 DP=1 的路径 | attention 按配置划分 DP 组 |
| 通信范围 | TP / CP / EP 各自在副本内部 | attention 分组执行；MoE 等部分仍可能跨 attention DP 组通信 |
| 调度与 GPU KV | 每个副本独立维护 | 随 attention DP 组及算子布局组织 |
| 请求分发 | 外层 controller 选择一个副本 | controller 向相应 DP worker 分发 |
| 主要用途 | 多副本扩容，同时保留单实例执行路径 | 调整 attention 与 MoE 等模块的并行布局、计算和 KV 冗余 |

这里比较的是 **fake-DP 与开启 DP-attention 的部署**；普通 DP 多副本本身也可以是独立实例，不能把所有 `--dp-size` 用法都理解为同一个全局通信世界。

Fake-DP 的配置意图是保持实际 `dp_size=1`，由 `fake_dp_size` 决定外层副本数。它不是把 `--fake-dp 2` 和 `--dp-size 2` 相乘得到四组副本；当前 controller 使用两者的最大值计算副本数。评估该功能时，应按其预期独立副本配置使用。

### 启动和请求路由

启动时，engine 发现 `fake_dp_size > 1` 后创建 DataParallelController。controller 走独立副本启动路径，为每组分配独立的分布式通信端口和 GPU 区间，再分别启动组内 scheduler 进程、加载模型。对外仍共用服务入口和 tokenizer / detokenizer IPC 通道。

请求分发取决于负载均衡配置，可以轮询、按请求数或 token 负载选择副本。在 PD 分离的 `follow_bootstrap_room` 模式下，controller 按 `bootstrap_room % 副本数` 选择目标，以匹配 KV bootstrap 的副本路由。当前 fake-DP 路径会忽略外部 `routed_dp_rank` 的直接指定，避免外部默认 rank=0 导致所有请求都落到副本 0。

GPU KV 池彼此独立；如果启用 AttentionStore 等外部缓存，仍可能复用同一外部缓存服务或命名空间。当前实现用副本 dp_id 区分注册信息，因此“独立 GPU KV”不代表外部缓存服务也必须部署两套。

### 性能含义

Fake-DP 提供多组模型并行服务的能力，主要影响总吞吐、请求排队和并发容量。它不会直接把一个请求分摊给两个副本，也不能保证吞吐线性翻倍。是否优于单组 TP8，取决于 TP4 的单组执行效率、模型能否装下、KV 容量、请求负载和路由均衡。

例如 TP4 能装下模型且单组效率较高时，两组 TP4 可能比单组 TP8 更适合处理大量独立请求；若权重或运行资源必须使用 8 卡才能容纳，则不能用两组 TP4 替代。

### 源码定位

以上基于分析时的本地仓库实现：

- `python/sglang/srt/arg_groups/fields/parallel.py`：`fake_dp_size` 定义与 `--fake-dp` 别名。
- `python/sglang/srt/entrypoints/engine.py`：`_launch_scheduler_processes()` 判断是否创建 controller。
- `python/sglang/srt/managers/data_parallel_controller.py`：副本数计算、`launch_dp_schedulers()`、GPU 分配和请求分发。
- `python/sglang/srt/disaggregation/common/conn.py`：对外公布 fake-DP 副本数，供 KV bootstrap 路由使用。
- `python/sglang/srt/managers/scheduler.py`：AttentionStore 注册时使用 controller 分配的副本 dp_id。

相关概念：[TP](./tensor_parallelism.md)、[EP](./expert_parallelism.md)、[PD 分离](./disaggregation.md)、[KV Cache](./kv_cache_size.md)。

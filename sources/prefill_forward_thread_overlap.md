% Disagg Prefill：Forward 线程与组 Batch Overlap

### 优化结论与适用范围

本文分析 commit `c833093f7c3797eab3f9e98f1a3187ccfb997f04`（`feat(prefill): thread-based forward overlap for disagg prefill (gated)`），以该提交时的代码为准。

核心变化是把当前 batch 的 `run_batch(N)` 移到独立 worker 线程，让 scheduler 主线程提前接收请求、组建下一批 `N+1`。与此同时，上一批的 `process_batch_result(N-1)` 也可以与当前 forward 重叠。

这项优化用于 **PD 分离的 Prefill 实例、PP=1、已开启 enable_overlap** 的路径。它没有改写原有 overlap 循环，而是增加一个默认关闭的新循环。这里的“组 batch”指 admission / 调度准备，包括请求筛选、chunk 推进、缓存与资源准备等，不只是拼接一个 Python 列表。

### 先区分两种 overlap

`run_batch()` 是 CPU 上的一次函数调用，其中包含 forward 准备、模型执行调用、CUDA 工作提交和结果搬运安排。**函数返回不等于 GPU forward 已经完成。** CUDA stream/event 负责约束设备端依赖，后续结果处理在需要时等待结果就绪。

因此需要区分：

- **原有 CPU–GPU overlap**：scheduler 提交 CUDA 工作后，GPU 可以继续计算，CPU 继续处理调度。
- **新增线程 overlap**：即使 `run_batch()` 还没有返回，scheduler 主线程也能执行其他工作。它隐藏的是原来被 `run_batch()` 占住的主线程时间。

不能把提交前的 overlap 循环理解为“所有 CPU 和 GPU 工作都完全串行”，也不能把 worker collect 理解为“无条件同步整块 GPU”。

### 四种模式对比

| 模式 | run_batch 在哪里执行 | 下一批 admission 能否与尚未返回的 run_batch 并发 | 结果处理顺序 |
|---|---|---|---|
| 普通循环，enable_overlap 关闭 | scheduler 主线程 | 否 | 当前批 run → 当前批 process |
| 原有 overlap 循环 | scheduler 主线程 | 否；但仍可能与已提交的 GPU 工作重叠 | 当前批 run → 上一批 process → 当前批 sample |
| 新线程循环，AHEAD 关闭 | worker；提交后立即 collect | 否 | 延续原有流水顺序，验证 off-thread 执行 |
| 新线程循环，AHEAD 开启 | worker | **是** | 组当前批 → collect/sample 上一批 → submit 当前批 → process 上一批 |

下文中 `A(N)` 表示接收和组建第 N 批，`F(N)` 表示主机侧 `run_batch(N)` 调用，`P(N)` 表示结果处理，`S(N)` 表示按需延迟采样。图示只表达顺序和并发关系，不按实际耗时比例绘制。

### 没有 overlap 的普通循环

对应 `event_loop_normal_disagg_prefill`，主要顺序是：

```text
Scheduler： A(N) → F(N) → P(N) → 处理 KV inflight → A(N+1) → F(N+1) → P(N+1)
Worker：    无
```

当前批结果处理完之后，主循环才进入下一轮。虽然底层 CUDA 调用仍有异步语义，但这里没有把 batch 结果处理延后一轮形成调度流水。

### 原有 overlap：GPU 异步，run_batch 仍占主线程

对应 `event_loop_overlap_disagg_prefill`：

```text
Scheduler： A(N) → F(N) → P(N-1) → 处理 KV inflight → S(N) → A(N+1) → F(N+1)
GPU：              已提交的 forward 可以在主机函数返回后继续执行
Worker：    无
```

源码的关键顺序：

```python
batch = get_next_disagg_prefill_batch_to_run(...).batch_to_run
result = run_batch(batch)
_apply_war_barrier()
result_queue.append((batch.copy(), result))
process_batch_result(previous_batch, previous_result)
process_disagg_prefill_inflight_queue()
launch_batch_sample_if_needed(result, batch)
last_batch = batch
```

它已经通过 CUDA 异步执行和延后一批的结果处理实现 overlap；瓶颈是主线程仍要先走完 `run_batch(N)` 的主机路径，之后才能进入下一轮组 batch。

### 线程开启但 AHEAD 关闭：没有新增并发

对应新循环中的 `if not ahead` 分支：

```text
Scheduler： A(N) → submit(N) → [立即 collect 等待] → P(N-1) → S(N) → A(N+1)
Worker：                       F(N) ────────────┘
```

`_run_batch_on_fwd_thread()` 只做 `_fwd_submit()` 后立即 `_fwd_collect()`。主线程等待期间没有去组下一批，因此这是验证 forward 移到线程后是否正确的基线。“无 overlap”在这个模式中特指没有新增的 scheduler–worker 并发；原有 CUDA 异步机制仍然存在。

### AHEAD 开启：当前 forward 与下一批组 batch 重叠

对应 `event_loop_overlap_disagg_prefill_threaded` 的 AHEAD 分支，稳定运行时可以按下表理解：

| 顺序 | Scheduler 主线程 | Forward worker |
|---|---|---|
| 1 | `process_batch_result(N-1)`、处理 KV inflight | 执行此前已 submit 的 `run_batch(N)` |
| 2 | 下一轮 retire AS writes、ingest、bootstrap、**组 batch N+1** | 继续 `run_batch(N)` |
| 3 | `collect(N)`，必要时等待 worker 返回 | 返回 result 和 batch copy，随后空闲 |
| 4 | `sample(N)`；为 N+1 准备 staging、snapshot 并 early-send cached prefix | 空闲 |
| 5 | `submit(N+1)` | 开始 `run_batch(N+1)` |
| 6 | **`process_batch_result(N)`**、处理 KV inflight | 继续 `run_batch(N+1)` |
| 7 | 下一轮 **组 batch N+2** | 继续 `run_batch(N+1)` |

```text
时间向下             Scheduler 主线程                  Forward worker
────────────────────────────────────────────────────────────────────
                     submit(N) ───────────────────────→ F(N)
                     P(N-1)                              │
                     处理 KV inflight                    │
                     retire / ingest / A(N+1)            │
                     collect(N) ←────────────────────────┘
                     S(N)                            worker 空闲
                     snapshot / early-send(N+1)
                     submit(N+1) ─────────────────────→ F(N+1)
                     P(N)                                │
                     处理 KV inflight                    │
                     retire / ingest / A(N+2)            │
                     collect(N+1) ←──────────────────────┘
```

实际循环的关键结构如下，省略了暂停、空闲和异常分支：

```python
retire_ascache_writes()
ingest_requests()
batch = get_next_disagg_prefill_batch_to_run(...).batch_to_run

if inflight is not None:
    previous_result, previous_copy = _fwd_collect()
    launch_batch_sample_if_needed(previous_result, inflight)

if batch:
    snapshot = _snapshot_batch_for_forward(batch)
    for req in snapshot.reqs:
        maybe_send_cached_prefix_chunk(req)
    _fwd_submit(snapshot, early_send_prefix=False)

if previous_result is not None:
    process_batch_result(previous_copy, previous_result)
process_disagg_prefill_inflight_queue()

inflight = snapshot if batch else None
last_batch = batch
```

**重要顺序是先组下一批，再 collect 上一批；先 submit 当前批，再 process 上一批。** 前者让 admission 与前向调用并发，后者让结果处理与新的前向调用并发。启动时没有上一批可处理，结束时即使没有新 batch，也会 collect/process 已提交的批次，并继续推进 KV transfers。

### 哪些工作仍然必须排序？

**Sampling 要在 worker 空闲时执行。** `launch_batch_sample_if_needed()` 在存在延迟采样函数时使用 `forward_stream` 和 `future_map`，与 `run_batch()` 共用这些结构。因此必须先 collect，再 sample，然后才 submit 下一次 forward；它不是可以任意丢到另一个线程的工作。

**只有一个 forward worker。** 输入、输出队列容量均为 1，循环在 collect 上一批后才提交下一批。这里并行的是 scheduler 工作与 forward 主机路径，并非同时运行两个独立的 `run_batch()`。

**GPU 上仍有依赖等待。** forward stream 等待 schedule stream 的准备工作；WAR barrier 让后续共享缓冲区写入等待此前 forward 的共享读取。CPU 上组 batch 提前进行，不代表对应设备写入可以越过依赖，也不保证所有标为并发的工作都能完全隐藏。

### 为什么不能直接把 live batch 交给线程？

下一轮 admission 会过滤或重绑定 batch 字段；worker 若直接读取同一个 live batch，可能拿到被下一轮修改的状态。提交前的 `_snapshot_batch_for_forward()` 做了以下隔离：

| 对象 | 处理方式 | 原因 |
|---|---|---|
| ScheduleBatch | `copy.copy(batch)` 浅拷贝 | admission 重绑定 live batch 字段时，snapshot 仍保留原引用 |
| sampling_info | `copy_for_forward()` | filter 会原地修改；forward 副本移除 penalizer orchestrator |
| spec_info | 存在时浅拷贝 | 隔离 batch filter 对该对象的修改 |
| reqs 中的 Req | 保留真实对象引用 | 结果处理必须修改真实请求的生命周期状态 |

尤其不能把 chunked request 深拷贝后丢给结果处理。admission 会增加真实请求的 `inflight_middle_chunks`，process 必须在同一个对象上递减；若递减落到临时副本，真实请求就可能一直被当成 middle chunk，无法释放 req_to_token slot 和 KV。snapshot 也不意味着永久共享一个可随意修改的 reqs 列表：live batch 的字段重绑定由不同的 batch 对象承接，而列表中的 Req 身份保持一致。

worker 在 `run_batch()` 和 WAR barrier 之后还会生成 `batch.copy()`，供 scheduler 处理本次 forward 的结果。`SamplingBatchInfo.copy_for_forward()` 同时被改为可重复调用，支持提交前隔离后，`run_batch()` 内再隔离一次。

### Cached prefix 为什么放到主线程发？

原有 `run_batch()` 会在 suffix forward 前发送已缓存的 prefix KV。线程并发后，同一个 continuing chunked request 的 KV sender 还可能被上一批的 `process_batch_result()` 用来发送 KV chunk。

AHEAD 模式把 `maybe_send_cached_prefix_chunk()` 放在 scheduler 主线程、submit 之前执行，并传入 `early_send_prefix=False`，防止 worker 再次发送。这样保留 prefix 发送与 suffix forward 重叠的机会，同时把相关 sender 的主机调用留在同一个线程顺序执行。

### 开启方法与收益边界

```bash
export SGLANG_ENABLE_PREFILL_FWD_THREAD=1
export SGLANG_PREFILL_FWD_THREAD_AHEAD=1
```

两个开关默认都是 False。只设置第一个时运行线程验证模式；两个都设置才启用 admit-ahead。dispatch 还要求 disagg PREFILL、PP=1、enable_overlap；PP>1 使用其专用循环，普通非 PD prefill 和 decode 不通过这个开关进入新路径。

收益来自隐藏 forward 主机路径与 admission / 结果处理之间原本串行的部分。实际收益受 CPU 调度成本、Python 线程执行、GPU 依赖、AS load-back、KV 发送和输入负载影响，不能把它等同于 GPU 算力提升。

commit message 报告：在 hit=0.8 的验证中，没有观察到 req_to_token 泄漏，bench 能完成，uncache ITPS 相对 **thread-no-overlap 基线** 提升约 7%～13%。这是提交说明中的测试结果，不是本文复测，也不是相对旧默认 overlap 循环的直接收益数据。

### 同一提交中的其他优化

- **跨请求合并 SWA staging**：prefill 结果处理先收集 `inflight_middle_chunks <= 0` 的请求，一次规划各请求已提交的 checkpoint，把原来逐请求的 gather / host sync / copy 合并成跨请求批处理；必须在 AttentionStore write 消费 staging slots 之前进行。
- **非整页尾部修正**：`kv_to_page_indices_pinned()` 的页数改为向上取整，发送侧也使用相同规则，避免 strided slice 包含尾页而 host buffer 按向下取整分配导致 copy 尺寸不匹配。

这两项与新增 forward 线程位于同一提交，分析整次 commit 的性能变化时应一并考虑。

### 源码定位

以下路径和函数均指 commit `c833093f7c3797eab3f9e98f1a3187ccfb997f04` 的版本，可用 `git show <commit>:<path>` 查看：

- `python/sglang/srt/disaggregation/prefill.py`：三个 prefill event loop、worker 队列、snapshot、结果处理及 prefix KV 发送。
- `python/sglang/srt/managers/scheduler.py`：`dispatch_event_loop()`、`run_batch()`、`_apply_war_barrier()`、`launch_batch_sample_if_needed()`、批量 SWA staging。
- `python/sglang/srt/environ.py`：两个线程开关及默认值。
- `python/sglang/srt/sampling/sampling_batch_info.py`：`copy_for_forward()` 的重复调用支持。
- `python/sglang/srt/mem_cache/common.py`：pinned page-index buffer 的尾页处理。

相关概念：[PD 分离](./disaggregation.md)、[PD 设计](./disaggregation_design.md)、[Chunked Prefill](./chunked_prefill.md)、[Batch 设计](./batch_design.md)、[CUDA Stream](./cuda_stream.md)。

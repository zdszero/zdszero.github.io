% SBO

### 双流 overlap 小算子

H100 有 132 个 SM。一些小算子（如 decode 时 batch=1 的 LayerNorm、RoPE、act_quant）可能只需要几个 SM，剩余 SM 空闲。把两个这样的小算子放到两个 stream 上，GPU 硬件调度器会将它们分配到不同 SM 上**物理并行执行**。

```
单流:  [LayerNorm_q ████░░░░░░░░][LayerNorm_k ████░░░░░░░░]  ← SM 大量闲置
双流:  [LayerNorm_q ████░░░░░░░░]                              ← 同时跑
       [LayerNorm_k ████░░░░░░░░]                              ← 时间减半
```

#### Example

- q_a_layernorm ‖ kv_a_layernorm
- q_b_proj ‖ Indexer
- indexer 内部
    - wq_b ‖ wk + k_norm
    - rotate_activation(q) ‖ rotate_activation(k)
    - act_quant(q) ‖ act_quant(k)

```python
if self.alt_stream is not None and get_is_capture_mode():
    current_stream = torch.cuda.current_stream()
    self.alt_stream.wait_stream(current_stream)
    q = self.q_a_layernorm(q)
    with torch.cuda.stream(self.alt_stream):
        k_nope = self.kv_a_layernorm(k_nope)
    current_stream.wait_stream(self.alt_stream)
```


在普通模式下做可能是负收益：

1. **动态开销**：每次前向传播都要创建/管理流同步，开销较大
2. **小批量不划算**：对于小 batch size，并行收益可能小于同步开销

**Prefill 时双流 overlap 通常无效**：这也是为什么 NSA 代码中 overlap 只在 decode 启用。Prefill 时 token 数量大，单个 GEMM 已经能占满所有 SM，放两个 stream 反而可能因为资源竞争导致两个都变慢。

#### overlap vs fusion

**为什么不直接写 Fused Kernel**

1. 算子类型不同，fusion 不自然

NSA 中 overlap 的典型例子：`q_b_proj`（GEMM）‖ `indexer`（包含 GEMM + MQA + topk 的复杂流水线）。这两个操作的计算模式、访存模式、数据流完全不同，没有共享数据，fuse 在一起没有意义——你无法把一个矩阵乘法和一个完整的 indexer pipeline 写成一个 kernel。

Fused kernel 的价值在于**减少中间结果的显存读写**（kernel fusion 消除 intermediate buffer），比如把 LayerNorm + 激活 + 量化 fuse 成一个 kernel。前提是这些操作是**同一数据流上的串行依赖**。

2. 双流 overlap 的对象是两条独立数据流

```
数据流 A: q_lora → wq_b → query
数据流 B: x → wk → key → k_norm
```

A 和 B 没有数据依赖，没有中间 buffer 可以消除。Fuse 它们意味着写一个 kernel 同时处理两组不相关的输入，这本质上就是在 kernel 内部做手动调度——GPU 的硬件调度器已经能做这件事了。

3. 开发成本差异巨大

---

**Fused kernel 更优的场景**：
- 多个 memory-bound 算子串联在同一数据上（如 LayerNorm → ReLU → Quantize）
- Fusion 能消除中间 tensor 的全局内存读写
- 单个 kernel 就能用满 GPU

**双流 overlap 更优的场景**：
- 两个算子操作不同数据，没有 fusion 收益
- 单个算子无法用满 SM（decode batch 小）
- 算子来自不同库（如 cuBLAS GEMM vs 自定义 triton kernel），无法 fuse


% Dualpipe

### Mircro Batch

### DS Prefill

![deepseek prefill overlap](../../../docs/WikiImage/image_2025-09-30-15-32-16.png)

分为四个阶段：

- `attn + shared` vs `combine`
- `attn` vs `dispatch`
- `dispatch` vs `mlp`
- `mlp + shared` vs `combine`

每层的时间计算方法如下，就是看每个阶段的 dispatch 和 combine 有没有被 overlap 住：

```python
# whatever need 20us for issue and hook kernel launch
# attn + shared vs combine
combine_overlaped1 = (attn_gate_pure + shared_time) > combine_time
combine_overlaped1_time = (
    attn_gate_pure + shared_time + 20 if combine_overlaped1 else combine_time
)
# attn vs dispatch
dispatch_overlaped1 = attn_gate_pure > dispatch_time
dispatch_overlaped1_time = (
    attn_gate_pure + 20 if dispatch_overlaped1 else dispatch_time
)
# dispatch vs moe-layer
dispatch_overlaped2 = moe_layer_time > dispatch_time
dispatch_overlaped2_time = (
    moe_layer_time + 20 if dispatch_overlaped2 else dispatch_time
)
# mlp + shared  vs combine
combine_overlaped2 = (shared_time + moe_layer_time) > combine_time
combine_overlaped2_time = (
    shared_time + moe_layer_time + 20 if combine_overlaped2 else combine_time
    )
```

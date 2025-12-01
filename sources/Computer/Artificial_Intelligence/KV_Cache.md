% KV Cache

### Process

1. load model
2. init mem pool
3. init cuda graph

### Static Dynamic

- static memory
    - model weight
    - kv cache pool
- dynamic memory
    - cuda graph

### Mem Pool

```
min_per_gpu_mem = self.init_torch_distributed()
total_gpu_mem = min_per_gpu_mem
available_gpu_memory = get_available_gpu_memory(
    self.device, self.gpu_id, distributed=self.tp_size > 1
)
rest_memory = available_gpu_memory - total_gpu_mem * (1 - frac_mem_staic)
max_num_tokens = rest_memory * (1 << 30) / cell_size
```

deepseek on gb200:

 ```
[Mem] total_gpu_mem = 174.8406982421875
[Mem] avail_gpu_mem = 78.7196044921875
[Mem] rest_memory = 69.97756958007812
[Mem] cell_size = 70272
[Mem] max_num_tokens = 1069242
 ```

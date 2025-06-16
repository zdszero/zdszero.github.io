% cooperative group

```cuda
namespace cg = cooperative_group;

// thread block
cg::thread_block block = cg::this_thread_block();
// thread warp
cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
```

- `warp.thread_rank()`: index of thread in warp
- `warp.size()`: generally is 32

If you want to use cooperative group to build reduce operations:

```cpp
int idx = blockIdx.x * warp.meta_group_size();
float sum = 0.0f;
float *x = inp + idx * C;
for (int i = warp.thread_rank(); i < C; i += warp.size()) {
    sum += x[i];
}
sum = cg::reduce(warp, sum, cg::plus<float>{});
```


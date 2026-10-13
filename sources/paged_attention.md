% Paged Attention

PagedAttention 的核心思想是：

> KV cache 不连续存储，而是分 block（类似操作系统分页）

PagedAttention 解决的是：

1. **避免内存碎片**

因为 request 长度不同，如果连续存：

```
req1: 100
req2: 1000
req3: 5
```

删除 req2 后会出现巨大空洞。

Paged 把 KV 切成固定 block，比如 16 tokens/block：

```
[block1][block2][block3]...
```

释放时只回收 block。

2. **支持共享**

prefix cache / speculative decode 里：

多个 request 可能共享前缀 token

只需要让：

```
req_to_token_pool
```

指向相同 token_index

而 token_to_kv_pool 指向同一 block

无需复制 KV。

3. **支持动态扩展**

decode 时：

* 新 token
* 从 free block list 拿一个 block
* 绑定到 token_id

完全 O(1) 扩展。

#### Example

假设：

block_size = 4 token

req1 输入 6 token：

```
req_to_token_pool[1] = [0,1,2,3,4,5]
```

token_to_kv_pool:

```
0 -> block0 offset0
1 -> block0 offset1
2 -> block0 offset2
3 -> block0 offset3
4 -> block1 offset0
5 -> block1 offset1
```

block1 剩下两个空位可以给别的 request 用。

这就是 paged 的意义。


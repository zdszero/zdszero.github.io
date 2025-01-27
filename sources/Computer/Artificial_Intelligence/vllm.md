% vllm

The task of language model is to model the probability of a list of tokens:

$P(x) = P(x_1) \cdot P(x_2 | x_1) \cdots P(x_n | x_1, \dots, x_{n-1})$


### Block Space Manager

- __How do system manage logical and physical blocks?__

Every `Sequence` track its logical blocks.

`BlockSpaceManager` use `Dict[seq_id, BlockTable]` to track sequence's physical blocks.

------

- __How to get block memory size?__

`cache block size = block_size * num_layers * num_heads * dim_head * 2 * dtype_size`

`* 2` is to store key and value

### Block Table

a list of `PhysicalTokenBlock`

### Scheduler

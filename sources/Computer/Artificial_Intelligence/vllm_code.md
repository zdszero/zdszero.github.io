% vllm v0.1.2 代码阅读

```
llm.run
    → step
        → run_workers("execute_model")
            → worker.execute_model
                → modelrunner.execute_model
        → decode_seq
        → stop_seq
        → free_finished_seq
```

- 序列组元数据中包含哪些信息？

block_tables 即一个字典，将所有的序列ID映射到对应的块表block_table，block_table实际上就是List[PhysicalTokenBlock]

- 如何将序列组的数据封装为神经网络的输入参数？

- 执行模型时的参数都是什么？各有什么含义？

input_ids 的类型为 torch.Tensor，应该就是将

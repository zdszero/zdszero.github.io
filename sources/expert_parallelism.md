% EP

### Fused vs EP

* fused moe 相比于 ep moe 的 **优点**：
    * 没有不均衡的问题
* fused moe 相比于 ep moe 的 **缺点**：
    * 存在重复计算的部分，计算量大 8 倍
        * moe gate
        * moe topk
        * quant（如果需要的话）
    * 无法做多机部署
* **差不多** 的部分：
    * 通信量的差异：fused moe 要大一些，需要根据 attention tp size 进行论证
        * attention tp size == 8，两者通信量差不多
            * all reduce = reduce scatter + dispatch （假设 dispatch 是 bf16 的话两者一致，实际上 deepseek dispatch 是 fp8 所以 ep moe 还是要小 1/4）
            * all reduce = combine + all gather
        * attention tp size == 1，两者通信量差不多（假设 dispatch 是 bf16 的话两者一致，实际上 deepseek dispatch 是 fp8 所以 ep moe 还是要小 1/2）
            * all gather = dispatch
            * reduce scatter = combine
* 需要根据实现来 **动态分析** 的部分：
    * TP 模式的 bmm 和 deep gemm 的 group gemm 的比较，实际上 deep gemm 表现更好



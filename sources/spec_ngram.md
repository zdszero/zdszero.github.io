% 投机解码（二）：不要 draft 模型的 n-gram 方案

> 系列总览见 [投机解码方案对比](./speculative_decoding.md)。上一篇：[从 Blockwise 到 Speculative Sampling](./spec_origins.md)。

### 问题：draft 模型本身也要钱

Speculative Sampling 的收益公式里，c（draft / target 单步耗时比）直接出现在分母里。独立小模型要训练、要部署、要占显存，还要串行跑 γ 次。如果某类任务的输出大段**复制**输入或历史，那么「猜」根本不需要模型：查一下前面出现过的接续就够了，c 近似为 0。

### 三条路线

2023 年底到 2024 年初出现了三种无模型的 draft，区别在于「从哪里查」：

| 工作 | 查什么 | 怎么验 | 论文结果 |
| :--- | :--- | :--- | :--- |
| Prompt Lookup Decoding（Saxena，2023.11） | 当前 prompt 里最后 n-gram（n ≤ 3）的上一次出现，取其后 10 个 token | 链 | Mistral-7B 摘要 / QA 约 2.4×（greedy）；开放对话收益很小 |
| REST（He 等，[arXiv 2311.08252](https://arxiv.org/abs/2311.08252)） | 外部数据存储（The Stack 27 GB、UltraChat 12 GB）的后缀数组，最长后缀匹配 | 候选放进 Trie，验最常见的 64 个前缀（树） | CodeLlama-7B HumanEval 2.36×，Vicuna MT-Bench 1.62～1.77× |
| Lookahead Decoding（Fu 等，[arXiv 2402.02057](https://arxiv.org/abs/2402.02057)） | target 自己做 Jacobi 迭代的轨迹，从中收集 n-gram 池 | lookahead 分支和 verify 分支用一个 mask 放进同一次前向 | 单卡 1.5～2.3×；每步 FLOPs 约为原来 120 倍 |

它们的共同 insight 是：**draft 不必是语言模型，只要是一个「足够经常猜对」的分布**。代价也是共同的：

- 对开放式生成几乎没用，命中率取决于输出和语料 / 上文的重合度。
- REST 没有「上下文内」能力，比如猜不出用户刚起的变量名；Prompt Lookup 只看 prompt；Lookahead 用算力换步数，到了算力受限的大 batch 反而可能变慢。

### SGLang 的 NGRAM：PLD 和 REST 的合体

SGLang 的 `NGRAM` 算法（`speculative/ngram_worker.py`，核心在 `speculative/cpp_ngram/`）把前两条路线合在了一起：

- **在线 trie**：每个请求取最近 `max_trie_depth`（默认 18）个 token 去 trie 里查后续。verify 结束后把这段窗口插回去。trie 由所有请求共享，所以别的请求生成过的片段也能命中，这比 PLD 只看自己的 prompt 更进一步。
- **外部语料**：可以预加载 JSONL 语料建成后缀自动机，运行时用 `POST /add_external_corpus` 追加（`external_corpus_manager.py`），相当于 REST 的数据存储。
- **树形 verify**：查到的多个接续按 BFS 展开成树，宽度在 `--speculative-ngram-min/max-bfs-breadth`（默认 1～10）之间，默认 12 个 draft token，用 SpecInfer 的树形 mask 一次验完。

它没有 draft 模型，`has_draft_kv()` 为 False，树只存在于 verify 的 attention mask 里，也就不存在「verify 后 draft 状态要对齐」的问题。限制是不支持 DP attention。

适用场景很明确：代码编辑、长文改写、RAG 回答、带模板的结构化输出。开放式聊天请换成后面几篇里读 target hidden 的方案。

### 小结

n-gram 方案把 draft 成本压到接近零，换来的是命中率完全依赖「输出是不是在复制已有文本」。SGLang 的 NGRAM 用共享 trie 加可选外部语料，覆盖了 Prompt Lookup 和 REST 两种来源，并用树形 verify 弥补单条查找命中率不稳的问题。想在开放式生成上也猜得准，draft 就得看到 target 的内部状态，这是下一篇 [Medusa 与 EAGLE](./spec_medusa_eagle.md) 的出发点。

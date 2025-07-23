% Hyperparameters

Transformer hyperparameter questions you might have had in 224n..

* How much bigger should the feedforward size be compared to hidden size?
* How many heads, and should num_heads always divide hidden size?
* What should my vocab size be?

And other model setting questions

* Do people even regularize these huge LMs?
* How do people scale these models - very deep or very wide?

### FeedForward

$$FFN(x) = max(0, xW_1 + b_1)W_2 + b_2$$

There are two dimensions:

- $d_{ff}$: feed forward dimension
- $d_{model}$: model dimension

convention: $d_{ff} = 4 d_{ff}$

__Exception #1 - GLU varient__

Remember that GLU variants scale down by $2/3^{\text{rd}}$. This means most GLU variants have
$d_{ff} = \frac{8}{3}d_{model}$. This is mostly what happens. Some notable such examples.

| Model       | $d_{ff}/d_{model}$ |
|-------------|--------------------|
| PaLM        | 4                  |
| Mistral 7B  | 3.5                |
| LLaMA-2 70B | 3.5                |
| LLaMA 70B   | 2.68               |
| Qwen 14B    | 2.67               |
| DeepSeek 67B| 2.68               |
| Yi 34B      | 2.85               |

### Aspect Ratio

Should my model be deep or wide? How deep and how wide?

Most models are surprisingly consistent on this too.

| Model | $d_{model} / n_{layer}$ |
|:-:|:-:|
| BLOOM | 205 |
| T5 v1.1. | 171 |
| PaLM | 156 |
| GPT3/OPT/Mistral/Qwen | 128 |
| LLama/LLama2 | 102 |

### Vocab Size

将两张表格转换为 Markdown 格式：

**Monolingual models - 30-50k vocab**

| Model              | Token count |
|--------------------|-------------|
| Original transformer | 37000       |
| GPT                | 40257       |
| GPT2/3             | 50257       |
| T5/T5v1.1          | 32128       |
| LLaMA              | 32000       |

**Multilingual / production systems 100-250k**

| Model       | Token count |
|-------------|-------------|
| mT5         | 250000      |
| PaLM        | 256000      |
| GPT4        | 100276      |
| Command A   | 255000      |
| DeepSeek    | 100000      |
| Qwen 15B    | 152064      |
| Yi          | 64000       |

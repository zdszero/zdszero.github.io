% Godel Incompleteness Proof

### 形式系统

A formal system is a math concept that consists of a set of symbols and a set of rules for manipulating these symbols.

These rules define how to generate, derive or transform symbols, creating a series of formal expressions. Here are some concepts:

- symbols（符号）
- formulas（公式）
- axioms（公理）
- derivation rules（推导规则）
- theorems（定理）

-----

### 哥德尔不完备定理证明

__基于哥德尔数（Gödel number）建立映射关系__

> typographical properties of long chains of symbols can be talked about in an indirect but perfectly accurate manner by instead talking about the properties of prime factorizations of large integers.

基本思想：将形式系统的演绎过程映射为数字的因数分解过程

将符号用 $1 \sim 12$ 的数字代替，其他代表变量（variable）的符号 $x, y, z$ 等用大于 12 的素数表示（比如$13, 17, 19, \cdots$)。

用这些符号的组合（factorized combination of symbols）可以将公式（formula）映射为一个哥德尔数。

> In sum, every expression inthe system, whether an elementary sign, a sequence of signs,or a sequence of sequences, can be assigned a unique Gödelnumber.

我们可以为每一个符号以及符号的任意组合都对应到一个哥德尔数，不管它是正确的还是错误的。

| constant sign | godel number | usual meaning |
|:-:|:-:|:-:|
| $\sim$ | 1 | not |
| $\vee$ | 2 | or |
| $\supset$ | 3 | if...then... |
| $\exists$ | 4 | there is an... |
| $=$ | 5 | equals |
| $0$ | 6 | zero |
| $s$ | 7 | the successor of |
| $($ | 8 | punctuation mark |
| $)$ | 9 | punctuation mark |
| $,$ | 10 | punctuation mark |
| $+$ | 11 | plus |
| $\times$ | 12 | times |

$\text{a and b}$ 也就是 $\sim (\sim a \vee \sim b)$

比如命题 $0 = 0$，可以被转化为 $2^6 \times 3^5 \times 5^6$

命题 $0 \ne 0$ 也就是 $\sim (0 = 0)$，可以被转化为 $2 \times 3^8 \times 5^6 \times 7^5 \times 11^6 \times 13^9$

__元数学的计算__

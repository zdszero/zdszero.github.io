% SWA

将 MHA 的计算量从 \(O(T^2)\) 变为了 \(O(T)\)，将访存量变为了从 \(O(T)\) 变成了 \(O(W)\)

- MHA 的计算量（单层）：\(2BT^2HD\)
- SWA 的计算量（单层）：\(4BWHD\)
- MHA 需要存储的 kv cache：\(B L T (d_{qk} + d_{v}) H_{kv}\)
- SWA 需要存储的 kv cache：\(B L W (d_{qk} + d_{v}) H_{kv}\)

推理引擎中有一个参数为 `--swa-full-tokens-ratio`，表示 SWA 和 GQA 占用显存的比值。

可以通过以上公式进行估算，以 mimo v2 flash 为例：

- 总共 48 层，39 层 swa，9 层 gqa
- swa 层：
    - `window_size=128`
    - `num_heads=64`
    - `num_kv_heads=8`
- gqa 层：
    - `num_heads=64`
    - `num_kv_heads=4`
- swa 和 gqa 层 \(d_{qk}\) 和 \(d_{v}\) 相同

定义最大序列长度为 \(T\)，swa 和 gqa 需要占用的 kv cache 比值为：

\[
\frac{L_{\text{swa}} \cdot W \cdot 8}{L_{\text{gqa}} \cdot T \cdot 4} = \frac{39 \cdot 128 \cdot 8}{9 \cdot T \cdot 4}
\]


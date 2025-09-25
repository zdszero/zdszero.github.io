% types

### fp8

__E4M3__

1 sign bit + 4 exponent bits + 3 mantissa bits

__E5M2__

1 sign bit + 5 exponent bits + 2 mantissa bits

- E4M3: Often used for model weights where you need reasonable precision
- E5M2: Often used for gradients/activations where extreme values need to be represented

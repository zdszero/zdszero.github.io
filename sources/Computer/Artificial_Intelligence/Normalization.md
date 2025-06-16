% Normalization

Normalization in deep learning refers to the process of transforming data to conform to specific statistical properties.

There are several ways: standardization or min-max normalization.

### Standardization

$$x' = \frac{x - \mu}{\sigma}$$

One common type is standardization, where each data point is adjusted by subtracting the mean of its column and then dividing by the standard deviation. This transformation results in a new column where the mean is zero and the standard deviation is one.

### min-max normalization

data is scaled to fit within a given range

### Why

__Nomalize data to improve training efficiency__

Imagine you’re training a neural network, and as you update the weights, some of them start getting really big. When that happens, the activations tied to those weights also become large, making it harder for your model to learn effectively. It slows things down and can cause problems in training.

Normalization helps fix this by keeping activations within a stable range. This not only makes the training process more stable but also speeds it up, allowing your model to learn more efficiently.

__Avoid internal covariate shift__ 

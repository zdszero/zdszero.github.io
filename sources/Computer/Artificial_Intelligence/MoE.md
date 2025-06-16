% MoE

Exploiting scale in both training data and model size has been central to the success of deep learning. When datasets are sufficiently large, increasing the capacity (number of parameters) of neural networks can give much better prediction accuracy.

### Idea

The basic idea of MoE is split the FFN into multiple sub-networks(experts), for each input token, only part of sub-networks(experts) are activated. Different sub-networks behavior as different “experts”, during training they absorb different information and knowledge from the dataset, during inferencing only part of experts are activated based on the input token.

- __Gating Network__: This small neural network takes the input and learns to determine which experts are most relevant for processing that specific input. It produces scores or probabilities for each expert

### Dense/Sparse Layer

#### Dense Layer

FFNN (Feedforward Neural Network)

An FFNN allows the model to use the contextual information created by the attention mechanism, transforming it further to capture more complex relationships in the data.

MLP is a type of FFNN

![dense layer](../../../docs/WikiImage/image_2025-01-07-17-40-48.png){ width=500px }

#### Sparse Layer

Only activate a portion of the parameters

Each expert learns different information during training

![sparse layer](../../../docs/WikiImage/image_2025-01-07-17-43-21.png){ width=500px }

### Expert

### What to learn

<u>Mixtral Paper</u>

![Example Expert](../../../docs/WikiImage/image_2025-01-07-17-46-22.png)

------

[MoE walkthrough](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)

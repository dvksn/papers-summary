# Architecture
Here is the famous picture of the transformer architecture:

![image](https://github.com/dvksn/papers-summary/assets/18422658/82848146-eea7-4ffb-b50b-de64b92f2cca)
## Encoder
It's a stack of 6 identical layers. Each layer has a Multi-head self attention block and a feed forward network. Residual connection is applied around each of the two sublayer. This goes through a layernorm (Also called post-layernorm). In the new architectures layernorm is applied before feeding to MHA and FFA.
h1 = LayerNorm(x + MHA(x))
h2 = LayerNorm(h1 + FFA(h1))

## Decoder
The decoder also has 6 identical layers. It has an extra block for Masked Multi-Head cross attention between encoder and decoder. Also, Decoder has **Causal** Self Attention instead of Self Attention of Encoder.

That's it for the uber level architecture overview! Now, let's dive into details of each component.

## Attention
It's a weighted sum of values with weight calculated as the compatibility between keys and queries.
### Scaled dot product attention
![image](https://github.com/dvksn/papers-summary/assets/18422658/7863cc10-decd-48ae-830a-b6f5e031c178)




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
#### Why not additive attention?
Additive attention calculated using single hidden layer Feed forward network. In practice scaled dot product attention is faster due to matrix multiplications.
For large values of dk, additive attention outperforms dot product attention. Since dot products grows with larger dk, pushing softmax into regions where it has small gradients. That's dot product attention is scaled down by sqrt(dk).

### Multi-head attention
Instead of performing single attention function, authors first projects query, key, value in dk, dk and dv dimension respectively. Attention operation is done on multiple heads in parallel and final output is concatenated. This helps in learning features from the input just like convolution operation where multiple filters are learned.
MultiHead(Q, K, V ) = Concat(head1, ..., headh)WO

## Position-wise Feed forward network
Each encoder/decoder layer contains a feed forward network.
FFN(x) = max(0, xW1 + b1)W2 + b2
This is where most of the computation happens as the size of the weight matrix is usually large as compared to dmodel. (dff = 2048, dmodel = 512)

## Embeddings and softmax
The input embedding layer of encoder, input embedding layer of decoder and the output linear transformation layer shares the same weight matrix.
In the two embedding layers, weights are multiplied by sqrt(dk).

## Positional Encoding
A sinusoidal is used to get the positional encodings. This adds the position information of each token.

## Regularization
1. Layer Norm
2. Dropout
3. Label Smoothing: To reduce the overconfidence in predictions. It can hurt perplexity.




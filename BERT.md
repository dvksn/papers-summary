# Architecture
It is very similar to encoder part of the transformer ie. It contains multi-head non-causal self attention blocks and feed forward layers.

# How does it understands the language?
BERT is trained on two training taks. 1. Masked Token Modeling (MLM) and 2. Next Sentence Prediction (NSP). These two tasks helps in developing language understanding.
## MLM
1. Since BERT uses both left and right context, to make it understand the language it modifies 15% of the token randomly. 80% of these tokens are replaced with [Mask] token, 10% are replaced with random word and rest 10% are kept exactly same.
2. The final hidden vectors corresponding to these 15% tokens are fed into an output softmax over vocabulary.
3. Since masked tokens are not present in the fine-tuning task, to close this mismatch in pre-training and fine-tuning only 80% of the time [Mask] token was used.

## NSP
1. To make the model learn relationship between two sentences, BERT was trained on Next sentence prediction task. It helps in Question-Answering, Natural Language Inference tasks.
2. To train on NSP, BERT used monolingual dataset where 50% of the time sentence A was followed by sentence B (Label = IsNext) and 50% of the time sentence B was not the next sentence (Label = NotNext).
3. It's a classification task where CLS token is used for prediction.

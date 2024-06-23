# Architecture
It is very similar to encoder part of the transformer ie. It contains multi-head non-causal self attention blocks and feed forward layers.
BERT Base has 110M parameters while BERT Large has 340M params.

# How does it understand the language?
BERT is trained on two training taks. 1. Masked Token Modeling (MLM) and 2. Next Sentence Prediction (NSP). These two tasks helps in developing language understanding.
## MLM
1. Since BERT uses both left and right context, to make it understand the language it modifies 15% of the token randomly. 80% of these tokens are replaced with [Mask] token, 10% are replaced with random word and rest 10% are kept exactly same.
2. The final hidden vectors corresponding to these 15% tokens are fed into an output softmax over vocabulary.
3. Since masked tokens are not present in the fine-tuning task, to close this mismatch in pre-training and fine-tuning only 80% of the time [Mask] token was used.

## NSP
1. To make the model learn relationship between two sentences, BERT was trained on Next sentence prediction task. It helps in Question-Answering, Natural Language Inference tasks.
2. To train on NSP, BERT used monolingual dataset where 50% of the time sentence A was followed by sentence B (Label = IsNext) and 50% of the time sentence B was not the next sentence (Label = NotNext).
3. It's a classification task where CLS token is used for prediction.

# Data for Pre-training
BookCorpus (800M words) and English Wikipedia (2,500 words).
## Input/Output Representation
1. Words are tokenized using WordPiece tokenizer and WordPiece embeddings are used for tokens.
2. A CLS token is added at the start of the sequence.
3. Sentence pairs (Each "sentence" are typically much longer than single sentence(or shorter too)) are packed together into a single sequence separated by SEP token.
4. The combined length of the sequence is <= 512 tokens.
5. A learned embedding is added to every token representing whether it belongs to first sentence or second sentence.
6. LM masking applied after WordPiece tokenization.
7. Position embedding is added to each token.
8. The input is constructed by summing token embedding, segement embedding, position embedding.

# Pre-training
To speed-up the training pr-training is done for sequence length of 128 tokens for 90% of the total steps. Rest of the training is done on sequence of 512 to learn position embeddings.
ADAM optimizer, dropout is used

# Experiments
## GLUE
For GLUE classification tasks, one layer is added on top of CLS token hidden state (dimension = H) to get output for K classes. cross-entropy loss is used to fine-tune.
## SQuAD v1.1
It's a collection of 100k crowd-sourced question/answer pairs. Given a question and a passage from Wikipedia containing the answer, the task is to predict the answer text span in the passage.
Two new vectors, start and end are added in fine-tuning. The probability of a token being the start of the answer span is computed as dot product between final hidden state of the token and the start vector. Same thing is done for end vector.
The maximum value S.Ti + E.Tj is used for prediction where i<=j
Training objective is the sum of the log-likelihoods of the correct start and end positions.
## SQuAD v2.0
An improvement over v1.1 as it has passages where answer is not present. [CLS] token comes to rescue in this case as for passages where answer is not present it is assumed that answer has start and end span at CLS position. 
Score(null) = S.C + E.C where C is the hidden state of the CLS token. 
A answer span is predicted when S.Ti + E.Tj is larger than Score(null) by some margin.
## SWAG
Given a sentence the task is to choose the most plausible continuation among four choices. Four input sequences are created one for each possible continuation. CLS token representation is used for calculating softmax values.

# Ablation Studies
1. Removing NSP task hurts performance on question answering tasks
2. The MLM pre-training converges slowly as compared to GPT as only 15% tokens are masked. Whereas in GPT model loss is calculated for every token.
3. Random masking results in worse performance than replacing all 15% with [Mask] token.



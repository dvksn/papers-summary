# Why we need it?
So, LLMs have come a long way. From using only 512 tokens in the context to having 2 million tokens context in Gemini 1.5Pro. RAG system was developed when LLMs were able to handle only a small context length. Now, a new RAG architecture was required to exploit the capabilities of new LLMs. 
Also, in the previous architecture retriever's task was to find snapshots of the text which are most relevant to the question and since the retriever units were small (typically 100 words chunks), It was like finding needle in the haystack. Whereas the reader's task was quite simple to just find the most relevant chunk from a fixed number of chunks. This led to less recall if we restricted the number of retrieved units to a low number as well as the continuity of the answer was affected as only small chunk size was allowed.

# What are the changes?
## Long Retrieval Units
In the LongRAG grouping of multiple related documents is done to have more than 4k tokens in one unit. This reduced the overall number of units from 22mn to 600k.

## Long Retriever
It will look for coarse relevant information for a given query. 4-8 units are combined to form context for Reader.

## Long Reader
It prompts LLM like GPT4 or Gemini with around 30k tokens context of retriever units to get the answer. For 

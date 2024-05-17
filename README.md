# Nano-GPT for Shakespearean writing style
 The aim of this project is to deepen the understanding of "Transformers," which have become essential elements of modern architecture. These 
architectural advancements rely on the "self-attention" mechanism, aiming to fully understand and harness the potential of this technology.

### Every step is explained in the code 

## what you will find in this project 
* Preparation of the database, definition of the vocabulary size, and creation of a simple letter-to-integer encoder/decoder for Shakespearean texts.
* Implementation of "token embeddings" and "positional embeddings" to represent letters as vectors and their positions in blocks (sentences).
* Implementation of the "self-attention" mechanism: the dot product between keys, queries, and values (calculating "affinities" between embeddings).
* Implementation of the "Multi-Head Attention" block as described in the famous paper "Attention is All You Need, 2017" (excluding the cross-attention part).
* Training of the model and successful generation of texts in the Shakespearean writing style with a validation loss of 1.56.

#### Project Tools
* pytorch 2.2.2 
* pytorch-cuda 12.1


# Next Character Prediction with MLP

## Overview

This project implements a MLP for next character prediction, leveraging the Shakespeare dataset. The goal is to generate coherent text by predicting the next character based on the preceding context. Inspired by Andrej Karpathy's blog post on the effectiveness of RNNs, this implementation explores the text generation that mimics the style of Shakespeare.

## Dataset

The Shakespeare dataset consists of a collection of texts written by William Shakespeare, which serves as a rich source for training the character prediction model.

# Model
This model is designed for predicting the next character in a sequence using an embedding layer followed by fully connected (linear) layers.

## Model Components

### 1. Embedding Layer
- **Purpose**: Converts input indices (characters) into dense vector representations of specified dimension (`emb_dim`).
- **Layer**: `self.emb = nn.Embedding(self.vocab_size, self.emb_dim)`

### 2. Linear Layers
- **Hidden Layers**: 
  - A list of linear layers is defined to transform the input progressively.
  - Each layer applies a linear transformation followed by a non-linear activation function (sigmoid).
- **First Linear Layer**: 
  - Maps the flattened embeddings to a hidden representation.
  - **Layer**: `self.lin1 = nn.Linear(self.block_size * self.emb_dim, self.hidden_size)`
- **Output Layer**: 
  - Maps from the hidden representation back to the size of the vocabulary.
  - **Layer**: `self.lin2 = nn.Linear(self.hidden_size, self.vocab_size)`


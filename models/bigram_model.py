
import torch
import torch.nn as nn


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        # Convert each character ID into a learnable vector
        self.token_embedding_table = nn.Embedding(
            vocab_size, vocab_size
        )

    def forward(self, index):

        # Look up the embedding for each input character
        logits = self.token_embedding_table(index)

        return logits

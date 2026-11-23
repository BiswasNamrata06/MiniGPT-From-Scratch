
import torch
import torch.nn as nn

# Configuration
vocab_size = 65
block_size = 8
n_embd = 32

# Token embedding table
token_embedding_table = nn.Embedding(
    vocab_size,
    n_embd
)

# Position embedding table
position_embedding_table = nn.Embedding(
    block_size,
    n_embd
)

# Example token IDs
idx = torch.tensor([[1, 5, 10, 20, 30, 15, 8, 2]])

# Get batch and sequence length
B, T = idx.shape

# Token embeddings
token_embeddings = token_embedding_table(idx)

# Position IDs: 0, 1, 2, ..., T-1
positions = torch.arange(T)

# Position embeddings
position_embeddings = position_embedding_table(positions)

# Add both embeddings
x = token_embeddings + position_embeddings

print("Token embedding shape:", token_embeddings.shape)
print("Position embedding shape:", position_embeddings.shape)
print("Final embedding shape:", x.shape)

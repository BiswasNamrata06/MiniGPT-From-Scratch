
import torch
import torch.nn as nn

# Number of unique characters
vocab_size = 10

# Size of each embedding vector
embedding_dim = 4

# Create embedding layer
embedding = nn.Embedding(
    vocab_size,
    embedding_dim
)

# Example character IDs
input_ids = torch.tensor([1, 2, 3, 4])

# Convert IDs into vectors
output = embedding(input_ids)

print("Input IDs:")
print(input_ids)

print("\nEmbedding vectors:")
print(output)

print("\nOutput shape:")
print(output.shape)


import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Create example input
x = torch.randn(4, 8)

# 2. Embedding dimension
embedding_dim = 8

# 3. Create Query, Key and Value layers
query_layer = nn.Linear(embedding_dim, embedding_dim, bias=False)
key_layer = nn.Linear(embedding_dim, embedding_dim, bias=False)
value_layer = nn.Linear(embedding_dim, embedding_dim, bias=False)

# 4. Calculate Q, K and V
Q = query_layer(x)
K = key_layer(x)
V = value_layer(x)

# 5. Calculate attention scores
scores = Q @ K.T

# 6. Scale scores
scores = scores / (embedding_dim ** 0.5)

# 7. Create causal mask
block_size = x.shape[0]

mask = torch.tril(
    torch.ones(block_size, block_size)
)

print("Causal mask:")
print(mask)

# 8. Block future tokens
scores = scores.masked_fill(mask == 0, float("-inf"))

# 9. Apply softmax
weights = F.softmax(scores, dim=-1)

# 10. Calculate output
output = weights @ V

print("\nAttention weights:")
print(weights)

print("\nOutput shape:", output.shape)
print("\nSum of each attention row:")
print(weights.sum(dim=-1))

print("\nUpper triangular attention weights:")
print(torch.triu(weights, diagonal=1))
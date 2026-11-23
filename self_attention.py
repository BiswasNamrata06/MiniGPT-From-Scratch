
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Example input
# 4 tokens, each represented by 8 numbers
x = torch.randn(4, 8)

# 2. Define embedding dimension
embedding_dim = 8

# 3. Create Query, Key and Value layers
query_layer = nn.Linear(embedding_dim, embedding_dim, bias=False)
key_layer = nn.Linear(embedding_dim, embedding_dim, bias=False)
value_layer = nn.Linear(embedding_dim, embedding_dim, bias=False)

# 4. Calculate Q, K and V
Q = query_layer(x)
K = key_layer(x)
V = value_layer(x)

print("Query shape:", Q.shape)
print("Key shape:", K.shape)
print("Value shape:", V.shape)

# 5. Calculate attention scores
scores = Q @ K.T

# 6. Scale scores
scores = scores / (embedding_dim ** 0.5)

# 7. Apply softmax
weights = F.softmax(scores, dim=-1)

# 8. Calculate attention output
output = weights @ V

print("\nAttention weights:")
print(weights)

print("\nAttention output:")
print(output)

print("\nOutput shape:", output.shape)

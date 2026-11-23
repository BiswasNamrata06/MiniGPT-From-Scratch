
import torch
from models.multi_head_attention import MultiHeadAttention

# Model configuration
batch_size = 2
block_size = 8
n_embd = 32
num_heads = 4
head_size = n_embd // num_heads

# Create random input
x = torch.randn(batch_size, block_size, n_embd)

# Create multi-head attention
attention = MultiHeadAttention(
    num_heads,
    head_size,
    n_embd,
    block_size
)

# Pass input through the model
output = attention(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("\nMulti-head attention test successful!")

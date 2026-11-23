
import torch
from models.feed_forward import FeedForward

# Configuration
batch_size = 2
block_size = 8
n_embd = 32

# Create random input
x = torch.randn(batch_size, block_size, n_embd)

# Create Feed-Forward Network
mlp = FeedForward(n_embd)

# Pass input through network
output = mlp(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

print("\nFeed-Forward Network test successful!")

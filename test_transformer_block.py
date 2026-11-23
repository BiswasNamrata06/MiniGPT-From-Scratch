
import torch

from models.transformer_block import TransformerBlock

# Configuration
batch_size = 2
block_size = 8
n_embd = 32
num_heads = 4

# Random input
x = torch.randn(batch_size, block_size, n_embd)

# Create Transformer Block
block = TransformerBlock(
    n_embd,
    num_heads,
    block_size
)

# Pass input through the block
output = block(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

print("\nTransformer Block test successful!")

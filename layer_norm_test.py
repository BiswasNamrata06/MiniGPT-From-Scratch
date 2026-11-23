
import torch
import torch.nn as nn

# Create sample input
x = torch.tensor([
    [10.0, 20.0, 30.0, 40.0],
    [5.0, 15.0, 25.0, 35.0]
])

# Create Layer Normalization
layer_norm = nn.LayerNorm(4)

# Apply normalization
output = layer_norm(x)

print("Original input:")
print(x)

print("\nNormalized output:")
print(output)

print("\nMean of each row:")
print(output.mean(dim=-1))

print("\nVariance of each row:")
print(output.var(dim=-1, unbiased=False))

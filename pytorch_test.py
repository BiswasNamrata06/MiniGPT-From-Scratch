
import torch

# Check PyTorch version
print("PyTorch Version:", torch.__version__)

# Create our first tensor
x = torch.tensor([1, 2, 3, 4, 5])

print("Our First Tensor:", x)

# Perform mathematical operations
print("Tensor + 10:", x + 10)
print("Tensor * 2:", x * 2)

# Check the shape
print("Tensor Shape:", x.shape)

# Check the device
print("Device:", x.device)

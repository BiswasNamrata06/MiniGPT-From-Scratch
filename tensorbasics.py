
import torch

# 0D Tensor (Scalar)
scalar = torch.tensor(5)

# 1D Tensor (Vector)
vector = torch.tensor([1, 2, 3, 4])

# 2D Tensor (Matrix)
matrix = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

# 3D Tensor
tensor_3d = torch.tensor([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])

# Display tensors and their properties
print("Scalar:", scalar)
print("Scalar Shape:", scalar.shape)
print("Scalar Dimensions:", scalar.ndim)

print("\nVector:", vector)
print("Vector Shape:", vector.shape)
print("Vector Dimensions:", vector.ndim)

print("\nMatrix:", matrix)
print("Matrix Shape:", matrix.shape)
print("Matrix Dimensions:", matrix.ndim)

print("\n3D Tensor:", tensor_3d)
print("3D Tensor Shape:", tensor_3d.shape)
print("3D Tensor Dimensions:", tensor_3d.ndim)

print("Accessing Elemnts")
x = torch.tensor([
    [10, 20, 30],
    [40, 50, 60]
])

print(x[0])
print(x[1])

print(x[0][1])
print(x[1][2])

print("MAreix Mutiplication")
A = torch.tensor([
    [1, 2],
    [3, 4]
])

B = torch.tensor([
    [5, 6],
    [7, 8]
])

result = torch.matmul(A, B)

print(result)

print("REshaping Tensors")
import torch

x = torch.tensor([1, 2, 3, 4, 5, 6])

y = x.reshape(2, 3)

print("Original:", x)
print("Reshaped:", y)
print("New Shape:", y.shape)
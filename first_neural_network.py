
import torch
import torch.nn as nn

# 1. Training data
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
Y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

# 2. Create a simple neural network
model = nn.Linear(1, 1)

# 3. Define loss function
criterion = nn.MSELoss()

# 4. Define optimizer
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

# 5. Training loop
for epoch in range(1000):

    # Forward propagation
    predictions = model(X)

    # Calculate loss
    loss = criterion(predictions, Y)

    # Clear old gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update weights
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch [{epoch + 1}/1000], "
            f"Loss: {loss.item():.4f}"
        )

# 6. Test the trained model
test_input = torch.tensor([[5.0]])

with torch.no_grad():
    prediction = model(test_input)

print("\nPrediction for x = 5:", prediction.item())

# Display learned parameters
for name, parameter in model.named_parameters():
    print(name, parameter.data)

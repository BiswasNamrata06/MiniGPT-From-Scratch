
import torch
import torch.nn as nn
from models.bigram_model import BigramLanguageModel

# 1. Training text
text = "hello world"

# 2. Create vocabulary
characters = sorted(set(text))
vocab_size = len(characters)

char_to_int = {
    ch: i for i, ch in enumerate(characters)
}

int_to_char = {
    i: ch for i, ch in enumerate(characters)
}

# 3. Encode the text
data = torch.tensor(
    [char_to_int[ch] for ch in text],
    dtype=torch.long
)

# 4. Set training parameters
batch_size = 4
block_size = 8
learning_rate = 0.01
training_steps = 1000

# 5. Create a batch of examples
def get_batch():

    # Random starting positions
    ix = torch.randint(
        len(data) - block_size,
        (batch_size,)
    )

    # Input and next-character targets
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x, y


# 6. Create the model
model = BigramLanguageModel(vocab_size)

# 7. Create loss function and optimiser
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

# 8. Training loop
for step in range(training_steps):

    # Get training examples
    x, y = get_batch()

    # Forward pass
    logits = model(x)

    # Calculate loss
    B, T, C = logits.shape

    logits = logits.view(B * T, C)
    targets = y.view(B * T)

    loss = loss_function(logits, targets)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()

    # Update model weights
    optimizer.step()

    if (step + 1) % 100 == 0:
        print(
            f"Step {step+1}, Loss: {loss.item():.4f}"
        )


# 9. Generate text
print("\nGenerated text:")

context = torch.zeros((1, 1), dtype=torch.long)

for _ in range(100):

    # Predict next character
    logits = model(context)

    # Get scores for the last position
    logits = logits[:, -1, :]

    # Convert scores into probabilities
    probabilities = torch.softmax(logits, dim=-1)

    # Sample the next character
    next_character = torch.multinomial(
        probabilities, num_samples=1
    )

    # Add new character to context
    context = torch.cat(
        (context, next_character), dim=1
    )

# Decode generated character IDs
generated = "".join(
    int_to_char[i.item()] for i in context[0]
)

print(generated)


import torch
import torch.nn as nn
from models.bigram_model import BigramLanguageModel

# 1. Load Shakespeare dataset
with open("data/tinyshakespeare.txt", "r", encoding="utf-8") as file:
    text = file.read()

print("Dataset characters:", len(text))

# 2. Create vocabulary
characters = sorted(set(text))
vocab_size = len(characters)

char_to_int = {ch: i for i, ch in enumerate(characters)}
int_to_char = {i: ch for i, ch in enumerate(characters)}

# 3. Encode entire dataset
data = torch.tensor(
    [char_to_int[ch] for ch in text],
    dtype=torch.long
)

# 4. Split into training and validation data
split_index = int(0.9 * len(data))

train_data = data[:split_index]
val_data = data[split_index:]

print("Vocabulary size:", vocab_size)
print("Training characters:", len(train_data))
print("Validation characters:", len(val_data))

# 5. Training settings
batch_size = 64
block_size = 8
learning_rate = 0.001
training_steps = 5000
eval_interval = 500
eval_iters = 100

# 6. Create batches
def get_batch(split):

    source = train_data if split == "train" else val_data

    ix = torch.randint(
        len(source) - block_size,
        (batch_size,)
    )

    x = torch.stack([source[i:i+block_size] for i in ix])
    y = torch.stack([source[i+1:i+block_size+1] for i in ix])

    return x, y


# 7. Evaluate model performance
@torch.no_grad()
def estimate_loss():

    model.eval()

    results = {}

    for split in ["train", "val"]:

        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):

            x, y = get_batch(split)

            logits = model(x)

            B, T, C = logits.shape

            loss = loss_function(
                logits.view(B * T, C),
                y.view(B * T)
            )

            losses[k] = loss.item()

        results[split] = losses.mean().item()

    model.train()

    return results


# 8. Create model and optimiser
model = BigramLanguageModel(vocab_size)

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

# 9. Train the model
print("\nStarting training...\n")

for step in range(training_steps):

    if step % eval_interval == 0:
        losses = estimate_loss()

        print(
            f"Step {step}: "
            f"Training Loss = {losses['train']:.4f}, "
            f"Validation Loss = {losses['val']:.4f}"
        )

    x, y = get_batch("train")

    logits = model(x)

    B, T, C = logits.shape

    logits = logits.view(B * T, C)
    targets = y.view(B * T)

    loss = loss_function(logits, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 10. Final evaluation
final_losses = estimate_loss()

print("\nTraining complete!")
print("Final Training Loss:", final_losses["train"])
print("Final Validation Loss:", final_losses["val"])

# 11. Generate Shakespeare-style text
print("\nGenerated text:\n")

model.eval()

context = torch.zeros((1, 1), dtype=torch.long)

with torch.no_grad():

    for _ in range(500):

        # Bigram model only needs the latest character
        logits = model(context[:, -1:])

        # Convert logits into probabilities
        probabilities = torch.softmax(logits[:, -1, :], dim=-1)

        # Sample next character
        next_character = torch.multinomial(
            probabilities, num_samples=1
        )

        # Append to context
        context = torch.cat((context, next_character), dim=1)

generated_text = "".join(
    int_to_char[i.item()] for i in context[0]
)

print(generated_text)

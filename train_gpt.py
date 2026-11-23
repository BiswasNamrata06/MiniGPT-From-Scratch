
import torch
import torch.nn as nn
from models.gpt_model import GPTLanguageModel

# 1. Configuration
batch_size = 16
block_size = 64
max_iters = 2000
eval_interval = 200
learning_rate = 0.001
eval_iters = 20

n_embd = 64
num_heads = 4
num_layers = 2

device = "cpu"

# 2. Load Shakespeare dataset
with open("data/tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 3. Create character vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(numbers):
    return "".join([itos[int(i)] for i in numbers])

data = torch.tensor(encode(text), dtype=torch.long)

# 4. Split training and validation data
split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

# 5. Create batches
def get_batch(split_name):
    source = train_data if split_name == "train" else val_data

    ix = torch.randint(
        len(source) - block_size - 1,
        (batch_size,)
    )

    x = torch.stack([source[i:i + block_size] for i in ix])
    y = torch.stack([source[i + 1:i + block_size + 1] for i in ix])

    return x.to(device), y.to(device)

# 6. Create model
model = GPTLanguageModel(
    vocab_size=vocab_size,
    n_embd=n_embd,
    block_size=block_size,
    num_heads=num_heads,
    num_layers=num_layers
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

# 7. Evaluate training and validation loss
@torch.no_grad()
def estimate_loss():
    model.eval()
    results = {}

    for split_name in ["train", "val"]:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            x, y = get_batch(split_name)
            logits, loss = model(x, y)
            losses[k] = loss.item()

        results[split_name] = losses.mean().item()

    model.train()
    return results

# 8. Training loop

# 8. Training loop
for step in range(max_iters):

    if step % eval_interval == 0:
        losses = estimate_loss()

        print(
            f"Step {step}: "
            f"Train Loss {losses['train']:.4f}, "
            f"Validation Loss {losses['val']:.4f}"
        )

    x, y = get_batch("train")

    logits, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    # Prevent excessively large gradients
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    # IMPORTANT: Update weights inside the loop
    optimizer.step()



# 9. Save model checkpoint
torch.save({
    "model_state_dict": model.state_dict(),
    "vocab_size": vocab_size,
    "stoi": stoi,
    "itos": itos,
    "config": {
        "n_embd": n_embd,
        "block_size": block_size,
        "num_heads": num_heads,
        "num_layers": num_layers
    }
}, "outputs/minigpt_checkpoint.pth")

print("\nModel saved successfully!")

# 10. Generate Shakespeare-like text
model.eval()

start = torch.tensor(
    [[stoi["\n"]]],
    dtype=torch.long,
    device=device
)

generated = model.generate(start, max_new_tokens=300)

print("\nGenerated text:\n")
print(decode(generated[0].tolist()))

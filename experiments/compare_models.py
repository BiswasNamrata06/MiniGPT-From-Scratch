
import torch
import torch.nn.functional as F

from models.bigram_model import BigramLanguageModel
from models.gpt_model import GPTLanguageModel

# Configuration
batch_size = 16
block_size = 64
training_steps = 500
learning_rate = 0.001
eval_iters = 20

device = "cpu"

# Load dataset
with open("data/tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Character vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}

data = torch.tensor(
    [stoi[c] for c in text],
    dtype=torch.long
)

# Split dataset
split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]


def get_batch(source):
    ix = torch.randint(
        len(source) - block_size - 1,
        (batch_size,)
    )

    x = torch.stack([
        source[i:i + block_size] for i in ix
    ])

    y = torch.stack([
        source[i + 1:i + block_size + 1] for i in ix
    ])

    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model):
    model.eval()
    losses = []

    for _ in range(eval_iters):
        x, y = get_batch(val_data)

        logits, _ = model(x, y) if isinstance(
            model, GPTLanguageModel
        ) else (model(x), None)

        if isinstance(model, GPTLanguageModel):
            # GPT returns flattened logits when targets are provided
            loss = F.cross_entropy(
                logits,
                y.view(-1)
            )
        else:
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1)
            )

        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def train_model(model, name):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    model.train()

    for step in range(training_steps):

        x, y = get_batch(train_data)

        if isinstance(model, GPTLanguageModel):
            logits, loss = model(x, y)
        else:
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1)
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0
        )

        optimizer.step()

        if (step + 1) % 100 == 0:
            print(f"{name} step {step + 1}: loss {loss.item():.4f}")

    validation_loss = evaluate(model)

    print(f"{name} validation loss: {validation_loss:.4f}")

    return validation_loss


# Create both models
bigram = BigramLanguageModel(vocab_size)

gpt = GPTLanguageModel(
    vocab_size=vocab_size,
    n_embd=64,
    block_size=block_size,
    num_heads=4,
    num_layers=2
)

# Train and compare
print("\nTraining Bigram Model...")
bigram_loss = train_model(bigram, "Bigram")

print("\nTraining MiniGPT...")
gpt_loss = train_model(gpt, "MiniGPT")

print("\n========== FINAL COMPARISON ==========")
print(f"Bigram validation loss: {bigram_loss:.4f}")
print(f"MiniGPT validation loss: {gpt_loss:.4f}")

if gpt_loss < bigram_loss:
    print("MiniGPT achieved lower validation loss!")
else:
    print("Bigram achieved lower validation loss in this run.")

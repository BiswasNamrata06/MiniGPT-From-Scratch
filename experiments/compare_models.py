import torch
import torch.nn.functional as F

from models.bigram_model import BigramLanguageModel
from models.gpt_model import GPTLanguageModel

torch.manual_seed(1337)

# Configuration
batch_size = 16
block_size = 64
training_steps = 3000
eval_iters = 200
log_interval = 500
learning_rates = {"Bigram": 0.01, "MiniGPT": 0.001}

device = "cpu"

# Load dataset
with open("data/tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}

data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]


def get_batch(source):
    ix = torch.randint(len(source) - block_size - 1, (batch_size,))
    x = torch.stack([source[i:i + block_size] for i in ix])
    y = torch.stack([source[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# Fixed evaluation batches: both models are scored on exactly the same text
val_batches = [get_batch(val_data) for _ in range(eval_iters)]
train_eval_batches = [get_batch(train_data) for _ in range(eval_iters)]


def compute_loss(model, x, y):
    out = model(x)
    # GPT returns (logits, loss); Bigram returns logits only
    logits = out[0] if isinstance(out, tuple) else out
    return F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))


@torch.no_grad()
def evaluate(model, batches):
    model.eval()
    losses = [compute_loss(model, x, y).item() for x, y in batches]
    model.train()
    return sum(losses) / len(losses)


def train_model(model, name):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rates[name])
    model.train()

    for step in range(1, training_steps + 1):
        x, y = get_batch(train_data)
        loss = compute_loss(model, x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % log_interval == 0:
            print(f"{name} step {step}: batch loss {loss.item():.4f}")

    train_loss = evaluate(model, train_eval_batches)
    val_loss = evaluate(model, val_batches)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{name}: train {train_loss:.4f} | val {val_loss:.4f} | params {n_params:,}")
    return val_loss


print(f"Random-guess loss (ln {vocab_size}): {torch.log(torch.tensor(float(vocab_size))).item():.4f}")

print("\nTraining Bigram model...")
bigram_loss = train_model(BigramLanguageModel(vocab_size), "Bigram")

print("\nTraining MiniGPT...")
gpt = GPTLanguageModel(
    vocab_size=vocab_size,
    n_embd=64,
    block_size=block_size,
    num_heads=4,
    num_layers=2
)
gpt_loss = train_model(gpt, "MiniGPT")

print("\n========== FINAL COMPARISON ==========")
print(f"Bigram validation loss:  {bigram_loss:.4f}")
print(f"MiniGPT validation loss: {gpt_loss:.4f}")
print(f"Reduction in validation loss: {(bigram_loss - gpt_loss) / bigram_loss * 100:.1f}%")
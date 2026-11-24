
import torch
from models.gpt_model import GPTLanguageModel

# Load checkpoint
checkpoint = torch.load(
    "outputs/minigpt_checkpoint.pth",
    map_location="cpu",
    weights_only=True
)

# Recover configuration
config = checkpoint["config"]
vocab_size = checkpoint["vocab_size"]

stoi = checkpoint["stoi"]
itos = checkpoint["itos"]

# Recreate model
model = GPTLanguageModel(
    vocab_size=vocab_size,
    n_embd=config["n_embd"],
    block_size=config["block_size"],
    num_heads=config["num_heads"],
    num_layers=config["num_layers"]
)

# Load trained weights
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

# Encoding and decoding
def encode(text):
    unknown = sorted(set(c for c in text if c not in stoi))
    if unknown:
        raise ValueError(f"Characters not in the vocabulary: {unknown}")
    return [stoi[c] for c in text]

def decode(numbers):
    return "".join([itos[int(i)] for i in numbers])

# Choose a starting prompt


prompt = "ROMEO:"
context = torch.tensor([encode(prompt)])

settings = [
    {"temperature": 0.7, "top_k": 3},
    {"temperature": 1.0, "top_k": 10},
    {"temperature": 1.2, "top_k": 20},
    {"temperature": 1.0, "top_k": None}
]

for setting in settings:

    generated = model.generate(
        context,
        max_new_tokens=200,
        temperature=setting["temperature"],
        top_k=setting["top_k"]
    )

    print("\nSettings:", setting)
    print(decode(generated[0].tolist()))

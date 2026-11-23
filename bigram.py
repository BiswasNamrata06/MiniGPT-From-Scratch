
import torch

# Our sample training text
text = "hello world"

# Create vocabulary
characters = sorted(list(set(text)))
vocab_size = len(characters)

# Create mappings
char_to_int = {
    character: index
    for index, character in enumerate(characters)
}

int_to_char = {
    index: character
    for index, character in enumerate(characters)
}

# Encode the text
encoded = torch.tensor(
    [char_to_int[c] for c in text],
    dtype=torch.long
)

print("Text:", text)
print("Vocabulary:", characters)
print("Vocabulary size:", vocab_size)
print("Encoded text:", encoded)

# Create input and target
x = encoded[:-1]
y = encoded[1:]

print("\nInput:", x)
print("Target:", y)

# Decode and display the pairs
for i in range(len(x)):
    print(
        f"Input: {int_to_char[x[i].item()]!r} "
        f"-> Target: {int_to_char[y[i].item()]!r}"
    )

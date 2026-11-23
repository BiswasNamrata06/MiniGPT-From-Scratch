
import torch

text = "hello"

# Build vocabulary
characters = sorted(set(text))

# Character-to-number mapping
char_to_int = {
    character: index
    for index, character in enumerate(characters)
}

# Convert text into numbers
encoded = [char_to_int[c] for c in text]

print("Vocabulary:", characters)
print("Mapping:", char_to_int)
print("Encoded text:", encoded)

# Create input-target pairs
for i in range(len(encoded) - 1):
    input_character = encoded[i]
    target_character = encoded[i + 1]

    print(
        "Input:", text[i],
        "| Target:", text[i + 1],
        "| Numbers:", input_character, "->", target_character
    )

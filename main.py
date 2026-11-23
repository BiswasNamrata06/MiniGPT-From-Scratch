
# MiniGPT - Character Level Tokenizer

text = "Artificial Intelligence"

# Step 1: Find all unique characters
characters = sorted(list(set(text)))

# Step 2: Calculate vocabulary size
vocab_size = len(characters)

print("Characters:", characters)
print("Vocabulary Size:", vocab_size)

# Step 3: Create character-to-integer mapping
char_to_int = {
    character: index
    for index, character in enumerate(characters)
}

# Step 4: Create integer-to-character mapping
int_to_char = {
    index: character
    for index, character in enumerate(characters)
}

# Step 5: Encoding function
def encode(text):
    return [char_to_int[character] for character in text]

# Step 6: Decoding function
def decode(numbers):
    return "".join([int_to_char[number] for number in numbers])

# Step 7: Test our tokenizer
encoded_text = encode(text)
decoded_text = decode(encoded_text)

print("Original Text:", text)
print("Encoded Text:", encoded_text)
print("Decoded Text:", decoded_text)

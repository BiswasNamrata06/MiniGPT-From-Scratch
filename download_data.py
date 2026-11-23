
from pathlib import Path
from urllib.request import urlopen

# Dataset URL
url = (
    "https://raw.githubusercontent.com/"
    "karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)

# Download dataset
print("Downloading Shakespeare dataset...")

with urlopen(url) as response:
    text = response.read().decode("utf-8")

# Save inside data folder
data_folder = Path("data")
data_folder.mkdir(exist_ok=True)

file_path = data_folder / "tinyshakespeare.txt"

file_path.write_text(text, encoding="utf-8")

print("Download complete!")
print("Characters:", len(text))
print("Saved to:", file_path)
print("\nFirst 500 characters:\n")
print(text[:500])

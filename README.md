# MiniGPT — From Scratch

A character-level, decoder-only GPT language model implemented from scratch in PyTorch and trained on the Tiny Shakespeare dataset. The project is inspired by Andrej Karpathy's *Let's build GPT: from scratch, in code, spelled out*.

It covers the full path from raw text to generated text: character tokenization, embeddings, causal self-attention, Transformer blocks, next-character prediction, training, checkpointing and sampling. Each concept was first built in a small standalone script (see [Learning Path](#learning-path)) before being assembled into the final model.

## Features

- Character-level tokenization (65-character vocabulary)
- Token embeddings and learned positional embeddings
- Causal (masked) multi-head self-attention
- Feed-forward network, residual connections and layer normalization (pre-LN, as in GPT-2)
- Next-character prediction with cross-entropy loss
- AdamW optimizer with gradient clipping and dropout
- Checkpoint saving and loading (weights, vocabulary and configuration)
- Temperature and top-k sampling
- Bigram vs Transformer comparison experiment

## Model Architecture

```text
Input text
    |
Character tokenization
    |
Token embedding + learned positional embedding
    |
N x Transformer block
    |    +-- LayerNorm -> causal multi-head self-attention -> residual add
    |    +-- LayerNorm -> feed-forward network (ReLU, 4x wide) -> residual add
    |
Final LayerNorm
    |
Linear layer (embedding size -> vocabulary size)
    |
Softmax -> sample next character
```

### Configuration used for training (`train_gpt.py`)

| Setting | Value |
|---|---|
| Vocabulary size | 65 characters |
| Context length (`block_size`) | 64 characters |
| Embedding size (`n_embd`) | 64 |
| Attention heads / head size | 4 / 16 |
| Transformer blocks | 2 |
| Dropout | 0.2 |
| Parameters | 112,193 |
| Batch size | 16 |
| Training steps | 2,000 |
| Optimizer | AdamW, learning rate 0.001, gradient clipping at 1.0 |

## Dataset

[Tiny Shakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare): a concatenation of Shakespeare's plays with 1,115,394 characters and 65 unique characters. The first 90% is used for training and the last 10% for validation.

## Tech Stack

- Python 3
- PyTorch

## Project Structure

```text
MiniGPT-From-Scratch/
├── data/
│   └── tinyshakespeare.txt        # created by download_data.py
├── models/
│   ├── bigram_model.py            # baseline model
│   ├── multi_head_attention.py    # single attention head + multi-head attention
│   ├── feed_forward.py            # feed-forward network
│   ├── transformer_block.py       # attention + feed-forward with residuals and layer norm
│   └── gpt_model.py               # full MiniGPT model with generate()
├── experiments/
│   └── compare_models.py          # Bigram vs MiniGPT comparison
├── outputs/
│   └── minigpt_checkpoint.pth     # created by train_gpt.py (not tracked by git)
├── train_gpt.py                   # trains MiniGPT and saves the checkpoint
├── generate.py                    # loads the checkpoint and generates text
├── download_data.py               # downloads the dataset
├── train_shakespeare_bigram.py    # trains the Bigram baseline on Shakespeare
├── train_bigram.py                # tiny Bigram demo on "hello world"
├── test_*.py                      # smoke tests for each module
├── (learning scripts)             # one-concept scripts, see Learning Path
├── requirements.txt
└── README.md
```

## Installation

Run all commands from the project root.

### 1. Clone the repository

```bash
git clone https://github.com/BiswasNamrata06/MiniGPT-From-Scratch.git
cd MiniGPT-From-Scratch
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

For a CPU-only PyTorch installation, if needed:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 4. Download the dataset

```bash
python download_data.py
```

## Usage

### Train

```bash
python train_gpt.py
```

The script loads the dataset, builds the character vocabulary, splits the data 90/10, trains for 2,000 steps, prints training and validation loss every 200 steps, saves a checkpoint to `outputs/minigpt_checkpoint.pth`, and prints a 300-character sample.

At step 0 the loss should be close to 4.17, which is ln(65), the loss of guessing uniformly over 65 characters.

### Generate text

Training must be run first, because `generate.py` loads the saved checkpoint.

```bash
python generate.py
```

The script samples from the prompt `ROMEO:` under four settings:

| Temperature | Top-k | Effect |
|---|---|---|
| 0.7 | 3 | Most conservative: cleaner spelling, more repetition |
| 1.0 | 10 | Balanced |
| 1.2 | 20 | More random |
| 1.0 | none | Samples from the full distribution |

Edit `prompt` and `settings` in `generate.py` to experiment. Prompts may only contain characters that appear in the 65-character vocabulary.

### Compare Bigram and MiniGPT

```bash
python -m experiments.compare_models
```

Run this from the project root (the `-m` form is required so that the `models` package can be imported).

## Results

<!-- Replace every TODO with the numbers printed by `python -m experiments.compare_models`, then delete this comment. -->

| Model | Parameters | Training loss | Validation loss |
|---|---:|---:|---:|
| Random guessing (ln 65) | n/a | 4.1744 | 4.1744 |
| Bigram language model | 4,225 | 2.4584 | 2.4869 |
| MiniGPT | 112,193 | 1.8636 | 1.9687 |

Experiment setup: both models were trained for 3,000 steps (Bigram with learning rate 0.01, MiniGPT with 0.001), using AdamW, batch size 16, context length 64 and a fixed random seed. Both were evaluated on the same 200 validation batches.

The Bigram model sees only the single previous character, while MiniGPT can attend to the previous 64 characters. The gap between their validation losses shows what attention and longer context contribute. Note that MiniGPT also has about 27 times more parameters than the Bigram model, so part of its advantage comes from size.

## Reproducibility

- Random seed is fixed at 1337 in the training and comparison scripts.
- Results can still differ slightly between machines and PyTorch versions.
- Validation loss is averaged over a limited number of random batches, so small differences between runs are expected.

## Hardware

Developed and trained on CPU (Intel laptop). The scripts set `device = "cpu"`. The model is deliberately small so that training takes minutes on consumer hardware.

## Limitations

- The model sees only the last 64 characters, so it cannot keep track of long-range structure.
- It is small (about 112 thousand parameters) and trained on about one million characters.
- It learns the style and spelling patterns of Shakespeare, not the meaning of the text. Generated text looks like Shakespeare but does not make sense.
- Positional embeddings are learned, so the context length cannot be extended without retraining.
- Generation recomputes the whole context at every step (no key-value cache).

## Learning Path

The project was built up one concept at a time. These standalone scripts show each step:

| Step | Script |
|---|---|
| Character tokenizer | `main.py` |
| Next-character input/target pairs | `next_character.py` |
| Tensors | `tensorbasics.py` |
| A first neural network | `first_neural_network.py` |
| Softmax | `softmax_test.py` |
| Embeddings | `embedding_test.py` |
| Positional embeddings | `positional_embedding.py` |
| Self-attention | `self_attention.py` |
| Causal attention (masking) | `causal_attention.py` |
| Layer normalization | `layer_norm_test.py` |
| Bigram model | `train_bigram.py`, `train_shakespeare_bigram.py` |
| Multi-head attention, feed-forward, Transformer block, GPT | `models/` |

## Future Improvements

- Byte Pair Encoding (BPE) tokenization
- Learning-rate warm-up and cosine decay
- A larger dataset and model
- Batched attention heads (or PyTorch's built-in scaled dot-product attention)
- GELU activation and dropout on attention weights
- Key-value cache for faster generation
- GPU training
- An interactive text-generation interface

## References

- Andrej Karpathy, [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html)

## Author

**Namrata Biswas**, Computer Science Engineering 

GitHub: [BiswasNamrata06](https://github.com/BiswasNamrata06)
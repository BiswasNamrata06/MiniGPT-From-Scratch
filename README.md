# MiniGPT — From Scratch

A character-level, decoder-only GPT language model implemented in PyTorch, inspired by Andrej Karpathy's *Let's build GPT: from scratch, in code, spelled out*.

This project explores the fundamentals of Large Language Models (LLMs), including tokenization, embeddings, causal self-attention, Transformer blocks, next-token prediction, model training, and autoregressive text generation.

## Features

- Character-level text tokenization
- Token and positional embeddings
- Causal self-attention
- Multi-head attention
- Feed-forward neural networks
- Residual connections and Layer Normalization
- Decoder-only Transformer architecture
- Next-character prediction using cross-entropy loss
- Temperature-based sampling
- Top-K sampling
- Model checkpoint saving and loading
- Bigram vs Transformer comparison experiment

## Model Architecture

The model follows a decoder-only Transformer architecture.

```text
Input Text
    |
Character Tokenization
    |
Token Embeddings + Positional Embeddings
    |
Transformer Blocks
    |
    +-- Causal Multi-Head Self-Attention
    |
    +-- Feed-Forward Network
    |
    +-- Residual Connections
    |
    +-- Layer Normalization
    |
Final Layer Normalization
    |
Linear Language Model Head
    |
Next-Token Prediction
    |
Generated Text
```

## Dataset

The model is trained on the Tiny Shakespeare dataset, a collection of Shakespeare's works commonly used for character-level language modelling experiments.

## Tech Stack

- Python
- PyTorch
- NumPy
- Matplotlib

## Project Structure

```text
MiniGPT-From-Scratch/
|
├── data/
|   └── tinyshakespeare.txt
|
├── models/
|   ├── bigram_model.py
|   ├── multi_head_attention.py
|   ├── feed_forward.py
|   ├── transformer_block.py
|   └── gpt_model.py
|
├── outputs/
|   └── minigpt_checkpoint.pth
|
├── train_gpt.py
├── generate.py
├── download_data.py
├── compare_models.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/BiswasNamrata06/MiniGPT-From-Scratch.git
cd MiniGPT-From-Scratch
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

For CPU-only PyTorch installation, if needed:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 4. Download the dataset

```bash
python download_data.py
```

## Training

Train the GPT model using:

```bash
python train_gpt.py
```

The training script:
- Loads the Shakespeare dataset.
- Creates training and validation splits.
- Generates batches of input and target sequences.
- Performs next-token prediction.
- Calculates cross-entropy loss.
- Updates model parameters using AdamW.
- Saves a trained checkpoint in the outputs directory.

## Text Generation

After training, generate text using:

```bash
python generate.py
```

You can experiment with:
- Different starting prompts
- Temperature values
- Top-K sampling settings
- Generation lengths

## Model Comparison

The project also includes an experiment comparing a simple Bigram model against the Transformer-based MiniGPT model.

Run:

```bash
python compare_models.py
```

Compare the validation losses to investigate how access to a longer context and attention mechanisms affects next-character prediction.
## Model Performance

MiniGPT was evaluated against a simple Bigram Language Model using the Tiny Shakespeare dataset.

| Model | Validation Loss |
|---|---:|
| Bigram Language Model | 4.0518 |
| MiniGPT | 2.3859 |

**Result:** MiniGPT achieved approximately 41.1% lower validation loss than the Bigram baseline.

This demonstrates the advantage of using causal self-attention and Transformer blocks to learn contextual patterns in character sequences.

*Note: Results are from the current experimental configuration and training run.*

## Hardware

Developed and tested using:
- Python 3
- PyTorch
- Intel Iris Xe integrated graphics
- CPU-based training

The implementation uses a relatively small Transformer configuration to make experimentation possible on consumer hardware.

## Learning Objectives

This project was developed to understand the internal working of GPT-style language models rather than relying on pre-trained language-model APIs.

Key concepts explored include:
- Neural network fundamentals
- Character-level language modelling
- Embedding layers
- Attention mechanisms
- Transformer architecture
- Autoregressive generation
- Model training and evaluation

## Future Improvements

- Byte Pair Encoding (BPE) tokenization
- Learning-rate scheduling
- More efficient attention implementations
- Larger training datasets
- Training performance optimisation
- Interactive text-generation interface

## References

- Andrej Karpathy — [Let's build GPT: from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## Author

**Namrata Biswas**

Computer Science Engineering Student

GitHub: [BiswasNamrata06](https://github.com/BiswasNamrata06)
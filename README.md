# LLM Training Pipeline (MiniGPT)

A proof-of-concept, end-to-end pipeline for training a small Transformer-style language model on a cleaned subset of English Wikipedia. Built to understand and explain the full lifecycle behind modern LLMs: **dataset → cleaning → tokenization → training → checkpointing/logging → text generation**.

---

## Why this exists

I built this as a “from scratch” LLM proof of concept to prove I could, and to develop the same kind of practical understanding I previously built through a graduate computer vision project. With LLMs being so prominent right now, I wanted to be able to describe them clearly—from the training pipeline up.

---

## Repository layout

- `wikipedia_llm.ipynb` — main notebook (data prep, tokenization, model, training, generation)
- `tokenizer/` — Byte-Level BPE artifacts (`vocab.json`, `merges.txt`)
- `tokenizer_hf/` — exported Hugging Face tokenizer format (reloadable via `from_pretrained`)
- `training_log.csv` — training/validation loss history (CSV)
- `requirements.txt` — dependencies

---

## Key features

- **Dataset ingestion (Hugging Face Datasets)** and **custom cleaning**
- **Byte-Level BPE tokenizer** (loaded from `tokenizer/`, exported to `tokenizer_hf/`)
- **Next-token prediction dataset** built from a long token stream and sliced into fixed windows
- **Checkpointing + resume** (`checkpoint.pt`)
- **Durable CSV logging** (`training_log.csv` with flush + fsync)
- **Text generation** with **temperature** and **top-k** sampling

---

## What I built vs. what I used

### Built in this project
- Checkpointing (save + resume)
- Training metric logging to CSV (durable writes)
- Data cleaning / preprocessing
- Dataset slicing into fixed-length sequences for next-token prediction
- Sampling-based text generation helper

### Libraries relied on
- **Hugging Face Datasets**: dataset access (Wikipedia snapshot)
- **PyTorch**: model/training primitives + **AdamW** optimizer (weight decay handling)
- **tokenizers / transformers**: Byte-Level BPE tokenizer + `PreTrainedTokenizerFast` wrapper

---

## How it works

### 1) Data: Wikipedia subset + cleaning
The notebook loads a subset of English Wikipedia:
- dataset: `wikimedia/wikipedia`
- config: `20231101.en`
- split: `train[:10%]`

Then it:
- removes wiki-style headings (e.g., `== Heading ==`)
- removes bracketed content (often citations)
- normalizes whitespace
- filters out short entries (keeps text > 200 characters)

### 2) Tokenization: Byte-Level BPE
A Byte-Level BPE tokenizer is loaded from:
- `tokenizer/vocab.json`
- `tokenizer/merges.txt`

Then wrapped as a Hugging Face fast tokenizer and saved to `tokenizer_hf/`.

### 3) Build a next-token dataset
Text is tokenized and concatenated into one long token stream and saved to `train.pt`.  
The dataset is then sliced into fixed-length blocks:
- `block_size = 1024`
- inputs `x = ids[start:end]`
- targets `y = ids[start+1:end+1]`

### 4) Model: MiniGPT (small Transformer)
Model highlights:
- token + positional embeddings
- stack of `nn.TransformerEncoderLayer` blocks
- layer norm + linear head to vocabulary logits

Default config in the notebook:
- `n_embd = 128`
- `n_layer = 4`
- `nhead = 4`
- `block_size = 1024`

> Note: This is intentionally small and understandable end-to-end. A natural next step is adding an explicit causal attention mask for strictly autoregressive attention.

### 5) Training: resume + durable logging
Training uses:
- `AdamW(lr=3e-4)`
- `CrossEntropyLoss`
- train/validation split (`val_split = 0.1`)
- checkpoints saved to `checkpoint.pt`
- metrics appended to `training_log.csv` with flush + fsync for crash-safe progress tracking

---

## Quickstart

### 1) Create an environment
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```
## Notes
## Hardware notes (early NVIDIA Blackwell support)

This project was developed early in the NVIDIA Blackwell lifecycle, and training on my GPU was initially blocked by ecosystem support (the GPU wasn’t recognized by the software stack). The fix was version alignment across the driver + CUDA + PyTorch build:

- Confirm driver + GPU visibility:
  - `nvidia-smi`
- Use a driver version that supports the architecture
- Install a CUDA toolkit compatible with the driver
- Install a PyTorch build that recognizes the new architecture  
  - (Often newer releases/nightlies early on)

Validate with:

- `torch.cuda.is_available()`
- Basic CUDA tensor ops before launching long runs

**Key lesson:** For brand-new GPU architectures, the model code is often fine—the blocker is usually driver/toolkit/framework support catching up.

---

## Quick interview answers

### 1) What was the goal of this project?
Build a proof-of-concept LLM end-to-end to prove I could, and to develop the same kind of practical understanding I previously built through a graduate computer vision project. With LLMs being so prominent right now, I wanted to be able to explain them clearly from the training pipeline up.

### 2) What did you build vs. what libraries did you rely on?
I implemented the practical training infrastructure—checkpointing and logging—plus the data cleaning workflow. Hugging Face provided the raw dataset access, and PyTorch provided the training primitives and AdamW optimizer (including weight decay handling).

### 3) What was the biggest technical challenge, and how did you solve it?
The hardest part was getting training working on an NVIDIA Blackwell GPU early in its lifecycle, when support across the driver/CUDA/PyTorch stack was still catching up. I resolved it by aligning versions across the NVIDIA driver, CUDA, and a PyTorch build that recognized the new architecture, then validating GPU availability in the runtime before running longer training jobs.

### 4) If you had one more week, what would you improve next?
I’d add a clean CLI and lightweight data visualizations (loss curves, throughput, eval snapshots) so convergence and training behavior are easier to inspect at a glance.

# Attenta

[![Rust](https://img.shields.io/badge/Rust-1.70%2B-orange)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Vaswani%202017-blue)](https://arxiv.org/abs/1706.03762)

**Attenta** — a from-scratch Rust implementation of the Transformer architecture from the
landmark paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017).
Trained on Indian labor law documents for legal document processing.

## What is Attenta?

Attenta is a complete, self-contained implementation of the Transformer model — no PyTorch,
no TensorFlow, no Candle. Just pure Rust using [`ndarray`](https://docs.rs/ndarray/) for tensor
operations. Every component from the paper is implemented from the ground up:

- **Multi-head attention** with scaled dot-product
- **Position-wise feed-forward networks**
- **Positional encoding** (sinusoidal)
- **Layer normalization** and residual connections
- **Adam optimizer** with warmup LR schedule (Equation 3)
- **Label smoothing** and **dropout**
- **Greedy and beam search** decoding
- **BLEU-4** evaluation
- **Checkpoint** save/load/averaging
- **Attention visualization** (Figures 3-5)
- **BPE tokenization** and bucket batching

## Project Structure

```
attention/                        # Project root
├── transformer/                  # Rust Transformer crate ("attenta")
│   ├── Cargo.toml
│   ├── Cargo.lock
│   ├── src/
│   │   ├── lib.rs                # Library entry point (all modules re-exported)
│   │   ├── main.rs               # Demo binary — runs the full pipeline
│   │   ├── model.rs              # Transformer model, config, positional encoding
│   │   ├── attention.rs          # Scaled dot-product + multi-head attention
│   │   ├── encoder.rs            # Encoder stack (N identical layers)
│   │   ├── decoder.rs            # Decoder stack (N identical layers)
│   │   ├── tensor_ops.rs         # Core ops: matmul, softmax, ReLU, layer norm, dropout
│   │   ├── backward.rs           # Backward pass gradients
│   │   ├── train.rs              # Training loop, Adam, StepTimer, metrics
│   │   ├── loss.rs               # Label smoothing CE, standard CE, perplexity
│   │   ├── optim.rs              # Adam optimizer + warmup LR schedule
│   │   ├── bleu.rs               # BLEU-4 score with n-gram precision + brevity penalty
│   │   ├── checkpoint.rs         # Save/load/average model weights (JSON)
│   │   ├── visualize.rs          # Attention weight extraction (Figures 3-5)
│   │   ├── batch.rs              # Bucket batching by sequence length (Section 5.1)
│   │   ├── data.rs               # Synthetic + legal dataset generators
│   │   ├── tokenizer.rs          # BPE tokenizer, save/load, train
│   │   ├── experiments.rs        # Model variation experiments (Table 3)
│   │   └── utils.rs              # Public API re-exports
│   └── examples/
│       └── train_legal.rs        # Training script for legal documents
├── data/                         # Indian legal act JSON files
│   ├── A2013-14.json             # Apprentices Act 1961
│   ├── llp act 2008.json         # Limited Liability Partnerships Act 2008
│   ├── Maternity Benefit Act 1961.json
│   ├── Payment of wages 1937.json
│   ├── Sexual Harassment of Women At Workplace Act2013.json
│   ├── The Code on Wages 2019.json
│   ├── The Industrial Employment.json
│   ├── The Insolvency And Bankruptcy Code 2016.json
│   ├── The Inter-State Migrant Workmen...Act 1979.json
│   ├── The Minimum Wages Act 1948.json
│   ├── The Payment Of Gratuity Act 1972.json
│   ├── the_industrial_disputes_act_1947.json
│   ├── wages 2017.json
│   └── The Apprentices Act1961.json
├── config.yaml                    # Configuration (Legal AI RAG system)
├── requirements.txt               # Python dependencies (for reference)
├── README.md                      # This file
├── PRD.MD                         # Product Requirements Document
├── TODO.md                        # Implementation checklist
└── UPDATE_PLAN.md                 # Update plan and roadmap
```

## Quick Start

### Prerequisites

- [Rust](https://www.rust-lang.org/) 1.70+ (tested with 1.70+)
- On some Windows systems, set `RUST_MIN_STACK=33554432` before compiling

### Run the Demo

```bash
cd transformer
RUST_MIN_STACK=33554432 cargo run --release
```

This runs the complete pipeline demonstrating every section of the paper:
- Model architecture (§3.1-3.5)
- Bucket batching (§5.1)
- Synthetic dataset generation (§5.1)
- Training on real legal data (§5.1-5.4)
- Adam optimizer with warmup (§5.3, Eq 3)
- Greedy + beam search decoding (§6.1)
- Attention visualization (Figures 3-5)
- Checkpoint save/load/average (§5.3, §6.1)
- BLEU evaluation (Table 2)
- Model variation experiments (Table 3)

### Train on Legal Documents

```bash
cd transformer
RUST_MIN_STACK=33554432 cargo run --release --example train_legal [STEPS] [DATA_DIR] [CHECKPOINT_DIR]
```

Example:

```bash
# Train for 500 steps on the built-in legal data
cargo run --release --example train_legal 500 ../data/ ./checkpoints/

# Train for 1000 steps on custom data
cargo run --release --example train_legal 1000 /path/to/my_data/ ./my_checkpoints/
```

Parameters:
- `STEPS` — number of training steps (default: 100)
- `DATA_DIR` — directory containing JSON legal act files (default: `../data/`)
- `CHECKPOINT_DIR` — directory for saving checkpoints (default: `./checkpoints/`)

### Run Tests

```bash
cd transformer
RUST_MIN_STACK=33554432 cargo test
```

All **85 unit tests** pass, covering:
- Tensor operations (matmul, softmax, ReLU, layer norm, dropout, masks)
- Attention mechanisms (scaled dot-product, multi-head)
- Forward/backward passes for all layer types
- Loss functions (label smoothing, cross-entropy, perplexity)
- Training loop correctness (loss decreases over steps)
- BLEU score calculations
- Checkpoint save/load/average roundtrip
- Tokenizer encode/decode/train
- Model variation experiments

## Architecture Overview

### Mathematical Operations (`tensor_ops.rs`)

| Operation | Implementation | Notes |
|-----------|---------------|-------|
| Matrix multiplication | Blocked 64×64 tiling | Cache-efficient, sequential |
| Softmax | Numerically stable | Subtracts row-wise max before exp |
| Layer normalization | Per-row normalization | ε=1e-6 |
| Dropout | Inverted dropout | Scales by 1/(1-p) during training |
| Embedding lookup | Row gather | O(1) per token |
| Xavier init | Glorot uniform | limit = sqrt(6/(fan_in+fan_out)) |
| Normal init | Box-Muller transform | For embedding initialization |

### Model Architecture (`model.rs`)

```
Transformer
├── src_embeddings       [src_vocab, d_model]
├── tgt_embeddings       [tgt_vocab, d_model]  (shared w/ output projection)
├── positional_encoding  [max_len, d_model]    (sinusoidal)
├── encoder              Encoder
│   ├── N × EncoderLayer
│   │   ├── MultiHeadAttention (self-attention)
│   │   ├── FeedForward (2 linear layers + ReLU)
│   │   ├── LayerNorm (×2)
│   │   └── Residual connections
│   └── LayerNorm (final)
├── decoder              Decoder
│   ├── N × DecoderLayer
│   │   ├── MultiHeadAttention (masked self-attention)
│   │   ├── MultiHeadAttention (encoder-decoder cross-attention)
│   │   ├── FeedForward (2 linear layers + ReLU)
│   │   ├── LayerNorm (×3)
│   │   └── Residual connections
│   └── LayerNorm (final)
└── output_projection    Linear(d_model → tgt_vocab)
```

### Training Pipeline (`train.rs`)

The training loop implements the full forward → loss → backward → update cycle:

1. **Forward pass**: Embed tokens + add positional encoding → encoder stack → decoder stack (with cached activations) → output projection → logits
2. **Loss**: Label smoothing cross-entropy (ε=0.1 per Section 5.4)
3. **Backward pass**: Gradient computation through all layers using cached activations
4. **Gradient clipping**: Global norm clipped to 1.0
5. **Weight tying**: Output projection synchronized with target embeddings
6. **Adam update**: β₁=0.9, β₂=0.98, ε=1e-9 with warmup LR schedule

### LR Schedule (Equation 3)

```
lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
```

### Data Pipeline

1. **Legal dataset loading**: Reads JSON files from `data/` directory
2. **BPE tokenization**: Trains a Byte Pair Encoding tokenizer on the legal text
3. **Sequence generation**: Creates (src, tgt_in, tgt_out) triples where:
   - `src` = tokenized legal section text
   - `tgt_in` = BOS + src[:-1] (shifted input for decoder)
   - `tgt_out` = src (target output for loss)
4. **Bucket batching**: Groups sequences of similar length to minimize padding

## Performance Optimizations

- **Blocked matrix multiplication**: 64×64 tiling for L1 cache efficiency
- **Batch RNG in dropout**: Single RNG instance per dropout call (was per-element)
- **Numerically stable softmax**: Max-subtraction prevents overflow
- **Release build**: `opt-level=3` with link-time optimization
- **Weight tying**: Reduces parameters and improves generalization (§3.4)

## Configuration

Default configuration follows the paper's base model (§5.3, Table 3):

```rust
TransformerConfig {
    src_vocab: 37000,      // Source vocabulary size
    tgt_vocab: 37000,      // Target vocabulary size
    d_model: 512,          // Model dimension
    n_heads: 8,            // Number of attention heads
    d_ff: 2048,            // Feed-forward hidden dimension
    n_layers: 6,           // Number of encoder/decoder layers
    max_len: 512,          // Maximum sequence length
    dropout: 0.1,          // Dropout rate
    pad_id: 0,             // Padding token ID
    bos_id: 1,             // Beginning-of-sequence token ID
    eos_id: 2,             // End-of-sequence token ID
    label_smoothing: 0.1,  // Label smoothing epsilon
    warmup_steps: 4000,    // Adam warmup steps
}
```

## Paper Coverage

| Paper Section | Implementation | File(s) |
|---------------|----------------|---------|
| §3.1 Encoder/Decoder Stacks | ✅ | `encoder.rs`, `decoder.rs` |
| §3.2.1 Scaled Dot-Product Attention | ✅ | `attention.rs:10` |
| §3.2.2 Multi-Head Attention | ✅ | `attention.rs:65` |
| §3.3 Position-wise FFN | ✅ | `attention.rs:159` |
| §3.4 Embeddings + Weight Tying | ✅ | `model.rs:63` |
| §3.5 Positional Encoding | ✅ | `model.rs:517` |
| §5.1 Bucket Batching | ✅ | `batch.rs` |
| §5.1 Data Pipeline | ✅ | `data.rs` |
| §5.1 BPE Tokenization | ✅ | `tokenizer.rs` |
| §5.2 Training | ✅ | `train.rs:696` |
| §5.3 Adam Optimizer | ✅ | `train.rs:16`, `optim.rs` |
| §5.3 LR Schedule (Eq 3) | ✅ | `optim.rs:104` |
| §5.3 Gradient Clipping | ✅ | `train.rs:897` |
| §5.4 Label Smoothing | ✅ | `loss.rs:12` |
| §6.1 Greedy Decoding | ✅ | `model.rs:199` |
| §6.1 Beam Search | ✅ | `model.rs:276` |
| §6.1 Max Output Length (input+50) | ✅ | `model.rs:217` |
| Table 2 BLEU | ✅ | `bleu.rs` |
| Table 3 Experiments | ✅ | `experiments.rs` |
| Figures 3-5 Visualization | ✅ | `visualize.rs` |
| §6.1 Checkpoint Averaging | ✅ | `checkpoint.rs:37` |

## Dependencies

- **ndarray** — N-dimensional array operations
- **rand** — Random number generation (Xavier init, dropout, sampling)
- **serde** / **serde_json** — Checkpoint serialization

No external deep learning frameworks.

## License

MIT

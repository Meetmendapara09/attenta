# Quickstart Guide

## Prerequisites

- Rust 1.70+ (https://www.rust-lang.org/)
- On some Windows systems, set `RUST_MIN_STACK=33554432` before compiling to prevent stack overflow

## Run the Demo

```bash
cd transformer
RUST_MIN_STACK=33554432 cargo run --release
```

This runs the complete pipeline demonstrating every major component of the Transformer:
- Model architecture (encoder/decoder stacks, multi-head attention, positional encoding)
- Bucket batching and synthetic dataset generation
- Training on real legal data
- Adam optimizer with warmup LR schedule
- Greedy + beam search decoding
- Attention visualization
- Checkpoint save/load/average
- BLEU evaluation
- Model variation experiments

## Train on Legal Documents

```bash
cd transformer
RUST_MIN_STACK=33554432 cargo run --release --example train_legal [STEPS] [DATA_DIR] [CHECKPOINT_DIR]
```

Examples:

```bash
# Train for 500 steps on the built-in Indian legal data
cargo run --release --example train_legal 500 ../data/ ./checkpoints/

# Train for 1000 steps on custom legal data
cargo run --release --example train_legal 1000 /path/to/my_data/ ./my_checkpoints/
```

Parameters:
- `STEPS` — number of training steps (default: 100)
- `DATA_DIR` — directory containing JSON legal act files (default: `../data/`)
- `CHECKPOINT_DIR` — directory for saving checkpoints (default: `./checkpoints/`)

## Run Tests

```bash
cd transformer
RUST_MIN_STACK=33554432 cargo test
```

All 85 unit tests pass, covering:
- Tensor operations (matmul, softmax, ReLU, layer norm, dropout, masks)
- Attention mechanisms (scaled dot-product, multi-head)
- Forward/backward passes for all layer types
- Loss functions (label smoothing, cross-entropy, perplexity)
- Training loop correctness (loss decreases over steps)
- BLEU score calculations
- Checkpoint save/load/average roundtrip
- Tokenizer encode/decode/train
- Model variation experiments

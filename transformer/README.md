# Attenta Transformer

A from-scratch Rust implementation of the Transformer architecture from
["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017).

## Quick Start

```bash
# Build and run the demo (shows all paper sections)
RUST_MIN_STACK=33554432 cargo run --release

# Run all tests
RUST_MIN_STACK=33554432 cargo test

# Train on Indian legal documents
RUST_MIN_STACK=33554432 cargo run --release --example train_legal [STEPS] [DATA_DIR] [CHECKPOINT_DIR]

# Build optimized release binary
RUST_MIN_STACK=33554432 cargo build --release
```

## Project Structure

```
src/
├── lib.rs              # Library entry — re-exports all modules
├── main.rs             # Demo binary — orchestrates the full pipeline
├── model.rs            # Transformer model, TransformerConfig, positional encoding (§3.4-3.5)
├── attention.rs        # Scaled dot-product attention (§3.2.1), multi-head attention (§3.2.2),
│                       #  feed-forward network (§3.3), layer normalization (§3.1), linear layer
├── encoder.rs          # Encoder layer + stack with residual connections (§3.1)
├── decoder.rs          # Decoder layer + stack with masked self-attention + cross-attention (§3.1)
├── tensor_ops.rs       # Core tensor operations: matmul, softmax, ReLU, layer norm, dropout,
│                       #  embedding lookup, masks, Xavier/normal/uniform init
├── backward.rs         # Backward pass: matmul, softmax, ReLU, attention, layer norm gradients
├── train.rs            # Adam optimizer state (§5.3), training loop, StepTimer, TrainingMetrics
├── loss.rs             # Label smoothing CE (§5.4), cross-entropy, perplexity, log-softmax
├── optim.rs            # Adam optimizer + warmup LR schedule (§5.3, Eq 3)
├── bleu.rs             # BLEU-4 with n-gram precision + brevity penalty (Table 2)
├── checkpoint.rs       # Save/load/average checkpoints in JSON (§5.3, §6.1)
├── visualize.rs        # Attention weight extraction for Figures 3-5
├── batch.rs            # Bucket batching by sequence length (§5.1)
├── data.rs             # Synthetic dataset + legal JSON dataset loader
├── tokenizer.rs        # BPE tokenizer: train, encode, decode, save/load
├── experiments.rs      # Model variation experiments (Table 3)
└── utils.rs            # Public API re-exports
```

## Key Implementation Details

### Blocked Matrix Multiplication (`tensor_ops.rs:10`)
Uses 64×64 block tiling along the reduction dimension to keep working sets in L1 cache.

### Numerically Stable Softmax (`tensor_ops.rs:52`)
Subtracts row-wise maximum before exponentiation to prevent overflow.

### Weight Tying (`model.rs:115`)
The output projection matrix is the transpose of the target embedding matrix (§3.4),
reducing parameters and improving translation quality.

### Adam with Warmup (`optim.rs:47`)
Implements Equation 3 from the paper:
```
lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
```

### Gradient Clipping (`train.rs:950`)
Global gradient norm is clipped to 1.0 to prevent gradient explosion during training.

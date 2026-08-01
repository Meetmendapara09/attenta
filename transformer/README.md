# Attenta Transformer

Rust crate for the Transformer model from "Attention Is All You Need" (Vaswani et al., 2017).

## Quick Start

```bash
# Run the demo
RUST_MIN_STACK=33554432 cargo run --release

# Train on legal data
RUST_MIN_STACK=33554432 cargo run --release --example train_legal 500 ../data/ ./checkpoints/

# Run tests
RUST_MIN_STACK=33554432 cargo test
```

See the [root README](https://github.com/Meetmendapara09/attenta) for project-level documentation and
[docs/](https://github.com/Meetmendapara09/attenta/tree/main/docs) for modular documentation.

## Module Map

| Module | Paper Section | Purpose |
|--------|--------------|---------|
| `model.rs` | Section3.1-3.5 | Transformer model, config, positional encoding |
| `attention.rs` | Section3.2-3.3 | Scaled dot-product, multi-head, FFN, layer norm |
| `encoder.rs` | Section3.1 | Encoder layer + stack |
| `decoder.rs` | Section3.1 | Decoder layer + stack |
| `tensor_ops.rs` | - | matmul, softmax, dropout, init |
| `backward.rs` | Section5.2 | Gradient computation |
| `train.rs` | Section5.2-5.3 | Training loop, Adam, StepTimer, metrics |
| `loss.rs` | Section5.4 | Label smoothing, cross-entropy, perplexity |
| `optim.rs` | Section5.3 | Adam + warmup LR schedule (Eq 3) |
| `bleu.rs` | Section6.2 | BLEU-4 with n-gram precision + brevity penalty |
| `checkpoint.rs` | Section6.1 | Save/load/average checkpoints |
| `visualize.rs` | Fig 3-5 | Attention weight extraction |
| `batch.rs` | Section5.1 | Bucket batching by sequence length |
| `data.rs` | Section5.1 | Legal dataset loading from JSON |
| `tokenizer.rs` | Section5.1 | BPE tokenizer |
| `experiments.rs` | Table 3 | Model variation experiments |
| `utils.rs` | - | Public API re-exports |

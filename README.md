# Attenta

[![Rust](https://img.shields.io/badge/Rust-1.70%2B-orange)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Vaswani%202017-blue)](https://arxiv.org/abs/1706.03762)

**Attenta** — a from-scratch Rust implementation of the Transformer architecture from
["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017).

Trained on Indian legal documents. No PyTorch, no TensorFlow — pure Rust.

## Quick Start

```bash
cd transformer
RUST_MIN_STACK=33554432 cargo run --release
```

See [Quickstart Guide](docs/quickstart.md) for detailed instructions.

## Project Structure

```
attention/                        # Project root
├── transformer/                  # Rust Transformer crate ("attenta")
│   ├── Cargo.toml
│   ├── Cargo.lock
│   ├── src/                      # All source modules
│   ├── examples/
│   │   └── train_legal.rs        # Training script for legal documents
│   └── README.md                 # Crate-specific docs
├── data/                         # Indian legal act JSON files
├── docs/                         # Documentation (modular)
│   ├── quickstart.md             # Installation, demo, testing
│   ├── architecture.md           # Model architecture + operations
│   ├── training.md               # Training guide + checkpoints
│   └── paper_coverage.md         # Paper section → implementation mapping
├── config.yaml                   # Configuration
├── requirements.txt             # Python dependencies (for reference)
├── README.md                     # This file
└── LICENSE                       # MIT
```

## Documentation

| Topic | Document |
|-------|----------|
| Installation & usage | [docs/quickstart.md](docs/quickstart.md) |
| Architecture overview | [docs/architecture.md](docs/architecture.md) |
| Training guide | [docs/training.md](docs/training.md) |
| Paper coverage | [docs/paper_coverage.md](docs/paper_coverage.md) |

## Key Features

- Multi-head attention with scaled dot-product
- Positional encoding (sinusoidal)
- Layer normalization + residual connections
- Adam optimizer with warmup LR schedule (Eq 3)
- Label smoothing cross-entropy (§5.4)
- Greedy + beam search decoding (§6.1)
- BLEU-4 evaluation
- Checkpoint save/load/averaging (§6.1)
- Attention visualization (Figures 3-5)
- BPE tokenization + bucket batching
- 85 unit tests covering all components
- Blocked 64×64 matmul for cache efficiency
- Dropout optimized with batch RNG

## License

MIT

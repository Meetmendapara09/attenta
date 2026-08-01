# Training Guide

## Configuration

Default configuration follows the paper's base model (Section 5.3, Table 3):

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

## Training on Legal Documents

```bash
cd transformer
RUST_MIN_STACK=33554432 cargo run --release --example train_legal [STEPS] [DATA_DIR] [CHECKPOINT_DIR]
```

Example:

```bash
cargo run --release --example train_legal 500 ../data/ ./checkpoints/
```

## Checkpoints

The training pipeline saves checkpoints in JSON format:
- Periodic checkpoint every 50 steps: `checkpoint_step_{N}.json`
- Final checkpoint: `checkpoint_final.json`

Checkpoint files contain:
- Model weights (all parameter matrices)
- Optimizer state (Adam moments)
- Configuration parameters
- Training step number

## Training on Custom Data

1. Create a directory with JSON files matching the legal act format:
```json
{
    "sections": [
        {
            "act": "Act Name",
            "section_number": "1",
            "section_title": "Title",
            "text": "Section text content...",
            "keywords": ["keyword1", "keyword2"],
            "document_types_applicable": ["type1"]
        }
    ]
}
```

2. Run training with your data directory:
```bash
cargo run --release --example train_legal 1000 /path/to/my_data/ ./checkpoints/
```

## Performance Notes

- Use `--release` for optimized builds (10-50x speedup over debug)
- The default build uses `opt-level=3` with LTO
- Matmul uses 64×64 blocked tiling for cache efficiency
- Dropout uses batched RNG for performance

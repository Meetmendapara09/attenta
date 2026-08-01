# Paper Coverage

Full mapping of "Attention Is All You Need" (Vaswani et al., 2017) sections to implementation.

## Model Architecture

| Paper Section | Implementation | File(s) |
|---------------|----------------|---------|
| §3.1 Encoder/Decoder Stacks | ✅ | `encoder.rs`, `decoder.rs` |
| §3.2.1 Scaled Dot-Product Attention | ✅ | `attention.rs:10` |
| §3.2.2 Multi-Head Attention | ✅ | `attention.rs:65` |
| §3.3 Position-wise FFN | ✅ | `attention.rs:159` |
| §3.4 Embeddings + Weight Tying | ✅ | `model.rs:63` |
| §3.5 Positional Encoding | ✅ | `model.rs:517` |

## Training

| Paper Section | Implementation | File(s) |
|---------------|----------------|---------|
| §5.1 Bucket Batching | ✅ | `batch.rs` |
| §5.1 Data Pipeline | ✅ | `data.rs` |
| §5.1 BPE Tokenization | ✅ | `tokenizer.rs` |
| §5.2 Training | ✅ | `train.rs:696` |
| §5.3 Adam Optimizer | ✅ | `train.rs:16`, `optim.rs` |
| §5.3 LR Schedule (Eq 3) | ✅ | `optim.rs:104` |
| §5.3 Gradient Clipping | ✅ | `train.rs:897` |
| §5.4 Label Smoothing | ✅ | `loss.rs:12` |

## Inference & Evaluation

| Paper Section | Implementation | File(s) |
|---------------|----------------|---------|
| §6.1 Greedy Decoding | ✅ | `model.rs:199` |
| §6.1 Beam Search | ✅ | `model.rs:276` |
| §6.1 Max Output Length (input+50) | ✅ | `model.rs:217` |
| Table 2 BLEU | ✅ | `bleu.rs` |
| Table 3 Experiments | ✅ | `experiments.rs` |
| Figures 3-5 Visualization | ✅ | `visualize.rs` |
| §6.1 Checkpoint Averaging | ✅ | `checkpoint.rs:37` |

# Paper Coverage

Full mapping of "Attention Is All You Need" (Vaswani et al., 2017) sections to implementation.

## Model Architecture

| Paper Section | Implementation | File(s) |
|---------------|----------------|---------|
| Section3.1 Encoder/Decoder Stacks | ✅ | `encoder.rs`, `decoder.rs` |
| Section3.2.1 Scaled Dot-Product Attention | ✅ | `attention.rs:10` |
| Section3.2.2 Multi-Head Attention | ✅ | `attention.rs:65` |
| Section3.3 Position-wise FFN | ✅ | `attention.rs:159` |
| Section3.4 Embeddings + Weight Tying | ✅ | `model.rs:63` |
| Section3.5 Positional Encoding | ✅ | `model.rs:517` |

## Training

| Paper Section | Implementation | File(s) |
|---------------|----------------|---------|
| Section5.1 Bucket Batching | ✅ | `batch.rs` |
| Section5.1 Data Pipeline | ✅ | `data.rs` |
| Section5.1 BPE Tokenization | ✅ | `tokenizer.rs` |
| Section5.2 Training | ✅ | `train.rs:696` |
| Section5.3 Adam Optimizer | ✅ | `train.rs:16`, `optim.rs` |
| Section5.3 LR Schedule (Eq 3) | ✅ | `optim.rs:104` |
| Section5.3 Gradient Clipping | ✅ | `train.rs:897` |
| Section5.4 Label Smoothing | ✅ | `loss.rs:12` |

## Inference & Evaluation

| Paper Section | Implementation | File(s) |
|---------------|----------------|---------|
| Section6.1 Greedy Decoding | ✅ | `model.rs:199` |
| Section6.1 Beam Search | ✅ | `model.rs:276` |
| Section6.1 Max Output Length (input+50) | ✅ | `model.rs:217` |
| Table 2 BLEU | ✅ | `bleu.rs` |
| Table 3 Experiments | ✅ | `experiments.rs` |
| Figures 3-5 Visualization | ✅ | `visualize.rs` |
| Section6.1 Checkpoint Averaging | ✅ | `checkpoint.rs:37` |

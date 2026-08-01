# Update Plan — Attention Is All You Need Implementation

## Status as of this session

- All 69 tests pass except 1: `bleu::tests::test_corpus_bleu` (short sentences cause total_ngrams=0 → 0.0 precision → ln(0) = -inf → score=0)
- All core architecture, forward/backward, training loop, checkpointing, BLEU, inference, and visualization are implemented
- ~85% complete per the TODO.md

## Remaining Work

### Step 1: Fix failing test in `bleu.rs`
- `corpus_bleu()` returns 0.0 when `total_ngrams[n] == 0` → fix smoothing to handle this case

### Step 2: Phase 5a — Create `batch.rs` (Sequence length batching)
- `BucketBatcher` struct for grouping sequences by approximate length
- Bucket width parameter (paper batches sequences of similar length together)
- Shuffle within buckets

### Step 3: Phase 5b — Create `data.rs` (Synthetic WMT-like dataset)
- Variable-length sequence generation
- Train/validation split
- Configurable min/max sequence length, vocab size
- Source-target pair generation

### Step 4: Phase 5c — Extend `train.rs`
- Add `StepTimer` for throughput measurement (tokens/sec)
- Add checkpoint saving every N steps during training
- Integrate BLEU evaluation during training
- Keep existing backward/optimization intact; add new modular helpers

### Step 5: Phase 5d — Update `main.rs`
- Wire up BucketBatcher, synthetic dataset, extended training loop
- Show throughput, checkpointing, BLEU eval

### Step 6: Phase 7 — Create `experiments.rs` (Table 3 experiments)
- Experiment runner varying: N (2,4,6), d_model (128,256,512), d_ff (512,1024,2048), h (2,4,8), dropout (0.0,0.1,0.2,0.3), label_smoothing (0.0,0.1,0.2)
- Fixed training steps per config
- Output results table matching Table 3

### Step 7: Phase 8 — Compile, test, verify
- `cargo build --release`
- `cargo test` (all 70+ tests passing)
- `cargo run --release` (full demo pipeline)


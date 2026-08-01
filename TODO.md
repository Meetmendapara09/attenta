# Implementation Plan: Complete "Attention Is All You Need" Paper

## Phase 1: Foundation
- [x] Codebase analysis complete (existing implementation: ~85%)

## Phase 2: Checkpointing (5.3, 6.1)
- [x] Add checkpoint save/load (JSON format) for model weights
- [x] Add checkpoint averaging (last 5 for base model, last 20 for big model)
- [x] Add optimizer state serialization

## Phase 3: BLEU Score Evaluation (Table 2)
- [x] Implement n-gram precision calculation
- [x] Implement brevity penalty
- [x] Add BLEU-4 scoring function
- [x] Add corpus-level BLEU evaluation
- [x] Fixed bug: corpus_bleu was returning 0 due to ln(0) when short sequences had no n-grams at higher orders

## Phase 4: Proper Inference (6.1)
- [x] Implement max output length = input_length + 50
- [x] Add early termination in greedy decode and beam search
- [x] Update `translate()`/`translate_beam()` APIs

## Phase 5: Training Improvements (5.1-5.2)
- [x] Add `batch.rs` — `BucketBatcher` for length-based sequence batching (Section 5.1)
- [x] Add `data.rs` — `SyntheticDataset` generator with variable-length sequences and train/val split
- [x] Extended `train.rs` — `StepTimer`, `TrainingMetrics`, `extended_train()` loop with checkpointing
- [x] Weight tying: fix output projection sync on every step during training

## Phase 6: Attention Visualization (Figures 3-5)
- [x] Add attention weight extraction from encoder/decoder
- [x] Add visualization output (JSON format for rendering)

## Phase 7: Model Variation Experiments (Table 3)
- [x] Add `experiments.rs` — experiment runner for varying N, d_model, d_ff, h, dropout, label_smoothing
- [x] Table 3 results formatted table + JSON output generator

## Phase 8: Testing & Verification
- [x] Fixed corpus_bleu test failure (ln(0) precision edge case)
- [x] All 85 unit tests pass (68 original + 17 new: batch(3) + data(4) + tokenizer(7) + experiments(1) + corpus_bleu(1) + train(1))
- [x] Clean warnings - known dead code left intentionally (utility functions, print_stats, etc.)
- [x] Release build compiles successfully (cargo build --release)
- [x] Full demo pipeline runs correctly (cargo run --release)
- [x] Backpropagation training loop verified: loss decreases over 100 steps
- [x] Greedy/beam search decoding verified
- [x] Tokenizer encode/decode/save/load roundtrip verified

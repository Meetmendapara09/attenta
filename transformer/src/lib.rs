//! Attenta: A from-scratch Rust implementation of "Attention Is All You Need".
//!
//! This crate provides a complete implementation of the Transformer architecture
//! (Vaswani et al., 2017) in pure Rust, including:
//! - Scaled dot-product and multi-head attention
//! - Encoder-decoder stacks with residual connections and layer normalization
//! - Position-wise feed-forward networks
//! - Positional encoding (sinusoidal)
//! - Adam optimizer with warmup LR schedule
//! - Label smoothing, dropout, and gradient clipping
//! - Greedy and beam search decoding
//! - BLEU-4 evaluation
//! - Checkpoint save/load/averaging
//! - Attention visualization
//! - Bucket batching and BPE tokenization
//! - Full backpropagation training loop

pub mod attention;
pub mod backward;
pub mod batch;
pub mod bleu;
pub mod checkpoint;
pub mod data;
pub mod decoder;
pub mod encoder;
pub mod experiments;
pub mod loss;
pub mod model;
pub mod optim;
pub mod tensor_ops;
pub mod tokenizer;
pub mod train;
pub mod utils;
pub mod visualize;

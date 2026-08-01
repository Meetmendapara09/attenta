// Public API re-exports for library consumers.
// These are kept for potential library usage but suppressed warnings
// since they may be used by external consumers.
#![allow(unused_imports)]

pub use crate::attention::{
    scaled_dot_product_attention, FeedForward, LayerNorm, Linear, MultiHeadAttention,
};

pub use crate::decoder::{Decoder, DecoderLayer};

pub use crate::encoder::{Encoder, EncoderLayer};

pub use crate::loss::{cross_entropy_loss, label_smoothing_loss, perplexity};

pub use crate::model::{generate_positional_encoding, Transformer, TransformerConfig};

pub use crate::optim::{lr_schedule, Adam};

pub use crate::tensor_ops::{
    causal_mask, embedding_lookup, layer_norm, make_decoder_mask, make_src_mask, matmul,
    normal_init, padding_mask, relu, scaled_init, softmax, transpose, xavier_init,
};

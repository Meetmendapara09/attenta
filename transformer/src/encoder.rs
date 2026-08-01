use ndarray::Array2;

use crate::attention::{FeedForward, LayerNorm, MultiHeadAttention};
use crate::tensor_ops::dropout;

/// Encoder Layer.
///
/// Each encoder layer has two sub-layers:
///   1. Multi-head self-attention
///   2. Position-wise feed-forward network
///
/// Each sub-layer has a residual connection followed by layer normalization:
///   LayerNorm(x + Sublayer(x))
///
/// Paper: Section 3.1
pub struct EncoderLayer {
    pub(crate) self_attn: MultiHeadAttention,
    pub(crate) feed_forward: FeedForward,
    pub(crate) norm_1: LayerNorm,
    pub(crate) norm_2: LayerNorm,
    pub(crate) dropout_rate: f64,
}

impl EncoderLayer {
    pub fn new(
        self_attn: MultiHeadAttention,
        feed_forward: FeedForward,
        norm_1: LayerNorm,
        norm_2: LayerNorm,
        dropout_rate: f64,
    ) -> Self {
        Self {
            self_attn,
            feed_forward,
            norm_1,
            norm_2,
            dropout_rate,
        }
    }

    /// x: [seq_len, d_model]
    /// mask: optional additive mask for padding
    pub fn forward(&self, x: &Array2<f64>, mask: Option<&Array2<f64>>, train: bool) -> Array2<f64> {
        // Sub-layer 1: Self-attention with residual + LayerNorm
        let sublayer1 = self.self_attn.forward(x, x, x, mask, train);
        let sublayer1 = dropout(&sublayer1, self.dropout_rate, train);
        let x = self.norm_1.forward(&(x + &sublayer1));

        // Sub-layer 2: Feed-forward with residual + LayerNorm
        let sublayer2 = self.feed_forward.forward(&x, train);
        let sublayer2 = dropout(&sublayer2, self.dropout_rate, train);
        self.norm_2.forward(&(&x + &sublayer2))
    }

    pub fn num_params(&self) -> usize {
        self.self_attn.num_params()
            + self.feed_forward.num_params()
            + self.norm_1.num_params()
            + self.norm_2.num_params()
    }
}

/// Encoder: stack of N identical layers.
///
/// Paper: Section 3.1
pub struct Encoder {
    pub(crate) layers: Vec<EncoderLayer>,
    pub(crate) norm: LayerNorm,
}

impl Encoder {
    pub fn new(layers: Vec<EncoderLayer>, norm: LayerNorm) -> Self {
        Self { layers, norm }
    }

    /// src: [src_len, d_model] (after embedding + positional encoding)
    pub fn forward(
        &self,
        src: &Array2<f64>,
        mask: Option<&Array2<f64>>,
        train: bool,
    ) -> Array2<f64> {
        let mut x = src.clone();
        for layer in &self.layers {
            x = layer.forward(&x, mask, train);
        }
        self.norm.forward(&x)
    }

    pub fn num_params(&self) -> usize {
        self.layers.iter().map(|l| l.num_params()).sum::<usize>() + self.norm.num_params()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn make_test_encoder_layer(d_model: usize, n_heads: usize, dropout: f64) -> EncoderLayer {
        let mut rng = rand::thread_rng();
        use crate::attention::{FeedForward, LayerNorm, Linear, MultiHeadAttention};
        use crate::tensor_ops::xavier_init;

        let w_q = Linear::new(
            xavier_init(d_model, d_model, &mut rng),
            Array1::zeros(d_model),
        );
        let w_k = Linear::new(
            xavier_init(d_model, d_model, &mut rng),
            Array1::zeros(d_model),
        );
        let w_v = Linear::new(
            xavier_init(d_model, d_model, &mut rng),
            Array1::zeros(d_model),
        );
        let w_o = Linear::new(
            xavier_init(d_model, d_model, &mut rng),
            Array1::zeros(d_model),
        );
        let self_attn = MultiHeadAttention::new(d_model, n_heads, w_q, w_k, w_v, w_o, dropout);

        let w1 = Linear::new(
            xavier_init(d_model, d_model * 2, &mut rng),
            Array1::zeros(d_model * 2),
        );
        let w2 = Linear::new(
            xavier_init(d_model * 2, d_model, &mut rng),
            Array1::zeros(d_model),
        );
        let feed_forward = FeedForward::new(w1, w2, dropout);

        EncoderLayer::new(
            self_attn,
            feed_forward,
            LayerNorm::new(Array1::ones(d_model), Array1::zeros(d_model)),
            LayerNorm::new(Array1::ones(d_model), Array1::zeros(d_model)),
            dropout,
        )
    }

    #[test]
    fn test_encoder_layer_output_shape() {
        let d_model = 16;
        let n_heads = 4;
        let layer = make_test_encoder_layer(d_model, n_heads, 0.0);
        let x = ndarray::Array2::from_shape_fn((5, d_model), |_| rand::random::<f64>());
        let out = layer.forward(&x, None, false);
        assert_eq!(out.shape(), &[5, d_model]);
    }

    #[test]
    fn test_encoder_layer_with_mask() {
        let d_model = 16;
        let n_heads = 4;
        let layer = make_test_encoder_layer(d_model, n_heads, 0.0);
        let x = ndarray::Array2::from_shape_fn((4, d_model), |_| rand::random::<f64>());
        // Mask one position
        let mut mask = ndarray::Array2::zeros((1, 4));
        mask[[0, 3]] = f64::NEG_INFINITY;
        let out = layer.forward(&x, Some(&mask), false);
        assert_eq!(out.shape(), &[4, d_model]);
    }

    #[test]
    fn test_encoder_stack_output_shape() {
        let d_model = 16;
        let n_heads = 4;
        let layer1 = make_test_encoder_layer(d_model, n_heads, 0.0);
        let layer2 = make_test_encoder_layer(d_model, n_heads, 0.0);
        let layers = vec![layer1, layer2];
        let norm = crate::attention::LayerNorm::new(Array1::ones(d_model), Array1::zeros(d_model));
        let encoder = Encoder::new(layers, norm);
        let x = ndarray::Array2::from_shape_fn((6, d_model), |_| rand::random::<f64>());
        let out = encoder.forward(&x, None, false);
        assert_eq!(out.shape(), &[6, d_model]);
    }

    #[test]
    fn test_encoder_num_params() {
        let d_model = 16;
        let n_heads = 4;
        let layer1 = make_test_encoder_layer(d_model, n_heads, 0.0);
        let layer2 = make_test_encoder_layer(d_model, n_heads, 0.0);
        let layers = vec![layer1, layer2];
        let norm = crate::attention::LayerNorm::new(Array1::ones(d_model), Array1::zeros(d_model));
        let encoder = Encoder::new(layers, norm);
        assert!(encoder.num_params() > 0);
    }
}

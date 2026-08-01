use ndarray::Array2;

use crate::attention::{FeedForward, LayerNorm, MultiHeadAttention};
use crate::tensor_ops::dropout;

/// Decoder Layer.
///
/// Each decoder layer has three sub-layers:
///   1. Masked multi-head self-attention (prevents attending to subsequent positions)
///   2. Multi-head attention over encoder output (encoder-decoder attention)
///   3. Position-wise feed-forward network
///
/// Each sub-layer has a residual connection followed by layer normalization.
///
/// Paper: Section 3.1
pub struct DecoderLayer {
    pub(crate) self_attn: MultiHeadAttention,
    pub(crate) cross_attn: MultiHeadAttention,
    pub(crate) feed_forward: FeedForward,
    pub(crate) norm_1: LayerNorm,
    pub(crate) norm_2: LayerNorm,
    pub(crate) norm_3: LayerNorm,
    pub(crate) dropout_rate: f64,
}

impl DecoderLayer {
    pub fn new(
        self_attn: MultiHeadAttention,
        cross_attn: MultiHeadAttention,
        feed_forward: FeedForward,
        norm_1: LayerNorm,
        norm_2: LayerNorm,
        norm_3: LayerNorm,
        dropout_rate: f64,
    ) -> Self {
        Self {
            self_attn,
            cross_attn,
            feed_forward,
            norm_1,
            norm_2,
            norm_3,
            dropout_rate,
        }
    }

    /// x:          [tgt_len, d_model]
    /// enc_output: [src_len, d_model]
    /// src_mask:   additive mask [1, src_len]
    /// tgt_mask:   additive mask [tgt_len, tgt_len]
    pub fn forward(
        &self,
        x: &Array2<f64>,
        enc_output: &Array2<f64>,
        src_mask: Option<&Array2<f64>>,
        tgt_mask: Option<&Array2<f64>>,
        train: bool,
    ) -> Array2<f64> {
        // Sub-layer 1: Masked self-attention + residual + LayerNorm
        let sublayer1 = self.self_attn.forward(x, x, x, tgt_mask, train);
        let sublayer1 = dropout(&sublayer1, self.dropout_rate, train);
        let x = self.norm_1.forward(&(x + &sublayer1));

        // Sub-layer 2: Encoder-decoder attention + residual + LayerNorm
        let sublayer2 = self
            .cross_attn
            .forward(&x, enc_output, enc_output, src_mask, train);
        let sublayer2 = dropout(&sublayer2, self.dropout_rate, train);
        let x = self.norm_2.forward(&(&x + &sublayer2));

        // Sub-layer 3: Feed-forward + residual + LayerNorm
        let sublayer3 = self.feed_forward.forward(&x, train);
        let sublayer3 = dropout(&sublayer3, self.dropout_rate, train);
        self.norm_3.forward(&(&x + &sublayer3))
    }

    pub fn num_params(&self) -> usize {
        self.self_attn.num_params()
            + self.cross_attn.num_params()
            + self.feed_forward.num_params()
            + self.norm_1.num_params()
            + self.norm_2.num_params()
            + self.norm_3.num_params()
    }
}

/// Decoder: stack of N identical layers.
///
/// Paper: Section 3.1
pub struct Decoder {
    pub(crate) layers: Vec<DecoderLayer>,
    pub(crate) norm: LayerNorm,
}

impl Decoder {
    pub fn new(layers: Vec<DecoderLayer>, norm: LayerNorm) -> Self {
        Self { layers, norm }
    }

    /// tgt:         [tgt_len, d_model]
    /// enc_output:  [src_len, d_model]
    /// src_mask:    additive mask [1, src_len]
    /// tgt_mask:    additive mask [tgt_len, tgt_len]
    pub fn forward(
        &self,
        tgt: &Array2<f64>,
        enc_output: &Array2<f64>,
        src_mask: Option<&Array2<f64>>,
        tgt_mask: Option<&Array2<f64>>,
        train: bool,
    ) -> Array2<f64> {
        let mut x = tgt.clone();
        for layer in &self.layers {
            x = layer.forward(&x, enc_output, src_mask, tgt_mask, train);
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

    fn make_test_decoder_layer(d_model: usize, n_heads: usize, dropout: f64) -> DecoderLayer {
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
        let cross_attn = MultiHeadAttention::new(d_model, n_heads, w_q, w_k, w_v, w_o, dropout);

        let w1 = Linear::new(
            xavier_init(d_model, d_model * 2, &mut rng),
            Array1::zeros(d_model * 2),
        );
        let w2 = Linear::new(
            xavier_init(d_model * 2, d_model, &mut rng),
            Array1::zeros(d_model),
        );
        let feed_forward = FeedForward::new(w1, w2, dropout);

        DecoderLayer::new(
            self_attn,
            cross_attn,
            feed_forward,
            LayerNorm::new(Array1::ones(d_model), Array1::zeros(d_model)),
            LayerNorm::new(Array1::ones(d_model), Array1::zeros(d_model)),
            LayerNorm::new(Array1::ones(d_model), Array1::zeros(d_model)),
            dropout,
        )
    }

    #[test]
    fn test_decoder_layer_output_shape() {
        let d_model = 16;
        let n_heads = 4;
        let layer = make_test_decoder_layer(d_model, n_heads, 0.0);
        let tgt = ndarray::Array2::from_shape_fn((4, d_model), |_| rand::random::<f64>());
        let enc = ndarray::Array2::from_shape_fn((6, d_model), |_| rand::random::<f64>());
        let out = layer.forward(&tgt, &enc, None, None, false);
        assert_eq!(out.shape(), &[4, d_model]);
    }

    #[test]
    fn test_decoder_layer_with_masks() {
        let d_model = 16;
        let n_heads = 4;
        let layer = make_test_decoder_layer(d_model, n_heads, 0.0);
        let tgt = ndarray::Array2::from_shape_fn((3, d_model), |_| rand::random::<f64>());
        let enc = ndarray::Array2::from_shape_fn((5, d_model), |_| rand::random::<f64>());
        let src_mask = ndarray::Array2::zeros((1, 5));
        let tgt_mask =
            crate::tensor_ops::causal_mask(3)
                .mapv(|x| if x == 0.0 { f64::NEG_INFINITY } else { 0.0 });
        let out = layer.forward(&tgt, &enc, Some(&src_mask), Some(&tgt_mask), false);
        assert_eq!(out.shape(), &[3, d_model]);
    }

    #[test]
    fn test_decoder_stack_output_shape() {
        let d_model = 16;
        let n_heads = 4;
        let layer1 = make_test_decoder_layer(d_model, n_heads, 0.0);
        let layer2 = make_test_decoder_layer(d_model, n_heads, 0.0);
        let layers = vec![layer1, layer2];
        let norm = crate::attention::LayerNorm::new(Array1::ones(d_model), Array1::zeros(d_model));
        let decoder = Decoder::new(layers, norm);
        let tgt = ndarray::Array2::from_shape_fn((4, d_model), |_| rand::random::<f64>());
        let enc = ndarray::Array2::from_shape_fn((6, d_model), |_| rand::random::<f64>());
        let out = decoder.forward(&tgt, &enc, None, None, false);
        assert_eq!(out.shape(), &[4, d_model]);
    }

    #[test]
    fn test_decoder_num_params() {
        let d_model = 16;
        let n_heads = 4;
        let layer1 = make_test_decoder_layer(d_model, n_heads, 0.0);
        let layer2 = make_test_decoder_layer(d_model, n_heads, 0.0);
        let layers = vec![layer1, layer2];
        let norm = crate::attention::LayerNorm::new(Array1::ones(d_model), Array1::zeros(d_model));
        let decoder = Decoder::new(layers, norm);
        assert!(decoder.num_params() > 0);
    }
}

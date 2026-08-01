use ndarray::{Array1, Array2};

use crate::tensor_ops::{dropout, matmul, softmax, transpose};

/// Scaled Dot-Product Attention.
///
/// Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
///
/// Paper: Section 3.2.1, Equation (1)
pub fn scaled_dot_product_attention(
    q: &Array2<f64>,
    k: &Array2<f64>,
    v: &Array2<f64>,
    mask: Option<&Array2<f64>>,
) -> (Array2<f64>, Array2<f64>) {
    let d_k = q.ncols() as f64;
    let scores = matmul(q, &transpose(k)) / d_k.sqrt();

    let scores = if let Some(m) = mask {
        &scores + m
    } else {
        scores
    };

    let weights = softmax(&scores);
    let output = matmul(&weights, v);
    (output, weights)
}

/// Linear projection layer: y = x @ w + b.
#[derive(Clone)]
pub struct Linear {
    pub w: Array2<f64>,
    pub b: Array1<f64>,
}

impl Linear {
    pub fn new(w: Array2<f64>, b: Array1<f64>) -> Self {
        Self { w, b }
    }

    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut result = matmul(x, &self.w);
        for mut row in result.rows_mut() {
            row += &self.b;
        }
        result
    }

    /// Number of parameters (weights + biases).
    pub fn num_params(&self) -> usize {
        self.w.len() + self.b.len()
    }
}

/// Multi-Head Attention.
///
/// MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
/// where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
///
/// Paper: Section 3.2.2
///
/// Uses single [d_model, d_model] projection matrices per Q/K/V,
/// sliced into h heads of dimension d_k = d_model / h.
pub struct MultiHeadAttention {
    pub(crate) d_model: usize,
    pub(crate) n_heads: usize,
    pub(crate) d_k: usize,
    pub(crate) w_q: Linear,
    pub(crate) w_k: Linear,
    pub(crate) w_v: Linear,
    pub(crate) w_o: Linear,
    pub(crate) dropout_rate: f64,
}

impl MultiHeadAttention {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        w_q: Linear,
        w_k: Linear,
        w_v: Linear,
        w_o: Linear,
        dropout_rate: f64,
    ) -> Self {
        assert_eq!(d_model % n_heads, 0, "d_model must be divisible by n_heads");
        let d_k = d_model / n_heads;
        Self {
            d_model,
            n_heads,
            d_k,
            w_q,
            w_k,
            w_v,
            w_o,
            dropout_rate,
        }
    }

    /// Forward pass.
    ///
    /// q, k, v: [seq_len, d_model]
    /// mask: optional additive mask [target_len, source_len] (0=allow, -inf=block)
    pub fn forward(
        &self,
        q: &Array2<f64>,
        k: &Array2<f64>,
        v: &Array2<f64>,
        mask: Option<&Array2<f64>>,
        train: bool,
    ) -> Array2<f64> {
        let seq_len = q.nrows();

        // Single linear projections: [seq_len, d_model]
        let q_proj = self.w_q.forward(q);
        let k_proj = self.w_k.forward(k);
        let v_proj = self.w_v.forward(v);

        // Attention for each head, then concatenate
        let mut concat = ndarray::Array2::zeros((seq_len, self.d_model));
        for h in 0..self.n_heads {
            let start = h * self.d_k;
            let end = start + self.d_k;

            // Extract head slices: [seq_len, d_k]
            let q_h = q_proj.slice(ndarray::s![.., start..end]).to_owned();
            let k_h = k_proj.slice(ndarray::s![.., start..end]).to_owned();
            let v_h = v_proj.slice(ndarray::s![.., start..end]).to_owned();

            let (head_out, _) = scaled_dot_product_attention(&q_h, &k_h, &v_h, mask);
            let head_out = dropout(&head_out, self.dropout_rate, train);

            // Write back into concat buffer
            for i in 0..seq_len {
                for j in 0..self.d_k {
                    concat[[i, start + j]] = head_out[[i, j]];
                }
            }
        }

        // Final linear projection
        self.w_o.forward(&concat)
    }

    /// Number of parameters.
    pub fn num_params(&self) -> usize {
        self.w_q.num_params()
            + self.w_k.num_params()
            + self.w_v.num_params()
            + self.w_o.num_params()
    }
}

/// Position-wise Feed-Forward Network.
///
/// FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
///
/// Paper: Section 3.3, Equation (2)
pub struct FeedForward {
    pub(crate) w_1: Linear,
    pub(crate) w_2: Linear,
    pub(crate) dropout_rate: f64,
}

impl FeedForward {
    pub fn new(w_1: Linear, w_2: Linear, dropout_rate: f64) -> Self {
        Self {
            w_1,
            w_2,
            dropout_rate,
        }
    }

    pub fn forward(&self, x: &Array2<f64>, train: bool) -> Array2<f64> {
        let x = self.w_1.forward(x);
        let x = crate::tensor_ops::relu(&x);
        let x = dropout(&x, self.dropout_rate, train);
        self.w_2.forward(&x)
    }

    pub fn num_params(&self) -> usize {
        self.w_1.num_params() + self.w_2.num_params()
    }
}

/// Layer Normalization.
///
/// Paper: Section 3.1
pub struct LayerNorm {
    pub(crate) w: Array1<f64>,
    pub(crate) b: Array1<f64>,
}

impl LayerNorm {
    pub fn new(w: Array1<f64>, b: Array1<f64>) -> Self {
        Self { w, b }
    }

    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        crate::tensor_ops::layer_norm(x, &self.w, &self.b)
    }

    pub fn num_params(&self) -> usize {
        self.w.len() + self.b.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};
    use rand::Rng;

    #[test]
    fn test_linear_forward() {
        let w = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let b = arr1(&[0.5, 0.5]);
        let layer = Linear::new(w, b);
        let x = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let y = layer.forward(&x);
        assert!((y[[0, 0]] - 1.5).abs() < 1e-10);
        assert!((y[[0, 1]] - 2.5).abs() < 1e-10);
        assert!((y[[1, 0]] - 3.5).abs() < 1e-10);
        assert!((y[[1, 1]] - 4.5).abs() < 1e-10);
    }

    #[test]
    fn test_linear_num_params() {
        let w = Array2::zeros((10, 20));
        let b = Array1::zeros(20);
        let layer = Linear::new(w, b);
        assert_eq!(layer.num_params(), 220);
    }

    #[test]
    fn test_scaled_dot_product_attention_shape() {
        let q = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        let k = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let v = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let (out, weights) = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(weights.shape(), &[2, 3]);
    }

    #[test]
    fn test_scaled_dot_product_attention_weights_sum_to_one() {
        let q = arr2(&[[1.0, 0.0]]);
        let k = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let v = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let (_, weights) = scaled_dot_product_attention(&q, &k, &v, None);
        let row_sum: f64 = weights.row(0).sum();
        assert!((row_sum - 1.0).abs() < 1e-10, "weights sum to {}", row_sum);
    }

    #[test]
    fn test_multi_head_attention_output_shape() {
        let d_model = 16;
        let n_heads = 4;
        let mut rng = rand::thread_rng();
        let w_q = Linear::new(
            crate::tensor_ops::xavier_init(d_model, d_model, &mut rng),
            Array1::zeros(d_model),
        );
        let w_k = Linear::new(
            crate::tensor_ops::xavier_init(d_model, d_model, &mut rng),
            Array1::zeros(d_model),
        );
        let w_v = Linear::new(
            crate::tensor_ops::xavier_init(d_model, d_model, &mut rng),
            Array1::zeros(d_model),
        );
        let w_o = Linear::new(
            crate::tensor_ops::xavier_init(d_model, d_model, &mut rng),
            Array1::zeros(d_model),
        );
        let mha = MultiHeadAttention::new(d_model, n_heads, w_q, w_k, w_v, w_o, 0.0);
        let x = Array2::from_shape_fn((5, d_model), |_| rng.gen::<f64>());
        let out = mha.forward(&x, &x, &x, None, false);
        assert_eq!(out.shape(), &[5, d_model]);
    }

    #[test]
    fn test_feed_forward_output_shape() {
        let mut rng = rand::thread_rng();
        let w1 = Linear::new(
            crate::tensor_ops::xavier_init(16, 32, &mut rng),
            Array1::zeros(32),
        );
        let w2 = Linear::new(
            crate::tensor_ops::xavier_init(32, 16, &mut rng),
            Array1::zeros(16),
        );
        let ffn = FeedForward::new(w1, w2, 0.0);
        let x = Array2::from_shape_fn((4, 16), |_| rng.gen::<f64>());
        let y = ffn.forward(&x, false);
        assert_eq!(y.shape(), &[4, 16]);
    }

    #[test]
    fn test_layer_norm_output_shape() {
        let ln = LayerNorm::new(Array1::ones(8), Array1::zeros(8));
        let x = arr2(&[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]);
        let y = ln.forward(&x);
        assert_eq!(y.shape(), &[1, 8]);
    }
}

use ndarray::{Array1, Array2, Axis};
use rand::Rng;

/// Blocked/tiling matrix multiplication for cache efficiency.
///
/// C = A @ B
/// A: [m, k], B: [k, n] -> C: [m, n]
///
/// Uses 64x64 blocks to keep working set in L1 cache.
/// Innermost loop is over j (output columns) for contiguous writes.
pub fn matmul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    assert_eq!(a.ncols(), b.nrows(), "matmul: dimension mismatch");
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();
    let mut c = Array2::zeros((m, n));

    let block = 64;
    for bk in (0..k).step_by(block) {
        let bk_end = (bk + block).min(k);
        for bi in (0..m).step_by(block) {
            let bi_end = (bi + block).min(m);
            for bj in (0..n).step_by(block) {
                let bj_end = (bj + block).min(n);
                for i in bi..bi_end {
                    for l in bk..bk_end {
                        let a_val = a[[i, l]];
                        for j in bj..bj_end {
                            c[[i, j]] += a_val * b[[l, j]];
                        }
                    }
                }
            }
        }
    }
    c
}

/// Transpose a 2D matrix.
pub fn transpose(a: &Array2<f64>) -> Array2<f64> {
    a.t().to_owned()
}

/// Numerically stable softmax along the last axis.
///
/// Subtracts row-wise max before exp to prevent overflow.
pub fn softmax(a: &Array2<f64>) -> Array2<f64> {
    let mut result = a.clone();
    for mut row in result.rows_mut() {
        let max = row.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
        let mut sum = 0.0;
        for x in row.iter_mut() {
            *x = (*x - max).exp();
            sum += *x;
        }
        let inv_sum = 1.0 / sum;
        for x in row.iter_mut() {
            *x *= inv_sum;
        }
    }
    result
}

/// ReLU activation: max(0, x).
pub fn relu(a: &Array2<f64>) -> Array2<f64> {
    a.mapv(|x| x.max(0.0))
}

/// Layer normalization along the last dimension.
///
/// y = (x - mean) / sqrt(var + eps) * w + b
///
/// Paper: Section 3.1, applied around each sub-layer.
pub fn layer_norm(x: &Array2<f64>, w: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let eps = 1e-6;

    let mean = x.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1));
    let var = x.var_axis(Axis(1), 0.0).insert_axis(Axis(1));
    let std = (&var + eps).mapv(f64::sqrt);

    let x_norm = (x - &mean) / &std;
    &x_norm * w + b
}

/// Embedding lookup: gather rows from embedding matrix by token indices.
pub fn embedding_lookup(embeddings: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
    let d = embeddings.ncols();
    let mut result = Array2::zeros((indices.len(), d));
    for (i, &idx) in indices.iter().enumerate() {
        result.row_mut(i).assign(&embeddings.row(idx));
    }
    result
}

/// Xavier/Glorot uniform initialization.
///
/// Draws from U[-limit, limit] where limit = sqrt(6 / (fan_in + fan_out)).
pub fn xavier_init(rows: usize, cols: usize, rng: &mut impl Rng) -> Array2<f64> {
    let limit = (6.0 / (rows + cols) as f64).sqrt();
    Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-limit..limit))
}

/// Scaled initialization per Section 5.3.
///
/// Attention weights scaled by 1 / sqrt(d_model).
#[allow(dead_code)]
pub fn scaled_init(rows: usize, cols: usize, d_model: usize, rng: &mut impl Rng) -> Array2<f64> {
    let std = 1.0 / (d_model as f64).sqrt();
    normal_init(rows, cols, std, rng)
}

/// Normal (Gaussian) initialization using Box-Muller transform.
pub fn normal_init(rows: usize, cols: usize, std: f64, rng: &mut impl Rng) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |_| {
        let u1: f64 = rng.gen();
        let u2: f64 = rng.gen();
        (u1.sqrt() * (-2.0 * u2.ln()).cos()) * std
    })
}

/// Uniform initialization in [-limit, limit].
#[allow(dead_code)]
pub fn uniform_init(rows: usize, cols: usize, limit: f64, rng: &mut impl Rng) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-limit..limit))
}

/// Dropout: randomly zero out elements with probability p, scale survivors by 1/(1-p).
pub fn dropout(x: &Array2<f64>, p: f64, train: bool) -> Array2<f64> {
    if !train || p == 0.0 {
        return x.clone();
    }
    let (nrows, ncols) = x.dim();
    let total = nrows * ncols;
    let keep_prob = 1.0 - p;
    let scale = 1.0 / keep_prob;
    let mut rng = rand::thread_rng();
    let mut mask_flat: Vec<f64> = Vec::with_capacity(total);
    mask_flat.resize_with(total, || {
        if rng.gen::<f64>() < p {
            0.0
        } else {
            scale
        }
    });
    let mask = ndarray::Array2::from_shape_vec((nrows, ncols), mask_flat).unwrap();
    x * &mask
}

/// Create a causal (lower-triangular) mask for decoder self-attention.
///
/// Returns [seq_len, seq_len] with 1.0 where attention is allowed, 0.0 where blocked.
pub fn causal_mask(seq_len: usize) -> Array2<f64> {
    let mut mask = Array2::zeros((seq_len, seq_len));
    for i in 0..seq_len {
        for j in 0..=i {
            mask[[i, j]] = 1.0;
        }
    }
    mask
}

/// Create padding mask from token ids.
///
/// Returns [seq_len] with 1.0 for real tokens, 0.0 for padding.
pub fn padding_mask(token_ids: &[usize], pad_id: usize) -> Array1<f64> {
    Array1::from_shape_fn(token_ids.len(), |i| {
        if token_ids[i] == pad_id {
            0.0
        } else {
            1.0
        }
    })
}

/// Combine causal + padding masks into additive mask for attention scores.
///
/// Returns [tgt_len, tgt_len] where allowed=0.0, blocked=f64::NEG_INFINITY.
pub fn make_decoder_mask(causal: &Array2<f64>, padding: &Array1<f64>) -> Array2<f64> {
    let seq_len = causal.nrows();
    let mut mask = Array2::zeros((seq_len, seq_len));
    for i in 0..seq_len {
        for j in 0..seq_len {
            if causal[[i, j]] == 0.0 || padding[j] == 0.0 {
                mask[[i, j]] = f64::NEG_INFINITY;
            }
        }
    }
    mask
}

/// Combine source padding into additive mask for encoder-decoder attention.
///
/// Returns [1, src_len] where allowed=0.0, blocked=f64::NEG_INFINITY.
pub fn make_src_mask(padding: &Array1<f64>) -> Array2<f64> {
    let src_len = padding.len();
    let mut mask = Array2::zeros((1, src_len));
    for j in 0..src_len {
        if padding[j] == 0.0 {
            mask[[0, j]] = f64::NEG_INFINITY;
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_matmul_basic() {
        let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let b = arr2(&[[5.0, 6.0], [7.0, 8.0]]);
        let c = matmul(&a, &b);
        // [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3*5+4*7, 3*6+4*8] = [43, 50]
        assert!((c[[0, 0]] - 19.0).abs() < 1e-10);
        assert!((c[[0, 1]] - 22.0).abs() < 1e-10);
        assert!((c[[1, 0]] - 43.0).abs() < 1e-10);
        assert!((c[[1, 1]] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_matmul_identity() {
        let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let eye = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let c = matmul(&a, &eye);
        for i in 0..2 {
            for j in 0..2 {
                assert!((c[[i, j]] - a[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_matmul_transpose() {
        let a = arr2(&[[1.0, 2.0, 3.0]]);
        let b = arr2(&[[4.0], [5.0], [6.0]]);
        let c = matmul(&a, &b);
        // 1*4 + 2*5 + 3*6 = 32
        assert!((c[[0, 0]] - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_transpose() {
        let a = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let at = transpose(&a);
        assert_eq!(at.shape(), &[3, 2]);
        assert!((at[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((at[[2, 1]] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let a = arr2(&[[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]]);
        let s = softmax(&a);
        for i in 0..2 {
            let row_sum: f64 = s.row(i).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-10,
                "row {} sums to {}",
                i,
                row_sum
            );
        }
    }

    #[test]
    fn test_softmax_uniform() {
        let a = arr2(&[[1.0, 1.0, 1.0]]);
        let s = softmax(&a);
        for j in 0..3 {
            assert!((s[[0, j]] - 1.0 / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values should not overflow
        let a = arr2(&[[1000.0, 1001.0, 1002.0]]);
        let s = softmax(&a);
        let row_sum: f64 = s.row(0).sum();
        assert!((row_sum - 1.0).abs() < 1e-6);
        assert!(s[[0, 2]] > s[[0, 0]]);
    }

    #[test]
    fn test_relu() {
        let a = arr2(&[[-1.0, 2.0, 0.0], [3.0, -4.0, 5.0]]);
        let r = relu(&a);
        assert!((r[[0, 0]] - 0.0).abs() < 1e-10);
        assert!((r[[0, 1]] - 2.0).abs() < 1e-10);
        assert!((r[[0, 2]] - 0.0).abs() < 1e-10);
        assert!((r[[1, 0]] - 3.0).abs() < 1e-10);
        assert!((r[[1, 1]] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_layer_norm_zero_mean() {
        let x = arr2(&[[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]);
        let w = Array1::ones(4);
        let b = Array1::zeros(4);
        let y = layer_norm(&x, &w, &b);
        // Each row should have approximately zero mean
        for i in 0..2 {
            let mean = y.row(i).mean().unwrap();
            assert!(mean.abs() < 1e-5, "row {} mean = {}", i, mean);
        }
    }

    #[test]
    fn test_layer_norm_unit_variance() {
        let x = arr2(&[[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]);
        let w = Array1::ones(4);
        let b = Array1::zeros(4);
        let y = layer_norm(&x, &w, &b);
        for i in 0..2 {
            let var = y.row(i).var_axis(Axis(0), 0.0);
            let var_val = *var.iter().next().unwrap();
            assert!(
                (var_val - 1.0_f64).abs() < 1e-2,
                "row {} var = {}",
                i,
                var_val
            );
        }
    }

    #[test]
    fn test_embedding_lookup() {
        let emb = arr2(&[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]);
        let ids = [2, 0, 1];
        let out = embedding_lookup(&emb, &ids);
        assert_eq!(out.shape(), &[3, 2]);
        assert!((out[[0, 0]] - 0.5).abs() < 1e-10);
        assert!((out[[1, 1]] - 0.2).abs() < 1e-10);
        assert!((out[[2, 0]] - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_xavier_init_range() {
        let mut rng = rand::thread_rng();
        let w = xavier_init(128, 128, &mut rng);
        let limit = (6.0_f64 / 256.0).sqrt();
        for &v in w.iter() {
            assert!(v >= -limit - 1e-10 && v <= limit + 1e-10);
        }
    }

    #[test]
    fn test_dropout_no_train() {
        let x = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let y = dropout(&x, 0.5, false);
        for i in 0..2 {
            for j in 0..2 {
                assert!((y[[i, j]] - x[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_dropout_zero_rate() {
        let x = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let y = dropout(&x, 0.0, true);
        for i in 0..2 {
            for j in 0..2 {
                assert!((y[[i, j]] - x[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_dropout_preserves_structure() {
        let x = arr2(&[[1.0, 2.0, 3.0, 4.0, 5.0]]);
        // With p=0.5 and 100 samples, some should be zeroed and some preserved
        let mut has_zero = false;
        let mut has_nonzero = false;
        for _ in 0..100 {
            let y = dropout(&x, 0.5, true);
            for j in 0..5 {
                if y[[0, j]] == 0.0 {
                    has_zero = true;
                }
                if (y[[0, j]] - x[[0, j]] * 2.0).abs() < 1e-10 {
                    has_nonzero = true;
                }
            }
            if has_zero && has_nonzero {
                break;
            }
        }
        assert!(has_zero, "dropout should zero some elements");
        assert!(has_nonzero, "dropout should preserve some elements");
    }

    #[test]
    fn test_causal_mask() {
        let m = causal_mask(4);
        assert_eq!(m.shape(), &[4, 4]);
        // Upper triangle should be 0
        assert!((m[[0, 1]] - 0.0).abs() < 1e-10);
        assert!((m[[0, 2]] - 0.0).abs() < 1e-10);
        assert!((m[[0, 3]] - 0.0).abs() < 1e-10);
        assert!((m[[1, 2]] - 0.0).abs() < 1e-10);
        // Diagonal and lower should be 1
        assert!((m[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((m[[1, 0]] - 1.0).abs() < 1e-10);
        assert!((m[[1, 1]] - 1.0).abs() < 1e-10);
        assert!((m[[3, 3]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_padding_mask() {
        let ids = vec![5, 0, 3, 0, 7];
        let mask = padding_mask(&ids, 0);
        let expected = arr1(&[1.0, 0.0, 1.0, 0.0, 1.0]);
        for i in 0..5 {
            assert!((mask[i] - expected[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_make_src_mask() {
        let padding = arr1(&[1.0, 0.0, 1.0]);
        let mask = make_src_mask(&padding);
        assert_eq!(mask.shape(), &[1, 3]);
        assert!((mask[[0, 0]] - 0.0).abs() < 1e-10);
        assert!(mask[[0, 1]].is_infinite());
        assert!((mask[[0, 2]] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_make_decoder_mask() {
        let causal = causal_mask(3);
        let padding = arr1(&[1.0, 1.0, 0.0]);
        let mask = make_decoder_mask(&causal, &padding);
        // Position 2 is padding, so column 2 should be -inf
        assert!(mask[[0, 2]].is_infinite());
        assert!(mask[[1, 2]].is_infinite());
        assert!(mask[[2, 2]].is_infinite());
        // Upper triangle positions should be -inf
        assert!(mask[[0, 1]].is_infinite());
        assert!(mask[[0, 2]].is_infinite());
        // Lower triangle positions should be 0
        assert!((mask[[1, 0]] - 0.0).abs() < 1e-10);
        assert!((mask[[2, 0]] - 0.0).abs() < 1e-10);
        assert!((mask[[2, 1]] - 0.0).abs() < 1e-10);
    }
}

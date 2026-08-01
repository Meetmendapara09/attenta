use ndarray::{Array1, Array2, Axis};

/// Backward pass for blocked matmul: C = A @ B.
///
/// Returns (dA, dB).
pub fn matmul_backward(
    grad: &Array2<f64>,
    a: &Array2<f64>,
    b: &Array2<f64>,
) -> (Array2<f64>, Array2<f64>) {
    // dA = grad @ B^T,  dB = A^T @ grad
    let d_a = crate::tensor_ops::matmul(grad, &crate::tensor_ops::transpose(b));
    let d_b = crate::tensor_ops::matmul(&crate::tensor_ops::transpose(a), grad);
    (d_a, d_b)
}

/// Backward pass for softmax.
///
/// Given dy and softmax output y, compute dx = y * (dy - sum(y * dy, axis=-1)).
pub fn softmax_backward(grad: &Array2<f64>, softmax_out: &Array2<f64>) -> Array2<f64> {
    // dx_i = y_i * (dy_i - sum_j(y_j * dy_j))
    let dot = (softmax_out * grad).sum_axis(Axis(1)).insert_axis(Axis(1));
    softmax_out * &(grad - &dot)
}

/// Backward pass for ReLU.
pub fn relu_backward(grad: &Array2<f64>, input: &Array2<f64>) -> Array2<f64> {
    grad.mapv(|g| if g > 0.0 { g } else { 0.0 }) * &input.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
}

/// Backward pass for layer normalization.
///
/// Returns (dx, dw, db).
#[allow(dead_code)]
pub fn layer_norm_backward(
    grad: &Array2<f64>,
    input: &Array2<f64>,
    w: &Array1<f64>,
    mean: &Array1<f64>,
    var: &Array1<f64>,
) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let eps = 1e-6;
    let d_model = input.ncols() as f64;
    let mean_2d = mean.clone().insert_axis(Axis(1));
    let var_2d = var.clone().insert_axis(Axis(1));
    let std_inv = (&var_2d + eps).mapv(|v| 1.0 / v.sqrt());

    // Normalized input: x_hat = (x - mean) / std
    let x_hat = (input - &mean_2d) * &std_inv;

    // dw = sum(grad * x_hat, axis=0)
    let dw = (grad * &x_hat).sum_axis(Axis(0));

    // db = sum(grad, axis=0)
    let db = grad.sum_axis(Axis(0));

    // dx = (1/std) * (grad - mean(grad) - x_hat * mean(grad * x_hat)) / sqrt(d_model)
    let grad_times_w = grad * w;
    let mean_gw = grad_times_w.sum_axis(Axis(1)).insert_axis(Axis(1)) / d_model;
    let mean_gw_xhat = (grad_times_w.clone() * &x_hat)
        .sum_axis(Axis(1))
        .insert_axis(Axis(1))
        / d_model;

    let dx = &std_inv * &(&grad_times_w - &mean_gw - &(&x_hat * &mean_gw_xhat));
    (dx, dw, db)
}

/// Backward pass for inverted dropout.
///
/// During training, mask is a boolean array where true = kept, false = dropped.
/// Scale factor was 1/(1-p).
pub fn dropout_backward(grad: &Array2<f64>, mask: &Array2<bool>, scale: f64) -> Array2<f64> {
    grad * &mask.mapv(|m| if m { scale } else { 0.0 })
}

/// Backward pass for linear layer: y = x @ w + b.
///
/// Returns (dx, dw, db).
pub fn linear_backward(
    grad: &Array2<f64>,
    input: &Array2<f64>,
    w: &Array2<f64>,
) -> (Array2<f64>, Array2<f64>, Array1<f64>) {
    let (dx, dw) = matmul_backward(grad, input, w);
    let db = grad.sum_axis(Axis(0));
    (dx, dw, db)
}

/// Backward pass for scaled dot-product attention.
///
/// Returns (dq, dk, dv).
pub fn attention_backward(
    grad: &Array2<f64>,
    weights: &Array2<f64>,
    q: &Array2<f64>,
    k: &Array2<f64>,
    v: &Array2<f64>,
    d_k: f64,
) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    // dV = weights^T @ grad
    let d_v = crate::tensor_ops::matmul(&crate::tensor_ops::transpose(weights), grad);

    // d_weights = grad @ V^T
    let d_weights = crate::tensor_ops::matmul(grad, &crate::tensor_ops::transpose(v));

    // Softmax backward: d_scores = weights * (d_weights - sum(weights * d_weights, axis=-1))
    let dot = (weights * &d_weights)
        .sum_axis(Axis(1))
        .insert_axis(Axis(1));
    let d_scores = weights * &(&d_weights - &dot);

    // dQ = d_scores @ K / sqrt(d_k)
    let d_q = crate::tensor_ops::matmul(&d_scores, k) / d_k.sqrt();

    // dK = d_scores^T @ Q / sqrt(d_k)
    let d_k_out =
        crate::tensor_ops::matmul(&crate::tensor_ops::transpose(&d_scores), q) / d_k.sqrt();

    (d_q, d_k_out, d_v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_matmul_backward_shapes() {
        let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let b = arr2(&[[5.0, 6.0], [7.0, 8.0]]);
        let grad = arr2(&[[1.0, 1.0], [1.0, 1.0]]);
        let (d_a, d_b) = matmul_backward(&grad, &a, &b);
        assert_eq!(d_a.shape(), a.shape());
        assert_eq!(d_b.shape(), b.shape());
    }

    #[test]
    fn test_matmul_backward_values() {
        let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let b = arr2(&[[5.0, 6.0], [7.0, 8.0]]);
        let grad = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let (d_a, _d_b) = matmul_backward(&grad, &a, &b);
        // dA = grad @ B^T = [[5,7],[6,8]]
        assert!((d_a[[0, 0]] - 5.0).abs() < 1e-10);
        assert!((d_a[[0, 1]] - 7.0).abs() < 1e-10);
        assert!((d_a[[1, 0]] - 6.0).abs() < 1e-10);
        assert!((d_a[[1, 1]] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_backward_identity() {
        // For uniform softmax with uniform grad, grad should pass through
        let softmax_out = arr2(&[[0.25, 0.25, 0.25, 0.25]]);
        let grad = arr2(&[[1.0, 1.0, 1.0, 1.0]]);
        let dx = softmax_backward(&grad, &softmax_out);
        // sum(y * dy) = 1.0, so dx = y * (1.0 - 1.0) = 0
        for i in 0..4 {
            assert!(dx[[0, i]].abs() < 1e-10);
        }
    }

    #[test]
    fn test_relu_backward() {
        let input = arr2(&[[-1.0, 2.0, 0.0, 3.0]]);
        let grad = arr2(&[[1.0, 1.0, 1.0, 1.0]]);
        let dx = relu_backward(&grad, &input);
        assert!((dx[[0, 0]] - 0.0).abs() < 1e-10); // -1 -> 0
        assert!((dx[[0, 1]] - 1.0).abs() < 1e-10); // 2 -> 1
        assert!((dx[[0, 2]] - 0.0).abs() < 1e-10); // 0 -> 0
        assert!((dx[[0, 3]] - 1.0).abs() < 1e-10); // 3 -> 1
    }

    #[test]
    fn test_linear_backward_shapes() {
        let grad = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let input = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let w = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let (dx, dw, db) = linear_backward(&grad, &input, &w);
        assert_eq!(dx.shape(), input.shape());
        assert_eq!(dw.shape(), w.shape());
        assert_eq!(db.shape(), &[2]);
    }

    #[test]
    fn test_dropout_backward_masked() {
        let grad = arr2(&[[2.0, 4.0, 6.0]]);
        let mask = arr2(&[[true, false, true]]);
        let scale = 2.0; // p = 0.5
        let dx = dropout_backward(&grad, &mask, scale);
        assert!((dx[[0, 0]] - 4.0).abs() < 1e-10); // 2.0 * 2.0
        assert!((dx[[0, 1]] - 0.0).abs() < 1e-10); // masked
        assert!((dx[[0, 2]] - 12.0).abs() < 1e-10); // 6.0 * 2.0
    }
}

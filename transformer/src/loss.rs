use ndarray::{Array2, Axis};

/// Label smoothing cross-entropy loss.
///
/// Paper: Section 5.4
/// "During training, we employed label smoothing of value epsilon_ls = 0.1.
///  This hurts perplexity, as the model learns to be more unsure, but
///  improves accuracy and BLEU score."
///
/// Computes: LS_CE = -(1 - eps) * sum(y_true * log(softmax(logits)))
///                     - (eps / V) * sum(log(softmax(logits)))
pub fn label_smoothing_loss(
    logits: &Array2<f64>,
    targets: &[usize],
    eps: f64,
    pad_id: usize,
) -> f64 {
    let (seq_len, vocab_size) = logits.dim();
    let vocab_f = vocab_size as f64;
    let log_probs = log_softmax(logits);

    let mut total_loss = 0.0;
    let mut count = 0;

    for (i, &tgt) in targets.iter().enumerate() {
        if i >= seq_len || tgt == pad_id {
            continue;
        }

        // True distribution: (1 - eps) at target, eps/V everywhere
        for j in 0..vocab_size {
            let smooth_target = if j == tgt {
                (1.0 - eps) + eps / vocab_f
            } else {
                eps / vocab_f
            };
            total_loss -= smooth_target * log_probs[[i, j]];
        }
        count += 1;
    }

    if count == 0 {
        0.0
    } else {
        total_loss / count as f64
    }
}

/// Standard (non-smoothed) cross-entropy loss for inference evaluation.
pub fn cross_entropy_loss(logits: &Array2<f64>, targets: &[usize], pad_id: usize) -> f64 {
    let (seq_len, _) = logits.dim();
    let log_probs = log_softmax(logits);

    let mut total_loss = 0.0;
    let mut count = 0;

    for (i, &tgt) in targets.iter().enumerate() {
        if i >= seq_len || tgt == pad_id {
            continue;
        }
        total_loss -= log_probs[[i, tgt]];
        count += 1;
    }

    if count == 0 {
        0.0
    } else {
        total_loss / count as f64
    }
}

/// Compute perplexity from loss: ppl = exp(loss).
pub fn perplexity(loss: f64) -> f64 {
    loss.exp()
}

/// Compute log-softmax along the last axis (numerically stable).
pub(crate) fn log_softmax(x: &Array2<f64>) -> Array2<f64> {
    let max = x.fold_axis(Axis(1), f64::NEG_INFINITY, |acc, &v| acc.max(v));
    let max = max.insert_axis(Axis(1));
    let log_sum = (x - &max).mapv(f64::exp).sum_axis(Axis(1)).mapv(f64::ln);
    let log_sum = log_sum.insert_axis(Axis(1));
    x - &max - &log_sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_label_smoothing_loss_basic() {
        // Perfect prediction: logit for correct class is very high
        let logits = Array2::from_shape_fn((3, 5), |(_, j)| if j == 2 { 10.0 } else { 0.0 });
        let targets = vec![2, 2, 2];
        let loss = label_smoothing_loss(&logits, &targets, 0.1, usize::MAX);
        assert!(loss >= 0.0, "loss should be non-negative");
        assert!(loss < 1.0, "loss for perfect prediction should be small");
    }

    #[test]
    fn test_cross_entropy_perfect() {
        let logits = Array2::from_shape_fn((2, 4), |(_, j)| if j == 1 { 100.0 } else { 0.0 });
        let targets = vec![1, 1];
        let loss = cross_entropy_loss(&logits, &targets, usize::MAX);
        assert!(
            loss < 0.01,
            "loss near 0 for perfect prediction, got {}",
            loss
        );
    }

    #[test]
    fn test_perplexity() {
        let ppl = perplexity(0.0);
        assert!((ppl - 1.0).abs() < 1e-6);
    }
}

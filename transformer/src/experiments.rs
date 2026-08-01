/// Model variation experiments matching Table 3 from the paper.
///
/// Paper Table 3: "Variations on the Transformer architecture"
/// Tests the effect of varying N, d_model, d_ff, h, dropout, and label_smoothing
/// on model performance (loss/perplexity).
///
/// The paper reports BLEU scores on WMT 2014, but here we report
/// loss/perplexity on synthetic copy tasks as a proxy.
use std::time::Instant;

use crate::loss::perplexity;
use crate::model::{Transformer, TransformerConfig};
use crate::train::{train_step, AdamState};

/// Results from a single experiment run.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExperimentResult {
    /// Experiment label
    pub label: String,
    /// Configuration
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub n_layers: usize,
    pub dropout: f64,
    pub label_smoothing: f64,
    /// Training results
    pub final_loss: f64,
    pub final_perplexity: f64,
    pub training_time_secs: f64,
}

/// Generate a single copy task sample for training.
///
/// src = random tokens, tgt_in = BOS + src[:-1], tgt_out = src (predict next token).
pub fn single_copy_sample(
    seq_len: usize,
    vocab: usize,
    bos_id: usize,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let src: Vec<usize> = (0..seq_len)
        .map(|_| (rand::random::<usize>() % (vocab - 3)) + 3)
        .collect();
    let mut tgt_in = vec![bos_id];
    tgt_in.extend_from_slice(&src[..seq_len - 1]);
    let tgt_out = src.clone();
    (src, tgt_in, tgt_out)
}

/// Run a single experiment with the given configuration.
///
/// Trains for `n_steps` on a synthetic copy task and returns the result.
pub fn run_experiment(
    label: &str,
    config: TransformerConfig,
    n_steps: usize,
    seq_len: usize,
    _batch_size: usize,
) -> ExperimentResult {
    let mut model = Transformer::new(config.clone());
    let mut adam = AdamState::new(config.d_model, config.warmup_steps, config.n_layers);
    for layer in &model.decoder.layers {
        adam.init_dec_layer(layer);
    }
    for layer in &model.encoder.layers {
        adam.init_enc_layer(layer);
    }
    adam.init_param("out_proj_w", (config.d_model, config.tgt_vocab));
    adam.init_param("tgt_emb", (config.tgt_vocab, config.d_model));
    adam.init_param("src_emb", (config.src_vocab, config.d_model));

    let start = Instant::now();
    let mut final_loss = 0.0;

    for step in 1..=n_steps {
        let (src, tgt_in, tgt_out) = single_copy_sample(seq_len, config.src_vocab, config.bos_id);
        let loss = train_step(&mut model, &src, &tgt_in, &tgt_out, &mut adam);
        final_loss = loss;

        if step % 20 == 0 || step == n_steps {
            let ppl = perplexity(loss);
            println!(
                "    {} step {}/{}: loss={:.4} ppl={:.2}",
                label, step, n_steps, loss, ppl
            );
        }
    }

    let elapsed = start.elapsed().as_secs_f64();

    ExperimentResult {
        label: label.to_string(),
        d_model: config.d_model,
        n_heads: config.n_heads,
        d_ff: config.d_ff,
        n_layers: config.n_layers,
        dropout: config.dropout,
        label_smoothing: config.label_smoothing,
        final_loss,
        final_perplexity: perplexity(final_loss),
        training_time_secs: elapsed,
    }
}

/// Run all Table 3 experiments and print results.
///
/// This tests variations of:
/// - N (number of layers): 2, 4, 6
/// - d_model: 128, 256, 512 (paper base)
/// - d_ff: 512, 1024, 2048 (paper base)
/// - h (heads): 2, 4, 8
/// - dropout: 0.0, 0.1, 0.2, 0.3
/// - label_smoothing: 0.0, 0.1, 0.2
#[allow(dead_code)]
pub fn run_experiments(
    n_steps: usize,
    seq_len: usize,
    batch_size: usize,
    output_path: &str,
) -> Vec<ExperimentResult> {
    let base_config = TransformerConfig {
        src_vocab: 50,
        tgt_vocab: 50,
        d_model: 64,
        n_heads: 4,
        d_ff: 128,
        n_layers: 2,
        max_len: 64,
        dropout: 0.1,
        pad_id: 0,
        bos_id: 1,
        eos_id: 2,
        label_smoothing: 0.1,
        warmup_steps: 100,
    };

    let mut results = Vec::new();

    println!("\n=== Running Table 3 Model Variation Experiments ===\n");
    println!(
        "Base config: d_model={}, n_heads={}, d_ff={}, n_layers={}",
        base_config.d_model, base_config.n_heads, base_config.d_ff, base_config.n_layers
    );
    println!(
        "  dropout={}, label_smoothing={}",
        base_config.dropout, base_config.label_smoothing
    );
    println!("  Training steps per config: {}\n", n_steps);

    // 1. Vary number of layers (N)
    println!("--- Varying N (number of layers) ---");
    for n_layers in [2, 4, 6] {
        let mut cfg = base_config.clone();
        cfg.n_layers = n_layers;
        let label = format!("N={}", n_layers);
        let result = run_experiment(&label, cfg, n_steps, seq_len, batch_size);
        results.push(result);
    }

    // 2. Vary d_model
    println!("\n--- Varying d_model ---");
    for d_model in [32, 64, 128] {
        let mut cfg = base_config.clone();
        cfg.d_model = d_model;
        // Adjust d_ff to maintain 4x ratio
        cfg.d_ff = d_model * 4;
        let label = format!("d_model={}", d_model);
        let result = run_experiment(&label, cfg, n_steps, seq_len, batch_size);
        results.push(result);
    }

    // 3. Vary d_ff
    println!("\n--- Varying d_ff ---");
    for d_ff in [64, 128, 256] {
        let mut cfg = base_config.clone();
        cfg.d_ff = d_ff;
        let label = format!("d_ff={}", d_ff);
        let result = run_experiment(&label, cfg, n_steps, seq_len, batch_size);
        results.push(result);
    }

    // 4. Vary number of heads
    println!("\n--- Varying h (heads) ---");
    for n_heads in [2, 4, 8] {
        let mut cfg = base_config.clone();
        cfg.n_heads = n_heads;
        // Ensure d_model is divisible by n_heads
        cfg.d_model = 64.max((64 / n_heads) * n_heads);
        let label = format!("h={}", n_heads);
        let result = run_experiment(&label, cfg, n_steps, seq_len, batch_size);
        results.push(result);
    }

    // 5. Vary dropout
    println!("\n--- Varying dropout ---");
    for dropout in [0.0, 0.1, 0.2, 0.3] {
        let mut cfg = base_config.clone();
        cfg.dropout = dropout;
        let label = format!("dropout={}", dropout);
        let result = run_experiment(&label, cfg, n_steps, seq_len, batch_size);
        results.push(result);
    }

    // 6. Vary label smoothing
    println!("\n--- Varying label_smoothing ---");
    for ls in [0.0, 0.1, 0.2] {
        let mut cfg = base_config.clone();
        cfg.label_smoothing = ls;
        let label = format!("ls={}", ls);
        let result = run_experiment(&label, cfg, n_steps, seq_len, batch_size);
        results.push(result);
    }

    // Print summary table
    println!("\n\n=== Table 3 Results Summary ===");
    println!(
        "{:<20} {:>8} {:>8} {:>8} {:>8} {:>8} {:>12} {:>12} {:>12}",
        "Variant", "d_model", "n_heads", "d_ff", "N", "dropout", "ls", "Loss", "PPL"
    );
    println!("{}", "-".repeat(110));
    for r in &results {
        println!(
            "{:<20} {:>8} {:>8} {:>8} {:>8} {:>8.1} {:>12} {:>12.4} {:>12.2}",
            r.label,
            r.d_model,
            r.n_heads,
            r.d_ff,
            r.n_layers,
            r.dropout,
            format!("{:.1}", r.label_smoothing),
            r.final_loss,
            r.final_perplexity
        );
    }

    // Save to JSON
    if !output_path.is_empty() {
        let json = serde_json::to_string_pretty(&results).unwrap_or_else(|_| "[]".to_string());
        std::fs::write(output_path, json)
            .unwrap_or_else(|_| println!("Warning: Could not write results to {}", output_path));
        println!("\nResults saved to: {}", output_path);
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_experiment_produces_result() {
        let config = TransformerConfig {
            src_vocab: 30,
            tgt_vocab: 30,
            d_model: 16,
            n_heads: 4,
            d_ff: 32,
            n_layers: 1,
            max_len: 32,
            dropout: 0.0,
            pad_id: 0,
            bos_id: 1,
            eos_id: 2,
            label_smoothing: 0.1,
            warmup_steps: 50,
        };
        let result = run_experiment("test", config, 10, 8, 4);
        assert!(result.final_loss >= 0.0);
        assert!(result.final_perplexity >= 1.0);
        assert!(result.training_time_secs >= 0.0);
    }
}

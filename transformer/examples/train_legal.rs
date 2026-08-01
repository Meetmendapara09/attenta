//! Training example: Train the Transformer on Indian legal documents.
//!
//! Run with: cargo run --release --example train_legal [STEPS] [DATA_DIR] [CHECKPOINT_DIR]
//!
//! Defaults: 100 steps, ../data/, ./checkpoints/
//!
//! Paper: Section 5.1 — "We used the WMT 2014 English-German dataset..."
//! Here we train on Indian labor law documents instead.

use attenta::data::{LegalDataset, SyntheticDatasetConfig};
use attenta::loss::perplexity;
use attenta::model::{Transformer, TransformerConfig};
use attenta::train::{StepTimer, TrainingMetrics};
use attenta::checkpoint::save_checkpoint;
use attenta::train::train_step;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_steps: usize = args.get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let data_dir: String = args.get(2).cloned().unwrap_or_else(|| "../data/".to_string());
    let checkpoint_dir: String = args.get(3).cloned().unwrap_or_else(|| "./checkpoints/".to_string());

    println!("==================================================================");
    println!("  Attenta — Training on Indian Legal Documents");
    println!("  Dataset: {}", data_dir);
    println!("  Steps: {}", n_steps);
    println!("  Checkpoints: {}", checkpoint_dir);
    println!("==================================================================");

    std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create checkpoint dir");

    // Load and tokenize legal data
    println!("\n[1/4] Loading legal dataset...");
    let legal_dataset = LegalDataset::from_directory(
        &data_dir,
        800,    // vocab_size
        64,     // max_len
        0,      // pad_id
        1,      // bos_id
        2,      // eos_id
        0.9,    // train_ratio
    ).expect("Failed to load legal dataset");

    legal_dataset.print_stats();

    // Build model from dataset config
    let config = TransformerConfig {
        src_vocab: legal_dataset.config.src_vocab,
        tgt_vocab: legal_dataset.config.tgt_vocab,
        d_model: 128,
        n_heads: 4,
        d_ff: 256,
        n_layers: 4,
        max_len: 64,
        dropout: 0.1,
        pad_id: 0,
        bos_id: 1,
        eos_id: 2,
        label_smoothing: 0.1,
        warmup_steps: 4000,
    };

    println!("\n[2/4] Building Transformer model...");
    println!("  d_model={}, n_heads={}, n_layers={}, d_ff={}", config.d_model, config.n_heads, config.n_layers, config.d_ff);
    println!("  Parameters: {:>10}", {
        let m = Transformer::new(config.clone());
        m.num_parameters()
    });

    let mut model = Transformer::new(config.clone());
    let mut adam = model.init_adam();

    // Training loop
    println!("\n[3/4] Training...");
    let mut timer = StepTimer::new();
    let mut metrics = TrainingMetrics::new();

    for step in 1..=n_steps {
        timer.step_begin();

        // Sample a random training example
        let idx = rand::random::<usize>() % legal_dataset.train_data.len();
        let (src, tgt_in, tgt_out) = &legal_dataset.train_data[idx];

        let loss = train_step(&mut model, src, tgt_in, tgt_out, &mut adam);
        let tokens = src.len() + tgt_in.len();
        timer.step_end(tokens);
        metrics.record(loss);

        let ppl = perplexity(loss);
        if step % 10 == 0 || step == n_steps {
            println!(
                "  Step {:>5}/{} | loss={:.4} | ppl={:.2} | lr={:.2e} | {:.0} tok/s",
                step, n_steps, loss, ppl, adam.learning_rate(), timer.avg_throughput
            );
        }

        // Checkpoint every 50 steps
        if step % 50 == 0 {
            let ckpt_path = format!("{}/checkpoint_step_{}.json", checkpoint_dir.trim_end_matches('/'), step);
            let _ = save_checkpoint(&model, &ckpt_path);
            println!("  Saved checkpoint: {}", ckpt_path);
        }
    }

    // Save final checkpoint
    let final_path = format!("{}/checkpoint_final.json", checkpoint_dir.trim_end_matches('/'));
    save_checkpoint(&model, &final_path).expect("Failed to save final checkpoint");
    println!("\n  Final checkpoint saved: {}", final_path);

    // Validation
    println!("\n[4/4] Validation...");
    if !legal_dataset.val_data.is_empty() {
        let val_count = legal_dataset.val_data.len().min(10);
        let mut val_loss = 0.0;
        for i in 0..val_count {
            let (src, tgt_in, tgt_out) = &legal_dataset.val_data[i];
            val_loss += train_step(&mut model, src, tgt_in, tgt_out, &mut adam);
        }
        val_loss /= val_count as f64;
        println!("  Validation loss: {:.4} | ppl: {:.2}", val_loss, perplexity(val_loss));
    }

    println!("\n==================================================================");
    println!("  Training Summary");
    println!("==================================================================");
    println!("  Total steps:     {}", metrics.total_steps);
    println!("  Average loss:    {:.4}", metrics.avg_loss());
    println!("  Best loss:       {:.4}", metrics.best_loss);
    println!("  Avg step time:   {:.3}s", timer.avg_step_time);
    println!("  Total elapsed:   {:.1}s", timer.total_elapsed());
    println!("  Throughput:      {:.0} tok/s", timer.total_throughput());
    println!("  Checkpoints:     {}", checkpoint_dir);
    println!("==================================================================");
}

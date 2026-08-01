/// Synthetic WMT-like dataset generator for training and evaluation.
///
/// Paper 5.1: "We used the WMT 2014 English-German dataset..."
/// This module generates synthetic data that mimics variable-length
/// sequence pairs for testing the training pipeline.
use rand::Rng;
use std::fs;
use serde::Deserialize;

/// Configuration for synthetic dataset generation.
#[derive(Debug, Clone)]
pub struct SyntheticDatasetConfig {
    /// Source vocabulary size
    pub src_vocab: usize,
    /// Target vocabulary size
    pub tgt_vocab: usize,
    /// Special token IDs
    pub pad_id: usize,
    pub bos_id: usize,
    pub eos_id: usize,
    /// Minimum sequence length (excluding special tokens)
    pub min_len: usize,
    /// Maximum sequence length (excluding special tokens)
    pub max_len: usize,
    /// Number of training samples
    pub train_samples: usize,
    /// Number of validation samples
    pub val_samples: usize,
    /// If true, target is a shifted copy of source (for copy task)
    /// If false, target is a random permutation of source tokens
    pub copy_task: bool,
}

impl Default for SyntheticDatasetConfig {
    fn default() -> Self {
        Self {
            src_vocab: 100,
            tgt_vocab: 100,
            pad_id: 0,
            bos_id: 1,
            eos_id: 2,
            min_len: 5,
            max_len: 20,
            train_samples: 1000,
            val_samples: 100,
            copy_task: true,
        }
    }
}

/// A synthetic dataset with train/validation splits.
///
/// Each sample is (src_tokens, tgt_input_tokens, tgt_output_tokens).
/// - src_tokens: source sequence tokens
/// - tgt_input_tokens: BOS + target sequence (decoder input)
/// - tgt_output_tokens: target sequence + EOS (what decoder should predict)
#[derive(Debug, Clone)]
pub struct SyntheticDataset {
    pub config: SyntheticDatasetConfig,
    pub train_data: Vec<(Vec<usize>, Vec<usize>, Vec<usize>)>,
    pub val_data: Vec<(Vec<usize>, Vec<usize>, Vec<usize>)>,
}

impl SyntheticDataset {
    /// Generate a new synthetic dataset with the given configuration.
    pub fn new(config: SyntheticDatasetConfig) -> Self {
        let mut rng = rand::thread_rng();
        let train_data = Self::generate_samples(&config, config.train_samples, &mut rng);
        let val_data = Self::generate_samples(&config, config.val_samples, &mut rng);
        Self {
            config,
            train_data,
            val_data,
        }
    }

    /// Generate random samples.
    fn generate_samples(
        config: &SyntheticDatasetConfig,
        count: usize,
        rng: &mut impl Rng,
    ) -> Vec<(Vec<usize>, Vec<usize>, Vec<usize>)> {
        let mut samples = Vec::with_capacity(count);

        for _ in 0..count {
            let seq_len = rng.gen_range(config.min_len..=config.max_len);
            let src: Vec<usize> = (0..seq_len)
                .map(|_| rng.gen_range(3..config.src_vocab))
                .collect();

            let (tgt_input, tgt_output) = if config.copy_task {
                // Copy task: target = source (for language modeling style)
                let mut tgt_in = vec![config.bos_id];
                tgt_in.extend_from_slice(&src[..seq_len - 1]);
                let tgt_out = src.clone();
                (tgt_in, tgt_out)
            } else {
                // Random target: different tokens from src
                let tgt: Vec<usize> = (0..seq_len)
                    .map(|_| rng.gen_range(3..config.tgt_vocab))
                    .collect();
                let mut tgt_in = vec![config.bos_id];
                tgt_in.extend_from_slice(&tgt[..seq_len - 1]);
                (tgt_in, tgt)
            };

            samples.push((src, tgt_input, tgt_output));
        }

        samples
    }

    /// Print dataset statistics.
    pub fn print_stats(&self) {
        println!("  SyntheticDataset stats:");
        println!("    Vocab (src/tgt): {}/{}", self.config.src_vocab, self.config.tgt_vocab);
        println!("    Sequence length range: {}-{}", self.config.min_len, self.config.max_len);
        println!("    Train samples: {}", self.train_data.len());
        println!("    Val samples: {}", self.val_data.len());
        println!("    Task type: {}", if self.config.copy_task { "copy" } else { "translation" });

        // Compute average lengths
        let avg_src_len: f64 = self.train_data.iter().map(|(s, _, _)| s.len() as f64).sum::<f64>() / self.train_data.len() as f64;
        let avg_tgt_len: f64 = self.train_data.iter().map(|(_, _, t)| t.len() as f64).sum::<f64>() / self.train_data.len() as f64;
        println!("    Avg src len: {:.1}", avg_src_len);
        println!("    Avg tgt len: {:.1}", avg_tgt_len);
    }
}

/// JSON structure for legal act documents.
#[derive(Debug, Deserialize)]
struct ActDocument {
    sections: Vec<ActSection>,
}

#[derive(Debug, Deserialize)]
struct ActSection {
    #[allow(dead_code)]
    section_number: String,
    section_title: String,
    text: String,
    keywords: Vec<String>,
}

/// Dataset loaded from real-world JSON legal text files.
#[derive(Debug, Clone)]
pub struct LegalDataset {
    pub train_data: Vec<(Vec<usize>, Vec<usize>, Vec<usize>)>,
    pub val_data: Vec<(Vec<usize>, Vec<usize>, Vec<usize>)>,
    #[allow(dead_code)]
    pub tokenizer: crate::tokenizer::BPETokenizer,
    pub config: SyntheticDatasetConfig,
}

impl LegalDataset {
    /// Load legal text from all JSON files in a directory, train a BPE tokenizer,
    /// and produce (src, tgt_input, tgt_output) triples.
    pub fn from_directory(
        dir: &str,
        vocab_size: usize,
        max_len: usize,
        pad_id: usize,
        bos_id: usize,
        eos_id: usize,
        train_ratio: f64,
    ) -> Result<Self, String> {
        let _rng = rand::thread_rng();
        let mut all_texts: Vec<String> = Vec::new();

        let entries = fs::read_dir(dir)
            .map_err(|e| format!("Failed to read directory '{}': {}", dir, e))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("{}", e))?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let content = fs::read_to_string(&path)
                .map_err(|e| format!("Failed to read '{}': {}", path.display(), e))?;
            let doc: ActDocument = serde_json::from_str(&content)
                .map_err(|e| format!("Failed to parse '{}': {}", path.display(), e))?;
            for section in &doc.sections {
                let text = format!(
                    "{} {} {}",
                    section.section_title,
                    section.text,
                    section.keywords.join(" ")
                );
                all_texts.push(text);
            }
        }

        if all_texts.is_empty() {
            return Err("No text found in JSON files".into());
        }

        let all_text_refs: Vec<&str> = all_texts.iter().map(|s| s.as_str()).collect();

        let tokenizer = crate::tokenizer::BPETokenizer::train(
            &all_text_refs,
            vocab_size,
            pad_id,
            bos_id,
            eos_id,
            3,
        );

        let mut samples: Vec<(Vec<usize>, Vec<usize>, Vec<usize>)> = Vec::new();
        for text in all_texts {
            let tokens = tokenizer.encode(&text, max_len);
            if tokens.is_empty() || tokens.len() < 2 {
                continue;
            }
            let mut tgt_in = vec![bos_id];
            tgt_in.extend_from_slice(&tokens[..tokens.len() - 1]);
            let tgt_out = tokens.clone();
            samples.push((tokens, tgt_in, tgt_out));
        }

        if samples.is_empty() {
            return Err("No valid samples generated".into());
        }

        let n_train = ((samples.len() as f64) * train_ratio).round() as usize;
        let n_train = n_train.max(1).min(samples.len() - 1);
        let train_data = samples[..n_train].to_vec();
        let val_data = samples[n_train..].to_vec();

        let config = SyntheticDatasetConfig {
            src_vocab: tokenizer.vocab_size(),
            tgt_vocab: tokenizer.vocab_size(),
            pad_id,
            bos_id,
            eos_id,
            min_len: 2,
            max_len,
            train_samples: train_data.len(),
            val_samples: val_data.len(),
            copy_task: true,
        };

        Ok(Self {
            train_data,
            val_data,
            tokenizer,
            config,
        })
    }

    pub fn print_stats(&self) {
        println!("  LegalDataset stats:");
        println!("    Vocab (src/tgt): {}/{}", self.config.src_vocab, self.config.tgt_vocab);
        println!("    Train samples: {}", self.train_data.len());
        println!("    Val samples:   {}", self.val_data.len());
        let avg_src_len: f64 = self.train_data.iter().map(|(s, _, _)| s.len() as f64).sum::<f64>() / self.train_data.len() as f64;
        let avg_tgt_len: f64 = self.train_data.iter().map(|(_, _, t)| t.len() as f64).sum::<f64>() / self.train_data.len() as f64;
        println!("    Avg src len: {:.1}", avg_src_len);
        println!("    Avg tgt len: {:.1}", avg_tgt_len);
    }
}

/// Pad a batch of sequences to the same length.
///
/// Takes a batch of sequences and pads each to `max_len` with `pad_id`.
/// Returns (padded_tensor, original_lengths).
pub fn pad_batch(sequences: &[Vec<usize>], pad_id: usize) -> (Vec<Vec<usize>>, Vec<usize>) {
    if sequences.is_empty() {
        return (vec![], vec![]);
    }
    let max_len = sequences.iter().map(|s| s.len()).max().unwrap_or(0);
    let lengths: Vec<usize> = sequences.iter().map(|s| s.len()).collect();
    let padded: Vec<Vec<usize>> = sequences
        .iter()
        .map(|s| {
            let mut padded = s.clone();
            padded.resize(max_len, pad_id);
            padded
        })
        .collect();
    (padded, lengths)
}

/// Convert a batch of (src, tgt_input, tgt_output) triples into
/// padded batches for training.
pub fn prepare_train_batch(
    batch: &[(Vec<usize>, Vec<usize>, Vec<usize>)],
    pad_id: usize,
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let src_batch: Vec<Vec<usize>> = batch.iter().map(|(s, _, _)| s.clone()).collect();
    let tgt_in_batch: Vec<Vec<usize>> = batch.iter().map(|(_, t, _)| t.clone()).collect();
    let tgt_out_batch: Vec<Vec<usize>> = batch.iter().map(|(_, _, o)| o.clone()).collect();

    let (src_padded, _) = pad_batch(&src_batch, pad_id);
    let (tgt_in_padded, _) = pad_batch(&tgt_in_batch, pad_id);
    let (tgt_out_padded, _) = pad_batch(&tgt_out_batch, pad_id);

    (src_padded, tgt_in_padded, tgt_out_padded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_generation() {
        let config = SyntheticDatasetConfig {
            src_vocab: 50,
            tgt_vocab: 50,
            pad_id: 0,
            bos_id: 1,
            eos_id: 2,
            min_len: 3,
            max_len: 8,
            train_samples: 100,
            val_samples: 20,
            copy_task: true,
        };
        let dataset = SyntheticDataset::new(config);
        assert_eq!(dataset.train_data.len(), 100);
        assert_eq!(dataset.val_data.len(), 20);

        // Check structure of first sample
        let (src, tgt_in, tgt_out) = &dataset.train_data[0];
        assert!(!src.is_empty());
        assert!(!tgt_in.is_empty());
        assert!(!tgt_out.is_empty());
        assert_eq!(tgt_in[0], 1); // BOS
    }

    #[test]
    fn test_pad_batch() {
        let sequences = vec![vec![1, 2, 3], vec![4, 5], vec![6]];
        let (padded, lengths) = pad_batch(&sequences, 0);
        assert_eq!(padded.len(), 3);
        assert_eq!(padded[0], vec![1, 2, 3]);
        assert_eq!(padded[1], vec![4, 5, 0]);
        assert_eq!(padded[2], vec![6, 0, 0]);
        assert_eq!(lengths, vec![3, 2, 1]);
    }

    #[test]
    fn test_prepare_train_batch() {
        let batch = vec![
            (vec![3, 4, 5], vec![1, 3, 4], vec![3, 4, 5]),
            (vec![6, 7], vec![1, 6], vec![6, 7]),
        ];
        let (src, tgt_in, tgt_out) = prepare_train_batch(&batch, 0);
        assert_eq!(src.len(), 2);
        assert_eq!(tgt_in.len(), 2);
        assert_eq!(tgt_out.len(), 2);
        // Should be padded to max length
        assert_eq!(src[0].len(), 3);
        assert_eq!(src[1].len(), 3);
    }

    #[test]
    fn test_config_default() {
        let config = SyntheticDatasetConfig::default();
        assert_eq!(config.pad_id, 0);
        assert_eq!(config.bos_id, 1);
        assert_eq!(config.eos_id, 2);
        assert!(config.min_len <= config.max_len);
    }
}


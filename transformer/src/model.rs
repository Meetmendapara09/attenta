use ndarray::{s, Array1, Array2};
use rand::Rng;
use rand::{rngs::StdRng, SeedableRng};

use crate::attention::{FeedForward, LayerNorm, Linear, MultiHeadAttention};
use crate::decoder::{Decoder, DecoderLayer};
use crate::encoder::{Encoder, EncoderLayer};
use crate::tensor_ops::{
    causal_mask, dropout, embedding_lookup, make_decoder_mask, make_src_mask, normal_init,
    padding_mask, xavier_init,
};

/// Transformer configuration.
///
/// Paper: Section 5.3 (Table 3 — base model)
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    pub src_vocab: usize,
    pub tgt_vocab: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub n_layers: usize,
    pub max_len: usize,
    pub dropout: f64,
    pub pad_id: usize,
    pub bos_id: usize,
    pub eos_id: usize,
    pub label_smoothing: f64,
    pub warmup_steps: usize,
}

impl Default for TransformerConfig {
    /// Default: base model from the paper.
    fn default() -> Self {
        Self {
            src_vocab: 37000,
            tgt_vocab: 37000,
            d_model: 512,
            n_heads: 8,
            d_ff: 2048,
            n_layers: 6,
            max_len: 512,
            dropout: 0.1,
            pad_id: 0,
            bos_id: 1,
            eos_id: 2,
            label_smoothing: 0.1,
            warmup_steps: 4000,
        }
    }
}

/// Transformer Model.
///
/// The full encoder-decoder Transformer from "Attention Is All You Need".
/// Paper: Section 3
pub struct Transformer {
    pub(crate) config: TransformerConfig,
    pub(crate) src_embeddings: Array2<f64>,
    pub(crate) tgt_embeddings: Array2<f64>,
    pub(crate) positional_encoding: Array2<f64>,
    pub(crate) encoder: Encoder,
    pub(crate) decoder: Decoder,
    /// Pre-softmax linear transformation (shares weights with tgt_embeddings transpose).
    pub(crate) output_projection: Linear,
    pub(crate) scale: f64,
}

impl Transformer {
    /// Build a new Transformer with a deterministic seed (for testing).
    pub fn new_seeded(config: TransformerConfig, seed: u64) -> Self {
        Self::new_with_rng(config, StdRng::seed_from_u64(seed))
    }

    /// Build a new Transformer from config.
    ///
    /// Weight initialization follows Section 5.3:
    /// - Embeddings: normal(0, 1) then scaled by sqrt(d_model)
    /// - Attention weights: Xavier uniform
    /// - Output projection: shared with tgt_embeddings transpose (Section 3.4)
    pub fn new(config: TransformerConfig) -> Self {
        Self::new_with_rng(config, rand::thread_rng())
    }

    /// Build a new Transformer with a specific RNG (for deterministic testing).
    pub fn new_with_rng<R: Rng + rand::RngCore>(config: TransformerConfig, mut rng: R) -> Self {
        let d = config.d_model;
        let n = config.n_heads;
        let df = config.d_ff;
        let nl = config.n_layers;

        // Embeddings scaled by sqrt(d_model) per Section 3.4
        let scale = (d as f64).sqrt();
        let src_embeddings = normal_init(config.src_vocab, d, 1.0, &mut rng) * scale;
        let tgt_embeddings = normal_init(config.tgt_vocab, d, 1.0, &mut rng) * scale;
        let positional_encoding = generate_positional_encoding(config.max_len, d);

        // Build encoder
        let mut enc_layers = Vec::with_capacity(nl);
        for _ in 0..nl {
            enc_layers.push(Self::build_encoder_layer(
                d,
                n,
                df,
                config.dropout,
                &mut rng,
            ));
        }
        let enc = Encoder::new(enc_layers, Self::build_layer_norm(d));

        // Build decoder
        let mut dec_layers = Vec::with_capacity(nl);
        for _ in 0..nl {
            dec_layers.push(Self::build_decoder_layer(
                d,
                n,
                df,
                config.dropout,
                &mut rng,
            ));
        }
        let dec = Decoder::new(dec_layers, Self::build_layer_norm(d));

        // Output projection: shared with tgt embedding transpose (Section 3.4)
        let output_projection = Linear::new(
            tgt_embeddings.t().to_owned(),
            Array1::zeros(config.tgt_vocab),
        );

        Self {
            config,
            src_embeddings,
            tgt_embeddings,
            positional_encoding,
            encoder: enc,
            decoder: dec,
            output_projection,
            scale,
        }
    }

    /// Forward pass for training.
    ///
    /// src: [batch][src_len] token ids
    /// tgt: [batch][tgt_len] token ids (shifted right)
    /// Returns: [batch][tgt_len][tgt_vocab] logits
    pub fn forward(&self, src: &[Vec<usize>], tgt: &[Vec<usize>], train: bool) -> Vec<Array2<f64>> {
        let batch = src.len();
        let mut all_logits = Vec::with_capacity(batch);

        for b in 0..batch {
            // Source padding mask
            let src_pad = padding_mask(&src[b], self.config.pad_id);
            let src_mask = make_src_mask(&src_pad);

            // Embed source + positional encoding
            let src_emb = Self::embed_and_encode(
                &self.src_embeddings,
                &self.positional_encoding,
                &src[b],
                self.scale,
                self.config.dropout,
                train,
            );

            // Encode
            let enc_output = self.encoder.forward(&src_emb, Some(&src_mask), train);

            // Target causal + padding mask
            let tgt_pad = padding_mask(&tgt[b], self.config.pad_id);
            let tgt_causal = causal_mask(tgt[b].len());
            let tgt_mask = make_decoder_mask(&tgt_causal, &tgt_pad);

            // Embed target + positional encoding
            let tgt_emb = Self::embed_and_encode(
                &self.tgt_embeddings,
                &self.positional_encoding,
                &tgt[b],
                self.scale,
                self.config.dropout,
                train,
            );

            // Decode
            let dec_output = self.decoder.forward(
                &tgt_emb,
                &enc_output,
                Some(&src_mask),
                Some(&tgt_mask),
                train,
            );

            // Project to vocabulary logits (weight tying: same matrix as tgt_embeddings)
            let logits = self.output_projection.forward(&dec_output);
            all_logits.push(logits);
        }

        all_logits
    }

    /// Greedy decoding: generate output sequence by always picking the highest-probability token.
    ///
    /// Paper: Section 6.1
    ///
    /// Per 6.1:
    /// - Maximum output length = input_length + 50
    /// - Early termination when possible
    /// - Output does NOT include the initial BOS token
    pub fn greedy_decode(&self, src: &[usize], max_len: usize) -> Vec<usize> {
        let src_mask = make_src_mask(&padding_mask(src, self.config.pad_id));
        let src_emb = Self::embed_and_encode(
            &self.src_embeddings,
            &self.positional_encoding,
            src,
            self.scale,
            0.0,
            false,
        );
        let enc_output = self.encoder.forward(&src_emb, Some(&src_mask), false);

        // Start with BOS token
        let mut output = vec![self.config.bos_id];
        let mut eos_count = 0;
        const EARLY_TERMINATION_EOS_THRESHOLD: usize = 2;

        // Compute max_len based on 6.1: input_length + 50, capped at config.max_len
        let effective_max_len = max_len.min(src.len() + 50).min(self.config.max_len);

        for _ in 0..effective_max_len {
            let tgt_emb = Self::embed_and_encode(
                &self.tgt_embeddings,
                &self.positional_encoding,
                &output,
                self.scale,
                0.0,
                false,
            );
            let tgt_causal = causal_mask(output.len());
            let tgt_mask = tgt_causal.mapv(|x| if x == 0.0 { f64::NEG_INFINITY } else { 0.0 });

            let dec_output = self.decoder.forward(
                &tgt_emb,
                &enc_output,
                Some(&src_mask),
                Some(&tgt_mask),
                false,
            );

            let logits = self.output_projection.forward(&dec_output);
            let last_logits = logits.row(logits.nrows() - 1);

            // Greedy: pick argmax
            let next_token = last_logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();

            if next_token == self.config.eos_id {
                eos_count += 1;
                // Early termination: if we see EOS back-to-back, stop
                if eos_count >= EARLY_TERMINATION_EOS_THRESHOLD {
                    break;
                }
                continue;
            }
            eos_count = 0;
            output.push(next_token);
        }

        // Remove BOS token from output (convenience API returns only generated tokens)
        output.remove(0);
        output
    }

    /// Beam search decoding.
    ///
    /// Paper: Section 6.1: "beam size of 4 and length penalty alpha = 0.6"
    ///
    /// Per 6.1:
    /// - Maximum output length = input_length + 50
    /// - Beam size of 4 and length penalty α = 0.6
    /// - Early termination when possible
    /// - Output does NOT include the initial BOS token
    pub fn beam_search(
        &self,
        src: &[usize],
        max_len: usize,
        beam_size: usize,
        alpha: f64,
    ) -> Vec<usize> {
        let src_mask = make_src_mask(&padding_mask(src, self.config.pad_id));
        let src_emb = Self::embed_and_encode(
            &self.src_embeddings,
            &self.positional_encoding,
            src,
            self.scale,
            0.0,
            false,
        );
        let enc_output = self.encoder.forward(&src_emb, Some(&src_mask), false);

        // Compute max_len based on 6.1: input_length + 50, capped at config.max_len
        let effective_max_len = max_len.min(src.len() + 50).min(self.config.max_len);

        // Each beam: (score, token_sequence) — score is cumulative log-probability
        let mut beams: Vec<(f64, Vec<usize>)> = vec![(0.0, vec![self.config.bos_id])];
        let mut completed: Vec<(f64, Vec<usize>)> = Vec::new();

        for _ in 0..effective_max_len {
            let mut candidates: Vec<(f64, Vec<usize>)> = Vec::new();

            // Only run forward pass on active beams (not ended with EOS)
            let active: Vec<_> = beams
                .iter()
                .filter(|(_, seq)| *seq.last().unwrap() != self.config.eos_id)
                .collect();

            if active.is_empty() {
                break;
            }

            for (score, seq) in active {
                let tgt_emb = Self::embed_and_encode(
                    &self.tgt_embeddings,
                    &self.positional_encoding,
                    seq,
                    self.scale,
                    0.0,
                    false,
                );
                let tgt_causal = causal_mask(seq.len());
                let tgt_mask = tgt_causal.mapv(|x| if x == 0.0 { f64::NEG_INFINITY } else { 0.0 });

                let dec_output = self.decoder.forward(
                    &tgt_emb,
                    &enc_output,
                    Some(&src_mask),
                    Some(&tgt_mask),
                    false,
                );
                let logits = self.output_projection.forward(&dec_output);
                let last_logits = logits.row(logits.nrows() - 1);

                // Log-softmax
                let max_val = last_logits.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let sum_exp: f64 = last_logits.mapv(|x| (x - max_val).exp()).sum();

                // Pre-allocate the base sequence
                let base = seq.clone();
                let base_score = score;

                for (tok_idx, &logit) in last_logits.iter().enumerate() {
                    if tok_idx == self.config.pad_id {
                        continue;
                    }
                    let log_prob = logit - max_val - sum_exp.ln();
                    let mut new_seq = base.clone();
                    new_seq.push(tok_idx);

                    // If EOS is generated, move to completed with length penalty
                    if tok_idx == self.config.eos_id {
                        let len = new_seq.len() as f64;
                        let adjusted_score = (base_score + log_prob) / len.powf(alpha);
                        completed.push((adjusted_score, new_seq));
                    } else {
                        candidates.push((base_score + log_prob, new_seq));
                    }
                }
            }

            if candidates.is_empty() {
                break;
            }

            // Keep top-k beams
            candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            beams = candidates.into_iter().take(beam_size).collect();

            // Early termination: if top completed beam is better than any active beam
            if !completed.is_empty() {
                completed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                if completed[0].0 >= beams[0].0 {
                    break;
                }
            }
        }

        // Add remaining active beams as candidates with length penalty
        for (score, seq) in beams {
            if seq.len() > 1 {
                let len = seq.len() as f64;
                completed.push((score / len.powf(alpha), seq));
            }
        }

        // Return best completed sequence (without BOS token)
        completed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        completed
            .into_iter()
            .next()
            .map(|(_, mut seq)| {
                seq.remove(0); // Remove BOS
                seq
            })
            .unwrap_or_else(Vec::new)
    }

    /// Total number of parameters.
    pub fn num_parameters(&self) -> usize {
        let emb = self.src_embeddings.len() + self.tgt_embeddings.len();
        let enc = self.encoder.num_params();
        let dec = self.decoder.num_params();
        let proj = self.output_projection.num_params();
        emb + enc + dec + proj
    }

    /// Initialize an Adam optimizer state for this model.
    ///
    /// Paper: Section 5.3 — "We used the Adam optimizer with beta_1 = 0.9, beta_2 = 0.98, eps = 1e-9."
    pub fn init_adam(&self) -> crate::train::AdamState {
        let mut adam = crate::train::AdamState::new(
            self.config.d_model,
            self.config.warmup_steps,
            self.config.n_layers,
        );
        for layer in &self.decoder.layers {
            adam.init_dec_layer(layer);
        }
        for layer in &self.encoder.layers {
            adam.init_enc_layer(layer);
        }
        adam.init_param("out_proj_w", (self.config.d_model, self.config.tgt_vocab));
        adam.init_param("tgt_emb", (self.config.tgt_vocab, self.config.d_model));
        adam.init_param("src_emb", (self.config.src_vocab, self.config.d_model));
        adam
    }

    /// Access the model config.
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// Convenience: translate src token ids using greedy decoding.
    ///
    /// Per 6.1: max output length = input_length + 50
    pub fn translate(&self, src: &[usize]) -> Vec<usize> {
        let max_len = src.len() + 50;
        self.greedy_decode(src, max_len)
    }

    /// Convenience: translate src token ids using beam search.
    ///
    /// Per 6.1: beam size of 4, length penalty α = 0.6, max output length = input_length + 50
    pub fn translate_beam(&self, src: &[usize]) -> Vec<usize> {
        let max_len = src.len() + 50;
        self.beam_search(src, max_len, 4, 0.6)
    }

    /// Embed tokens, add positional encoding, apply dropout.
    fn embed_and_encode(
        embeddings: &Array2<f64>,
        pe: &Array2<f64>,
        tokens: &[usize],
        scale: f64,
        dropout_rate: f64,
        train: bool,
    ) -> Array2<f64> {
        let emb = embedding_lookup(embeddings, tokens) * scale;
        let seq_len = tokens.len();
        let pe_slice = pe.slice(s![..seq_len, ..]);
        let emb = &emb + &pe_slice;
        dropout(&emb, dropout_rate, train)
    }

    fn build_encoder_layer(
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        dropout: f64,
        rng: &mut impl rand::Rng,
    ) -> EncoderLayer {
        let w_q = Linear::new(xavier_init(d_model, d_model, rng), Array1::zeros(d_model));
        let w_k = Linear::new(xavier_init(d_model, d_model, rng), Array1::zeros(d_model));
        let w_v = Linear::new(xavier_init(d_model, d_model, rng), Array1::zeros(d_model));
        let w_o = Linear::new(xavier_init(d_model, d_model, rng), Array1::zeros(d_model));
        let self_attn = MultiHeadAttention::new(d_model, n_heads, w_q, w_k, w_v, w_o, dropout);

        let w1 = Linear::new(xavier_init(d_model, d_ff, rng), Array1::zeros(d_ff));
        let w2 = Linear::new(xavier_init(d_ff, d_model, rng), Array1::zeros(d_model));
        let feed_forward = FeedForward::new(w1, w2, dropout);

        EncoderLayer::new(
            self_attn,
            feed_forward,
            Self::build_layer_norm(d_model),
            Self::build_layer_norm(d_model),
            dropout,
        )
    }

    fn build_decoder_layer(
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        dropout: f64,
        rng: &mut impl rand::Rng,
    ) -> DecoderLayer {
        let w_q = Linear::new(xavier_init(d_model, d_model, rng), Array1::zeros(d_model));
        let w_k = Linear::new(xavier_init(d_model, d_model, rng), Array1::zeros(d_model));
        let w_v = Linear::new(xavier_init(d_model, d_model, rng), Array1::zeros(d_model));
        let w_o = Linear::new(xavier_init(d_model, d_model, rng), Array1::zeros(d_model));
        let self_attn = MultiHeadAttention::new(d_model, n_heads, w_q, w_k, w_v, w_o, dropout);

        let w_q = Linear::new(xavier_init(d_model, d_model, rng), Array1::zeros(d_model));
        let w_k = Linear::new(xavier_init(d_model, d_model, rng), Array1::zeros(d_model));
        let w_v = Linear::new(xavier_init(d_model, d_model, rng), Array1::zeros(d_model));
        let w_o = Linear::new(xavier_init(d_model, d_model, rng), Array1::zeros(d_model));
        let cross_attn = MultiHeadAttention::new(d_model, n_heads, w_q, w_k, w_v, w_o, dropout);

        let w1 = Linear::new(xavier_init(d_model, d_ff, rng), Array1::zeros(d_ff));
        let w2 = Linear::new(xavier_init(d_ff, d_model, rng), Array1::zeros(d_model));
        let feed_forward = FeedForward::new(w1, w2, dropout);

        DecoderLayer::new(
            self_attn,
            cross_attn,
            feed_forward,
            Self::build_layer_norm(d_model),
            Self::build_layer_norm(d_model),
            Self::build_layer_norm(d_model),
            dropout,
        )
    }

    fn build_layer_norm(d_model: usize) -> LayerNorm {
        LayerNorm::new(Array1::ones(d_model), Array1::zeros(d_model))
    }
}

/// Positional Encoding using sine and cosine functions of different frequencies.
///
/// PE_(pos, 2i)   = sin(pos / 10000^(2i/d_model))
/// PE_(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
///
/// Paper: Section 3.5
pub fn generate_positional_encoding(max_len: usize, d_model: usize) -> Array2<f64> {
    let mut pe = Array2::zeros((max_len, d_model));
    let inv_freq = Array1::from_shape_fn(d_model / 2, |i| {
        1.0 / 10000f64.powf(2.0 * i as f64 / d_model as f64)
    });

    for pos in 0..max_len {
        let angles = &inv_freq * pos as f64;
        for (i, &angle) in angles.iter().enumerate() {
            pe[[pos, 2 * i]] = angle.sin();
            if 2 * i + 1 < d_model {
                pe[[pos, 2 * i + 1]] = angle.cos();
            }
        }
    }
    pe
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> TransformerConfig {
        TransformerConfig {
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
            warmup_steps: 100,
        }
    }

    #[test]
    fn test_transformer_forward_shape() {
        let t = Transformer::new(small_config());
        let src = vec![vec![3, 5, 7, 8, 9], vec![4, 6, 8, 10, 11]];
        let tgt = vec![vec![3, 5, 7, 8], vec![4, 6, 8, 10]];
        let logits = t.forward(&src, &tgt, false);
        assert_eq!(logits.len(), 2);
        assert_eq!(logits[0].shape(), &[4, 30]);
        assert_eq!(logits[1].shape(), &[4, 30]);
    }

    #[test]
    fn test_transformer_forward_train_vs_eval() {
        let t = Transformer::new(small_config());
        let src = vec![vec![3, 5, 7, 8]];
        let tgt = vec![vec![3, 5, 7, 8]];
        // With dropout=0.0, train and eval should give the same result
        let out_train = t.forward(&src, &tgt, true);
        let out_eval = t.forward(&src, &tgt, false);
        for i in 0..4 {
            for j in 0..30 {
                assert!(
                    (out_train[0][[i, j]] - out_eval[0][[i, j]]).abs() < 1e-10,
                    "dropout=0 should produce identical results"
                );
            }
        }
    }

    #[test]
    fn test_greedy_decode_produces_tokens() {
        let t = Transformer::new_seeded(small_config(), 42);
        let src = vec![3, 5, 7, 8, 9];
        let decoded = t.greedy_decode(&src, 20);
        assert!(!decoded.is_empty(), "greedy decode should produce tokens");
        assert!(
            decoded.len() <= 21,
            "decoded should be at most max_len+1 (BOS + 20), got {}",
            decoded.len()
        );
    }

    #[test]
    fn test_beam_search_produces_tokens() {
        let t = Transformer::new_seeded(small_config(), 42);
        let src = vec![3, 5, 7, 8, 9];
        let decoded = t.beam_search(&src, 20, 2, 0.6);
        assert!(!decoded.is_empty(), "beam search should produce tokens");
    }

    #[test]
    fn test_translate_convenience() {
        let t = Transformer::new_seeded(small_config(), 42);
        let src = vec![3, 5, 7];
        let decoded = t.translate(&src);
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_num_parameters_positive() {
        let t = Transformer::new(small_config());
        let n = t.num_parameters();
        assert!(n > 0, "model should have parameters, got {}", n);
    }

    #[test]
    fn test_config_accessor() {
        let cfg = small_config();
        let t = Transformer::new(cfg);
        assert_eq!(t.config().d_model, 16);
        assert_eq!(t.config().n_heads, 4);
        assert_eq!(t.config().n_layers, 1);
    }

    #[test]
    fn test_positional_encoding_shape() {
        let pe = generate_positional_encoding(32, 16);
        assert_eq!(pe.shape(), &[32, 16]);
    }

    #[test]
    fn test_positional_encoding_values_in_range() {
        let pe = generate_positional_encoding(32, 16);
        for v in pe.iter() {
            assert!(*v >= -1.0 && *v <= 1.0, "PE values should be in [-1, 1]");
        }
    }

    #[test]
    fn test_positional_encoding_zero_position() {
        let pe = generate_positional_encoding(32, 16);
        // PE(0, 2i) = sin(0) = 0
        for i in 0..8 {
            assert!((pe[[0, 2 * i]]).abs() < 1e-10);
        }
        // PE(0, 2i+1) = cos(0) = 1
        for i in 0..8 {
            assert!((pe[[0, 2 * i + 1]] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_weight_tying_dimensions() {
        let cfg = small_config();
        let d_model = cfg.d_model;
        let tgt_vocab = cfg.tgt_vocab;
        let t = Transformer::new(cfg);
        // output_projection.w is [d_model, tgt_vocab] (transposed from embedding matrix)
        assert_eq!(t.output_projection.w.nrows(), d_model);
        assert_eq!(t.output_projection.w.ncols(), tgt_vocab);
    }
}

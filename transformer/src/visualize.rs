/// Attention visualization utilities for Figures 3-5 from the paper.
///
/// Extracts attention weights from encoder/decoder layers for visualization.
/// Outputs JSON-formatted data that can be rendered as attention heatmaps.
use ndarray::{s, Array2};

use crate::decoder::Decoder;
use crate::encoder::Encoder;
use crate::model::Transformer;
use crate::tensor_ops::{
    causal_mask, embedding_lookup, make_decoder_mask, make_src_mask, padding_mask,
};

/// Attention weights from a single layer/head combination.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AttentionWeights {
    /// Layer name (e.g., "encoder_layer_0", "decoder_layer_3_self", "decoder_layer_3_cross")
    pub layer: String,
    /// Head index (0..n_heads-1)
    pub head: usize,
    /// Source tokens (for labeling axes)
    pub source_tokens: Vec<usize>,
    /// Target tokens (for labeling axes, or same as source for self-attention)
    pub target_tokens: Vec<usize>,
    /// Weight matrix [target_len, source_len] — attention probabilities
    pub weights: Vec<Vec<f64>>,
}

/// Extract all attention weights from the encoder during forward pass.
///
/// Returns attention weights for all encoder self-attention layers and heads.
pub fn extract_encoder_attention(
    encoder: &Encoder,
    src_tokens: &[usize],
    src_emb: &Array2<f64>,
    src_mask: &Array2<f64>,
) -> Vec<AttentionWeights> {
    let mut results = Vec::new();
    let n_heads = if encoder.layers.is_empty() {
        0
    } else {
        encoder.layers[0].self_attn.n_heads
    };
    let d_k = if encoder.layers.is_empty() {
        1
    } else {
        encoder.layers[0].self_attn.d_k
    };

    let mut x = src_emb.clone();

    for (li, layer) in encoder.layers.iter().enumerate() {
        let q_proj = layer.self_attn.w_q.forward(&x);
        let k_proj = layer.self_attn.w_k.forward(&x);
        let _v_proj = layer.self_attn.w_v.forward(&x);

        for h in 0..n_heads {
            let start = h * d_k;
            let end = start + d_k;

            let q_h = q_proj.slice(s![.., start..end]).to_owned();
            let k_h = k_proj.slice(s![.., start..end]).to_owned();

            // Compute attention scores: Q @ K^T / sqrt(d_k)
            let scores = {
                let qk = q_h.dot(&k_h.t());
                &qk / (d_k as f64).sqrt()
            };

            // Apply mask
            let masked_scores = &scores + src_mask;

            // Softmax to get weights
            let weights = crate::tensor_ops::softmax(&masked_scores);

            // Convert to Vec<Vec<f64>>
            let rows = weights.nrows();
            let cols = weights.ncols();
            let mut weight_vec = Vec::with_capacity(rows);
            for i in 0..rows {
                let mut row = Vec::with_capacity(cols);
                for j in 0..cols {
                    row.push(weights[[i, j]]);
                }
                weight_vec.push(row);
            }

            results.push(AttentionWeights {
                layer: format!("encoder_layer_{}", li),
                head: h,
                source_tokens: src_tokens.to_vec(),
                target_tokens: src_tokens.to_vec(),
                weights: weight_vec,
            });
        }

        // Forward through layer for next iteration
        x = layer.forward(&x, Some(src_mask), false);
    }

    results
}

/// Extract all attention weights from the decoder during forward pass.
///
/// Returns both self-attention and cross-attention weights for all decoder layers and heads.
pub fn extract_decoder_attention(
    decoder: &Decoder,
    tgt_tokens: &[usize],
    src_tokens: &[usize],
    enc_output: &Array2<f64>,
    tgt_emb: &Array2<f64>,
    src_mask: &Array2<f64>,
    tgt_mask: &Array2<f64>,
) -> Vec<AttentionWeights> {
    let mut results = Vec::new();

    if decoder.layers.is_empty() {
        return results;
    }

    let n_heads = decoder.layers[0].self_attn.n_heads;
    let d_k = decoder.layers[0].self_attn.d_k;

    let mut x = tgt_emb.clone();

    for (li, layer) in decoder.layers.iter().enumerate() {
        // Self-attention
        let sa_q = layer.self_attn.w_q.forward(&x);
        let sa_k = layer.self_attn.w_k.forward(&x);

        for h in 0..n_heads {
            let start = h * d_k;
            let end = start + d_k;

            let q_h = sa_q.slice(s![.., start..end]).to_owned();
            let k_h = sa_k.slice(s![.., start..end]).to_owned();

            let scores = {
                let qk = q_h.dot(&k_h.t());
                &qk / (d_k as f64).sqrt()
            };

            let masked_scores = &scores + tgt_mask;
            let weights = crate::tensor_ops::softmax(&masked_scores);

            let rows = weights.nrows();
            let cols = weights.ncols();
            let mut weight_vec = Vec::with_capacity(rows);
            for i in 0..rows {
                let mut row = Vec::with_capacity(cols);
                for j in 0..cols {
                    row.push(weights[[i, j]]);
                }
                weight_vec.push(row);
            }

            results.push(AttentionWeights {
                layer: format!("decoder_layer_{}_self", li),
                head: h,
                source_tokens: tgt_tokens.to_vec(),
                target_tokens: tgt_tokens.to_vec(),
                weights: weight_vec,
            });
        }

        // Self-attention + residual + norm
        let sa_out = layer.self_attn.forward(&x, &x, &x, Some(tgt_mask), false);
        let x_norm1 = layer.norm_1.forward(&(&x + &sa_out));

        // Cross-attention
        let ca_q = layer.cross_attn.w_q.forward(&x_norm1);
        let ca_k = layer.cross_attn.w_k.forward(enc_output);

        for h in 0..n_heads {
            let start = h * d_k;
            let end = start + d_k;

            let q_h = ca_q.slice(s![.., start..end]).to_owned();
            let k_h = ca_k.slice(s![.., start..end]).to_owned();

            let scores = {
                let qk = q_h.dot(&k_h.t());
                &qk / (d_k as f64).sqrt()
            };

            let masked_scores = &scores + src_mask;
            let weights = crate::tensor_ops::softmax(&masked_scores);

            let rows = weights.nrows();
            let cols = weights.ncols();
            let mut weight_vec = Vec::with_capacity(rows);
            for i in 0..rows {
                let mut row = Vec::with_capacity(cols);
                for j in 0..cols {
                    row.push(weights[[i, j]]);
                }
                weight_vec.push(row);
            }

            results.push(AttentionWeights {
                layer: format!("decoder_layer_{}_cross", li),
                head: h,
                source_tokens: src_tokens.to_vec(),
                target_tokens: tgt_tokens.to_vec(),
                weights: weight_vec,
            });
        }

        // Continue forward pass
        let ca_out =
            layer
                .cross_attn
                .forward(&x_norm1, enc_output, enc_output, Some(src_mask), false);
        let x_norm2 = layer.norm_2.forward(&(&x_norm1 + &ca_out));
        let ffn_out = layer.feed_forward.forward(&x_norm2, false);
        x = layer.norm_3.forward(&(&x_norm2 + &ffn_out));
    }

    results
}

/// Extract all attention weights from the full Transformer.
///
/// Returns a vector of all attention weights across all layers and heads.
pub fn extract_all_attention(
    transformer: &Transformer,
    src_tokens: &[usize],
    tgt_tokens: &[usize],
) -> Vec<AttentionWeights> {
    let config = &transformer.config;
    let scale = transformer.scale;

    // Encoder
    let src_emb = embedding_lookup(&transformer.src_embeddings, src_tokens) * scale;
    let pe_slice = transformer
        .positional_encoding
        .slice(s![..src_tokens.len(), ..])
        .to_owned();
    let src_emb = &src_emb + &pe_slice;

    let src_pad = padding_mask(src_tokens, config.pad_id);
    let src_mask = make_src_mask(&src_pad);

    let mut results = Vec::new();

    // Extract encoder attention
    results.extend(extract_encoder_attention(
        &transformer.encoder,
        src_tokens,
        &src_emb,
        &src_mask,
    ));

    // Get encoder output
    let enc_output = transformer
        .encoder
        .forward(&src_emb, Some(&src_mask), false);

    // Decoder
    let tgt_emb = embedding_lookup(&transformer.tgt_embeddings, tgt_tokens) * scale;
    let pe_slice = transformer
        .positional_encoding
        .slice(s![..tgt_tokens.len(), ..])
        .to_owned();
    let tgt_emb = &tgt_emb + &pe_slice;

    let tgt_pad = padding_mask(tgt_tokens, config.pad_id);
    let tgt_causal = causal_mask(tgt_tokens.len());
    let tgt_mask = make_decoder_mask(&tgt_causal, &tgt_pad);

    // Extract decoder attention
    results.extend(extract_decoder_attention(
        &transformer.decoder,
        tgt_tokens,
        src_tokens,
        &enc_output,
        &tgt_emb,
        &src_mask,
        &tgt_mask,
    ));

    results
}

/// Save attention weights to a JSON file for visualization.
pub fn save_attention_viz(results: &[AttentionWeights], path: &str) -> Result<(), String> {
    let json =
        serde_json::to_string_pretty(results).map_err(|e| format!("Serialization error: {}", e))?;
    std::fs::write(path, json).map_err(|e| format!("Write error: {}", e))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Transformer, TransformerConfig};

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
    fn test_extract_encoder_attention() {
        let config = small_config();
        let transformer = Transformer::new(config.clone());
        let src = vec![3, 5, 7, 8];

        let src_emb =
            embedding_lookup(&transformer.src_embeddings, &src) * (config.d_model as f64).sqrt();
        let pe_slice = transformer
            .positional_encoding
            .slice(s![..src.len(), ..])
            .to_owned();
        let src_emb = &src_emb + &pe_slice;

        let src_pad = padding_mask(&src, config.pad_id);
        let src_mask = make_src_mask(&src_pad);

        let attn = extract_encoder_attention(&transformer.encoder, &src, &src_emb, &src_mask);
        assert!(!attn.is_empty(), "Should extract encoder attention");
        assert_eq!(attn.len(), config.n_heads * config.n_layers);

        // Check weights sum to ~1.0 per row
        for a in &attn {
            for row in &a.weights {
                let sum: f64 = row.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-6,
                    "Row sum should be ~1.0, got {}",
                    sum
                );
            }
        }
    }

    #[test]
    fn test_extract_all_attention() {
        let config = small_config();
        let transformer = Transformer::new(config.clone());
        let src = vec![3, 5, 7, 8];
        let tgt = vec![1, 3, 5, 7]; // BOS + tokens

        let results = extract_all_attention(&transformer, &src, &tgt);
        assert!(!results.is_empty(), "Should extract attention");

        // Should have encoder + decoder self + decoder cross
        let expected_enc = config.n_heads * config.n_layers;
        let expected_dec_self = config.n_heads * config.n_layers;
        let expected_dec_cross = config.n_heads * config.n_layers;
        assert_eq!(
            results.len(),
            expected_enc + expected_dec_self + expected_dec_cross
        );
    }
}

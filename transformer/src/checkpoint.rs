use std::fs;

use ndarray::{Array1, Array2};

use crate::decoder::DecoderLayer;
use crate::encoder::EncoderLayer;
use crate::model::{Transformer, TransformerConfig};

/// Save model weights and config to a JSON file.
///
/// Format: JSON with all weight matrices serialized as flat arrays.
pub fn save_checkpoint(transformer: &Transformer, path: &str) -> Result<(), String> {
    let data = serialize_model(transformer);
    let json =
        serde_json::to_string_pretty(&data).map_err(|e| format!("Serialization error: {}", e))?;
    fs::write(path, json).map_err(|e| format!("Write error: {}", e))?;
    Ok(())
}

/// Load model weights from a JSON checkpoint file.
///
/// Returns a new Transformer with the loaded weights.
pub fn load_checkpoint(path: &str, config: &TransformerConfig) -> Result<Transformer, String> {
    let json = fs::read_to_string(path).map_err(|e| format!("Read error: {}", e))?;
    let data: serde_json::Value =
        serde_json::from_str(&json).map_err(|e| format!("Parse error: {}", e))?;
    let mut transformer = Transformer::new(config.clone());
    deserialize_model(&data, &mut transformer)?;
    Ok(transformer)
}

/// Average multiple checkpoints and save the result.
///
/// Paper 6.1: "For the base models, we used a single model obtained by
/// averaging the last 5 checkpoints... For the big models, we averaged
/// the last 20 checkpoints."
pub fn average_checkpoints(
    paths: &[String],
    config: &TransformerConfig,
    output_path: &str,
) -> Result<Transformer, String> {
    if paths.is_empty() {
        return Err("No checkpoint paths provided".to_string());
    }

    // Load all checkpoints
    let models: Vec<Transformer> = paths
        .iter()
        .map(|p| load_checkpoint(p, config))
        .collect::<Result<Vec<_>, _>>()?;

    // Start with the first model
    let mut averaged = Transformer::new(config.clone());

    // Sum all weights
    // Embeddings
    for m in &models {
        averaged.src_embeddings = &averaged.src_embeddings + &m.src_embeddings;
        averaged.tgt_embeddings = &averaged.tgt_embeddings + &m.tgt_embeddings;
    }

    // Encoder layers
    for li in 0..config.n_layers {
        for m in &models {
            let l = &m.encoder.layers[li];
            let al = &mut averaged.encoder.layers[li];
            al.self_attn.w_q.w = &al.self_attn.w_q.w + &l.self_attn.w_q.w;
            al.self_attn.w_k.w = &al.self_attn.w_k.w + &l.self_attn.w_k.w;
            al.self_attn.w_v.w = &al.self_attn.w_v.w + &l.self_attn.w_v.w;
            al.self_attn.w_o.w = &al.self_attn.w_o.w + &l.self_attn.w_o.w;
            al.self_attn.w_o.b = &al.self_attn.w_o.b + &l.self_attn.w_o.b;
            al.feed_forward.w_1.w = &al.feed_forward.w_1.w + &l.feed_forward.w_1.w;
            al.feed_forward.w_1.b = &al.feed_forward.w_1.b + &l.feed_forward.w_1.b;
            al.feed_forward.w_2.w = &al.feed_forward.w_2.w + &l.feed_forward.w_2.w;
            al.feed_forward.w_2.b = &al.feed_forward.w_2.b + &l.feed_forward.w_2.b;
            al.norm_1.w = &al.norm_1.w + &l.norm_1.w;
            al.norm_1.b = &al.norm_1.b + &l.norm_1.b;
            al.norm_2.w = &al.norm_2.w + &l.norm_2.w;
            al.norm_2.b = &al.norm_2.b + &l.norm_2.b;
        }
    }

    // Decoder layers
    for li in 0..config.n_layers {
        for m in &models {
            let l = &m.decoder.layers[li];
            let al = &mut averaged.decoder.layers[li];
            al.self_attn.w_q.w = &al.self_attn.w_q.w + &l.self_attn.w_q.w;
            al.self_attn.w_k.w = &al.self_attn.w_k.w + &l.self_attn.w_k.w;
            al.self_attn.w_v.w = &al.self_attn.w_v.w + &l.self_attn.w_v.w;
            al.self_attn.w_o.w = &al.self_attn.w_o.w + &l.self_attn.w_o.w;
            al.self_attn.w_o.b = &al.self_attn.w_o.b + &l.self_attn.w_o.b;
            al.cross_attn.w_q.w = &al.cross_attn.w_q.w + &l.cross_attn.w_q.w;
            al.cross_attn.w_k.w = &al.cross_attn.w_k.w + &l.cross_attn.w_k.w;
            al.cross_attn.w_v.w = &al.cross_attn.w_v.w + &l.cross_attn.w_v.w;
            al.cross_attn.w_o.w = &al.cross_attn.w_o.w + &l.cross_attn.w_o.w;
            al.cross_attn.w_o.b = &al.cross_attn.w_o.b + &l.cross_attn.w_o.b;
            al.feed_forward.w_1.w = &al.feed_forward.w_1.w + &l.feed_forward.w_1.w;
            al.feed_forward.w_1.b = &al.feed_forward.w_1.b + &l.feed_forward.w_1.b;
            al.feed_forward.w_2.w = &al.feed_forward.w_2.w + &l.feed_forward.w_2.w;
            al.feed_forward.w_2.b = &al.feed_forward.w_2.b + &l.feed_forward.w_2.b;
            al.norm_1.w = &al.norm_1.w + &l.norm_1.w;
            al.norm_1.b = &al.norm_1.b + &l.norm_1.b;
            al.norm_2.w = &al.norm_2.w + &l.norm_2.w;
            al.norm_2.b = &al.norm_2.b + &l.norm_2.b;
            al.norm_3.w = &al.norm_3.w + &l.norm_3.w;
            al.norm_3.b = &al.norm_3.b + &l.norm_3.b;
        }
    }

    // Encoder/Decoder final norms
    for m in &models {
        averaged.encoder.norm.w = &averaged.encoder.norm.w + &m.encoder.norm.w;
        averaged.encoder.norm.b = &averaged.encoder.norm.b + &m.encoder.norm.b;
        averaged.decoder.norm.w = &averaged.decoder.norm.w + &m.decoder.norm.w;
        averaged.decoder.norm.b = &averaged.decoder.norm.b + &m.decoder.norm.b;
    }

    // Output projection bias
    for m in &models {
        averaged.output_projection.b = &averaged.output_projection.b + &m.output_projection.b;
    }

    // Divide by number of models
    let n = models.len() as f64;
    averaged.src_embeddings = &averaged.src_embeddings / n;
    averaged.tgt_embeddings = &averaged.tgt_embeddings / n;

    for li in 0..config.n_layers {
        let al = &mut averaged.encoder.layers[li];
        al.self_attn.w_q.w = &al.self_attn.w_q.w / n;
        al.self_attn.w_k.w = &al.self_attn.w_k.w / n;
        al.self_attn.w_v.w = &al.self_attn.w_v.w / n;
        al.self_attn.w_o.w = &al.self_attn.w_o.w / n;
        al.self_attn.w_o.b = &al.self_attn.w_o.b / n;
        al.feed_forward.w_1.w = &al.feed_forward.w_1.w / n;
        al.feed_forward.w_1.b = &al.feed_forward.w_1.b / n;
        al.feed_forward.w_2.w = &al.feed_forward.w_2.w / n;
        al.feed_forward.w_2.b = &al.feed_forward.w_2.b / n;
        al.norm_1.w = &al.norm_1.w / n;
        al.norm_1.b = &al.norm_1.b / n;
        al.norm_2.w = &al.norm_2.w / n;
        al.norm_2.b = &al.norm_2.b / n;

        let al = &mut averaged.decoder.layers[li];
        al.self_attn.w_q.w = &al.self_attn.w_q.w / n;
        al.self_attn.w_k.w = &al.self_attn.w_k.w / n;
        al.self_attn.w_v.w = &al.self_attn.w_v.w / n;
        al.self_attn.w_o.w = &al.self_attn.w_o.w / n;
        al.self_attn.w_o.b = &al.self_attn.w_o.b / n;
        al.cross_attn.w_q.w = &al.cross_attn.w_q.w / n;
        al.cross_attn.w_k.w = &al.cross_attn.w_k.w / n;
        al.cross_attn.w_v.w = &al.cross_attn.w_v.w / n;
        al.cross_attn.w_o.w = &al.cross_attn.w_o.w / n;
        al.cross_attn.w_o.b = &al.cross_attn.w_o.b / n;
        al.feed_forward.w_1.w = &al.feed_forward.w_1.w / n;
        al.feed_forward.w_1.b = &al.feed_forward.w_1.b / n;
        al.feed_forward.w_2.w = &al.feed_forward.w_2.w / n;
        al.feed_forward.w_2.b = &al.feed_forward.w_2.b / n;
        al.norm_1.w = &al.norm_1.w / n;
        al.norm_1.b = &al.norm_1.b / n;
        al.norm_2.w = &al.norm_2.w / n;
        al.norm_2.b = &al.norm_2.b / n;
        al.norm_3.w = &al.norm_3.w / n;
        al.norm_3.b = &al.norm_3.b / n;
    }

    averaged.encoder.norm.w = &averaged.encoder.norm.w / n;
    averaged.encoder.norm.b = &averaged.encoder.norm.b / n;
    averaged.decoder.norm.w = &averaged.decoder.norm.w / n;
    averaged.decoder.norm.b = &averaged.decoder.norm.b / n;
    averaged.output_projection.b = &averaged.output_projection.b / n;

    // Weight tying: sync output projection with tgt_embeddings
    averaged.output_projection.w = averaged.tgt_embeddings.t().to_owned();

    // Save averaged checkpoint
    save_checkpoint(&averaged, output_path)?;
    Ok(averaged)
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

fn serialize_model(transformer: &Transformer) -> serde_json::Value {
    let config = &transformer.config;
    let mut map = serde_json::Map::new();

    // Config
    map.insert("config".to_string(), serialize_config(config));

    // Embeddings
    map.insert(
        "src_embeddings".to_string(),
        array2_to_json(&transformer.src_embeddings),
    );
    map.insert(
        "tgt_embeddings".to_string(),
        array2_to_json(&transformer.tgt_embeddings),
    );

    // Encoder layers
    let enc_layers: Vec<serde_json::Value> = transformer
        .encoder
        .layers
        .iter()
        .map(serialize_encoder_layer)
        .collect();
    map.insert(
        "encoder_layers".to_string(),
        serde_json::Value::Array(enc_layers),
    );

    // Encoder norm
    map.insert(
        "encoder_norm_w".to_string(),
        array1_to_json(&transformer.encoder.norm.w),
    );
    map.insert(
        "encoder_norm_b".to_string(),
        array1_to_json(&transformer.encoder.norm.b),
    );

    // Decoder layers
    let dec_layers: Vec<serde_json::Value> = transformer
        .decoder
        .layers
        .iter()
        .map(serialize_decoder_layer)
        .collect();
    map.insert(
        "decoder_layers".to_string(),
        serde_json::Value::Array(dec_layers),
    );

    // Decoder norm
    map.insert(
        "decoder_norm_w".to_string(),
        array1_to_json(&transformer.decoder.norm.w),
    );
    map.insert(
        "decoder_norm_b".to_string(),
        array1_to_json(&transformer.decoder.norm.b),
    );

    // Output projection
    map.insert(
        "output_proj_b".to_string(),
        array1_to_json(&transformer.output_projection.b),
    );

    serde_json::Value::Object(map)
}

fn serialize_config(config: &TransformerConfig) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    map.insert(
        "src_vocab".to_string(),
        serde_json::Value::Number(config.src_vocab.into()),
    );
    map.insert(
        "tgt_vocab".to_string(),
        serde_json::Value::Number(config.tgt_vocab.into()),
    );
    map.insert(
        "d_model".to_string(),
        serde_json::Value::Number(config.d_model.into()),
    );
    map.insert(
        "n_heads".to_string(),
        serde_json::Value::Number(config.n_heads.into()),
    );
    map.insert(
        "d_ff".to_string(),
        serde_json::Value::Number(config.d_ff.into()),
    );
    map.insert(
        "n_layers".to_string(),
        serde_json::Value::Number(config.n_layers.into()),
    );
    map.insert(
        "max_len".to_string(),
        serde_json::Value::Number(config.max_len.into()),
    );
    map.insert(
        "dropout".to_string(),
        serde_json::Value::Number(
            serde_json::Number::from_f64(config.dropout).unwrap_or(serde_json::Number::from(0)),
        ),
    );
    map.insert(
        "pad_id".to_string(),
        serde_json::Value::Number(config.pad_id.into()),
    );
    map.insert(
        "bos_id".to_string(),
        serde_json::Value::Number(config.bos_id.into()),
    );
    map.insert(
        "eos_id".to_string(),
        serde_json::Value::Number(config.eos_id.into()),
    );
    map.insert(
        "label_smoothing".to_string(),
        serde_json::Value::Number(
            serde_json::Number::from_f64(config.label_smoothing)
                .unwrap_or(serde_json::Number::from(0)),
        ),
    );
    map.insert(
        "warmup_steps".to_string(),
        serde_json::Value::Number(config.warmup_steps.into()),
    );
    serde_json::Value::Object(map)
}

fn serialize_encoder_layer(layer: &EncoderLayer) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    map.insert(
        "self_attn_wq".to_string(),
        array2_to_json(&layer.self_attn.w_q.w),
    );
    map.insert(
        "self_attn_wk".to_string(),
        array2_to_json(&layer.self_attn.w_k.w),
    );
    map.insert(
        "self_attn_wv".to_string(),
        array2_to_json(&layer.self_attn.w_v.w),
    );
    map.insert(
        "self_attn_wo".to_string(),
        array2_to_json(&layer.self_attn.w_o.w),
    );
    map.insert(
        "self_attn_bo".to_string(),
        array1_to_json(&layer.self_attn.w_o.b),
    );
    map.insert(
        "ffn_w1".to_string(),
        array2_to_json(&layer.feed_forward.w_1.w),
    );
    map.insert(
        "ffn_b1".to_string(),
        array1_to_json(&layer.feed_forward.w_1.b),
    );
    map.insert(
        "ffn_w2".to_string(),
        array2_to_json(&layer.feed_forward.w_2.w),
    );
    map.insert(
        "ffn_b2".to_string(),
        array1_to_json(&layer.feed_forward.w_2.b),
    );
    map.insert("norm1_w".to_string(), array1_to_json(&layer.norm_1.w));
    map.insert("norm1_b".to_string(), array1_to_json(&layer.norm_1.b));
    map.insert("norm2_w".to_string(), array1_to_json(&layer.norm_2.w));
    map.insert("norm2_b".to_string(), array1_to_json(&layer.norm_2.b));
    serde_json::Value::Object(map)
}

fn serialize_decoder_layer(layer: &DecoderLayer) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    map.insert(
        "self_attn_wq".to_string(),
        array2_to_json(&layer.self_attn.w_q.w),
    );
    map.insert(
        "self_attn_wk".to_string(),
        array2_to_json(&layer.self_attn.w_k.w),
    );
    map.insert(
        "self_attn_wv".to_string(),
        array2_to_json(&layer.self_attn.w_v.w),
    );
    map.insert(
        "self_attn_wo".to_string(),
        array2_to_json(&layer.self_attn.w_o.w),
    );
    map.insert(
        "self_attn_bo".to_string(),
        array1_to_json(&layer.self_attn.w_o.b),
    );
    map.insert(
        "cross_attn_wq".to_string(),
        array2_to_json(&layer.cross_attn.w_q.w),
    );
    map.insert(
        "cross_attn_wk".to_string(),
        array2_to_json(&layer.cross_attn.w_k.w),
    );
    map.insert(
        "cross_attn_wv".to_string(),
        array2_to_json(&layer.cross_attn.w_v.w),
    );
    map.insert(
        "cross_attn_wo".to_string(),
        array2_to_json(&layer.cross_attn.w_o.w),
    );
    map.insert(
        "cross_attn_bo".to_string(),
        array1_to_json(&layer.cross_attn.w_o.b),
    );
    map.insert(
        "ffn_w1".to_string(),
        array2_to_json(&layer.feed_forward.w_1.w),
    );
    map.insert(
        "ffn_b1".to_string(),
        array1_to_json(&layer.feed_forward.w_1.b),
    );
    map.insert(
        "ffn_w2".to_string(),
        array2_to_json(&layer.feed_forward.w_2.w),
    );
    map.insert(
        "ffn_b2".to_string(),
        array1_to_json(&layer.feed_forward.w_2.b),
    );
    map.insert("norm1_w".to_string(), array1_to_json(&layer.norm_1.w));
    map.insert("norm1_b".to_string(), array1_to_json(&layer.norm_1.b));
    map.insert("norm2_w".to_string(), array1_to_json(&layer.norm_2.w));
    map.insert("norm2_b".to_string(), array1_to_json(&layer.norm_2.b));
    map.insert("norm3_w".to_string(), array1_to_json(&layer.norm_3.w));
    map.insert("norm3_b".to_string(), array1_to_json(&layer.norm_3.b));
    serde_json::Value::Object(map)
}

fn array2_to_json(a: &Array2<f64>) -> serde_json::Value {
    let shape = a.shape();
    let data: Vec<f64> = a.iter().copied().collect();
    let mut map = serde_json::Map::new();
    map.insert(
        "shape".to_string(),
        serde_json::Value::Array(vec![
            serde_json::Value::Number(shape[0].into()),
            serde_json::Value::Number(shape[1].into()),
        ]),
    );
    map.insert(
        "data".to_string(),
        serde_json::Value::Array(
            data.into_iter()
                .map(|v| {
                    serde_json::Value::Number(
                        serde_json::Number::from_f64(v).unwrap_or(serde_json::Number::from(0)),
                    )
                })
                .collect(),
        ),
    );
    serde_json::Value::Object(map)
}

fn array1_to_json(a: &Array1<f64>) -> serde_json::Value {
    let data: Vec<f64> = a.iter().copied().collect();
    serde_json::Value::Array(
        data.into_iter()
            .map(|v| {
                serde_json::Value::Number(
                    serde_json::Number::from_f64(v).unwrap_or(serde_json::Number::from(0)),
                )
            })
            .collect(),
    )
}

fn deserialize_model(
    data: &serde_json::Value,
    transformer: &mut Transformer,
) -> Result<(), String> {
    let obj = data.as_object().ok_or("Expected JSON object")?;

    // Embeddings
    if let Some(v) = obj.get("src_embeddings") {
        transformer.src_embeddings = json_to_array2(v)?;
    }
    if let Some(v) = obj.get("tgt_embeddings") {
        transformer.tgt_embeddings = json_to_array2(v)?;
    }

    // Encoder layers
    if let Some(serde_json::Value::Array(layers)) = obj.get("encoder_layers") {
        for (i, layer_data) in layers.iter().enumerate() {
            if i < transformer.encoder.layers.len() {
                deserialize_encoder_layer(layer_data, &mut transformer.encoder.layers[i])?;
            }
        }
    }

    // Encoder norm
    if let Some(v) = obj.get("encoder_norm_w") {
        transformer.encoder.norm.w = json_to_array1(v)?;
    }
    if let Some(v) = obj.get("encoder_norm_b") {
        transformer.encoder.norm.b = json_to_array1(v)?;
    }

    // Decoder layers
    if let Some(serde_json::Value::Array(layers)) = obj.get("decoder_layers") {
        for (i, layer_data) in layers.iter().enumerate() {
            if i < transformer.decoder.layers.len() {
                deserialize_decoder_layer(layer_data, &mut transformer.decoder.layers[i])?;
            }
        }
    }

    // Decoder norm
    if let Some(v) = obj.get("decoder_norm_w") {
        transformer.decoder.norm.w = json_to_array1(v)?;
    }
    if let Some(v) = obj.get("decoder_norm_b") {
        transformer.decoder.norm.b = json_to_array1(v)?;
    }

    // Output projection bias
    if let Some(v) = obj.get("output_proj_b") {
        transformer.output_projection.b = json_to_array1(v)?;
    }

    // Weight tying: sync output projection with tgt_embeddings
    transformer.output_projection.w = transformer.tgt_embeddings.t().to_owned();

    Ok(())
}

fn deserialize_encoder_layer(
    data: &serde_json::Value,
    layer: &mut EncoderLayer,
) -> Result<(), String> {
    let obj = data.as_object().ok_or("Expected object")?;
    if let Some(v) = obj.get("self_attn_wq") {
        layer.self_attn.w_q.w = json_to_array2(v)?;
    }
    if let Some(v) = obj.get("self_attn_wk") {
        layer.self_attn.w_k.w = json_to_array2(v)?;
    }
    if let Some(v) = obj.get("self_attn_wv") {
        layer.self_attn.w_v.w = json_to_array2(v)?;
    }
    if let Some(v) = obj.get("self_attn_wo") {
        layer.self_attn.w_o.w = json_to_array2(v)?;
    }
    if let Some(v) = obj.get("self_attn_bo") {
        layer.self_attn.w_o.b = json_to_array1(v)?;
    }
    if let Some(v) = obj.get("ffn_w1") {
        layer.feed_forward.w_1.w = json_to_array2(v)?;
    }
    if let Some(v) = obj.get("ffn_b1") {
        layer.feed_forward.w_1.b = json_to_array1(v)?;
    }
    if let Some(v) = obj.get("ffn_w2") {
        layer.feed_forward.w_2.w = json_to_array2(v)?;
    }
    if let Some(v) = obj.get("ffn_b2") {
        layer.feed_forward.w_2.b = json_to_array1(v)?;
    }
    if let Some(v) = obj.get("norm1_w") {
        layer.norm_1.w = json_to_array1(v)?;
    }
    if let Some(v) = obj.get("norm1_b") {
        layer.norm_1.b = json_to_array1(v)?;
    }
    if let Some(v) = obj.get("norm2_w") {
        layer.norm_2.w = json_to_array1(v)?;
    }
    if let Some(v) = obj.get("norm2_b") {
        layer.norm_2.b = json_to_array1(v)?;
    }
    Ok(())
}

fn deserialize_decoder_layer(
    data: &serde_json::Value,
    layer: &mut DecoderLayer,
) -> Result<(), String> {
    let obj = data.as_object().ok_or("Expected object")?;
    if let Some(v) = obj.get("self_attn_wq") {
        layer.self_attn.w_q.w = json_to_array2(v)?;
    }
    if let Some(v) = obj.get("self_attn_wk") {
        layer.self_attn.w_k.w = json_to_array2(v)?;
    }
    if let Some(v) = obj.get("self_attn_wv") {
        layer.self_attn.w_v.w = json_to_array2(v)?;
    }
    if let Some(v) = obj.get("self_attn_wo") {
        layer.self_attn.w_o.w = json_to_array2(v)?;
    }
    if let Some(v) = obj.get("self_attn_bo") {
        layer.self_attn.w_o.b = json_to_array1(v)?;
    }
    if let Some(v) = obj.get("cross_attn_wq") {
        layer.cross_attn.w_q.w = json_to_array2(v)?;
    }
    if let Some(v) = obj.get("cross_attn_wk") {
        layer.cross_attn.w_k.w = json_to_array2(v)?;
    }
    if let Some(v) = obj.get("cross_attn_wv") {
        layer.cross_attn.w_v.w = json_to_array2(v)?;
    }
    if let Some(v) = obj.get("cross_attn_wo") {
        layer.cross_attn.w_o.w = json_to_array2(v)?;
    }
    if let Some(v) = obj.get("cross_attn_bo") {
        layer.cross_attn.w_o.b = json_to_array1(v)?;
    }
    if let Some(v) = obj.get("ffn_w1") {
        layer.feed_forward.w_1.w = json_to_array2(v)?;
    }
    if let Some(v) = obj.get("ffn_b1") {
        layer.feed_forward.w_1.b = json_to_array1(v)?;
    }
    if let Some(v) = obj.get("ffn_w2") {
        layer.feed_forward.w_2.w = json_to_array2(v)?;
    }
    if let Some(v) = obj.get("ffn_b2") {
        layer.feed_forward.w_2.b = json_to_array1(v)?;
    }
    if let Some(v) = obj.get("norm1_w") {
        layer.norm_1.w = json_to_array1(v)?;
    }
    if let Some(v) = obj.get("norm1_b") {
        layer.norm_1.b = json_to_array1(v)?;
    }
    if let Some(v) = obj.get("norm2_w") {
        layer.norm_2.w = json_to_array1(v)?;
    }
    if let Some(v) = obj.get("norm2_b") {
        layer.norm_2.b = json_to_array1(v)?;
    }
    if let Some(v) = obj.get("norm3_w") {
        layer.norm_3.w = json_to_array1(v)?;
    }
    if let Some(v) = obj.get("norm3_b") {
        layer.norm_3.b = json_to_array1(v)?;
    }
    Ok(())
}

fn json_to_array2(v: &serde_json::Value) -> Result<Array2<f64>, String> {
    let obj = v.as_object().ok_or("Expected object for array2")?;
    let shape = obj
        .get("shape")
        .and_then(|s| s.as_array())
        .ok_or("Missing shape")?;
    let rows = shape[0].as_u64().ok_or("Invalid rows")? as usize;
    let cols = shape[1].as_u64().ok_or("Invalid cols")? as usize;
    let data = obj
        .get("data")
        .and_then(|d| d.as_array())
        .ok_or("Missing data")?;
    let values: Vec<f64> = data.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect();
    Ok(Array2::from_shape_vec((rows, cols), values).map_err(|e| format!("Shape error: {}", e))?)
}

fn json_to_array1(v: &serde_json::Value) -> Result<Array1<f64>, String> {
    let arr = v.as_array().ok_or("Expected array for array1")?;
    let values: Vec<f64> = arr.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect();
    Ok(Array1::from_vec(values))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::TransformerConfig;

    fn test_config() -> TransformerConfig {
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
    fn test_save_load_roundtrip() {
        let config = test_config();
        let t1 = Transformer::new(config.clone());
        let path = "test_checkpoint.json";
        save_checkpoint(&t1, path).unwrap();
        let t2 = load_checkpoint(path, &config).unwrap();
        // Clean up
        let _ = std::fs::remove_file(path);

        // Verify weights match
        assert_eq!(t1.src_embeddings.shape(), t2.src_embeddings.shape());
        assert_eq!(t1.tgt_embeddings.shape(), t2.tgt_embeddings.shape());
        assert_eq!(t1.encoder.layers.len(), t2.encoder.layers.len());
        assert_eq!(t1.decoder.layers.len(), t2.decoder.layers.len());

        // Check a few values
        for i in 0..t1.src_embeddings.len() {
            assert!(
                (t1.src_embeddings.as_slice().unwrap()[i]
                    - t2.src_embeddings.as_slice().unwrap()[i])
                    .abs()
                    < 1e-10
            );
        }
    }

    #[test]
    fn test_average_checkpoints() {
        let config = test_config();
        let t1 = Transformer::new(config.clone());
        let t2 = Transformer::new(config.clone());

        let p1 = "test_cp1.json";
        let p2 = "test_cp2.json";
        let p_out = "test_avg.json";
        save_checkpoint(&t1, p1).unwrap();
        save_checkpoint(&t2, p2).unwrap();

        let avg = average_checkpoints(&[p1.to_string(), p2.to_string()], &config, p_out).unwrap();

        // Clean up
        let _ = std::fs::remove_file(p1);
        let _ = std::fs::remove_file(p2);
        let _ = std::fs::remove_file(p_out);

        // Averaged model should have same shape
        assert_eq!(avg.src_embeddings.shape(), t1.src_embeddings.shape());
    }
}

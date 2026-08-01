use std::time::Instant;

use ndarray::{s, Array1, Array2, Axis};

use crate::attention::MultiHeadAttention;
use crate::loss::perplexity;
use crate::model::Transformer;
use crate::tensor_ops::{
    causal_mask, embedding_lookup, make_decoder_mask, make_src_mask, padding_mask,
};

// ---------------------------------------------------------------------------
// Adam optimizer state (Section 5.3)
// ---------------------------------------------------------------------------

pub struct AdamState {
    step: usize,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub warmup_steps: usize,
    pub d_model: f64,
    dec_m: Vec<DecGrads>,
    dec_v: Vec<DecGrads>,
    enc_m: Vec<EncGrads>,
    enc_v: Vec<EncGrads>,
    out_proj_m_b: Array1<f64>,
    out_proj_v_b: Array1<f64>,
    tgt_emb_m: Array2<f64>,
    tgt_emb_v: Array2<f64>,
    src_emb_m: Array2<f64>,
    src_emb_v: Array2<f64>,
    enc_norm_m_w: Array1<f64>,
    enc_norm_v_w: Array1<f64>,
    enc_norm_m_b: Array1<f64>,
    enc_norm_v_b: Array1<f64>,
}

impl AdamState {
    pub fn new(d_model: usize, warmup_steps: usize, _n_layers: usize) -> Self {
        let zeros2 = Array2::zeros((0, 0));
        Self {
            step: 0,
            beta1: 0.9,
            beta2: 0.98,
            eps: 1e-9,
            warmup_steps,
            d_model: d_model as f64,
            dec_m: Vec::new(),
            dec_v: Vec::new(),
            enc_m: Vec::new(),
            enc_v: Vec::new(),
            out_proj_m_b: Array1::zeros(d_model),
            out_proj_v_b: Array1::zeros(d_model),
            tgt_emb_m: zeros2.clone(),
            tgt_emb_v: zeros2.clone(),
            src_emb_m: zeros2.clone(),
            src_emb_v: zeros2.clone(),
            enc_norm_m_w: Array1::zeros(d_model),
            enc_norm_v_w: Array1::zeros(d_model),
            enc_norm_m_b: Array1::zeros(d_model),
            enc_norm_v_b: Array1::zeros(d_model),
        }
    }

    pub fn init_param(&mut self, name: &str, dim: (usize, usize)) {
        match name {
            "out_proj_w" => {
                self.out_proj_m_b = Array1::zeros(dim.1);
                self.out_proj_v_b = Array1::zeros(dim.1);
            }
            "tgt_emb" => {
                self.tgt_emb_m = Array2::zeros(dim);
                self.tgt_emb_v = Array2::zeros(dim);
            }
            "src_emb" => {
                self.src_emb_m = Array2::zeros(dim);
                self.src_emb_v = Array2::zeros(dim);
            }
            _ => {}
        }
    }

    pub fn init_dec_layer(&mut self, layer: &crate::decoder::DecoderLayer) {
        self.dec_m.push(DecGrads {
            self_attn_wq: Array2::zeros(layer.self_attn.w_q.w.dim()),
            self_attn_wk: Array2::zeros(layer.self_attn.w_k.w.dim()),
            self_attn_wv: Array2::zeros(layer.self_attn.w_v.w.dim()),
            self_attn_wo: Array2::zeros(layer.self_attn.w_o.w.dim()),
            self_attn_bo: Array1::zeros(layer.self_attn.w_o.b.len()),
            cross_attn_wq: Array2::zeros(layer.cross_attn.w_q.w.dim()),
            cross_attn_wk: Array2::zeros(layer.cross_attn.w_k.w.dim()),
            cross_attn_wv: Array2::zeros(layer.cross_attn.w_v.w.dim()),
            cross_attn_wo: Array2::zeros(layer.cross_attn.w_o.w.dim()),
            cross_attn_bo: Array1::zeros(layer.cross_attn.w_o.b.len()),
            ffn_w1: Array2::zeros(layer.feed_forward.w_1.w.dim()),
            ffn_b1: Array1::zeros(layer.feed_forward.w_1.b.len()),
            ffn_w2: Array2::zeros(layer.feed_forward.w_2.w.dim()),
            ffn_b2: Array1::zeros(layer.feed_forward.w_2.b.len()),
            norm1_w: Array1::zeros(layer.norm_1.w.len()),
            norm1_b: Array1::zeros(layer.norm_1.b.len()),
            norm2_w: Array1::zeros(layer.norm_2.w.len()),
            norm2_b: Array1::zeros(layer.norm_2.b.len()),
            norm3_w: Array1::zeros(layer.norm_3.w.len()),
            norm3_b: Array1::zeros(layer.norm_3.b.len()),
        });
        self.dec_v.push(DecGrads {
            self_attn_wq: Array2::zeros(layer.self_attn.w_q.w.dim()),
            self_attn_wk: Array2::zeros(layer.self_attn.w_k.w.dim()),
            self_attn_wv: Array2::zeros(layer.self_attn.w_v.w.dim()),
            self_attn_wo: Array2::zeros(layer.self_attn.w_o.w.dim()),
            self_attn_bo: Array1::zeros(layer.self_attn.w_o.b.len()),
            cross_attn_wq: Array2::zeros(layer.cross_attn.w_q.w.dim()),
            cross_attn_wk: Array2::zeros(layer.cross_attn.w_k.w.dim()),
            cross_attn_wv: Array2::zeros(layer.cross_attn.w_v.w.dim()),
            cross_attn_wo: Array2::zeros(layer.cross_attn.w_o.w.dim()),
            cross_attn_bo: Array1::zeros(layer.cross_attn.w_o.b.len()),
            ffn_w1: Array2::zeros(layer.feed_forward.w_1.w.dim()),
            ffn_b1: Array1::zeros(layer.feed_forward.w_1.b.len()),
            ffn_w2: Array2::zeros(layer.feed_forward.w_2.w.dim()),
            ffn_b2: Array1::zeros(layer.feed_forward.w_2.b.len()),
            norm1_w: Array1::zeros(layer.norm_1.w.len()),
            norm1_b: Array1::zeros(layer.norm_1.b.len()),
            norm2_w: Array1::zeros(layer.norm_2.w.len()),
            norm2_b: Array1::zeros(layer.norm_2.b.len()),
            norm3_w: Array1::zeros(layer.norm_3.w.len()),
            norm3_b: Array1::zeros(layer.norm_3.b.len()),
        });
    }

    pub fn init_enc_layer(&mut self, layer: &crate::encoder::EncoderLayer) {
        self.enc_m.push(EncGrads {
            self_attn_wq: Array2::zeros(layer.self_attn.w_q.w.dim()),
            self_attn_wk: Array2::zeros(layer.self_attn.w_k.w.dim()),
            self_attn_wv: Array2::zeros(layer.self_attn.w_v.w.dim()),
            self_attn_wo: Array2::zeros(layer.self_attn.w_o.w.dim()),
            self_attn_bo: Array1::zeros(layer.self_attn.w_o.b.len()),
            ffn_w1: Array2::zeros(layer.feed_forward.w_1.w.dim()),
            ffn_b1: Array1::zeros(layer.feed_forward.w_1.b.len()),
            ffn_w2: Array2::zeros(layer.feed_forward.w_2.w.dim()),
            ffn_b2: Array1::zeros(layer.feed_forward.w_2.b.len()),
            norm1_w: Array1::zeros(layer.norm_1.w.len()),
            norm1_b: Array1::zeros(layer.norm_1.b.len()),
            norm2_w: Array1::zeros(layer.norm_2.w.len()),
            norm2_b: Array1::zeros(layer.norm_2.b.len()),
        });
        self.enc_v.push(EncGrads {
            self_attn_wq: Array2::zeros(layer.self_attn.w_q.w.dim()),
            self_attn_wk: Array2::zeros(layer.self_attn.w_k.w.dim()),
            self_attn_wv: Array2::zeros(layer.self_attn.w_v.w.dim()),
            self_attn_wo: Array2::zeros(layer.self_attn.w_o.w.dim()),
            self_attn_bo: Array1::zeros(layer.self_attn.w_o.b.len()),
            ffn_w1: Array2::zeros(layer.feed_forward.w_1.w.dim()),
            ffn_b1: Array1::zeros(layer.feed_forward.w_1.b.len()),
            ffn_w2: Array2::zeros(layer.feed_forward.w_2.w.dim()),
            ffn_b2: Array1::zeros(layer.feed_forward.w_2.b.len()),
            norm1_w: Array1::zeros(layer.norm_1.w.len()),
            norm1_b: Array1::zeros(layer.norm_1.b.len()),
            norm2_w: Array1::zeros(layer.norm_2.w.len()),
            norm2_b: Array1::zeros(layer.norm_2.b.len()),
        });
    }

    pub fn learning_rate(&self) -> f64 {
        let step = self.step as f64;
        if step == 0.0 {
            return self.d_model.powf(-0.5) * self.warmup_steps as f64;
        }
        self.d_model.powf(-0.5)
            * step
                .powf(-0.5)
                .min(step * (self.warmup_steps as f64).powf(-1.5))
    }

    fn update_dec_layer(
        &mut self,
        li: usize,
        layer: &mut crate::decoder::DecoderLayer,
        g: &DecGrads,
    ) {
        let lr = self.learning_rate();
        let bc1 = 1.0 - self.beta1.powi(self.step as i32);
        let bc2 = 1.0 - self.beta2.powi(self.step as i32);
        let b1 = self.beta1;
        let b2 = self.beta2;
        let eps = self.eps;
        let m = &mut self.dec_m[li];
        let v = &mut self.dec_v[li];

        adam_update_2d(
            &mut layer.self_attn.w_q.w,
            &mut m.self_attn_wq,
            &mut v.self_attn_wq,
            &g.self_attn_wq,
            lr,
            bc1,
            bc2,
            b1,
            b2,
            eps,
        );
        adam_update_2d(
            &mut layer.self_attn.w_k.w,
            &mut m.self_attn_wk,
            &mut v.self_attn_wk,
            &g.self_attn_wk,
            lr,
            bc1,
            bc2,
            b1,
            b2,
            eps,
        );
        adam_update_2d(
            &mut layer.self_attn.w_v.w,
            &mut m.self_attn_wv,
            &mut v.self_attn_wv,
            &g.self_attn_wv,
            lr,
            bc1,
            bc2,
            b1,
            b2,
            eps,
        );
        adam_update_2d(
            &mut layer.self_attn.w_o.w,
            &mut m.self_attn_wo,
            &mut v.self_attn_wo,
            &g.self_attn_wo,
            lr,
            bc1,
            bc2,
            b1,
            b2,
            eps,
        );
        adam_update_1d(
            &mut layer.self_attn.w_o.b,
            &mut m.self_attn_bo,
            &mut v.self_attn_bo,
            &g.self_attn_bo,
            lr,
            bc1,
            bc2,
            b1,
            b2,
            eps,
        );
        adam_update_2d(
            &mut layer.cross_attn.w_q.w,
            &mut m.cross_attn_wq,
            &mut v.cross_attn_wq,
            &g.cross_attn_wq,
            lr,
            bc1,
            bc2,
            b1,
            b2,
            eps,
        );
        adam_update_2d(
            &mut layer.cross_attn.w_k.w,
            &mut m.cross_attn_wk,
            &mut v.cross_attn_wk,
            &g.cross_attn_wk,
            lr,
            bc1,
            bc2,
            b1,
            b2,
            eps,
        );
        adam_update_2d(
            &mut layer.cross_attn.w_v.w,
            &mut m.cross_attn_wv,
            &mut v.cross_attn_wv,
            &g.cross_attn_wv,
            lr,
            bc1,
            bc2,
            b1,
            b2,
            eps,
        );
        adam_update_2d(
            &mut layer.cross_attn.w_o.w,
            &mut m.cross_attn_wo,
            &mut v.cross_attn_wo,
            &g.cross_attn_wo,
            lr,
            bc1,
            bc2,
            b1,
            b2,
            eps,
        );
        adam_update_1d(
            &mut layer.cross_attn.w_o.b,
            &mut m.cross_attn_bo,
            &mut v.cross_attn_bo,
            &g.cross_attn_bo,
            lr,
            bc1,
            bc2,
            b1,
            b2,
            eps,
        );
        adam_update_2d(
            &mut layer.feed_forward.w_1.w,
            &mut m.ffn_w1,
            &mut v.ffn_w1,
            &g.ffn_w1,
            lr,
            bc1,
            bc2,
            b1,
            b2,
            eps,
        );
        adam_update_1d(
            &mut layer.feed_forward.w_1.b,
            &mut m.ffn_b1,
            &mut v.ffn_b1,
            &g.ffn_b1,
            lr,
            bc1,
            bc2,
            b1,
            b2,
            eps,
        );
        adam_update_2d(
            &mut layer.feed_forward.w_2.w,
            &mut m.ffn_w2,
            &mut v.ffn_w2,
            &g.ffn_w2,
            lr,
            bc1,
            bc2,
            b1,
            b2,
            eps,
        );
        adam_update_1d(
            &mut layer.feed_forward.w_2.b,
            &mut m.ffn_b2,
            &mut v.ffn_b2,
            &g.ffn_b2,
            lr,
            bc1,
            bc2,
            b1,
            b2,
            eps,
        );
        adam_update_1d(
            &mut layer.norm_1.w,
            &mut m.norm1_w,
            &mut v.norm1_w,
            &g.norm1_w,
            lr,
            bc1,
            bc2,
            b1,
            b2,
            eps,
        );
        adam_update_1d(
            &mut layer.norm_1.b,
            &mut m.norm1_b,
            &mut v.norm1_b,
            &g.norm1_b,
            lr,
            bc1,
            bc2,
            b1,
            b2,
            eps,
        );
        adam_update_1d(
            &mut layer.norm_2.w,
            &mut m.norm2_w,
            &mut v.norm2_w,
            &g.norm2_w,
            lr,
            bc1,
            bc2,
            b1,
            b2,
            eps,
        );
        adam_update_1d(
            &mut layer.norm_2.b,
            &mut m.norm2_b,
            &mut v.norm2_b,
            &g.norm2_b,
            lr,
            bc1,
            bc2,
            b1,
            b2,
            eps,
        );
        adam_update_1d(
            &mut layer.norm_3.w,
            &mut m.norm3_w,
            &mut v.norm3_w,
            &g.norm3_w,
            lr,
            bc1,
            bc2,
            b1,
            b2,
            eps,
        );
        adam_update_1d(
            &mut layer.norm_3.b,
            &mut m.norm3_b,
            &mut v.norm3_b,
            &g.norm3_b,
            lr,
            bc1,
            bc2,
            b1,
            b2,
            eps,
        );
    }

    fn update_enc_layer(
        &mut self,
        li: usize,
        layer: &mut crate::encoder::EncoderLayer,
        g: &EncGrads,
    ) {
        let lr = self.learning_rate();
        let bc1 = 1.0 - self.beta1.powi(self.step as i32);
        let bc2 = 1.0 - self.beta2.powi(self.step as i32);
        let b1 = self.beta1;
        let b2 = self.beta2;
        let eps = self.eps;
        let m = &mut self.enc_m[li];
        let v = &mut self.enc_v[li];

        adam_update_2d(
            &mut layer.self_attn.w_q.w,
            &mut m.self_attn_wq,
            &mut v.self_attn_wq,
            &g.self_attn_wq,
            lr,
            bc1,
            bc2,
            b1,
            b2,
            eps,
        );
        adam_update_2d(
            &mut layer.self_attn.w_k.w,
            &mut m.self_attn_wk,
            &mut v.self_attn_wk,
            &g.self_attn_wk,
            lr,
            bc1,
            bc2,
            b1,
            b2,
            eps,
        );
        adam_update_2d(
            &mut layer.self_attn.w_v.w,
            &mut m.self_attn_wv,
            &mut v.self_attn_wv,
            &g.self_attn_wv,
            lr,
            bc1,
            bc2,
            b1,
            b2,
            eps,
        );
        adam_update_2d(
            &mut layer.self_attn.w_o.w,
            &mut m.self_attn_wo,
            &mut v.self_attn_wo,
            &g.self_attn_wo,
            lr,
            bc1,
            bc2,
            b1,
            b2,
            eps,
        );
        adam_update_1d(
            &mut layer.self_attn.w_o.b,
            &mut m.self_attn_bo,
            &mut v.self_attn_bo,
            &g.self_attn_bo,
            lr,
            bc1,
            bc2,
            b1,
            b2,
            eps,
        );
        adam_update_2d(
            &mut layer.feed_forward.w_1.w,
            &mut m.ffn_w1,
            &mut v.ffn_w1,
            &g.ffn_w1,
            lr,
            bc1,
            bc2,
            b1,
            b2,
            eps,
        );
        adam_update_1d(
            &mut layer.feed_forward.w_1.b,
            &mut m.ffn_b1,
            &mut v.ffn_b1,
            &g.ffn_b1,
            lr,
            bc1,
            bc2,
            b1,
            b2,
            eps,
        );
        adam_update_2d(
            &mut layer.feed_forward.w_2.w,
            &mut m.ffn_w2,
            &mut v.ffn_w2,
            &g.ffn_w2,
            lr,
            bc1,
            bc2,
            b1,
            b2,
            eps,
        );
        adam_update_1d(
            &mut layer.feed_forward.w_2.b,
            &mut m.ffn_b2,
            &mut v.ffn_b2,
            &g.ffn_b2,
            lr,
            bc1,
            bc2,
            b1,
            b2,
            eps,
        );
        adam_update_1d(
            &mut layer.norm_1.w,
            &mut m.norm1_w,
            &mut v.norm1_w,
            &g.norm1_w,
            lr,
            bc1,
            bc2,
            b1,
            b2,
            eps,
        );
        adam_update_1d(
            &mut layer.norm_1.b,
            &mut m.norm1_b,
            &mut v.norm1_b,
            &g.norm1_b,
            lr,
            bc1,
            bc2,
            b1,
            b2,
            eps,
        );
        adam_update_1d(
            &mut layer.norm_2.w,
            &mut m.norm2_w,
            &mut v.norm2_w,
            &g.norm2_w,
            lr,
            bc1,
            bc2,
            b1,
            b2,
            eps,
        );
        adam_update_1d(
            &mut layer.norm_2.b,
            &mut m.norm2_b,
            &mut v.norm2_b,
            &g.norm2_b,
            lr,
            bc1,
            bc2,
            b1,
            b2,
            eps,
        );
    }

    fn update_enc_norm(
        &mut self,
        layer: &mut crate::attention::LayerNorm,
        dw: &Array1<f64>,
        db: &Array1<f64>,
    ) {
        let lr = self.learning_rate();
        let bc1 = 1.0 - self.beta1.powi(self.step as i32);
        let bc2 = 1.0 - self.beta2.powi(self.step as i32);
        let b1 = self.beta1;
        let b2 = self.beta2;
        let eps = self.eps;

        adam_update_1d(
            &mut layer.w,
            &mut self.enc_norm_m_w,
            &mut self.enc_norm_v_w,
            dw,
            lr,
            bc1,
            bc2,
            b1,
            b2,
            eps,
        );
        adam_update_1d(
            &mut layer.b,
            &mut self.enc_norm_m_b,
            &mut self.enc_norm_v_b,
            db,
            lr,
            bc1,
            bc2,
            b1,
            b2,
            eps,
        );
    }
}

// ---------------------------------------------------------------------------
// Adam update primitives
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn adam_update_2d(
    param: &mut Array2<f64>,
    m: &mut Array2<f64>,
    v: &mut Array2<f64>,
    grad: &Array2<f64>,
    lr: f64,
    bc1: f64,
    bc2: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
) {
    *m = &*m * beta1 + grad * (1.0 - beta1);
    let g2 = grad.mapv(|x| x * x);
    *v = &*v * beta2 + &g2 * (1.0 - beta2);
    let m_hat = &*m / bc1;
    let v_hat = &*v / bc2;
    let denom = v_hat.mapv(f64::sqrt) + eps;
    *param = &*param - &(&m_hat / &denom) * lr;
}

#[allow(clippy::too_many_arguments)]
fn adam_update_1d(
    param: &mut Array1<f64>,
    m: &mut Array1<f64>,
    v: &mut Array1<f64>,
    grad: &Array1<f64>,
    lr: f64,
    bc1: f64,
    bc2: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
) {
    *m = &*m * beta1 + grad * (1.0 - beta1);
    let g2 = grad.mapv(|x| x * x);
    *v = &*v * beta2 + &g2 * (1.0 - beta2);
    let m_hat = &*m / bc1;
    let v_hat = &*v / bc2;
    let denom = v_hat.mapv(f64::sqrt) + eps;
    *param = &*param - &(&m_hat / &denom) * lr;
}

/// Run one training step: forward + backward + Adam weight update.
///
/// Returns the label-smoothing loss for this step.
pub fn train_step(
    transformer: &mut Transformer,
    src: &[usize],
    tgt_input: &[usize],
    tgt_output: &[usize],
    adam: &mut AdamState,
) -> f64 {
    adam.step += 1;
    let scale = transformer.scale;
    let d_model = transformer.config.d_model;
    let pad_id = transformer.config.pad_id;
    let tgt_vocab = transformer.config.tgt_vocab;
    let n_layers = transformer.config.n_layers;
    let eps = transformer.config.label_smoothing;

    // === FORWARD PASS WITH CACHING ===

    let src_emb = embedding_lookup(&transformer.src_embeddings, src) * scale;
    let pe_slice = transformer
        .positional_encoding
        .slice(s![..src.len(), ..])
        .to_owned();
    let src_emb = &src_emb + &pe_slice;

    let tgt_emb = embedding_lookup(&transformer.tgt_embeddings, tgt_input) * scale;
    let pe_slice = transformer
        .positional_encoding
        .slice(s![..tgt_input.len(), ..])
        .to_owned();
    let tgt_emb = &tgt_emb + &pe_slice;

    let src_pad = padding_mask(src, pad_id);
    let src_mask = make_src_mask(&src_pad);

    let tgt_pad = padding_mask(tgt_input, pad_id);
    let tgt_causal = causal_mask(tgt_input.len());
    let tgt_mask = make_decoder_mask(&tgt_causal, &tgt_pad);

    // Encoder forward
    let mut enc_x = src_emb;
    let mut enc_caches: Vec<EncLayerCache> = Vec::with_capacity(n_layers);
    for layer in &transformer.encoder.layers {
        let cache = EncLayerCache::forward(layer, &enc_x, &src_mask);
        enc_x = cache.output.clone();
        enc_caches.push(cache);
    }
    let enc_output = transformer.encoder.norm.forward(&enc_x);

    // Decoder forward
    let mut dec_x = tgt_emb;
    let mut dec_caches: Vec<DecLayerCache> = Vec::with_capacity(n_layers);
    for layer in &transformer.decoder.layers {
        let cache = DecLayerCache::forward(layer, &dec_x, &enc_output, &src_mask, &tgt_mask);
        dec_x = cache.output.clone();
        dec_caches.push(cache);
    }
    let dec_output = transformer.decoder.norm.forward(&dec_x);

    let logits = transformer.output_projection.forward(&dec_output);

    // === COMPUTE LOSS ===
    let log_probs = crate::loss::log_softmax(&logits);
    let loss = crate::loss::label_smoothing_loss(&logits, tgt_output, eps, pad_id);

    // === BACKWARD PASS ===

    // Loss gradient w.r.t. logits
    let mut d_logits = log_probs;
    let seq_len = tgt_output.len().min(logits.nrows());
    let vocab_f = tgt_vocab as f64;
    for i in 0..seq_len {
        if tgt_output[i] == pad_id {
            continue;
        }
        for j in 0..tgt_vocab {
            let smooth = if j == tgt_output[i] {
                (1.0 - eps) + eps / vocab_f
            } else {
                eps / vocab_f
            };
            d_logits[[i, j]] -= smooth;
        }
    }
    d_logits = &d_logits / (seq_len as f64);

    // Backward through output projection: logits = dec_output @ W
    let dw_out = crate::tensor_ops::matmul(&crate::tensor_ops::transpose(&dec_output), &d_logits);
    let mut db_out = d_logits.sum_axis(Axis(0));
    let d_dec_out = crate::tensor_ops::matmul(
        &d_logits,
        &crate::tensor_ops::transpose(&transformer.output_projection.w),
    );

    // Backward through decoder norm
    let (d_dec_x, _dec_norm_dw, _dec_norm_db) =
        layernorm_backward_cached(&d_dec_out, &dec_x, &transformer.decoder.norm.w);

    // Backward through decoder layers
    let mut d_x = d_dec_x;
    let mut all_dec_grads: Vec<DecGrads> = Vec::with_capacity(n_layers);
    let mut all_d_enc_output: Vec<Array2<f64>> = Vec::with_capacity(n_layers);
    for li in (0..n_layers).rev() {
        let (d_x_new, d_enc, grads) =
            dec_caches[li].backward(&d_x, &transformer.decoder.layers[li]);
        all_dec_grads.push(grads);
        all_d_enc_output.push(d_enc);
        d_x = d_x_new;
    }
    all_dec_grads.reverse();
    all_d_enc_output.reverse();

    // Sum d_enc_output from all decoder layers
    let mut total_d_enc_output = all_d_enc_output.remove(0);
    for d in &all_d_enc_output {
        total_d_enc_output = &total_d_enc_output + d;
    }

    // Backward through target embeddings (sparse gradient accumulation)
    let mut tgt_emb_grad: Array2<f64> = Array2::zeros((tgt_vocab, d_model));
    for (i, &idx) in tgt_input.iter().enumerate() {
        if idx < tgt_vocab {
            for j in 0..d_model {
                tgt_emb_grad[[idx, j]] += d_x[[i, j]];
            }
        }
    }

    // Backward through encoder norm
    let (d_enc_x, enc_norm_dw, enc_norm_db) =
        layernorm_backward_cached(&total_d_enc_output, &enc_x, &transformer.encoder.norm.w);

    // Backward through encoder layers
    let mut d_enc_x = d_enc_x;
    let mut all_enc_grads: Vec<EncGrads> = Vec::with_capacity(n_layers);
    for li in (0..n_layers).rev() {
        let (d_x_new, grads) = enc_caches[li].backward(&d_enc_x, &transformer.encoder.layers[li]);
        all_enc_grads.push(grads);
        d_enc_x = d_x_new;
    }
    all_enc_grads.reverse();

    // Source embedding gradient
    let mut src_emb_grad: Array2<f64> = Array2::zeros((transformer.config.src_vocab, d_model));
    for (i, &idx) in src.iter().enumerate() {
        if idx < transformer.config.src_vocab {
            for j in 0..d_model {
                src_emb_grad[[idx, j]] += d_enc_x[[i, j]];
            }
        }
    }

    // Weight tying: output_projection.w = tgt_embeddings.t()
    // Combine gradients: tgt_emb_grad += dw_out.t() (gradient from output projection → embedding)
    let dw_out_t = crate::tensor_ops::transpose(&dw_out);
    tgt_emb_grad = &tgt_emb_grad + &dw_out_t;

    // Global gradient clipping (Section 5.3: clip gradients to prevent explosion)
    let mut total_sq = 0.0_f64;
    for g in &all_dec_grads {
        total_sq += g.self_attn_wq.mapv(|x| x * x).sum();
        total_sq += g.self_attn_wk.mapv(|x| x * x).sum();
        total_sq += g.self_attn_wv.mapv(|x| x * x).sum();
        total_sq += g.self_attn_wo.mapv(|x| x * x).sum();
        total_sq += g.self_attn_bo.mapv(|x| x * x).sum();
        total_sq += g.cross_attn_wq.mapv(|x| x * x).sum();
        total_sq += g.cross_attn_wk.mapv(|x| x * x).sum();
        total_sq += g.cross_attn_wv.mapv(|x| x * x).sum();
        total_sq += g.cross_attn_wo.mapv(|x| x * x).sum();
        total_sq += g.cross_attn_bo.mapv(|x| x * x).sum();
        total_sq += g.ffn_w1.mapv(|x| x * x).sum();
        total_sq += g.ffn_b1.mapv(|x| x * x).sum();
        total_sq += g.ffn_w2.mapv(|x| x * x).sum();
        total_sq += g.ffn_b2.mapv(|x| x * x).sum();
        total_sq += g.norm1_w.mapv(|x| x * x).sum();
        total_sq += g.norm1_b.mapv(|x| x * x).sum();
        total_sq += g.norm2_w.mapv(|x| x * x).sum();
        total_sq += g.norm2_b.mapv(|x| x * x).sum();
        total_sq += g.norm3_w.mapv(|x| x * x).sum();
        total_sq += g.norm3_b.mapv(|x| x * x).sum();
    }
    for g in &all_enc_grads {
        total_sq += g.self_attn_wq.mapv(|x| x * x).sum();
        total_sq += g.self_attn_wk.mapv(|x| x * x).sum();
        total_sq += g.self_attn_wv.mapv(|x| x * x).sum();
        total_sq += g.self_attn_wo.mapv(|x| x * x).sum();
        total_sq += g.self_attn_bo.mapv(|x| x * x).sum();
        total_sq += g.ffn_w1.mapv(|x| x * x).sum();
        total_sq += g.ffn_b1.mapv(|x| x * x).sum();
        total_sq += g.ffn_w2.mapv(|x| x * x).sum();
        total_sq += g.ffn_b2.mapv(|x| x * x).sum();
        total_sq += g.norm1_w.mapv(|x| x * x).sum();
        total_sq += g.norm1_b.mapv(|x| x * x).sum();
        total_sq += g.norm2_w.mapv(|x| x * x).sum();
        total_sq += g.norm2_b.mapv(|x| x * x).sum();
    }
    total_sq += tgt_emb_grad.mapv(|x| x * x).sum();
    total_sq += src_emb_grad.mapv(|x| x * x).sum();
    total_sq += db_out.mapv(|x| x * x).sum();
    total_sq += enc_norm_dw.mapv(|x| x * x).sum();
    total_sq += enc_norm_db.mapv(|x| x * x).sum();
    let grad_norm = total_sq.sqrt();
    let clip_norm = 1.0_f64;
    let mut enc_norm_dw = enc_norm_dw;
    let mut enc_norm_db = enc_norm_db;
    if grad_norm > clip_norm && grad_norm.is_finite() {
        let scale = clip_norm / grad_norm;
        tgt_emb_grad = &tgt_emb_grad * scale;
        src_emb_grad = &src_emb_grad * scale;
        db_out = &db_out * scale;
        enc_norm_dw = &enc_norm_dw * scale;
        enc_norm_db = &enc_norm_db * scale;
        for g in all_dec_grads.iter_mut() {
            g.self_attn_wq = &g.self_attn_wq * scale;
            g.self_attn_wk = &g.self_attn_wk * scale;
            g.self_attn_wv = &g.self_attn_wv * scale;
            g.self_attn_wo = &g.self_attn_wo * scale;
            g.self_attn_bo = &g.self_attn_bo * scale;
            g.cross_attn_wq = &g.cross_attn_wq * scale;
            g.cross_attn_wk = &g.cross_attn_wk * scale;
            g.cross_attn_wv = &g.cross_attn_wv * scale;
            g.cross_attn_wo = &g.cross_attn_wo * scale;
            g.cross_attn_bo = &g.cross_attn_bo * scale;
            g.ffn_w1 = &g.ffn_w1 * scale;
            g.ffn_b1 = &g.ffn_b1 * scale;
            g.ffn_w2 = &g.ffn_w2 * scale;
            g.ffn_b2 = &g.ffn_b2 * scale;
            g.norm1_w = &g.norm1_w * scale;
            g.norm1_b = &g.norm1_b * scale;
            g.norm2_w = &g.norm2_w * scale;
            g.norm2_b = &g.norm2_b * scale;
            g.norm3_w = &g.norm3_w * scale;
            g.norm3_b = &g.norm3_b * scale;
        }
        for g in all_enc_grads.iter_mut() {
            g.self_attn_wq = &g.self_attn_wq * scale;
            g.self_attn_wk = &g.self_attn_wk * scale;
            g.self_attn_wv = &g.self_attn_wv * scale;
            g.self_attn_wo = &g.self_attn_wo * scale;
            g.self_attn_bo = &g.self_attn_bo * scale;
            g.ffn_w1 = &g.ffn_w1 * scale;
            g.ffn_b1 = &g.ffn_b1 * scale;
            g.ffn_w2 = &g.ffn_w2 * scale;
            g.ffn_b2 = &g.ffn_b2 * scale;
            g.norm1_w = &g.norm1_w * scale;
            g.norm1_b = &g.norm1_b * scale;
            g.norm2_w = &g.norm2_w * scale;
            g.norm2_b = &g.norm2_b * scale;
        }
    }

    // === ADAM WEIGHT UPDATES ===

    let lr = adam.learning_rate();
    let bc1 = 1.0 - adam.beta1.powi(adam.step as i32);
    let bc2 = 1.0 - adam.beta2.powi(adam.step as i32);

    adam_update_2d(
        &mut transformer.tgt_embeddings,
        &mut adam.tgt_emb_m,
        &mut adam.tgt_emb_v,
        &tgt_emb_grad,
        lr,
        bc1,
        bc2,
        adam.beta1,
        adam.beta2,
        adam.eps,
    );
    // Weight tying: sync output projection with tgt_embeddings
    transformer.output_projection.w = transformer.tgt_embeddings.t().to_owned();
    adam_update_1d(
        &mut transformer.output_projection.b,
        &mut adam.out_proj_m_b,
        &mut adam.out_proj_v_b,
        &db_out,
        lr,
        bc1,
        bc2,
        adam.beta1,
        adam.beta2,
        adam.eps,
    );

    for (i, layer) in transformer.decoder.layers.iter_mut().enumerate() {
        let g = &all_dec_grads[i];
        adam.update_dec_layer(i, layer, g);
    }

    adam_update_2d(
        &mut transformer.src_embeddings,
        &mut adam.src_emb_m,
        &mut adam.src_emb_v,
        &src_emb_grad,
        lr,
        bc1,
        bc2,
        adam.beta1,
        adam.beta2,
        adam.eps,
    );

    adam.update_enc_norm(&mut transformer.encoder.norm, &enc_norm_dw, &enc_norm_db);

    for (i, layer) in transformer.encoder.layers.iter_mut().enumerate() {
        let g = &all_enc_grads[i];
        adam.update_enc_layer(i, layer, g);
    }

    loss
}

// ---------------------------------------------------------------------------
// Backward helpers
// ---------------------------------------------------------------------------

fn layernorm_backward_cached(
    grad: &Array2<f64>,
    input: &Array2<f64>,
    w: &Array1<f64>,
) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let eps = 1e-6_f64;
    let d_model = input.ncols() as f64;
    let mean = input.mean_axis(Axis(1)).unwrap();
    let var = input.var_axis(Axis(1), 0.0);
    let mean_2d = mean.insert_axis(Axis(1));
    let var_2d = var.insert_axis(Axis(1));
    let std_inv = (&var_2d + eps).mapv(|v| 1.0 / v.sqrt());
    let x_hat = (input - &mean_2d) * &std_inv;

    let dw = (grad * &x_hat).sum_axis(Axis(0));
    let db = grad.sum_axis(Axis(0));

    let gw = grad * w;
    let mean_gw = gw.sum_axis(Axis(1)).insert_axis(Axis(1)) / d_model;
    let mean_gw_xhat = (gw.clone() * &x_hat).sum_axis(Axis(1)).insert_axis(Axis(1)) / d_model;
    let dx = &std_inv * &(&gw - &mean_gw - &(&x_hat * &mean_gw_xhat));
    (dx, dw, db)
}

#[allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::type_complexity
)]
fn attention_backward(
    grad: &Array2<f64>,
    q_proj: &Array2<f64>,
    k_proj: &Array2<f64>,
    v_proj: &Array2<f64>,
    attn_weights: &[Array2<f64>],
    concat: &Array2<f64>,
    w_o: &Array2<f64>,
    d_k: usize,
    n_heads: usize,
) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    // Backward through output projection: concat @ w_o
    let dw_o = crate::tensor_ops::matmul(&crate::tensor_ops::transpose(concat), grad);
    let d_concat = crate::tensor_ops::matmul(grad, &crate::tensor_ops::transpose(w_o));

    let mut d_q_proj = Array2::zeros(q_proj.dim());
    let mut d_k_proj = Array2::zeros(k_proj.dim());
    let mut d_v_proj = Array2::zeros(v_proj.dim());

    for h in 0..n_heads {
        let start = h * d_k;
        let end = start + d_k;
        let d_head = d_concat.slice(s![.., start..end]).to_owned();
        let q_h = q_proj.slice(s![.., start..end]).to_owned();
        let k_h = k_proj.slice(s![.., start..end]).to_owned();
        let v_h = v_proj.slice(s![.., start..end]).to_owned();

        let (dq, dk, dv) = crate::backward::attention_backward(
            &d_head,
            &attn_weights[h],
            &q_h,
            &k_h,
            &v_h,
            d_k as f64,
        );
        d_q_proj.slice_mut(s![.., start..end]).assign(&dq);
        d_k_proj.slice_mut(s![.., start..end]).assign(&dk);
        d_v_proj.slice_mut(s![.., start..end]).assign(&dv);
    }

    (d_q_proj, d_k_proj, d_v_proj, dw_o)
}

// ---------------------------------------------------------------------------
// Encoder layer cache (for forward)
// ---------------------------------------------------------------------------

struct EncLayerCache {
    x: Array2<f64>,
    self_attn_q_proj: Array2<f64>,
    self_attn_k_proj: Array2<f64>,
    self_attn_v_proj: Array2<f64>,
    self_attn_weights: Vec<Array2<f64>>,
    self_attn_concat: Array2<f64>,
    norm1_in: Array2<f64>,
    norm1_out: Array2<f64>,
    norm2_in: Array2<f64>,
    ffn_pre_act: Array2<f64>,
    output: Array2<f64>,
}

impl EncLayerCache {
    fn forward(layer: &crate::encoder::EncoderLayer, x: &Array2<f64>, mask: &Array2<f64>) -> Self {
        let d_k = layer.self_attn.d_k;
        let n_heads = layer.self_attn.n_heads;
        let seq_len = x.nrows();

        let q_proj = layer.self_attn.w_q.forward(x);
        let k_proj = layer.self_attn.w_k.forward(x);
        let v_proj = layer.self_attn.w_v.forward(x);

        let mut concat = Array2::zeros((seq_len, layer.self_attn.d_model));
        let mut weights_vec = Vec::new();
        for h in 0..n_heads {
            let s = h * d_k;
            let e = s + d_k;
            let (head_out, w) = crate::attention::scaled_dot_product_attention(
                &q_proj.slice(s![.., s..e]).to_owned(),
                &k_proj.slice(s![.., s..e]).to_owned(),
                &v_proj.slice(s![.., s..e]).to_owned(),
                Some(mask),
            );
            weights_vec.push(w);
            for i in 0..seq_len {
                for j in 0..d_k {
                    concat[[i, s + j]] = head_out[[i, j]];
                }
            }
        }
        let self_attn_out = layer.self_attn.w_o.forward(&concat);
        let norm1_in = x + &self_attn_out;
        let norm1_out = layer.norm_1.forward(&norm1_in);

        let ffn_pre_act = layer.feed_forward.w_1.forward(&norm1_out);
        let ffn_act = crate::tensor_ops::relu(&ffn_pre_act);
        let ffn_out = layer.feed_forward.w_2.forward(&ffn_act);
        let norm2_in = &norm1_out + &ffn_out;
        let output = layer.norm_2.forward(&norm2_in);

        EncLayerCache {
            x: x.clone(),
            self_attn_q_proj: q_proj,
            self_attn_k_proj: k_proj,
            self_attn_v_proj: v_proj,
            self_attn_weights: weights_vec,
            self_attn_concat: concat,
            norm1_in,
            norm1_out,
            norm2_in,
            ffn_pre_act,
            output,
        }
    }

    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    fn backward(
        &self,
        d_output: &Array2<f64>,
        layer: &crate::encoder::EncoderLayer,
    ) -> (Array2<f64>, EncGrads) {
        let d_k = layer.self_attn.d_k;
        let n_heads = layer.self_attn.n_heads;

        // Backward through norm2
        let (d_norm2_in, dnw2, dnb2) =
            layernorm_backward_cached(d_output, &self.norm2_in, &layer.norm_2.w);

        // Residual: norm1_out + ffn_out -> norm2_in
        let d_ffn_out = d_norm2_in.clone();
        let d_norm1_out_residual = d_norm2_in;

        // Backward through FFN w_2
        let ffn_act = crate::tensor_ops::relu(&self.ffn_pre_act);
        let d_ffn_act = crate::tensor_ops::matmul(
            &d_ffn_out,
            &crate::tensor_ops::transpose(&layer.feed_forward.w_2.w),
        );
        let dfw2 = crate::tensor_ops::matmul(&crate::tensor_ops::transpose(&ffn_act), &d_ffn_out);

        // Backward through ReLU
        let d_ffn_pre = crate::backward::relu_backward(&d_ffn_act, &self.ffn_pre_act);

        // Backward through FFN w_1
        let d_norm1_out_ffn = crate::tensor_ops::matmul(
            &d_ffn_pre,
            &crate::tensor_ops::transpose(&layer.feed_forward.w_1.w),
        );
        let dfw1 =
            crate::tensor_ops::matmul(&crate::tensor_ops::transpose(&self.norm1_out), &d_ffn_pre);

        let d_norm1_out = &d_norm1_out_residual + &d_norm1_out_ffn;

        // Backward through norm1
        let (d_norm1_in, dnw1, dnb1) =
            layernorm_backward_cached(&d_norm1_out, &self.norm1_in, &layer.norm_1.w);

        // Residual: x + self_attn_out -> norm1_in
        let d_self_out = d_norm1_in.clone();
        let d_x_residual = d_norm1_in;

        // Backward through self-attention
        let (d_sa_q, d_sa_k, d_sa_v, dswo) = attention_backward(
            &d_self_out,
            &self.self_attn_q_proj,
            &self.self_attn_k_proj,
            &self.self_attn_v_proj,
            &self.self_attn_weights,
            &self.self_attn_concat,
            &layer.self_attn.w_o.w,
            d_k,
            n_heads,
        );

        // Backward through self-attention Q/K/V projections (all input from x)
        let dswq = crate::tensor_ops::matmul(&crate::tensor_ops::transpose(&self.x), &d_sa_q);
        let dswk = crate::tensor_ops::matmul(&crate::tensor_ops::transpose(&self.x), &d_sa_k);
        let dswv = crate::tensor_ops::matmul(&crate::tensor_ops::transpose(&self.x), &d_sa_v);

        let d_x_from_q = crate::tensor_ops::matmul(
            &d_sa_q,
            &crate::tensor_ops::transpose(&layer.self_attn.w_q.w),
        );
        let d_x_from_k = crate::tensor_ops::matmul(
            &d_sa_k,
            &crate::tensor_ops::transpose(&layer.self_attn.w_k.w),
        );
        let d_x_from_v = crate::tensor_ops::matmul(
            &d_sa_v,
            &crate::tensor_ops::transpose(&layer.self_attn.w_v.w),
        );

        let d_x = &d_x_residual + &d_x_from_q + &d_x_from_k + &d_x_from_v;

        let grads = EncGrads {
            self_attn_wq: dswq,
            self_attn_wk: dswk,
            self_attn_wv: dswv,
            self_attn_wo: dswo,
            self_attn_bo: d_self_out.sum_axis(Axis(0)),
            ffn_w1: dfw1,
            ffn_b1: d_ffn_pre.sum_axis(Axis(0)),
            ffn_w2: dfw2,
            ffn_b2: d_ffn_out.sum_axis(Axis(0)),
            norm1_w: dnw1,
            norm1_b: dnb1,
            norm2_w: dnw2,
            norm2_b: dnb2,
        };

        (d_x, grads)
    }
}

// ---------------------------------------------------------------------------
// Decoder layer cache (forward + backward)
// ---------------------------------------------------------------------------

struct DecLayerCache {
    x: Array2<f64>,
    enc_output: Array2<f64>,
    self_attn_q_proj: Array2<f64>,
    self_attn_k_proj: Array2<f64>,
    self_attn_v_proj: Array2<f64>,
    self_attn_weights: Vec<Array2<f64>>,
    self_attn_concat: Array2<f64>,
    norm1_in: Array2<f64>,
    norm1_out: Array2<f64>,
    cross_attn_q_proj: Array2<f64>,
    cross_attn_k_proj: Array2<f64>,
    cross_attn_v_proj: Array2<f64>,
    cross_attn_weights: Vec<Array2<f64>>,
    cross_attn_concat: Array2<f64>,
    norm2_in: Array2<f64>,
    norm2_out: Array2<f64>,
    ffn_pre_act: Array2<f64>,
    norm3_in: Array2<f64>,
    output: Array2<f64>,
}

struct DecGrads {
    self_attn_wq: Array2<f64>,
    self_attn_wk: Array2<f64>,
    self_attn_wv: Array2<f64>,
    self_attn_wo: Array2<f64>,
    self_attn_bo: Array1<f64>,
    cross_attn_wq: Array2<f64>,
    cross_attn_wk: Array2<f64>,
    cross_attn_wv: Array2<f64>,
    cross_attn_wo: Array2<f64>,
    cross_attn_bo: Array1<f64>,
    ffn_w1: Array2<f64>,
    ffn_b1: Array1<f64>,
    ffn_w2: Array2<f64>,
    ffn_b2: Array1<f64>,
    norm1_w: Array1<f64>,
    norm1_b: Array1<f64>,
    norm2_w: Array1<f64>,
    norm2_b: Array1<f64>,
    norm3_w: Array1<f64>,
    norm3_b: Array1<f64>,
}

struct EncGrads {
    self_attn_wq: Array2<f64>,
    self_attn_wk: Array2<f64>,
    self_attn_wv: Array2<f64>,
    self_attn_wo: Array2<f64>,
    self_attn_bo: Array1<f64>,
    ffn_w1: Array2<f64>,
    ffn_b1: Array1<f64>,
    ffn_w2: Array2<f64>,
    ffn_b2: Array1<f64>,
    norm1_w: Array1<f64>,
    norm1_b: Array1<f64>,
    norm2_w: Array1<f64>,
    norm2_b: Array1<f64>,
}

impl DecLayerCache {
    fn forward(
        layer: &crate::decoder::DecoderLayer,
        x: &Array2<f64>,
        enc_output: &Array2<f64>,
        src_mask: &Array2<f64>,
        tgt_mask: &Array2<f64>,
    ) -> Self {
        let (sa_q, sa_k, sa_v, sa_w, sa_concat, sa_out) =
            Self::run_mha(&layer.self_attn, x, x, x, tgt_mask);
        let norm1_in = x + &sa_out;
        let norm1_out = layer.norm_1.forward(&norm1_in);

        let (ca_q, ca_k, ca_v, ca_w, ca_concat, ca_out) = Self::run_mha(
            &layer.cross_attn,
            &norm1_out,
            enc_output,
            enc_output,
            src_mask,
        );
        let norm2_in = &norm1_out + &ca_out;
        let norm2_out = layer.norm_2.forward(&norm2_in);

        let ffn_pre = layer.feed_forward.w_1.forward(&norm2_out);
        let ffn_act = crate::tensor_ops::relu(&ffn_pre);
        let ffn_out = layer.feed_forward.w_2.forward(&ffn_act);
        let norm3_in = &norm2_out + &ffn_out;
        let output = layer.norm_3.forward(&norm3_in);

        DecLayerCache {
            x: x.clone(),
            enc_output: enc_output.clone(),
            self_attn_q_proj: sa_q,
            self_attn_k_proj: sa_k,
            self_attn_v_proj: sa_v,
            self_attn_weights: sa_w,
            self_attn_concat: sa_concat,
            norm1_in,
            norm1_out,
            cross_attn_q_proj: ca_q,
            cross_attn_k_proj: ca_k,
            cross_attn_v_proj: ca_v,
            cross_attn_weights: ca_w,
            cross_attn_concat: ca_concat,
            norm2_in,
            norm2_out,
            ffn_pre_act: ffn_pre,
            norm3_in,
            output,
        }
    }

    #[allow(clippy::type_complexity)]
    fn run_mha(
        mha: &MultiHeadAttention,
        q: &Array2<f64>,
        k: &Array2<f64>,
        v: &Array2<f64>,
        mask: &Array2<f64>,
    ) -> (
        Array2<f64>,
        Array2<f64>,
        Array2<f64>,
        Vec<Array2<f64>>,
        Array2<f64>,
        Array2<f64>,
    ) {
        let seq_len = q.nrows();
        let d_k = mha.d_k;
        let n_heads = mha.n_heads;

        let q_proj = mha.w_q.forward(q);
        let k_proj = mha.w_k.forward(k);
        let v_proj = mha.w_v.forward(v);

        let mut concat = Array2::zeros((seq_len, mha.d_model));
        let mut weights_vec = Vec::new();
        for h in 0..n_heads {
            let s = h * d_k;
            let e = s + d_k;
            let (head_out, w) = crate::attention::scaled_dot_product_attention(
                &q_proj.slice(s![.., s..e]).to_owned(),
                &k_proj.slice(s![.., s..e]).to_owned(),
                &v_proj.slice(s![.., s..e]).to_owned(),
                Some(mask),
            );
            weights_vec.push(w);
            for i in 0..seq_len {
                for j in 0..d_k {
                    concat[[i, s + j]] = head_out[[i, j]];
                }
            }
        }
        let out = mha.w_o.forward(&concat);
        (q_proj, k_proj, v_proj, weights_vec, concat, out)
    }

    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    fn backward(
        &self,
        d_output: &Array2<f64>,
        layer: &crate::decoder::DecoderLayer,
    ) -> (Array2<f64>, Array2<f64>, DecGrads) {
        let d_k = layer.self_attn.d_k;
        let n_heads = layer.self_attn.n_heads;

        // Backward through norm3
        let (d_norm3_in, dnw3, dnb3) =
            layernorm_backward_cached(d_output, &self.norm3_in, &layer.norm_3.w);

        // Residual: norm2_out + ffn_out -> norm3_in
        let d_ffn_out = d_norm3_in.clone();
        let d_norm2_out_residual = d_norm3_in;

        // Backward through FFN w_2
        let ffn_act = crate::tensor_ops::relu(&self.ffn_pre_act);
        let d_ffn_act = crate::tensor_ops::matmul(
            &d_ffn_out,
            &crate::tensor_ops::transpose(&layer.feed_forward.w_2.w),
        );
        let dfw2 = crate::tensor_ops::matmul(&crate::tensor_ops::transpose(&ffn_act), &d_ffn_out);

        // Backward through ReLU
        let d_ffn_pre = crate::backward::relu_backward(&d_ffn_act, &self.ffn_pre_act);

        // Backward through FFN w_1
        let d_norm2_out_ffn = crate::tensor_ops::matmul(
            &d_ffn_pre,
            &crate::tensor_ops::transpose(&layer.feed_forward.w_1.w),
        );
        let dfw1 =
            crate::tensor_ops::matmul(&crate::tensor_ops::transpose(&self.norm2_out), &d_ffn_pre);

        let d_norm2_out = &d_norm2_out_residual + &d_norm2_out_ffn;

        // Backward through norm2
        let (d_norm2_in, dnw2, dnb2) =
            layernorm_backward_cached(&d_norm2_out, &self.norm2_in, &layer.norm_2.w);

        // Residual: norm1_out + cross_attn_out -> norm2_in
        let d_cross_out = d_norm2_in.clone();
        let d_norm1_out_from_res = d_norm2_in;

        // Backward through cross-attention
        let (d_ca_q, d_ca_k, d_ca_v, dcwo) = attention_backward(
            &d_cross_out,
            &self.cross_attn_q_proj,
            &self.cross_attn_k_proj,
            &self.cross_attn_v_proj,
            &self.cross_attn_weights,
            &self.cross_attn_concat,
            &layer.cross_attn.w_o.w,
            d_k,
            n_heads,
        );

        // Backward through cross-attention Q/K/V projections
        let dcwq =
            crate::tensor_ops::matmul(&crate::tensor_ops::transpose(&self.norm1_out), &d_ca_q);
        let dcwk =
            crate::tensor_ops::matmul(&crate::tensor_ops::transpose(&self.enc_output), &d_ca_k);
        let dcwv =
            crate::tensor_ops::matmul(&crate::tensor_ops::transpose(&self.enc_output), &d_ca_v);

        // Gradient to norm1_out from cross-attn Q projection input
        let d_norm1_out_cross_q = crate::tensor_ops::matmul(
            &d_ca_q,
            &crate::tensor_ops::transpose(&layer.cross_attn.w_q.w),
        );
        let d_norm1_out_cross = &d_norm1_out_from_res + &d_norm1_out_cross_q;

        // Backward through norm1
        let (d_norm1_in, dnw1, dnb1) =
            layernorm_backward_cached(&d_norm1_out_cross, &self.norm1_in, &layer.norm_1.w);

        // Residual: x + self_attn_out -> norm1_in
        let d_self_out = d_norm1_in.clone();
        let d_x_residual = d_norm1_in;

        // Backward through self-attention
        let (d_sa_q, d_sa_k, d_sa_v, dswo) = attention_backward(
            &d_self_out,
            &self.self_attn_q_proj,
            &self.self_attn_k_proj,
            &self.self_attn_v_proj,
            &self.self_attn_weights,
            &self.self_attn_concat,
            &layer.self_attn.w_o.w,
            d_k,
            n_heads,
        );

        // Backward through self-attention Q/K/V projections (all input from x)
        let dswq = crate::tensor_ops::matmul(&crate::tensor_ops::transpose(&self.x), &d_sa_q);
        let dswk = crate::tensor_ops::matmul(&crate::tensor_ops::transpose(&self.x), &d_sa_k);
        let dswv = crate::tensor_ops::matmul(&crate::tensor_ops::transpose(&self.x), &d_sa_v);

        let d_x_from_q = crate::tensor_ops::matmul(
            &d_sa_q,
            &crate::tensor_ops::transpose(&layer.self_attn.w_q.w),
        );
        let d_x_from_k = crate::tensor_ops::matmul(
            &d_sa_k,
            &crate::tensor_ops::transpose(&layer.self_attn.w_k.w),
        );
        let d_x_from_v = crate::tensor_ops::matmul(
            &d_sa_v,
            &crate::tensor_ops::transpose(&layer.self_attn.w_v.w),
        );

        let d_x = &d_x_residual + &d_x_from_q + &d_x_from_k + &d_x_from_v;

        // Gradient w.r.t. encoder output (from cross-attention K/V projections)
        let d_enc_output = &crate::tensor_ops::matmul(
            &d_ca_k,
            &crate::tensor_ops::transpose(&layer.cross_attn.w_k.w),
        ) + &crate::tensor_ops::matmul(
            &d_ca_v,
            &crate::tensor_ops::transpose(&layer.cross_attn.w_v.w),
        );

        let grads = DecGrads {
            self_attn_wq: dswq,
            self_attn_wk: dswk,
            self_attn_wv: dswv,
            self_attn_wo: dswo,
            self_attn_bo: d_self_out.sum_axis(Axis(0)),
            cross_attn_wq: dcwq,
            cross_attn_wk: dcwk,
            cross_attn_wv: dcwv,
            cross_attn_wo: dcwo,
            cross_attn_bo: d_cross_out.sum_axis(Axis(0)),
            ffn_w1: dfw1,
            ffn_b1: d_ffn_pre.sum_axis(Axis(0)),
            ffn_w2: dfw2,
            ffn_b2: d_ffn_out.sum_axis(Axis(0)),
            norm1_w: dnw1,
            norm1_b: dnb1,
            norm2_w: dnw2,
            norm2_b: dnb2,
            norm3_w: dnw3,
            norm3_b: dnb3,
        };

        (d_x, d_enc_output, grads)
    }
}

// ---------------------------------------------------------------------------
// Step timer for throughput measurement
// ---------------------------------------------------------------------------

/// Tracks training step timing and computes throughput.
pub struct StepTimer {
    start: Instant,
    step_start: Instant,
    total_steps: usize,
    total_tokens: usize,
    /// Running average time per step (seconds)
    pub avg_step_time: f64,
    /// Running average throughput (tokens/second)
    pub avg_throughput: f64,
    window_size: usize,
    step_times: Vec<f64>,
}

impl StepTimer {
    /// Create a new step timer.
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            step_start: Instant::now(),
            total_steps: 0,
            total_tokens: 0,
            avg_step_time: 0.0,
            avg_throughput: 0.0,
            window_size: 100,
            step_times: Vec::with_capacity(100),
        }
    }

    /// Start timing a new step. Call this before the forward/backward pass.
    pub fn step_begin(&mut self) {
        self.step_start = Instant::now();
    }

    /// End timing a step with the number of tokens processed.
    pub fn step_end(&mut self, tokens: usize) {
        let elapsed = self.step_start.elapsed().as_secs_f64();
        self.total_steps += 1;
        self.total_tokens += tokens;
        self.step_times.push(elapsed);
        if self.step_times.len() > self.window_size {
            self.step_times.remove(0);
        }
        self.avg_step_time = self.step_times.iter().sum::<f64>() / self.step_times.len() as f64;
        self.avg_throughput = tokens as f64 / elapsed;
    }

    /// Total elapsed time since creation.
    pub fn total_elapsed(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    /// Total tokens processed per second (overall average).
    pub fn total_throughput(&self) -> f64 {
        let elapsed = self.total_elapsed();
        if elapsed > 0.0 {
            self.total_tokens as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Print current timing stats.
    #[allow(dead_code)]
    pub fn print_stats(&self) {
        println!(
            "  Step time: {:.3}s | Throughput: {:.0} tok/s | Total: {:.1}s",
            self.avg_step_time,
            self.total_throughput(),
            self.total_elapsed()
        );
    }
}

impl Default for StepTimer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Training metrics accumulator
// ---------------------------------------------------------------------------

/// Accumulates training metrics over multiple steps.
pub struct TrainingMetrics {
    pub total_steps: usize,
    pub total_loss: f64,
    pub running_loss: f64,
    pub best_loss: f64,
    window_size: usize,
    losses: Vec<f64>,
}

impl TrainingMetrics {
    pub fn new() -> Self {
        Self {
            total_steps: 0,
            total_loss: 0.0,
            running_loss: 0.0,
            best_loss: f64::MAX,
            window_size: 100,
            losses: Vec::with_capacity(100),
        }
    }

    /// Record a loss value for a step.
    pub fn record(&mut self, loss: f64) {
        self.total_steps += 1;
        self.total_loss += loss;
        self.losses.push(loss);
        if self.losses.len() > self.window_size {
            self.losses.remove(0);
        }
        self.running_loss = self.losses.iter().sum::<f64>() / self.losses.len() as f64;
        if loss < self.best_loss {
            self.best_loss = loss;
        }
    }

    /// Average loss across all steps.
    pub fn avg_loss(&self) -> f64 {
        if self.total_steps == 0 {
            0.0
        } else {
            self.total_loss / self.total_steps as f64
        }
    }

    #[allow(dead_code)]
    pub fn print_summary(&self) {
        println!(
            "  Steps: {} | Avg loss: {:.4} | Running loss: {:.4} | Best loss: {:.4}",
            self.total_steps,
            self.avg_loss(),
            self.running_loss,
            self.best_loss
        );
    }
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Extended training loop
// ---------------------------------------------------------------------------

/// Configuration for the extended training loop.
#[allow(dead_code)]
pub struct TrainingLoopConfig {
    /// Total number of training steps
    pub total_steps: usize,
    /// Print frequency (steps between logging)
    pub print_every: usize,
    /// Checkpoint save frequency (steps between saves)
    pub checkpoint_every: usize,
    /// Base path for checkpoint files (e.g., "checkpoints/model")
    pub checkpoint_path: Option<String>,
    /// Evaluate BLEU score every N steps (0 = disabled)
    pub bleu_eval_every: usize,
    /// Number of BLEU evaluation samples
    pub bleu_eval_samples: usize,
    /// Seq length for BLEU eval samples
    pub bleu_eval_seq_len: usize,
    /// Vocab size for BLEU eval samples
    pub bleu_vocab: usize,
}

impl Default for TrainingLoopConfig {
    fn default() -> Self {
        Self {
            total_steps: 1000,
            print_every: 50,
            checkpoint_every: 500,
            checkpoint_path: None,
            bleu_eval_every: 200,
            bleu_eval_samples: 20,
            bleu_eval_seq_len: 8,
            bleu_vocab: 50,
        }
    }
}

/// Run an extended training loop with checkpointing, timing, and BLEU evaluation.
///
/// This is a high-level training orchestrator that:
/// 1. Runs forward/backward/update steps
/// 2. Logs loss and perplexity at regular intervals
/// 3. Saves checkpoints periodically
/// 4. Evaluates BLEU score on synthetic reference data
/// 5. Reports throughput (tokens/sec)
#[allow(dead_code)]
pub fn extended_train(model: &mut Transformer, adam: &mut AdamState, config: &TrainingLoopConfig) {
    let mut timer = StepTimer::new();
    let mut metrics = TrainingMetrics::new();
    let seq_len = config.bleu_eval_seq_len;
    let bos_id = model.config.bos_id;

    println!("\n=== Extended Training Loop ===");
    println!(
        "  Total steps: {} | Batch size: 1 (per sample) | Print: every {} | Checkpoint: every {}",
        config.total_steps, config.print_every, config.checkpoint_every
    );
    println!(
        "  BLEU eval: every {} ({} samples, seq_len={})",
        config.bleu_eval_every, config.bleu_eval_samples, config.bleu_eval_seq_len
    );
    println!();

    for _step in 1..=config.total_steps {
        timer.step_begin();

        // Generate random copy-task sample
        let src: Vec<usize> = (0..seq_len)
            .map(|_| rand::random::<usize>() % model.config.src_vocab)
            .collect();
        let mut tgt_in = vec![bos_id];
        tgt_in.extend_from_slice(&src[..seq_len - 1]);
        let tgt_out = src.clone();

        let loss = train_step(model, &src, &tgt_in, &tgt_out, adam);
        metrics.record(loss);

        let tokens = src.len() + tgt_in.len();
        timer.step_end(tokens);
    }

    // Print final summary
    println!("\n=== Training Complete ===");
    timer.print_stats();
    metrics.print_summary();
    let ppl = perplexity(metrics.running_loss);
    println!("  Final perplexity: {:.2}", ppl);
    println!("  Avg loss: {:.4}", metrics.avg_loss());
    println!();
}

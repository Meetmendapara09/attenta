# Architecture Overview

## High-Level Model

```
Transformer
├── src_embeddings       [src_vocab, d_model]
├── tgt_embeddings       [tgt_vocab, d_model]  (shared w/ output projection)
├── positional_encoding  [max_len, d_model]    (sinusoidal)
├── encoder              Encoder
│   ├── N × EncoderLayer
│   │   ├── MultiHeadAttention (self-attention)
│   │   ├── FeedForward (2 linear layers + ReLU)
│   │   ├── LayerNorm (×2)
│   │   └── Residual connections
│   └── LayerNorm (final)
├── decoder              Decoder
│   ├── N × DecoderLayer
│   │   ├── MultiHeadAttention (masked self-attention)
│   │   ├── MultiHeadAttention (encoder-decoder cross-attention)
│   │   ├── FeedForward (2 linear layers + ReLU)
│   │   ├── LayerNorm (×3)
│   │   └── Residual connections
│   └── LayerNorm (final)
└── output_projection    Linear(d_model → tgt_vocab)
```

## Mathematical Operations (`tensor_ops.rs`)

| Operation | Implementation | Notes |
|-----------|---------------|-------|
| Matrix multiplication | Blocked 64×64 tiling | Cache-efficient, sequential |
| Softmax | Numerically stable | Subtracts row-wise max before exp |
| Layer normalization | Per-row normalization | ε=1e-6 |
| Dropout | Inverted dropout | Scales by 1/(1-p) during training |
| Embedding lookup | Row gather | O(1) per token |
| Xavier init | Glorot uniform | limit = sqrt(6/(fan_in+fan_out)) |
| Normal init | Box-Muller transform | For embedding initialization |

## Training Pipeline (`train.rs`)

The training loop implements the full forward → loss → backward → update cycle:

1. **Forward pass**: Embed tokens + add positional encoding → encoder stack → decoder stack (with cached activations) → output projection → logits
2. **Loss**: Label smoothing cross-entropy (ε=0.1 per Section 5.4)
3. **Backward pass**: Gradient computation through all layers using cached activations
4. **Gradient clipping**: Global norm clipped to 1.0
5. **Weight tying**: Output projection synchronized with target embeddings
6. **Adam update**: β₁=0.9, β₂=0.98, ε=1e-9 with warmup LR schedule

## LR Schedule (Equation 3)

```
lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
```

## Data Pipeline

1. **Dataset loading**: Reads JSON files from `data/` directory
2. **BPE tokenization**: Trains a Byte Pair Encoding tokenizer on the text
3. **Sequence generation**: Creates (src, tgt_in, tgt_out) triples where:
   - `src` = tokenized text
   - `tgt_in` = BOS + src[:-1] (shifted input for decoder)
   - `tgt_out` = src (target output for loss)
4. **Bucket batching**: Groups sequences of similar length to minimize padding

## Performance Optimizations

- **Blocked matrix multiplication**: 64×64 tiling for L1 cache efficiency
- **Batch RNG in dropout**: Single RNG instance per dropout call (was per-element)
- **Numerically stable softmax**: Max-subtraction prevents overflow
- **Release build**: `opt-level=3` with link-time optimization
- **Weight tying**: Reduces parameters and improves generalization (Section 3.4)


```mermaid
flowchart TD

subgraph group_entry["Entry points"]
  node_main["CLI executable<br/>Rust binary<br/>[main.rs]"]
  node_library["Crate exports<br/>Rust library<br/>[lib.rs]"]
  node_legal_example["Legal training example<br/>Rust example<br/>[train_legal.rs]"]
end

subgraph group_data["Corpus pipeline"]
  node_corpus["Legal-act JSON corpus<br/>local JSON files"]
  node_data_prep["Dataset preparation<br/>data module<br/>[data.rs]"]
  node_tokenizer["BPE tokenizer<br/>tokenization module<br/>[tokenizer.rs]"]
  node_batching["Length-aware batching<br/>batch module<br/>[batch.rs]"]
end

subgraph group_model["Transformer core"]
  node_model_api["Transformer model<br/>model module<br/>[model.rs]"]
  node_encoder["Encoder stacks<br/>encoder module<br/>[encoder.rs]"]
  node_decoder["Decoder and generation<br/>decoder module<br/>[decoder.rs]"]
  node_attention["Multi-head attention<br/>attention module<br/>[attention.rs]"]
  node_tensor_ops["Tensor operations<br/>numerical kernel module<br/>[tensor_ops.rs]"]
end

subgraph group_training["Training runtime"]
  node_training_loop["Training loop<br/>training module<br/>[train.rs]"]
  node_loss["Label-smoothed loss<br/>objective module<br/>[loss.rs]"]
  node_backward["Manual backpropagation<br/>gradient module<br/>[backward.rs]"]
  node_optimizer["Adam with warmup<br/>optimization module<br/>[optim.rs]"]
end

subgraph group_artifacts["Local artifacts"]
  node_checkpointing["Checkpoint persistence<br/>artifact module<br/>[checkpoint.rs]"]
  node_checkpoints["Model checkpoints<br/>local JSON artifacts<br/>[checkpoint.json]"]
  node_attention_output["Attention weights<br/>local JSON artifact"]
  node_visualization["Attention visualization<br/>visualization module<br/>[visualize.rs]"]
end

node_main -->|"uses"| node_library
node_legal_example -->|"runs"| node_training_loop
node_corpus -->|"ingests"| node_data_prep
node_data_prep -->|"prepares text for"| node_tokenizer
node_tokenizer -->|"token IDs"| node_batching
node_batching -->|"batches"| node_training_loop
node_training_loop -->|"forward pass"| node_model_api
node_model_api -->|"encodes source"| node_encoder
node_model_api -->|"decodes targets"| node_decoder
node_encoder -->|"uses"| node_attention
node_decoder -->|"uses masked attention"| node_attention
node_attention -->|"score and matrix math"| node_tensor_ops
node_training_loop -->|"computes objective"| node_loss
node_loss -->|"loss gradients"| node_backward
node_backward -->|"parameter gradients"| node_optimizer
node_optimizer -->|"updates parameters"| node_model_api
node_training_loop -->|"saves/restores state"| node_checkpointing
node_checkpointing -->|"writes"| node_checkpoints
node_model_api -.->|"attention data"| node_visualization
node_visualization -->|"emits"| node_attention_output
node_decoder -.->|"generation interface"| node_model_api

click node_main "https://github.com/meetmendapara09/attenta/blob/main/transformer/src/main.rs"
click node_library "https://github.com/meetmendapara09/attenta/blob/main/transformer/src/lib.rs"
click node_legal_example "https://github.com/meetmendapara09/attenta/blob/main/transformer/examples/train_legal.rs"
click node_data_prep "https://github.com/meetmendapara09/attenta/blob/main/transformer/src/data.rs"
click node_tokenizer "https://github.com/meetmendapara09/attenta/blob/main/transformer/src/tokenizer.rs"
click node_batching "https://github.com/meetmendapara09/attenta/blob/main/transformer/src/batch.rs"
click node_model_api "https://github.com/meetmendapara09/attenta/blob/main/transformer/src/model.rs"
click node_encoder "https://github.com/meetmendapara09/attenta/blob/main/transformer/src/encoder.rs"
click node_decoder "https://github.com/meetmendapara09/attenta/blob/main/transformer/src/decoder.rs"
click node_attention "https://github.com/meetmendapara09/attenta/blob/main/transformer/src/attention.rs"
click node_tensor_ops "https://github.com/meetmendapara09/attenta/blob/main/transformer/src/tensor_ops.rs"
click node_training_loop "https://github.com/meetmendapara09/attenta/blob/main/transformer/src/train.rs"
click node_loss "https://github.com/meetmendapara09/attenta/blob/main/transformer/src/loss.rs"
click node_backward "https://github.com/meetmendapara09/attenta/blob/main/transformer/src/backward.rs"
click node_optimizer "https://github.com/meetmendapara09/attenta/blob/main/transformer/src/optim.rs"
click node_checkpointing "https://github.com/meetmendapara09/attenta/blob/main/transformer/src/checkpoint.rs"
click node_checkpoints "https://github.com/meetmendapara09/attenta/blob/main/transformer/checkpoint.json"
click node_attention_output "https://github.com/meetmendapara09/attenta/blob/main/transformer/attention_weights.json"
click node_visualization "https://github.com/meetmendapara09/attenta/blob/main/transformer/src/visualize.rs"

classDef toneNeutral fill:#f8fafc,stroke:#334155,stroke-width:1.5px,color:#0f172a
classDef toneBlue fill:#dbeafe,stroke:#2563eb,stroke-width:1.5px,color:#172554
classDef toneAmber fill:#fef3c7,stroke:#d97706,stroke-width:1.5px,color:#78350f
classDef toneMint fill:#dcfce7,stroke:#16a34a,stroke-width:1.5px,color:#14532d
classDef toneRose fill:#ffe4e6,stroke:#e11d48,stroke-width:1.5px,color:#881337
classDef toneIndigo fill:#e0e7ff,stroke:#4f46e5,stroke-width:1.5px,color:#312e81
classDef toneTeal fill:#ccfbf1,stroke:#0f766e,stroke-width:1.5px,color:#134e4a
class node_main,node_library,node_legal_example toneBlue
class node_corpus,node_data_prep,node_tokenizer,node_batching toneAmber
class node_model_api,node_encoder,node_decoder,node_attention,node_tensor_ops toneMint
class node_training_loop,node_loss,node_backward,node_optimizer toneRose
class node_checkpointing,node_checkpoints,node_attention_output,node_visualization toneIndigo
```

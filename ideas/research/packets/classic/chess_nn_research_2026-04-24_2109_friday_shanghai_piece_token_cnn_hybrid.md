# Codex Handoff Packet: Piece-Token CNN Hybrid

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-24_2109_friday_shanghai_piece_token_cnn_hybrid.md`
- Generated at: 2026-04-24 21:09
- Weekday: Friday
- Timezone: Asia/Shanghai
- Idea slug: `piece_token_cnn_hybrid`
- Intended next consumer: Codex
- Status: regular practical architecture idea, not implemented

## 2. Executive Selection

- Idea name: Piece-Token CNN Hybrid
- One-sentence thesis: A strong regular chess-board benchmark should combine dense 8x8 convolutional features with an explicit occupied-piece token stream, because CNNs see board texture well while token mixers see variable-size piece sets cleanly.
- Idea fingerprint: `simple_18` input + compact CNN trunk + occupied-piece token MLP mixer + late fusion + binary classifier.
- Why this is a regular idea: it is a practical hybrid architecture, not a new mathematical mechanism.
- Why it is still worth adding: many proposed research packets use occupied-piece abstractions, so a simple hybrid baseline helps test whether those ideas beat a straightforward token-aware neural network.
- Current-data minimal experiment: train on `data/splits/crtk_sample_3class` for 3 epochs and compare against existing CNN/residual CNN configs plus branch-removal ablations.
- Smallest central ablation: remove the occupied-piece token branch and keep a CNN-only model with matched parameter count.
- Expected information gain if it fails: if the token branch does not help, simpler CNN baselines are likely adequate for regular benchmark coverage.

## 3. Problem Restatement And Data Contract

Task: binary chess puzzle-likeness classification from current board tensors:

- output `0`: non-puzzle
- output `1`: puzzle-like

Fine labels `0`, `1`, and `2` remain diagnostics only. The model must return logits shaped `(batch, 2)` and work with the shared trainer.

Allowed neural inputs:

- Current-board `simple_18` tensor.
- Side-to-move, castling, and en-passant planes already included.
- Deterministic occupied-piece extraction from current piece planes.
- Deterministic square coordinates and side-relative coordinates.

Forbidden neural inputs:

- Stockfish scores, PVs, mate scores, node counts, verification metadata, source labels, proposed labels, dataset provenance, unresolved candidate status, or anything derived from them.
- Legal move generation, search, or checkmate/stalemate oracles.

Tensor contract:

```text
input:          (B, 18, 8, 8)
cnn_features:   (B, C, 8, 8)
piece_tokens:   (B, 32, F)
token_latents:  (B, 32, D)
fused:          (B, H)
logits:         (B, 2)
```

## 4. Architecture Specification

Module names:

- `Simple18PieceTokenExtractor`
- `BoardCNNTrunk`
- `PieceTokenMixer`
- `CNNTokenFusionHead`

Forward pass:

1. Input `(B, 18, 8, 8)`.
2. CNN path:
   - 3 or 4 convolution blocks with batchnorm/ReLU/dropout.
   - Width default `48`.
   - Output pooled features from mean and max pooling.
3. Token path:
   - Extract up to 32 occupied pieces.
   - Token feature vector includes:
     - piece type one-hot,
     - side-relative own/opponent flag,
     - color,
     - absolute rank/file,
     - side-relative rank/file,
     - castling/en-passant scalar context.
   - Encode tokens with MLP to `token_dim=64`.
   - Run `num_token_mixer_layers=2` lightweight mixer layers:

```text
token = token + MLP(token)
global_token_summary = mean/max/sum pooling over occupied tokens
token = token + gate(global_token_summary)
```

No self-attention is required for the first version.

4. Fusion:
   - Concatenate CNN pooled vector, token pooled vector, material/count summary, and a small interaction vector:

```text
interaction = cnn_proj(cnn_vec) * token_proj(token_vec)
```

5. MLP head returns `(B, 2)`.

Default shape sketch:

```text
input:        (B, 18, 8, 8)
cnn map:      (B, 48, 8, 8)
cnn pooled:   (B, 96)
tokens:       (B, 32, F)
token embed:  (B, 32, 64)
token pooled: (B, 192)  # mean/max/sum
fused:        (B, 384-512)
logits:       (B, 2)
```

Default config:

```yaml
model:
  name: piece_token_cnn_hybrid
  input_channels: 18
  num_classes: 2
  cnn_width: 48
  cnn_blocks: 4
  token_dim: 64
  token_mixer_layers: 2
  fusion_hidden: 192
  dropout: 0.1
  use_batchnorm: true
  include_interaction: true
  ablation: none
```

Parameter estimate:

- Around 250k to 700k depending on fusion width.

Complexity:

- Small CNN maps are only `8x8`.
- Token path is capped at 32 pieces and uses MLP/set pooling only.

## 5. Why This Is Useful

The existing simple CNN sees all squares but has no explicit variable-size piece-set representation. Many exotic ideas introduce token extraction, set pooling, role subspaces, or piece-level bottlenecks. This hybrid gives a sane regular comparator:

- If a fancy piece-token model cannot beat this hybrid, the fancy mechanism may not be adding much.
- If the token branch does not beat CNN-only, future token-heavy ideas need stronger justification.
- If the hybrid improves near-puzzle diagnostics, it becomes a useful baseline for the research loop.

## 6. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| `cnn_only_matched` | Remove token branch and increase CNN/head params to match | Occupied-piece token stream adds value | If it matches, token branch is unnecessary. |
| `token_only` | Remove CNN trunk and classify from token branch | Dense board texture matters | If it matches, CNN path may be redundant. |
| `no_interaction_fusion` | Remove multiplicative CNN-token interaction | Cross-branch interaction matters | If it matches, simple concatenation is enough. |
| `material_token_only` | Token branch uses only material/piece counts, no coordinates | Piece geometry matters | If it matches, token path is material shortcut. |
| `shuffle_token_coordinates` | Shuffle token square coordinates within each sample | Token coordinate semantics matter | If it matches, token path ignores geometry. |
| `single_token_layer` | Use one token MLP layer | Token mixing depth matters | If it matches, keep the simpler version. |

## 7. Benchmark And Falsification Criteria

Compare against:

- `configs/bench_cnn_small_simple18.yaml`
- `configs/bench_cnn_medium_simple18.yaml`
- `configs/bench_cnn_deep_simple18.yaml`
- `configs/bench_residual_small_simple18.yaml`
- `configs/bench_residual_medium_simple18.yaml`
- `configs/bench_multiscale_cnn_mixer_simple18.yaml` if that idea gets implemented.

Metrics:

- AUROC.
- Accuracy and balanced accuracy.
- F1.
- Calibration.
- Fine-label `0/1/2 -> predicted 0/1` confusion matrix.
- Class `1` recall or precision at matched fine-label-`0` false-positive rate if tooling supports it.

Success threshold:

- Beats best same-budget CNN/residual baseline by at least `+0.5` AUROC point, or improves class-`1` recall at matched fine-label-`0` FPR by at least `+1.0` point.
- Beats `cnn_only_matched`.
- Token branch diagnostics show nontrivial coordinate use: coordinate shuffle hurts performance.

Failure threshold:

- `cnn_only_matched` matches the full hybrid.
- `material_token_only` or coordinate shuffle matches full token branch.
- Gains disappear against a matched-parameter residual CNN.

Use as a regular baseline if:

- It consistently beats current CNNs and has stable training without exotic machinery.

Abandon if:

- The token branch does not add value over CNN-only controls.

## 8. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260424_piece_token_cnn_hybrid/idea.yaml` | Create | Idea metadata. |
| `ideas/20260424_piece_token_cnn_hybrid/architecture.md` | Create | Architecture details from this packet. |
| `ideas/20260424_piece_token_cnn_hybrid/ablations.md` | Create | Ablation table. |
| `src/chess_nn_playground/models/trunk/piece_token_cnn_hybrid.py` | Create | Token extractor, CNN trunk, token mixer, fusion head, builder. |
| `src/chess_nn_playground/models/registry.py` | Update | Register `piece_token_cnn_hybrid`. |
| `configs/bench_piece_token_cnn_hybrid_simple18.yaml` | Create | Main config. |
| `configs/bench_piece_token_cnn_hybrid_cnn_only.yaml` | Create | Central ablation config. |
| `tests/test_piece_token_cnn_hybrid_forward.py` | Create | Forward shape, finite logits, token mask handling, ablation smoke tests. |

## 9. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-24_2109_friday_shanghai_piece_token_cnn_hybrid.md
  generated_at: 2026-04-24 21:09
  weekday: Friday
  timezone: Asia/Shanghai
  idea_slug: piece_token_cnn_hybrid
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260424_piece_token_cnn_hybrid
  name: Piece-Token CNN Hybrid
  slug: piece_token_cnn_hybrid
  status: draft
  created_at: 2026-04-24
  author: Codex
  short_thesis: A compact hybrid of dense board CNN features and occupied-piece token features may be a stronger regular benchmark for chess puzzle-likeness classification.
  novelty_claim: Practical architecture baseline, not a high-novelty research mechanism.
  expected_advantage: Combines board texture and variable-size piece-set information while remaining easy to train and ablate.
  central_falsification_ablation: cnn_only_matched
  target_task: coarse_binary
  input_representation: simple_18
  output_heads: binary logits
  compute_notes: Small 8x8 CNN plus capped 32-token MLP mixer; should be practical.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/bench_piece_token_cnn_hybrid_simple18.yaml
  model_path: src/chess_nn_playground/models/trunk/piece_token_cnn_hybrid.py
  latest_result_path: null
  notes: Treat as a regular token-aware baseline candidate.
```

```yaml
config_yaml:
  run:
    name: bench_piece_token_cnn_hybrid_simple18
    output_dir: results
  seed: 42
  deterministic: true
  mode: coarse_binary
  device: nvidia
  data:
    train_path: data/splits/crtk_sample_3class/split_train.parquet
    val_path: data/splits/crtk_sample_3class/split_val.parquet
    test_path: data/splits/crtk_sample_3class/split_test.parquet
    encoding: simple_18
    cache_features: false
  model:
    name: piece_token_cnn_hybrid
    input_channels: 18
    num_classes: 2
    cnn_width: 48
    cnn_blocks: 4
    token_dim: 64
    token_mixer_layers: 2
    fusion_hidden: 192
    dropout: 0.1
    use_batchnorm: true
    include_interaction: true
    ablation: none
  training:
    epochs: 3
    batch_size: 512
    num_workers: 0
    learning_rate: 0.001
    weight_decay: 0.0001
    class_weighting: balanced
    early_stopping_patience: 2
    mixed_precision: false
```

```yaml
model_spec:
  model_name: piece_token_cnn_hybrid
  file_path: src/chess_nn_playground/models/trunk/piece_token_cnn_hybrid.py
  builder_function: build_piece_token_cnn_hybrid_from_config
  input_shape: [batch, 18, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - Simple18PieceTokenExtractor
    - BoardCNNTrunk
    - PieceTokenMixer
    - CNNTokenFusionHead
  required_config_fields:
    - input_channels
    - num_classes
    - cnn_width
    - cnn_blocks
    - token_dim
    - token_mixer_layers
    - fusion_hidden
    - include_interaction
    - ablation
  expected_parameter_count: 250000-700000
  expected_memory_notes: Token path is capped at 32 occupied pieces; CNN maps stay 8x8.
```

## 10. Prompt Maintenance Notes

This should be treated as a regular baseline candidate. If implemented and strong, update the deep research prompt so future high-novelty ideas compare against it.

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add `Piece-Token CNN Hybrid` to baseline list if it outperforms current CNNs. | Future research packets should beat a simple token-aware baseline, not only pure CNNs. | `Current baselines already exist` |
| Mark CNN+piece-token late fusion as regular engineering unless a new falsifiable bottleneck is added. | Prevents future prompts from treating basic token fusion as high novelty. | `Common Approaches Rejected` |

## 11. Final Sanity Check

- Stored as a Markdown file in `ideas/research/packets/classic/`: yes
- Regular practical architecture idea: yes
- No forbidden engine features used as inputs: yes
- Does not fabricate labels: yes
- Minimal current-data experiment exists: yes
- Uses shared trainer contract: yes
- Includes central ablations: yes
- Gives implementation hooks for future Codex work: yes

# Codex Research Batch: Residual-Inspired Architecture Candidates

## File Metadata

- Filename: `chess_nn_research_2026-04-24_2054_friday_shanghai_residual_inspired_batch.md`
- Generated at: 2026-04-24 20:54
- Weekday: Friday
- Timezone: Asia/Shanghai
- Intended next consumer: Codex
- Status: draft research batch, not implemented

## Purpose

This file explores residual-inspired neural architectures without proposing "just use a ResNet." The central idea is to use residuals as measured objects:

- residual correction against a frozen baseline,
- residual convergence defects in an unrolled fixed-point process,
- residuals between coarse and fine board representations,
- residuals against simple independence or material-only explanations,
- residual uncertainty around near-puzzle examples.

These are architecture candidates, not implemented ideas or benchmark results.

## Shared Data Contract

All candidates target the current binary puzzle-likeness task:

- output `0`: non-puzzle
- output `1`: puzzle-like

Fine labels `0`, `1`, and `2` are diagnostics only. The first experiment for each candidate should use `simple_18`, the current `crtk_sample_3class` splits, and the shared trainer.

Allowed inputs:

- Current board occupancy, side-to-move, castling/en-passant planes.
- Deterministic square coordinates and safe current-board transforms.
- A frozen baseline model's train-split-derived logits or latent vectors only when the baseline is trained strictly on training data and applied at inference without using validation/test labels.

Forbidden inputs:

- Stockfish scores, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, dataset provenance, unresolved candidate status, or anything derived from them.
- Engine search, forced-line search, legal mate/stalemate oracles, or future game outcomes.

## Ranked Shortlist

| Rank | Candidate | Residual object | Why expand it |
|---|---|---|---|
| 1 | Fixed-Point Residual Defect Network | Norms and directions of unrolled latent update defects | Most distinct from ordinary ResNet depth and has a clean convergence-defect falsifier. |
| 2 | Baseline Logit Residual Adapter | Learned correction to a frozen simple CNN's logits and latent uncertainty | Practical second-stage test of what the baseline misses. |
| 3 | Coarse-to-Fine Board Residual Pyramid | Fine board evidence not explained by coarse pooled boards | Simple to implement and directly residual-inspired. |
| 4 | Independence Residual Interaction Network | Signed residuals from product-of-marginals board explanations | Tests interactions left unexplained by material/square marginals. |
| 5 | Residual Calibration Error Field | Spatial features that predict baseline calibration residuals | Useful if the current baseline is accurate but poorly calibrated on near-puzzles. |

The best next full packet is `Fixed-Point Residual Defect Network`: it is residual-inspired without being a routine residual CNN, and the central diagnostic is observable even if performance is mediocre.

## Candidate 1: Fixed-Point Residual Defect Network

### Thesis

Puzzle-like positions may be harder for a learned board-state operator to equilibrate. Instead of classifying only the final latent, classify from the residual defects of an unrolled update process:

```text
r_t = F(h_t, x) - h_t
h_{t+1} = h_t + alpha_t r_t
```

The hypothesis is that puzzle-like positions create larger, more directional, or more oscillatory residual defects than ordinary non-puzzle positions.

### Fingerprint

```text
current-board encoder
+ shared latent residual update operator
+ residual norm/direction/oscillation summaries across unrolled steps
+ binary puzzle-likeness head
```

### Why It Is Not Just A ResNet

A standard ResNet classifies from the final feature after residual blocks. This model classifies from the residual trajectory itself: norms, signed changes, cosine similarities between defects, contraction ratios, and final fixed-point error. The residual path is the bottleneck and the falsifier.

### Architecture Sketch

1. Encode input `(B, 18, 8, 8)` into `h_0 in R^D`, default `D=128`.
2. Apply a shared residual update block `T=6` times:

```text
u_t = update_mlp([h_t, global_board_embed])
r_t = u_t - h_t
h_{t+1} = h_t + alpha * r_t
```

3. Compute residual trajectory features:
   - `||r_t||_2`
   - `||r_t||_1`
   - cosine similarity `cos(r_t, r_{t-1})`
   - contraction ratio `||r_t|| / (||r_{t-1}|| + eps)`
   - final defect `||F(h_T, x) - h_T||`
   - signed low-rank projections of `r_t`
4. Classify from trajectory features plus optionally the final latent.

### Tensor Contract

```text
input:        (B, 18, 8, 8)
h_path:       (B, T + 1, D)
r_path:       (B, T, D)
defect_stats: (B, T * S + projections)
logits:       (B, 2)
```

Default:

```yaml
model:
  name: fixed_point_residual_defect
  input_channels: 18
  num_classes: 2
  latent_dim: 128
  steps: 6
  update_hidden: 256
  alpha: 0.5
  projection_dim: 16
  include_final_latent: true
  ablation: none
```

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `final_latent_only` | Classify from `h_T` without residual path stats | Tests whether residual defects are the signal. |
| `untied_residual_blocks` | Use ordinary untied residual blocks with matched parameter count | Tests whether fixed-point/shared-update dynamics matter. |
| `random_update_operator` | Freeze random update block and train only the head | Tests whether learned dynamics matter. |
| `defect_norm_only` | Remove residual directions and cosine oscillation | Tests whether direction/oscillation adds signal beyond hardness. |
| `single_step` | Use only one residual update | Tests whether trajectory information matters. |

### Implementation Hook

- Model file: `src/chess_nn_playground/models/fixed_point_residual.py`
- Registry name: `fixed_point_residual_defect`
- Main config: `configs/bench_fixed_point_residual_simple18.yaml`
- Central ablation config: `configs/bench_fixed_point_residual_final_only.yaml`
- Tests: forward shape, finite logits, deterministic repeated forward, ablation modes.

### Success/Failure

Success:

- Main model beats `final_latent_only` and `untied_residual_blocks` on AUROC or class-`1` recall at matched fine-label-`0` false-positive rate.
- Residual trajectory diagnostics differ between fine labels `0`, `1`, and `2`.

Failure:

- Final latent only matches the full model.
- Untied ordinary residual blocks match or beat it with no residual diagnostic difference.

## Candidate 2: Baseline Logit Residual Adapter

### Thesis

The existing simple CNN likely has systematic errors. A small residual adapter can test what information remains after the baseline logit and latent representation are known:

```text
logits = frozen_baseline_logits + correction(x, baseline_latent)
```

This is residual-inspired in the practical modeling sense: model the correction to a known baseline, not the whole function from scratch.

### Fingerprint

```text
frozen trained simple CNN
+ small correction network over board and baseline latent
+ residual logit correction diagnostics
+ binary puzzle-likeness head
```

### Risk

This is close to ensembling. It should only be used as a diagnostic second-stage architecture, not as the main novelty claim. The correction must be small, and all baseline weights must be trained only on training data.

### Architecture Sketch

1. Train or load a baseline `simple_cnn` checkpoint trained on the train split.
2. Freeze the baseline.
3. Expose its penultimate latent `z_base` and logits `l_base`.
4. Train a small adapter:

```text
c = adapter([small_board_embed(x), z_base, l_base])
l = l_base + beta * c
```

5. Export correction magnitude and correction direction diagnostics.

### Tensor Contract

```text
input:        (B, 18, 8, 8)
base_latent:  (B, D_base)
base_logits:  (B, 2)
correction:   (B, 2)
logits:       (B, 2)
```

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `baseline_only` | Use frozen baseline logits directly | Tests whether adapter adds value. |
| `adapter_without_board` | Adapter sees only baseline latent/logits | Tests whether raw board adds residual information. |
| `board_only_adapter` | Adapter sees board but not baseline latent/logits | Tests whether correction is really residual. |
| `random_frozen_baseline` | Use a random frozen baseline latent/logit source | Tests whether the trained baseline representation matters. |
| `large_adapter_control` | Increase adapter size to CNN-like capacity | Checks whether gains are just a second model. |

### Implementation Hook

- Model file: `src/chess_nn_playground/models/baseline_residual_adapter.py`
- Registry name: `baseline_logit_residual_adapter`
- Main config: `configs/bench_baseline_residual_adapter_simple18.yaml`
- Needs a clear checkpoint path field:

```yaml
model:
  baseline_checkpoint_path: results/<baseline_run>/checkpoint_best.pt
  correction_scale: 0.5
  freeze_baseline: true
```

### Success/Failure

Success:

- Adapter improves class-`1` recall or calibration at matched fine-label-`0` FPR while keeping correction magnitudes small.

Failure:

- Board-only adapter matches it, random baseline matches it, or gains require a large adapter that is effectively an ensemble.

## Candidate 3: Coarse-to-Fine Board Residual Pyramid

### Thesis

A puzzle-like position may be present in details not explained by coarse board summaries. Build a residual pyramid over the board: classify from what remains after each scale's coarse reconstruction explains the finer scale.

### Fingerprint

```text
current-board tensor
+ deterministic coarse pooling and learned safe upsampling
+ fine-minus-coarse residual maps by scale/channel
+ binary puzzle-likeness head
```

### Why It Is Distinct

- Not wavelet scattering: residuals are learned coarse reconstructions, not fixed wavelet bands.
- Not ordinary CNN: the architecture explicitly exposes reconstruction residual maps.
- Not masked codec: no label-free masked prediction objective is required.

### Architecture Sketch

1. Build coarse views:
   - `8x8 -> 4x4 -> 2x2 -> 1x1` using average pooling over channels.
2. Upsample each coarse view back to the finer scale.
3. Compute residual maps:

```text
e_4 = x_8 - up(decoder_4(pool_4(x_8)))
e_2 = pool_4(x_8) - up(decoder_2(pool_2(x_8)))
e_1 = pool_2(x_8) - up(decoder_1(pool_1(x_8)))
```

4. Feed residual energies, signed residual maps, and a small residual CNN head into binary classifier.

### Tensor Contract

```text
input:         (B, 18, 8, 8)
residual_8:    (B, R, 8, 8)
residual_4:    (B, R, 4, 4)
residual_2:    (B, R, 2, 2)
stats:         (B, features)
logits:        (B, 2)
```

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `coarse_only` | Use only pooled coarse maps | Tests whether fine residuals help. |
| `fine_only_cnn` | Use matched small CNN on original board | Tests whether pyramid residuals beat ordinary local filters. |
| `random_upsampler` | Freeze random upsampling decoders | Tests whether learned coarse explanation matters. |
| `residual_energy_only` | Use only norms, no signed residual maps | Tests whether residual structure matters beyond magnitude. |

### Implementation Hook

- Model file: `src/chess_nn_playground/models/coarse_fine_residual.py`
- Registry name: `coarse_fine_residual_pyramid`
- Main config: `configs/bench_coarse_fine_residual_simple18.yaml`
- Central ablation config: `configs/bench_coarse_fine_residual_coarse_only.yaml`

### Success/Failure

Success:

- Beats `coarse_only` and a matched small CNN, especially on class-`1` diagnostics.

Failure:

- Fine-only CNN or residual-energy-only matches the full model.

## Candidate 4: Independence Residual Interaction Network

### Thesis

Some puzzle-like signals may be interactions that remain after subtracting a simple independence explanation of board occupancy. Instead of modeling all piece-square interactions directly, compute signed residuals:

```text
observed(piece, square) - expected(piece) * expected(square)
```

and classify from these residual interaction maps.

### Fingerprint

```text
current-board occupancy
+ product-of-marginals or low-rank independence baseline
+ signed occupancy residual maps
+ interaction residual classifier
```

### Relationship To Prior Research

This is adjacent to high-order constellation and Mobius/ANOVA ideas, but the central object is a residual against an explicitly simple independence model, not explicit degree-2/3 occupied-piece tuple enumeration.

### Architecture Sketch

1. From `simple_18`, compute safe occupancy summaries:
   - piece/channel marginals
   - square marginals
   - side-relative rank/file marginals
2. Build an expected occupancy tensor from a low-rank product model.
3. Compute signed residual maps `r = x_piece_planes - expected`.
4. Feed residual maps plus residual summary stats to a small classifier.

### Tensor Contract

```text
input:       (B, 18, 8, 8)
expected:    (B, 12, 8, 8)
residual:    (B, 12, 8, 8)
stats:       (B, features)
logits:      (B, 2)
```

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `raw_piece_planes` | Use original piece planes with matched head | Tests whether residualization helps. |
| `marginals_only` | Use piece/square marginals without residual maps | Tests whether interaction residuals add signal. |
| `random_expected_tensor` | Use random expected maps with matched means | Tests whether product baseline matters. |
| `material_only_expected` | Expected tensor from material only | Tests whether square marginals matter. |

### Implementation Hook

- Model file: `src/chess_nn_playground/models/independence_residual.py`
- Registry name: `independence_residual_interaction`
- Main config: `configs/bench_independence_residual_simple18.yaml`
- Central ablation config: `configs/bench_independence_residual_marginals_only.yaml`

### Success/Failure

Success:

- Beats raw-piece-plane matched head and marginals-only control.

Failure:

- Raw planes or marginals match it, suggesting the residual subtraction is not useful.

## Candidate 5: Residual Calibration Error Field

### Thesis

If the existing CNN has good accuracy but poor reliability on near-puzzles, a residual calibration architecture can predict where the baseline is likely overconfident. The model learns a spatial "calibration error field" and uses it to adjust logits or produce diagnostics.

### Fingerprint

```text
frozen or jointly trained board classifier
+ spatial error-field branch
+ residual logit temperature/correction
+ calibration and fine-label diagnostic outputs
```

### Relationship To Prior Research

This is adjacent to credal/evidential and sample-wise temperature ideas, but residual-inspired because it explicitly models calibration residuals of a base classifier.

### Architecture Sketch

1. Train a small CNN classifier normally.
2. Add an error-field branch from intermediate feature maps:

```text
error_field = conv_head(features)
temperature = softplus(pool(error_field)) + eps
correction = small_mlp(pool(error_field))
logits = raw_logits / temperature + correction
```

3. Regularize the correction to be small.
4. Export temperature, correction norm, and error-field heatmaps for diagnostics.

### Tensor Contract

```text
input:        (B, 18, 8, 8)
features:     (B, C, 8, 8)
error_field:  (B, K, 8, 8)
temperature:  (B, 1)
correction:   (B, 2)
logits:       (B, 2)
```

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `raw_logits_only` | No calibration residual branch | Tests whether residual calibration helps. |
| `global_temperature_only` | Single learned scalar temperature | Tests whether sample/spatial calibration matters. |
| `correction_only` | Remove temperature, keep additive correction | Separates calibration from correction. |
| `temperature_only` | Remove additive correction | Separates correction from calibration. |

### Implementation Hook

- Model file: `src/chess_nn_playground/models/residual_calibration.py`
- Registry name: `residual_calibration_field`
- Main config: `configs/bench_residual_calibration_simple18.yaml`
- Central ablation config: `configs/bench_residual_calibration_global_temp.yaml`

### Success/Failure

Success:

- Improves calibration and class-`1` diagnostics without hurting AUROC.

Failure:

- Global temperature or raw logits match it.

## Cross-Candidate Benchmark Rules

Use the same basic run setup unless the candidate explicitly needs a frozen baseline checkpoint:

```yaml
seed: 42
deterministic: true
mode: coarse_binary
data:
  train_path: data/splits/crtk_sample_3class/split_train.parquet
  val_path: data/splits/crtk_sample_3class/split_val.parquet
  test_path: data/splits/crtk_sample_3class/split_test.parquet
  encoding: simple_18
  cache_features: false
model:
  input_channels: 18
  num_classes: 2
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

Compare against:

- `configs/bench_cnn_small_simple18.yaml`
- `configs/bench_cnn_medium_simple18.yaml`
- `configs/bench_residual_small_simple18.yaml`
- A same-parameter ordinary residual CNN where relevant.
- Each candidate's central non-residual ablation.

Required diagnostics:

- AUROC, accuracy, balanced accuracy, F1, calibration.
- Fine-label `0/1/2 -> predicted 0/1` confusion matrix for main model and central ablation.
- Class `1` recall or precision at matched fine-label-`0` false-positive rate where available.
- Candidate-specific residual diagnostics:
  - residual defect trajectory,
  - logit correction norm,
  - pyramid residual energy by scale,
  - independence residual maps,
  - calibration temperature/correction histograms.

## Prompt Maintenance Notes

If one of these residual-inspired ideas is implemented and fails, add the corresponding anti-duplicate rule to the deep research prompt:

| Candidate | Anti-duplicate rule if it fails |
|---|---|
| Fixed-Point Residual Defect Network | Do not repeat shared-update residual-defect trajectory classifiers with only different step counts, latent widths, or update MLP sizes. |
| Baseline Logit Residual Adapter | Do not repeat frozen-baseline logit correction adapters unless the residual target or falsifier changes beyond adapter size. |
| Coarse-to-Fine Board Residual Pyramid | Do not repeat coarse-to-fine board residual pyramids with only different pooling scales or decoder widths. |
| Independence Residual Interaction Network | Do not repeat product-of-marginals occupancy residual maps with only different low-rank expected tensors. |
| Residual Calibration Error Field | Do not repeat spatial residual calibration branches with only different temperature/correction head widths. |

## Final Sanity Check

- Stored as a Markdown file in `ideas/research_packets/`: yes
- Residual-inspired but not merely "use ResNet": yes
- Includes multiple architecture candidates: yes
- Avoids forbidden engine/search/source features: yes
- Does not fabricate labels: yes
- Keeps current-data minimal experiments possible: yes
- Includes central ablations: yes
- Gives implementation hooks for future Codex work: yes

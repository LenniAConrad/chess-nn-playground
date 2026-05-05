# Codex Research Batch: Additional Architecture Candidates 3

## File Metadata

- Filename: `chess_nn_research_2026-04-24_2118_friday_shanghai_architecture_batch_3.md`
- Generated at: 2026-04-24 21:18
- Weekday: Friday
- Timezone: Asia/Shanghai
- Intended next consumer: Codex
- Status: draft architecture batch, not implemented

## Purpose

This batch adds more candidate neural architectures that are not just variations of CNN depth or standard Transformers. The goal is to keep expanding the design space while preserving clear ablations and current-data feasibility.

These are not implementation commits. They are research candidates to promote into full packets only if selected.

## Shared Data Contract

All candidates target the current binary puzzle-likeness task:

- output `0`: non-puzzle
- output `1`: puzzle-like

Fine labels `0`, `1`, and `2` remain diagnostics only. The first implementation for each candidate should use `simple_18`, the existing `crtk_sample_3class` train/val/test splits, and the shared trainer.

Forbidden as model inputs:

- Stockfish scores, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, dataset provenance, unresolved candidate status, or anything derived from them.
- Engine search, forced-line search, legal mate/stalemate oracles, or future game outcomes.

Allowed:

- Current board occupancy, side-to-move, castling/en-passant planes, deterministic square coordinates, side-relative coordinates, and deterministic transforms of current board tensors.

## Ranked Shortlist

| Rank | Candidate | Main object | Why expand it |
|---|---|---|---|
| 1 | Kernel Mean Prototype Network | Kernel mean embeddings of occupied piece sets | Strong token-set baseline between simple pooling and exotic geometry. |
| 2 | TensorSketch Interaction Network | Random sketch of high-order board interactions | Tests higher-order interactions without explicit Mobius tuple enumeration. |
| 3 | Maxout Region Signature Network | Winner identities and margins in maxout feature banks | Gives a simple activation-region bottleneck with clear controls. |
| 4 | Spline Board Surface Network | Learned low-degree board surface coefficients and residuals | Practical smooth-geometry baseline distinct from CNN filters. |
| 5 | Boundary-Condition Disagreement CNN | Disagreement across padding/boundary assumptions | Cheap way to test edge/context sensitivity. |
| 6 | Piece-Drop Stability Network | Prediction stability under deterministic safe piece masks | Regular robustness idea; useful but adjacent to masking packets. |

Best next full packet from this batch: `Kernel Mean Prototype Network`.

## Candidate 1: Kernel Mean Prototype Network

### Thesis

Puzzle-like positions may be separable by the distribution of occupied piece tokens in a learned kernel feature space. Instead of attending to pieces or computing pairwise transport, embed the occupied-piece set as a kernel mean and compare it to learned prototype embeddings.

### Fingerprint

```text
occupied piece tokens
+ random Fourier / learned kernel features
+ mean and covariance embeddings
+ distances to learned prototype means
+ binary puzzle-likeness head
```

### Why It Is Distinct

- Not optimal transport: no coupling or Sinkhorn plan.
- Not sparse witness: no selected subset hides the rest.
- Not attention: no query-token softmax.
- Not Grassmannian/matrix-pencil: no eigenspace or generalized spectrum.

### Architecture Sketch

1. Extract up to 32 occupied tokens from `simple_18`.
2. Token features:
   - piece type,
   - own/opponent flag,
   - color,
   - rank/file,
   - side-relative rank/file,
   - castling/en-passant context.
3. Map tokens to kernel features:

```text
phi(t) = sqrt(2 / m) * cos(W t + b)
```

with fixed random Fourier features or learned low-rank features.
4. Pool set embedding:

```text
mu = mean_i phi(t_i)
second = mean_i phi(t_i) * phi(t_i)
```

5. Compare to `P` learned prototype means:

```text
d_p = ||mu - proto_p||_2^2
cos_p = cosine(mu, proto_p)
```

6. Classify from mean embedding, second-moment summary, distances, and token count/material stats.

### Tensor Contract

```text
input:       (B, 18, 8, 8)
tokens:      (B, 32, F)
kernel_phi:  (B, 32, M), default M=128
mean_embed:  (B, M)
proto_dists: (B, P), default P=16
logits:      (B, 2)
```

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `linear_mean_only` | Replace kernel features with raw-token mean/max pooling | Kernel embedding matters | If it matches, kernel features are unnecessary. |
| `random_prototypes_only` | Freeze random prototypes | Learned prototypes matter | If it matches, prototype learning is unnecessary. |
| `material_only_tokens` | Use only piece counts/types | Geometry matters | If it matches, model is material shortcut. |
| `coordinate_shuffle` | Shuffle token coordinates within sample | Token geometry matters | If it matches, coordinates are ignored. |
| `no_second_moment` | Remove second-moment features | Distribution shape matters beyond mean | If it matches, use simpler mean embedding. |

### Implementation Hook

- Model file: `src/chess_nn_playground/models/kernel_mean_prototype.py`
- Registry name: `kernel_mean_prototype`
- Main config: `configs/bench_kernel_mean_prototype_simple18.yaml`
- Central ablation config: `configs/bench_kernel_mean_prototype_linear_mean.yaml`
- Tests: forward shape, finite logits, token mask handling, deterministic fixed random features.

### Decision Rule

Promote if it beats raw token mean pooling and CNN baselines or improves class `1` diagnostics at matched fine-label `0` false-positive rate. Drop if material-only or coordinate shuffle matches full model.

## Candidate 2: TensorSketch Interaction Network

### Thesis

Some puzzle-like signals may require high-order interactions among piece-square facts. Exact high-order tuple enumeration is expensive and overlaps with Mobius/ANOVA packets, but TensorSketch can approximate polynomial-kernel interactions with a compact randomized sketch.

### Fingerprint

```text
board feature vector
+ CountSketch / TensorSketch polynomial interaction map
+ degree-2/3/4 sketch features
+ compact MLP head
```

### Why It Is Distinct

- Adjacent to high-order constellation ideas, but does not enumerate occupied tuples or learn explicit Mobius terms.
- More practical as a randomized feature map baseline for high-order interactions.
- The central ablation is degree-1 and shuffled sketch signs.

### Architecture Sketch

1. Build compact board vector `x_vec` from:
   - flattened `simple_18`,
   - material counts,
   - side-to-move/castling/en-passant features.
2. Project into `D=512` base features.
3. Apply TensorSketch maps for degrees 2 and 3:

```text
sketch_d(x) approx polynomial_kernel_degree_d(x)
```

4. Concatenate degree-1, degree-2, and degree-3 sketches.
5. MLP head returns logits.

### Tensor Contract

```text
input:      (B, 18, 8, 8)
base:       (B, 512)
sketch2:    (B, S), default S=512
sketch3:    (B, S)
features:   (B, 1536)
logits:     (B, 2)
```

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `degree1_only` | Remove polynomial sketches | High-order interactions matter | If it matches, sketches are unnecessary. |
| `degree2_only` | Remove degree-3 sketch | Third-order interactions matter | If it matches, degree 3 is unnecessary. |
| `random_sign_reshuffle` | Reshuffle CountSketch signs at eval/train consistently | Specific sketch structure matters | If it matches, features may be random capacity. |
| `matched_mlp` | MLP on base vector with matched parameter count | Sketch beats ordinary MLP capacity | If it matches, sketch is unnecessary. |
| `material_only_base` | Base vector from material only | Board geometry matters | If it matches, interaction sketch is shortcut. |

### Implementation Hook

- Model file: `src/chess_nn_playground/models/tensor_sketch_interaction.py`
- Registry name: `tensor_sketch_interaction`
- Main config: `configs/bench_tensor_sketch_interaction_simple18.yaml`
- Central ablation config: `configs/bench_tensor_sketch_interaction_degree1.yaml`
- Tests: deterministic sketch shape, finite logits, ablation smoke tests.

### Decision Rule

Promote if degree-2/3 sketches beat degree-1 and matched MLP controls. Drop if material-only or matched MLP matches full model.

## Candidate 3: Maxout Region Signature Network

### Thesis

Puzzle-like boards may fall into distinctive piecewise-linear activation regions. A maxout bank can expose those regions directly by reporting winner identities, margins, and region-transition statistics.

### Fingerprint

```text
current-board features
+ maxout feature banks
+ winner index / margin / entropy signatures
+ binary puzzle-likeness head
```

### Why It Is Distinct

- Not a residual CNN or attention model.
- Uses activation-region signatures as features, not only final hidden values.
- Easy to falsify with value-only and random-bank controls.

### Architecture Sketch

1. Small CNN stem maps board to feature vector `h`.
2. Maxout banks:

```text
y_j = max_{k=1..K} (a_{j,k}^T h + b_{j,k})
winner_j = argmax_k
margin_j = top1 - top2
```

3. Features:
   - maxout values,
   - margins,
   - soft winner probabilities,
   - winner histogram over banks,
   - entropy of winner distribution.
4. MLP head returns logits.

### Tensor Contract

```text
input:      (B, 18, 8, 8)
h:          (B, H)
maxout:     (B, J)
winners:    (B, J, K) soft one-hot
margins:    (B, J)
logits:     (B, 2)
```

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `values_only` | Remove winner/margin signatures | Region identity matters | If it matches, signatures are unnecessary. |
| `random_maxout_bank` | Freeze random maxout bank | Learned regions matter | If it matches, region learning is unnecessary. |
| `relu_mlp_matched` | Replace maxout with ReLU MLP at matched params | Maxout regions matter | If it matches, ordinary MLP is enough. |
| `margin_only` | Use only top1-top2 margins | Boundary proximity matters | If it matches, values are not needed. |

### Implementation Hook

- Model file: `src/chess_nn_playground/models/maxout_region_signature.py`
- Registry name: `maxout_region_signature`
- Main config: `configs/bench_maxout_region_signature_simple18.yaml`
- Central ablation config: `configs/bench_maxout_region_signature_values_only.yaml`

### Decision Rule

Promote if winner/margin signatures beat values-only and ReLU MLP controls. Drop if random maxout bank matches learned bank.

## Candidate 4: Spline Board Surface Network

### Thesis

Chess boards may benefit from a smooth geometric baseline that is not convolutional. Fit learned tensor-product spline surfaces to piece planes and classify from low-degree surface coefficients plus residual maps.

### Fingerprint

```text
current-board planes
+ tensor-product spline basis projection
+ smooth coefficients and fine residuals
+ binary puzzle-likeness head
```

### Why It Is Distinct

- Not wavelet scattering: spline basis is smooth low-degree control geometry.
- Not masked codec: no reconstruction pretraining.
- Not CNN: fixed projection basis plus compact residual head.

### Architecture Sketch

1. Precompute tensor-product B-spline or Bernstein basis over the 8x8 grid.
2. Project each piece plane onto `K` basis functions:

```text
coeff = basis_pinv @ plane
recon = basis @ coeff
residual = plane - recon
```

3. Feed coefficients, residual energies, and optionally residual maps to a small head.

### Tensor Contract

```text
input:       (B, 18, 8, 8)
coeffs:      (B, 18, K)
residuals:   (B, 18, 8, 8)
stats:       (B, features)
logits:      (B, 2)
```

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `coefficients_only` | Remove residual maps | Fine deviations matter | If it matches, residual maps unnecessary. |
| `residual_energy_only` | Use residual norms only | Residual structure matters | If it matches, maps unnecessary. |
| `random_basis` | Replace spline basis with random orthogonal basis | Smooth board geometry matters | If it matches, spline structure unnecessary. |
| `cnn_matched_params` | Matched small CNN | Spline basis helps beyond CNN capacity | If it matches, use CNN. |

### Implementation Hook

- Model file: `src/chess_nn_playground/models/spline_board_surface.py`
- Registry name: `spline_board_surface`
- Main config: `configs/bench_spline_board_surface_simple18.yaml`
- Central ablation config: `configs/bench_spline_board_surface_random_basis.yaml`

### Decision Rule

Promote if spline basis beats random basis and matched CNN. Drop if coefficients-only or random basis matches full model.

## Candidate 5: Boundary-Condition Disagreement CNN

### Thesis

Chess board edges matter: pawns, rooks, kings, and tactics behave differently near boundaries. A CNN's padding convention imposes a boundary assumption. Run a shared CNN under multiple boundary conditions and classify from disagreement.

### Fingerprint

```text
current-board tensor
+ shared CNN under zero/reflect/circular/edge-mask padding
+ boundary-condition latent disagreement
+ binary puzzle-likeness head
```

### Why It Is Distinct

- Regular and practical, but not just a deeper CNN.
- Tests edge-condition sensitivity directly.
- No engine/search features.

### Architecture Sketch

1. Same CNN weights are applied to board variants with different padding modes:
   - zero,
   - reflect,
   - circular,
   - learned edge-mask padding channel.
2. Pool latent for each boundary mode.
3. Features:
   - mean latent,
   - variance across modes,
   - pairwise distances,
   - logit disagreement if each mode has a small shared head.
4. Final head returns logits.

### Tensor Contract

```text
input:       (B, 18, 8, 8)
latents:     (B, M, D), M=4 boundary modes
disagree:    (B, features)
logits:      (B, 2)
```

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `zero_padding_only` | Use only standard zero padding | Boundary disagreement matters | If it matches, multi-boundary unnecessary. |
| `mean_latent_only` | Remove disagreement features | Disagreement carries signal | If it matches, ensemble mean is enough. |
| `independent_cnn_modes` | Separate weights per mode | Tests if shared-boundary sensitivity matters | If independent wins only by capacity, not useful. |
| `random_boundary_labels` | Shuffle boundary-mode identities | Specific boundary modes matter | If it matches, disagreement may be generic noise. |

### Implementation Hook

- Model file: `src/chess_nn_playground/models/boundary_disagreement_cnn.py`
- Registry name: `boundary_disagreement_cnn`
- Main config: `configs/bench_boundary_disagreement_cnn_simple18.yaml`
- Central ablation config: `configs/bench_boundary_disagreement_zero_only.yaml`

### Decision Rule

Promote if disagreement features beat zero-padding-only and mean-latent-only controls. Drop if ordinary CNN matches it.

## Candidate 6: Piece-Drop Stability Network

### Thesis

Puzzle-like positions may be less stable under deterministic removal of specific safe current-board evidence groups. Instead of forcing a classifier to use sparse witnesses, measure how a small encoder's latent changes when piece groups are dropped.

### Fingerprint

```text
current-board tensor
+ deterministic piece-group masks
+ encoder latent stability contrasts
+ binary puzzle-likeness head
```

### Relationship To Prior Research

Adjacent to sparse witness, masked codec, and attention perturbation ideas. Use this only as a regular robustness candidate, not as a high-novelty packet.

### Architecture Sketch

1. Define deterministic masks:
   - own minor pieces,
   - own major pieces,
   - opponent minor pieces,
   - opponent major pieces,
   - center pieces,
   - king-neighborhood pieces.
2. Run a shared small encoder on original board and masked boards.
3. Compute latent deltas:

```text
delta_m = ||z(x) - z(mask_m(x))||
```

4. Classify from original latent plus stability vector and delta ratios.

### Tensor Contract

```text
input:        (B, 18, 8, 8)
variants:     (B, M, 18, 8, 8)
latents:      (B, M + 1, D)
stability:    (B, M)
logits:       (B, 2)
```

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `original_only` | Use only original board latent | Stability matters | If it matches, masking unnecessary. |
| `random_masks` | Random masks with matched piece counts | Semantic groups matter | If it matches, groups unnecessary. |
| `material_masks_only` | Drop by piece value only | Geometry groups matter | If it matches, semantic masks too simple. |
| `delta_only` | Use stability deltas without original latent | Tests whether stability alone carries signal | If it matches, useful diagnostic. |

### Implementation Hook

- Model file: `src/chess_nn_playground/models/piece_drop_stability.py`
- Registry name: `piece_drop_stability`
- Main config: `configs/bench_piece_drop_stability_simple18.yaml`
- Central ablation config: `configs/bench_piece_drop_stability_original_only.yaml`

### Decision Rule

Promote only if semantic masks beat random masks and original-only encoder. Otherwise it is just expensive augmentation.

## Recommended Expansion Order From This Batch

1. `Kernel Mean Prototype Network`
2. `TensorSketch Interaction Network`
3. `Boundary-Condition Disagreement CNN`
4. `Maxout Region Signature Network`
5. `Spline Board Surface Network`
6. `Piece-Drop Stability Network`

## Shared Benchmark Setup

Each candidate should start from:

```yaml
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
- `configs/bench_piece_token_cnn_hybrid_simple18.yaml` if implemented

Required diagnostics:

- AUROC, balanced accuracy, F1, calibration.
- Fine-label `0/1/2 -> predicted 0/1` confusion matrix.
- Central ablation delta.
- Class `1` recall or precision at matched fine-label `0` false-positive rate where available.

## Prompt Maintenance Notes

If any candidate is implemented and fails, add a focused anti-duplicate rule:

| Candidate | Anti-duplicate rule if it fails |
|---|---|
| Kernel Mean Prototype | Do not repeat kernel-mean occupied-token prototypes with only different random feature counts or prototype counts. |
| TensorSketch Interaction | Do not repeat TensorSketch/polynomial-kernel board interaction maps with only different sketch size or degree. |
| Maxout Region Signature | Do not repeat maxout activation-region signature models with only different bank sizes. |
| Spline Board Surface | Do not repeat spline/smooth-basis board surface models with only different basis order or control counts. |
| Boundary-Condition Disagreement | Do not repeat shared-CNN boundary-padding disagreement models with only different padding modes. |
| Piece-Drop Stability | Do not repeat deterministic piece-drop stability classifiers with only different mask groups. |

## Final Sanity Check

- Stored as a Markdown file in `ideas/research_packets/`: yes
- Adds multiple fresh candidates: yes
- Avoids forbidden engine/search/source features: yes
- Does not fabricate labels: yes
- Keeps current-data minimal experiments possible: yes
- Includes central ablations and stop conditions: yes
- Gives implementation hooks for future Codex work: yes

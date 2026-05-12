# Codex Handoff Packet: Polar-Procrustes Alignment Bottleneck

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-24_2104_friday_shanghai_polar_procrustes.md`
- Generated at: 2026-04-24 21:04
- Weekday: Friday
- Timezone: Asia/Shanghai
- Idea slug: `polar_procrustes`
- Intended next consumer: Codex
- Status: draft research packet, not implemented

## 2. Executive Selection

- Idea name: Polar-Procrustes Alignment Bottleneck
- High-level linear algebra concept: polar decomposition, orthogonal Procrustes alignment, and matrix strain spectra.
- One-sentence thesis: Puzzle-like positions may be characterized by how poorly learned side/role summaries can be aligned by an orthogonal transform, and Procrustes residuals can test this relative matrix geometry without move generation or engine metadata.
- Idea fingerprint: current-board occupied tokens + learned own/opponent role matrices + polar decomposition of cross-covariance + Procrustes residual/strain spectrum + binary puzzle-likeness head.
- Why this is not a common CNN/ResNet/Transformer variant: the central computation is the optimal orthogonal alignment between learned role matrices via SVD/polar decomposition, not convolution, residual blocks, square-token attention, attack graphs, or LC0-style towers.
- Current-data minimal experiment: train on `simple_18` using the existing `crtk_sample_3class` train/val/test splits for 3 epochs, compare against same-budget `simple_18` CNN/residual baselines plus `separate_matrix_stats_only`.
- Smallest central falsification ablation: keep separate own/opponent matrix norms, singular values, role masses, and token summaries, but remove Procrustes alignment residuals and polar strain features.
- Expected information gain if it fails: a clean failure rules out polar/Procrustes relative-alignment bottlenecks before trying larger role matrices or CNN-fused versions.

## 3. Problem Restatement And Data Contract

The task is binary chess puzzle-likeness classification:

- output `0`: non-puzzle
- output `1`: puzzle-like

Fine labels:

- `0`: known non-puzzle
- `1`: verified near-puzzle
- `2`: verified puzzle

The model trains on the binary target while the standard report keeps fine-label `0/1/2 -> predicted 0/1` diagnostics.

Allowed neural inputs:

- Current board occupancy from `simple_18`.
- Side-to-move, castling, and en-passant planes already present in the encoding.
- Deterministic square coordinates and side-relative coordinates.
- Own/opponent partition derived from side-to-move and current piece planes.

Forbidden neural inputs:

- Stockfish scores, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, dataset provenance, unresolved candidate status, or anything derived from them.
- Engine search, forced-line search, checkmate/stalemate oracles, future game outcomes, or label-informed masks.

Tensor contract:

```text
input:            (B, 18, 8, 8)
occupied_tokens:  (B, N, F), N <= 32
own_matrix:       (B, R, D)
opp_matrix:       (B, R, D)
cross_cov:        (B, D, D) or (B, R, R)
polar_features:   (B, features)
logits:           (B, 2)
```

First implementation should use side-to-move-relative own/opponent occupied tokens and `simple_18`.

Leakage checklist:

- Own/opponent is derived from current board and side-to-move only.
- Fine labels are evaluation-only.
- No move list, attack graph, engine feature, source metadata, or legal oracle is used.

## 4. Research Map

External research anchors are conceptual only. No external citation was verified during generation.

| Source or concept | Borrowed | Not copied |
|---|---|---|
| Orthogonal Procrustes problem | The optimal orthogonal matrix aligning two learned matrices and its residual norm. | No supervised shape alignment labels and no external geometry dataset. |
| Polar decomposition | Decompose a cross-covariance matrix into orthogonal alignment and symmetric positive semidefinite strain. | No physical deformation model and no iterative polar solver requirement. |
| Singular-value spectra | Singular values of cross-covariance and strain indicate alignment strength by mode. | No generic SVD-only compression as the whole classifier. |

Candidate search trace:

| Candidate mechanism | Why not selected |
|---|---|
| Direct cosine similarity between own/opponent pooled vectors | Too simple; loses matrix-mode alignment and is a weak linear algebra bottleneck. |
| Canonical correlation between own/opponent sets | Close to Grassmannian/principal-angle packet already stored. |
| Matrix-pencil comparison of own/opponent forms | Already covered by the matrix-pencil packet. |
| Full attention between own and opponent pieces | Too close to attention-inspired batches and less cleanly linear algebraic. |

Concept-to-operator mapping:

| High-level concept | Concrete operator/object in this idea | Tensor contract | Central falsifier | Why not a duplicate |
|---|---|---|---|---|
| Orthogonal Procrustes | `min_Q ||XQ - Y||_F` for learned own/opponent role matrices | `(B, R, D), (B, R, D) -> (B, features)` | separate matrix stats only | Not determinant volume, not Grassmannian angles, not matrix pencil. |
| Polar decomposition | `C = U H` for cross-covariance `C = X^T Y` | `(B, D, D) -> U, singular/strain stats` | random orthogonal alignment | Tests optimal alignment structure specifically. |
| Alignment residual | `||X Q_star - Y||_F` and per-role residuals | `(B, R, D) -> (B, residual stats)` | identity alignment only | Tests whether best rotation matters. |

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN | `src/chess_nn_playground/models/trunk/cnn.py` | Already present and tests local learned filters, not own/opponent matrix alignment. |
| Residual CNN | `src/chess_nn_playground/models/trunk/residual_cnn.py` | More residual depth is ordinary scaling. |
| LC0-style CNN/residual CNN | Existing 112-plane configs | Too close to engine-network conventions. |
| Vanilla ViT over 64 squares | Common square-token Transformer | Too broad and does not isolate Procrustes geometry. |
| Cross-attention between own and opponent pieces | Attention-inspired candidate | Would test attention routing, not polar alignment. |
| Grassmannian principal-angle bottleneck | Local 2026-04-24 packet | Compares subspace orientation; this compares full learned matrices by optimal orthogonal alignment and strain. |
| Matrix-pencil generalized spectrum | Local 2026-04-24 packet | Compares quadratic forms; this aligns two role matrices by Procrustes/polar decomposition. |
| Hyperparameter tuning | Existing configs | Not a research architecture. |
| Ensembling | Any leaderboard ensemble | Would hide whether alignment residuals matter. |

## 6. Mathematical Thesis

Let `x` be a current board. Extract occupied tokens and split them into side-to-move-relative sets:

```text
S_own(x) = own occupied pieces
S_opp(x) = opponent occupied pieces
```

A shared token encoder maps tokens to embeddings. Role pooling produces two matrices:

```text
X(x) in R^{R x D}   # own role matrix
Y(x) in R^{R x D}   # opponent role matrix
```

Rows correspond to learned role summaries. The orthogonal Procrustes problem asks for:

```text
Q_star(x) = argmin_{Q^T Q = I} ||X(x) Q - Y(x)||_F
```

If `C = X^T Y = U Sigma V^T`, then:

```text
Q_star = U V^T
min_Q ||XQ - Y||_F^2 = ||X||_F^2 + ||Y||_F^2 - 2 * sum_i sigma_i(C)
```

The polar decomposition `C = Q_star H` separates the best orthogonal alignment `Q_star` from the symmetric strain `H`.

Core hypothesis:

Puzzle-like positions may involve a mismatch or sharply structured alignment between own and opponent role summaries. A tactical position is not only "own pieces have features" or "opponent pieces have features"; it may be about whether the two sets can be brought into alignment by a coherent role rotation, and where that alignment fails.

Proposition:

The Procrustes residual is invariant to a shared right-orthogonal coordinate change in the embedding space.

Proof sketch:

Let `X' = XW` and `Y' = YW` for orthogonal `W`. Then:

```text
min_Q ||X'Q - Y'||_F = min_Q ||XWQ - YW||_F
```

Set `Q' = W Q W^T`; the feasible set of orthogonal matrices is unchanged, and Frobenius norm is orthogonally invariant. Therefore the residual measures relative alignment rather than arbitrary embedding coordinates.

What is actually proven:

- The Procrustes residual and cross-covariance singular values are invariant to shared orthogonal coordinate changes.
- The residual is a specific relative matrix comparison, not a separate summary of own and opponent matrices.
- Removing Procrustes terms while preserving separate singular values directly falsifies the claimed alignment signal.

What remains hypothesized:

- That learned role matrices correspond to chess-relevant side/role summaries.
- That puzzle-like positions have distinctive alignment residuals or strain spectra.
- That the current split is not dominated by material or source artifacts.

Counterexamples:

- Labels are driven by material imbalance or source artifacts.
- Own/opponent matrices are too sparse in low-material positions.
- Tactical signal requires legal move consequences and cannot be captured by static role alignment.
- Separate own/opponent matrix spectra already contain all useful information.

Self-critique:

This idea may collapse into material comparison: own and opponent piece sets differ by count and type, and the Procrustes residual may simply measure that. Mandatory controls must include material-only matrices, count-matched role pooling, and separate-matrix-stats-only ablations. If those match, the alignment story is not doing real work.

## 7. Architecture Specification

Module names:

- `Simple18OwnOpponentTokenExtractor`
- `PieceSquareTokenEncoder`
- `RoleMatrixPooler`
- `PolarProcrustesLayer`
- `PolarProcrustesHead`

Forward pass:

1. Decode `simple_18` into occupied tokens.
2. Split tokens into own/opponent relative to side-to-move.
3. Build token features:
   - piece type one-hot,
   - own/opponent flag,
   - color,
   - absolute coordinates,
   - side-relative coordinates,
   - castling/en-passant scalars.
4. Encode tokens with a shared MLP:

```text
(B, N, F) -> (B, N, D), default D=48
```

5. Pool role matrices for own and opponent:

```text
X = role_pool(own_tokens) -> (B, R, D), default R=8
Y = role_pool(opp_tokens) -> (B, R, D)
```

Role pooling can use learned role queries with masked softmax over each side's tokens. This is not the main attention idea; it is just a differentiable set-to-matrix adapter.

6. Optionally center and normalize rows:

```text
X = row_layer_norm(X)
Y = row_layer_norm(Y)
```

7. Compute cross-covariance:

```text
C = X^T Y / R     # (B, D, D)
```

For lower compute, use `C_role = X Y^T / D` with shape `(B, R, R)`. First implementation should use the smaller role-space matrix if needed.

8. Compute SVD:

```text
C = U Sigma V^T
Q_star = U V^T
H_spectrum = Sigma
```

9. Compute features:
   - Procrustes residual `||X Q_star - Y||_F`,
   - normalized residual,
   - singular values of `C`,
   - nuclear norm, spectral norm, stable rank,
   - identity residual `||X - Y||_F`,
   - improvement from alignment `||X - Y||_F - ||X Q_star - Y||_F`,
   - per-role residual norms,
   - separate singular values of `X` and `Y`.
10. MLP head returns `(B, 2)`.

Shapes:

```text
input:       (B, 18, 8, 8)
tokens:      (B, 32, F)
embeddings:  (B, 32, 48)
X, Y:        (B, 8, 48)
C:           (B, 48, 48) or (B, 8, 8)
features:    about (B, 120-260)
logits:      (B, 2)
```

Parameter estimate:

- 60k to 150k depending on token dimension and head width.

Complexity:

- Role pooling: `O(B * N * R * D)`.
- SVD of `D x D` matrix: `O(B * D^3)`, manageable for `D=48`.
- If this is too expensive, use `R x R` role cross-covariance with `R=8`.

Required config fields:

```yaml
model:
  name: polar_procrustes_alignment
  input_channels: 18
  num_classes: 2
  token_dim: 48
  role_count: 8
  head_hidden: 128
  matrix_space: embedding
  normalize_rows: true
  include_separate_spectra: true
  ablation: none
```

Encoding adapters:

- `simple_18`: supported first.
- `lc0_static_112`: fail closed unless current-board piece channels and auxiliary semantics are explicitly mapped.
- `lc0_bt4_112`: optional later, using only known current-slot piece planes for deterministic own/opponent token extraction. History planes must not drive deterministic token extraction.

Pseudocode:

```python
tokens, mask, side_mask = extract_own_opponent_tokens_simple18(x)
h = token_encoder(tokens)
own_h = h.masked_fill(~own_mask[..., None], 0)
opp_h = h.masked_fill(~opp_mask[..., None], 0)
x_mat = role_pool(own_h, own_mask)
y_mat = role_pool(opp_h, opp_mask)
if normalize_rows:
    x_mat = row_norm(x_mat)
    y_mat = row_norm(y_mat)
c = x_mat.transpose(-1, -2) @ y_mat / role_count
u, sigma, vh = torch.linalg.svd(c, full_matrices=False)
q = u @ vh
aligned = x_mat @ q
residual = torch.linalg.norm(aligned - y_mat, dim=(-2, -1))
features = build_procrustes_features(x_mat, y_mat, sigma, residual, ablation)
return head(features)
```

## 8. Loss, Training, And Regularization

- Primary loss: existing balanced cross entropy on coarse binary labels.
- Optimizer: AdamW.
- Learning rate: `0.001`.
- Weight decay: `0.0001`.
- Batch size: `512` if using `R x R` SVD or `256` if using `D x D` SVD with autograd memory pressure.
- Optional regularization:
  - role entropy penalty to prevent all roles pooling the same tokens;
  - small row-norm stabilization penalty;
  - keep optional penalties off for the first benchmark unless role collapse appears.
- Determinism: seed `42`, deterministic true. Check SVD determinism in CPU smoke tests.
- Fair comparison: same splits, same binary target, same 3-epoch budget, same artifact pipeline.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| `separate_matrix_stats_only` | Keep separate singular values/norms of `X` and `Y`, remove cross-covariance and Procrustes features | Relative alignment matters | If it matches, Procrustes alignment is unnecessary. |
| `identity_alignment_only` | Use `||X - Y||` without optimal `Q_star` | Optimal orthogonal alignment matters | If it matches, polar decomposition is unnecessary. |
| `random_orthogonal_alignment` | Replace `Q_star` with fixed random orthogonal matrix | Learned sample-specific alignment matters | If it matches, any rotation/control feature is enough. |
| `batch_shuffled_opponent` | Pair own matrix `X(x)` with opponent matrix `Y(x')` from another sample | Own/opponent relation is sample-specific | If it matches, cross-side relation is not meaningful. |
| `material_only_matrices` | Build `X,Y` from piece counts/types only, no coordinates | Board geometry matters beyond material | If it matches, model is material shortcut. |
| `role_pool_mean_only` | Replace learned role pooling with side-wise mean/max pooling | Role matrix structure matters | If it matches, role matrix adapter is unnecessary. |
| `singular_values_only` | Use singular values of `C`, remove residual and improvement features | Procrustes residuals add beyond strain spectrum | If it matches, residual diagnostics are not needed. |

## 10. Benchmark And Falsification Criteria

Baselines:

- `configs/bench_cnn_small_simple18.yaml`
- `configs/bench_cnn_medium_simple18.yaml`
- `configs/bench_residual_small_simple18.yaml`
- A matched parameter own/opponent mean-pooling model if available.

Metrics:

- AUROC.
- Accuracy and balanced accuracy.
- F1.
- Calibration.
- Required fine-label `0/1/2 -> predicted 0/1` confusion matrix for main and central ablations.
- Class `1` recall or precision at matched fine-label-`0` false-positive rate where available.

Additional diagnostics:

- Procrustes residual distribution by fine label.
- Alignment improvement `identity_residual - procrustes_residual`.
- Singular values of cross-covariance by fine label.
- Role-pool entropy and role mass by side.
- Delta between main and `separate_matrix_stats_only`.

Success threshold:

- Main model beats the best same-budget `simple_18` CNN/residual baseline by at least `+1.0` AUROC point, or improves class-`1` recall at matched fine-label-`0` FPR by at least `+2.0` points.
- Main beats `separate_matrix_stats_only`, `identity_alignment_only`, and `batch_shuffled_opponent` by at least `+0.5` AUROC point or a clear class-`1` diagnostic gain.

Failure threshold:

- Separate matrix stats, identity residuals, material-only matrices, or batch-shuffled opponent controls match the main model.
- Role pooling collapses so every role sees the same average piece set.

Abandon if:

- Procrustes alignment residuals do not beat separate stats and shuffled opponent controls.

Scale if:

- Main beats central ablations and Procrustes residual distributions separate fine labels `0`, `1`, and `2`.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260424_polar_procrustes/idea.yaml` | Create | Idea metadata copied from machine-readable block. |
| `ideas/20260424_polar_procrustes/math_thesis.md` | Create | Section 6 mathematical thesis. |
| `ideas/20260424_polar_procrustes/architecture.md` | Create | Section 7 architecture details. |
| `ideas/20260424_polar_procrustes/ablations.md` | Create | Section 9 ablations. |
| `ideas/20260424_polar_procrustes/trainer_notes.md` | Create | SVD, batch-size, and diagnostics notes. |
| `src/chess_nn_playground/models/polar_procrustes.py` | Create | Token extractor, role matrix pooler, Procrustes layer, builder. |
| `src/chess_nn_playground/models/registry.py` | Update | Register `polar_procrustes_alignment`. |
| `configs/bench_polar_procrustes_simple18.yaml` | Create | Main config. |
| `configs/bench_polar_procrustes_separate_stats.yaml` | Create | Central ablation config. |
| `configs/bench_polar_procrustes_identity.yaml` | Create | Identity-alignment ablation config. |
| `tests/test_polar_procrustes_forward.py` | Create | Forward shape, finite logits, nonnegative residuals, ablation smoke tests. |

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-24_2104_friday_shanghai_polar_procrustes.md
  generated_at: 2026-04-24 21:04
  weekday: Friday
  timezone: Asia/Shanghai
  idea_slug: polar_procrustes
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260424_polar_procrustes
  name: Polar-Procrustes Alignment Bottleneck
  slug: polar_procrustes
  status: draft
  created_at: 2026-04-24
  author: Codex
  short_thesis: Puzzle-like positions may show distinctive Procrustes alignment residuals and polar strain spectra between learned own/opponent role matrices.
  novelty_claim: Uses polar decomposition and orthogonal Procrustes alignment, not CNN depth, self-attention, determinant volume, Grassmannian angles, matrix pencils, attack graphs, move deltas, OT, topology, or score fields.
  expected_advantage: Captures relative own/opponent role alignment with clean separate-stats and shuffled-opponent falsifiers.
  central_falsification_ablation: separate_matrix_stats_only
  target_task: coarse_binary
  input_representation: simple_18
  output_heads: binary logits
  compute_notes: SVD over role or embedding cross-covariance; use role-space matrix if embedding-space SVD is too heavy.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/bench_polar_procrustes_simple18.yaml
  model_path: src/chess_nn_playground/models/polar_procrustes.py
  latest_result_path: null
  notes: Must run separate-stats, identity-alignment, material-only, and shuffled-opponent controls before scaling.
```

```yaml
config_yaml:
  run:
    name: bench_polar_procrustes_simple18
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
    name: polar_procrustes_alignment
    input_channels: 18
    num_classes: 2
    token_dim: 48
    role_count: 8
    head_hidden: 128
    matrix_space: embedding
    normalize_rows: true
    include_separate_spectra: true
    ablation: none
  training:
    epochs: 3
    batch_size: 256
    num_workers: 0
    learning_rate: 0.001
    weight_decay: 0.0001
    class_weighting: balanced
    early_stopping_patience: 2
    mixed_precision: false
```

```yaml
model_spec:
  model_name: polar_procrustes_alignment
  file_path: src/chess_nn_playground/models/polar_procrustes.py
  builder_function: build_polar_procrustes_alignment_from_config
  input_shape: [batch, 18, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - Simple18OwnOpponentTokenExtractor
    - PieceSquareTokenEncoder
    - RoleMatrixPooler
    - PolarProcrustesLayer
    - PolarProcrustesHead
  required_config_fields:
    - input_channels
    - num_classes
    - token_dim
    - role_count
    - matrix_space
    - normalize_rows
    - ablation
  expected_parameter_count: 60000-150000
  expected_memory_notes: Main SVD is batch * token_dim * token_dim for embedding-space or batch * role_count * role_count for role-space.
```

```yaml
research_continuity:
  idea_fingerprint: own/opponent occupied-piece role matrices + polar decomposition of cross-covariance + Procrustes residual/strain spectrum + binary puzzle-likeness
  already_researched_family_overlap: Adjacent to high-level linear-algebra packets only; not a graph Laplacian, sheaf, move-delta, OT, topology, attention model, or residual CNN.
  closest_duplicate_risk: Could be confused with Grassmannian angles or matrix pencils; distinguish by optimal orthogonal alignment and Procrustes residual against shuffled-opponent controls.
  do_not_repeat_if_this_fails:
    - Polar/Procrustes own-opponent role alignment classifiers with only different role counts, token dimensions, or matrix-space choices.
    - Cross-covariance polar strain spectra over the same side-relative matrices rescued by a larger CNN or attention wrapper.
  suggested_next_search_directions:
    - If Procrustes residual partly helps, test stricter role-pooling regularization before scaling token dimension.
    - If separate stats match, abandon alignment residuals and treat side-wise matrix summaries as the useful part.
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add this packet to imported memory with fingerprint `own/opponent role matrices + polar Procrustes alignment residual`. | Prevents repeats under polar decomposition, orthogonal alignment, or Procrustes terminology. | `Imported Research Memory` |
| Add anti-duplicate wording for Procrustes/polar classifiers unless matrix construction or falsifier changes materially. | Avoids future packets that only change role count or embedding dimension. | Anti-duplicate rules after matrix-pencil/subspace ideas |
| Require `separate_matrix_stats_only`, `identity_alignment_only`, and `batch_shuffled_opponent` controls for future alignment models. | These isolate optimal alignment from separate side summaries and sample-independent shortcuts. | `Ablation Plan` requirements |

## 14. Final Sanity Check

- Stored as a Markdown file in `ideas/research/packets/classic/`: yes
- High-level linear algebra concept selected: yes, polar decomposition and orthogonal Procrustes alignment
- No forbidden engine features used as inputs: yes
- Does not fabricate labels: yes
- Not a routine CNN/ResNet/Transformer variant: yes
- Minimal current-data experiment exists: yes
- Falsification criterion is concrete: yes
- Codex can implement without asking for missing architecture details: yes
- Prompt maintenance notes included for Codex: yes
- Repetition check against imported and local packets completed: yes
- Distinct from Grassmannian principal-angle packet: yes
- Distinct from matrix-pencil packet: yes
- Distinct from determinant/log-volume packet: yes
- Not a tactical sheaf/Hodge variant or one-ply move-delta pooling/spectrum/landscape variant: yes
- Not a current-board piece-target/material-target Sinkhorn/OT transport variant: yes
- Not a deterministic nuisance-orthogonal projection bottleneck variant: yes
- Not an exact near-duplicate of imported ordinal, sparse-witness, ray-language, Mobius-constellation, or pseudo-likelihood packets: yes
- Not an exact near-duplicate of imported orbit-symmetry, tempo-intervention, credal-evidence, rule-partition-invariance, kinematic-commutator, or masked-codec packets: yes
- Not an exact near-duplicate of imported cubical Euler/Betti topology, Hall-defect overload, or king-cage/king-escape path-DP packets: yes
- Not an exact near-duplicate of imported FCA/Galois-closure, denoising-score-field, or non-backtracking-edge-walk packets: yes

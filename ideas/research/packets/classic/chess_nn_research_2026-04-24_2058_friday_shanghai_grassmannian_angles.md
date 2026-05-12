# Codex Handoff Packet: Grassmannian Principal-Angle Bottleneck

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-24_2058_friday_shanghai_grassmannian_angles.md`
- Generated at: 2026-04-24 20:58
- Weekday: Friday
- Timezone: Asia/Shanghai
- Idea slug: `grassmannian_angles`
- Intended next consumer: Codex
- Status: draft research packet, not implemented

## 2. Executive Selection

- Idea name: Grassmannian Principal-Angle Bottleneck
- High-level linear algebra concept: Grassmannians, principal angles between subspaces, and canonical-correlation spectra.
- One-sentence thesis: Puzzle-like positions may be characterized by relative geometry between learned role subspaces, and principal-angle spectra can test this without move generation, engine metadata, attack graphs, or ordinary Transformer attention.
- Idea fingerprint: current-board occupied tokens + learned role-gated covariance subspaces + principal-angle/canonical-correlation spectra + binary puzzle-likeness head.
- Why this is not a common CNN/ResNet/Transformer variant: the central computation is eigendecomposition of learned token covariances and SVD of subspace cross-Gram matrices, not convolution, residual depth, square-token self-attention, or LC0 tower copying.
- Current-data minimal experiment: train on `simple_18` using the existing `crtk_sample_3class` train/val/test splits for 3 epochs, compare with same-budget `simple_18` CNN/residual baselines and the angle-removal ablation.
- Smallest central falsification ablation: keep each role subspace's eigenvalue spectrum and pooled token statistics, but replace all cross-subspace principal-angle spectra with constants or batch-shuffled angle spectra.
- Expected information gain if it fails: a clean failure rules out Grassmannian role-subspace geometry as a useful bottleneck on this split and prevents future variants that only change subspace counts or dimensions.

## 3. Problem Restatement And Data Contract

The task is binary chess puzzle-likeness classification from current board positions:

- output `0`: non-puzzle
- output `1`: puzzle-like

The source fine labels are:

- fine label `0`: known non-puzzle
- fine label `1`: verified near-puzzle
- fine label `2`: verified puzzle

The model trains on the binary target while evaluation keeps the fine-label `0/1/2 -> predicted 0/1` diagnostic matrix.

Allowed neural inputs:

- Current board occupancy from `simple_18`.
- Side-to-move, castling, and en-passant planes already present in the encoding.
- Deterministic square coordinates and side-relative coordinates.
- Learned role gates derived only from current board tokens.

Forbidden neural inputs:

- Stockfish scores, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, dataset provenance, unresolved candidate-pool status, or anything derived from them.
- Engine search, forced-line search, checkmate/stalemate oracles, or future game outcomes.

Tensor contract:

```text
input:          (B, 18, 8, 8)
tokens:         (B, N, F), N <= 32 occupied tokens or N = 64 square tokens
embeddings:     (B, N, D)
role_gates:     (B, R, N)
role_bases:     (B, R, D, K)
angle_spectra:  (B, R_pairs, K)
logits:         (B, 2)
```

First implementation should use occupied-piece tokens because subspace geometry over pieces is more interpretable and cheaper than 64 square-token subspaces.

Leakage checklist:

- Subspaces are built from current-board token embeddings only.
- Fine labels are not model inputs.
- No attack graph, move list, engine feature, source metadata, or legal oracle appears in the model.

## 4. Research Map

External research anchors are conceptual only. No external citation was verified during generation.

| Source or concept | Borrowed | Not copied |
|---|---|---|
| Grassmannian geometry | A `K`-dimensional subspace of `R^D` as a point on the Grassmann manifold `Gr(K, D)`. | No Riemannian optimization loop and no claim that chess positions literally form a smooth manifold. |
| Principal angles between subspaces | Singular values of `Q_a^T Q_b` measure canonical alignment between two orthonormal bases. | No canonical-correlation dataset pairing or statistical CCA objective. |
| Spectral covariance methods | Top eigenvectors of role-gated token covariance define stable token-order-invariant subspaces. | No graph Laplacian spectrum, attack graph, or sheaf operator. |

Candidate search trace:

| Candidate mechanism | Why not selected |
|---|---|
| Plain SVD of the entire board tensor | Too likely to become generic low-rank compression and too insensitive to role structure. |
| Krylov subspace over a learned board operator | Interesting but would need stronger operator-design choices and risks becoming another graph/diffusion model. |
| Schur complement between king-region and non-king-region covariance blocks | Strong linear algebra concept, but more brittle and less clearly implementable than principal angles. |
| Matrix pencil / generalized eigenvalue board classifier | Novel, but hard to define safe matrix pairs without leaking hand-built chess semantics or becoming unstable. |

Concept-to-operator mapping:

| High-level concept | Concrete operator/object in this idea | Tensor contract | Central falsifier | Why not a duplicate |
|---|---|---|---|---|
| Grassmannian subspace | Top-`K` eigenspace of each role-gated token covariance | `(B, R, D, D) -> (B, R, D, K)` | keep eigenvalues, remove cross-role angles | Not determinant log-volume; it compares subspace orientations, not volume. |
| Principal angles | Singular values of `Q_a^T Q_b` for role basis pairs | `(B, D, K) x (B, D, K) -> (B, K)` | batch-shuffle or constant angle spectra | Not attention; no query-token softmax routes. |
| Canonical-correlation spectrum | Sorted cosines of principal angles | `(B, R_pairs, K)` | spectra-only-without-pair-identity | Tests role-pair geometry specifically. |

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN | `src/chess_nn_playground/models/cnn.py` | Already present and tests local learned filters, not subspace angles. |
| Residual CNN | `src/chess_nn_playground/models/residual_cnn.py` | Extra residual capacity is ordinary scaling. |
| LC0-style CNN/residual CNN | Existing 112-plane configs | Too close to engine-network conventions. |
| Vanilla ViT over 64 squares | Common Transformer baseline | Attention capacity is too broad and does not isolate Grassmannian geometry. |
| Plain GNN over square adjacency | Generic graph network | Too close to ordinary message passing and imported graph families. |
| Determinantal volume bottleneck | New local draft packet | Measures log-volume of Gram matrices; this idea measures relative orientation between role subspaces. |
| Mobius/ANOVA constellation | Imported constellation packet | Explicit tuple interactions differ from covariance eigenspaces and principal angles. |
| Hyperparameter tuning | Existing configs | Not a research architecture. |
| Ensembling | Any leaderboard ensemble | Would obscure whether subspace geometry matters. |

## 6. Mathematical Thesis

Let `x` be a current board. Extract occupied tokens:

```text
S(x) = {(t_i, s_i)}_{i=1}^N, N <= 32
```

where `t_i` is piece/color/side-relative type and `s_i` is square coordinate. A token encoder maps each token to `h_i in R^D`. A role gate `g_{r,i}(x) in [0, 1]` defines a weighted covariance:

```text
C_r(x) = sum_i g_{r,i}(x) (h_i - mu_r)(h_i - mu_r)^T + eps I
mu_r = sum_i g_{r,i} h_i / (sum_i g_{r,i} + eps)
```

Let `Q_r(x) in R^{D x K}` be the top-`K` orthonormal eigenvectors of `C_r(x)`. This basis represents a point on the Grassmannian `Gr(K, D)` because the subspace matters, not the particular basis.

For role pair `(r, s)`, define:

```text
M_{r,s}(x) = Q_r(x)^T Q_s(x)
sigma_{r,s,1:K}(x) = singular_values(M_{r,s}(x))
theta_{r,s,j}(x) = arccos(clamp(sigma_{r,s,j}, -1, 1))
```

The `sigma` values are cosines of the principal angles between the two role subspaces.

Core hypothesis:

Puzzle-like positions may not merely have unusual individual pieces or local motifs. They may arrange piece-role evidence so that learned subspaces align, separate, or become nearly orthogonal in ways associated with tactical tension. Principal angles are a direct measure of this relative role geometry.

Proposition:

The principal-angle spectrum between `span(Q_r)` and `span(Q_s)` is invariant to token ordering and to orthonormal basis changes inside each subspace.

Proof sketch:

Token ordering does not change `C_r` because it is a sum. Replacing `Q_r` by `Q_r U` and `Q_s` by `Q_s V` for orthogonal `U, V in R^{K x K}` changes `M` to `U^T M V`, which preserves singular values. Therefore the bottleneck uses subspace geometry rather than arbitrary basis coordinates.

What is actually proven:

- The angle spectra are permutation-invariant over occupied tokens.
- The angle spectra are invariant to basis rotation inside each learned subspace.
- Removing cross-role angle spectra while preserving eigenvalue spectra directly falsifies the claimed role-geometry signal.

What remains hypothesized:

- That the learned role gates discover chess-relevant subspaces.
- That puzzle-like positions have distinctive principal-angle spectra.
- That the current split is not dominated by material/source shortcuts.

Counterexamples:

- Labels are mostly material, phase, or source artifacts.
- The role gates collapse into identical subspaces for all positions.
- Useful signal requires explicit move consequences rather than static subspace geometry.
- Angle spectra are too coarse and lose square-specific tactics.

Self-critique:

This idea is mathematically clean but may be too abstract for chess. Principal angles do not know about legal moves, pins, checks, or forced lines. The experiment is still worth running because the model is small, leakage-safe, and has a decisive ablation: if eigenvalues plus pooled token stats match full principal angles, Grassmannian geometry should be abandoned.

## 7. Architecture Specification

Module names:

- `Simple18OccupiedTokenExtractor`
- `PieceSquareTokenEncoder`
- `RoleGatedCovarianceSubspaces`
- `PrincipalAngleSpectrum`
- `GrassmannianAngleHead`

Forward pass:

1. Decode `simple_18` piece planes into up to 32 occupied tokens and masks.
2. Build token features:
   - piece type one-hot,
   - own/opponent flag relative to side-to-move,
   - color,
   - absolute square coordinates,
   - side-relative square coordinates,
   - castling/en-passant scalars broadcast to tokens.
3. Encode tokens with an MLP:

```text
(B, 32, F) -> (B, 32, D), default D=64
```

4. Compute `R` role gates:

```text
(B, 32, D) -> (B, R, 32), default R=8
```

5. For each role, compute weighted mean and covariance:

```text
C_r in (B, D, D)
```

6. Use `torch.linalg.eigh` to get top `K` eigenvectors:

```text
Q_r in (B, D, K), default K=6
lambda_r in (B, K)
```

7. For each role pair, compute SVD of `Q_r^T Q_s`:

```text
sigma_{r,s} in (B, K)
theta_{r,s} = arccos(sigma)
```

8. Pool features:
   - all principal-angle cosine spectra,
   - all angle spectra,
   - eigenvalue spectra per role,
   - gate mass per role,
   - pairwise angle entropy and min/max angle,
   - optional token count/material summaries for control.
9. MLP head returns `(B, 2)`.

Shapes:

```text
input:          (B, 18, 8, 8)
tokens:         (B, 32, F)
embeddings:     (B, 32, 64)
role_gates:     (B, 8, 32)
covariances:    (B, 8, 64, 64)
role_bases:     (B, 8, 64, 6)
pair_spectra:   (B, 28, 6)
features:       about (B, 350-700)
logits:         (B, 2)
```

Parameter estimate:

- 80k to 180k depending on token encoder and head width.
- Linear algebra buffers are not trainable.

Complexity:

- Covariance construction: `O(B * R * N * D^2)`.
- Eigh per role: `O(B * R * D^3)`, with default `D=64`.
- Pair SVD: `O(B * R^2 * K^3)`, small for `K=6`.

Memory:

- Covariance tensor costs `B * R * D * D` floats.
- With `B=512`, `R=8`, `D=64`, this is about 16.8M floats before autograd overhead. If memory is high, reduce `D` to 48 or compute role covariances in chunks.

Required config fields:

```yaml
model:
  name: grassmannian_principal_angles
  input_channels: 18
  num_classes: 2
  token_dim: 64
  role_count: 8
  subspace_dim: 6
  head_hidden: 128
  covariance_eps: 0.001
  include_eigenvalues: true
  include_angle_diagnostics: true
  ablation: none
```

Encoding adapters:

- `simple_18`: supported first.
- `lc0_static_112`: fail closed unless current-board piece channels and auxiliary semantics are explicitly mapped.
- `lc0_bt4_112`: optional later; deterministic token extraction may use only known current-slot piece planes. History planes should not be used for deterministic token extraction.

Pseudocode:

```python
tokens, mask = extract_occupied_tokens_simple18(x)
h = token_encoder(tokens)
gates = gate_mlp(h).transpose(1, 2).sigmoid()
role_features = []
bases = []
eigvals = []
for r in range(role_count):
    w = gates[:, r] * mask
    mu = weighted_mean(h, w)
    centered = h - mu[:, None, :]
    cov = weighted_cov(centered, w, eps=covariance_eps)
    vals, vecs = torch.linalg.eigh(cov)
    q = vecs[..., -subspace_dim:]
    bases.append(q)
    eigvals.append(vals[..., -subspace_dim:])
spectra = []
for a, b in role_pairs:
    cross = bases[a].transpose(-1, -2) @ bases[b]
    sigma = torch.linalg.svdvals(cross).clamp(0.0, 1.0)
    spectra.append(sigma)
features = build_features(spectra, eigvals, gates, mask, ablation)
return head(features)
```

## 8. Loss, Training, And Regularization

- Primary loss: existing balanced cross entropy on coarse binary labels.
- Auxiliary regularization:
  - Optional role-gate entropy penalty to avoid all roles selecting all tokens.
  - Optional role-diversity penalty on mean gates to prevent collapse into identical role masks.
  - Keep both off in the first benchmark unless collapse is severe.
- Optimizer: AdamW.
- Learning rate: `0.001`.
- Weight decay: `0.0001`.
- Batch size: start with `256` if eigendecomposition memory is high; otherwise `512`.
- Determinism: seed `42`, deterministic mode true. Be aware that some GPU eigensolvers may have nondeterministic kernels; CPU smoke tests are still deterministic.
- Fair comparison: same splits, same 3-epoch budget, same artifacts, same binary target.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| `no_cross_angles` | Keep role eigenvalues, gate masses, token stats; replace principal-angle spectra with constants | Cross-role subspace geometry matters | If it matches, Grassmannian angles are unnecessary. |
| `batch_shuffled_angles` | Shuffle angle spectra across samples within a batch | Angles are sample-specific evidence | If it matches, angle spectra are not tied to the board. |
| `random_role_gates` | Freeze random role gates with same distribution | Learned role subspaces matter | If it matches, learned gating is unnecessary. |
| `pooled_token_head` | Replace subspaces with mean/max token pooling and matched head | Subspace geometry beats ordinary set pooling | If it matches, the linear algebra bottleneck adds no signal. |
| `eigenvalues_only` | Keep within-role covariance spectra but remove cross-role angles | Relative orientation matters beyond variance magnitude | If it matches, only covariance scale matters. |
| `material_only_tokens` | Use only piece type/material features, remove coordinates | Geometry matters beyond material | If it matches, labels may be material dominated. |
| `no_orthonormalization` | Use raw role means and dot products instead of eigenspace bases | Principal-angle invariance matters | If it matches, ordinary dot-product summaries suffice. |

## 10. Benchmark And Falsification Criteria

Baselines:

- `configs/bench_cnn_small_simple18.yaml`
- `configs/bench_cnn_medium_simple18.yaml`
- `configs/bench_residual_small_simple18.yaml`
- A matched parameter occupied-token mean/max pooling model if available.

Metrics:

- AUROC.
- Accuracy and balanced accuracy.
- F1.
- Calibration.
- Required fine-label `0/1/2 -> predicted 0/1` confusion matrix for main and central ablations.
- Class `1` recall or precision at matched fine-label-`0` false-positive rate where tooling supports it.

Additional diagnostics:

- Mean role-gate mass per role.
- Mean pairwise principal-angle spectra by fine label.
- Role collapse checks: average pairwise angle between role subspaces.
- Delta between main and `no_cross_angles`.

Success threshold:

- Main model beats the best same-budget `simple_18` CNN/residual baseline by at least `+1.0` AUROC point, or improves class-`1` recall at matched fine-label-`0` FPR by at least `+2.0` points.
- Main beats `no_cross_angles` and `eigenvalues_only` by at least `+0.5` AUROC point or a clear class-`1` diagnostic gain.

Failure threshold:

- `no_cross_angles`, `eigenvalues_only`, or pooled-token head matches the main model.
- Role gates collapse and angle spectra become nearly constant across samples.

Abandon if:

- Cross-role angle spectra do not beat eigenvalue-only or batch-shuffled controls.

Scale if:

- Main beats the central ablations and angle diagnostics show distinct spectra for fine labels `0`, `1`, and `2`.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260424_grassmannian_angles/idea.yaml` | Create | Idea metadata copied from machine-readable block. |
| `ideas/20260424_grassmannian_angles/math_thesis.md` | Create | Section 6 mathematical thesis. |
| `ideas/20260424_grassmannian_angles/architecture.md` | Create | Section 7 architecture details. |
| `ideas/20260424_grassmannian_angles/ablations.md` | Create | Section 9 ablations. |
| `ideas/20260424_grassmannian_angles/trainer_notes.md` | Create | Batch-size and eigensolver determinism notes. |
| `src/chess_nn_playground/models/grassmannian_angles.py` | Create | Token extractor, covariance subspaces, principal-angle spectra, builder. |
| `src/chess_nn_playground/models/registry.py` | Update | Register `grassmannian_principal_angles`. |
| `configs/bench_grassmannian_angles_simple18.yaml` | Create | Main config. |
| `configs/bench_grassmannian_angles_no_cross.yaml` | Create | Central ablation config. |
| `configs/bench_grassmannian_angles_eigen_only.yaml` | Create | Eigenvalues-only ablation config. |
| `tests/test_grassmannian_angles_forward.py` | Create | Forward shape, finite logits, angle range, ablation smoke tests. |

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-24_2058_friday_shanghai_grassmannian_angles.md
  generated_at: 2026-04-24 20:58
  weekday: Friday
  timezone: Asia/Shanghai
  idea_slug: grassmannian_angles
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260424_grassmannian_angles
  name: Grassmannian Principal-Angle Bottleneck
  slug: grassmannian_angles
  status: draft
  created_at: 2026-04-24
  author: Codex
  short_thesis: Puzzle-like positions may have distinctive principal-angle spectra between learned role-gated occupied-piece subspaces.
  novelty_claim: Uses Grassmannian subspace geometry and canonical-angle spectra, not CNN depth, self-attention, determinant volume, attack graphs, move deltas, OT, topology, or score fields.
  expected_advantage: Captures relative orientation between tactical role subspaces with a clean angle-removal falsifier.
  central_falsification_ablation: no_cross_angles
  target_task: coarse_binary
  input_representation: simple_18
  output_heads: binary logits
  compute_notes: Eigh over role covariance matrices costs batch * role_count * token_dim^3; start with token_dim 48 or 64.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/bench_grassmannian_angles_simple18.yaml
  model_path: src/chess_nn_playground/models/grassmannian_angles.py
  latest_result_path: null
  notes: Must run no-cross-angle, eigenvalues-only, and pooled-token controls before scaling.
```

```yaml
config_yaml:
  run:
    name: bench_grassmannian_angles_simple18
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
    name: grassmannian_principal_angles
    input_channels: 18
    num_classes: 2
    token_dim: 64
    role_count: 8
    subspace_dim: 6
    head_hidden: 128
    covariance_eps: 0.001
    include_eigenvalues: true
    include_angle_diagnostics: true
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
  model_name: grassmannian_principal_angles
  file_path: src/chess_nn_playground/models/grassmannian_angles.py
  builder_function: build_grassmannian_principal_angles_from_config
  input_shape: [batch, 18, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - Simple18OccupiedTokenExtractor
    - PieceSquareTokenEncoder
    - RoleGatedCovarianceSubspaces
    - PrincipalAngleSpectrum
    - GrassmannianAngleHead
  required_config_fields:
    - input_channels
    - num_classes
    - token_dim
    - role_count
    - subspace_dim
    - covariance_eps
    - ablation
  expected_parameter_count: 80000-180000
  expected_memory_notes: Covariance tensor is batch * role_count * token_dim * token_dim floats; reduce token_dim or batch size if needed.
```

```yaml
research_continuity:
  idea_fingerprint: occupied-piece tokens + role-gated covariance eigenspaces + principal-angle/canonical-correlation spectra + binary puzzle-likeness
  already_researched_family_overlap: Adjacent only to determinant volume and generic spectral methods; not a graph Laplacian, sheaf, move-delta, OT, topology, attention, or residual CNN idea.
  closest_duplicate_risk: Could be mistaken for determinant log-volume because both use linear algebra over token embeddings; distinguish by cross-subspace principal angles and no-cross-angle falsifier.
  do_not_repeat_if_this_fails:
    - Grassmannian/principal-angle role-subspace classifiers with only different role counts, token dimensions, or subspace dimensions.
    - Canonical-correlation spectra over the same occupied-token subspaces rescued by a larger CNN or attention wrapper.
  suggested_next_search_directions:
    - If angles partially help, test stronger role-gate constraints before increasing token dimension.
    - If eigenvalues-only matches, treat covariance scale as the useful part and abandon Grassmannian orientation.
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add this packet to imported memory with fingerprint `role-gated covariance eigenspaces + principal-angle spectra`. | Prevents repeats under Grassmannian, CCA, canonical-angle, or subspace-alignment terminology. | `Imported Research Memory` |
| Add anti-duplicate wording for subspace-angle classifiers unless the subspace construction or falsifier is materially different. | Avoids future packets that only change role count or subspace dimension. | Anti-duplicate rules after determinant/log-volume ideas |
| Require `no_cross_angles`, `eigenvalues_only`, and `batch_shuffled_angles` controls for future Grassmannian/subspace models. | These controls isolate cross-subspace geometry from within-subspace variance and generic set pooling. | `Ablation Plan` requirements |

## 14. Final Sanity Check

- Stored as a Markdown file in `ideas/research/packets/classic/`: yes
- High-level linear algebra concept selected: yes, Grassmannians and principal angles
- No forbidden engine features used as inputs: yes
- Does not fabricate labels: yes
- Not a routine CNN/ResNet/Transformer variant: yes
- Minimal current-data experiment exists: yes
- Falsification criterion is concrete: yes
- Codex can implement without asking for missing architecture details: yes
- Prompt maintenance notes included for Codex: yes
- Repetition check against imported and local packets completed: yes
- Distinct from determinant/log-volume packet: yes
- Not a tactical sheaf/Hodge variant or one-ply move-delta pooling/spectrum/landscape variant: yes
- Not a current-board piece-target/material-target Sinkhorn/OT transport variant: yes
- Not a deterministic nuisance-orthogonal projection bottleneck variant: yes
- Not an exact near-duplicate of imported ordinal, sparse-witness, ray-language, Mobius-constellation, or pseudo-likelihood packets: yes
- Not an exact near-duplicate of imported orbit-symmetry, tempo-intervention, credal-evidence, rule-partition-invariance, kinematic-commutator, or masked-codec packets: yes
- Not an exact near-duplicate of imported cubical Euler/Betti topology, Hall-defect overload, or king-cage/king-escape path-DP packets: yes
- Not an exact near-duplicate of imported FCA/Galois-closure, denoising-score-field, or non-backtracking-edge-walk packets: yes

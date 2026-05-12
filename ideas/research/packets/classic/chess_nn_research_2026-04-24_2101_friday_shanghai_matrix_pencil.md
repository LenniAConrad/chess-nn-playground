# Codex Handoff Packet: Matrix-Pencil Generalized Spectrum Bottleneck

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-24_2101_friday_shanghai_matrix_pencil.md`
- Generated at: 2026-04-24 21:01
- Weekday: Friday
- Timezone: Asia/Shanghai
- Idea slug: `matrix_pencil`
- Intended next consumer: Codex
- Status: draft research packet, not implemented

## 2. Executive Selection

- Idea name: Matrix-Pencil Generalized Spectrum Bottleneck
- High-level linear algebra concept: matrix pencils, generalized eigenvalue problems, and generalized Rayleigh quotients.
- One-sentence thesis: Puzzle-like positions may be characterized by directional dominance between two learned board-energy forms, and a generalized eigenvalue spectrum can test that relative geometry more directly than separate covariance spectra.
- Idea fingerprint: current-board occupied tokens + two learned PSD role-energy matrices + generalized eigenvalues of `(A, B)` + Rayleigh quotient diagnostics + binary puzzle-likeness head.
- Why this is not a common CNN/ResNet/Transformer variant: the central computation is solving a small generalized symmetric eigenproblem `A v = lambda B v`, not convolution, residual stacking, square-token self-attention, attack-graph propagation, or LC0 tower copying.
- Current-data minimal experiment: train on `simple_18` using the existing `crtk_sample_3class` train/val/test splits for 3 epochs, compare against same-budget `simple_18` CNN/residual baselines plus `separate_spectra_only`.
- Smallest central falsification ablation: keep eigenvalues and trace/Frobenius summaries of `A` and `B` separately, but remove all generalized eigenvalues and generalized Rayleigh quotient features.
- Expected information gain if it fails: a clean failure rules out generalized relative-spectrum board-energy bottlenecks before trying larger matrix pencils, alternate roles, or nonlinear wrappers.

## 3. Problem Restatement And Data Contract

The task is binary chess puzzle-likeness classification from current board positions:

- output `0`: non-puzzle
- output `1`: puzzle-like

Fine labels:

- `0`: known non-puzzle
- `1`: verified near-puzzle
- `2`: verified puzzle

Fine labels are used only by evaluation diagnostics, not by the neural architecture. Reports should preserve the rectangular fine-label `0/1/2 -> predicted 0/1` matrix.

Allowed neural inputs:

- Current board occupancy from `simple_18`.
- Side-to-move, castling, and en-passant planes already present in `simple_18`.
- Deterministic square coordinates and side-relative coordinates.
- Learned matrix factors derived only from current-board token embeddings.

Forbidden neural inputs:

- Stockfish scores, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, dataset provenance, unresolved candidate status, or anything derived from them.
- Engine search, forced-line search, legal mate/stalemate oracles, future game outcomes, or label-informed masks.

Tensor contract:

```text
input:             (B, 18, 8, 8)
tokens:            (B, N, F), N <= 32 occupied tokens
embeddings:        (B, N, D)
matrix_factors_A:  (B, M, K)
matrix_factors_B:  (B, M, K)
matrices_A_B:      (B, M, M)
gen_eigs:          (B, M)
logits:            (B, 2)
```

First implementation should use occupied-piece tokens, not all 64 squares, to keep matrix solves small and interpretable.

Leakage checklist:

- `A` and `B` are built only from current-board token embeddings.
- No engine/search/source metadata enters the model.
- Fine labels remain evaluation-only.
- No pseudo-legal move generation, attack graph, or checkmate oracle is needed.

## 4. Research Map

External research anchors are conceptual only. No external citation was verified during generation.

| Source or concept | Borrowed | Not copied |
|---|---|---|
| Generalized eigenvalue problem | The spectrum of `A v = lambda B v` measures directions where quadratic form `A` dominates `B`. | No physics simulation, no finite-element PDE, and no external graph Laplacian. |
| Matrix pencils | Treat `(A, B)` as the object, not either matrix alone. | No polynomial eigenvalue problem and no matrix pencil from move/search dynamics. |
| Generalized Rayleigh quotient | `R(v) = (v^T A v) / (v^T B v)` gives an interpretable dominance ratio. | No hand-labeled tactical energy and no engine-derived objective. |

Candidate search trace:

| Candidate mechanism | Why not selected |
|---|---|
| Schur-complement conditional covariance | Strong concept, but more sensitive to block partition choices and matrix inverses. |
| Jordan-form or pseudospectrum classifier | Too numerically unstable and hard to justify for symmetric board features. |
| Singular-value decomposition of one board matrix | Too close to generic spectral compression and less falsifiable than a matrix-pair comparison. |
| Grassmannian principal angles | Already stored as a separate packet; this idea uses generalized dominance spectra, not subspace-angle orientation. |

Concept-to-operator mapping:

| High-level concept | Concrete operator/object in this idea | Tensor contract | Central falsifier | Why not a duplicate |
|---|---|---|---|---|
| Matrix pencil | Learned pair `(A(x), B(x))` of PSD board-energy matrices | `(B, M, M), (B, M, M)` | separate spectra only | Not determinant volume or Grassmannian angles; the relative generalized spectrum is the object. |
| Generalized eigenvalues | Eigenvalues of `B^{-1/2} A B^{-1/2}` with `B` regularized | `(B, M, M) -> (B, M)` | shuffled matrix pairing | Tests whether `A` and `B` relation is sample-specific. |
| Rayleigh quotient | Probed values `(z^T A z)/(z^T B z)` for learned probe directions | `(B, P, M)` | trace-ratio only | Tests directional dominance beyond scalar matrix size. |

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN | `src/chess_nn_playground/models/trunk/cnn.py` | Already present and tests local filters, not matrix-pencil spectra. |
| Residual CNN | `src/chess_nn_playground/models/trunk/residual_cnn.py` | More residual depth is ordinary scaling. |
| LC0-style CNN/residual CNN | Existing 112-plane configs | Too close to engine-network conventions. |
| Vanilla ViT over 64 squares | Common square-token Transformer | Too broad and does not isolate generalized eigenvalue geometry. |
| Plain GNN over board squares | Generic graph network | Ordinary message passing and too near imported graph families. |
| Determinantal volume bottleneck | Local 2026-04-24 packet | Uses log-volume of one Gram object, not relative spectra of a matrix pair. |
| Grassmannian principal-angle bottleneck | Local 2026-04-24 packet | Uses subspace orientation; this uses generalized dominance of quadratic forms. |
| Hyperparameter tuning | Existing configs | Not a research architecture. |
| Ensembling | Any leaderboard ensemble | Would obscure whether the pencil spectrum matters. |

## 6. Mathematical Thesis

Let `x` be a current board and let `S(x) = {(t_i, s_i)}_{i=1}^N` be its occupied piece tokens, with `N <= 32`. A token encoder maps each token to `h_i in R^D`.

The model constructs two low-rank positive semidefinite matrices:

```text
A(x) = U_A(x)^T U_A(x) + eps I_M
B(x) = U_B(x)^T U_B(x) + eps I_M
```

where `U_A, U_B in R^{K x M}` are learned summaries of current-board tokens. Intuitively, `A` and `B` are two learned board-energy forms. They need not correspond to fixed chess concepts, but one can initialize or regularize them toward broad role families such as side-to-move versus opponent, central versus peripheral, or high-mobility-shaped versus low-mobility-shaped token summaries without using legal move generation.

The generalized eigenvalues `lambda_j(x)` solve:

```text
A(x) v_j = lambda_j(x) B(x) v_j
```

Equivalently, since `B` is positive definite after regularization:

```text
C(x) = L_B(x)^{-1} A(x) L_B(x)^{-T}
lambda_j(x) = eigenvalues(C(x))
```

where `B = L_B L_B^T` is the Cholesky factorization.

Core hypothesis:

Puzzle-like positions differ from ordinary positions in directions where one learned board-energy form strongly dominates another. These directional dominance ratios are captured by generalized eigenvalues and generalized Rayleigh quotients:

```text
R_x(v) = (v^T A(x) v) / (v^T B(x) v)
```

A CNN can approximate many functions, but it does not force the model to expose this relative-spectrum object.

Proposition:

For positive definite `B`, the generalized eigenvalues of `(A, B)` are stationary values of the generalized Rayleigh quotient `R_x(v)`.

Proof sketch:

Constrain `v^T B v = 1`. Stationarity of `v^T A v - lambda(v^T B v - 1)` gives:

```text
A v = lambda B v
```

Thus the eigenvalues are the extremal directional dominance ratios between the two learned quadratic forms.

What is actually proven:

- The generalized spectrum measures relative, not absolute, board-energy geometry.
- The generalized eigenvalues are invariant under congruent changes of coordinates applied to both forms.
- The `separate_spectra_only` ablation removes the relative pencil interaction while preserving each matrix's standalone spectrum.

What remains hypothesized:

- That learned `A` and `B` discover useful safe board-energy forms.
- That puzzle-like positions create distinctive generalized dominance ratios.
- That this bottleneck adds information beyond material and token pooling.

Counterexamples:

- Labels are dominated by material, phase, or source artifacts.
- `A` and `B` collapse to nearly proportional matrices, making generalized eigenvalues nearly constant.
- Useful signal requires explicit move consequences rather than static board quadratic forms.
- Separate eigenvalues of `A` and `B` already contain all useful signal.

Self-critique:

This is an abstract linear-algebra bottleneck. Without strong controls, it could become just another learned low-rank MLP feature extractor. The key ablations are therefore mandatory: separate spectra only, shuffled matrix pairing, trace-ratio only, and random factor matrices. If those match the main model, the matrix-pencil idea should be abandoned.

## 7. Architecture Specification

Module names:

- `Simple18OccupiedTokenExtractor`
- `PieceSquareTokenEncoder`
- `LowRankBoardMatrixPair`
- `GeneralizedSpectrumLayer`
- `MatrixPencilHead`

Forward pass:

1. Decode `simple_18` piece planes into up to 32 occupied tokens and masks.
2. Build token features:
   - piece type one-hot,
   - own/opponent relative to side-to-move,
   - color,
   - absolute square coordinates,
   - side-relative square coordinates,
   - castling/en-passant scalar features broadcast to tokens.
3. Encode tokens with an MLP:

```text
(B, 32, F) -> (B, 32, D), default D=64
```

4. Produce two low-rank matrix factor sets:

```text
U_A = factor_A(tokens, mask) -> (B, K, M)
U_B = factor_B(tokens, mask) -> (B, K, M)
```

Default `K=16`, `M=16`.

One simple implementation:

```text
token_weights_A: (B, 32, K)
token_values_A:  (B, 32, M)
U_A[k, :] = sum_i weight_A[i, k] * value_A[i, :]
```

and similarly for `B`.

5. Build regularized PSD matrices:

```text
A = U_A^T U_A / K + eps I
B = U_B^T U_B / K + eps I
```

6. Compute generalized eigenvalues:

```text
L = cholesky(B)
Y = solve_triangular(L, A)
C = solve_triangular(L, Y.transpose(-1, -2)).transpose(-1, -2)
eigvals = eigvalsh(C)
```

A simpler and stable implementation can use:

```text
B_inv_A = cholesky_solve(A, L)
C_sym = 0.5 * (B_inv_A + B_inv_A.transpose(-1, -2))
eigvals = eigvalsh(C_sym)
```

but the whitened symmetric form is preferred.

7. Build feature vector:
   - sorted generalized eigenvalues,
   - log generalized eigenvalues,
   - spread `max - min`,
   - condition-like ratio `max / min`,
   - trace ratio `tr(A) / tr(B)`,
   - separate eigenvalues of `A` and `B`,
   - Frobenius norms,
   - optional learned probe Rayleigh quotients.
8. MLP head returns `(B, 2)` logits.

Shapes:

```text
input:          (B, 18, 8, 8)
tokens:         (B, 32, F)
embeddings:     (B, 32, 64)
U_A, U_B:       (B, 16, 16)
A, B:           (B, 16, 16)
gen_eigs:       (B, 16)
features:       about (B, 100-220)
logits:         (B, 2)
```

Parameter estimate:

- 70k to 160k depending on token encoder and head width.
- Matrix operations have no learned parameters.

Complexity:

- Factor construction: `O(B * N * K * M)`.
- Matrix construction: `O(B * K * M^2)`.
- Cholesky/eigendecomposition: `O(B * M^3)`, small for `M=16`.

Required config fields:

```yaml
model:
  name: matrix_pencil_generalized_spectrum
  input_channels: 18
  num_classes: 2
  token_dim: 64
  factor_rank: 16
  matrix_dim: 16
  head_hidden: 128
  matrix_eps: 0.001
  include_separate_spectra: true
  include_rayleigh_probes: true
  probe_count: 8
  ablation: none
```

Encoding adapters:

- `simple_18`: supported first.
- `lc0_static_112`: fail closed unless current-board piece channels and auxiliary semantics are explicitly mapped.
- `lc0_bt4_112`: optional later, using only known current-slot piece planes for deterministic token extraction. History planes should not drive deterministic matrix construction unless a separate learned adapter is documented.

Pseudocode:

```python
tokens, mask = extract_occupied_tokens_simple18(x)
h = token_encoder(tokens)
wa = torch.softmax(weight_a(h).masked_fill(~mask[..., None], -inf), dim=1)
wb = torch.softmax(weight_b(h).masked_fill(~mask[..., None], -inf), dim=1)
va = value_a(h)
vb = value_b(h)
u_a = torch.einsum("bnk,bnm->bkm", wa, va)
u_b = torch.einsum("bnk,bnm->bkm", wb, vb)
a = u_a.transpose(-1, -2) @ u_a / factor_rank + eps * eye
b = u_b.transpose(-1, -2) @ u_b / factor_rank + eps * eye
gen = generalized_eigvals_symmetric(a, b)
features = build_pencil_features(a, b, gen, ablation)
return head(features)
```

## 8. Loss, Training, And Regularization

- Primary loss: existing balanced cross entropy on coarse binary labels.
- Optimizer: AdamW.
- Learning rate: `0.001`.
- Weight decay: `0.0001`.
- Batch size: `512` should be feasible because matrices are only `16 x 16`; reduce to `256` if GPU memory is tight.
- Regularization:
  - optional penalty on `condition_number(B)` or minimum eigenvalue instability;
  - optional diversity penalty to keep `A` and `B` from becoming proportional;
  - keep optional penalties off for first benchmark unless numerical instability appears.
- Determinism: seed `42`, deterministic true. Cholesky/eig operations should be checked in CPU smoke tests.
- Fair comparison: same splits, same binary target, same 3-epoch budget, same artifact pipeline.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| `separate_spectra_only` | Keep eigenvalues/norms/traces of `A` and `B`, remove generalized eigenvalues | Relative matrix-pencil geometry matters | If it matches, the pencil is unnecessary. |
| `trace_ratio_only` | Use only `tr(A)/tr(B)` and scalar norms | Directional dominance matters beyond scalar size | If it matches, generalized eigenspectrum is overkill. |
| `batch_shuffled_B` | Pair each `A(x)` with `B(x')` from another sample in batch | Sample-specific relation between forms matters | If it matches, `A` and `B` are independent shortcuts. |
| `random_factors` | Freeze factor builders randomly and train only the head | Learned matrix factors matter | If it matches, the learned board matrices are unnecessary. |
| `single_matrix_spectrum` | Use only eigenvalues of one learned PSD matrix | A matrix pair matters beyond one spectral object | If it matches, generalized comparison is unnecessary. |
| `mean_pool_head` | Replace matrices with mean/max token pooling and matched MLP | Pencil beats ordinary set pooling | If it matches, linear algebra bottleneck adds no signal. |
| `material_only_tokens` | Remove coordinates and use only material/piece identity features | Board geometry matters | If it matches, labels may be material dominated. |

## 10. Benchmark And Falsification Criteria

Baselines:

- `configs/bench_cnn_small_simple18.yaml`
- `configs/bench_cnn_medium_simple18.yaml`
- `configs/bench_residual_small_simple18.yaml`
- A matched parameter occupied-token pooling model if available.

Metrics:

- AUROC.
- Accuracy and balanced accuracy.
- F1.
- Calibration.
- Required fine-label `0/1/2 -> predicted 0/1` confusion matrix for the main model and central ablations.
- Class `1` recall or precision at matched fine-label-`0` false-positive rate where available.

Additional diagnostics:

- Mean generalized eigenvalue spectrum by fine label.
- Condition number of `B`.
- Proportionality diagnostic `||A / tr(A) - B / tr(B)||_F`.
- Delta between main and `separate_spectra_only`.
- Delta between main and `batch_shuffled_B`.

Success threshold:

- Main model beats the best same-budget `simple_18` CNN/residual baseline by at least `+1.0` AUROC point, or improves class-`1` recall at matched fine-label-`0` FPR by at least `+2.0` points.
- Main beats `separate_spectra_only` and `trace_ratio_only` by at least `+0.5` AUROC point or a clear class-`1` diagnostic gain.

Failure threshold:

- `separate_spectra_only`, `trace_ratio_only`, `single_matrix_spectrum`, or mean-pool head matches the main model.
- `batch_shuffled_B` does not hurt performance.
- `A` and `B` become nearly proportional for most samples.

Abandon if:

- The generalized eigenspectrum does not beat separate spectra and shuffled-pair controls.

Scale if:

- The generalized spectrum beats central controls and the diagnostic spectrum separates fine labels `0`, `1`, and `2`.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260424_matrix_pencil/idea.yaml` | Create | Idea metadata copied from machine-readable block. |
| `ideas/20260424_matrix_pencil/math_thesis.md` | Create | Section 6 mathematical thesis. |
| `ideas/20260424_matrix_pencil/architecture.md` | Create | Section 7 architecture details. |
| `ideas/20260424_matrix_pencil/ablations.md` | Create | Section 9 ablations. |
| `ideas/20260424_matrix_pencil/trainer_notes.md` | Create | Numerical stability and diagnostic notes. |
| `src/chess_nn_playground/models/matrix_pencil.py` | Create | Token extractor, PSD matrix pair builder, generalized spectrum layer, builder. |
| `src/chess_nn_playground/models/registry.py` | Update | Register `matrix_pencil_generalized_spectrum`. |
| `configs/bench_matrix_pencil_simple18.yaml` | Create | Main config. |
| `configs/bench_matrix_pencil_separate_spectra.yaml` | Create | Central ablation config. |
| `configs/bench_matrix_pencil_shuffled_b.yaml` | Create | Batch-shuffled `B` ablation config. |
| `tests/test_matrix_pencil_forward.py` | Create | Forward shape, finite logits, positive eigenvalue checks, ablation smoke tests. |

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-24_2101_friday_shanghai_matrix_pencil.md
  generated_at: 2026-04-24 21:01
  weekday: Friday
  timezone: Asia/Shanghai
  idea_slug: matrix_pencil
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260424_matrix_pencil
  name: Matrix-Pencil Generalized Spectrum Bottleneck
  slug: matrix_pencil
  status: draft
  created_at: 2026-04-24
  author: Codex
  short_thesis: Puzzle-like positions may have distinctive generalized eigenvalue spectra between two learned current-board PSD energy forms.
  novelty_claim: Uses matrix pencils and generalized Rayleigh spectra, not CNN depth, self-attention, determinant volume, Grassmannian angles, attack graphs, move deltas, OT, topology, or score fields.
  expected_advantage: Captures directional dominance of one learned board-energy form relative to another with a clean separate-spectra falsifier.
  central_falsification_ablation: separate_spectra_only
  target_task: coarse_binary
  input_representation: simple_18
  output_heads: binary logits
  compute_notes: Generalized eigensolve over 16x16 matrices is cheap; monitor B condition number.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/bench_matrix_pencil_simple18.yaml
  model_path: src/chess_nn_playground/models/matrix_pencil.py
  latest_result_path: null
  notes: Must run separate-spectra, trace-ratio, and shuffled-B controls before scaling.
```

```yaml
config_yaml:
  run:
    name: bench_matrix_pencil_simple18
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
    name: matrix_pencil_generalized_spectrum
    input_channels: 18
    num_classes: 2
    token_dim: 64
    factor_rank: 16
    matrix_dim: 16
    head_hidden: 128
    matrix_eps: 0.001
    include_separate_spectra: true
    include_rayleigh_probes: true
    probe_count: 8
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
  model_name: matrix_pencil_generalized_spectrum
  file_path: src/chess_nn_playground/models/matrix_pencil.py
  builder_function: build_matrix_pencil_generalized_spectrum_from_config
  input_shape: [batch, 18, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - Simple18OccupiedTokenExtractor
    - PieceSquareTokenEncoder
    - LowRankBoardMatrixPair
    - GeneralizedSpectrumLayer
    - MatrixPencilHead
  required_config_fields:
    - input_channels
    - num_classes
    - token_dim
    - factor_rank
    - matrix_dim
    - matrix_eps
    - ablation
  expected_parameter_count: 70000-160000
  expected_memory_notes: Main matrices are batch * 2 * matrix_dim * matrix_dim floats; default matrix_dim 16 is small.
```

```yaml
research_continuity:
  idea_fingerprint: occupied-piece tokens + learned PSD matrix pair + generalized eigenvalue/Rayleigh quotient spectrum + binary puzzle-likeness
  already_researched_family_overlap: Adjacent to determinant volume and Grassmannian subspace ideas only at the broad linear-algebra level; not a graph Laplacian, sheaf, move-delta, OT, topology, attention, or residual CNN idea.
  closest_duplicate_risk: Could be confused with Grassmannian angles or determinant log-volume; distinguish by generalized eigenvalues of a matrix pair and separate-spectra/shuffled-B falsifiers.
  do_not_repeat_if_this_fails:
    - Matrix-pencil/generalized-eigenvalue board-energy classifiers with only different matrix dimensions, factor ranks, or token encoders.
    - Rayleigh quotient spectra over the same occupied-token PSD matrix pairs rescued by a larger CNN or attention wrapper.
  suggested_next_search_directions:
    - If generalized spectra partly help, test constrained constructions of A and B before increasing matrix dimension.
    - If separate spectra match, abandon matrix pencils and treat single-matrix spectral summaries as the useful part.
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add this packet to imported memory with fingerprint `learned PSD matrix pair + generalized eigenvalue spectrum`. | Prevents repeats under matrix pencil, generalized Rayleigh, or relative spectrum terminology. | `Imported Research Memory` |
| Add anti-duplicate wording for matrix-pencil classifiers unless the matrix-pair construction or falsifier changes materially. | Avoids future packets that only change matrix dimension or factor rank. | Anti-duplicate rules after Grassmannian/subspace ideas |
| Require `separate_spectra_only`, `trace_ratio_only`, and `batch_shuffled_B` controls for future matrix-pencil models. | These isolate the generalized relative spectrum from standalone matrix size and sample-independent shortcuts. | `Ablation Plan` requirements |

## 14. Final Sanity Check

- Stored as a Markdown file in `ideas/research/packets/classic/`: yes
- High-level linear algebra concept selected: yes, matrix pencils and generalized eigenvalues
- No forbidden engine features used as inputs: yes
- Does not fabricate labels: yes
- Not a routine CNN/ResNet/Transformer variant: yes
- Minimal current-data experiment exists: yes
- Falsification criterion is concrete: yes
- Codex can implement without asking for missing architecture details: yes
- Prompt maintenance notes included for Codex: yes
- Repetition check against imported and local packets completed: yes
- Distinct from Grassmannian principal-angle packet: yes
- Distinct from determinant/log-volume packet: yes
- Not a tactical sheaf/Hodge variant or one-ply move-delta pooling/spectrum/landscape variant: yes
- Not a current-board piece-target/material-target Sinkhorn/OT transport variant: yes
- Not a deterministic nuisance-orthogonal projection bottleneck variant: yes
- Not an exact near-duplicate of imported ordinal, sparse-witness, ray-language, Mobius-constellation, or pseudo-likelihood packets: yes
- Not an exact near-duplicate of imported orbit-symmetry, tempo-intervention, credal-evidence, rule-partition-invariance, kinematic-commutator, or masked-codec packets: yes
- Not an exact near-duplicate of imported cubical Euler/Betti topology, Hall-defect overload, or king-cage/king-escape path-DP packets: yes
- Not an exact near-duplicate of imported FCA/Galois-closure, denoising-score-field, or non-backtracking-edge-walk packets: yes

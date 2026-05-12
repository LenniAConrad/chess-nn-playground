# Codex Handoff Packet: Determinantal Tactical Volume Bottleneck

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-24_2044_friday_shanghai_determinantal_volume.md`
- Generated at: 2026-04-24 20:44
- Weekday: Friday
- Timezone: Asia/Shanghai
- Idea slug: `determinantal_volume`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: Determinantal Tactical Volume Bottleneck
- One-sentence thesis: Puzzle-like positions may concentrate the occupied pieces into a low-dimensional learned tactical subspace, and a log-determinant bottleneck can test that hypothesis without move generation, engine labels, attack graphs, or ordinary CNN capacity scaling.
- Idea fingerprint: current-board occupied piece tokens + learned role-gated PSD kernels + logdet/eigenvalue volume summaries + binary puzzle-likeness head.
- Why this is not a common CNN/ResNet/Transformer variant: the central computation is a determinant of a role-gated Gram matrix over occupied tokens, not convolution, residual depth, square-token attention, or LC0 plane copying.
- Current-data minimal experiment: train on `simple_18` using `data/splits/crtk_sample_3class/{split_train,split_val,split_test}.parquet` for 3 epochs, compare with the existing small/medium `simple_18` CNN and residual CNN baselines, and inspect binary metrics plus the required fine-label `0/1/2 -> predicted 0/1` matrices.
- Smallest central falsification ablation: replace each role log-determinant with a diagonal-only trace feature that preserves token gates, token norms, material, and role marginals but removes all determinant volume interaction.
- Expected information gain if it fails: a clean failure says that global occupied-token volume collapse/diversity is not a useful inductive bottleneck on the current split, ruling out a family of DPP/logdet-style tactical concentration models before trying larger versions.

## 3. Problem Restatement And Data Contract

Task: classify a chess position as binary non-puzzle (`0`) versus puzzle-like (`1`). The source fine labels are `0` known non-puzzle, `1` verified near-puzzle, and `2` verified puzzle. The model trains on coarse binary labels while reports retain the rectangular fine-label diagnostic matrix.

Allowed neural inputs:

- Current board piece occupancy from `simple_18`.
- Side-to-move, castling, and en-passant planes already present in the encoding.
- Deterministic square coordinates and side-relative coordinates.
- Deterministic token masks derived only from current board occupancy.

Forbidden neural inputs:

- Stockfish scores, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, dataset provenance, unresolved candidate-pool status, or any feature computed from them.
- Legal move counts, checkmate/stalemate oracles, engine search, forced lines, or future game outcomes.

Tensor contract:

- Input: `(batch, C, 8, 8)`, first implementation `C=18`.
- Occupied-token extraction: `(batch, N_max=32, token_features)`.
- Output logits: `(batch, 2)`.

Leakage checklist:

- The determinant kernel sees only current occupied pieces, square coordinates, side-to-move, and optional castling/en-passant planes.
- Fine labels are used only by the trainer/evaluation pipeline, not by the model.
- No source or engine metadata is part of token features.

## 4. Research Map

External research anchors are conceptual only. No external citation was verified during generation.

| Source or concept | Borrowed | Not copied |
|---|---|---|
| Determinantal point processes / logdet diversity objectives, unverified reference: Kulesza and Taskar, "Determinantal Point Processes for Machine Learning" | The idea that `log det(I + K_S)` measures volume/diversity of a selected set under a PSD kernel. | No probabilistic DPP sampling objective, no MAP subset selection, and no claim that chess positions are DPP draws. |
| Kernel methods and Gram determinants | The determinant of a token Gram matrix as a coordinate-free test of rank, collapse, and interaction among occupied pieces. | No SVM, Gaussian process, or kernel classifier. |
| Neural set functions | The occupied pieces are treated as a variable-size set with permutation-invariant readout. | No DeepSets average-pooling-only bottleneck and no attention over all square tokens as the core operator. |

Candidate search trace:

| Candidate mechanism | Why not selected |
|---|---|
| Low-rank tensor factorization over occupied triples | Too close to the imported Mobius/ANOVA constellation family. |
| Pairwise tactical payoff game with replicator dynamics | Interesting, but a learned payoff matrix over occupied pieces is less directly falsifiable than deleting logdet interactions. |
| Spectral graph features over attack edges | Too close to imported sheaf/Hodge/graph and non-backtracking edge-walk packets. |
| Learned square-token Transformer with determinant regularizer | The Transformer would dominate the mechanism and make the determinant hard to falsify. |

Concept-to-operator mapping:

| High-level concept | Concrete operator/object in this idea | Tensor contract | Central falsifier | Why not a duplicate |
|---|---|---|---|---|
| Log-volume / determinant | Role-gated Gram matrix `K_r = D_r Phi A_r A_r^T Phi^T D_r + eps I` over occupied tokens | `(B, R, N, N) -> (B, R, stats)` | diagonal-only trace with same gates and norms | Not sheaf/graph, not move-delta, not OT, not Mobius degree-2/3 enumeration. |
| Set bottleneck | Fixed maximum 32 occupied piece tokens with masks | `(B, C, 8, 8) -> (B, 32, F)` | material-only token summary | It keeps a determinant interaction rather than simple pooling. |

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN | `src/chess_nn_playground/models/cnn.py` | Already present and tests learned local texture, not determinant volume collapse. |
| Residual CNN | `src/chess_nn_playground/models/residual_cnn.py` | Extra residual capacity is ordinary scaling. |
| LC0-style CNN/residual CNN | Existing `lc0_bt4_112` configs | Too close to copied engine-network input conventions and baseline towers. |
| Vanilla ViT over 64 squares | Common square-token Transformer | Too generic and likely to become attention capacity rather than a falsifiable chess mechanism. |
| Plain GNN on board adjacency | Generic grid graph network | Too close to ordinary local message passing. |
| Attack-defense graph spectral network | Imported sheaf/Hodge/non-backtracking region | Already researched as structured tactical graphs. |
| One-ply move-delta set pooling | Imported move-delta packets | It enumerates move consequences, which this idea deliberately avoids. |
| Hyperparameter tuning | Existing configs | Not a research architecture. |
| Ensembling | Any leaderboard ensemble | Would hide whether the determinant bottleneck helps. |

## 6. Mathematical Thesis

Let `x` be a current-board tensor. Let `S(x) = {(t_i, s_i)}_{i=1}^N` be the occupied piece set, where `t_i` is piece/color/side-to-move-relative type and `s_i` is square coordinate. A token encoder maps each piece to `phi_i in R^d`. For role `r`, a nonnegative gate `g_{ri}` and role matrix `A_r in R^{d x q}` define:

```text
K_r(x) = D_r Phi A_r A_r^T Phi^T D_r + eps I_N
D_r = diag(sqrt(g_{r1}), ..., sqrt(g_{rN}))
V_r(x) = log det K_r(x)
```

The hypothesis is that puzzle-like positions differ from non-puzzles not only by local motifs, but by how occupied pieces collapse into, or span, learned tactical role subspaces. A determinant is sensitive to joint linear dependence among all gated tokens: if many tokens become redundant under a role kernel, the volume shrinks; if several independent tactical factors coexist, the volume grows.

Proposition: for fixed gates and PSD role kernel, `log det(K_r)` is invariant to permutation of the occupied tokens and changes only through the spectrum of the role-gated token covariance. Therefore it cannot use token ordering and cannot reduce to a single local convolutional filter.

Proof sketch: token permutation conjugates `K_r` by a permutation matrix `P`. Since `det(P K_r P^T) = det(K_r)`, `V_r` is set-invariant. The determinant equals the product of eigenvalues, so it responds to rank and volume rather than only sums of token features.

What is actually proven:

- The bottleneck is permutation-invariant over occupied tokens.
- The central feature uses joint interactions through the determinant.
- Removing off-diagonal kernel entries removes volume interactions while preserving diagonal token magnitudes.

What remains hypothesized:

- That puzzle-likeness correlates with learned role-volume collapse or diversity.
- That the current split is not dominated by source artifacts that a CNN can exploit more easily.

Counterexamples:

- Datasets where labels are mostly material imbalance, opening phase, or source artifacts.
- Puzzles requiring exact one-ply move consequences rather than static piece-role geometry.
- Positions where determinant features are swamped by material count unless ablations are strict.

Self-critique: a flexible token encoder could learn class shortcuts before the determinant matters. The smallest ablation specifically preserves gates and norms while deleting off-diagonal determinant interactions, so the first experiment can distinguish true log-volume signal from token-summary signal.

## 7. Architecture Specification

Module names:

- `Simple18OccupiedTokenExtractor`
- `PieceSquareTokenEncoder`
- `RoleGatedPSDVolume`
- `DeterminantalVolumeHead`

Forward pass:

1. Decode `simple_18` piece planes into up to 32 occupied tokens and a boolean mask.
2. Build token features: piece type one-hot, own/opponent flag, color, side-to-move-relative square coordinates, absolute square coordinates, and optional castling/en-passant scalars broadcast to tokens.
3. Encode tokens with a small MLP: `(B, 32, F) -> (B, 32, d)`, default `d=48`.
4. Compute role gates with sigmoid MLP: `(B, 32, d) -> (B, R, 32)`, default `R=8`.
5. For each role, project tokens through `A_r`: `(B, R, 32, q)`, default `q=16`.
6. Build `K_r = Z_r Z_r^T + eps I` using masked tokens.
7. Compute logdet, trace, normalized logdet per active token, top eigenvalue ratio, and gate mass per role.
8. Feed role statistics to an MLP head returning `(B, 2)`.

Shapes:

```text
input:          (B, 18, 8, 8)
tokens:         (B, 32, F)
token_embed:    (B, 32, 48)
role_gates:     (B, 8, 32)
role_proj:      (B, 8, 32, 16)
gram:           (B, 8, 32, 32)
stats:          (B, 8 * 6)
logits:         (B, 2)
```

Parameter estimate: 60k to 140k depending on token hidden size, role count, and head width.

Complexity: `O(B * R * N^3)` for Cholesky/logdet with `N <= 32`, plus `O(B * R * N * q^2)` projections. This is small relative to normal convolutional towers.

Required config fields:

```yaml
model:
  name: determinantal_tactical_volume
  input_channels: 18
  num_classes: 2
  token_dim: 48
  role_count: 8
  role_rank: 16
  head_hidden: 128
  determinant_eps: 0.001
  ablation: none
```

Encoding adapters:

- `simple_18`: supported first; exact piece planes are known.
- `lc0_static_112`: fail closed unless current-board piece-channel semantics are explicitly mapped.
- `lc0_bt4_112`: later support may use only the current history slot if its plane mapping is explicit; older history planes must not drive deterministic token extraction.

Pseudocode:

```python
tokens, mask = extract_occupied_tokens_simple18(x)
h = token_encoder(tokens)
gates = gate_mlp(h).transpose(1, 2)
stats = []
for r in range(role_count):
    z = role_projectors[r](h) * gates[:, r, :, None].sqrt()
    z = z * mask[:, :, None]
    gram = z @ z.transpose(-1, -2)
    gram = gram + determinant_eps * eye32 + inactive_token_mask(mask)
    stats.append(volume_stats(gram, gates[:, r], mask))
return head(torch.cat(stats, dim=-1))
```

## 8. Loss, Training, And Regularization

- Primary loss: existing coarse-binary cross entropy with balanced class weighting.
- Auxiliary loss: optional small entropy penalty on role gates to avoid all roles selecting all pieces; default off for the first benchmark.
- Batch size: 512 should be feasible.
- Optimizer: AdamW, learning rate `0.001`, weight decay `0.0001`.
- Determinism: keep seed `42`, deterministic mode true, and no stochastic token sampling in the main model.
- Fair comparison: same splits, same 3-epoch budget, same reporting artifacts, same binary target as baseline configs.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| `diagonal_trace_only` | Replace logdet/eigen stats with diagonal trace and gate masses using the same token encoder and gates | Off-diagonal determinant volume carries signal | If it matches, determinant interactions are unnecessary. |
| `random_orthogonal_role_kernel` | Freeze random role projections with matched rank/norm | Learned tactical subspace matters | If it matches, learning role kernels is not important. |
| `material_square_shuffle` | Shuffle square coordinates among same piece types within each sample | Geometry of piece placement matters beyond material | If it matches, the model likely uses material/type shortcuts. |
| `material_only_tokens` | Remove coordinates, keep piece type/color/material and side-to-move | Spatial role volume matters | If it matches, the idea collapsed into material summary. |
| `mean_pool_set_head` | Replace determinant stats with mean/max pooled token embeddings at matched parameter count | Log-volume bottleneck beats ordinary set pooling | If it matches, DPP-style volume is not adding useful structure. |

## 10. Benchmark And Falsification Criteria

Compare against:

- `configs/bench_cnn_small_simple18.yaml`
- `configs/bench_cnn_medium_simple18.yaml`
- `configs/bench_residual_small_simple18.yaml`
- any existing `simple_18` residual result with the same split and epoch budget

Inspect:

- AUROC, accuracy, balanced accuracy, F1, calibration.
- Fine-label `0/1/2 -> predicted 0/1` confusion matrix for main model and central ablations.
- Class `1` recall at matched fine-label-`0` false-positive rate if threshold tooling is available.

Success threshold:

- Main model improves over the best same-budget `simple_18` baseline by at least `+1.0` AUROC point or improves class-`1` recall by at least `+2.0` points at matched fine-label-`0` FPR.
- The main model beats `diagonal_trace_only` by at least `+0.5` AUROC point or a clear class-`1` diagnostic gain.

Failure threshold:

- Main model trails the small CNN by more than `1.0` AUROC point and the diagonal-only ablation is equal or better.

Abandon if:

- `material_only_tokens` or `diagonal_trace_only` matches the main model under the same budget.

Scale if:

- Main beats both ordinary set pooling and diagonal-only trace while retaining clean fine-label diagnostics.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260424_determinantal_volume/idea.yaml` | Create | Idea metadata copied from machine-readable block below. |
| `ideas/20260424_determinantal_volume/math_thesis.md` | Create | Section 6 thesis. |
| `ideas/20260424_determinantal_volume/architecture.md` | Create | Section 7 architecture. |
| `ideas/20260424_determinantal_volume/ablations.md` | Create | Section 9 ablations. |
| `src/chess_nn_playground/models/determinantal_volume.py` | Create | Token extractor, logdet bottleneck, builder function. |
| `src/chess_nn_playground/models/registry.py` | Update | Register `determinantal_tactical_volume`. |
| `configs/bench_determinantal_volume_simple18.yaml` | Create | Main training config. |
| `configs/bench_determinantal_volume_trace_ablation.yaml` | Create | Central ablation config. |
| `tests/test_determinantal_volume_forward.py` | Create | Forward shape, finite logits, permutation-invariance smoke test over token order if extractor exposes a lower-level API. |

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-24_2044_friday_shanghai_determinantal_volume.md
  generated_at: 2026-04-24 20:44
  weekday: Friday
  timezone: Asia/Shanghai
  idea_slug: determinantal_volume
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260424_determinantal_volume
  name: Determinantal Tactical Volume Bottleneck
  slug: determinantal_volume
  status: draft
  created_at: 2026-04-24
  author: Codex
  short_thesis: Puzzle-like positions may differ by learned occupied-piece role-volume collapse or diversity measured through log-determinants of role-gated PSD token kernels.
  novelty_claim: Uses determinant/log-volume bottlenecks over occupied pieces, not CNNs, Transformers, attack sheaves, one-ply move deltas, optimal transport, topology, FCA, score fields, or non-backtracking walks.
  expected_advantage: A small, permutation-invariant global interaction bottleneck may isolate tactical concentration that local CNNs and simple set pooling do not express cleanly.
  central_falsification_ablation: diagonal_trace_only
  target_task: coarse_binary
  input_representation: simple_18
  output_heads: binary logits
  compute_notes: O(batch * roles * 32^3) logdet bottleneck; small because occupied pieces are capped at 32.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/bench_determinantal_volume_simple18.yaml
  model_path: src/chess_nn_playground/models/determinantal_volume.py
  latest_result_path: null
  notes: First experiment should include diagonal-only and material-only controls.
```

```yaml
config_yaml:
  run:
    name: bench_determinantal_volume_simple18
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
    name: determinantal_tactical_volume
    input_channels: 18
    num_classes: 2
    token_dim: 48
    role_count: 8
    role_rank: 16
    head_hidden: 128
    determinant_eps: 0.001
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
  model_name: determinantal_tactical_volume
  file_path: src/chess_nn_playground/models/determinantal_volume.py
  builder_function: build_determinantal_tactical_volume_from_config
  input_shape: [batch, 18, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - Simple18OccupiedTokenExtractor
    - PieceSquareTokenEncoder
    - RoleGatedPSDVolume
    - DeterminantalVolumeHead
  required_config_fields:
    - input_channels
    - num_classes
    - token_dim
    - role_count
    - role_rank
    - determinant_eps
    - ablation
  expected_parameter_count: 60000-140000
  expected_memory_notes: Gram tensor is batch * role_count * 32 * 32 floats.
```

```yaml
research_continuity:
  idea_fingerprint: occupied piece tokens + role-gated PSD Gram matrices + logdet/eigen volume bottleneck + binary puzzle-likeness
  already_researched_family_overlap: Adjacent only to generic set functions and determinant diversity objectives; not a sheaf, graph, move-delta, OT, topology, FCA, score-field, or non-backtracking model.
  closest_duplicate_risk: Could be mistaken for Mobius/ANOVA constellations because determinant expands into high-order interactions; distinguish by the low-rank PSD log-volume bottleneck and diagonal-only falsifier.
  do_not_repeat_if_this_fails:
    - DPP/logdet occupied-piece volume bottlenecks with only different role counts or token dimensions.
    - Determinant kernels over the same current-board occupied-token set rescued by a larger CNN wrapper.
  suggested_next_search_directions:
    - Non-determinant matrix functionals with stronger chess semantics only if the determinant ablation shows partial signal.
    - Data-source artifact controls before scaling if material-only ablations are competitive.
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add this packet to imported memory if implemented or rejected, with fingerprint `occupied-token role-gated PSD logdet volume`. | Prevents future near-duplicates such as "DPP chess bottleneck with more roles." | `Imported Research Memory` |
| Add an anti-duplicate rule blocking determinant/log-volume token kernels unless the central matrix functional or falsifier is genuinely different. | Avoids repeating the same determinant idea under diversity, rank, or volume terminology. | Anti-duplicate rules after Mobius/ANOVA warning |
| Require diagonal-only and material/coordinate-shuffle controls for determinant or matrix-functional set models. | The easiest shortcut is material/type summary without true off-diagonal interaction. | `Ablation Plan` requirements |

## 14. Final Sanity Check

- Downloadable Markdown file created: yes
- Filename follows required date/time/day/timezone/slug pattern: yes
- No forbidden engine features used as inputs: yes
- Does not fabricate labels: yes
- Not a routine CNN/ResNet/Transformer variant: yes
- Minimal current-data experiment exists: yes
- Falsification criterion is concrete: yes
- Codex can implement without asking for missing architecture details: yes
- Prompt maintenance notes included for Codex: yes
- Repetition check against imported research packets completed: yes
- Not a tactical sheaf/Hodge variant or one-ply move-delta pooling/spectrum/landscape variant: yes
- Not a current-board piece-target/material-target Sinkhorn/OT transport variant: yes
- Not a deterministic nuisance-orthogonal projection bottleneck variant: yes
- Not an exact near-duplicate of imported ordinal, sparse-witness, ray-language, Mobius-constellation, or pseudo-likelihood packets: yes
- Not an exact near-duplicate of imported orbit-symmetry, tempo-intervention, credal-evidence, rule-partition-invariance, kinematic-commutator, or masked-codec packets: yes
- Not an exact near-duplicate of imported cubical Euler/Betti topology, Hall-defect overload, or king-cage/king-escape path-DP packets: yes
- Not an exact near-duplicate of imported FCA/Galois-closure, denoising-score-field, or non-backtracking-edge-walk packets: yes

# Codex Handoff Packet: Soft King-Cage Path Bottleneck Network

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-21_0812_tuesday_los_angeles_king_cage_dp.md`
- Generated at: 2026-04-21 08:12 America/Los_Angeles
- Weekday: Tuesday
- Timezone: `los_angeles`
- Idea slug: `king_cage_dp`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: Soft King-Cage Path Bottleneck Network
- One-sentence thesis: A chess position is puzzle-like partly when one king is separated from the broader board by a low-dimensional, rule-derived barrier field, so a classifier should benefit from a differentiable soft shortest-path bottleneck that measures king escape energy without using engine analysis or game-tree search.
- Idea fingerprint: current board planes -> deterministic pseudo-legal attack and occupancy maps -> monotone nonnegative king-barrier field -> temperature-smoothed Bellman-Ford dynamic program on the fixed 8-neighbor king-step grid from each king to outer Chebyshev shells -> cage energy/path-entropy/distance-field bottleneck fused with a small CNN -> binary logits.
- Why this is not a common CNN/ResNet/Transformer variant: the central layer explicitly sums over an exponential family of board paths by dynamic programming and has a degree-preserving topology-destroying ablation; a plain convolution, ResNet, or square Transformer has no forced path/barrier computation.
- Current-data minimal experiment: train the model on `simple_18` for the existing `crtk_sample_3class` split, using the shared coarse-binary trainer for 3 epochs, then compare test AUROC/balanced accuracy and the `3x2` fine-label diagnostic against the current simple CNN and residual CNN.
- Smallest central falsification ablation: keep the same board trunk, barrier fields, king positions, shell sizes, barrier histograms, material/side-to-move features, and parameter count, but run the soft Bellman-Ford recurrence on a fixed degree-preserving random graph over 64 squares instead of the true 8-neighbor chessboard grid.
- Expected information gain if it fails: if the randomized-topology DP matches the real-topology DP, king-cage path topology is not carrying useful puzzle signal beyond attack/occupancy counts, and future cycles should avoid soft shortest-path/min-cut king-cage variants.

## 3. Problem Restatement And Data Contract

The task is binary chess puzzle-likeness classification from a single board position. The model input is a tensor `(batch, C, 8, 8)` and the output is logits `(batch, 2)`, where output `0` means non-puzzle and output `1` means puzzle-like. The available fine labels are:

- fine label `0`: known non-puzzle
- fine label `1`: verified near-puzzle
- fine label `2`: verified puzzle

The benchmark is coarse binary with `y_binary = 0` for fine label `0` and `y_binary = 1` for fine labels `1` and `2`. Reports must still include the rectangular diagnostic matrix:

```text
true fine label 0/1/2 -> predicted binary output 0/1
```

The required benchmark split is:

```text
data/splits/crtk_sample_3class/split_train.parquet
data/splits/crtk_sample_3class/split_val.parquet
data/splits/crtk_sample_3class/split_test.parquet
```

Do not point the current trainer directly at the roughly 45M-row full Parquet dataset until streaming support exists.

Available encodings are `simple_18`, `lc0_static_112`, and `lc0_bt4_112`. The first experiment should use `simple_18` because its channel semantics are explicit: 12 piece planes plus side-to-move, castling, and en-passant planes. The model can later support `lc0_static_112` and `lc0_bt4_112` only through fail-closed adapters that know exactly which channels correspond to current piece occupancy and side-to-move.

Leakage checklist:

- Safe neural inputs: current board occupancy, deterministic board coordinates, side-to-move, castling/en-passant planes already present in the encoding, and pseudo-legal attack geometry derived only from the current board.
- Safe diagnostic targets: fine labels used only for reporting the `3x2` matrix and near-puzzle diagnostics, never as input features.
- Unsafe neural inputs: Stockfish scores, PVs, mate scores, node counts, verification metadata, source labels, proposed labels, dataset provenance, or unresolved-candidate status.
- Unsafe unless separately justified and ablated: full legal-move generation, legal move counts, checkmate/stalemate oracles, forced-line search, and move-tree consequences.
- This idea uses pseudo-legal attack pressure maps and occupancy barriers, not legal move counts, mate/stalemate detection, or game-tree search.
- For `lc0_static_112` and `lc0_bt4_112`, deterministic geometry may use only known current-board channels. History channels may be passed to a learned neural trunk but must never be interpreted by the rule-geometry adapter unless channel semantics are explicitly documented.

## 4. Research Map

External sources used:

| Source | What is borrowed | What is not copied |
|---|---|---|
| Arthur Mensch and Mathieu Blondel, "Differentiable Dynamic Programming for Structured Prediction and Attention", ICML 2018, https://arxiv.org/abs/1802.03676 | The general principle that replacing a hard max/min in a DP recurrence by a smooth convex-regularized operator yields a differentiable layer. | No Viterbi, sequence labeling, or attention architecture is copied. |
| Marco Cuturi and Mathieu Blondel, "Soft-DTW: a Differentiable Loss Function for Time-Series", ICML 2017, https://proceedings.mlr.press/v70/cuturi17a.html | The soft-minimum view of a combinatorial alignment/path objective and the idea that gradients represent softened path responsibility. | No time-series alignment loss or DTW recurrence is used. |
| Brandon Amos and J. Zico Kolter, "OptNet: Differentiable Optimization as a Layer in Neural Networks", ICML 2017, https://proceedings.mlr.press/v70/amos17a.html | The design pattern of inserting an optimization-like constrained computation as a neural module. | No quadratic program, OptNet solver, or implicit QP differentiation is used. |
| Marin Vlastelica Pogančić et al., "Differentiation of Blackbox Combinatorial Solvers", ICLR 2020, https://openreview.net/forum?id=BkevoJSYPB | The broader motivation for combinatorial solver layers, including shortest-path solvers, inside learning systems. | No black-box solver, Dijkstra call, MIP solver, or perturbed black-box gradient estimator is used. |
| Ryo Yonetani et al., "Path Planning using Neural A* Search", ICML 2021, https://arxiv.org/abs/2009.07476 | The path-planning analogy: learn a cost/guidance map, then run a differentiable search-like computation. | No A* search, path-supervision labels, expert demonstrations, or planning dataset are used. |
| Zhaocheng Zhu et al., "Neural Bellman-Ford Networks", NeurIPS 2021, https://openreview.net/forum?id=DEsIX_D_vR | The idea that path aggregations can be computed by Bellman-Ford-like recurrences. | No link prediction, knowledge graph relations, or NBFNet message-passing architecture is copied. |

Candidate search trace:

| Serious candidate considered | Why it was not selected |
|---|---|
| Cubical persistent homology of attack-pressure fields | Interesting, but too close to the imported static attack-defense/topological families and harder to implement/debug on an 8x8 board than a soft path DP. |
| Label-safe selective prediction head for near-puzzles | Useful for deployment calibration, but mostly a head/loss idea; it does not add a strong board-level inductive bias for puzzle-likeness. |
| Inferred-environment adversarial information bottleneck | Could attack source artifacts, but it risks relying on unknown dataset provenance and may become another nuisance-suppression variant without a crisp chess operator. |
| Neural-symbolic DNF over deterministic tactical predicates | Potentially interpretable, but overlaps with sparse witness and high-order constellation packets unless it introduces a truly new predicate algebra. |
| Rule-only static exchange evaluator over capture sequences | Chess-relevant, but too close to legal move-tree consequences and leakage-prone unless heavily constrained; it also edges toward one-ply/multi-ply move-delta families. |
| Spectral compression of king-zone pressure maps | Easier than path DP, but likely collapses to static attack-map features and lacks a decisive topology-destroying falsifier. |
| Causal invariance over material/phase/color environments | Already represented by imported rule-partition invariance packets; new environments are not clearly available in the current split. |
| Differentiable board edit distance to stored puzzle motifs | Risks memorization, motif-library leakage, and near-duplicate behavior with generative compression/pseudo-likelihood packets. |
| Energy-based latent variable over "tactical tension" | Too unconstrained without engine-free latent supervision; easy to reduce to an ordinary MLP on CNN features. |
| Plain path-based GNN between all squares | Too close to a generic square GNN and less falsifiable than an explicit fixed-grid DP. |
| Low-rank polynomial ring features around both kings | Too close to Möbius/ANOVA piece-constellation packets. |
| Masked piece restoration around kings | Too close to the imported masked board code-length/surprise codec packet. |

Concept-to-operator mapping:

| High-level concept | Concrete operator/object in this idea | Tensor contract | Central falsifier | Why not a duplicate |
|---|---|---|---|---|
| Differentiable dynamic programming | Temperature-smoothed Bellman-Ford recurrence on the fixed 8-neighbor king-step grid | barrier `[B, 2, 8, 8]` -> distance fields `[B, 2, R, Q, 8, 8]` and scalars `[B, 2, R, Q]` | Replace grid edges by a degree-preserving random graph while preserving barrier and count statistics | Imported packets do not use board-path escape DP as the central observable. |
| King-cage geometry | Minimum soft barrier from each king to outer Chebyshev shells | king maps `[B, 2, 8, 8]`, shells from coordinates and king positions | Shuffle barriers within king-centered shells, preserving radial histograms but destroying corridors | Not a static attack-defense graph; attacks become scalar costs on a spatial escape field. |
| Information bottleneck | Low-dimensional cage scalars and optional distance fields appended to a small CNN trunk | scalars `[B, F]`, fields `[B, F_map, 8, 8]` | Replace cage features by matched histograms and a parameter-matched MLP | Not closed-form nuisance projection and not a masked codec. |
| Path entropy / multiplicity | Compare soft escape energies across temperatures as a proxy for number of viable low-cost corridors | `Q` temperatures, default `[0.25, 0.75]` | Use only the lowest-temperature energy and remove multi-temperature spread | Not an ordinal, evidential, or credal head; uncertainty is about softened paths, not labels. |

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN | `src/chess_nn_playground/models/trunk/cnn.py` | Already implemented and too ordinary; it does not isolate a new chess-specific operator. |
| Residual CNN | `src/chess_nn_playground/models/trunk/residual_cnn.py` | Already implemented; adding residual depth would be routine width/depth tuning. |
| LC0-style CNN/residual CNN | Existing LC0 BT4-style CNN and residual variants | Already represented; copying LC0-style channels or blocks is not a new research mechanism. |
| Ordinary ViT over 64 squares | Generic Transformer baseline | Too generic, parameter-hungry for the current sample, and explicitly disallowed as a core idea. |
| Plain GNN on square adjacency | Generic board graph model | Too close to a standard graph baseline unless it has a sharper chess operator and falsifier. |
| Hyperparameter tuning | Existing benchmark configs | Does not test a new hypothesis about puzzle-likeness. |
| Ensembling | Any existing model ensemble | Improves robustness at higher cost but gives little scientific information about board structure. |
| Bigger CNN or deeper ResNet | Small/medium/deep variants | Explicitly disallowed as the core idea and likely confounds compute with inductive bias. |
| Static attack-defense graph classifier | Imported sheaf/Hodge/attack-defense families | Already heavily researched; this packet avoids attack graph edges as the central object. |
| One-ply move-delta pooling | Imported counterfactual move-delta families | It would enumerate move consequences and duplicate existing move-landscape ideas. |
| Sinkhorn/optimal-transport piece-target model | Imported transport families | Already represented; changing costs or temperatures would not be novel. |
| Ordinal cumulative head | Imported ordinal evidence ladder | A head-only label-ordering change is already covered. |
| Sparse witness-piece bottleneck | Imported sparse witness packet | Selecting a few pieces would duplicate existing sparse rationale mechanics. |
| Masked-board surprise codec | Imported masked code-length packet | Compression/surprise from masking is already researched and not the chosen operator. |

## 6. Mathematical Thesis

Let `X` be the space of current-board tensor encodings that decode to legal or dataset-provided chess positions with side-to-move. Let `F in {0,1,2}` be the fine label and `Y = 1[F >= 1]` the coarse binary target. Training data are drawn from an empirical distribution `P(X,F)` induced by the benchmark split. The model may use only deterministic functions of the current board and the provided input planes; it may not use engine evaluations, verification metadata, source labels, or move-tree consequences.

Chess symmetry assumptions are deliberately weak. The board is not invariant under arbitrary rotations or reflections because pawns, castling, en-passant, side-to-move, and board edges matter. The model computes both white-king and black-king cage features and then forms side-to-move-relative combinations, but it does not impose full color-flip or board-orbit quotient invariance.

Core hypothesis:

For a nontrivial subset of puzzle-like positions, especially tactics involving mating nets, trapped kings, overloaded defenders near the king, and forced king exposure, the label `Y=1` is statistically associated with an asymmetric increase in a rule-only king escape barrier. This barrier is not just the number of attacked adjacent squares; it is the soft minimum cost of paths from the king to a broader safe region through the current board's attacked and occupied squares.

Formal object:

Let `S` be the 64 board squares and let `G=(S,E)` be the undirected 8-neighbor king-step grid. For color `c`, let `k_c in S` be the current king square. Define deterministic current-board maps:

- `A_{\bar c}(i)`: clipped or `log1p` pseudo-legal attack count by the opponent of color `c` on square `i`, computed from the current board with blockers for sliders.
- `O_c(i)`: occupancy by color `c`.
- `O_{\bar c}(i)`: occupancy by the opponent.
- `D_c(i)`: Chebyshev distance from `k_c` to `i`.
- `B_edge(i)`: edge/corner coordinate features.

A small constrained barrier module produces nonnegative costs

```text
b_c(i) = softplus(theta_0
                  + softplus(theta_A) * log1p(A_{\bar c}(i))
                  + softplus(theta_own) * O_c(i)
                  + softplus(theta_opp) * O_{\bar c}(i)
                  + h_theta(local_coord_features_i))
```

where `h_theta` is a small learned local adapter. The monotone positive coefficients make attacked and occupied squares weakly harder, not easier, to traverse.

For shell radius `r`, define the target set

```text
T_r(k_c) = { i in S : d_infty(i, k_c) >= r }.
```

For temperature `tau > 0`, define the soft-min operator

```text
softmin_tau(z_1,...,z_m) = -tau * log(sum_j exp(-z_j / tau)).
```

Initialize

```text
V_0^{c,r,tau}(i) = 0 if i in T_r(k_c), else M
```

for a large finite `M`. Then iterate for `t = 0,...,T-1`:

```text
V_{t+1}^{c,r,tau}(i) =
  0                                                if i in T_r(k_c)
  softmin_tau({ V_t^{c,r,tau}(j) + b_c(j) : j in N_G(i) }) otherwise.
```

The cage energy is

```text
E_{c,r,tau}(X) = V_T^{c,r,tau}(k_c).
```

Use radii such as `r in {2,3,4,5}` and temperatures such as `tau in {0.25,0.75}`. The classifier receives `E_{c,r,tau}`, differences between the two kings, side-to-move-relative versions, and optionally the final distance fields `V_T`.

Variational principle:

For fixed `b_c`, `r`, and `T`, the recurrence computes

```text
E_{c,r,tau}(X)
= -tau * log sum_{p in P_T(k_c,T_r)} exp(-Cost_b(p) / tau)
```

where `P_T(k_c,T_r)` is the set of grid paths of length at most `T` from `k_c` to the target shell, and `Cost_b(p)` is the sum of entered-square barriers along the path, up to the usual finite-horizon convention. As `tau -> 0`, `E_{c,r,tau}` converges to the minimum path barrier. For `tau > 0`, gradients of `E` with respect to `b_c(i)` are proportional to the expected number of visits to square `i` under the Gibbs distribution over low-cost escape paths.

Proof sketch:

The statement follows by induction on `t`. At `t=0`, the partition function is one for target squares and approximately zero elsewhere because non-target states have cost `M`. Suppose `V_t` is the negative-temperature-scaled log partition over paths of length at most `t` from each square to the target. The recurrence expands one neighbor step and adds the entered-square barrier, while `softmin_tau` is exactly the negative log-sum-exp over those one-step path extensions. Differentiability follows from smoothness of log-sum-exp for finite `tau`. The zero clamping on target squares implements absorbing targets. The `tau -> 0` limit is the standard log-sum-exp convergence to the minimum.

What is actually proven:

- The DP layer computes a differentiable soft minimum over bounded-length grid escape paths.
- Its low-temperature limit recovers a minimum cumulative barrier over those paths.
- Its gradients identify soft path responsibility under the Gibbs path distribution.

What remains hypothesized:

- Puzzle-likeness is sufficiently correlated with king-cage path energy to improve classification.
- The current-board pseudo-legal attack and occupancy maps are adequate proxies for real king escape constraints.
- The bottleneck improves sample efficiency relative to a small CNN rather than merely adding a redundant feature.

Counterexamples where the idea should fail:

- Quiet material-winning puzzles with no king exposure or king-cage structure.
- Endgame studies where the key idea is opposition, zugzwang, or promotion geometry not captured by attack-barrier corridors.
- Tactical puzzles centered on a queen fork, skewer, or trapped non-king piece far from both kings.
- Positions where a legal defensive resource exists because an occupied/attacked soft barrier is not a true legal barrier.
- Positions with noisy near-puzzle labels where the visual cage is strong but the verified line is absent.

Self-critique:

The strongest objection is that the model may only rediscover "attacked squares near the king" or "king on edge" shortcuts, not true path topology. That objection is serious because pseudo-legal attack maps are strong and can correlate with tactics. The minimal experiment is still worth running because the degree-preserving random-graph DP, radial shell shuffle, and histogram-only controls directly test whether spatial corridor topology adds information beyond attack/occupancy counts, edge/corner location, material, side-to-move, and immediate king-ring pressure.

## 7. Architecture Specification

Module names:

- `SoftKingCagePathNet`
- `EncodingSemanticsAdapter`
- `RuleGeometryBuilder`
- `MonotoneBarrierField`
- `SoftKingEscapeDP`
- `CageFeatureFusionHead`

Forward-pass steps:

1. Input `x`: `[B, C, 8, 8]`.
2. `EncodingSemanticsAdapter` extracts current piece planes and side-to-move:
   - piece planes: `[B, 12, 8, 8]`
   - side-to-move scalar or plane: `[B, 1]` and/or `[B, 1, 8, 8]`
   - adapter must raise a clear error if channel semantics are unknown.
3. `RuleGeometryBuilder` computes deterministic maps:
   - king maps: `[B, 2, 8, 8]`
   - own/opponent occupancy maps per king color: `[B, 2, 2, 8, 8]`
   - clipped/log attack pressure by each side: `[B, 2, 8, 8]`
   - coordinate/ring maps relative to each king: `[B, 2, K_geo, 8, 8]`
4. `MonotoneBarrierField` builds per-king nonnegative barrier maps:
   - input maps per color: `[B, 2, K_barrier, 8, 8]`
   - output barriers: `[B, 2, 8, 8]`
5. `SoftKingEscapeDP` runs the smoothed Bellman-Ford recurrence for `R` radii and `Q` temperatures:
   - distance fields: `[B, 2, R, Q, 8, 8]`
   - cage scalars sampled at each king: `[B, 2 * R * Q]`
   - side-relative scalars: concatenate own-to-move, opponent-to-move, opponent-minus-own, max, min, and temperature spread, about `[B, 6 * R * Q]`.
6. Board trunk:
   - input `x`: `[B, C, 8, 8]`
   - stem conv to `[B, W, 8, 8]`, default `W=48`
   - two small residual blocks, still `[B, W, 8, 8]`
   - if `use_distance_fields=true`, concatenate flattened DP maps projected by `1x1` conv: `[B, W + W_dp, 8, 8]`
   - global average/max pool to `[B, 2*(W+W_dp)]`
7. Fusion head:
   - concatenate pooled trunk features and cage scalars: `[B, F_total]`
   - MLP `F_total -> 64 -> 2`
   - output logits `[B, 2]`.

Parameter-count estimate:

- For `simple_18`, default `W=48`, two residual blocks, small DP projection, and fusion MLP: roughly `140k-190k` trainable parameters.
- For `lc0_static_112` or `lc0_bt4_112`, the first convolution adds about `(112-18)*48*3*3 = 40,608` parameters, so expect roughly `180k-235k`.
- The deterministic geometry and DP recurrences have no trainable parameters except the small barrier field adapter.

FLOP/complexity estimate:

- Pseudo-legal attack geometry: `O(B * 64 * ray_directions * max_ray)` plus leaper/pawn masks; on 8x8 this is small.
- Soft DP: `O(B * 2 * R * Q * T * 64 * 8)` soft neighbor operations. With `R=4`, `Q=2`, `T=12`, this is about `98k` neighbor updates per sample.
- CNN trunk is tiny because all feature maps are `8x8`.

Candidate set and memory:

- No explicit path candidate set is materialized. The exponential path family is compressed by DP.
- Distance buffer memory is `O(B * 2 * R * Q * 64)` floats if only final fields are kept, and `O(B * 2 * R * Q * T * 64)` under naive autograd.
- For `B=512`, `R=4`, `Q=2`, final distance fields require about `512*2*4*2*64*4 = 2 MB`; storing all `T=12` states is about `24 MB` before framework overhead.
- Chunking plan: if memory becomes a problem, loop over `(radius, temperature)` pairs and keep only the final field/scalar; or set `use_distance_fields=false` and keep only cage scalars.

Required config fields:

```yaml
model:
  name: soft_king_cage_path_net
  input_channels: 18
  num_classes: 2
  encoding_adapter: simple_18
  trunk_width: 48
  trunk_blocks: 2
  barrier_hidden_channels: 16
  dp_radii: [2, 3, 4, 5]
  dp_temperatures: [0.25, 0.75]
  dp_steps: 12
  dp_big_m: 50.0
  use_distance_fields: true
  monotone_barrier: true
  ablation_mode: none
```

Encoding-adapter assumptions:

- `simple_18`: supported immediately; parse 12 piece planes and side-to-move/castling/en-passant planes according to the existing exporter contract.
- `lc0_static_112`: learned trunk may accept all 112 channels, but rule geometry may only run if the adapter can identify current piece and side-to-move channels exactly. Otherwise raise `ValueError`.
- `lc0_bt4_112`: same as `lc0_static_112`; unavailable history planes are irrelevant to rule geometry and should not be interpreted. History channels may flow through the learned trunk only.
- All adapters must fail closed when channel semantics are unknown.

Pseudocode:

```text
forward(x):
    parsed = adapter.parse(x)  # fail closed if semantics unknown
    geom = rule_geometry(parsed.pieces, parsed.side_to_move)
    barrier = monotone_barrier(geom)  # [B,2,8,8], nonnegative

    dp_fields, dp_scalars = soft_escape_dp(
        barrier=barrier,
        king_maps=geom.king_maps,
        radii=cfg.dp_radii,
        temperatures=cfg.dp_temperatures,
        steps=cfg.dp_steps,
        ablation_mode=cfg.ablation_mode,
    )

    trunk = board_trunk(x)
    if cfg.use_distance_fields:
        trunk = concat_channels(trunk, dp_project(flatten_color_radius_temp(dp_fields)))
    pooled = global_pool(trunk)

    cage = make_side_relative(dp_scalars, parsed.side_to_move)
    logits = fusion_head(concat(pooled, cage))
    return logits
```

The returned logits are exactly `(batch, num_classes)` and should work with the shared trainer, reports, confusion matrices, predictions, and leaderboard code.

## 8. Loss, Training, And Regularization

Primary loss:

- Balanced cross-entropy on the coarse binary target, using the existing `class_weighting: balanced` behavior.

Optional auxiliary loss:

- Optional small cage-only classification head from cage scalars, weighted by `aux_cage_loss_weight=0.05`.
- This auxiliary head is diagnostic and should be disabled in the cleanest first comparison unless Codex wants branch-utilization curves.
- No auxiliary target may use fine label `1` or `2` differently; it must use only the same coarse binary target.

Class weighting:

- Use balanced weights from the training split, matching existing benchmark behavior.

Batch size expectations:

- Default `batch_size: 512` on CPU/GPU for `simple_18`.
- If DP autograd memory is high, first set `use_distance_fields=false` before reducing batch size.

Learning rate and optimizer defaults:

- Optimizer: AdamW.
- Learning rate: `0.001`.
- Weight decay: `0.0001`.
- Epochs: `3` for the minimal experiment.
- Early stopping patience: `2`, unchanged from current configs.
- Mixed precision: `false` for deterministic first tests; enable later only after numeric stability is confirmed.

Regularizers:

- Dropout `0.10` in the fusion MLP.
- Barrier field coefficients for attack and occupancy should be monotone positive via `softplus`.
- Optional L2 penalty on local barrier adapter weights, not on monotone base coefficients.
- No data augmentation is required for the minimal experiment.

Determinism requirements:

- Use seed `42`.
- Keep `deterministic: true`.
- The central random-graph ablation must use a fixed saved seed and log the generated graph degree sequence.
- No stochastic graph resampling per batch for the main result.

What must stay unchanged for fair comparison:

- Same train/val/test Parquet split.
- Same coarse-binary target construction.
- Same report format including `3x2` diagnostic matrix.
- Same epoch budget, class weighting, batch size if hardware permits, and early stopping policy.
- Do not add extra data, engine annotations, puzzle verification metadata, or future full-dataset streaming.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| Degree-preserving random-grid DP | Replaces the true 8-neighbor grid with a fixed random graph over 64 squares with matched node degrees; keeps barriers, king sources, shell target counts, side-to-move, histograms, trunk, and parameter count | The real spatial path topology matters, not merely repeated soft aggregation over 64 nodes | If this matches the main model, abandon the central path-topology claim. |
| Shell-shuffled barrier DP | Randomly permutes barrier values within Chebyshev shells around each king, preserving radial distribution and local counts but destroying corridors | Corridor placement matters beyond king distance and attack/occupancy histograms | If this matches the main model, the model is using radial pressure only. |
| Histogram-only cage control | Replaces DP fields/scalars with per-king histograms/quantiles of attack, occupancy, barrier, edge distance, and ring counts; parameter-match the fusion MLP | Soft path aggregation adds value beyond count statistics | If this matches the main model, future work should use cheaper count features or discard the idea. |
| Immediate-ring-only control | Uses only ring-1 and ring-2 attacked/occupied/empty counts around each king | Multi-step escape geometry matters beyond adjacent-square king safety | If this matches the main model, the DP horizon is unnecessary. |
| No-DP param-matched trunk | Removes DP branch and adds a parameter-matched MLP/conv block to the board trunk | Gains are not just from extra parameters | If this matches the main model, the bottleneck is not earning its complexity. |
| Fixed-barrier DP | Uses a hand-set monotone barrier formula with no learned local adapter | Learned barrier calibration matters | If fixed performs the same, keep the simpler fixed version for interpretability. |
| Low-temperature-only DP | Uses only `tau=0.25`, removing temperature spread/path multiplicity | Path multiplicity/entropy adds value beyond the best corridor | If equal, remove multi-temperature features. |
| Distance-fields-off | Keeps only cage scalars, not final DP fields | Fine spatial DP maps add useful local detail | If equal, keep scalar-only model for lower memory and better falsifiability. |
| Attack-map CNN control | Feeds raw attack-pressure maps to a param-matched CNN without DP | The DP, not just attack maps, drives improvement | If equal, the idea reduces to static attack-map feature engineering. |
| Side-relative swap diagnostic | Swap own/opponent cage scalar ordering at evaluation or train a random-side twin | The side-to-move-relative asymmetry is meaningful | If little change, the branch may be side-insensitive or shortcutting. |

The first row is the smallest central falsification ablation. It preserves obvious nuisance statistics while destroying the proposed graph semantics.

This idea does not generate a legal move set, one-ply move-delta set, or candidate move bag. The analogous count-preserving controls are the histogram-only, immediate-ring-only, shell-shuffled, and attack-map CNN controls, which preserve material, side-to-move, king location, attack-count distributions, source-square/ring marginals, and barrier histograms while destroying path semantics.

## 10. Benchmark And Falsification Criteria

Baselines to compare against:

- Existing `simple_18` simple CNN.
- Existing `simple_18` residual CNN.
- If already available in the leaderboard, include small/medium/deep variants as context, but the primary comparison is against parameter-near simple/residual CNNs.
- Optional later comparison: `lc0_bt4_112` CNN/residual after fail-closed adapter support exists.

Metrics to inspect:

- Validation and test accuracy.
- Balanced accuracy.
- AUROC.
- AUPRC.
- F1 at the selected threshold.
- Cross-entropy loss.
- Expected calibration error if already supported.
- `3x2` fine-label diagnostic confusion matrix.
- Fine label `1` recall and precision at matched fine-label-`0` false-positive rate.

Near-puzzle diagnostic:

- On validation, choose a threshold for each model that matches the baseline fine-label-`0` false-positive rate, preferably using a fixed target such as `5%` if feasible.
- On test, report fine-label-`1` recall, fine-label-`1` precision among predicted positives, and fine-label-`2` recall at that matched false-positive rate.
- This diagnostic matters because near-puzzles are the likely ambiguity boundary where a cage bottleneck may help or fail.

Required confusion outputs:

- Main model: fine label `0/1/2 -> predicted 0/1`.
- Every central ablation: same `3x2` matrix.
- Include thresholded predictions and probability scores in the shared predictions artifact.

Required artifacts:

- `results/<run>/metrics.json`
- `results/<run>/confusion_3x2.csv`
- `results/<run>/predictions.parquet`
- `results/<run>/model_config.yaml`
- `results/<run>/ablation_summary.md`
- `results/<run>/king_cage_diagnostics.parquet` with cage scalars, threshold, fine label, prediction, and correctness.
- Optional but useful: a small PNG grid of top false positives/false negatives with king-cage fields, if existing plotting utilities permit.

Success threshold:

- Test AUROC improves by at least `0.010` over the best parameter-near `simple_18` CNN/residual baseline, or balanced accuracy improves by at least `0.008`, and
- fine-label-`1` recall at matched fine-label-`0` false-positive rate improves by at least `0.030`, and
- the degree-preserving random-grid DP loses at least half of the main model's gain over the no-DP baseline or loses at least `0.005` AUROC.

Failure threshold:

- Main model is within `0.003` AUROC and `0.005` balanced accuracy of the no-DP param-matched trunk, or
- random-grid DP and shell-shuffled DP match the main model within noise, or
- fine-label-`1` recall does not improve at matched fine-label-`0` FPR.

What result would make me abandon the idea:

- If both topology-destroying controls match the main model and the histogram-only control captures the same near-puzzle behavior, then the path/barrier thesis is false for the current task and should not be repeated as a king-cage DP, min-cut, soft shortest-path, or path-entropy variant.

What result would justify scaling:

- The main model beats the strongest `simple_18` baseline on AUROC/balanced accuracy, improves class-`1` recall at matched false-positive rate, and the real-grid DP clearly beats the randomized-topology and histogram controls.
- Only after that should Codex try `lc0_static_112` or `lc0_bt4_112` with fail-closed rule adapters and longer training.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_0812_king_cage_dp/idea.yaml` | Create | Machine-readable idea metadata copied from the `idea_yaml` block. |
| `ideas/20260421_0812_king_cage_dp/math_thesis.md` | Create | Section 6 mathematical thesis, theorem/proof sketch, hypotheses, counterexamples, and self-critique. |
| `ideas/20260421_0812_king_cage_dp/architecture.md` | Create | Module contracts, tensor shapes, DP recurrence, parameter estimate, and pseudocode. |
| `ideas/20260421_0812_king_cage_dp/implementation_notes.md` | Create | Encoding adapter fail-closed rules, attack-map construction notes, numerical stability notes for `softmin_tau`, and random-graph ablation details. |
| `ideas/20260421_0812_king_cage_dp/trainer_notes.md` | Create | Loss, class weighting, deterministic training, metrics, and unchanged benchmark assumptions. |
| `ideas/20260421_0812_king_cage_dp/ablations.md` | Create | Ablation table and required count-preserving controls. |
| `ideas/20260421_0812_king_cage_dp/train.py` | Create | Thin idea-local entrypoint that calls the shared trainer with this config; no custom trainer unless necessary. |
| `ideas/20260421_0812_king_cage_dp/config.yaml` | Create | Runnable minimal config copied from `config_yaml`, plus model-specific fields. |
| `ideas/20260421_0812_king_cage_dp/report_template.md` | Create | Required benchmark, confusion, near-puzzle diagnostic, ablation comparison, and abandonment/scaling decision template. |
| `ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` | Update | Add prompt maintenance notes from Section 13 after consuming this packet; preserve all hard leakage, label, falsification, and anti-duplicate rules. |
| `src/chess_nn_playground/models/trunk/soft_king_cage_path.py` | Create | PyTorch implementation of `SoftKingCagePathNet`, `EncodingSemanticsAdapter`, `RuleGeometryBuilder`, `MonotoneBarrierField`, and `SoftKingEscapeDP`. |
| `src/chess_nn_playground/models/registry.py` | Modify | Register `soft_king_cage_path_net` builder. |
| `configs/soft_king_cage_path_simple18.yaml` | Create | Minimal 3-epoch `simple_18` experiment config. |
| `configs/soft_king_cage_path_simple18_random_grid_ablation.yaml` | Create | Same as main config with `ablation_mode: random_grid_degree_preserving`. |
| `configs/soft_king_cage_path_simple18_histogram_ablation.yaml` | Create | Same as main config with `ablation_mode: histogram_only`. |
| `tests/test_soft_king_cage_path.py` | Create | Shape tests, deterministic output tests, finite logits, gradient flow through DP, and registry construction. |
| `tests/test_rule_geometry_adapters.py` | Create | `simple_18` adapter parsing tests and fail-closed tests for unknown LC0 semantics. |
| `tests/test_soft_escape_dp_ablation.py` | Create | Verify real-grid and random-grid adjacency degree sequence, DP output shape, and no path candidate materialization. |

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0812_tuesday_los_angeles_king_cage_dp.md
  generated_at: 2026-04-21 08:12 America/Los_Angeles
  weekday: Tuesday
  timezone: los_angeles
  idea_slug: king_cage_dp
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260421_0812_king_cage_dp
  name: Soft King-Cage Path Bottleneck Network
  slug: king_cage_dp
  status: draft
  created_at: 2026-04-21 08:12 America/Los_Angeles
  author: ChatGPT Pro
  short_thesis: Puzzle-like positions often contain asymmetric king escape barriers; a differentiable soft shortest-path bottleneck can expose that structure without engine search.
  novelty_claim: Uses a fixed 8-neighbor king-step board topology and smoothed Bellman-Ford escape energy, not attack sheaves, move-delta sets, transport, pseudo-likelihood, orbit pooling, or masked code-length.
  expected_advantage: Better near-puzzle recall at matched non-puzzle false-positive rate when puzzle-likeness is tied to king cages or mating-net geometry.
  central_falsification_ablation: Degree-preserving random-grid soft DP preserving barriers, king sources, shell target counts, histograms, material, side-to-move, trunk, and parameter count.
  target_task: coarse_binary
  input_representation: simple_18 first; lc0_static_112/lc0_bt4_112 only with fail-closed current-board adapters
  output_heads: binary logits; optional cage-only auxiliary logits for diagnostics
  compute_notes: DP cost O(B*2*R*Q*T*64*8), no path candidate materialization; default R=4, Q=2, T=12.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/soft_king_cage_path_simple18.yaml
  model_path: src/chess_nn_playground/models/trunk/soft_king_cage_path.py
  latest_result_path: null
  notes: Do not use engine scores, legal move counts, mate/stalemate oracles, source labels, proposed labels, or dataset provenance.
```

```yaml
config_yaml:
  run:
    name: soft_king_cage_path_simple18
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
    name: soft_king_cage_path_net
    input_channels: 18
    num_classes: 2
    encoding_adapter: simple_18
    trunk_width: 48
    trunk_blocks: 2
    barrier_hidden_channels: 16
    dp_radii: [2, 3, 4, 5]
    dp_temperatures: [0.25, 0.75]
    dp_steps: 12
    dp_big_m: 50.0
    use_distance_fields: true
    monotone_barrier: true
    ablation_mode: none
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
  model_name: soft_king_cage_path_net
  file_path: src/chess_nn_playground/models/trunk/soft_king_cage_path.py
  builder_function: build_soft_king_cage_path_net
  input_shape: [batch, C, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - EncodingSemanticsAdapter
    - RuleGeometryBuilder
    - MonotoneBarrierField
    - SoftKingEscapeDP
    - CageFeatureFusionHead
  required_config_fields:
    - input_channels
    - num_classes
    - encoding_adapter
    - dp_radii
    - dp_temperatures
    - dp_steps
    - dp_big_m
    - trunk_width
    - trunk_blocks
    - ablation_mode
  expected_parameter_count: 140k-190k for simple_18; 180k-235k for 112-channel LC0 encodings
  expected_memory_notes: No explicit path set; final DP fields about 2 MB for batch 512, R=4, Q=2; naive autograd across 12 steps about 24 MB before overhead.
```

```yaml
research_continuity:
  idea_fingerprint: current-board occupancy and pseudo-legal attack maps -> monotone king barrier field -> smoothed Bellman-Ford escape energy over fixed 8-neighbor board grid -> cage scalars/distance fields fused with small CNN
  already_researched_family_overlap: Adjacent only in using rule-derived attack pressure; not a sheaf/Hodge/attack graph, move-delta, Sinkhorn/OT, nuisance projection, ordinal, sparse witness, ray automaton, ANOVA constellation, pseudo-likelihood, orbit, tempo, credal, kinematic commutator, or masked-codec packet.
  closest_duplicate_risk: Static attack-defense graph models; the distinguishing object is a king-centered board-path escape DP with topology-destroying graph ablation.
  do_not_repeat_if_this_fails:
    - soft Bellman-Ford king escape bottlenecks
    - differentiable min-cut or shortest-path king-cage variants
    - king-cage path entropy over attack/occupancy barrier fields
    - radial shell shuffled barrier variants of the same idea
  suggested_next_search_directions:
    - label-safe selective prediction that is not ordinal or credal if classification accuracy saturates
    - source-artifact suppression only if genuine non-forbidden environments become available
    - non-king strategic motif operators that avoid move-delta, transport, masked-codec, and constellation duplicates
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add "king-cage soft shortest-path/min-cut DP over attack/occupancy barriers" to the imported memory after this packet is implemented or fails. | Prevents the next research pass from renaming this mechanism as a min-cut, maze, escape-energy, or path-entropy model. | `Imported Research Memory` |
| Require any future attack-derived topology idea to include both degree-preserving graph randomization and histogram/radial-shell controls. | Attack maps are strong shortcuts; this makes topology claims falsifiable. | `Depth requirements` or `Ablation Plan` guidance |
| Add a line that LC0 rule adapters must fail closed unless current-board channel semantics are documented. | Avoids accidental interpretation of history or unknown channels as deterministic geometry. | `Problem Restatement And Data Contract` |
| Add near-puzzle recall at matched fine-label-0 false-positive rate as a preferred diagnostic for ambiguous class `1`. | It gives repeated cycles a stable way to compare behavior on the near-puzzle boundary. | `Benchmark And Falsification Criteria` |
| If this idea fails by matching histogram controls, add a warning against repeating "global topology over scalar attack pressure" without a new observable. | A failure would indicate the apparent structure is just count/ring statistics. | `Research Continuity` |

## 14. Final Sanity Check

- Downloadable Markdown file created: yes
- Filename follows required date/time/day/timezone/slug pattern: yes, `chess_nn_research_2026-04-21_0812_tuesday_los_angeles_king_cage_dp.md`
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
- Not an exact near-duplicate of imported ordinal, sparse-witness, ray-language, Möbius-constellation, or pseudo-likelihood packets: yes
- Not an exact near-duplicate of imported orbit-symmetry, tempo-intervention, credal-evidence, rule-partition-invariance, kinematic-commutator, or masked-codec packets: yes

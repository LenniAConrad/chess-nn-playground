# Codex Handoff Packet: King Escape Percolation Network

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-21_0811_tuesday_pacific_king_escape_percolation.md`
- Generated at: 2026-04-21 08:11 America/Los_Angeles
- Weekday: Tuesday
- Timezone: Pacific
- Idea slug: `king_escape_percolation`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: **King Escape Percolation Network**
- One-sentence thesis: A puzzle-like chess position often contains a frozen-board tactical cage around one king, so expose a differentiable soft shortest-path/free-energy operator that measures whether a king has low-cost escape corridors through current-board occupancy and pseudo-legal attack hazards, then test whether corridor connectivity improves binary puzzle classification beyond ordinary CNN features.
- Idea fingerprint: `current board -> pseudo-legal attack hazard fields -> side-aware king-seeded 8-neighbor softmin escape dynamic program -> escape free-energy maps/vectors -> small fusion classifier -> binary puzzle-likeness logits`.
- Why this is not a common CNN/ResNet/Transformer variant: The central computation is not learned convolutional receptive-field growth or attention over squares; it is a fixed, rule-conditioned dynamic program whose recurrence computes a smooth minimum over all frozen-board king escape paths up to a bounded length.
- Current-data minimal experiment: Train `KingEscapePercolationNet` on `simple_18` using `data/splits/crtk_sample_3class/split_train.parquet`, validate on `split_val.parquet`, test on `split_test.parquet`, keep the shared coarse-binary trainer/reporting path, and export the usual `3x2` fine-label diagnostic matrix.
- Smallest central falsification ablation: Replace each king's attack/occupancy cost field by a seeded, Chebyshev-ring-preserving and occupancy-bin-preserving shuffle before the dynamic program; this preserves material, side-to-move, king location, cost histogram, ring marginals, and obvious attack-density shortcuts while destroying connected escape corridors.
- Expected information gain if it fails: If the shuffled-cost ablation matches the full model, then corridor topology is not adding information beyond static attack density/ring counts, and future work should stop proposing king-cage connectivity variants unless new evidence or a different label regime appears.

## 3. Problem Restatement And Data Contract

The project is `chess-nn-playground`. The task is chess puzzle-likeness classification from a single board position. The model returns binary logits for:

- output `0`: non-puzzle
- output `1`: puzzle-like

The available fine labels are diagnostic source labels:

- fine label `0`: known non-puzzle
- fine label `1`: verified near-puzzle
- fine label `2`: verified puzzle

The default training/evaluation target is coarse binary. Unless the existing trainer already defines a different mapping, use `fine_label == 0 -> binary 0` and `fine_label in {1,2} -> binary 1`. Reports must still include the rectangular diagnostic matrix:

```text
true fine label 0/1/2 -> predicted binary output 0/1
```

Allowed input tensors are current project encodings with shape `(batch, C, 8, 8)`. The model must return logits with shape `(batch, 2)`. The minimal first experiment should use:

```text
data/splits/crtk_sample_3class/split_train.parquet
data/splits/crtk_sample_3class/split_val.parquet
data/splits/crtk_sample_3class/split_test.parquet
encoding: simple_18
```

The full Parquet dataset of roughly 45M rows must not be used directly until streaming support exists.

Leakage checklist:

- Safe as neural inputs or rule-derived tensors: deterministic board coordinates, piece occupancy, side-to-move, castling/en-passant planes already present in the encoding, and pseudo-legal attack geometry derived only from the current board.
- Leakage-prone unless explicitly justified, engine-free, label-independent, and ablated: full legal-move generation, move counts, checkmate/stalemate oracles, forced-line search, move-tree consequences, and any feature that asks whether a move wins, mates, or is legal after king-safety filtering.
- Forbidden as neural inputs: Stockfish or other engine evaluations, principal variations, node counts, mate scores, verification metadata, source labels, proposed labels, dataset provenance, row origin, and any unresolved candidate-pool status.
- `lc0_static_112` and `lc0_bt4_112`: deterministic geometry may use only explicitly mapped current-board piece planes, side-to-move, castling, and en-passant semantics. History channels may be consumed only by a learned neural adapter, not by the rule geometry. If Codex cannot verify the current-board channel mapping, the adapter must fail closed with a clear error.

## 4. Research Map

External sources used for the selected mechanism:

| Source | What is borrowed | What is not copied |
|---|---|---|
| Peter G. Doyle and J. Laurie Snell, *Random Walks and Electric Networks*, arXiv/math/0001057 and Dartmouth PDF, https://arxiv.org/abs/math/0001057 | The interpretation of escape/hitting behavior and conductance as a global connectivity property of a weighted graph. | No exact electrical-network solve is required; no theorem from chess or puzzle labels is imported. |
| Leo Grady, “Random Walks for Image Segmentation,” IEEE TPAMI 2006, DOI `10.1109/TPAMI.2006.233`, PubMed: https://pubmed.ncbi.nlm.nih.gov/17063682/ | The idea that seeded graph-walk or boundary-hitting probabilities expose object connectivity in an image-like grid. | No image segmentation labels, graph-cut segmentation, or pixel-label propagation objective is copied. |
| Arthur Mensch and Mathieu Blondel, “Differentiable Dynamic Programming for Structured Prediction and Attention,” ICML/PMLR 2018, https://proceedings.mlr.press/v80/mensch18a.html | The general principle of smoothing max/min recurrences so a dynamic program can live inside a neural network. | This idea does not copy their sequence models, Viterbi setup, attention mechanism, or training tasks. |
| Marco Cuturi and Mathieu Blondel, “Soft-DTW: a Differentiable Loss Function for Time-Series,” ICML/PMLR 2017, https://proceedings.mlr.press/v70/cuturi17a.html | A precedent for replacing hard dynamic-programming minima by soft minima and differentiating through the relaxed value. | No time-series alignment, DTW loss, or clustering objective is used. |
| Brandon Amos and J. Zico Kolter, “OptNet: Differentiable Optimization as a Layer in Neural Networks,” ICML/PMLR 2017, https://proceedings.mlr.press/v70/amos17a.html | The high-level lesson that constrained algorithmic layers can encode dependencies ordinary feed-forward layers may need many examples to learn. | No quadratic programming layer or OptNet solver is used. |

Candidate search trace for serious mechanisms considered but not selected:

| Candidate mechanism | Why it was considered | Why it lost to the final idea |
|---|---|---|
| Persistent homology of king-zone attack/occupancy filtrations | It would directly test whether holes, rings, and connected components around a king predict puzzle-likeness. | Differentiable cubical persistence on `8x8` boards is implementation-heavy, harder to debug, and has a less direct falsifier than ring-preserving corridor shuffles. |
| Soft Boolean tactical-motif satisfiability layer | A differentiable DNF/SAT layer over pins, forks, overloaded pieces, and king nets could encode human tactic concepts. | Handwritten motif libraries risk becoming brittle, incomplete, and close to checkmate/forcing-line oracles if expanded aggressively. |
| Label-safe selective classifier for fine-label-1 ambiguity | Near-puzzles are ambiguous, so abstention/calibration could be useful without changing input features. | It is more of an output decision policy than a board-structure inductive bias, and it is too close in spirit to existing ordinal/credal uncertainty packets. |
| Self-discovered environment adversarial bottleneck | It might suppress source artifacts without using provenance labels. | Learned environments can collapse, are difficult to falsify, and overlap the imported rule-partition invariance family unless the environment construction is genuinely new. |
| Tropical/morphological neural filters on attack fields | Min-plus and max-plus filters are naturally connected to cages and barriers. | A stack of learned morphological filters would look too much like another CNN unless tied to an explicit path objective; the final softmin DP keeps the morphology but gives it a chess-specific seed and target. |
| Energy-based masked piece deletion sensitivity | Puzzle positions might be fragile under deletion of a single blocker or defender. | Deletion sensitivity is close to sparse witness and masked-codec families, and it risks becoming a counterfactual piece-set search rather than a single current-board operator. |
| Low-rank latent motif dictionary with VQ bottleneck | A discrete motif code could be interpretable and source-artifact resistant. | Without a strong rule-derived operator it becomes generic representation learning, and masked/code-length packets already cover nearby generative compression ideas. |
| Full legal king-mobility diffusion | Direct legal king escapes are semantically tempting for mating nets. | Legal move generation and king-safety filtering drift toward checkmate/stalemate oracle leakage; the selected operator deliberately uses frozen-board pseudo-legal hazards only. |

Concept-to-operator mapping:

| High-level concept | Concrete operator/object in this idea | Tensor contract | Central falsifier | Why not a duplicate |
|---|---|---|---|---|
| Escape probability / conductance | King-seeded soft free energy to board boundary or outer king rings on the `8x8` king-adjacency grid | cost maps `(B, 2, T_tau, 8, 8)` plus scalar path free energies `(B, 2, T_tau, S)` | Ring/bin-preserving hazard shuffle before DP | Uses square-grid path connectivity, not attack-defense graph message passing, sheaf restrictions, Hodge energy, or piece-target transport. |
| Differentiable dynamic programming | Softmin recurrence over 8-neighbor king steps with fixed finite horizon | initial seed `(B,2,8,8)`, cost `(B,2,8,8)`, output distance maps `(B,2,len(tau),snapshots,8,8)` | Replace grid neighbors by degree/ring-preserving random neighbors | The learned classifier cannot choose arbitrary attention over 64 squares; it receives a constrained path log-partition. |
| Percolation/corridor topology | Connectivity of low-cost squares from king square to boundary under attack/occupancy barriers | per-side scalar and map descriptors | Count-only ring histogram and shuffled-cost controls | Tests connected corridors rather than total attack count, material, or king-zone density. |
| Information bottleneck | Small vector of escape free energies and reachable masses fused with a small CNN stem | vector `(B,V)` with `V` about 48-96 | Percolation-only versus CNN-only versus fused model | Not a closed-form nuisance projection and not a masked-board code-length model. |

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN on `simple_18` | `src/chess_nn_playground/models/trunk/cnn.py` | Already exists and tests generic local pattern learning without a new chess-specific operator. |
| Residual CNN depth/width variants | `src/chess_nn_playground/models/trunk/residual_cnn.py` | Already represented; changing depth or channels is ordinary capacity tuning, not a research idea. |
| LC0-style CNN on `lc0_static_112` or `lc0_bt4_112` | Existing LC0 BT4-style CNN variants | Too close to imported/baseline LC0 encoders and does not introduce a falsifiable mechanism. |
| LC0-style residual CNN | Existing LC0 residual CNN variants | A stronger residual tower could help but would not explain what structural signal distinguishes puzzle-like positions. |
| Ordinary ViT over 64 square tokens | Generic vanilla Transformer | Explicitly disallowed and likely data-hungry; attention weights alone are not a board operator. |
| Plain GNN on square adjacency | Generic square-graph neural network | Too close to a CNN with a different implementation and lacks a chess-specific falsifier. |
| Static attack-defense graph message passing | Imported tactical sheaf/Hodge/attack-defense families | Already heavily explored; more edge labels or pooling would be a near-duplicate. |
| One-ply move-delta set with attention/DeepSets | Imported rule-only counterfactual move-delta families | Already researched and risks leaning on move-count or legal-action shortcuts. |
| Entropic piece-target transport | Imported optimal-transport packets | Current-board transport bottlenecks are already represented; temperature or target-bucket changes are not novel. |
| Ordinal or cumulative near-puzzle head | Imported ordinal evidence ladder | Useful for labels but not a new input-side inductive bias, and the family is explicitly covered. |
| Sparse witness-piece bottleneck | Imported sparse witness-piece bottleneck | Deleting or selecting a few pieces is already represented and could become another witness mask. |
| Hyperparameter tuning | All current baselines | Learning rate, optimizer, dropout, and batch-size tuning are necessary engineering, not an original research mechanism. |
| Ensembling | Any trained model collection | It can improve metrics while hiding whether the new structural hypothesis is true. |

## 6. Mathematical Thesis

### Input space definition

Let `X_C = R^{C x 8 x 8}` be the tensor space for an encoding. For the minimal experiment, `C=18` and the first adapter decodes the current position into piece occupancy

```text
P(x) in {0,1}^{2 x 6 x 8 x 8},
```

where the color axis is `{white, black}` and piece axis is `{pawn, knight, bishop, rook, queen, king}` under the project's established channel convention. Let `m(x) in {white, black}` be side-to-move.

For each side `s`, let `K_s(x) in V` be the square of the side `s` king, where `V = {0,...,7} x {0,...,7}`. Let `A_s(x,v)` be a deterministic pseudo-legal attack count from side `s` onto square `v`, computed only from current-board occupancy and chess attack geometry. Sliding attacks stop at the first occupied blocker. No legal-move filtering, no checkmate/stalemate oracle, and no engine analysis is allowed.

### Label/target definition

The training target is `Y in {0,1}` with `Y=0` for fine label `0` and `Y=1` for fine labels `1` and `2`, unless the existing trainer's coarse-binary convention already defines this mapping. The fine label `L in {0,1,2}` is diagnostic only and must not be a neural input.

### Data distribution assumptions

The sample distribution is assumed to contain both tactical and non-tactical chess positions whose superficial material, phase, side-to-move, and source-process artifacts may differ across labels. The hypothesis does not assume every puzzle is a king attack. It assumes only that a nontrivial subset of puzzle-like positions has a static king-cage signature that is not fully captured by material counts or local attack density.

### Allowed symmetry or equivariance assumptions

Chess is not invariant under arbitrary rotations or vertical reflections because pawns, castling, and side-to-move break those symmetries. This model may share the same escape operator between the two colors after explicitly swapping “defender” and “attacker” roles and using side-aware pawn attack directions. It must not impose full board rotation/reflection invariance. Horizontal file-mirror augmentation is not central and should not be used in the minimal experiment unless the existing benchmark already applies it consistently to all baselines.

### Core hypothesis

Let `s` be a defender color and `a = 1-s` be the opposing color. Define a nonnegative frozen-board escape cost field

```text
c_s(x,v) = base + learned_nonnegative_geo_cost_s(x,v) + lambda_occ * occ(x,v),
```

with the defender king square exempted from the occupancy penalty. The cost features include attacker attack counts/types, defender occupancy, attacker occupancy, Chebyshev distance from `K_s`, distance to board edge, and side-to-move role bits. The central hypothesis is:

```text
H: P(Y=1 | x) is better approximated when the classifier has direct access to the low-cost path free energies from each king to escape basins than when it sees only local CNN features or attack-density histograms.
```

In human terms, some puzzle-like positions are produced by barriers and corridors: defenders may have many apparently empty squares, but those squares do not connect into safe escape routes once current attacks and blockers are considered.

### Formal object or operator

Let `G_K = (V,E_K)` be the undirected king-adjacency graph on the chessboard with self-loops, so `(u,v) in E_K` when `u=v` or the Chebyshev distance between `u` and `v` is `1`. Let `B_edge` be the set of board-edge squares. Optionally also define outer king rings `B_r = {v: d_infty(v,K_s) >= r}` for `r in {2,3,4}`.

For a temperature `tau > 0`, horizon `T`, and side `s`, define the soft dynamic-programming value:

```text
D_{s,tau,0}(v) = 0 if v = K_s(x), else +infty
D_{s,tau,t+1}(v) = c_s(x,v) - tau * log sum_{u in N_K(v)} exp(-D_{s,tau,t}(u) / tau)
```

where `N_K(v)` are king-neighbors plus self-loop predecessors inside the board. The edge escape free energy is

```text
F_{s,tau,T}^{edge}(x) = -tau * log sum_{v in B_edge} exp(-D_{s,tau,T}(v) / tau).
```

The model also exports reachable-mass summaries such as

```text
M_{s,tau,T,alpha}(x) = mean_v sigmoid((alpha - D_{s,tau,T}(v)) / rho),
```

plus selected `D` or reachability maps for CNN fusion.

### Variational principle and optimization objective

For fixed `x`, `s`, `tau`, and `T`, the dynamic program computes a log-partition over all length-`T` frozen-board king paths. If `P_T(K_s,v)` is the set of length-`T` paths from `K_s` to `v`, and path cost is `C_s(p)=sum_{i=1}^T c_s(x,p_i)`, then

```text
D_{s,tau,T}(v) = -tau * log sum_{p in P_T(K_s,v)} exp(-C_s(p)/tau).
```

Thus `F^{edge}` is a soft minimum over all paths that end on the board boundary. As `tau -> 0`, it converges to the ordinary minimum path cost. For finite `tau`, it rewards both cheap paths and multiplicity of cheap paths. A caged king should have high barrier free energy and low reachable mass; an exposed but mobile king should have lower free energy or many low-cost corridors.

The training objective is weighted cross-entropy over the binary label:

```text
min_theta E[ w_Y * CE(Y, h_theta(x, Phi_escape(x))) ] + lambda_TV * TV(c_s) + lambda_L2 * ||theta||_2^2,
```

where `Phi_escape` is the deterministic/low-parameter escape operator output and `h_theta` is the fusion classifier.

### Proposition

For any nonnegative cost field `c_s`, temperature `tau > 0`, and horizon `T`, the recurrence above satisfies

```text
D_{s,tau,T}(v) = -tau * log sum_{p in P_T(K_s,v)} exp(-C_s(p)/tau).
```

Consequently,

```text
lim_{tau -> 0+} D_{s,tau,T}(v) = min_{p in P_T(K_s,v)} C_s(p),
```

assuming at least one length-`T` path exists, which it does because self-loops are allowed.

### Proof sketch or derivation

The base case is immediate: there is exactly one zero-length path from `K_s` to itself and no zero-length path to any other square. For the induction step, every length-`t+1` path ending at `v` is a length-`t` path ending at some predecessor `u in N_K(v)` followed by the final square `v`, adding cost `c_s(x,v)`. Summing `exp(-cost/tau)` over all such predecessor-extended paths gives the recurrence. The zero-temperature limit follows from the standard log-sum-exp soft-min identity.

### What is actually proven

The recurrence exactly computes a smooth free energy over bounded frozen-board king paths under the chosen cost field, and its zero-temperature limit is the bounded-horizon shortest escape-path cost. The layer is differentiable in the learned cost parameters wherever finite `tau` is used.

### What remains only hypothesized

It is not proven that puzzle labels are caused by king cages. It is also not proven that the cost field learned from pseudo-legal attack maps is the right approximation of human tactical pressure. The hypothesis is empirical: the frozen-board escape free energy should improve classification, especially class-`1` near-puzzle recall at matched fine-label-`0` false-positive rate, and the connectivity-destroying ablation should lose that gain.

### Counterexamples where the idea should fail

- Quiet endgame studies where the tactic is zugzwang-like and not visible as a current-board king cage.
- Material tactics such as loose-piece forks where the opponent king has normal escape corridors.
- Positions where an exposed king is tactically safe because the opponent lacks forcing moves; the static hazard field may overpredict puzzle-likeness.
- Locked positions where a king is spatially boxed in but no tactic exists; the operator may produce false positives.
- Underpromotion, stalemate motifs, and long forcing lines that require legal move-tree reasoning.

### Self-critique

The strongest objection is that the operator may rediscover simple shortcuts: attacked-square count near the king, material/phase imbalance, or source-process artifacts correlated with castled kings. The minimal experiment is still worth running because the central ablation preserves these obvious marginals while destroying only corridor connectivity. If the full model beats the ablation and improves class-`1` recall at fixed fine-label-`0` false-positive rate, the evidence specifically supports connected escape geometry rather than generic king-danger density.

## 7. Architecture Specification

### Module names

Implement the main model in:

```text
src/chess_nn_playground/models/trunk/king_escape_percolation.py
```

Suggested classes/functions:

- `EncodingGeometryAdapter`
- `PseudoLegalAttackMaps`
- `EscapeCostField`
- `SoftMinEscapeDP`
- `KingEscapePercolationBlock`
- `KingEscapePercolationNet`
- `build_king_escape_percolation(config)`

### Forward-pass steps

1. **Input**
   - `x`: `(B, C, 8, 8)`.

2. **Encoding adapter**
   - Decode current-board piece planes into `pieces`: `(B, 2, 6, 8, 8)`.
   - Decode side-to-move into `stm`: `(B,)` or `(B,1)`.
   - Optionally decode castling/en-passant for the learned stem, but not for path costs unless the channel semantics are explicit.
   - Fail closed for unknown channel mappings.

3. **Pseudo-legal attack maps**
   - Compute `attack_counts`: `(B, 2, 1, 8, 8)`.
   - Compute optional attacker-type flags/counts: `(B, 2, 6, 8, 8)`.
   - Pawns use side-aware directions. Knights/kings use fixed kernels. Sliding pieces scan rays until the first occupied square. Do not generate legal moves.

4. **Per-defender geometric features**
   - For each defender side `s`, construct `geo_s`: `(B, F_geo, 8, 8)`, with `F_geo` expected around `24-32`.
   - Include defender piece planes, attacker piece planes, total occupancy, attacker attack count, attacker attack type flags, defender defense count, normalized Chebyshev distance to defender king, normalized distance to board edge, and side-to-move role bits.

5. **Learn constrained cost field**
   - Shared `1x1` MLP over `geo_s`: `F_geo -> cost_hidden_dim -> 1`.
   - `raw_cost`: `(B, 2, 1, 8, 8)`.
   - `cost = softplus(raw_cost) + base_cost + occupancy_barrier * occ`, with the defender king square occupancy penalty set to zero.
   - Clip or normalize cost to `[cost_min, cost_max]` for numerical stability.

6. **Soft escape dynamic program**
   - For each side and each `tau in escape_taus`, initialize `D_0`: `(B, 2, 1, 8, 8)` with `0` at the king and `large_value` elsewhere.
   - Iterate `escape_steps` times using eight shifted predecessor maps plus self-loop and a `logsumexp` softmin.
   - Save maps at configured `dp_snapshots`, for example `[1,2,3,4,6,8,12]`.
   - Output:
     - `escape_maps`: `(B, 2 * len(taus) * len(snapshots), 8, 8)` after squeezing singleton channels.
     - `escape_vec`: `(B, V_escape)`, containing edge free energies, ring free energies, reachable masses, side-to-move aligned differences, and per-side asymmetry features.

7. **Small learned board stem**
   - `stem(x)`: `(B, 32, 8, 8)` using a small convolutional stem, not a deep residual tower.
   - Suggested: one `3x3` conv, GroupNorm, SiLU, and one lightweight depthwise-separable residual block. This gives the model enough local pattern capacity without making the DP irrelevant.

8. **Fusion**
   - Concatenate `stem_out` and `escape_maps`: `(B, 32 + P_escape, 8, 8)`.
   - Apply a shallow `1x1` or `3x3` fusion block to `(B, 48 or 64, 8, 8)`.
   - Global mean-pool and max-pool the fused map: `(B, 96 or 128)`.
   - Concatenate `escape_vec`: `(B, 96/128 + V_escape)`.

9. **Classifier head**
   - MLP with hidden size `128`, dropout `0.05-0.10`, output logits `(B, 2)`.
   - Return logits only, so the shared trainer remains compatible.

### Parameter-count estimate

For `simple_18` with `cost_hidden_dim=16`, stem width `32`, fusion width `64`, and MLP hidden `128`, expected parameter count is roughly `60k-100k`, depending on the exact fusion block. For `lc0_*` with `C=112`, the first convolution adds about `27k` parameters relative to `simple_18`; the escape operator itself is almost unchanged.

### FLOP and complexity estimate

- Pseudo-legal attack maps: `O(B * 64 * ray_count * ray_length)` for sliding scans plus small fixed-kernel operations for leapers. On an `8x8` board this is minor.
- Soft DP: `O(B * 2 * len(escape_taus) * escape_steps * 64 * 9)` softmin terms. With `B=512`, `len(taus)=3`, and `escape_steps=12`, this is about `21 million` predecessor terms, which is modest on GPU.
- Memory for saved DP maps: `B * 2 * len(taus) * len(snapshots) * 64 * 4` bytes. With `B=512`, `3` temperatures, and `7` snapshots, this is about `5.5 MB` for maps before fusion.
- No generated move/candidate set is used. Therefore there is no candidate-set memory term. If Codex later adds optional target anchors, it must document memory as `O(B * max_candidates * candidate_dim)` and add chunking; do not add that in the first experiment.

### Required config fields

```yaml
model:
  name: king_escape_percolation
  input_channels: 18
  num_classes: 2
  encoding: simple_18
  cost_hidden_dim: 16
  escape_taus: [0.08, 0.25, 0.75]
  escape_steps: 12
  dp_snapshots: [1, 2, 3, 4, 6, 8, 12]
  occupancy_barrier: 3.0
  base_cost: 0.05
  cost_max: 8.0
  stem_width: 32
  fusion_width: 64
  classifier_hidden_dim: 128
  dropout: 0.1
  ablation_mode: none
```

### Encoding-adapter assumptions

- `simple_18`: Use the project's established 12 piece-plane order plus side-to-move, castling, and en-passant planes. If the project exposes channel metadata, use it. If not, put the convention in one adapter class and test it with synthetic boards.
- `lc0_static_112`: Use only explicitly identified current-board occupancy planes for rule geometry. All other planes may enter the learned stem but must not influence pseudo-legal attack maps unless semantics are known.
- `lc0_bt4_112`: Use only the latest/current board slice for rule geometry. Historical planes may enter the learned stem. If BT4 current slices cannot be located with certainty, raise `ValueError` and skip LC0 experiments.

### Pseudocode

```text
forward(x):
    pieces, stm, aux = adapter.decode_current_board(x)
    attacks = pseudo_legal_attack_maps(pieces)

    escape_maps, escape_vec = percolation_block(
        pieces=pieces,
        attacks=attacks,
        side_to_move=stm,
        ablation_mode=config.ablation_mode,
    )

    local = small_stem(x)
    fused_map = fusion_conv(concat_channel(local, escape_maps))
    pooled = concat(mean_pool(fused_map), max_pool(fused_map), escape_vec)
    logits = classifier(pooled)
    return logits
```

## 8. Loss, Training, And Regularization

- Primary loss: weighted cross-entropy over two logits, using the benchmark's coarse-binary labels.
- Optional auxiliary loss: none required. Keep the first run simple.
- Optional regularizers:
  - `lambda_cost_tv = 1e-4` total-variation penalty on learned cost maps to discourage arbitrary checkerboard costs.
  - Standard `weight_decay = 1e-4`.
  - Dropout `0.05-0.10` in the final MLP only.
- Class weighting: use the existing `balanced` class weighting from benchmark configs.
- Batch size expectations: start with `512` for `simple_18`. If DP map storage increases memory, reduce to `256` before changing the architecture.
- Learning-rate and optimizer defaults: AdamW or the existing trainer default optimizer, `learning_rate=0.001`, `weight_decay=0.0001`.
- Epochs: minimal experiment uses `3` epochs with early stopping patience `2`, matching the current lightweight benchmark pattern.
- Determinism requirements: set seed `42`, enable deterministic PyTorch mode where the project already does so, and make all shuffle/random-graph ablations use explicit seeded permutations stored in the report.
- Keep unchanged for fair comparison: train/val/test split paths, binary label mapping, class weighting policy, evaluation thresholds selected on validation data, metrics, report templates, and confusion-matrix generation.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| `ring_bin_cost_shuffle` | Shuffles each side's cost field within Chebyshev-distance rings from that king and within coarse occupancy/attack bins before the DP. | Central claim: connected low-cost corridors matter beyond cost histograms and ring marginals. | If this matches the full model, the DP is exploiting density shortcuts, not escape topology. |
| `ring_hist_vector_only` | Removes DP maps and feeds only per-ring histograms of occupancy, attack counts, and learned costs. | Tests whether the soft path operator adds information beyond obvious king-zone summaries. | If this matches the full model, future work should use simpler king-zone statistics or abandon this family. |
| `degree_preserving_neighbor_shuffle` | Replaces the 8-neighbor board graph with a fixed seeded random neighbor table preserving each square's degree and edge/corner/center category. | Tests whether board-grid corridor semantics matter rather than repeated soft aggregation. | If performance is unchanged, the recurrence acts like generic smoothing. |
| `cnn_only_param_matched` | Removes `escape_maps` and `escape_vec`; increases fusion/head width to match parameters. | Tests whether improvements are due only to extra capacity. | If equal, the structured operator is not justified. |
| `percolation_only` | Removes the raw learned CNN stem and classifies only from escape maps/vectors. | Tests whether the operator alone carries a strong puzzle signal. | If it works but fusion does not improve, the CNN stem may be overfitting or suppressing the bottleneck. |
| `no_attack_cost` | Cost field sees occupancy, distances, and side-to-move role bits but no pseudo-legal attack maps. | Tests whether attack semantics are necessary for the cage signal. | If unchanged, the model is likely using occupancy/king placement shortcuts. |
| `no_occupancy_barrier` | Removes or sets near-zero the deterministic occupancy barrier while keeping attacks. | Tests whether blockers and corridors, not only attacked squares, matter. | If unchanged, connectivity through blockers is not important for this dataset. |
| `single_temperature_hardmin` | Uses one very low temperature and no multi-temperature path multiplicity summaries. | Tests whether multiplicity of cheap escape paths is useful beyond shortest path. | If better, simplify the operator; if worse, keep multi-temperature free energy. |
| `wrong_attacker_semantics` | Uses defender's own attack map as hazard around its king, preserving attack density scale but breaking attacker/defender meaning. | Tests side-role semantics. | If unchanged, attack maps are not being used semantically. |
| `stm_alignment_removed` | Keeps both kings' features but removes side-to-move aligned asymmetry features. | Tests whether “side to move attacking opponent king” is part of the signal. | If unchanged, puzzle-likeness may be side-independent or already encoded elsewhere. |
| `king_seed_blind` | Seeds the DP from a fixed center/ring proxy instead of the true king square while preserving global maps. | Tests whether king-specific anchoring matters. | If unchanged, the method is not actually measuring king escape. |

For every central ablation, export the same `3x2` fine-label diagnostic matrix as the main model. For the shuffle ablations, also save the seed and a note describing exactly what marginals were preserved.

## 10. Benchmark And Falsification Criteria

Baselines to compare against:

- Existing simple CNN on `simple_18`.
- Existing residual CNN on `simple_18`, using the same split and training budget.
- Current best lightweight baseline in the project leaderboard for the same split, if available.
- `cnn_only_param_matched` ablation from this packet.
- `ring_bin_cost_shuffle` central ablation.
- Optional after the simple experiment: LC0-style CNN/residual CNN on `lc0_static_112` or `lc0_bt4_112`, but only if the channel mapping is already reliable.

Metrics to inspect:

- Validation and test accuracy.
- Balanced accuracy.
- AUROC.
- AUPRC.
- F1 at validation-selected threshold.
- Cross-entropy/log loss.
- Expected calibration error if the project already computes it.
- Fine-label `0/1/2 -> predicted 0/1` confusion for the main model and every central ablation.

Near-puzzle diagnostic:

- Select a threshold on validation that matches the fine-label-`0` false-positive rate of the best existing `simple_18` baseline, or use a fixed fine-label-`0` FPR of `5%` if no baseline threshold is available.
- At that threshold report fine-label-`1` recall, fine-label-`2` recall, positive precision, and the full `3x2` matrix.
- The key diagnostic is class-`1` recall at matched fine-label-`0` FPR, because near-puzzles are the label slice most likely to reveal whether the model learned puzzle structure rather than memorizing obvious verified puzzles.

Required artifacts:

- Config YAML for main and ablations.
- Model checkpoint path or training-log reference.
- Metrics JSON/CSV for train/val/test.
- Fine-label confusion matrices for main and central ablations.
- Prediction file with row id if available, true fine label, binary target, logits, probability, and predicted class.
- Percolation diagnostics: distributions of `F_edge` and reachable mass by fine label, and correlation with material count/phase if those nuisances are easy to compute.
- Report Markdown using the packet's report template.

Success threshold:

- Main model improves test AUROC by at least `0.01` or test AUPRC by at least `0.02` over the best comparable `simple_18` baseline, and
- fine-label-`1` recall at matched fine-label-`0` FPR improves by at least `3` absolute percentage points, and
- the central `ring_bin_cost_shuffle` ablation loses at least half of the main model's gain over the param-matched CNN.

Failure threshold:

- Main model differs from the param-matched CNN by less than `0.003` AUROC and less than `0.005` AUPRC, or
- `ring_bin_cost_shuffle` is statistically indistinguishable from the full model across the reported metrics, or
- class-`1` recall at matched fine-label-`0` FPR is worse than the best baseline.

Result that would make me abandon the idea:

- The full model, count-only ring histogram, and ring/bin shuffled ablation all perform the same within run noise. That outcome means the proposed connectivity operator is not doing useful work on this benchmark.

Result that would justify scaling:

- The full model clears the success threshold and central ablations show that true board-grid corridors matter. Then scale to longer training, try `lc0_static_112` with fail-closed geometry adapters, and consider adding a percolation-only interpretability report before any larger architecture changes.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_kep_king_escape_percolation/idea.yaml` | Create | Machine-readable idea metadata from Section 12. |
| `ideas/20260421_kep_king_escape_percolation/math_thesis.md` | Create | Mathematical thesis, proposition, proof sketch, falsifier, and self-critique from Sections 6 and 9. |
| `ideas/20260421_kep_king_escape_percolation/architecture.md` | Create | Architecture specification, tensor shapes, DP recurrence, parameter/complexity estimates, and adapter assumptions. |
| `ideas/20260421_kep_king_escape_percolation/implementation_notes.md` | Create | Notes on pseudo-legal attack-map computation, fail-closed encoders, numerical stability, and seeded ablations. |
| `ideas/20260421_kep_king_escape_percolation/trainer_notes.md` | Create | Training settings, unchanged benchmark requirements, class weighting, threshold selection, and reporting instructions. |
| `ideas/20260421_kep_king_escape_percolation/ablations.md` | Create | Ablation table and required interpretation. |
| `ideas/20260421_kep_king_escape_percolation/train.py` | Create | Thin idea-specific launcher that loads the config and calls the shared trainer; do not duplicate trainer logic. |
| `ideas/20260421_kep_king_escape_percolation/config.yaml` | Create | Minimal `simple_18` experiment config from Section 12. |
| `ideas/20260421_kep_king_escape_percolation/report_template.md` | Create | Template with metrics, `3x2` matrices, near-puzzle matched-FPR diagnostic, and percolation histograms. |
| `ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` | Update | Add this packet to imported memory and add anti-duplicate notes for king escape percolation if it fails. Preserve all hard leakage and falsification rules. |
| `src/chess_nn_playground/models/trunk/king_escape_percolation.py` | Create | `torch.nn.Module` implementation of adapters, pseudo-legal attack maps, cost field, softmin DP, fusion, and logits return. |
| `src/chess_nn_playground/models/registry.py` | Update | Register `king_escape_percolation` builder without disturbing existing models. |
| `configs/king_escape_percolation_simple18.yaml` | Create | Main benchmark config pointing at current `crtk_sample_3class` split and `simple_18`. |
| `configs/king_escape_percolation_simple18_shuffle_ablation.yaml` | Create | Same config with `model.ablation_mode: ring_bin_cost_shuffle`. |
| `configs/king_escape_percolation_simple18_cnn_only.yaml` | Create | Same config with DP disabled and parameter-matched CNN-only control. |
| `tests/test_king_escape_percolation.py` | Create | Focused tests for output shape, deterministic adapter failures, synthetic attack maps, DP finite outputs, and ablation seed determinism. |
| `tests/test_model_registry_king_escape_percolation.py` | Create or update | Ensure registry can instantiate the model and produce `(B,2)` logits from `(B,18,8,8)`. |

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0811_tuesday_pacific_king_escape_percolation.md
  generated_at: "2026-04-21 08:11 America/Los_Angeles"
  weekday: Tuesday
  timezone: Pacific
  idea_slug: king_escape_percolation
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260421_kep
  name: King Escape Percolation Network
  slug: king_escape_percolation
  status: draft
  created_at: "2026-04-21 08:11 America/Los_Angeles"
  author: ChatGPT Pro
  short_thesis: A differentiable soft shortest-path/free-energy operator over frozen-board king escape corridors can expose puzzle-like tactical cages not captured by ordinary CNN features.
  novelty_claim: Uses current-board pseudo-legal attack hazards as cell costs in a king-seeded square-grid percolation dynamic program; not an attack-defense sheaf, move-delta set, Sinkhorn transport, nuisance projection, ordinal head, or masked codec.
  expected_advantage: Better near-puzzle recall at matched non-puzzle false-positive rate, especially for positions with mating-net or king-cage structure.
  central_falsification_ablation: ring_bin_cost_shuffle preserving king-centered ring and occupancy/attack-bin marginals while destroying corridor connectivity before the DP.
  target_task: coarse_binary
  input_representation: simple_18 first; lc0_static_112/lc0_bt4_112 only with fail-closed current-board geometry adapters
  output_heads: two-class logits
  compute_notes: Soft DP cost is O(batch * 2 sides * temperatures * steps * 64 * 9); no move candidate set is generated.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/king_escape_percolation_simple18.yaml
  model_path: src/chess_nn_playground/models/trunk/king_escape_percolation.py
  latest_result_path: null
  notes: Export main and ablation 3x2 fine-label confusion matrices plus class-1 recall at matched fine-label-0 false-positive rate.
```

```yaml
config_yaml:
  run:
    name: king_escape_percolation_simple18
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
    name: king_escape_percolation
    input_channels: 18
    num_classes: 2
    encoding: simple_18
    cost_hidden_dim: 16
    escape_taus: [0.08, 0.25, 0.75]
    escape_steps: 12
    dp_snapshots: [1, 2, 3, 4, 6, 8, 12]
    occupancy_barrier: 3.0
    base_cost: 0.05
    cost_max: 8.0
    stem_width: 32
    fusion_width: 64
    classifier_hidden_dim: 128
    dropout: 0.1
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
  model_name: king_escape_percolation
  file_path: src/chess_nn_playground/models/trunk/king_escape_percolation.py
  builder_function: build_king_escape_percolation
  input_shape: [batch, 18, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - EncodingGeometryAdapter
    - PseudoLegalAttackMaps
    - EscapeCostField
    - SoftMinEscapeDP
    - KingEscapePercolationBlock
    - KingEscapePercolationNet
  required_config_fields:
    - input_channels
    - num_classes
    - encoding
    - cost_hidden_dim
    - escape_taus
    - escape_steps
    - dp_snapshots
    - occupancy_barrier
    - base_cost
    - cost_max
    - stem_width
    - fusion_width
    - classifier_hidden_dim
    - dropout
    - ablation_mode
  expected_parameter_count: "approximately 60k-100k for simple_18; about 27k more for a 112-channel learned stem"
  expected_memory_notes: "DP snapshots use B * 2 * len(taus) * len(snapshots) * 64 floats; with B=512, taus=3, snapshots=7 this is about 5.5 MB before fusion. No move candidate set is generated."
```

```yaml
research_continuity:
  idea_fingerprint: "current-board pseudo-legal attack/occupancy hazard maps + king-seeded 8-neighbor softmin escape dynamic program + free-energy/reachable-mass bottleneck + binary puzzle-likeness logits"
  already_researched_family_overlap: "Uses safe pseudo-legal attack maps, but not as an attack-defense graph/sheaf/Hodge object; no move-delta set, no Sinkhorn transport, no nuisance projection, no ordinal/credal head, no masked board codec."
  closest_duplicate_risk: "May be mistaken for a static attack-defense graph model because attack maps feed the cost field; distinguish it by the square-grid path free-energy operator and ring/bin-preserving corridor-shuffle falsifier."
  do_not_repeat_if_this_fails:
    - king escape softmin path free-energy from current board
    - board-edge or outer-ring escape percolation from king seeds
    - attack/occupancy cost-field corridor models with only density/ring-histogram controls
    - degree-preserving randomized king-grid neighbor ablations as the sole novelty
  suggested_next_search_directions:
    - label-safe selective prediction that changes decision policy without another ordinal or credal head
    - causal invariance using genuinely external data-source shifts if provenance becomes available and is not used as input
    - non-move-tree generative motif discovery that is not masked-board code length or pseudo-likelihood
    - calibrated active-error analysis of fine-label-1 near-puzzles at matched false-positive rates
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add `King Escape Percolation Network` to the imported research memory after implementation. | Prevents the next research pass from re-proposing soft king-corridor/free-energy path models as fresh ideas. | `Imported Research Memory` |
| Add an anti-duplicate fingerprint: `current-board attack/occupancy hazard field + king-seeded soft shortest-path/random-walk/percolation/free-energy corridor operator + binary puzzle-likeness`. | Captures the reusable family, not just this name. | `Research Continuity` or anti-duplicate rules below imported fingerprints |
| Require any future attack-map-as-cost idea to include a semantics-destroying ablation that preserves king-centered ring histograms and attack/occupancy marginals. | Avoids confusing attack-density shortcuts for structural corridor learning. | `Depth requirements` and `Ablation Plan` guidance |
| Clarify that LC0 geometry adapters must fail closed unless current-board channel semantics are verified. | Prevents accidental use of history/provenance-like channels in deterministic rule geometry. | `Project Context You Must Respect` under available encodings |
| Keep the matched fine-label-0 FPR diagnostic for class-1 recall as a required near-puzzle check. | It is a sharp diagnostic for whether ideas help ambiguous near-puzzles rather than only obvious puzzles. | `Benchmark And Falsification Criteria` requirements |

## 14. Final Sanity Check

- Downloadable Markdown file created: Yes
- Filename follows required date/time/day/timezone/slug pattern: Yes, `chess_nn_research_2026-04-21_0811_tuesday_pacific_king_escape_percolation.md`
- No forbidden engine features used as inputs: Yes
- Does not fabricate labels: Yes
- Not a routine CNN/ResNet/Transformer variant: Yes
- Minimal current-data experiment exists: Yes, `simple_18` on the existing `crtk_sample_3class` split
- Falsification criterion is concrete: Yes, the ring/bin-preserving cost shuffle and param-matched CNN controls are central
- Codex can implement without asking for missing architecture details: Yes
- Prompt maintenance notes included for Codex: Yes
- Repetition check against imported research packets completed: Yes
- Not a tactical sheaf/Hodge variant or one-ply move-delta pooling/spectrum/landscape variant: Yes
- Not a current-board piece-target/material-target Sinkhorn/OT transport variant: Yes
- Not a deterministic nuisance-orthogonal projection bottleneck variant: Yes
- Not an exact near-duplicate of imported ordinal, sparse-witness, ray-language, Möbius-constellation, or pseudo-likelihood packets: Yes
- Not an exact near-duplicate of imported orbit-symmetry, tempo-intervention, credal-evidence, rule-partition-invariance, kinematic-commutator, or masked-codec packets: Yes

# Codex Handoff Packet: Kinematic Commutator Bottleneck Network

## 1. File Metadata

- Filename: chess_nn_research_2026-04-21_0728_tuesday_local_kinematic_commutator.md
- Generated at: 2026-04-21 07:28:00 America/Los_Angeles
- Weekday: Tuesday
- Timezone: local
- Idea slug: kinematic_commutator
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: Kinematic Commutator Bottleneck Network, abbreviated KCBN.
- One-sentence thesis: Puzzle-like positions should be enriched for non-commuting interactions between different chess motion geometries, so expose Lie-bracket features `[K_i(x), K_j(x)]h` from rule-only current-board motion operators instead of asking a generic CNN to discover these second-order ordered interactions.
- Idea fingerprint: current-board deterministic piece occupancy and side-aware pseudo-legal motion operators; sparse operator bank for rays, leapers, pawns, and king adjacency; learned square features; pairwise Lie commutator maps; pooled commutator bottleneck; binary puzzle-likeness logits; no engine metadata, labels, move tree, Stockfish score, PV, node count, or verification provenance as input.
- Why this is not a common CNN/ResNet/Transformer variant: the central features are explicit non-commutative operator brackets over chess kinematics, not deeper convolution, residual stacking, square-token attention, ordinary GNN message passing, or LC0-style plane processing.
- Closest baseline or common method it resembles: a fixed-rule non-commutative operator neural network or algebraic signal-processing layer, but instantiated with chess motion operators and falsified by commutator-destroying controls rather than by tuning graph-filter depth.
- Current-data minimal experiment: train `KinematicCommutatorClassifier` on `simple_18` for the existing `crtk_sample_3class` train/val/test split with the shared binary trainer for 3 epochs, then run the same report artifacts as the current baselines.
- Smallest central falsification ablation: replace every Lie bracket `K_i K_j h - K_j K_i h` with the symmetric product summary `K_i K_j h + K_j K_i h` while preserving operator count, pair count, tensor shape, parameter count, and pooling head.
- Expected information gain if it fails: a clean failure says that ordered non-commutativity of rule-only piece motion is not carrying label signal beyond first-order motion maps, mobility-like degree statistics, material, and ordinary local CNN features on this split.

## 3. Problem Restatement And Data Contract

The task is binary chess puzzle-likeness classification from a single board-position tensor. The model outputs logits for:

- output `0`: non-puzzle
- output `1`: puzzle-like

The available fine labels are diagnostic source classes, not model inputs:

- fine label `0`: known non-puzzle
- fine label `1`: verified near-puzzle
- fine label `2`: verified puzzle

The benchmark is binary, but every report must include the rectangular `3x2` diagnostic matrix:

`true fine label 0/1/2 -> predicted binary output 0/1`.

The model must accept a tensor `(batch, C, 8, 8)` and return logits `(batch, 2)`. The minimal experiment uses:

- train: `data/splits/crtk_sample_3class/split_train.parquet`
- val: `data/splits/crtk_sample_3class/split_val.parquet`
- test: `data/splits/crtk_sample_3class/split_test.parquet`
- encoding: `simple_18`

The full roughly 45M-row Parquet dataset must not be used directly until streaming support exists.

Allowed neural-network inputs are current-board encodings and deterministic rule-derived geometry from the current board. Forbidden neural-network inputs are Stockfish scores, PVs, mate scores, node counts, verification metadata, source labels, proposed labels, data provenance, or anything derived from labels.

Leakage checklist:

- Safe: deterministic board coordinates, piece occupancy, side-to-move, castling/en-passant planes already present in the encoding, and pseudo-legal attack or reach geometry derived only from the current board.
- Risky and not used by the first experiment: full legal-move generation, legal move counts, checkmate/stalemate oracles, forced-line search, generated child positions, or move-tree consequences. These would need an explicit rule-only, engine-free justification and ablation before use.
- Never allowed: engine evaluation, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, or dataset provenance as neural-network inputs.
- For `simple_18`, the deterministic operator bank may read the 12 piece planes and side-to-move plane if the channel map is known.
- For `lc0_static_112` and `lc0_bt4_112`, current-board channels may be used for deterministic geometry only when Codex has an explicit channel-semantics map. History planes may be consumed only by a learned neural adapter, not by the rule geometry generator. If current-board piece-plane semantics are unknown, the adapter must fail closed before training.

## 4. Research Map

External ideas used:

1. Taco Cohen and Max Welling, "Group Equivariant Convolutional Networks," arXiv:1602.07576, https://arxiv.org/abs/1602.07576. Borrowed: the discipline of encoding a known geometric prior to reduce sample complexity. Not copied: group convolutions, rotation/reflection equivariance, or image-style symmetry assumptions.
2. Michael Bronstein, Joan Bruna, Taco Cohen, and Petar Veličković, "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges," arXiv:2104.13478, https://arxiv.org/abs/2104.13478. Borrowed: the geometric-learning view that task structure should determine the admissible operators. Not copied: any graph neural network, gauge network, or sheaf-like construction.
3. Mauricio Velasco et al., "Graph neural networks and non-commuting operators," NeurIPS 2024 / arXiv:2411.04265, https://arxiv.org/abs/2411.04265. Borrowed: non-commuting operator polynomials can be a distinct neural primitive. Not copied: graphon tuple theory, transferability theorems, or generic graph-tuple layers.
4. Alejandro Parada-Mayorga, Landon Butler, and Alejandro Ribeiro, "Convolutional Filters and Neural Networks with Non Commutative Algebras," arXiv:2108.09923 and IEEE TSP 2023, https://arxiv.org/abs/2108.09923. Borrowed: the idea that non-commutative convolutional signal models process information through operator algebras. Not copied: their spectral representation, stability proof, or network architecture.
5. Baker-Campbell-Hausdorff / Lie-bracket background, e.g. https://en.wikipedia.org/wiki/Baker%E2%80%93Campbell%E2%80%93Hausdorff_formula. Borrowed: the basic fact that commutators are the first correction term for order-dependent operator composition. Not copied: no continuous Lie-group dynamics or quantum-mechanical model is used.
6. Chess rules are used only as deterministic kinematic constraints. No engine-analysis paper, engine heuristic, or puzzle-verification method is used.

Candidate search trace:

| Candidate mechanism considered | Why it was serious | Why it lost to KCBN |
|---|---|---|
| Side-to-move causal invariance across exact color/180-degree transforms and file mirrors | It directly targets source artifacts and superficial orientation shortcuts | It is close to augmentation/adversarial invariance and may not add a distinct board-structure observable beyond existing encodings |
| Masked board denoising compression with puzzle classifier on residual surprisal | It fits minimum-description-length intuition and could suppress common material patterns | It risks duplicating the imported static-geometry pseudo-likelihood/description-length family unless class-conditioned and unary controls are redesigned |
| Cubical persistent homology of attack-pressure scalar fields | It is a genuinely different topological observable over board fields | It is harder to implement robustly, has more numerical choices, and is still adjacent to static attack/defense geometry without as crisp a central ablation |
| Selective prediction / evidential uncertainty for class `1` ambiguity | It respects label ambiguity and could improve calibration | It is mostly a head/loss change unless paired with a new representation, and it risks being a weaker variant of ordinal or independent binary heads |
| Rule-only electrical resistance or min-cut between kings and high-value pieces | It gives a mathematically clear global vulnerability statistic | It is close to graph Laplacian energy and therefore too near the imported sheaf/Hodge/Laplacian line |
| Diffusion-style corruption recovery of legal-looking boards | It could learn tactical motifs without engine signals | It is expensive for the current loop and too likely to become a generic generative pretraining proposal |
| Wavelet/scattering transforms over board-coordinate and piece-type channels | It supplies stable multiscale features | It is too close to fixed convolutional feature engineering and lacks a chess-specific falsifier |
| Non-commutative chess kinematic operator brackets | It is rule-only, compact, implementable, not a sheaf/OT/move-delta model, and has a direct semantics-destroying ablation | Selected |

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN | `src/chess_nn_playground/models/cnn.py` | Already exists and mainly tests ordinary local pattern recognition, not a new tactical observable. |
| Residual CNN | `src/chess_nn_playground/models/residual_cnn.py` | Already exists and would make the research loop a depth/optimization comparison. |
| LC0-style CNN / residual CNN | Existing LC0 BT4-style CNN and residual CNN variants | Already covered by current baselines and too close to copying LC0-style plane processing. |
| Ordinary ViT over 64 square tokens | None or any future vanilla Transformer baseline | It is explicitly disallowed as a core idea and has no chess-specific falsifier. |
| Plain GNN on squares or pieces | Generic graph neural network baseline | It would likely reduce to ordinary message passing over adjacency, attack, or occupancy graphs already explored by sheaf/graph families. |
| Hyperparameter tuning | All existing baselines | It is not a research idea and would not isolate a new mechanism. |
| Ensembling CNN, residual, and LC0 models | Any leaderboard ensemble | It may improve metrics but gives poor scientific attribution and is explicitly disallowed as the core idea. |
| More data or training on the full Parquet file | Any current trainer | Streaming is not ready and "add more data" is not an architectural research contribution. |
| Another attack-defense sheaf/Hodge/tension/curvature model | Imported tactical sheaf packets | Already researched family; changing edge labels or pooling would be a duplicate. |
| Another one-ply move-delta DeepSets/attention/spectrum/free-energy model | Imported counterfactual move-delta packets | Already researched family; it would also risk drifting toward move-tree leakage. |
| Another Sinkhorn or piece-target optimal-transport bottleneck | Imported OT packets | Already researched family; temperature/cost changes would not be novel enough. |
| Deterministic material/phase residualization | Imported nuisance-orthogonal packet | Closed-form projection over nuisance features is already covered and has a different falsifier. |
| Ordinal cumulative head for fine labels | Imported ordinal ladder packet | Already covered and focused on label structure rather than board operator structure. |
| Sparse witness-piece bottleneck | Imported sparse witness packet | Already covered; selecting pieces or material witnesses is not the new mechanism here. |
| Ray-language automaton | Imported ray-language packet | KCBN uses operator algebra on square fields, not string automata over ray tokens. |
| Möbius / ANOVA piece-constellation interactions | Imported constellation packet | KCBN studies ordered operator products, not explicit high-order occupied-piece interactions. |
| Static-geometry board pseudo-likelihood ratio | Imported pseudo-likelihood packet | KCBN is discriminative and bracket-based, not a class-conditioned generative description-length ratio. |

## 6. Mathematical Thesis

Input space definition:

Let `S = {1,...,64}` be the chessboard squares. An encoded position is `x in X`, represented to the network as `X_tensor(x) in R^{C x 8 x 8}`. Let `O_x in {0,1}^S` be deterministic occupancy extracted from current-board piece planes when channel semantics are known. Let `h_theta(x): S -> R^d` be a learned square feature map produced by a channel adapter from the input tensor.

Label/target definition:

The binary target is `Y in {0,1}`, where `0` means non-puzzle and `1` means puzzle-like. Fine labels `0,1,2` are used only for diagnostics and must not be inputs.

Data distribution assumptions:

The training and test splits are sampled from the current `crtk_sample_3class` process. The central hypothesis assumes that, conditional on coarse nuisance variables such as material, side-to-move, and phase, verified near-puzzles and puzzles are more likely than non-puzzles to contain localized interactions where the order of applying two piece-motion geometries matters. This is not proven; it is the empirical claim under test.

Allowed symmetry or equivariance assumptions:

Chess is not invariant to arbitrary board rotations or reflections because pawns, castling, promotion direction, and side-to-move matter. KCBN assumes only that rule-derived motion operators are side-aware and color-aware. A side-to-move canonical transform may be implemented only if it exactly swaps colors, rotates the board 180 degrees for black-to-move, updates pawn direction, and maps castling/en-passant channels consistently. The first experiment does not require any nontrivial augmentation.

Core hypothesis:

Let `K_m(x)` be a deterministic sparse linear operator on square fields for motion type `m`, built only from current-board occupancy, board boundaries, side-aware pawn directions, and line-of-sight blockers for sliders. Puzzle-likeness is hypothesized to have useful conditional mutual information with the Lie-bracket fields

`B_ij(x) h_theta(x) = (K_i(x) K_j(x) - K_j(x) K_i(x)) h_theta(x)`

beyond the information in first-order fields `{K_i(x)h_theta(x)}` and nuisance summaries.

Formal object introduced by the idea:

The formal object is the current-board chess kinematic operator algebra

`A_x = <K_m(x) : m in M>`

over square fields, together with its degree-two Lie-bracket subspace

`L_x^(2) = span{[K_i(x), K_j(x)] : i < j}`.

Here `M` contains side-aware ray, leaper, pawn-attack, and king-neighborhood operators. For sliders, `K_m(x)_{t,s}=1` when square `t` is reachable from source square `s` along the motion direction with all intermediate squares empty in the current board. This is pseudo-legal reach geometry, not legal-move generation.

Proposition:

For fixed `x`, suppose a target function contains a term

`f(x) = phi(<u, (K_i K_j - K_j K_i) h_theta(x)>)`

for some nonzero `u` and nonlinear scalar `phi`, and suppose there exist feature configurations with identical first-order summaries `{K_i h, K_j h}` and identical symmetric second-order product `K_i K_j h + K_j K_i h`, but opposite commutator response. Then any architecture whose representation is invariant under exchanging the ordered products `K_iK_j` and `K_jK_i` cannot represent `f`, while a linear readout over the commutator field can represent the signed preactivation before `phi`.

Proof sketch:

An exchange-invariant representation maps the two configurations to the same representation because it preserves only first-order fields and symmetric second-order products. Therefore any readout from that representation gives the same value to both configurations. The commutator maps them to opposite vectors because `[K_i,K_j]h` changes sign when the two ordered products are exchanged. A head with weight `u` separates the pair. Thus the commutator adds an antisymmetric degree-two non-commutative polynomial feature unavailable to commutative or order-symmetrized summaries.

Variational principle / objective:

KCBN trains a discriminative classifier

`p_theta(Y=1 | x) = softmax(g_theta(Pool({[K_i(x),K_j(x)] h_theta(x)}_{(i,j) in P}), Pool(h_theta(x))))_1`

by minimizing class-balanced cross-entropy on the binary labels, optionally with a small bracket-energy sparsity regularizer. The learning objective does not claim that puzzles equal high bracket norm; it asks whether learned bracket patterns improve generalization under hard ablations.

What is actually proven:

The proposition proves only a representational separation between order-sensitive commutator features and order-symmetric summaries for constructed configurations. It also proves that if all chosen operators commute on a given board-feature pair, KCBN's bracket branch contributes zero on that pair.

What remains only hypothesized:

It is not proven that real puzzle labels depend on these commutators, that the split is free of confounding artifacts, or that degree-two brackets are the right order. It is also unproven that KCBN will outperform a strong CNN.

Counterexamples where the idea should fail:

- Quiet zugzwang, endgame tablebase-like ideas, or strategic maneuvers whose puzzle-likeness is not visible in current-board kinematic interference.
- Positions where one dominant motif is fully captured by a single first-order attack map and no operator-order interaction is needed.
- Dataset artifacts where labels correlate mostly with material imbalance, source, check frequency, or diagram style.
- Long forced lines where child positions, not current-board geometry, carry the signal.
- Boards where bracket magnitude is dominated by edge effects or mobility degree rather than tactical semantics.

Self-critique:

The strongest objection is that `K_iK_j - K_jK_i` may collapse to a fancy mobility, blocker, or board-edge statistic. The minimal experiment is still worth running because the central ablations preserve tensor shape, operator counts, degrees, and much of the compute while destroying ordered non-commutative semantics. If degree-preserving random operators or symmetric products match KCBN, the idea should be abandoned rather than tuned.

## 7. Architecture Specification

Module names:

- `KinematicCommutatorClassifier`
- `EncodingSemanticAdapter`
- `RuleMotionOperatorBank`
- `SparseMotionApply`
- `LieBracketPairBlock`
- `CommutatorPoolingHead`

Default model config:

- `input_channels: 18`
- `hidden_dim: 48`
- `operator_set: side_aware_basic_12`
- `num_operator_pairs: 28`
- `pair_chunk_size: 4`
- `commutator_abs: true`
- `include_first_order_control_branch: true`
- `first_order_branch_dim: 24`
- `dropout: 0.10`
- `num_classes: 2`

Forward-pass steps and shapes:

1. Input tensor: `x` has shape `(B, C, 8, 8)`.
2. `EncodingSemanticAdapter`:
   - learned square features `H0 = Conv1x1(C -> d)(x)`, shape `(B, d, 8, 8)`;
   - flatten to `H`, shape `(B, d, 64)`;
   - extract deterministic piece occupancy and side-to-move from current-board channels for `simple_18`, shape `(B, 12, 64)` plus side-to-move `(B,)`.
3. `RuleMotionOperatorBank`:
   - constructs or retrieves sparse operator edge lists for each motion type `m`.
   - default motion types: four orthogonal ray directions, four diagonal ray directions, aggregate knight jumps, aggregate king one-step adjacency, side-to-move pawn attacks, non-side-to-move pawn attacks.
   - sliders use current occupancy for line-of-sight blocker masks.
   - no full legal move generation, king-safety filtering, checkmate test, move counts, or child boards.
4. First-order optional branch:
   - compute `Y_m = K_m(x) H` for each operator, shape per operator `(B, d, 64)`;
   - pool mean and max over squares/operators into `(B, 2d)` after a small linear compressor.
5. Lie-bracket branch:
   - for each selected pair `(i,j)`, compute `C_ij = K_i(x)(K_j(x)H) - K_j(x)(K_i(x)H)`, shape `(B, d, 64)`;
   - use absolute value and signed mean features: maps `abs(C_ij)` plus scalar signed means;
   - process pairs in chunks of `pair_chunk_size` to avoid materializing all pair maps at once.
6. Pair map compression:
   - concatenate chunked commutator maps conceptually as `(B, P*d, 8, 8)`;
   - apply a shared pair-compression `1x1` projection from `P*d` to `d`, or chunk-accumulate through a shared `Linear(d -> d)` and sum over learned pair embeddings;
   - output `Hc`, shape `(B, d, 8, 8)`.
7. Pooling:
   - `mean_pool(Hc)` and `max_pool(Hc)` -> `(B, 2d)`;
   - commutator scalar stats per pair: mean absolute value and max absolute value -> `(B, 2P)`;
   - optional stem pool from `H0` -> `(B, 2d)`.
8. `CommutatorPoolingHead`:
   - concatenate pooled vectors, default `(B, 2d + 2d + 2P)` = `(B, 248)` for `d=48, P=28`;
   - MLP `248 -> 128 -> 2`;
   - return logits `(B, 2)`.

Parameter-count estimate:

- `simple_18` input adapter: about `18*48 + 48 = 912` parameters.
- Pair compression and pair embeddings: about `65k` parameters if implemented as a `1x1` projection over all `P*d` maps; less if chunk-shared.
- Classifier head: about `32k` parameters.
- Normalization, compressors, and biases: under `20k`.
- Expected total: `0.10M` to `0.16M` trainable parameters for the default model, depending on pair-compression implementation. For `lc0_*` with `112` input channels, add about `4.5k` adapter parameters.

FLOP / complexity estimate:

Let `E_m(x)` be the number of sparse edges in operator `K_m(x)`, `E_avg` the average per operator, `P` the number of commutator pairs, and `d` hidden channels. A first-order pass costs approximately `O(B*d*sum_m E_m)`. The commutator branch costs approximately

`O(B*d*sum_(i,j in P) (E_i + E_j))`.

For default `B=512`, `d=48`, `P=28`, and `E_avg` around a few hundred edges per operator, expect several hundred million sparse gather-add operations per batch. This is acceptable for a 3-epoch sample benchmark if implemented with vectorized gathers and pair chunking.

Candidate-set memory estimate and chunking plan:

KCBN does not generate a move candidate set. The large intermediate is the conceptual commutator map tensor `(B, P, d, 64)`. Its float32 memory is:

`4 * B * P * d * 64` bytes.

For `B=512, P=28, d=48`, this is about `176 MB` if fully materialized. Codex should process pairs in chunks, default `pair_chunk_size=4`, reducing this intermediate to about `25 MB` plus accumulators. Mixed precision is off by default for deterministic comparison, but the implementation should be compatible with AMP later.

Required config fields:

- `model.name`
- `model.input_channels`
- `model.num_classes`
- `model.hidden_dim`
- `model.operator_set`
- `model.num_operator_pairs`
- `model.pair_chunk_size`
- `model.encoding_adapter`
- `model.fail_closed_on_unknown_channels`
- `model.include_first_order_control_branch`
- `model.dropout`

Encoding support:

- First experiment should use only `simple_18` because the current 12 piece planes and side-to-move semantics are explicit enough for deterministic operator construction.
- `lc0_static_112` support is feasible only if Codex adds a tested channel map for current-board piece planes and side-to-move. The learned adapter may use all 112 channels, but the rule operator bank may read only current-board piece planes.
- `lc0_bt4_112` support is feasible under the same rule. Unavailable history planes are zero-filled by the current exporter; the deterministic geometry generator must ignore history channels unless a verified current-board channel map is present.
- All adapters must fail closed with a clear exception if channel semantics are unknown. They must not guess channel order.

Pseudocode:

    def forward(x):
        H0 = input_projection(x)                  # (B,d,8,8)
        H = flatten_squares(H0)                   # (B,d,64)
        piece_planes, stm = semantic_adapter.extract_current_board(x)
        ops = operator_bank(piece_planes, stm)    # sparse K_m(x), no move tree

        first_order_summary = first_order_pool(ops, H) if enabled else empty

        comm_summary_acc = []
        comm_map_acc = zeros(B, d, 8, 8)
        for pair_chunk in chunks(selected_pairs, pair_chunk_size):
            chunk_maps = []
            for i, j in pair_chunk:
                K_i, K_j = ops[i], ops[j]
                C = apply(K_i, apply(K_j, H)) - apply(K_j, apply(K_i, H))
                chunk_maps.append(abs(C).reshape(B, d, 8, 8))
                comm_summary_acc.append(pair_stats(C))
            comm_map_acc += pair_chunk_project(chunk_maps)

        pooled = concat(meanmax(comm_map_acc), first_order_summary, concat(comm_summary_acc))
        logits = head(dropout(pooled))
        return logits

How the model returns logits:

`KinematicCommutatorClassifier.forward(x)` returns a dense tensor of shape `(batch, num_classes)`, with `num_classes=2`, compatible with the shared trainer, reports, confusion matrices, predictions, and leaderboards.

## 8. Loss, Training, And Regularization

Primary loss:

- Class-balanced cross-entropy over the coarse binary target.

Optional auxiliary loss:

- Optional bracket-energy sparsity: `lambda_bracket_l1 * mean(abs(normalized_commutator_stats))`, default `lambda_bracket_l1 = 0.0` for the first benchmark. Turn it on only after the main and ablation runs, not for the central comparison.

Class weighting:

- Use existing `class_weighting: balanced` behavior from the shared trainer. Do not derive weights from fine labels except through the existing coarse target mapping.

Batch size expectations:

- Default `batch_size: 512` for parity with sample benchmarks.
- If memory is high, lower `pair_chunk_size` before lowering batch size. If batch size must change, repeat the strongest baseline with the same batch size.

Learning-rate and optimizer defaults:

- AdamW or the shared trainer's existing Adam-compatible default.
- `learning_rate: 0.001`
- `weight_decay: 0.0001`
- `epochs: 3`
- `early_stopping_patience: 2`

Regularizers:

- Dropout `0.10` in the pooling head.
- Operator-pair dropout `0.05` may be added after the first run but should not be used in the central first benchmark unless baselines have comparable dropout.
- No data augmentation in the first run unless the exact same augmentation is applied to the compared baselines.

Determinism requirements:

- `seed: 42`
- `deterministic: true`
- deterministic selected pair list
- deterministic sparse edge ordering
- fixed random seeds for randomized ablations
- report the git commit or patch hash if available

What must stay unchanged for fair comparison:

- split paths
- binary target mapping
- `3x2` fine-label diagnostic reporting
- optimizer family unless current configs require otherwise
- epochs and early-stopping policy
- batch size if memory allows
- encoding for same-family comparisons
- report artifacts, prediction schema, and leaderboard format

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| Symmetric product control | Replace `[K_i,K_j]H` with `K_iK_jH + K_jK_iH`; preserve pair count, maps, parameters, pooling | Ordered non-commutativity, not just second-order reach, matters | If it matches KCBN, the Lie-bracket claim is falsified |
| First-order only | Use `{K_iH}` maps and pooling but remove all pairwise products | Degree-two operator interactions add value beyond attack/reach maps | If it matches KCBN, brackets are not needed |
| Zero-commutator diagonal degree control | Replace each `K_i` by a diagonal matrix containing source out-degree or target in-degree; commutators vanish | Mobility degree alone should not explain performance | If it matches KCBN, the model is using count/mobility shortcuts |
| Degree-preserving randomized operator bank | For each motion type, use a fixed random sparse matrix with matched per-square in/out degree and approximate color/parity bins, gated by the same occupancy extraction | Chess geometry matters, not just sparse operator degree | If it matches KCBN, the chess kinematic semantics are not doing work |
| Empty-board operators | Ignore blockers for sliders while preserving board boundaries and motion directions | Current-board line-of-sight blockers matter | If it matches KCBN, blocker-dependent pseudo-legal geometry is unnecessary |
| Edge-effect masked run | Downweight or remove commutator stats from rim squares in the pooled head | Performance is not just boundary non-commutation | If metrics collapse only because of rim removal, the branch may be exploiting board-edge artifacts |
| Pair-label shuffle | Keep the same computed commutator maps but randomly permute pair embeddings/labels per run | Specific operator-pair identity matters | If it matches KCBN, the head may only use generic magnitude |
| Material/side nuisance summary | Train a tiny MLP on material counts, side-to-move, castling, and en-passant only | Establish shortcut floor | If close to KCBN, the split is dominated by trivial metadata-like signals |
| Matched-parameter shallow CNN | Replace commutator branch with a shallow parameter-matched `1x1/3x3` CNN head | KCBN is not merely benefiting from extra parameters | If it matches KCBN, the operator algebra is not justified |
| Full operator semantics with frozen random adapter | Freeze the input projection randomly and train only the commutator head | Learned square features are needed, not just raw channel algebra | If it performs well, raw occupancy brackets may already carry most signal |

For the structured operator, the key semantics-destroying ablation is the degree-preserving randomized operator bank. It preserves obvious shortcuts such as edge count, degree scale, source-square marginal pressure, and sparse compute pattern while destroying chess motion semantics. There is no generated move candidate set, but the count-only diagonal control and material/side nuisance summary provide the analogous controls for mobility and material shortcuts.

## 10. Benchmark And Falsification Criteria

Baselines to compare against:

- existing `simple_18` simple CNN, same split and training budget
- existing `simple_18` residual CNN, same split and training budget
- best already-recorded LC0 static/BT4 CNN or residual CNN as a context row, clearly marked as a different encoding if not retrained
- KCBN main model on `simple_18`
- all central KCBN ablations: symmetric product, first-order only, degree-preserving randomized operator bank, diagonal degree control, and material/side nuisance summary

Metrics to inspect:

- validation and test accuracy
- AUROC
- AUPRC
- macro F1 and positive-class F1
- balanced accuracy
- calibration: ECE or Brier if already available in reports
- rectangular `3x2` fine-label confusion matrix for every main and central ablation run

Near-puzzle diagnostic:

- Report class `1` recall at a matched fine-label-`0` false-positive rate. Use thresholds chosen so each model has the same false-positive rate on fine label `0` as the best simple_18 residual CNN, then compare recall on fine label `1`.
- Also report class `1` precision among predicted puzzle-like examples if the reporting code already supports it.

Required artifacts:

- trained checkpoint or pointer following existing convention
- validation and test metrics JSON/CSV
- predictions Parquet/CSV with probabilities and labels
- `3x2` diagnostic confusion matrix for main and ablations
- ablation summary table
- leaderboard row
- short report using `ideas/20260421_kinematic_commutator/report_template.md`

Success threshold:

- Primary success: KCBN beats the best same-encoding `simple_18` baseline by at least `+1.0` percentage point test AUROC or at least `+5.0` percentage points class-`1` recall at matched fine-label-`0` false-positive rate.
- Mechanism success: the symmetric-product and degree-preserving randomized-operator ablations are each at least `0.5` AUROC point worse than KCBN or materially worse on the matched-FPR class-`1` diagnostic.

Failure threshold:

- KCBN is within `0.2` AUROC point of first-order-only, symmetric-product, or degree-preserving random-operator controls and has no class-`1` diagnostic improvement.
- KCBN underperforms the simple CNN and residual CNN while random or degree-only controls match it.

What result would make me abandon the idea:

- Degree-preserving randomized operators or diagonal degree controls match or exceed KCBN on test AUROC and class-`1` matched-FPR recall. That would show the bracket semantics are unnecessary on this dataset.

What result would justify scaling:

- KCBN clears the success threshold and the central semantics-destroying ablations drop meaningfully. Then Codex may try a second block, larger hidden dimension, `lc0_static_112` with fail-closed current-board channel maps, and longer training. Scaling should not happen before the hard ablations.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_kinematic_commutator/idea.yaml` | Create | Machine-readable idea metadata copied from the `idea_yaml` block, including status `draft` until first benchmark. |
| `ideas/20260421_kinematic_commutator/math_thesis.md` | Create | Section 6 expanded only as needed; include proposition, proof sketch, hypotheses, and failure modes. |
| `ideas/20260421_kinematic_commutator/architecture.md` | Create | Section 7 with exact tensor shapes, operator definitions, chunking, and adapter fail-closed rules. |
| `ideas/20260421_kinematic_commutator/implementation_notes.md` | Create | Sparse edge-list construction, deterministic operator ordering, occupancy blocker logic, and no-move-tree guardrails. |
| `ideas/20260421_kinematic_commutator/trainer_notes.md` | Create | Loss, class weighting, deterministic flags, baseline parity, and report requirements. |
| `ideas/20260421_kinematic_commutator/ablations.md` | Create | Section 9 ablation plan with runnable config names. |
| `ideas/20260421_kinematic_commutator/train.py` | Create | Thin wrapper or documented command entry that calls the shared trainer with the idea config; do not duplicate trainer logic. |
| `ideas/20260421_kinematic_commutator/config.yaml` | Create | Default KCBN simple_18 config for the sample split. |
| `ideas/20260421_kinematic_commutator/report_template.md` | Create | Template requiring baseline comparison, metrics, `3x2` matrices, class-`1` matched-FPR diagnostic, and ablation verdict. |
| `ideas/all_ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` | Update | Add this idea fingerprint to imported memory after implementation; add anti-duplicate language for non-commutative chess kinematic Lie-bracket operator banks if it fails or succeeds. Preserve all leakage, label, and falsification constraints. |
| `src/chess_nn_playground/models/kinematic_commutator.py` | Create | `KinematicCommutatorClassifier`, adapter, operator bank, sparse apply, bracket block, pooling head, and ablation modes. |
| `src/chess_nn_playground/models/registry.py` | Update | Register builder name `kinematic_commutator_bottleneck`. |
| `configs/kinematic_commutator_simple18.yaml` | Create | Shared-trainer config pointing to the current split, `simple_18`, binary mode, and the new model. |
| `configs/kinematic_commutator_simple18_symmetric.yaml` | Create | Central symmetric-product ablation. |
| `configs/kinematic_commutator_simple18_first_order.yaml` | Create | First-order-only ablation. |
| `configs/kinematic_commutator_simple18_random_degree.yaml` | Create | Degree-preserving randomized operator ablation. |
| `configs/kinematic_commutator_simple18_degree_only.yaml` | Create | Diagonal degree/count-only ablation. |
| `tests/test_kinematic_commutator.py` | Create | Focused tests for forward shape, deterministic edge ordering, fail-closed adapter behavior, zero commutator for identical operators, nonzero bracket on a constructed blocker board, and ablation tensor-shape parity. |
| `tests/test_model_registry_kinematic_commutator.py` | Create if registry tests exist | Ensure the model builds from config and returns `(batch, 2)`. |

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0728_tuesday_local_kinematic_commutator.md
  generated_at: 2026-04-21T07:28:00-07:00
  weekday: Tuesday
  timezone: local
  idea_slug: kinematic_commutator
  format: markdown
```

```yaml
idea_yaml:
  idea_id: 20260421_kinematic_commutator
  name: Kinematic Commutator Bottleneck Network
  slug: kinematic_commutator
  status: draft
  created_at: 2026-04-21T07:28:00-07:00
  author: ChatGPT Pro
  short_thesis: Puzzle-like positions may be enriched for non-commuting interactions between deterministic chess motion operators, captured by Lie-bracket maps over learned square features.
  novelty_claim: Uses rule-only current-board kinematic operator commutators as the central bottleneck, not CNN depth, sheaf/Hodge tension, move-delta pooling, optimal transport, ordinal heads, sparse witnesses, ray automata, Möbius constellations, or pseudo-likelihood ratios.
  expected_advantage: Better class-1 near-puzzle recall at matched fine-label-0 false-positive rate by exposing ordered piece-geometry interference.
  central_falsification_ablation: Replace Lie brackets K_iK_jH - K_jK_iH with symmetric products K_iK_jH + K_jK_iH while preserving shape and parameter count.
  target_task: coarse_binary
  input_representation: simple_18 first; lc0_static_112 and lc0_bt4_112 only with fail-closed current-board channel maps
  output_heads: coarse_binary_logits
  compute_notes: Sparse current-board operator applications with pair chunking; no full legal move generation or child-board evaluation.
  implementation_status: not_implemented
  trainer_entrypoint: scripts/train_model.py
  config_path: configs/kinematic_commutator_simple18.yaml
  model_path: src/chess_nn_playground/models/kinematic_commutator.py
  latest_result_path: null
  notes: Run symmetric-product, first-order-only, random-degree, and degree-only ablations before scaling.
```

```yaml
config_yaml:
  run:
    name: kinematic_commutator_simple18
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
    name: kinematic_commutator_bottleneck
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

```yaml
model_spec:
  model_name: kinematic_commutator_bottleneck
  file_path: src/chess_nn_playground/models/kinematic_commutator.py
  builder_function: build_kinematic_commutator_bottleneck
  input_shape: [batch, C, 8, 8]
  output_shape: [batch, num_classes]
  key_modules:
    - EncodingSemanticAdapter
    - RuleMotionOperatorBank
    - SparseMotionApply
    - LieBracketPairBlock
    - CommutatorPoolingHead
  required_config_fields:
    - input_channels
    - num_classes
    - hidden_dim
    - operator_set
    - num_operator_pairs
    - pair_chunk_size
    - encoding_adapter
    - fail_closed_on_unknown_channels
    - include_first_order_control_branch
    - dropout
  expected_parameter_count: 0.10M-0.16M for simple_18 default hidden_dim=48 and 28 operator pairs
  expected_memory_notes: Conceptual commutator tensor is 4*B*P*d*64 bytes; use pair_chunk_size=4 to keep default B=512 intermediate around 25MB per chunk instead of about 176MB fully materialized.
```

```yaml
research_continuity:
  idea_fingerprint: current-board deterministic chess kinematic operators plus learned square features plus degree-two Lie commutator maps [K_i,K_j]H pooled for binary puzzle-likeness
  already_researched_family_overlap: Touches rule-derived attack/reach geometry but is not a sheaf/Hodge/Laplacian/tension model, not a one-ply move-delta model, not Sinkhorn/OT, not nuisance projection, not ordinal, not sparse witness, not ray automaton, not Möbius constellation, and not pseudo-likelihood.
  closest_duplicate_risk: Could be mistaken for a static attack graph model; the distinguishing falsifier is order-sensitive non-commutative brackets versus symmetric, first-order, and degree-preserving randomized operators.
  do_not_repeat_if_this_fails:
    - Do not propose another chess motion Lie-bracket or non-commutative operator-polynomial bottleneck with different pair labels, hidden size, or pooling.
    - Do not rescue the idea by adding deeper CNN layers before the symmetric-product and random-degree controls are beaten.
    - Do not repackage bracket norms as curvature, tension, or Hodge energy.
    - Do not convert it into a one-ply move-delta landscape.
  suggested_next_search_directions:
    - Label-safe selective prediction specifically for fine-label-1 ambiguity with independent-binary controls.
    - Causal invariance across source-like environments, material phases, and exact side-to-move transforms without closed-form nuisance projection.
    - Masked generative compression with strong unary/material and randomized-neighborhood controls, carefully distinguished from class-conditioned pseudo-likelihood.
    - Calibration-first models that improve abstention and near-puzzle diagnostics without changing input features.
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add KCBN to imported research memory after implementation with the fingerprint `current-board chess kinematic operators + Lie commutator [K_i,K_j]H + commutator bottleneck pooling`. | Prevents the next research cycle from repeating the same non-commutative operator idea under names like bracket curvature, operator algebra, or Lie tactical interference. | Imported Research Memory |
| Add an anti-duplicate clause: do not propose another non-commutative chess motion operator-polynomial model unless the formal observable is not degree-two Lie brackets of deterministic current-board motion operators. | This closes the obvious escape hatch of changing pair labels, adding nested brackets, or renaming brackets as curvature. | Anti-duplicate rules below imported fingerprints |
| Add a reusable hard control for operator-algebra models: symmetric-product, first-order-only, and degree-preserving randomized-operator ablations are mandatory. | Keeps future operator proposals falsifiable and prevents fancy algebra from hiding mobility/degree shortcuts. | Ablation requirements |
| Add a note that rule-only slider line-of-sight operators are allowed only when they do not generate legal child boards, legal move counts, checkmate/stalemate flags, or engine-derived consequences. | Clarifies the boundary between safe current-board pseudo-legal geometry and leakage-prone move generation. | Problem/data contract guidance |
| If KCBN fails by matching random-degree controls, add a warning that current split may reward degree/mobility statistics and future ideas should include nuisance-preserving controls early. | Converts failure into prompt memory rather than repeated architecture proposals. | Research Continuity guidance |

## 14. Final Sanity Check

- Downloadable Markdown file created: yes
- Filename follows required date/time/day/timezone/slug pattern: yes, `chess_nn_research_2026-04-21_0728_tuesday_local_kinematic_commutator.md`
- No forbidden engine features used as inputs: yes
- Does not fabricate labels: yes
- Not a routine CNN/ResNet/Transformer variant: yes
- Minimal current-data experiment exists: yes, `simple_18` on the existing `crtk_sample_3class` split
- Falsification criterion is concrete: yes, symmetric-product and degree-preserving randomized-operator controls must drop
- Codex can implement without asking for missing architecture details: yes
- Prompt maintenance notes included for Codex: yes
- Repetition check against imported research packets completed: yes
- Not a tactical sheaf/Hodge variant or one-ply move-delta pooling/spectrum/landscape variant: yes
- Not a current-board piece-target/material-target Sinkhorn/OT transport variant: yes
- Not a deterministic nuisance-orthogonal projection bottleneck variant: yes
- Not an exact near-duplicate of imported ordinal, sparse-witness, ray-language, Möbius-constellation, or pseudo-likelihood packets: yes

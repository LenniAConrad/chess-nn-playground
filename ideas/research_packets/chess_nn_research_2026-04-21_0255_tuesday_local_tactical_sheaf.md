# Codex Handoff Packet: Tactical Sheaf Tension Network

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-21_0255_tuesday_local_tactical_sheaf.md`
- Generated at: 2026-04-21 02:55:34 America/Los_Angeles
- Weekday: Tuesday
- Timezone: local
- Idea slug: `tactical_sheaf`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: Tactical Sheaf Tension Network, abbreviated `TSTN`.
- One-sentence thesis: classify chess puzzle-likeness by learning a side-aware cellular sheaf over pseudo-legal attack, defense, and x-ray relations, then pooling the sheaf-coboundary tension created by incompatible local tactical claims.
- Idea fingerprint: deterministic current-board tactical complex + directed attack/defense/x-ray edge stalks + learned relation-tied restriction maps + sheaf Dirichlet energy pooling + left-right partial equivariance only.
- Why this is not a common CNN/ResNet/Transformer variant: the core computation is not image convolution, residual scaling, full-board attention, or a standard square-token GNN; it constructs a sparse typed tactical complex from legal move geometry and uses a learned coboundary/Laplacian operator whose edge residuals are the primary classification signal.
- Current-data minimal experiment: train one `tactical_sheaf_tension_net` on `simple_18` with the provided train/val/test split, three deterministic seeds, binary cross-entropy from fine-label collapse `0 -> 0`, `{1,2} -> 1`, and report the standard fine-label confusion slices; only run `lc0_static_112` after the `simple_18` sanity pass beats or meaningfully differs from the best non-ensemble CNN-family baseline.
- Expected information gain if it fails: a clean failure, especially if the trivial-edge ablation matches it, would falsify the claim that explicitly modeled attack-defense sheaf tension adds useful puzzle-likeness information beyond ordinary spatial features on the current dataset; future cycles should then prefer engine-free differentiable search surrogates or causal split-invariance rather than another attack-graph sheaf.

## 3. Problem Restatement And Data Contract

Task: binary chess puzzle-likeness classification from board-position tensors.

Outputs:

- `0`: non-puzzle.
- `1`: puzzle-like.

Source fine labels, used for training-label collapse and reporting but never as input features:

- fine label `0`: known non-puzzle.
- fine label `1`: verified near-puzzle.
- fine label `2`: verified puzzle.

Binary target construction:

- `y_binary = 0` when `fine_label == 0`.
- `y_binary = 1` when `fine_label in {1, 2}`.
- Do not fabricate or alter class `1` or class `2`; unresolved candidates remain unresolved and are not promoted into verified examples.

Allowed neural-network inputs:

- Encoded board tensors from the existing encodings: `simple_18`, `lc0_static_112`, and `lc0_bt4_112`.
- Deterministic geometric features computed from the current encoded board, such as pseudo-legal attack rays, defense edges, blockers, piece identity, side to move, and current occupancy.
- Constants derived from chess rules, for example knight offsets or bishop ray directions.

Forbidden neural-network inputs:

- Stockfish scores.
- Principal variations.
- Engine node counts.
- Engine best moves.
- Puzzle-verification metadata.
- Source labels or proposed labels as features.
- Split membership or example identifiers.
- Any signal that encodes whether the position was verified as a puzzle or near-puzzle.

Tensor contract:

- Model input: `(batch, C, 8, 8)`.
- Model output: logits `(batch, num_classes)`, where `num_classes = 2` for the current benchmark.
- Internally, `TSTN` flattens the board to 64 square nodes but preserves a standard PyTorch `nn.Module` interface.

Benchmark split:

```text
data/splits/crtk_sample_3class/split_train.parquet
data/splits/crtk_sample_3class/split_val.parquet
data/splits/crtk_sample_3class/split_test.parquet
```

Leakage checklist for Codex:

- The tactical graph builder must read only the input tensor and fixed chess-rule geometry.
- It must not call Stockfish or any chess engine.
- It must not inspect target labels, fine labels, puzzle IDs, verification tags, or source metadata.
- It must not use proposed labels for unresolved candidates.
- For `lc0_bt4_112`, build the attack-defense sheaf from the current-board piece planes only; the full tensor may still feed the square encoder if that is how existing baselines treat the encoding.
- Fine labels `0/1/2` may be used only for target collapse, stratified reporting, and confusion-matrix slices.
- Any optional file-mirror consistency regularizer must transform the input tensor and castling-side channels correctly; disable it for encodings whose mirror transform is not already reliable.

## 4. Research Map

This idea borrows mathematical operators, not ready-made chess architectures.

| Paper or idea | URL | What is borrowed | What is not copied |
|---|---|---|---|
| Hansen and Ghrist, “Toward a Spectral Theory of Cellular Sheaves” | https://arxiv.org/abs/1808.01513 | The cellular-sheaf view of assigning vector spaces to cells and using a Hodge/sheaf Laplacian as a structured energy operator. | No spectral theorem is assumed to solve chess; no eigen-decomposition is proposed for training. |
| Hansen and Gebhart, “Sheaf Neural Networks” | https://arxiv.org/abs/2012.06333 | The idea that graph diffusion can be generalized by replacing a trivial graph Laplacian with a sheaf Laplacian carrying non-constant relation maps. | Their benchmark tasks and generic graph setup are not copied; the graph here is a chess-rule tactical complex. |
| Bodnar et al., “Neural Sheaf Diffusion” | https://arxiv.org/abs/2202.04579 | Learnable sheaf restriction maps and the motivation that non-trivial sheaves can help when neighboring labels/features are heterophilic or oversmoothed by ordinary GNNs. | This proposal does not use citation-network heterophily benchmarks, dense learned graph construction, or label-dependent graph learning. |
| Barbero et al., “Sheaf Neural Networks with Connection Laplacians” | https://proceedings.mlr.press/v196/barbero22a.html | Stability intuition from constrained/connection-like maps, here simplified to diagonal plus low-rank relation-tied restrictions with norm penalties. | No manifold tangent-space PCA or precomputed connection from continuous data is copied. |
| Directed cellular sheaf work, including “Sheaves Reloaded: A Directional Awakening” | https://arxiv.org/abs/2506.02842 | Source/target asymmetry is useful because chess attacks are directional; source and target restriction maps should not be forced identical. | Citation is treated as recent and only lightly verified; the proposal does not depend on any unverified theorem from it. |
| Carroll and Beel, “Finite Group Equivariant Neural Networks for Games” | https://arxiv.org/abs/2009.05027 | Board-game symmetry should be exploited only when the game rules justify it. | This proposal does not impose full dihedral symmetry; chess pawn direction, castling, and side-to-move break most image symmetries. |
| Romero et al., “Learning Equivariances and Partial Equivariances from Data” | https://openreview.net/pdf?id=jFfRcKVut98 | The warning that full equivariance can be wrong and that partial equivariance can be preferable. | No partial group convolution is copied; `TSTN` uses a hand-auditable left-right relation tying and an optional mirror-consistency term. |
| Miłosz and Kapusta, “Predicting Chess Puzzle Difficulty with Transformers” | https://arxiv.org/abs/2410.11078 | Puzzle quality/difficulty is not identical to engine strength; human-facing puzzle structure may require specialized inductive bias. | This project predicts binary puzzle-likeness from a position, not Glicko difficulty from puzzle move sequences, and it deliberately avoids an ordinary Transformer backbone. |
| Schütt, Huber, and André, “Estimating Chess Puzzle Difficulty Without Past Game Records...” | https://www.computer.org/csdl/proceedings-article/bigdata/2024/10826087 | Puzzle-related neural models can be motivated by human problem-solving structure rather than pure engine evaluation. | Their difficulty target, architecture, and use of puzzle sequences are not copied. |

Unverified or lightly verified citations: the recent directed-sheaf reference is included as directional inspiration only; Codex does not need it to implement or test this idea.

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Plain simple CNN over `simple_18` | simple CNN | Already covered; likely learns local piece textures but has no explicit long-range ray, pin, skewer, or overloaded-defender structure. |
| Standard residual CNN | residual CNN | Residual depth improves optimization but does not change the tactical relation object; this would be routine architecture scaling. |
| Small/medium/deep CNN variants | small/medium/deep CNN variants | Depth/width sweeps are explicitly disallowed as the core research idea and would not isolate a new inductive bias. |
| LC0-style CNN or residual CNN clone | LC0 BT4-style CNN and residual variants | Strong baseline family, but still a convolutional value/policy-style representation; copying it would be an LC0 clone rather than a puzzle-likeness hypothesis. |
| Ordinary ViT over 64 square tokens | no exact baseline, but standard Transformer family | Full attention can connect all squares but treats tactical lines as learned from scratch and is disallowed as an ordinary square Transformer. |
| Plain GNN-on-squares with adjacency or attacks | no exact baseline; nearest graph message-passing baseline | A square GNN would pass messages on edges, but without edge stalks, restriction maps, or a sheaf energy it collapses to generic message passing and is too close to a common graph baseline. |
| Hyperparameter tuning of batch size, LR, dropout, or optimizer | all baselines | Useful for polish but not a research idea; it would also blur falsification because gains could come from optimization rather than structure. |
| Ensembling CNNs, transformers, and graph models | possible benchmark ensemble | Explicitly rejected; ensembling can improve leaderboard metrics while adding little scientific information about puzzle-likeness. |
| More data or self-training unresolved candidates | outside current split | Disallowed as core; it risks label fabrication and would not answer whether the current split contains a learnable tactical-energy signal. |
| Engine-evaluation distillation | engine-trained chess evaluators | Forbidden leakage: Stockfish scores, PVs, node counts, and engine best moves cannot be neural-network inputs or targets for this experiment. |
| Handcrafted material/tactical score features concatenated to a CNN | feature-engineered baseline | Piece identity and attack geometry are allowed, but hand-scored material or tactic heuristics would make the model less clean; `TSTN` learns relation maps and tensions instead. |

## 6. Mathematical Thesis

Input space:

- Let `X in R^{C x 8 x 8}` be one encoded board position from an allowed encoding.
- Let `b(X)` be the board state decoded from the current-position planes: occupied squares, piece type, piece color, and side to move.
- No engine-evaluated quantity is part of `X` or `b(X)`.

Target definition:

- The supervised target is `Y in {0,1}` where verified near-puzzles and verified puzzles are collapsed into `1`.
- The reporting distribution still distinguishes true fine labels `0`, `1`, and `2` after prediction.

Distribution assumptions:

- Many verified puzzles and near-puzzles are generated by sparse forcing structure: attacks on kings or high-value targets, pins, skewers, overloads, loose pieces, back-rank motifs, discovered attacks, or constrained replies.
- Many non-puzzles have either low tactical tension or diffuse tactical structure that does not concentrate around a forcing target.
- The assumption is statistical, not universal. Quiet zugzwang, fortress, long maneuver, and endgame-study puzzles may violate it.

Symmetry and equivariance assumptions:

- Chess is not invariant under arbitrary rotations or reflections. Pawns have direction, side-to-move matters, castling rights distinguish king side from queen side, and rank reflection changes the rules.
- The only built-in symmetry proposed here is conservative left-right file reflection, tying relation maps for mirrored directions such as east/west and north-east/north-west after correctly remapping castling-side planes. This is an optional consistency regularizer and relation-tying rule, not a full D4 invariant model.
- Relation IDs are side-aware: a white pawn attack and a black pawn attack are expressed in a side-relative frame, but absolute rank information remains available through square embeddings.

Core hypothesis:

- Puzzle-like positions are better separated by learned inconsistency patterns over an attack-defense sheaf than by raw square textures alone.
- A tactical motif appears as a local obstruction to simultaneously satisfying all learned attack, defense, x-ray, and target-square compatibility constraints. The obstruction is measured as sheaf tension.

Formal object:

For a decoded board `b`, construct a directed typed tactical complex `G_b = (V, E, R)`:

- `V = {0, ..., 63}` are board squares.
- `E` contains directed typed relations `(u -> v, r)` produced by deterministic current-board chess geometry:
  - pseudo-legal piece control of empty squares;
  - attacks on enemy-occupied squares;
  - defenses of own occupied squares;
  - king-ring controls;
  - sliding-piece x-ray edges through exactly one blocker to the next occupied square, with blocker color encoded in the relation type.
- `R` is a finite relation set keyed by source piece type, source color role relative to side-to-move, coarse edge kind, and direction octant. Relation IDs are tied under safe left-right mirror equivalences.

Define a cellular sheaf `F_theta` over this directed 1-complex:

- Every square stalk is `F(v) = R^r`, where `r = fiber_dim`.
- Every edge stalk is `F(e) = R^r`.
- For each relation type `r_e`, learn source and target restriction maps:
  - `rho_src[r_e]: R^r -> R^r`.
  - `rho_dst[r_e]: R^r -> R^r`.
- Use diagonal plus low-rank maps for stability and parameter economy:
  - `rho(z) = diag(a_r) z + U_r (V_r^T z)` with small rank, e.g. `rank = 4`.

For a square cochain `h in C^0(G_b; F_theta)`, define the directed sheaf coboundary on an edge `e = (u -> v, r_e)` as:

```text
(delta_theta h)_e = rho_src[r_e] h_u - rho_dst[r_e] h_v
```

The sheaf tension energy is:

```text
E_theta(h; G_b) = sum_{e in E} w_e ||(delta_theta h)_e||_2^2
```

with degree-normalized nonnegative weights `w_e`.

The corresponding sheaf Laplacian is:

```text
L_theta = delta_theta^* W delta_theta
```

where `W` is diagonal with entries `w_e`.

Proposition:

For any fixed board-derived tactical complex `G_b`, any real restriction maps `rho_src`, `rho_dst`, and any nonnegative edge weights, `L_theta = delta_theta^* W delta_theta` is positive semidefinite, and `E_theta(h; G_b) = <h, L_theta h> >= 0`.

Proof sketch:

- `delta_theta` is a linear map from square cochains to edge cochains once `G_b` and relation maps are fixed.
- `W` is positive semidefinite because its edge weights are nonnegative.
- Therefore `<h, delta_theta^* W delta_theta h> = <delta_theta h, W delta_theta h> = sum_e w_e ||(delta_theta h)_e||^2 >= 0`.
- If relation maps are tied under a valid file-mirror permutation and the input square encoder is mirror-equivariant under the same permutation, then `E_theta` is invariant to that mirror action because the edge sum is merely reindexed.

Objective:

Learn an encoder `phi_theta(X) = h_0`, several stable sheaf-tension diffusion blocks, and a classifier head `g_theta` such that:

```text
logits = g_theta(pool({h_l}, {E_l}, edge_energy_histograms))
```

minimizes supervised binary cross-entropy on the training split.

What is proven:

- The energy is nonnegative and has a valid sheaf-Laplacian form.
- The operator is sparse and finite-range on board-derived tactical edges.
- Conservative file-mirror tying is safe only when all mirrored input channels, including castling-side channels if present, are correctly permuted.

What is hypothesized:

- High, localized, relation-specific sheaf tension is predictive of near-puzzle and puzzle labels.
- X-ray edges are crucial because many forcing motifs are invisible to one-hop piece-contact graphs.
- A learned non-trivial sheaf beats a trivial attack graph because attack and defense relations are heterophilic: an attacker and a target should not necessarily have similar embeddings.

Counterexamples and limitations:

- Quiet endgame studies or zugzwang positions may be puzzle-like with low attack-defense tension.
- Some non-puzzles may contain sharp-looking tactical tension that is tactically unsound after calculation.
- Pinned-piece legality is only approximated by pseudo-legal geometry and x-ray edges; exact legal move generation is intentionally avoided unless implemented as deterministic rules without engine evaluation.
- Two positions can share nearly identical attack graphs but differ in castling rights, en-passant, side-to-move, or irreversible-move context.
- The model may overfit relation IDs if relation tying, dropout, and ablations are not enforced.

## 7. Architecture Specification

Proposed model class:

```text
TacticalSheafTensionNet(nn.Module)
```

Proposed helper modules:

```text
BoardTensorDecoder
TacticalComplexBuilder
SquareStalkEncoder
SheafTensionBlock
TacticalEnergyPool
TacticalSheafHead
```

Forward-pass contract:

```text
input:  x          shape (B, C, 8, 8)
output: logits     shape (B, 2)
```

High-level pseudocode only:

```text
def forward(x):
    board_state = decoder(x, encoding)
    graph = tactical_complex_builder(board_state)

    square_tokens = flatten_board_channels(x)              # (B, 64, C)
    h = square_stalk_encoder(square_tokens, board_state)   # (B, 64, r)

    all_energy_stats = []
    for block in sheaf_blocks:
        h, stats = block(h, graph)                         # h: (B, 64, r)
        all_energy_stats.append(stats)

    pooled = tactical_energy_pool(h, graph, all_energy_stats)
    logits = head(pooled)                                  # (B, 2)
    return logits
```

`BoardTensorDecoder`:

- Input: `(B, C, 8, 8)` and `encoding` string.
- Output:
  - `piece_type`: `(B, 64)` integer in `{empty, P, N, B, R, Q, K}`.
  - `piece_color`: `(B, 64)` integer in `{empty, white, black}`.
  - `side_to_move`: `(B,)` integer.
  - optional castling/en-passant tensors if reliably available.
- For `lc0_bt4_112`, decode the current-position slice for graph construction. Do not build tactical edges from historical planes.

`TacticalComplexBuilder`:

- Builds a ragged sparse graph per batch. Implementation may flatten batch and store global square node indices.
- Outputs:
  - `edge_src`: `(E_total,)` long, global node index.
  - `edge_dst`: `(E_total,)` long, global node index.
  - `edge_rel`: `(E_total,)` long relation ID.
  - `edge_group`: `(E_total,)` coarse group ID for pooling.
  - `edge_weight`: `(E_total,)` float, nonnegative degree-normalized weight.
  - `batch_index_for_edge`: `(E_total,)` long.
- Maximum edges per position should be below roughly `2048`; typical positions should be much smaller.
- Edge construction rules:
  - Pawns: side-specific diagonal control only, not forward moves.
  - Knights: all valid L-shaped attacked squares.
  - Kings: adjacent squares, also mark king-ring relations around both kings.
  - Bishops/rooks/queens: ray control through empty squares until the first blocker; include first occupied square as attack or defense; add one x-ray edge from slider through the first blocker to the next occupied square on the same ray.
  - Do not test checkmate, search replies, evaluate legality with an engine, or use PVs.

Relation ID design:

```text
source_role    in {side_to_move_piece, non_side_to_move_piece}
source_piece   in {P, N, B, R, Q, K}
edge_kind      in {control_empty, attack_enemy, defend_own, xray_one_blocker}
direction_bin  in {N, S, E, W, NE, NW, SE, SW} after mapping knight jumps to the sign/octant of displacement
```

Nominal relation count: `2 * 6 * 4 * 8 = 384`, with optional left-right tying that shares parameters between mirrored direction bins when safe.

`SquareStalkEncoder`:

- Input: square token `(B, 64, C)` plus optional learned square coordinates and side-to-move embedding.
- Recommended structure:
  - linear `C -> hidden_dim`.
  - GELU.
  - linear `hidden_dim -> fiber_dim`.
  - layer norm on `fiber_dim`.
- Suggested default: `hidden_dim = 64`, `fiber_dim = 24`.
- This is not a convolutional backbone; it is a per-square stalk initializer.

`SheafTensionBlock`:

Inputs:

- `h`: `(B, 64, r)`, flattened internally to `(B*64, r)`.
- sparse graph from `TacticalComplexBuilder`.

Per-edge computation:

```text
h_src = h_flat[edge_src]                       # (E, r)
h_dst = h_flat[edge_dst]                       # (E, r)
a = rho_src[edge_rel](h_src)                   # (E, r)
b = rho_dst[edge_rel](h_dst)                   # (E, r)
delta = a - b                                  # (E, r)
energy = edge_weight * squared_norm(delta)     # (E,)
```

Laplacian-style update:

```text
grad_src += edge_weight * rho_src[edge_rel]^T(delta)
grad_dst -= edge_weight * rho_dst[edge_rel]^T(delta)
h_next = LayerNorm(h - softplus(step) * normalized_grad + residual_mlp(h))
```

Restriction-map parameterization:

```text
rho_rel(z) = diag(a_rel) z + U_rel (V_rel^T z)
```

Defaults:

- `num_blocks = 3`.
- `fiber_dim = 24`.
- `restriction_rank = 4`.
- `drop_edge_p = 0.05` during training only.
- `map_norm_clip` or spectral/norm penalty enabled.

`TacticalEnergyPool`:

Pool the final node fibers and all block energy statistics into one vector:

- global mean of `h`: `(r,)` per sample.
- global max of `h`: `(r,)` per sample.
- side-to-move piece-square mean and non-side-to-move piece-square mean: `(2r,)`.
- edge energy mean/max/top-3 mean per block.
- coarse edge-group energy histograms: control, attack, defense, x-ray, king-ring if implemented.
- target-square energy accumulation around both kings.

Expected pooled dimension with defaults: approximately `3r + 2r + num_blocks * (10 to 24)`; Codex may compute it dynamically.

`TacticalSheafHead`:

- MLP: `pooled_dim -> 128 -> 2`.
- Activation: GELU.
- Dropout: `0.10` after first head layer only.

Parameter estimate:

- `SquareStalkEncoder`: about `C*64 + 64*24`; for `simple_18`, roughly `2.7k`; for `lc0_*_112`, roughly `8.7k`.
- Restriction maps: `384` relation types, two maps per relation, diagonal plus rank-4 on `r=24`: roughly `384 * 2 * (24 + 24*4*2) = 165,888` raw parameters before mirror tying; mirror tying reduces this.
- Three residual MLPs on `r=24`: roughly `10k` to `20k` total depending on hidden width.
- Pooling head: usually below `50k`.
- Total default target: about `0.20M` to `0.35M` parameters, much smaller than most deep CNN/LC0-style baselines.

FLOP/complexity estimate:

- Tactical graph construction: `O(B * 64 * ray_length)` fixed small chess-rule work.
- Sheaf blocks: `O(num_blocks * E_total * fiber_dim * restriction_rank)` for low-rank restrictions plus scatter-adds.
- With `E <= 2048` per board, `num_blocks = 3`, `fiber_dim = 24`, `rank = 4`, the edge operator should remain comfortably below a standard medium CNN forward pass.

Config fields:

```text
model.name: tactical_sheaf_tension_net
model.encoding: simple_18 | lc0_static_112 | lc0_bt4_112
model.hidden_dim: 64
model.fiber_dim: 24
model.num_blocks: 3
model.restriction_rank: 4
model.relation_count: 384
model.tie_file_mirror_relations: true
model.use_xray_edges: true
model.use_king_ring_edges: true
model.edge_dropout: 0.05
model.head_dropout: 0.10
model.num_classes: 2
```

Encoding support:

- Required first: `simple_18`.
- Required after sanity pass: `lc0_static_112`.
- Optional if existing loaders already support it cleanly: `lc0_bt4_112`, with current-board graph construction only.

Logits interface:

- Return raw logits `(B, 2)`.
- Do not apply softmax in the model.

## 8. Loss, Training, And Regularization

Primary loss:

- Weighted binary/multiclass cross-entropy over two logits.
- Use class weights computed from `split_train.parquet` binary label prevalence only.
- Suggested optional label smoothing: `0.02`; run without smoothing in the central ablation if smoothing muddies interpretation.

Optional auxiliary losses:

1. Restriction norm regularizer:

```text
lambda_map_norm * mean_r (||U_r||_F^2 + ||V_r||_F^2 + ||a_r||_2^2)
```

Suggested `lambda_map_norm = 1e-5`.

2. Energy concentration regularizer, weak and optional:

```text
lambda_entropy * entropy(normalized_edge_energy_per_sample)
```

Use only if validation shows energy is completely diffuse. Default is off because it may over-impose the hypothesis.

3. File-mirror consistency, optional:

```text
lambda_mirror * KL(p(x) || p(mirror_file(x)))
```

Default `lambda_mirror = 0.0` until Codex verifies exact channel remapping for the chosen encoding. This must never fabricate fine labels; it is a prediction-consistency term only.

Batch size:

- Start with the same batch size used by the nearest CNN baseline for the selected encoding.
- If memory is lower than CNNs, do not increase batch size for the first comparison; keep it fixed for fairness.

Optimizer and LR:

- Use the project’s standard optimizer and LR schedule for non-LC0 custom models.
- If there is no standard, use AdamW with `lr = 3e-4`, `weight_decay = 1e-4`, cosine decay, and warmup no longer than `5%` of total steps.
- Do not tune optimizer settings as the research variable.

Regularizers:

- Edge dropout: `0.05` on tactical edges during training.
- Head dropout: `0.10`.
- Weight decay: same as baseline protocol.
- Gradient clipping: `1.0` if the sheaf operator is unstable in the first epoch.
- Restriction maps should be initialized near identity for defense/control relations and near small random maps for attack/x-ray relations, but keep initialization deterministic by seed.

Determinism:

- Run at least three seeds: `0`, `1`, `2` or the project’s standard seed set.
- Fix train/val/test split exactly.
- Log model parameter count, edge counts per batch, and mean edge energy to detect graph-construction bugs.

What must stay fixed for fair comparison:

- Same benchmark split.
- Same binary target collapse.
- Same allowed encoding.
- Same number of epochs or early-stopping rule as the comparison baseline.
- Same metric script.
- No ensembling.
- No additional data.
- No engine-derived supervision.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| Trivial sheaf ablation | Replace learned restriction maps with identity/scalar maps on the same tactical graph. | Non-trivial relation-specific sheaf maps matter beyond attack graph connectivity. | If equal or better, the sheaf claim is false; future work may still use attack graphs, but not learned sheaf tension. |
| Smallest central falsifier | Keep graph, pooling, and parameter count similar, but feed only ordinary message-passing sums instead of coboundary residual energies. | The classifier needs incompatibility/tension, not just sparse graph aggregation. | If this matches `TSTN`, the central energy-obstruction thesis should be abandoned. |
| No x-ray edges | Remove sliding-piece x-ray edges through one blocker. | Pins, skewers, batteries, and discovered attacks are important puzzle-likeness signals. | If no drop, current data may be dominated by direct attacks or the x-ray builder is too noisy. |
| No attack/defense distinction | Merge `attack_enemy`, `defend_own`, and `control_empty` edge kinds. | Heterophilic attack and homophilic defense relations need different restrictions. | If no drop, relation typing is over-engineered. |
| No side-to-move role | Relation IDs do not distinguish side-to-move pieces from opponent pieces. | Puzzle-likeness is side-conditioned, not just static board tension. | If no drop, the dataset may contain side-symmetric artifacts or side-to-move is already encoded strongly elsewhere. |
| No file-mirror tying | Untie mirrored direction parameters. | Conservative partial equivariance improves sample efficiency. | If untying improves clearly, mirror tying is too strong or channel remapping is wrong. |
| Energy-pool removal | Use only final node-fiber mean/max pooling. | Edge tension statistics are the useful signal, not only node embeddings after sheaf diffusion. | If no drop, the Laplacian update acts like an ordinary graph feature extractor and the energy readout is unnecessary. |
| CNN-stalk hybrid check | Add a tiny 3x3 convolution before `SquareStalkEncoder`. | Local texture might be needed before sheaf reasoning. | If this is the only winning version, report that the sheaf idea requires local CNN preprocessing and is less clean. |
| Encoding transfer | Same default model on `lc0_static_112` after `simple_18`. | The idea is encoding-robust and not a `simple_18` artifact. | If it fails only on LC0 encodings, the decoder/current-board slice mapping is likely wrong or LC0 planes already saturate the signal. |

## 10. Benchmark And Falsification Criteria

Baselines:

- Best existing simple CNN for the same encoding.
- Best existing residual CNN for the same encoding.
- Best existing small/medium/deep CNN variant for the same encoding.
- Best existing LC0-style CNN/residual baseline for LC0 encodings.
- The ablation baselines in Section 9, especially the trivial-sheaf and no-energy versions.

Metrics:

- Primary: validation and test ROC-AUC for binary puzzle-like classification.
- Secondary: PR-AUC, macro-F1, balanced accuracy, accuracy, negative-class false-positive rate, calibration/ECE if existing scripts support it.
- Required reporting slice: true fine label `0/1/2 -> predicted binary output 0/1` confusion table.
- Also report recall on fine label `2` at fixed fine label `0` false-positive rates if the benchmark scripts permit threshold sweeps.

Artifacts to save:

- Config YAML.
- Model parameter count.
- Training curves.
- Validation metrics by epoch.
- Test metrics for the chosen checkpoint.
- Fine-label confusion report.
- Ablation table.
- Edge-count and energy-stat sanity logs.

Success threshold:

- Strong success: mean over three seeds improves test ROC-AUC by at least `+1.5` percentage points over the best non-ensemble same-encoding baseline, with no worse than `+0.5` percentage point increase in fine-label-0 false-positive rate at the selected threshold.
- Targeted success: even without full ROC-AUC gain, improves fine-label-2 recall by at least `+3.0` percentage points at a matched fine-label-0 false-positive rate, while fine-label-1 behavior remains interpretable.
- Scientific success: the full model beats both the trivial-sheaf and no-energy ablations by at least `+1.0` ROC-AUC point or a similarly meaningful fixed-FPR recall margin.

Failure threshold:

- Mean ROC-AUC is within `±0.3` percentage points of the best ordinary CNN baseline and full `TSTN` does not beat the trivial-sheaf/no-energy ablations.
- Fine-label confusion shows no meaningful improvement on fine labels `1` or `2`.
- Edge energy histograms are non-informative or collapse to relation-count artifacts.

Abandon condition:

- Abandon this exact idea if the smallest central falsifier matches or beats full `TSTN` on the same encoding and seeds.
- Also abandon if implementation requires fragile label metadata, engine calls, or unresolved-candidate assumptions.

Scaling condition:

- Scale only if the default model clears scientific success on `simple_18`.
- Next scale: `fiber_dim = 32`, `num_blocks = 4`, `restriction_rank = 4`, `lc0_static_112`.
- Do not scale by adding an ensemble or a large CNN backbone; that would change the research question.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_tactical_sheaf/README.md` | Create | Copy this handoff summary, the hypothesis, and the final benchmark results after running. |
| `ideas/20260421_tactical_sheaf/research_notes.md` | Create | Short implementation notes, graph-builder decisions, edge-count sanity checks, and deviations from this packet. |
| `src/chess_nn_playground/models/tactical_sheaf_tension_net.py` | Create | `TacticalSheafTensionNet`, helper modules, sparse sheaf block, and pseudolegal tactical-complex builder. |
| `src/chess_nn_playground/models/registry.py` | Modify | Register `tactical_sheaf_tension_net` using the same registry conventions as existing models. |
| `configs/tactical_sheaf_simple18.yaml` | Create | Minimal `simple_18` config with defaults from Sections 7 and 8. |
| `configs/tactical_sheaf_lc0_static112.yaml` | Create after simple sanity | LC0 static config only after the `simple_18` run validates graph construction. |
| `tests/test_tactical_complex_builder.py` | Create if test framework exists | Focused deterministic tests for pawn direction, knight attacks, slider blocker stop, one-blocker x-ray edge, side-to-move relation role, and mirror relation tying. |
| `tests/test_tactical_sheaf_shapes.py` | Create if test framework exists | Forward-pass shape tests for `(B,C,8,8) -> (B,2)`, no-label graph construction, CPU smoke test, and CUDA smoke test if CI supports it. |
| `ideas/20260421_tactical_sheaf/ablation_plan.yaml` | Create | Machine-readable ablation list from Section 9. |
| `ideas/20260421_tactical_sheaf/results.md` | Create after experiments | Metrics, seed table, confusion matrices, and decision: scale, revise, or abandon. |
| `ideas/chatgpt_pro_deep_math_research_prompt.md` | Modify after consuming this output | Preserve hard constraints while adding reusable lessons, anti-duplicate rules, clearer output requirements, and failure-mode guidance discovered during implementation and benchmarking. |

Codex must update `ideas/chatgpt_pro_deep_math_research_prompt.md` after consuming this output. It should preserve leakage rules, label rules, falsification requirements, and anti-duplicate requirements. Suggested additions are in Section 13.

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0255_tuesday_local_tactical_sheaf.md
  generated_at: '2026-04-21T02:55:34-07:00'
  weekday: tuesday
  timezone: local
  idea_slug: tactical_sheaf
  artifact_type: markdown
  intended_next_consumer: Codex
```

```yaml
idea_yaml:
  idea_id: '20260421_tactical_sheaf'
  idea_name: Tactical Sheaf Tension Network
  model_name: tactical_sheaf_tension_net
  thesis: >-
    Learn puzzle-likeness from sheaf-coboundary tension on a deterministic
    current-board tactical complex of attacks, defenses, and x-rays.
  fingerprint:
    - deterministic_pseudolegal_tactical_complex
    - directed_attack_defense_xray_edges
    - relation_tied_source_target_restriction_maps
    - sheaf_dirichlet_energy_pooling
    - conservative_file_mirror_partial_equivariance
  allowed_encodings:
    - simple_18
    - lc0_static_112
    - lc0_bt4_112
  forbidden_inputs:
    - stockfish_scores
    - principal_variations
    - node_counts
    - engine_best_moves
    - verification_metadata
    - source_labels_as_features
    - proposed_labels_as_features
    - split_membership
  primary_test: simple_18_three_seed_current_split
  central_falsifier: no_energy_message_passing_same_graph
  abandon_if: full_model_does_not_beat_trivial_sheaf_and_no_energy_ablations
```

```yaml
config_yaml:
  model:
    name: tactical_sheaf_tension_net
    encoding: simple_18
    num_classes: 2
    hidden_dim: 64
    fiber_dim: 24
    num_blocks: 3
    restriction_rank: 4
    relation_count: 384
    tie_file_mirror_relations: true
    use_xray_edges: true
    use_king_ring_edges: true
    edge_dropout: 0.05
    head_dropout: 0.10
    max_edges_per_position: 2048
  data:
    train_path: data/splits/crtk_sample_3class/split_train.parquet
    val_path: data/splits/crtk_sample_3class/split_val.parquet
    test_path: data/splits/crtk_sample_3class/split_test.parquet
    binary_target_rule: '0->0, 1->1, 2->1'
    use_fine_labels_as_inputs: false
  training:
    loss: weighted_cross_entropy
    class_weights: from_train_binary_prevalence
    optimizer: project_default_or_adamw
    lr: 0.0003
    weight_decay: 0.0001
    label_smoothing: 0.02
    gradient_clip_norm: 1.0
    seeds: [0, 1, 2]
    deterministic: true
  regularization:
    restriction_norm_lambda: 0.00001
    mirror_consistency_lambda: 0.0
    energy_entropy_lambda: 0.0
  evaluation:
    metrics:
      - roc_auc
      - pr_auc
      - macro_f1
      - balanced_accuracy
      - fine_label_confusion_0_1_2_to_binary
    success_threshold: '+1.5pp ROC-AUC over best same-encoding non-ensemble baseline or +3.0pp fine-label-2 recall at matched fine-label-0 FPR'
    failure_threshold: 'within +/-0.3pp ROC-AUC and no win over trivial-sheaf/no-energy ablations'
```

```yaml
model_spec:
  class_name: TacticalSheafTensionNet
  module_path: src/chess_nn_playground/models/tactical_sheaf_tension_net.py
  input_shape: [batch, C, 8, 8]
  output_shape: [batch, 2]
  helper_modules:
    - BoardTensorDecoder
    - TacticalComplexBuilder
    - SquareStalkEncoder
    - SheafTensionBlock
    - TacticalEnergyPool
    - TacticalSheafHead
  graph_builder:
    node_count: 64
    edge_types:
      - control_empty
      - attack_enemy
      - defend_own
      - xray_one_blocker
      - optional_king_ring
    relation_key_fields:
      - source_role_relative_to_side_to_move
      - source_piece_type
      - edge_kind
      - direction_bin
    engine_calls_allowed: false
    label_access_allowed: false
  sheaf_operator:
    square_stalk_dim: 24
    edge_stalk_dim: 24
    restriction_parameterization: diagonal_plus_low_rank
    low_rank: 4
    laplacian_form: delta_transpose_weight_delta
    energy_pooling: true
  expected_parameter_count: '0.20M-0.35M default, depending on encoding and mirror tying'
```

```yaml
research_continuity:
  idea_fingerprint: deterministic_current_board_attack_defense_xray_sheaf_laplacian_energy
  closest_duplicate_risk: >-
    Could be mistaken for a plain attack-graph GNN or generic neural sheaf diffusion;
    distinguish it by requiring directed chess-rule relation IDs, one-blocker x-ray edges,
    source/target restriction maps, and explicit edge-energy pooling.
  do_not_repeat_if_this_fails:
    - learned_sheaf_laplacian_on_pseudolegal_attack_graph_without_new_supervision
    - attack_defense_graph_message_passing_as_primary_chess_puzzle_signal
    - relation_typed_square_gnn_without_differentiable_search_or_causal_split_tests
  suggested_next_search_directions:
    - engine_free_differentiable_one_ply_counterfactual_move_masking
    - causal_invariance_across_opening_endgame_material_slices
    - optimal_transport_between_attacker_and_defender_mass_without_sheaf_maps
    - information_bottleneck_on_candidate_target_squares
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add “If proposing a graph model, specify why it is not merely a plain GNN-on-squares and include a central ablation that removes the non-GNN ingredient.” | Prevents future prompts from accepting generic graph message passing as deep math. | Hard Constraints or Research Goal. |
| Add “For any symmetry claim, list the exact chess-rule quantities preserved and broken, including pawns, side-to-move, castling, and en-passant when encoded.” | Avoids invalid D4 or rank-reflection equivariance assumptions. | Required Markdown File Content, Section 6 guidance. |
| Add “When an idea constructs deterministic features from the board, explicitly state whether the construction uses legal move generation, pseudo-legal geometry, or engine evaluation.” | Sharpens leakage boundaries and makes implementation auditable. | Problem Restatement And Data Contract. |
| Add “Every idea must include the smallest ablation that can falsify its central mathematical claim, not just remove optional components.” | Makes failures informative and discourages architecture bloat. | Ablation Plan and Benchmark/Falsification Criteria. |
| Add “For LC0 history encodings, distinguish current-board planes used for deterministic geometry from history planes used only by the neural encoder.” | Prevents accidental temporal or metadata leakage and clarifies graph construction. | Problem Restatement And Data Contract. |
| Add “Codex should record edge/object counts or operator diagnostics for nonstandard architectures.” | Catches silent bugs in constructed complexes, sheaves, hypergraphs, or transport operators. | Implementation Plan For Codex. |

Do not weaken leakage rules, label rules, falsification requirements, or anti-duplicate requirements.

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

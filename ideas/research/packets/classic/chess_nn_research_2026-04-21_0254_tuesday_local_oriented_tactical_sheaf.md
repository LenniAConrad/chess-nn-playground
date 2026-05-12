# Codex Handoff Packet: Oriented Tactical Sheaf Laplacian

## 1. File Metadata

- Filename: `chess_nn_research_2026-04-21_0254_tuesday_local_oriented_tactical_sheaf.md`
- Generated at: 2026-04-21 02:54:49 UTC-07:00
- Weekday: Tuesday
- Timezone: local
- Idea slug: `oriented_tactical_sheaf`
- Intended next consumer: Codex

## 2. Executive Selection

- Idea name: Oriented Tactical Sheaf Laplacian
- One-sentence thesis: Classify puzzle-likeness by learning whether the board-only attack/defense incidence structure has a high tactical gluing defect under a side-to-move-oriented cellular sheaf Laplacian.
- Idea fingerprint: `side_to_move_canonical_board -> typed_attack_defense_pin_incidence_complex -> learned_sheaf_restriction_maps -> sheaf_heat_diffusion + gluing_defect_energy_pool -> binary_logits`
- Why this is not a common CNN/ResNet/Transformer variant: The core computation is not local grid convolution, residual depth, or square self-attention; it constructs a chess-rule-derived typed incidence complex from the current board and applies a learned cellular-sheaf coboundary/Laplacian whose edges are attacks, defenses, rays, and tactical triples.
- Current-data minimal experiment: Train `OrientedTacticalSheafNet` on `simple_18` using the existing `data/splits/crtk_sample_3class/{split_train,split_val,split_test}.parquet` split, compare against the best already-logged single-model CNN/residual/LC0-style baseline under the same training budget, and report binary metrics plus fine-label-to-binary confusion.
- Expected information gain if it fails: A clean failure, especially if degree-preserving random relation masks match the real attack/defense masks, would show that puzzle-likeness in this split is not captured by static tactical incidence/gluing tension and that later cycles should pivot toward temporal/search-surrogate or causal-invariance ideas rather than richer board sheaves.

## 3. Problem Restatement And Data Contract

Task: binary chess puzzle-likeness classification from board positions.

Binary outputs:

- `0`: non-puzzle
- `1`: puzzle-like

Fine source labels:

- fine label `0`: known non-puzzle, mapped to binary `0`
- fine label `1`: verified near-puzzle, mapped to binary `1`
- fine label `2`: verified puzzle, mapped to binary `1`

The benchmark reports the cross-tabulation:

```text
true fine label 0/1/2 -> predicted binary output 0/1
```

Available encodings:

- `simple_18`
- `lc0_static_112`
- `lc0_bt4_112`

Benchmark split:

```text
data/splits/crtk_sample_3class/split_train.parquet
data/splits/crtk_sample_3class/split_val.parquet
data/splits/crtk_sample_3class/split_test.parquet
```

Model interface:

- PyTorch module: `nn.Module`
- Input tensor: `(batch, C, 8, 8)`
- Output logits: `(batch, num_classes)`, with `num_classes = 2`

Allowed neural-network inputs:

- Board-position tensor from one of the supported encodings.
- Deterministic board-only features extracted from that tensor, such as occupancy, side-to-move orientation, pseudo-legal attacks, visible ray attacks, defenses, king rays, and pin/skewer candidate geometry.
- Fixed chess-rule masks that do not depend on engine analysis or labels.

Forbidden neural-network inputs:

- Stockfish scores, principal variations, node counts, best moves, mate distances, tablebase results, or any engine-derived quantities.
- Verification metadata, source labels, proposed labels, unresolved-candidate status, or dataset provenance flags.
- Fabricated class `1` or class `2` labels.
- Treating unresolved candidates as non-puzzles or puzzles.

Leakage checklist for Codex:

- The model must receive only `(batch, C, 8, 8)` tensors at forward time.
- The incidence complex must be computed only from the current board tensor and fixed chess rules.
- No parquet label column, fine label, split name, puzzle ID, verification field, or engine artifact may be used in model features.
- Any optional adapter that extracts current piece planes must use encoding metadata or raw tensor channels only; it must not use labels.
- Any auxiliary loss must be label-compatible with the binary target or self-regularizing; it must not create pseudo fine labels.
- Validation/test splits must remain untouched except for evaluation.

## 4. Research Map

This idea borrows mathematical machinery but not architecture code from the following papers and sources.

1. Jakob Hansen and Thomas Gebhart, "Sheaf Neural Networks"  
   URL: https://arxiv.org/abs/2012.06333  
   Borrowed: sheaf Laplacian as a generalization of graph diffusion for asymmetric, signed, relation-dependent data.  
   Not copied: their benchmark graphs and generic sheaf constructions are not reused; here the base complex is a chess tactical incidence complex built from board rules.

2. Cristian Bodnar, Francesco Di Giovanni, Benjamin Paul Chamberlain, Pietro Liò, Michael M. Bronstein, "Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing in GNNs"  
   URL: https://arxiv.org/abs/2202.04579  
   Borrowed: learned sheaf maps and the view that nontrivial sheaves can control diffusion better than ordinary graph Laplacians in heterophilic relational settings.  
   Not copied: no citation-network task, no generic GNN benchmark protocol, and no assumption that chess square labels are homophilic.

3. Jakob Hansen and Robert Ghrist, "Toward a Spectral Theory of Cellular Sheaves"  
   URL: https://arxiv.org/abs/1808.01513  
   Borrowed: spectral sheaf Laplacian intuition, especially that gluing inconsistency can be represented by coboundary energy.  
   Not copied: no topological theorem is claimed as a puzzle detector; the puzzle-likeness connection is explicitly a hypothesis.

4. Lek-Heng Lim, "Hodge Laplacians on Graphs"  
   URL: https://arxiv.org/abs/1507.05379  
   Borrowed: the linear-algebraic view that Laplacians arise from coboundary operators, making positive-semidefinite energy proofs simple.  
   Not copied: the proposed model uses cellular-sheaf restrictions and chess-specific typed relations, not a generic graph Hodge model.

5. David W. Romero and collaborators, "Learning Partial Equivariances from Data"  
   URL: https://arxiv.org/abs/2110.10211  
   Borrowed: the warning that full equivariance can be harmful when the domain only has partial or conditional symmetries.  
   Not copied: no partial group convolution layer is proposed. Chess is handled by a narrow side-to-move canonicalization, not by full rotation/reflection invariance.

6. Thomas S. Cohen and Max Welling, "Group Equivariant Convolutional Networks"  
   URL: https://arxiv.org/abs/1602.07576  
   Borrowed: the general idea of exploiting true symmetries to reduce sample complexity.  
   Not copied: this proposal rejects full board `D4` equivariance because pawns, castling, and side-to-move break many geometric symmetries.

7. Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka, "How Powerful are Graph Neural Networks?"  
   URL: https://arxiv.org/abs/1810.00826  
   Borrowed: a reason to avoid a plain message-passing GNN-on-squares as the central novelty.  
   Not copied: no WL-power claim is made for the proposed model; the sheaf operator is chosen for chess relation geometry, not generic graph isomorphism power.

8. Leela Chess Zero developer documentation, "Neural network topology" and backend input references  
   URLs: https://lczero.org/dev/backend/nn/ and https://lczero.org/dev/backend/interface/  
   Borrowed: awareness that LC0-family encodings use `112` input planes of shape `8x8`.  
   Not copied: no LC0 residual tower, policy/value head, MCTS data, or engine training target is used.

Unverifiable citations: none intentionally included. Codex should keep URLs but may refresh bibliographic metadata if the repository has a citation style.

## 5. Common Approaches Rejected

| Approach | Closest existing baseline | Why rejected |
|---|---|---|
| Simple CNN on `simple_18` | simple CNN | Already covered; local kernels may learn motifs but do not explicitly model long-range line-of-sight, pins, overloads, or attack-defense tension. |
| Standard residual CNN | residual CNN | Depth and skip connections are routine and likely duplicate existing baselines rather than testing a new inductive bias. |
| Small/medium/deep CNN variants | small/medium/deep CNN variants | Scaling width/depth is ordinary capacity tuning, explicitly disallowed as the core research idea. |
| LC0-style CNN or LC0-style residual tower | LC0 BT4-style CNN and residual variants | Too close to existing LC0-inspired baselines and risks becoming a clone of a chess-engine architecture rather than a puzzle-likeness classifier idea. |
| Ordinary ViT over 64 square tokens | no direct baseline, closest is generic square Transformer | Self-attention can learn interactions but has no built-in notion of legal attacks, ray blockers, defenders, or side-to-move asymmetry; it is also explicitly disallowed as an ordinary square Transformer. |
| Plain GNN-on-squares with legal-move edges | no direct baseline, closest is graph model alternative | Too generic; pairwise message passing alone does not represent relation-specific restriction maps or gluing defects, and it risks reducing to a handcrafted adjacency GCN. |
| Full `D4` group-equivariant CNN | no direct baseline, closest is CNN with symmetry augmentation | Chess is not fully rotation/reflection invariant: pawns have direction, castling rights are side-specific, and side-to-move matters. |
| Hyperparameter tuning of current baselines | all existing baselines | Does not create a new scientific hypothesis and is explicitly ruled out. |
| Ensembling multiple current models | any baseline ensemble | May improve score but gives low information gain and is explicitly ruled out. |
| More data, pseudo-labeling, or treating unresolved candidates as negatives | dataset expansion/pseudo-label workflow | Violates the current research constraint against fabricating class `1`/`2` labels or resolving unresolved candidates by assumption. |
| Engine-feature distillation from Stockfish/LC0 | no allowed baseline | Forbidden leakage: scores, PVs, node counts, mate depths, and verification metadata cannot be model inputs. |
| Vanilla spectral graph convolution on an 8x8 grid | simple graph spectral model | Grid spectra ignore chess movement geometry; a bishop ray and a rook ray should not be the same as adjacent pixels. |

## 6. Mathematical Thesis

### Input space

Let `E` be one of the encodings `simple_18`, `lc0_static_112`, or `lc0_bt4_112`. The raw input space is

```text
X_E subset R^{C_E x 8 x 8}.
```

Each `x in X_E` represents a legal or dataset-accepted chess position. The model may deterministically derive from `x` a current board occupancy tensor, side-to-move indicator, and piece-type/color occupancy estimates. When an encoding schema exposes current piece planes, extraction should be exact. If not, the fallback adapter may learn soft piece-state probes from raw channels, but it still receives no labels beyond the binary training target.

### Target definition

The binary target is

```text
y = 0 if fine_label = 0
y = 1 if fine_label in {1, 2}.
```

The model must never receive the fine label as input.

### Distribution assumptions

The fixed train/validation/test parquet split is treated as the experimental distribution. The research hypothesis does not require the split to be perfectly iid, but evaluation must use the prescribed split and must report fine-label-to-binary behavior to detect whether the method merely separates verified puzzles from non-puzzles while missing near-puzzles.

### Symmetry and equivariance assumptions

Chess has only conditional board symmetries for this task.

Accepted symmetry prior:

```text
color swap + 180-degree board rotation + side-to-move canonicalization.
```

This maps the mover's perspective to a common orientation: "us" to move, "them" defending, mover pawns advancing in the canonical forward direction.

Rejected symmetry prior:

```text
full rotation/reflection D4 equivariance.
```

Reason: pawns, castling, en-passant context when present, and side-to-move break arbitrary board rotations/reflections. A left-right file mirror may be rule-preserving if castling-side metadata is mirrored correctly, but it is not part of the minimal experiment.

### Core hypothesis

Puzzle-like positions are often characterized by local tactical constraints that cannot be made globally consistent: a piece is attacked but apparently defended; a defender is pinned; a king ray makes a normal defense invalid; an overloaded piece is required to satisfy multiple relations; a forcing move changes several attack/defense constraints at once. These are not just local patterns on the grid. They are gluing defects in a typed tactical incidence structure.

Hypothesis:

```text
A learned sheaf energy over attack/defense/pin incidence cells provides a more sample-efficient statistic for puzzle-likeness than ordinary spatial convolution or plain pairwise message passing.
```

### Formal object

For each board `x`, construct a finite cell complex `K(x)`.

0-cells:

```text
V = {64 board squares}
```

Each square has a stalk vector space:

```text
F(v) = R^s
```

with default `s = 8`.

1-cells are typed tactical relations:

```text
e = (u, v, r)
```

where `u` and `v` are squares and `r` is one of:

- `us_attacks_them_piece`
- `them_attacks_us_piece`
- `us_defends_us_piece`
- `them_defends_them_piece`
- `us_attacks_empty_near_king`
- `them_attacks_empty_near_king`
- `bishop_ray_visible`
- `rook_ray_visible`
- `queen_ray_visible`
- `knight_attack`
- `pawn_attack_forward_oriented`
- `king_adjacency`
- `king_ray_pin_candidate`

The exact list may be collapsed to 10-12 relation types for implementation. Edges are weighted by deterministic board-only masks:

```text
w_e(x) in [0, 1].
```

For exact piece extraction these weights are binary except for relation gates; for soft extraction they are differentiable probabilities.

Optional 2-cells are tactical triples:

```text
q = (attacker, target, defender, r_q)
```

when the attacker attacks a target and a same-color defender also defends that target, or when a ray piece, king, and blocker form a pin/skewer candidate. These 2-cells are not required for the smallest falsification experiment, but they are part of the full proposed model.

A learned cellular sheaf assigns restriction maps to each typed edge:

```text
rho_{e,u}^{(r)}: F(u) -> F(e)
rho_{e,v}^{(r)}: F(v) -> F(e)
```

with `F(e) = R^s`. Define the coboundary:

```text
(delta_rho h)_e = sqrt(w_e) * (rho_{e,v}^{(r)} h_v - sigma_r rho_{e,u}^{(r)} h_u)
```

where `sigma_r in {-1, +1}` is fixed by relation type. For attack relations, the sign separates attacker context from target context. For defense relations, the sign can be shared or learned through a bounded scalar gate.

The sheaf Laplacian is:

```text
L_rho(x) = delta_rho(x)^T delta_rho(x).
```

The gluing-defect energy is:

```text
E_rho(h; x) = ||delta_rho(x) h||_2^2 = h^T L_rho(x) h.
```

The model learns node states `h_v`, restriction maps, relation gates, and a classifier over pooled node states plus per-relation sheaf energies.

### Proposition

For any fixed board `x`, nonnegative edge weights `w_e(x)`, and real restriction maps `rho`, the operator

```text
L_rho(x) = delta_rho(x)^T delta_rho(x)
```

is symmetric positive semidefinite. Therefore the linear heat step

```text
h_{t+1} = (I - eta L_rho(x)) h_t
```

is stable in the Euclidean norm for `0 <= eta <= 2 / lambda_max(L_rho(x))`, and the sheaf energy is non-increasing under the same spectral step-size bound.

### Proof sketch

For any vector `h`,

```text
h^T L_rho h = h^T delta_rho^T delta_rho h = ||delta_rho h||_2^2 >= 0.
```

Thus `L_rho` is positive semidefinite and has nonnegative eigenvalues. In an eigenbasis, the heat step multiplies the component with eigenvalue `lambda` by `1 - eta lambda`. If `0 <= eta <= 2 / lambda_max`, then `|1 - eta lambda| <= 1` for every eigenvalue, so the step cannot amplify that component in Euclidean norm. The same eigenvalue-wise argument gives non-increase of `h^T L_rho h`.

### What is proven

- The board-conditioned sheaf Laplacian is a mathematically valid positive-semidefinite energy operator.
- The corresponding linear diffusion is stable under a simple spectral step-size bound.
- The construction respects the limited side-to-move canonical symmetry and does not require full board rotation/reflection invariance.

### What is hypothesized

- High or structured learned gluing-defect energy correlates with puzzle-likeness.
- Relation-specific sheaf maps help distinguish true tactical tension from harmless attacks/defenses.
- Side-to-move canonicalization improves sample efficiency without imposing false symmetries.
- Optional attack-target-defender 2-cells help class `1` near-puzzles, where tactics are present but may be less clean than verified puzzles.

### Counterexamples and failure modes

- Quiet zugzwang, opposition, fortress, or endgame-study puzzles may be puzzle-like with weak immediate attack/defense incidence.
- A non-puzzle blunder-rich position may have high tactical tension but no verified puzzle motif.
- Long forcing sequences requiring several quiet moves may not be visible from a static one-ply attack complex.
- Encodings whose current piece planes are difficult to extract may make the soft adapter learn slowly.
- Dataset artifacts may dominate the label signal; in that case a CNN could match or beat the sheaf model without learning chess tactics.
- Positions with unusual castling/en-passant context may not be fully represented if the minimal adapter ignores those planes.

## 7. Architecture Specification

### Proposed module name

```text
OrientedTacticalSheafNet
```

Suggested file:

```text
src/chess_nn_playground/models/oriented_tactical_sheaf.py
```

### Submodules

1. `BoardStateAdapter`
   - Input: `x` with shape `(B, C, 8, 8)`.
   - Output:
     - `square_raw`: `(B, 64, d_in)`
     - `piece_state`: `(B, 64, 13)` representing empty plus 12 mover-oriented piece states, exact or soft.
     - `occupancy`: `(B, 64)`
     - `side_info`: `(B, k_side)`
   - Preferred mode: exact current-piece extraction from encoding metadata.
   - Fallback mode: learned `1x1` projection from channels to soft piece-state probabilities.
   - The adapter may canonicalize black-to-move positions by 180-degree rotation and color swap if the encoding is not already mover-oriented.

2. `TacticalIncidenceBuilder`
   - Input:
     - `piece_state`: `(B, 64, 13)`
     - `occupancy`: `(B, 64)`
   - Output:
     - `edge_index`: dense or sparse representation of `(source_square, target_square, relation_type)`
     - `edge_weight`: `(B, E)` or `(B, R, 64, 64)`
     - optional `triad_index` and `triad_weight`
   - Uses precomputed masks for knight, king, pawn, rook ray, bishop ray, queen ray, and between-square blockers.
   - Visible ray relation for squares `i -> j` is active only if the piece type can move along the ray and all squares strictly between `i` and `j` are unoccupied.
   - Pin candidate relation is active for `(king, blocker, slider)` aligned on a rook/bishop/queen ray with the blocker between king and slider and no other blockers between them.

3. `SquareTokenEncoder`
   - Input:
     - `square_raw`: `(B, 64, d_in)`
     - `piece_state`: `(B, 64, 13)`
     - fixed coordinate features: `(64, 6)` such as rank, file, centered rank/file, edge distance, promotion-distance for mover, and king-zone indicator when available.
   - Output:
     - `h0`: `(B, 64, d_model)`
   - Default `d_model = 64`.

4. `SheafDiffusionBlock`
   - Inputs:
     - `h`: `(B, 64, d_model)`
     - relation masks: `(B, R, 64, 64)`
   - Internal projections:
     - `node_to_stalk`: `R^{d_model} -> R^s`, default `s = 8`
     - per-relation restriction maps `rho_src[r]`, `rho_dst[r]`: `(s, s)`
     - per-relation bounded scalar gate `g_r in (0, 2)`
   - Computes relation-wise coboundary defects:
     - `defect[b,r,i,j] = sqrt(w[b,r,i,j]) * (rho_dst[r] z[b,j] - sign[r] * rho_src[r] z[b,i])`
   - Applies a bounded sheaf heat update:
     - `z_update = -eta * delta^T defect`
     - `h = LayerNorm(h + stalk_to_node(z_update) + relation_mlp(h))`
   - Returns updated `h` and per-relation energies.

5. `TriadDefectPool` optional but enabled in the full model
   - Inputs:
     - `h`: `(B, 64, d_model)`
     - top `K` tactical triples by deterministic weight, default `K = 128`
   - Computes:
     - `||A h_attacker + B h_target + C h_defender||^2`
   - Output:
     - pooled triad statistics `(B, n_triad_stats)`.

6. `TacticalReadout`
   - Pools:
     - mean and max node embeddings
     - mover-piece pooled embeddings
     - opponent-piece pooled embeddings
     - per-relation sheaf energy mean/max
     - optional triad energy histogram or quantiles
   - Output:
     - logits `(B, 2)`

### Default tensor shapes

With `B` batch size, `C` input planes, `R = 12`, `d_model = 64`, `s = 8`, `L = 4`:

```text
x:                         (B, C, 8, 8)
piece_state:               (B, 64, 13)
relation_masks:            (B, 12, 64, 64)
h0:                        (B, 64, 64)
stalk_state z:             (B, 64, 8)
edge_defect dense form:    (B, 12, 64, 64, 8)
relation_energy:           (B, 12)
pooled_readout:            approx (B, 256 to 384)
logits:                    (B, 2)
```

Dense edge-defect tensors are acceptable for 64 squares. Codex may implement sparse edge lists later, but the minimal version should prioritize clarity and tests.

### Parameter estimate

Approximate default parameter count:

- Raw channel/token adapter: `C * 64 + 64`, about `1.2k` for `simple_18`, `7.2k` for `112`-plane encodings.
- Piece-state soft probe if needed: `C * 13 + 13`, about `247` for `simple_18`, `1.5k` for `112`.
- Square token encoder MLP: about `10k`.
- Four sheaf blocks:
  - restriction maps: `L * R * 2 * s * s = 4 * 12 * 2 * 8 * 8 = 6144`
  - stalk/node projections and MLPs: about `80k` to `160k`, depending on hidden expansion.
- Readout MLP: about `30k` to `60k`.

Expected total: `130k` to `260k` parameters, intentionally comparable to small baselines and far below a large residual tower.

### FLOP and complexity estimate

Dense relation implementation:

```text
O(B * L * R * 64^2 * s^2)
```

for restriction-map defect computation, plus

```text
O(B * L * 64 * d_model^2)
```

for token MLPs.

With `L=4`, `R=12`, `s=8`, this is roughly `12M` to `16M` multiply-adds per sample before MLP overhead if implemented naively dense. Because the board has only 64 squares, this is acceptable for a minimal experiment. A sparse implementation can reduce cost later.

### Pseudocode

```text
forward(x):
    board = BoardStateAdapter(x)
    masks = TacticalIncidenceBuilder(board.piece_state, board.occupancy)

    h = SquareTokenEncoder(board.square_raw, board.piece_state, coordinates)

    all_energy = []
    for block in sheaf_blocks:
        h, energy_r = block(h, masks.edge_weight)
        all_energy.append(energy_r)

    triad_stats = TriadDefectPool(h, masks.triads) if use_triads else empty

    readout = concat(
        mean_pool(h),
        max_pool(h),
        mover_piece_pool(h, board.piece_state),
        opponent_piece_pool(h, board.piece_state),
        per_relation_energy_stats(all_energy),
        triad_stats
    )

    logits = TacticalReadout(readout)
    return logits
```

### Encoding support

- `simple_18`: run first. Use exact current piece planes if existing repo metadata exposes them; otherwise use soft adapter fallback. Side-to-move canonicalization should be enabled if the encoding is absolute-color rather than mover-relative.
- `lc0_static_112`: supported after the `simple_18` smoke test. Prefer metadata-based extraction of current planes. LC0-family `112`-plane encodings may already be mover-relative; Codex should verify in the repository encoding implementation rather than assume.
- `lc0_bt4_112`: supported using the same adapter interface. The model should initially use only the current-board slice for incidence construction, while allowing the token adapter to read all channels. This prevents accidental temporal leakage while still making the architecture compatible with the tensor.

### Logits interface

The model must return raw logits:

```text
logits.shape == (batch, num_classes)
```

No sigmoid or softmax inside `forward`.

## 8. Loss, Training, And Regularization

Primary loss:

```text
torch.nn.CrossEntropyLoss(weight=train_binary_class_weights)
```

where class weights are computed only from the training split. If the existing benchmark uses unweighted CE for all baselines, run both unweighted and weighted only if that is already part of the fair benchmark protocol; do not create a hyperparameter search.

Optional auxiliary regularizers:

1. Sheaf map norm regularizer:
   ```text
   lambda_rho * sum_r (||rho_src[r]||_F^2 + ||rho_dst[r]||_F^2)
   ```
   Default `lambda_rho = 1e-5`.

2. Relation gate entropy or boundedness regularizer:
   ```text
   lambda_gate * sum_r (gate_r - 1)^2
   ```
   Default `lambda_gate = 1e-4`.

3. Energy budget regularizer:
   ```text
   lambda_energy * mean(log1p(E_rho))
   ```
   Default `lambda_energy = 0` for the first run. Enable only if training diverges.

No auxiliary labels, no pseudo-labels, and no engine targets.

Training defaults for the minimal experiment:

- Batch size: `256` if memory allows; otherwise `128`.
- Optimizer: `AdamW`.
- Learning rate: `3e-4`.
- Weight decay: `1e-4`.
- Epoch budget: same as the strongest existing baseline budget in the repo. If no standard exists, use `50` epochs with validation early stopping patience `8`.
- Scheduler: keep the repository default. If none exists, use cosine decay without restarts.
- Dropout: `0.10` in readout MLP and token MLP only.
- Gradient clipping: global norm `1.0`.
- Determinism: run seeds `0, 1, 2`; set PyTorch, NumPy, and data-loader seeds as existing benchmark permits.
- Mixed precision: allowed only if existing baselines use it or if all compared models are rerun under the same precision.

What must stay fixed for fair comparison:

- Same train/val/test split.
- Same encoding for a given comparison.
- Same binary label mapping.
- Same maximum epochs or wall-clock budget policy.
- Same early-stopping metric.
- Same data augmentation policy. Do not add symmetry augmentation unless all baselines are rerun with the identical legal symmetry policy.
- Same reporting scripts, especially fine-label-to-binary confusion.

## 9. Ablation Plan

| Ablation | What it removes or changes | Hypothesis tested | Failure interpretation |
|---|---|---|---|
| Degree-preserving random relation masks | Replace real tactical masks with random masks matching approximate edge counts per relation and board | The actual chess attack/defense geometry matters, not just extra dense mixing | If performance is unchanged, the central tactical-incidence claim is falsified. |
| Scalar graph Laplacian | Replace learned sheaf restriction maps with scalar normalized adjacency/Laplacian over the same edges | Nontrivial sheaf maps add value beyond a legal-move GNN | If scalar performs the same, future work should use simpler graph operators or abandon sheaf complexity. |
| No side-to-move canonicalization | Use absolute board orientation and color channels without mover perspective normalization | The limited chess symmetry prior improves sample efficiency | If unchanged, canonicalization is not important for this split or encodings are already normalized. |
| No pin/king-ray relation | Remove pin/skewer candidate edges while keeping ordinary attacks/defenses | King-ray geometry contributes to puzzle-likeness | If unchanged, static king-ray motifs may be rare or learned by other relations. |
| No triad defect pool | Remove attacker-target-defender 2-cell statistics | Tactical triples help beyond pairwise sheaf diffusion | If unchanged, keep the simpler 1-cell-only model. |
| Identity restrictions | Force `rho_src = rho_dst = I` for all relation types | Learned relation-specific maps are needed | If unchanged, relation-type masks alone are doing the work. |
| Relation labels collapsed | Keep all edges but use one shared relation type | Attack, defense, ray, pawn, knight, and king relations need different semantics | If unchanged, the typed-relation thesis is weak. |
| CNN-token readout only | Use `BoardStateAdapter` and `SquareTokenEncoder`, then pool without sheaf blocks | Improvements are not merely from the adapter/readout | If this matches full model, the sheaf operator should be abandoned. |
| Current-board-only LC0 mode | For `lc0_bt4_112`, restrict token adapter to current planes only | Historical BT4 planes are not required for the sheaf claim | If full BT4 helps but current-only does not, improvement may come from history channels rather than the proposed tactical operator. |

Smallest central falsifier: degree-preserving random relation masks. If random masks match real masks across three seeds within `0.3` percentage points macro-F1, the attack/defense incidence thesis should be considered failed.

## 10. Benchmark And Falsification Criteria

Baselines:

- Best existing simple CNN on the same encoding.
- Best existing residual CNN on the same encoding.
- Best existing small/medium/deep CNN variant on the same encoding.
- Best existing LC0 BT4-style CNN/residual variant for `lc0_bt4_112`.
- Any already-registered non-ensemble single model in the repository.

Primary metrics:

- Test macro-F1 for binary classification.
- Test AUROC if probabilities are already reported.
- Test accuracy.
- Fine-label-to-binary confusion:
  ```text
  fine 0 -> predicted 0/1
  fine 1 -> predicted 0/1
  fine 2 -> predicted 0/1
  ```
- Fine label `1` recall as puzzle-like: `P(pred=1 | fine=1)`.
- Fine label `0` false positive rate: `P(pred=1 | fine=0)`.

Artifacts to save:

- Model config YAML.
- Training logs for each seed.
- Validation and test metrics JSON.
- Fine-label confusion table.
- Ablation metrics table.
- Calibration curve or reliability data if the repo already has such artifact support.
- A small debug dump of relation density statistics, not containing labels or engine data.

Success threshold:

- On `simple_18`, mean over three seeds improves test macro-F1 by at least `+2.0` percentage points over the best comparable single-model baseline, or improves AUROC by at least `+1.5` percentage points, while reducing neither fine label `0` specificity nor fine label `2` recall by more than `1.0` percentage point.
- Secondary success: at matched fine label `0` false positive rate, fine label `1` recall improves by at least `+5.0` percentage points. This is valuable because near-puzzles are the most informative class for puzzle-likeness.

Failure threshold:

- Mean test macro-F1 is within `±0.5` percentage points of the best baseline, and none of the central ablations changes performance by more than `0.5` percentage points.
- Degree-preserving random masks are statistically indistinguishable from real masks across seeds.
- Training is unstable under the default bounded-step configuration even after gradient clipping, with no validation improvement over the adapter-only ablation.

Abandon condition:

- If the full model does not beat the adapter-only ablation and real relation masks do not beat randomized masks, do not repeat this family with more layers, bigger stalks, extra relation types, or a larger readout. That would become hyperparameter tuning, not new research.

Scaling condition:

- Scale to `lc0_static_112` and `lc0_bt4_112` only after `simple_18` shows either primary success or the secondary fine-label-`1` success.
- If `simple_18` succeeds but LC0 encodings fail, inspect adapter extraction and relation-density logs before rejecting the math idea.

## 11. Implementation Plan For Codex

| Path | Action | Contents |
|---|---|---|
| `ideas/20260421_oriented_tactical_sheaf/README.md` | Create | Human-readable summary of this handoff, experiment intent, leakage rules, and benchmark command examples. |
| `ideas/20260421_oriented_tactical_sheaf/handoff.md` | Create | Copy this Markdown packet verbatim or link to the downloaded artifact for traceability. |
| `src/chess_nn_playground/models/oriented_tactical_sheaf.py` | Create | `BoardStateAdapter`, `TacticalIncidenceBuilder`, `SquareTokenEncoder`, `SheafDiffusionBlock`, `TriadDefectPool`, `TacticalReadout`, and `OrientedTacticalSheafNet`. |
| `src/chess_nn_playground/models/registry.py` | Update | Register model name `oriented_tactical_sheaf` or `otsl`. Preserve existing model registrations. |
| `configs/oriented_tactical_sheaf_simple18.yaml` | Create | Minimal `simple_18` config with `d_model=64`, `stalk_dim=8`, `num_sheaf_layers=4`, `use_triads=true`, and leakage-safe adapter settings. |
| `configs/oriented_tactical_sheaf_lc0_static_112.yaml` | Create after smoke test | LC0 static config using the same model, with encoding-specific adapter metadata. |
| `configs/oriented_tactical_sheaf_lc0_bt4_112.yaml` | Create after smoke test | BT4 config; incidence builder should use current board only while token adapter can read all planes if fair baselines do. |
| `tests/test_oriented_tactical_sheaf.py` | Create | Shape tests, deterministic mask tests, PSD/sheaf-energy nonnegativity test, no-label-input smoke test, and forward/backward pass for `C=18` and `C=112`. |
| `tests/test_tactical_incidence_builder.py` | Create if test structure allows | Unit tests for knight, pawn, rook/bishop blocker visibility, and pin-candidate masks on synthetic board tensors. |
| `ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` | Update | After consuming this output, preserve hard constraints and add reusable lessons: reject repeating sheaf-only variants if randomized relation masks fail; require central falsifier ablation; require adapter leakage checks; clarify that full `D4` equivariance is unsafe for chess unless pawn/castling/side-to-move transformations are explicitly handled. |

Codex must update `ideas/research/prompts/chatgpt_pro_deep_math_research_prompt.md` after consuming this output. The update should preserve all leakage rules, label rules, falsification requirements, and anti-duplicate requirements while adding reusable lessons, anti-duplicate rules, clearer output requirements, or failure-mode guidance discovered from this research pass.

## 12. Machine-Readable Blocks

```yaml
download_artifact:
  filename: chess_nn_research_2026-04-21_0254_tuesday_local_oriented_tactical_sheaf.md
  artifact_type: markdown
  generated_at: "2026-04-21T02:54:49-07:00"
  weekday: Tuesday
  timezone_token: local
  idea_slug: oriented_tactical_sheaf
  intended_next_consumer: Codex
```

```yaml
idea_yaml:
  idea_id: "20260421_oriented_tactical_sheaf"
  idea_name: "Oriented Tactical Sheaf Laplacian"
  model_name: "oriented_tactical_sheaf"
  core_thesis: "Puzzle-likeness is detectable as learned gluing-defect energy on a side-to-move-oriented chess attack/defense incidence sheaf."
  input_encodings:
    - simple_18
    - lc0_static_112
    - lc0_bt4_112
  primary_minimal_encoding: simple_18
  binary_label_mapping:
    fine_0: 0
    fine_1: 1
    fine_2: 1
  forbidden_inputs:
    - stockfish_scores
    - principal_variations
    - node_counts
    - mate_distances
    - verification_metadata
    - source_labels_as_features
    - proposed_labels_as_features
    - unresolved_candidate_assumptions
  central_falsifier: "degree_preserving_random_relation_masks"
  success_threshold:
    macro_f1_pp_over_best_baseline: 2.0
    auroc_pp_over_best_baseline: 1.5
    fine1_recall_pp_at_matched_fine0_fpr: 5.0
  abandon_if:
    - "adapter_only_ablation_matches_full_model"
    - "random_relation_masks_match_real_masks"
    - "no_central_ablation_changes_metric_by_more_than_0_5pp"
```

```yaml
config_yaml:
  model:
    name: oriented_tactical_sheaf
    num_classes: 2
    input_shape: [null, null, 8, 8]
    d_model: 64
    stalk_dim: 8
    num_sheaf_layers: 4
    relation_types:
      - us_attacks_them_piece
      - them_attacks_us_piece
      - us_defends_us_piece
      - them_defends_them_piece
      - us_attacks_empty_near_king
      - them_attacks_empty_near_king
      - bishop_ray_visible
      - rook_ray_visible
      - queen_ray_visible
      - knight_attack
      - pawn_attack_forward_oriented
      - king_adjacency
      - king_ray_pin_candidate
    use_side_to_move_canonicalization: true
    use_triads: true
    topk_triad_cells: 128
    adapter:
      mode: schema_or_soft
      incidence_uses_current_board_only: true
      allow_token_adapter_all_channels: true
      require_no_label_features: true
    regularization:
      dropout: 0.10
      sheaf_map_l2: 0.00001
      relation_gate_l2_to_one: 0.0001
      energy_budget: 0.0
  data:
    split_dir: data/splits/crtk_sample_3class
    train_file: split_train.parquet
    val_file: split_val.parquet
    test_file: split_test.parquet
    encoding: simple_18
  training:
    loss: cross_entropy
    class_weighting: train_binary_counts
    optimizer: AdamW
    learning_rate: 0.0003
    weight_decay: 0.0001
    batch_size_preferred: 256
    batch_size_fallback: 128
    max_epochs: 50
    early_stopping_patience: 8
    gradient_clip_norm: 1.0
    seeds: [0, 1, 2]
  evaluation:
    metrics:
      - accuracy
      - macro_f1
      - auroc
      - fine_label_to_binary_confusion
      - fine1_recall_as_puzzle_like
      - fine0_false_positive_rate
```

```yaml
model_spec:
  class_name: OrientedTacticalSheafNet
  module_path: chess_nn_playground.models.oriented_tactical_sheaf
  input:
    tensor: x
    shape: [batch, channels, 8, 8]
    dtype: float32
  output:
    tensor: logits
    shape: [batch, 2]
    activation: none
  submodules:
    BoardStateAdapter:
      input_shape: [batch, channels, 8, 8]
      outputs:
        square_raw: [batch, 64, d_in]
        piece_state: [batch, 64, 13]
        occupancy: [batch, 64]
    TacticalIncidenceBuilder:
      outputs:
        edge_weight_dense: [batch, relation_types, 64, 64]
        triad_index_optional: [batch, topk_triad_cells, 3]
    SquareTokenEncoder:
      output_shape: [batch, 64, 64]
    SheafDiffusionBlock:
      repeats: 4
      stalk_dim: 8
      returns_energy: true
    TriadDefectPool:
      enabled: true
      topk: 128
    TacticalReadout:
      output_shape: [batch, 2]
  complexity:
    dense_relation_order: "O(batch * layers * relation_types * 64^2 * stalk_dim^2)"
    default_parameter_range: "130k-260k"
  tests_required:
    - forward_shape_c18
    - forward_shape_c112
    - backward_smoke
    - sheaf_energy_nonnegative
    - incidence_builder_piece_mask_examples
    - no_label_or_engine_feature_inputs
```

```yaml
research_continuity:
  idea_fingerprint: null
  closest_duplicate_risk: null
  do_not_repeat_if_this_fails: []
  suggested_next_search_directions: []
  notes:
    idea_fingerprint_text: "side-to-move canonical tactical incidence complex plus learned cellular-sheaf Laplacian and gluing-defect energy pooling"
    closest_duplicate_risk_text: "Could be confused with a legal-move GNN; distinguish by learned sheaf restriction maps, PSD coboundary energy, and central randomized-relation falsifier."
    do_not_repeat_if_this_fails_text:
      - "Do not retry with merely more sheaf layers, larger stalk dimension, or more relation types if randomized masks match real masks."
      - "Do not recast as a generic GNN-on-squares unless a new falsifiable mechanism is added."
    suggested_next_search_directions_text:
      - "Engine-free differentiable proof-number or forcing-line surrogate from legal move trees, with no engine scores."
      - "Causal invariance across encoding families to suppress dataset artifacts."
      - "Optimal-transport comparison between attack mass and defense mass around kings and high-value pieces."
      - "Information bottleneck models that isolate near-puzzle class 1 from verified puzzle class 2 without using fine labels as inputs."
```

## 13. Prompt Maintenance Notes For Codex

| Proposed prompt update | Why it helps | Exact section to edit |
|---|---|---|
| Add: "If a sheaf/graph incidence idea fails because randomized relation masks match real masks, future cycles must not propose the same family with only larger dimensions or more edge types." | Prevents anti-duplicate drift into routine scaling after a falsified structural hypothesis. | Research Goal or Common Approaches Rejected |
| Add: "For chess symmetry ideas, explicitly state which transformations preserve pawns, castling metadata, and side-to-move; full `D4` board symmetry is not allowed without a proof of label-preserving transformation." | Avoids false equivariance assumptions that can silently hurt chess tasks. | Hard Constraints or Mathematical Thesis instructions |
| Add: "Every proposed structured model must include one degree-preserving or semantics-destroying ablation that can falsify the central structure claim." | Forces information gain even when the model fails. | Ablation Plan |
| Add: "Architecture specs must identify how current board state is extracted from each encoding, or provide a label-free fallback adapter." | Prevents Codex from needing clarification and prevents accidental label/metadata leakage. | Architecture Specification |
| Add: "When using LC0-family encodings, distinguish current-board channels used for rule-derived incidence from history channels used only by learned token adapters." | Keeps deterministic chess-rule features tied to the current position and avoids unintended temporal assumptions. | Problem Restatement And Data Contract |
| Add: "For near-puzzle class `1`, require a metric at matched fine-label-0 false positive rate." | The class `1` behavior is central to puzzle-likeness and can be hidden by aggregate binary metrics. | Benchmark And Falsification Criteria |

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

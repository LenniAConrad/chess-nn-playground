# Codex Handoff Packet: Tactical Radius Filtration

## 1. File Metadata

- **Project:** `chess-nn-playground`
- **Packet type:** original research handoff
- **Generated:** 2026-04-28 08:57:39 new_york
- **Filename:** `chess_nn_research_2026-04-28_0857_tuesday_new_york_tactical_radius.md`
- **Idea name:** Tactical Radius Filtration, abbreviated `TRF`
- **Task:** binary chess puzzle classifier from a current-board tensor
- **Input:** `x: FloatTensor[B, C, 8, 8]`
- **Output:** `logit: FloatTensor[B]`
- **Binary target:** `y = 1[fine == 2]`; therefore fine labels `0` and `1` map to negative class `0`, and fine label `2` maps to positive class `1`.
- **Required report:** `3 x 2` diagnostic table with rows `fine in {0,1,2}` and columns `pred_bin in {0,1}`.
- **Forbidden inputs:** engine scores, principal variations, node counts, mate scores, best moves, verification metadata, source labels.
- **Selected operator:** a chess-rule scale space over tactical graph distance, not an image pyramid or ordinary multiscale attention.

## 2. Executive Selection

Select **Tactical Radius Filtration**.

The core bet is that a chess puzzle is not multiscale because the visual board has objects at different pixel sizes. Every square is already at the same spatial resolution. It is multiscale because tactical evidence lives at different **rule distances**: a loose piece may be one attack edge away, a fork may be two relation steps away, a king-net may be three or four relation steps away through attacks, defenses, pins, escape squares, and ray blockers.

`TRF` builds a deterministic, board-dependent graph from legal chess relations derivable from the current board tensor. It then creates exact tactical shells around every square and piece. Scale `r` means “reachable in exactly `r` chess-rule relation steps,” not “visible after downsampling,” “visible through a dilated kernel,” or “attended to at a larger window size.”

The selected model uses:

- no FPN;
- no CNN feature pyramid;
- no dilated convolution mixer;
- no wavelet scattering;
- no residual pyramid;
- no hypercolumn readout;
- no query-key square attention;
- no engine-derived labels or metadata.

The model should be tested against a square-local MLP, a small ordinary CNN baseline, and a same-parameter non-chess graph baseline. The operator only earns its place if the chess-defined scale is the reason for the lift.

## 3. Data Contract

### Input tensor

Expected input:

```text
x: FloatTensor[B, C, 8, 8]
```

The packet assumes `C` contains a deterministic current-board encoding with enough information to recover piece identity and side-to-move. The preferred channel schema is one of:

```text
Option A: absolute colors
[white_pawn, white_knight, white_bishop, white_rook, white_queen, white_king,
 black_pawn, black_knight, black_bishop, black_rook, black_queen, black_king,
 side_to_move, optional_state_planes...]

Option B: side-relative
[stm_pawn, stm_knight, stm_bishop, stm_rook, stm_queen, stm_king,
 opp_pawn, opp_knight, opp_bishop, opp_rook, opp_queen, opp_king,
 side_to_move_or_orientation_marker, optional_state_planes...]
```

`TRF` needs a `channel_schema` adapter. If the project already has a board tensor convention, implement only the adapter, not a new dataset format.

### Allowed current-board features

Allowed because they are part of the current state, not search output:

```text
piece placement
side to move
castling rights if already encoded in the current-board tensor
en-passant square if already encoded in the current-board tensor
halfmove/fullmove counters only if already part of the project baseline tensor
```

For this classifier, castling and en-passant can be ignored inside the rule graph unless already cheap to decode. Attack, defense, pin, ray, and king-zone structure are enough for the selected operator.

### Disallowed features

Do not pass or derive these into the model, sampler, batch metadata, loss, or diagnostics:

```text
engine score
centipawn score
win/draw/loss engine probability
principal variation
node count
search depth
mate score
best move
solution move
verification flag
puzzle source
source label
site label
curation metadata
```

### Labels

Training labels:

```text
fine: LongTensor[B] with values {0, 1, 2}
y: FloatTensor[B] = (fine == 2).float()
```

The model trains on one binary target. Fine labels are retained for diagnostics and stratified validation, not for a second predictive head in the default implementation.

### Split discipline

Use position-level leakage control:

```text
split_key = hash(canonical_fen_without_clocks_or_source)
```

If multiple rows share the same current board, all copies must stay in the same split. If the project stores game identifiers, an even stricter game-level split is acceptable.

### Required validation report

For every validation run, produce:

```text
diagnostic[fine_label, pred_bin] += 1
```

where:

```text
fine_label in {0, 1, 2}
pred_bin = 1[sigmoid(logit) >= threshold]
```

Report both counts and row-normalized rates:

```text
              pred 0      pred 1
fine 0       count/rate   count/rate
fine 1       count/rate   count/rate
fine 2       count/rate   count/rate
```

Use threshold `0.5` for the first run. Later sweeps may report an additional validation-tuned threshold, but the `0.5` table should remain for comparability.

## 4. Multiscale Research Background

Conventional computer-vision multiscale design usually changes image resolution or receptive-field size. Feature Pyramid Networks use a top-down architecture with lateral connections to build semantically strong feature maps at multiple image scales; that is exactly the family this packet avoids. Source: Lin et al., “Feature Pyramid Networks for Object Detection,” CVPR 2017, DOI `10.1109/CVPR.2017.106`, https://doi.org/10.1109/CVPR.2017.106.

Atrous/dilated convolution controls effective field of view without reducing feature-map resolution, and DeepLab-style atrous spatial pyramid pooling probes feature layers at multiple sampling rates. That is also forbidden here because it defines scale through spatial sampling, not chess rules. Source: Chen et al., “DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs,” IEEE TPAMI 2018, DOI `10.1109/TPAMI.2017.2699184`, https://doi.org/10.1109/TPAMI.2017.2699184.

Hypercolumns concatenate activations of CNN units above a pixel across layers to combine localization and semantics. `TRF` avoids this because the readout is not a layer-stack concatenation over image resolutions; it is a rule-shell summary over chess relations. Source: Hariharan et al., “Hypercolumns for Object Segmentation and Fine-Grained Localization,” CVPR 2015, DOI `10.1109/CVPR.2015.7298642`, https://doi.org/10.1109/CVPR.2015.7298642.

Wavelet scattering creates stable multiscale representations by cascading wavelet transforms, modulus nonlinearities, and averaging. `TRF` rejects wavelets because chess scale is not frequency-band scale; a knight fork and a pinned bishop can be far apart in pixel geometry while being adjacent in tactical relation space. Sources: Mallat, “Group Invariant Scattering,” Communications on Pure and Applied Mathematics 2012, DOI `10.1002/cpa.21413`, https://doi.org/10.1002/cpa.21413; Bruna and Mallat, “Invariant Scattering Convolution Networks,” IEEE TPAMI 2013, DOI `10.1109/TPAMI.2012.230`, https://doi.org/10.1109/TPAMI.2012.230.

Classic scale-space work is still useful as a metaphor: coarser representations should expose larger structures without inventing spurious ones. Perona and Malik’s anisotropic diffusion is especially relevant because it keeps semantically meaningful boundaries sharp instead of blurring everything uniformly. `TRF` borrows that principle, but the diffusion domain is a chess-rule graph, not an image grid. Source: Perona and Malik, “Scale-Space and Edge Detection Using Anisotropic Diffusion,” IEEE TPAMI 1990, DOI `10.1109/34.56205`, https://doi.org/10.1109/34.56205.

Transformer self-attention is a powerful general relation mechanism, but ordinary square-to-square attention is intentionally not selected. The proposed model does not learn all pairwise square affinities with query-key dot products; it constructs sparse chess-rule neighborhoods first and only learns typed shell transforms. Source: Vaswani et al., “Attention Is All You Need,” arXiv `1706.03762`, https://arxiv.org/abs/1706.03762.

AlphaZero-style chess networks demonstrate that board tensors can support strong chess learning without handcrafted evaluation functions, but AlphaZero uses self-play, policy/value training, and MCTS. This packet is narrower: one supervised puzzle logit from the current board only, with no search outputs. Source: Silver et al., “A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play,” Science 2018, DOI `10.1126/science.aar6404`, https://doi.org/10.1126/science.aar6404.

## 5. Candidate Search Trace

### Candidate A: board partition lattice

Define partitions by files, ranks, quadrants, center, king wings, pawn fronts, and promotion zones. Compute zeta summaries over partitions and use Möbius increments across refinements.

Decision: useful, but too static. It captures strategic geography better than tactical motifs. It can miss positions where the key relation is a knight fork, pin, x-ray, or overloaded defender crossing partition boundaries.

### Candidate B: piece-zone hierarchy

Define zones around each piece: own square, legal move zone, attack zone, defended zone, king zone, and high-value target zone.

Decision: strong candidate, but the hierarchy becomes arbitrary unless zones are nested and comparable across piece types. A queen’s zone and a knight’s zone are not naturally ordered.

### Candidate C: relation coarsening

Start with fine relation types, then merge them at coarser scales: exact piece attack at radius one, attack/defense chain at radius two, king pressure and material tension at radius three.

Decision: keep as a component. Relation coarsening is the right way to avoid exploding edge types at larger tactical radii.

### Candidate D: rule-derived diffusion over attack graph

Build an attack/defense/ray graph and run learned diffusion steps.

Decision: promising, but plain diffusion blurs immediate tactics into broad pressure. Exact-shell features are cleaner than repeated smoothing.

### Selected synthesis: Tactical Radius Filtration

Use a board-dependent chess relation graph, define a shortest-path tactical radius, compute exact shells by a filtration, and apply relation-coarsened transforms at each shell. This preserves the good parts of Candidates B, C, and D while avoiding a static partition-only model.

## 6. Rejected Common Approaches

### Feature pyramid CNN

Rejected because scale would be inherited from CNN depth and downsampling. On an `8 x 8` chessboard, downsampling destroys square identity almost immediately and does not define a chess concept of scale.

### Dilated CNN mixer or ASPP-style module

Rejected because dilation rate is a spatial sampling parameter. A bishop pin can be visible along a diagonal ray at distance five, while a knight fork can be visible at a geometric offset of `(1,2)`. Spatial dilation does not respect chess move primitives.

### Wavelet scattering

Rejected because frequency localization is not the target structure. Tactical motifs are sparse relational objects, not texture bands.

### Residual pyramid

Rejected because a residual pyramid would still treat coarser scale as a network-depth or resolution artifact. `TRF` has set-theoretic shell increments, not top-down residual feature fusion.

### Hypercolumn readout

Rejected because concatenating layer activations above each square does not force chess-rule scale. It can work, but it does not answer this research prompt.

### Ordinary multiscale attention

Rejected because full square attention can learn arbitrary pairwise correlations and may become a soft lookup over board layouts. The selected operator constructs legal-rule neighborhoods first. The learned part transforms typed shell aggregates; it does not learn query-key affinities over all square pairs.

### Engine-supervised or solution-move-supervised classifier

Rejected because it violates the input contract. The classifier must not use engine scores, PVs, node counts, mate scores, best moves, verification metadata, or source labels.

## 7. Mathematical Thesis

A chess puzzle position is multiscale in **tactical radius**.

Let:

```text
S = {0, ..., 63}
```

be the set of board squares. Let `B` be the current board decoded from `x`. For every rule relation type `t` in a finite relation set `T`, define a board-dependent adjacency matrix:

```text
A_t(B) in {0,1}^{64 x 64}
```

where:

```text
A_t(B)[u,v] = 1
```

means that square `v` is connected to square `u` by relation `t` under the current chess rules and occupancy.

The relation set should include at least:

```text
pawn_attack_stm
pawn_attack_opp
knight_attack
bishop_ray_contact
rook_ray_contact
queen_ray_contact
king_contact
same_color_defense
enemy_attack
slider_blocker
slider_xray_after_one_blocker
king_zone_stm
king_zone_opp
pawn_front_or_promotion_lane
```

Now define the untyped tactical contact graph:

```text
M(B) = I OR A_t(B) OR A_t(B)^T over all t in T
```

where `OR` is Boolean union. This graph is not a legal-move generator and does not ask what the best move is. It only describes current-board contacts and rule-derived zones.

Define chess tactical distance:

```text
d_B(u,v) = min { r >= 0 : M(B)^r[u,v] = 1 }
```

using Boolean matrix multiplication. If no path exists, `d_B(u,v) = infinity`.

Define closed tactical balls and exact tactical shells:

```text
P_r(B)[u,v] = 1[d_B(u,v) <= r]
Q_0(B) = I
Q_r(B) = P_r(B) AND NOT P_{r-1}(B), for r >= 1
```

`Q_r` is the `r`-th shell of the filtration. This is the central mathematical move: scale is not a spatial window. Scale is the minimum number of chess-rule contacts needed to connect two squares.

Interpretation:

```text
r = 0: the piece or empty-square identity itself
r = 1: direct attack, defense, ray contact, blocker, king-zone membership
r = 2: defender-of-attacker, attacker-of-defender, fork setup, pin support, escape-square pressure
r = 3: local tactical complex around king, queen, pinned piece, promotion lane, or overloaded defender
r = 4: broad tactical context; usually optional on 8 x 8
```

The thesis is falsifiable:

> Fine-2 puzzle positions should show a different profile across exact tactical shells than fine-0 and fine-1 positions. In particular, positives should have stronger radius-2 and radius-3 interaction signatures around kings, high-value pieces, blockers, and defended attackers than negatives with similar material and square-local features.

## 8. Multiscale Operator

### Name

`TacticalRadiusFiltration`

### Inputs

```text
x: FloatTensor[B, C, 8, 8]
channel_schema: object describing piece planes and side-to-move plane
R: max tactical radius, recommended R = 3 initially
d_model: hidden width, recommended 96 or 128
```

### Step 1: decode current board

Decode a non-gradient board state:

```text
piece_id: LongTensor[B, 64]      # empty, P, N, B, R, Q, K with color or side-relative sign
occupied: BoolTensor[B, 64]
stm: BoolTensor[B]
```

This decode is not a model output. It is an adapter from the existing tensor convention into rule masks.

### Step 2: square feature lift

Flatten the board:

```text
X = rearrange(x, "b c h w -> b (h w) c")
```

Apply a per-square lift:

```text
H0 = MLP_square([X, square_coord_features, side_to_move_feature])
H0: FloatTensor[B, 64, d_model]
```

This is equivalent to a shared `1 x 1` square MLP, not a CNN pyramid.

### Step 3: build typed chess relations

For each board in the batch, build:

```text
A_t: BoolTensor[B, 64, 64]
```

Use deterministic rule masks:

```text
knight jumps
king steps
pawn attack diagonals by color
rook rays stopped by occupancy
bishop rays stopped by occupancy
queen rays as rook-or-bishop rays
same-color defended contacts
enemy attacked contacts
first blocker on each slider ray
x-ray contact after exactly one blocker
king-zone squares around both kings
pawn-front and promotion-lane squares
```

Use pseudo-legal attack/contact masks, not engine search. Do not require checkmate detection, legal best moves, or PVs.

### Step 4: define relation coarsening by scale

Use a coarsening map:

```text
pi_r: T -> G_r
```

Recommended groups:

```text
G_1:
  own_direct_attack
  opp_direct_attack
  own_defense
  opp_defense
  slider_blocker
  slider_xray
  king_zone
  pawn_lane

G_2:
  attack_chain
  defense_chain
  attack_defense_collision
  ray_continuation
  king_zone_pressure
  pawn_chain_pressure

G_3:
  king_pressure_complex
  material_tension_complex
  escape_square_complex
  promotion_or_back_rank_complex
  open_line_complex
```

At larger radius, relation types should become coarser. That is intentional: exact relation identity is useful at radius one, while radius three should describe tactical complexes.

### Step 5: create exact-shell relation matrices

Let:

```text
M = I OR union_t A_t OR A_t^T
P_0 = I
P_r = 1[(M^r) > 0] OR P_{r-1}
Q_r = P_r AND NOT P_{r-1}
```

For grouped last-step shells, define:

```text
M_g = OR_{t: pi_r(t)=g} A_t
Q_{r,g} = 1[(P_{r-1} @ M_g) > 0] AND NOT P_{r-1}
```

where `@` is Boolean matrix multiplication. Normalize each shell:

```text
D_{r,g}[u,u] = max(1, sum_v Q_{r,g}[u,v])
N_{r,g} = D_{r,g}^{-1} Q_{r,g}
```

### Step 6: aggregate shell features

For each radius and relation group:

```text
Z_{r,g} = N_{r,g} H0
Z_{r,g}: FloatTensor[B, 64, d_model]
```

Project and sum groups:

```text
Y_r = sum_g Z_{r,g} W_{r,g} + H0 W_{r,self}
H_r = LayerNorm(GELU(Y_r))
H_r: FloatTensor[B, 64, d_model]
```

Important: compute each `H_r` from `H0` and the exact shell at radius `r`. Do not feed `H_{r-1}` into `H_r` as a residual pyramid. The nested structure is in `Q_r`, not in network depth.

### Step 7: rule-zone readout

Create deterministic masks:

```text
piece_mask
stm_piece_mask
opp_piece_mask
stm_king_zone_mask
opp_king_zone_mask
high_value_piece_mask   # K, Q, R; side-relative
slider_blocker_mask
```

Pool each radius:

```text
pool_piece_r      = mean_mask(H_r, piece_mask)
pool_stm_r        = mean_mask(H_r, stm_piece_mask)
pool_opp_r        = mean_mask(H_r, opp_piece_mask)
pool_stm_kzone_r  = mean_mask(H_r, stm_king_zone_mask)
pool_opp_kzone_r  = mean_mask(H_r, opp_king_zone_mask)
pool_blocker_r    = mean_mask(H_r, slider_blocker_mask)
```

Concatenate:

```text
V = concat_r([
  pool_piece_r,
  pool_stm_r,
  pool_opp_r,
  pool_stm_kzone_r,
  pool_opp_kzone_r,
  pool_blocker_r,
  shell_count_features_r
])
```

Then:

```text
logit = MLP_readout(V).squeeze(-1)
```

The default model returns only this one puzzle logit.

### Step 8: shell count features

Append a small set of non-engine scalar features derived from the same relation matrices:

```text
count_own_attacks_to_opp_king_zone
count_opp_attacks_to_stm_king_zone
count_defended_attackers
count_attacked_high_value_pieces
count_slider_xrays_to_king_or_queen
count_safe_or_attacked_escape_squares
```

These are not labels and not search outputs. They are current-board rule-field counts. Keep them low-dimensional so the neural shell features remain the main signal.

## 9. Architecture Tensor Contract

### Module signature

```python
class TacticalRadiusFiltrationClassifier(nn.Module):
    def __init__(
        self,
        channel_schema,
        d_model: int = 128,
        radius: int = 3,
        relation_groups: str = "default",
        use_shell_counts: bool = True,
    ): ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, 8, 8]
        # returns logits: [B]
```

### Shape contract

```text
x                         [B, C, 8, 8]
X_flat                    [B, 64, C]
piece_id                  [B, 64]
occupied                  [B, 64]
H0                        [B, 64, D]

A_t                       [B, T, 64, 64]      bool
M                         [B, 64, 64]         bool
P_r                       [B, 64, 64]         bool
Q_{r,g}                   [B, G_r, 64, 64]    bool or float
N_{r,g}                   [B, G_r, 64, 64]    float

Z_{r,g}                   [B, G_r, 64, D]
Y_r                       [B, 64, D]
H_r                       [B, 64, D]

zone_pools_per_r          [B, K_pool * D]
shell_count_features_r    [B, K_count]
V                         [B, (R+1) * (K_pool * D + K_count)]
logit                     [B]
```

### Recommended initial hyperparameters

```text
D = 128
R = 3
dropout = 0.10
readout_hidden = 256
activation = GELU
normalization = LayerNorm
batch size = project-dependent
```

### Dense versus sparse implementation

Because the board has only `64` squares, dense relation tensors are acceptable:

```python
Z = torch.einsum("bgij,bjd->bgid", N, H0)
```

Even with `G = 12`, this is small. Optimize later only if profiling proves it necessary.

### Invariance and augmentation

Recommended canonicalization:

```text
Orient board so side-to-move is always "up" from the model's perspective.
Swap colors into stm/opp channels when using side-relative encoding.
```

Recommended augmentations:

```text
horizontal mirror with correct file remapping
color/side-relative swap only when labels remain valid
do not apply arbitrary rotations because pawn direction and castling geometry are not rotation-invariant
```

### Output contract

The model emits:

```text
logit: FloatTensor[B]
```

No policy head. No value head. No best-move head. No fine-class head in the default version.

## 10. Training Objective

### Primary objective

Use weighted binary cross entropy:

```text
y_i = 1[fine_i == 2]

L_bce = BCEWithLogitsLoss(
    logit_i,
    y_i,
    pos_weight = N_negative / N_positive
)
```

If class imbalance is severe, combine `pos_weight` with a stratified batch sampler over `fine in {0,1,2}` so the model cannot ignore fine-2 positives.

### Calibration term

Optional but recommended:

```text
p_i = sigmoid(logit_i)
L_brier = mean((p_i - y_i)^2)
L = L_bce + 0.05 * L_brier
```

This helps the `3 x 2` diagnostic remain interpretable.

### Shell dropout

During training only, randomly drop entire shell groups with small probability:

```text
drop Q_{r,g} with p = 0.05 for r >= 1
```

This is not spatial dropout and not a pyramid trick. It tests whether the classifier can avoid brittle reliance on one relation group.

### Fine labels

Default training should not use a fine-class prediction head. Fine labels are used to form `y` and to produce the required diagnostic.

A separate ablation may add a stop-gradient fine probe for analysis, but that should not be part of the selected default model.

### Validation diagnostic

Every validation epoch logs:

```text
threshold = 0.5
pred_bin = 1[sigmoid(logit) >= threshold]

diag = zeros(3, 2)
for fine_i, pred_i in validation_batch:
    diag[fine_i, pred_i] += 1
```

Also log:

```text
binary AUROC
binary AUPRC
Brier score
fine-row false positive rate for fine 0
fine-row false positive rate for fine 1
fine-2 true positive rate
```

The `3 x 2` table is the required report. The extra metrics are supporting diagnostics.

## 11. Ablations

### A1: radius limit

Run:

```text
R = 0
R = 1
R = 2
R = 3
R = 4
```

Expected result: `R=0` underfits, `R=1` improves direct tactics, `R=2` or `R=3` gives the main lift, `R=4` may saturate or overfit.

### A2: exact shells versus closed balls

Compare:

```text
exact shell Q_r
closed ball P_r
```

Expected result: exact shells should be cleaner. If closed balls win decisively, the Möbius filtration thesis is weaker.

### A3: chess graph versus Chebyshev graph

Replace `M(B)` with an `8-neighbor` king-distance grid graph while preserving the same radius count.

Expected result: chess graph should win. If Chebyshev distance matches it, the operator is not proving a chess-specific scale advantage.

### A4: relation type shuffle

Preserve graph degree and shell sizes but randomly permute relation group labels within each batch.

Expected result: performance should drop. If it does not, typed relations are decorative.

### A5: no x-ray and blocker relations

Remove:

```text
slider_blocker
slider_xray_after_one_blocker
```

Expected result: drop on positions whose solution likely depends on pins, skewers, discovered attacks, back-rank themes, and overloaded blockers.

### A6: no king-zone masks

Remove:

```text
stm_king_zone_mask
opp_king_zone_mask
king_zone_pressure groups
```

Expected result: drop on mating/net-like puzzles or fine-2 positives with king pressure.

### A7: no shell count features

Remove scalar rule-field counts and keep only neural shell pools.

Expected result: small drop at most. A large drop means the model is mostly a handcrafted feature classifier, not learning useful shell representations.

### A8: MLP-only baseline

Flatten `x` and train a square-local or board-flat MLP with comparable parameter count.

Expected result: `TRF` should beat it on fine-2 recall at the same false-positive rate.

### A9: small CNN baseline

Use a simple non-pyramid CNN as a sanity baseline. Do not select it as the research operator.

Expected result: CNN may be competitive, but `TRF` should be more stable across board mirrors and less dependent on square-distance artifacts.

### A10: forbidden-feature audit

Run a deliberately clean loader that asserts these keys are absent:

```text
engine_score
pv
nodes
mate
best_move
source
verification
```

Expected result: model performance should be reproducible without any forbidden fields.

## 12. Falsification Criteria

Treat the idea as falsified or not worth merging if any of these hold after a fair sweep:

1. **No lift over non-chess scale.** The Chebyshev graph ablation matches `TRF` within the validation confidence interval while using comparable parameters.
2. **No lift over flat input.** A board-flat MLP or small CNN matches `TRF` on AUROC, AUPRC, and fine-2 recall at the same fine-0/fine-1 false-positive rate.
3. **Relation labels do not matter.** Relation type shuffling preserves performance.
4. **Radius does not matter.** `R=0` or `R=1` matches `R=2/3`, which would mean the multiscale thesis is unnecessary.
5. **Shells collapse.** Learned readout norms or ablation deltas show nearly all signal comes from local piece identity and material counts.
6. **Fine diagnostic fails.** The required `3 x 2` table shows fine-2 positives are not separated from fine-0/fine-1 negatives except by a threshold that creates unacceptable fine-1 false positives.
7. **Leakage suspicion.** Performance drops sharply when splitting by canonical FEN hash or game ID, suggesting duplicated positions or source artifacts drove the result.
8. **Forbidden data dependency.** Any path from engine output, best move, verification metadata, or source labels reaches the model, sampler, or loss.
9. **Symmetry brittleness.** Horizontal mirror augmentation or side-relative canonicalization changes predictions in ways inconsistent with chess symmetry.
10. **Cost unjustified.** Dense relation construction dominates runtime and does not buy measurable validation lift over simpler rule-count features.

A good result is not just “higher accuracy.” A good result is a visible, repeatable advantage specifically tied to tactical radius and typed chess relations.

## 13. Implementation Notes

### Suggested file target inside the project

Implement as:

```text
models/tactical_radius_filtration.py
```

and add a config entry such as:

```text
model.name = tactical_radius_filtration
model.d_model = 128
model.radius = 3
model.use_shell_counts = true
```

Do not create a policy head or engine-analysis dependency.

### Rule construction

For initial implementation, use precomputed `64 x 64` masks:

```text
knight_mask[colorless]
king_mask[colorless]
pawn_attack_mask[side]
rook_ray_squares[origin, direction, ordered]
bishop_ray_squares[origin, direction, ordered]
between_mask[u, v]
line_mask[u, v]
```

For each board:

1. Get occupancy.
2. For each slider origin, walk the ordered ray until the first blocker.
3. Add ray-contact edges to empty traversed squares and first blocker.
4. If exactly one blocker exists, continue to the next occupied square and add an `xray_after_one_blocker` edge.
5. Add attack/defense tags based on piece color and target occupancy.
6. Add king-zone relations around both kings.

This is deterministic board logic, not search.

### Python-chess option

Using `python-chess` for attack masks is acceptable for a first prototype if it is used only for current-board legal contacts. Do not call engines. Do not request legal best moves. Do not analyze positions.

If speed matters, replace `python-chess` with vectorized PyTorch or NumPy relation construction after the idea is validated.

### Handling ambiguous tensor schemas

If the existing `C` channels do not expose piece planes directly, add a schema adapter. Do not train a hidden decoder to infer pieces and then build rule masks from soft piece probabilities in the first implementation. The research question is about tactical-radius scale, not latent board parsing.

### Batching

Dense tensors are easiest:

```python
# N: [B, G, 64, 64]
# H0: [B, 64, D]
Z = torch.einsum("bgij,bjd->bgid", N.float(), H0)
```

Then:

```python
Y = sum(project_g(Z[:, g]) for g in range(G))
H_r = layer_norm(gelu(Y + self_project(H0)))
```

### Numeric details

Use:

```text
degree clamp min = 1
float32 for relation matrices after normalization
bool for construction
LayerNorm after each shell transform
dropout only after shell transform or readout MLP
```

### Diagnostics implementation

Add a validation utility:

```python
def fine_binary_diagnostic(fine, logits, threshold=0.5):
    pred = (logits.sigmoid() >= threshold).long()
    out = torch.zeros(3, 2, dtype=torch.long)
    for c in (0, 1, 2):
        for p in (0, 1):
            out[c, p] = ((fine == c) & (pred == p)).sum()
    return out
```

Log row-normalized rates with counts. Keep this exact table stable across experiments.

### Failure-mode checks

Add assertions:

```text
assert logits.shape == [B]
assert not batch contains forbidden keys
assert relation matrices are detached from labels
assert fine labels are not passed into forward()
assert best_move is not loaded
assert engine fields are not loaded
```

### Practical first run

Start small:

```text
D = 96
R = 3
G_1/G_2/G_3 default relation groups
batch size as large as memory allows
weighted BCE
threshold 0.5 diagnostic
```

If the first run is unstable, reduce relation groups before reducing radius. The radius is the research claim.

## 14. Prompt Maintenance

Keep the prompt anchored to this invariant:

> Scale must be mathematically defined in chess terms.

For this packet, that definition is tactical graph distance over current-board rule relations.

When extending the idea, acceptable variants include:

```text
piece-zone hierarchy added as extra masks
board partition lattice as a secondary readout
relation coarsening learned by a small categorical embedding
radius-specific scalar rule fields
side-relative canonicalization improvements
```

Do not let future revisions drift into:

```text
FPN
CNN pyramid
dilated CNN mixer
wavelet scattering
residual pyramid
hypercolumn readout
ordinary multiscale attention
policy-head supervision
engine-score distillation
best-move labels
source-label leakage
```

If the project needs a shorter name in configs, use:

```text
trf
```

If the project needs a one-line description, use:

```text
A chess-rule multiscale classifier that defines scale by exact tactical-radius shells over attack, defense, ray, x-ray, king-zone, and pawn-lane relations.
```

The strongest maintenance test is simple: if removing the chess-rule graph does not hurt, the idea has failed. If radius-2 and radius-3 shells produce a repeatable lift in the `3 x 2` fine diagnostic without forbidden inputs, keep developing it.

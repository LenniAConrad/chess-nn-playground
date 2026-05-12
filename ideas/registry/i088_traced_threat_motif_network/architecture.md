# Architecture

`Traced Threat Motif Network` (TTMN) is a bespoke PyTorch implementation
of the source packet's traced-motif algebra over the `puzzle_binary`
contract. It treats the side-to-move as `u` and the opponent as `t`,
maps the board to ten typed `(64 x 64)` group operators, evaluates a
fixed vocabulary of 24 traced-threat motifs through composition, and
returns one BCE logit plus motif and contest diagnostics.

## Implementation Binding

- Registered model name: `traced_threat_motif_network`
- Source implementation file: `src/chess_nn_playground/models/trunk/traced_threat_motif.py`
- Idea-local wrapper: `ideas/registry/i088_traced_threat_motif_network/model.py`

## Modules

`BoardStem` is a residual square encoder over the configured board planes.
It maps `(B, C, 8, 8)` board tensors through a `3x3` lift to `d_model`
channels, runs `stem_blocks` `ResidualBlock` units (GELU-gated `3x3` conv
pairs with a residual sum), pre-LayerNorms the per-square token sequence,
and returns both the token map `h` of shape `(B, 64, d_model)` and the
spatial `(B, d_model, 8, 8)` map. Mean and max pooling over `h` produce
the `2 d_model`-wide trunk pool consumed by the head.

`RelationMaskBuilder` builds a deterministic `(B, K = 36, 64, 64)` mask
that is zero everywhere a piece geometry forbids a relation. For each of
the twelve `(color, piece)` combinations it materialises three role-typed
geometries — `ctrl` (squares the piece attacks/threatens), `hit` (the
ctrl mask intersected with the enemy occupancy), and `quiet` (squares the
piece can move to without capture). Sliding pieces (`bishop`, `rook`,
`queen`) are gated by a fully-precomputed between-square clear-path mask.
The pawn double-step is gated by the corresponding middle-square clearance.

`RelationGate` is the only board-conditioned attention in the network. It
maps the per-token features `h` through one query and one key projection
*per* raw relation channel `k` (not per piece-pair), scores
`q[k] @ k[k]^T / sqrt(d)`, adds a per-channel bias, takes a softplus,
multiplies by the geometry mask, and row-normalises so the gated raw
operator `a_raw[k]` is row-substochastic and supported only on
geometrically legal pairs.

`GroupMixer` mixes the 36 raw relations into ten board-typed groups
through learned softmax weights over a fixed index pool per group. The
ten groups are

```text
{u_ctrl, u_hit, u_quiet, u_ray, u_jump,
 t_ctrl, t_hit, t_quiet, t_ray, t_jump}
```

with `ray` mixing only `(BISHOP, ROOK, QUEEN)` channels and `jump`
mixing only `(PAWN, KNIGHT, KING)` channels. The mixer is side-to-move
aware: each `u_*` operator selects from White raw channels when the
side-to-move is White and otherwise from Black, and vice versa for the
`t_*` operators.

`MotifComposer` evaluates the fixed 24-word motif vocabulary listed in
`MOTIF_WORDS`. For each word `w = g_1 ... g_L` it composes the operators
`M_w = A_{g_1} ... A_{g_L}` via `torch.bmm`, then reads four scalar
boundary contractions per board:

```text
trace(M_w) = (1 / 64) * tr(M_w)
mass(M_w)  = log(1 + sum(M_w))
king(M_w)  = u^T M_w k_t   (us-piece mass into enemy king)
value(M_w) = u^T M_w v_t   (us-piece mass into enemy material value)
```

`u` is the side-to-move piece occupancy, `k_t` is the enemy king plane,
and `v_t` is the enemy occupancy weighted by a learnable, softplus-
positive material value vector normalised by its max. The four
contractions stack into a `(B, 24 x 4)` motif feature block. A separate
monoidal block adds `loop2_u`, `loop2_t`, `parallel`, and `interaction`,
which are loop and parallel traces of `(A_{u_ctrl}, A_{t_ctrl})`.

`ContestPullback` computes per-square attacker / defender pressure as
the product of inbound `u_ctrl` and inbound `t_ctrl`, returning the
`(B, 8, 8)` heatmap and three scalar diagnostics (mean, top-4 mean,
distribution entropy).

`TracedThreatMotifNet` glues the trunk, mask builder, relation gate,
group mixer, motif composer, and contest pullback together. The forward
path computes:

1. `h, _ = stem(board)`.
2. `mask_raw = mask_builder(piece_planes)`.
3. `a_raw = relation_gate(h, mask_raw)`.
4. `groups = group_mixer(a_raw, side_to_move)`.
5. `motif_features, motif_scores, motif_extra = motif_composer(groups, piece_planes, side_to_move)`.
6. `contest_heatmap, contest_features = contest_pullback(groups)`.
7. `logits = head([cnn_pool, motif_features, contest_features])`.

The head is a LayerNorm-Linear-GELU-Dropout-Linear-GELU-Linear stack
that receives the concatenation of the trunk pool, motif block,
monoidal block, and contest scalars, and emits one BCE logit per board.

## Diagnostics

`forward(x)` returns a dict containing:

- `logits`: shape `(B,)`, BCE-compatible for the one-logit
  puzzle_binary head.
- `prob`: sigmoid of the puzzle logit.
- `motif_scores`: shape `(B, 24)`, the trace + king + value + 0.1 * mass
  combination per motif word.
- `top_motif_idx`: shape `(B, 5)`, indices of the top motifs.
- `contest_heatmap`: shape `(B, 8, 8)` per-square attacker-defender
  pressure.
- `contest_features`: `(B, 3)` mean, top-4 mean, and entropy of contest.
- `trace_closure`, `open_king_mass`, `open_value_mass`: scalar means
  over the motif vocabulary.
- `monoidal_features`: `(B, 4)` loop2_u, loop2_t, parallel, interaction.
- `parallel_loop2`, `interaction_loop`: scalar projections from the
  monoidal block.
- `raw_relation_density`, `gated_relation_density`: mean activity of
  the geometry-mask and gated raw relations.
- `mechanism_energy`, `proposal_profile_strength`,
  `proposal_keyword_count`: scalar reporting fields preserved for
  compatibility with the project's research-packet diagnostic schema.

`forward(x, return_diag=True)` additionally returns
`group_ctrl_mass`, the per-board `(B, 2)` total mass of the
`u_ctrl` and `t_ctrl` operators, used by ablation harnesses.

## Contract

- Input: `(B, C, 8, 8)` board tensor only with `C >= 13`. CRTK /
  verification / source / engine metadata is reporting-only and is not
  consumed.
- Output: dict with `logits` of shape `(B,)` for the one-logit
  puzzle_binary BCE-with-logits trainer, plus the diagnostics listed
  above.
- Target mapping: fine labels `0` and `1` map to binary target `0`;
  fine label `2` maps to binary target `1`.
- Side-to-move: the layer-12 plane is averaged per board to recover the
  scalar `stm` selector that switches the `u`/`t` perspective.
- Material value vector: a learnable, softplus-positive `(6,)`
  parameter normalised by its max so the value contraction stays in
  `[0, 1]` per square.
- Motif vocabulary: a fixed list of 24 traced-threat motif words over
  the ten-group alphabet. The vocabulary spans single-step `hit`
  attacks, two-step pin/fork patterns, ray-skewers, knight-fork
  variants, and four-step sacrifice/decoy patterns. The vocabulary is
  not learned; only the per-channel relation gate, the group-mixing
  softmax, the material value vector, and the head are trainable.

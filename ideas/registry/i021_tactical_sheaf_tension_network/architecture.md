# Architecture

`Tactical Sheaf Tension Network` (`TacticalSheafTensionNet`) implements the
source packet's side-aware cellular sheaf over pseudo-legal attack, defense,
control, and x-ray relations as a bespoke PyTorch model for the repo's
`puzzle_binary` task.

## Implementation Binding

- Registered model name: `tactical_sheaf_tension_network`
- Source implementation file: `src/chess_nn_playground/models/trunk/tactical_sheaf_tension.py`
- Idea-local wrapper: `ideas/registry/i021_tactical_sheaf_tension_network/model.py`

## Modules

`BoardTensorDecoder` reads a `(B, C, 8, 8)` board tensor and recovers per-square
piece type, piece color, and side-to-move. For `simple_18` it uses planes
`0..11` for piece occupancy and plane `12` for the side-to-move flag; for
`lc0_static_112` it uses planes `104..105` to recover side-to-move; for
`lc0_bt4_112` the side-to-move is treated as canonical white-to-move per the
encoding contract. Per-square `role in {0, 1, 2}` (empty / side-to-move /
opponent) is computed from `piece_color` and `side_to_move` so the sheaf is
side-aware without breaking pawn-direction information.

`SquareStalkEncoder` flattens the board to `(B, 64, C)`, concatenates a
deterministic 6-D `square_coords` (rank/file in `[0, 1]`, centered rank/file,
edge distance, square color parity), the per-square piece-type one-hot, the
role one-hot, and the broadcast side-to-move scalar, then runs
`LayerNorm -> Linear -> GELU -> Dropout -> Linear -> LayerNorm` to produce
per-square stalks `h0` of shape `(B, 64, fiber_dim)`.

`TacticalComplexBuilder` builds the directed typed tactical complex from chess
geometry alone (no engine, no labels). Per position and per source piece it
emits four edge kinds:

- `control_empty`: pseudo-legal candidate move into an empty square.
- `attack_enemy`: pseudo-legal capture candidate against an enemy piece.
- `defend_own`: pseudo-legal candidate ray that protects an own piece.
- `xray_one_blocker`: a one-blocker x-ray edge through the first blocker on
  rays for sliders (bishop / rook / queen).

For sliders, every direction follows the ray until the first occupied square;
that target generates the first edge (control / attack / defend depending on
target color), and one further x-ray edge is generated through the first
blocker if a second occupied square is reachable along the same ray. Pawn
edges use forward-diagonal capture candidates oriented by side-to-move and
role; knights and kings use their fixed offset sets.

Each directed edge carries:

- a `relation_id` indexing one of `RELATION_COUNT = role x piece x edge_kind x
  direction` typed restrictions (default `2 * 6 * 4 * 8 = 384`);
- a `relation_group` over `(control_empty, attack_enemy, defend_own,
  xray_one_blocker, king_ring)` for group-pooled statistics, where edges into
  squares inside the king ring of either side are promoted to `king_ring` so
  king-pressure tension can be read out separately;
- a degree-normalized `edge_weight = (deg(src) * deg(dst))^(-1/2)` over the
  combined endpoint degree.

Direction binning supports left-right file-mirror tying (default), so east
and west collapse to the same direction bin and the sheaf is partially
equivariant under file reflection only; pawn direction, castling, and
side-to-move asymmetries are intentionally not tied. Edges are clipped to
`max_edges_per_position` (default `2048`) to bound batched gather/scatter.

`DiagonalLowRankRestrictions` parameterizes the side-aware cellular sheaf's
restriction maps as `R[tau] = diag(d[tau]) + U[tau] V[tau]^T` per relation,
with separate parameters for the source (`R_src`) and target (`R_dst`) ends
of each typed edge. This is the diagonal-plus-low-rank form from the math
thesis: `diag` controls the spectral baseline and the low-rank update
captures relation-specific tactical bias without exploding the parameter
budget. The transpose method realizes `R^T` for the Laplacian-like adjoint
update.

`SheafTensionBlock` realizes one round of typed sheaf diffusion. For every
directed edge `e = (u -> v, tau)` it computes the (signed) coboundary

```text
delta_e = R_src^tau h_u - R_dst^tau h_v in R^{fiber_dim}
E_e = w_e * ||delta_e||_2^2
```

with `w_e` the degree-normalized edge weight. The Laplacian-like residual
update scatters `R_src^T (w_e * delta_e)` to source squares and
`-R_dst^T (w_e * delta_e)` to destination squares, divides by per-square
edge-weighted degree, and applies `h <- LayerNorm(h - eta * lap_update +
NodeMLP(h))` with a sigmoid-bounded `eta in (0, 1)`. Edge dropout multiplies
`w_e` by Bernoulli noise during training only. Each block emits per-batch
mean / weighted mean / max / top-3 edge tension and a per-relation-group
mean tension, plus a normalized edge count.

`TacticalEnergyPool` concatenates the final stalks' mean / max / std node
pools with stm-only and opponent-only weighted means (so the pool is
side-aware), the per-block tension and per-group tension stack, and four
board-count features (stm piece count, opponent piece count, total piece
count, edge-count proxy).

The classifier head is a `LayerNorm -> Linear -> GELU -> Dropout -> Linear`
MLP that produces one BCE-compatible puzzle logit when `num_classes = 1`.

## Diagnostics

`forward` returns a dict with:

- `logits` (BCE-compatible; shape `(B,)` when `num_classes = 1`).
- `sheaf_tension`: pooled per-block mean edge tension.
- `weighted_sheaf_tension`: edge-weight-normalized mean tension across blocks.
- `max_edge_tension`: per-block max tension, max-pooled across blocks.
- `top3_edge_tension`: per-block mean of the top-3 edge tensions.
- `edge_density`: fraction of `max_edges_per_position` actually emitted.
- `control_energy`, `attack_energy`, `defense_energy`, `xray_energy`,
  `king_ring_energy`: per-relation-group mean edge tension averaged over
  blocks.
- `side_piece_count`, `opponent_piece_count`: board-count proxies for
  reporting parity.

Diagnostics are reporting-only; they do not enter the training loss.

## Contract

- Input: `(B, C, 8, 8)` board tensor only. CRTK / verification / source /
  engine metadata is reporting-only and is not consumed.
- Output: dict with `logits` of shape `(B,)` for the one-logit
  `puzzle_binary` BCE-with-logits trainer, plus the diagnostics listed above.
- Symmetry: only file-mirror direction tying is applied by default; pawn
  direction, castling, and side-to-move asymmetries are intentionally not
  tied, matching the source packet's "left-right partial equivariance only"
  stance.

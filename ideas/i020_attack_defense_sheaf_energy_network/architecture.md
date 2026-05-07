# Architecture

`Attack-Defense Sheaf Energy Network` (`AttackDefenseSheafNet`) implements the
source packet's typed chess-incidence sheaf with ray-blocker visibility gates
and residual-energy readout as a bespoke PyTorch model for the repo's
`puzzle_binary` task.

## Implementation Binding

- Registered model name: `attack_defense_sheaf_energy_network`
- Source implementation file: `src/chess_nn_playground/models/attack_defense_sheaf.py`
- Idea-local wrapper: `ideas/i020_attack_defense_sheaf_energy_network/model.py`

## Modules

`SquareAdapter` flattens the `(B, C, 8, 8)` board tensor to `(B, 64, C)`,
appends the precomputed deterministic 6-D `square_coords` (rank/file in
`[0, 1]`, centered rank/file, edge distance, square color parity), and runs
the result through `LayerNorm -> Linear -> GELU -> Dropout -> Linear ->
LayerNorm` to produce per-square stalks `h0` of shape `(B, 64, d_model)`. The
adapter is encoding-agnostic, so `simple_18`, `lc0_static_112`, and
`lc0_bt4_112` all share the same trunk.

`AttackDefenseIncidence` precomputes the directed typed chess-incidence
complex from static geometry only (no engine, no legality oracle) and
registers the result as non-persistent buffers so they ride the module to GPU.
It carries:

- `edge_src`, `edge_dst`: `(E,)` long tensors of square indices in `[0, 63]`.
- `edge_type`: `(E,)` long tensor mapping each directed edge to a relation
  type.  Bucketed distance is folded into the relation key for ray edges so
  short and long pins/skewers are not collapsed.
- `relation_group`: `(E,)` long tensor over `("ray", "knight", "king",
  "pawn_up", "pawn_down")` for group-mean readouts.
- `edge_is_ray`: `(E,)` bool tensor; ray edges receive multiplicative blocker
  visibility, all other edges keep `q_e = 1`.
- `blocker_index`, `blocker_mask`: `(E, max_blockers)` tensors enumerating the
  intervening squares for each ray edge, padded with `-1` (and a boolean
  mask) so masked entries cannot contribute to visibility.
- `edge_geom`: `(E, 8)` deterministic geometry vector (normalized
  source/target rank/file, signed deltas, distance, ray flag).
- `square_coords`: `(64, 6)` per-square coordinate features.

The complex includes ray edges in eight chess directions at distances
1..`max_ray_length`, the eight knight offsets, the eight king-neighborhood
offsets, and oriented `pawn_up_attack`/`pawn_down_attack` diagonals. With
`tie_file_reflection=True` (default) east/west rays and the NE/NW and SE/SW
diagonals share a single relation type so left-right file reflection is the
only safe weak symmetry expressed; pawn direction, side-to-move and castling
asymmetries are not tied. The total count is on the order of 2.4k directed
typed edges, small enough for dense per-batch gather/scatter.

`OccupancyHead` (`LayerNorm -> Linear -> Sigmoid`) reads each stalk to a
single occupancy proxy `o_v in (0, 1)` per square. The occupancy proxy is
unsupervised and is used only by the visibility gate and as a diagnostic.

The ray visibility gate `_ray_visibility` follows the math thesis directly:
for each edge it gathers `o_m` for every blocker square `m in M_e` (replacing
masked / padded entries with `1.0` so they do not affect the product) and
computes
`q_e = prod_{m in M_e} (1 - o_m + eps)`.
Non-ray edges get `q_e = 1`.

`EdgeGate` is the learned tactical gate: it concatenates `[h_src, h_dst,
type_embed(tau), q_e]` and pushes the result through a 2-layer MLP with a
final `sigmoid`, yielding `a_e in (0, 1)`. The total gate is `gamma_e = q_e *
a_e` (the visibility multiplier is mixed in through the gate input and
through the residual scaling below).

`SheafDiffusionBlock` realizes one round of gated typed sheaf diffusion:

- Per-type diagonal-style restriction maps `R_src[tau], R_dst[tau] in
  R^{r x d}` are stored as `(num_types, sheaf_rank, d_model)` parameter
  tensors.
- Per-edge claims `R_src^tau h_u`, `R_dst^tau h_v` are computed with a single
  einsum over the gathered batch.
- The sheaf residual `c_e = R_src^tau h_u - R_dst^tau h_v in R^r` and the
  gated edge energy `E_e = gamma_e * ||c_e||_2^2` are formed.
- The Laplacian-like adjoint update scatters `gamma_e * c_e * R_src^tau` to
  the source square and `gamma_e * c_e * R_dst^tau` to the destination
  square, divides by `(deg_src + deg_dst)` (with `gamma_e` weights) and
  applies `h <- LayerNorm(h - eta * node_delta + NodeMLP(h))` with a
  sigmoid-bounded `eta in (0, 1)`.
- Edge dropout multiplies `gamma_e` by Bernoulli noise during training only.

`AttackDefenseSheafNet` stacks `num_blocks` diffusion blocks (the visibility
gate is recomputed at the trunk and reused across blocks). Per block we keep
mean / std / max / top-k edge energy, mean / std gate, mean visibility, and a
per-relation-group mean of edge energy.

The convergence readout (`_convergence_features`) implements the packet's
target-centered tension idea: it scatters edge energy and gate to destination
squares, divides to get mean incoming tension per square, scatters energy to
source squares to get outgoing pressure, and exposes the `(incoming -
outgoing)` net tension. The top-8 incoming-energy share over total incoming
energy is also reported as a saturation diagnostic.

The classifier (`LayerNorm -> Linear -> GELU -> Dropout -> Linear`) consumes
the concatenation of `[h.mean, h.amax, h.std]`, the per-block stats stack,
the convergence features, and four occupancy summaries, and produces one
BCE-compatible puzzle logit when `num_classes = 1`.

## Diagnostics

`forward` returns a dict with:

- `logits` (BCE-compatible; shape `(B,)` when `num_classes = 1`).
- `mechanism_energy`: `log1p` of the mean gated edge energy across blocks.
- `proposal_profile_strength`: mean gate across blocks (proxy for tactical
  density).
- `proposal_keyword_count`: a constant scalar carried for downstream
  reporting parity with sibling sheaf ideas; not used in training.
- `sheaf_tension`: raw mean gated edge energy across blocks.
- `ray_visibility_mean`: mean of `q_e` across edges (saturation diagnostic
  for the blocker gate).
- `gate_mean`, `edge_energy_mean`: pooled gate / energy diagnostics.
- `ray_energy`, `knight_energy`, `king_energy`, `pawn_energy`: per-relation-
  group mean edge energy from the final block.
- `convergence_tension`: mean per-square incoming gated edge energy from the
  final block.
- `defense_gap`: mean absolute net `(incoming - outgoing)` energy per square.
- `top_edge_tension`: mean of the top-k edge energies in the final block.
- `occupancy_proxy_mean`: mean of `o_v` across squares.

Diagnostics are reporting-only; they do not enter the training loss.

## Contract

- Input: `(B, C, 8, 8)` board tensor only. CRTK / verification / source /
  engine metadata is reporting-only and is not consumed.
- Output: dict with `logits` of shape `(B,)` for the one-logit
  `puzzle_binary` BCE-with-logits trainer, plus the diagnostics listed above.
- Symmetry: file-mirror tying is the only tied symmetry; pawn direction,
  castling, and side-to-move asymmetries are intentionally not enforced,
  matching the source packet's stance that chess is not D4-invariant.

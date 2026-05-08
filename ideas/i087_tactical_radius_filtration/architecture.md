# Architecture

`Tactical Radius Filtration` (TRF) implements the math thesis as a bespoke
PyTorch classifier. Scale is defined by exact tactical-radius shells over a
deterministic chess-rule contact graph, never by image resolution, dilation,
wavelet bands, residual pyramids, hypercolumns, or general square-pair
attention.

## Mechanism

The forward path follows the math thesis directly:

1. **Decode current board.** The simple_18 input tensor is parsed into pieces
   on `64` squares using the 12 piece-occupancy planes plus the side-to-move
   plane. No engine, search, or solution-move data is consumed.
2. **Square feature lift.** Each square is lifted by a shared `1 x 1` MLP on
   `[piece_planes, square_coord_features, side_to_move_feature]` to
   `H0 in R^{B x 64 x d_model}`.
3. **Build typed chess relations.** For each board, the
   `TacticalRadiusGraphBuilder` constructs deterministic boolean adjacency
   tensors for: own/opponent direct attacks, friendly defenses (own and
   enemy), first slider blockers, slider x-rays through one blocker, both
   king zones, and pawn front / promotion lanes. Pieces, pseudo-legal attack
   geometry, and ray walks stop at occupancy. A Chebyshev king-distance graph
   is provided as the A3 ablation.
4. **Relation coarsening.** Group counts shrink with radius:
   - radius 1 keeps the eight base relation groups,
   - radius 2 coarsens to attack/defense/collision/ray/king-zone/pawn-chain
     bundles,
   - radius >= 3 coarsens further to king-pressure, material-tension, escape,
     promotion, and open-line complexes.
5. **Exact-shell filtration.** Closed balls grow by Boolean matrix-multiplying
   the previous ball into the radius-r grouped relations. The exact shell is
   `Q_r = P_r AND NOT P_{r-1}` so radius `r` features only see contacts that
   first appear at exactly `r` chess-rule steps. A `shell_mode="closed_ball"`
   switch is exposed for the A2 ablation.
6. **Typed shell aggregation.** Each shell `Q_{r,g}` is row-normalized into
   `N_{r,g}` and used to mix `H0` with per-group projection weights:
   `Y_r = sum_g project_{r,g}(N_{r,g} H0) + W_{r,self} H0`, then
   `H_r = LayerNorm(GELU(Y_r))`. `H_r` is computed from `H0` and the exact
   shell `Q_r`, never from `H_{r-1}`, so the multiscale structure lives in
   the filtration rather than in network depth.
7. **Rule-zone readout.** Six deterministic masks (any piece, side-to-move
   pieces, opponent pieces, side-to-move king zone, opponent king zone,
   slider blockers) pool each `H_r` into mask-conditioned mean features. The
   pooled vector is concatenated with low-dimensional shell-count rule
   features (count of own attacks into opp king zone, opp attacks into stm
   king zone, defended attackers, x-rays into king/queen, contested squares,
   etc.) and fed through a `LayerNorm -> Linear -> GELU -> Dropout -> Linear`
   readout to a single BCE logit.
8. **Shell dropout.** During training only, entire shell groups at `r >= 1`
   are dropped with probability `shell_dropout` to break brittle reliance on
   any one relation group.

## Output Contract

Forward returns a dictionary:

- `logits`: `FloatTensor[B]`, the BCE puzzle logit for `puzzle_binary`.
- `radius_shell_counts`: per-radius mean shell mass, used for diagnostics.
- `topology_pressure`, `radius2_pressure`, `radius3_pressure`,
  `shell_count_hint`, `shell_readout_features`, `piece_pool_energy`,
  `mechanism_energy`, `proposal_profile_strength`,
  `proposal_keyword_count` — diagnostics consumed by the trainer.

## Ablations

The math thesis falsifiers are exposed as constructor flags:

- `radius` (A1): `R in {0, 1, 2, 3, 4}`.
- `shell_mode` (A2): `"exact"` vs `"closed_ball"`.
- `graph_mode` (A3): `"chess"` vs `"chebyshev"`.
- `use_xray` (A5): drop slider blocker / x-ray relations.
- `use_king_zone` (A6): drop both king-zone relations and king-zone masks.
- `use_shell_counts` (A7): drop scalar rule-field counts.
- `shell_dropout`: training-time random group dropout for robustness.

## Implementation Binding

- Registered model name: `tactical_radius_filtration`.
- Source implementation file:
  `src/chess_nn_playground/models/tactical_radius_filtration.py`.
- Idea-local wrapper: `ideas/i087_tactical_radius_filtration/model.py`.

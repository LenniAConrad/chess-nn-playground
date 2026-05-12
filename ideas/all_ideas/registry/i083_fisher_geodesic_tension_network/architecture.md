# Architecture

`Fisher-Geodesic Tension Network` (FGTN) is a board-only `puzzle_binary`
classifier whose decisive non-linearity is the **Fisher-Rao geodesic
excess** of source -> hinge -> sink categorical distributions on the
64-square simplex. It follows the markdown thesis from
`ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-28_0755_tuesday_new_york_fisher_geodesic.md`.

## Mechanism

1. **Compact residual board trunk.** A `Conv2d -> BatchNorm -> SiLU` stem
   maps the `simple_18` board tensor `(B, input_channels, 8, 8)` to a
   `(B, width, 8, 8)` feature field. `depth` `ResidualBlock` layers
   (two `Conv2d(3x3, padding=1) + BatchNorm + SiLU` with a skip
   connection) refine the trunk without ever down-sampling square
   identity. Pooled features `pooled = features.mean(dim=(2,3))` feed the
   readout and the route gate.
2. **Route distribution head.** `Conv2d(width -> routes * 3, kernel=1)`
   produces one `(B, routes, 3, 64)` tensor of logits. Softmax over the
   final 64-square axis (computed in float32) followed by
   `simplex_floor(p, eps) = (1 - 64*eps) p + eps` projects the routes
   into the open simplex `Delta^63_eps`. Per route `r`, the three slots
   are interpreted as the source distribution `p_r`, the hinge/tension
   distribution `h_r`, and the sink/criticality distribution `q_r`.
3. **Fisher-Rao geometry on the simplex.** For each route the trunk
   computes pairwise Fisher-Rao distances via the Bhattacharyya
   coefficient

   ```
   d_FR(p, q) = 2 * arccos( sum_i sqrt( p_i * q_i ) ).
   ```

   The geodesic excess and its directness ratio are

   ```
   E_r   = d_FR(p_r, h_r) + d_FR(h_r, q_r) - d_FR(p_r, q_r)
   rho_r = E_r / (d_FR(p_r, q_r) + eps).
   ```

   The optional spherical hinge angle `turn = pi - angle(v_hp, v_hq)` is
   computed from sphere-log directions in the square-root embedding when
   `use_angle=True`.
4. **Route gate and aggregation.** A `LayerNorm -> Linear` route-gate
   network turns the pooled board features into a softmax weight over
   routes (`gate.shape == (B, routes)`). The trunk concatenates the
   per-route excess, ratio, pairwise distances, optional turn, and
   gate-weighted / max scalars into a deterministic-length geometry
   feature vector `geom_feat`.
5. **Readout.** The puzzle logit comes from
   `LayerNorm -> Linear -> SiLU -> Dropout -> Linear -> SiLU -> Linear`
   over `[pooled, geom_feat]`. A separate geometry-only readout consumes
   `geom_feat` alone so the geometry-only ablation from the markdown is
   always available as `geometry_only_logits`. The main `logits` tensor
   has shape `(B,)` for the `puzzle_binary` BCE-with-logits trainer.

## Output Contract

Forward returns a dict whose `"logits"` entry is `(B,)` for the repository
`puzzle_binary` BCE-with-logits trainer. Diagnostic tensors saved to
prediction artefacts include:

- `geometry_only_logits`: `(B,)` ablation logit using only `geom_feat`.
- `route_probs`: `(B, routes, 3, 64)` softmax-floored route distributions
  (source, hinge, sink) on the categorical simplex.
- `route_excess`: `(B, routes)` Fisher-Rao geodesic excess per route.
- `direct_distance`: `(B, routes)` `d_FR(p_r, q_r)` per route.
- `route_ratio`: `(B, routes)` directness ratio
  `E_r / (d_FR(p_r, q_r) + eps)`.
- `route_gate`: `(B, routes)` softmax weight over routes derived from
  pooled board features only.
- `weighted_excess`, `max_excess`: `(B,)` gate-weighted and per-batch max
  excess summaries.
- `weighted_ratio`, `max_ratio`: `(B,)` gate-weighted and per-batch max
  directness summaries.
- `hinge_turn`, `weighted_turn`, `max_turn`: spherical hinge-angle
  diagnostics when `use_angle=True`.
- `geometry_features`: `(B, geom_dim)` concatenated geometry vector that
  feeds the readout and the geometry-only ablation.
- `fisher_geodesic_tension`, `information_surprisal`, `mechanism_energy`,
  `proposal_profile_strength`, `proposal_keyword_count`: legacy
  packet-family diagnostic aliases produced from the same geometry
  tensors so reporting parity with other packet folders is preserved.

## Leakage Guards

The forward pass consumes only the `simple_18` board tensor. The packet's
forbidden inputs (Stockfish scores, principal variations, node counts,
mate scores, best moves, verification metadata, source labels, source
identity) are never passed to the model. CRTK metadata is reporting-only.

## Implementation Binding

- Registered model name: `fisher_geodesic_tension_network`.
- Source implementation file: `src/chess_nn_playground/models/fisher_geodesic_tension.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i083_fisher_geodesic_tension_network/model.py`.

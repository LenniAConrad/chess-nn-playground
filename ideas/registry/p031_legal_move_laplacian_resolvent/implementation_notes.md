# Implementation Notes

- Central model code: `src/chess_nn_playground/models/primitives/legal_move_laplacian_resolvent.py`.
- Legal-move graph helper: `src/chess_nn_playground/models/primitives/legal_move_graph.py`.
- Idea-local wrapper: `ideas/registry/p031_legal_move_laplacian_resolvent/model.py`.
- Registry key: `legal_move_laplacian_resolvent`.
- Source primitive: `ideas/research/primitives/external_06_high_risk_legal_graph_delta_state_primitives.md`
  (Section 1, "Legal-Move Laplacian Pseudoinverse Propagation").

## Inputs

Only the `simple_18` `(B, 18, 8, 8)` current-board tensor is consumed.
The legal-move adjacency `A(x)` is built analytically:

- Piece-existence planes 0-11 give per-color piece masks.
- Side-to-move plane 12 selects the "own color".
- Blocker resolution uses the static (64, 64, 64) `between` table over the
  occupancy of all pieces (own + enemy), matching the i193 trunk's
  `DualStreamFeatureBuilder` semantics.
- Sliding-piece edges (B, R, Q) are blocked by the first occupied square
  on the ray. Knight and king edges are occlusion-free. Pawn pushes
  require the target square empty (and the intermediate square empty for
  the two-square push). Pawn captures fire on the diagonals regardless of
  occupancy -- the spec treats the "move graph" as the pseudo-legal threat
  graph, which is symmetric in this regard.
- Edges to own-color targets are dropped.

CRTK metadata, source labels, verification flags, and engine scores are
**not** consulted. The legal-move adjacency depends entirely on the
simple_18 piece planes and side-to-move plane.

## Stop-gradient contract

The legal-move adjacency, ``own_piece_mask``, and ``degree`` tensors are
computed inside ``torch.no_grad()``. Gradient flow runs entirely through:

- the i193 trunk (unchanged),
- the per-piece edge-weight vector ``piece_edge_weights``,
- the alpha logit ``alpha_logit``,
- the per-square feature projection ``square_feature_proj``,
- the mixing matrix ``theta``,
- the pool / delta / gate MLPs.

Trunk diagnostics fed into the gate are also detached so the head cannot
leak gradient back into the trunk's gate / pooling logic.

## Output dict contract

The model output is a ``dict[str, Tensor]`` following the i193 contract,
extended with:

- ``logits`` (rebound to ``base_logit + primitive_gate * primitive_delta_raw``)
- ``base_logit``
- ``primitive_delta`` (effective; zero in the ``zero_delta`` /
  ``trunk_only`` ablations)
- ``primitive_delta_raw`` (head MLP output, before gating)
- ``primitive_gate`` (sigmoid)
- ``primitive_gate_applied`` (zero in ``trunk_only``)
- ``primitive_gate_logit``
- ``primitive_gate_entropy``
- ``primitive_contribution`` (gate * delta)
- ``lmlpp_alpha`` (effective ``alpha`` per sample -- the same scalar
  broadcast over the batch)
- ``lmlpp_mean_feature_norm``
- ``lmlpp_max_feature_norm``
- ``lmlpp_degree_mean``
- ``lmlpp_neumann_terms`` (constant, broadcast over the batch)
- ``trunk_<name>`` for every diagnostic the i193 trunk produced.

All per-sample scalar tensors are emitted in the standard one-column-per-
key shape so the shared trainer copies them into ``predictions_<split>.parquet``.

## Ablation modes

See ``model.ALLOWED_ABLATIONS``. The primary falsifier is ``k1_gat_rebrand``
(K=1 collapse). ``shuffle_adjacency`` is the rule-feature falsifier.

## Why this is not a `ResearchPacketProbe` scaffold

The model is a bespoke `nn.Module` that wraps the bespoke i193
`ExchangeThenKingDualStreamNetwork` and adds:

- a per-square feature projector,
- a per-piece edge-weight parameter,
- an alpha logit,
- a mixing matrix Theta,
- pool / delta / gate MLPs.

It does not call `build_research_packet_probe_from_config`, does not
delegate to a shared CNN / MLP / NNUE / LC0 baseline builder, and has its
own forward pass. The `implementation_kind: bespoke_model` declaration is
consistent with the `audit_implementation_kinds.py` heuristics.

## Production upgrade path

The dense `(B, 64, 64)` matmuls in the Neumann series are the easy win
target for a future optimisation pass:

1. Switch the adjacency materialisation to ``torch.sparse_csr`` once the
   density-vs-kernel-utilisation tradeoff favours sparse on the deployment
   hardware.
2. Add power-iteration spectral clipping for ``alpha`` to enable larger
   ``K`` without numerical drift.
3. Optionally implement a Triton kernel for the (B, 64, 64) x (B, 64, d)
   batched matmul to fuse the K matmuls.

None of these change the model contract; they only change the resolvent
materialisation.

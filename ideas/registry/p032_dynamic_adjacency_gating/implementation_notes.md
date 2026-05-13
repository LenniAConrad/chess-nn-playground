# Implementation Notes

- Central model code: `src/chess_nn_playground/models/primitives/dynamic_adjacency_gating.py`.
- Legal-move graph helper: `src/chess_nn_playground/models/primitives/legal_move_graph.py`.
- Idea-local wrapper: `ideas/registry/p032_dynamic_adjacency_gating/model.py`.
- Registry key: `dynamic_adjacency_gating`.
- Source primitive: `ideas/research/primitives/external_25_dynamic_adjacency_rank_order_involution_gate.md`
  (Section 1, "Dynamic Adjacency-Conditioned Gating").

## Inputs

Only the `simple_18` `(B, 18, 8, 8)` current-board tensor is consumed.
The per-type legal-move masks are built analytically inside the forward
pass from the same machinery used by p031: piece planes 0-11, side-to-move
plane 12, and blocker occupancy.

CRTK metadata, source labels, verification flags, engine scores, and
principal variations are **not** consulted.

## Stop-gradient contract

The legal-move adjacency and the per-type masks are computed inside
``torch.no_grad()``. Gradient flow runs through:

- the i193 trunk (unchanged),
- ``square_feature_proj``,
- the eight per-type linear projections (``move_type_projections``),
- ``shared_projection`` (only when the ``single_move_type`` ablation is
  active),
- ``aggregator``, ``delta_head``, ``gate_head``.

The trunk diagnostics fed to ``gate_head`` are detached so the head
cannot leak gradient back into the trunk.

## Output dict contract

The output dict follows the i193 contract, extended with:

- ``logits`` (rebound to ``base_logit + gate * delta``)
- ``base_logit``
- ``primitive_delta`` / ``primitive_delta_raw``
- ``primitive_gate`` / ``primitive_gate_logit`` / ``primitive_gate_entropy``
- ``primitive_contribution`` (gate * delta)
- ``dag_total_degree`` (mean degree per sample)
- ``dag_degree_type_<code>`` for each of the 8 active move-type codes
- ``trunk_<name>`` for every diagnostic the i193 trunk produced

## Ablation modes

See ``model.ALLOWED_ABLATIONS``. The primary falsifier is
``single_move_type`` (per-type weight sharing collapse). ``soft_mask``
tests the hard-mask commitment. ``uniform_adjacency`` is the
adjacency-removal control. ``shuffle_adjacency`` is the rule-feature
falsifier.

## Why this is not a `ResearchPacketProbe` scaffold

The model is a bespoke `nn.Module` with a custom forward pass and a
unique per-move-type weight-sharing scheme. It does not call
``build_research_packet_probe_from_config`` and does not delegate to a
shared CNN / MLP / NNUE / LC0 baseline builder.

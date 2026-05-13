# Implementation Notes

- Central model code: `src/chess_nn_playground/models/primitives/sparse_legal_graph_transition.py`.
- Legal-move graph helper: `src/chess_nn_playground/models/primitives/legal_move_graph.py`.
- Idea-local wrapper: `ideas/registry/p035_sparse_legal_graph_transition/model.py`.
- Registry key: `sparse_legal_graph_transition`.
- Source primitive: `ideas/research/primitives/external_30_sparse_legal_graph_transition_delta_accumulator.md`
  (Section "primitive_sparse_transition_flow"; first-listed proposal).

## Inputs

Only the `simple_18` `(B, 18, 8, 8)` current-board tensor is consumed.
The legal-move adjacency is built analytically inside the forward pass
from the simple_18 piece planes, side-to-move plane, and blocker
occupancy.

CRTK metadata, source labels, verification flags, engine scores, and
principal variations are **not** consulted.

## Stop-gradient contract

The legal-move adjacency, ``own_piece_mask``, and ``degree`` are
computed inside ``torch.no_grad()``. Gradient flow runs through the
i193 trunk (unchanged), ``square_feature_proj``, the three edge
projections (``w_self``, ``w_neighbor``, ``w_interact``), the
``edge_norm``, the aggregator, the delta head, and the gate head.
Trunk diagnostics fed to the gate are detached.

## Output dict contract

The output dict follows the i193 contract, extended with:

- ``logits`` (rebound to ``base_logit + gate * delta``)
- ``base_logit``
- ``primitive_delta`` / ``primitive_delta_raw``
- ``primitive_gate`` / ``primitive_gate_logit`` / ``primitive_gate_entropy``
- ``primitive_contribution`` (gate * delta)
- ``slmgt_degree_mean`` (mean legal-move degree per sample)
- ``slmgt_edge_norm`` (mean L2 of the edge tensor per sample)
- ``slmgt_edge_max`` (max per-edge L2 across the board)
- ``trunk_<name>`` for every diagnostic the i193 trunk produced

## Ablation modes

See ``ALLOWED_ABLATIONS``. The primary falsifier is ``separable_phi``
(joint vs separable edge function). ``uniform_adjacency`` is the
rule-graph-removal control. ``shuffle_adjacency`` is the rule-feature
falsifier.

## Why this is not a `ResearchPacketProbe` scaffold

The model is a bespoke `nn.Module` with a joint edge function plus a
mean-aggregator GNN forward pass. It does not call
``build_research_packet_probe_from_config`` and does not delegate to a
shared CNN / MLP baseline builder.

## Production upgrade path

The pair tensor ``phi`` has shape (B, 64, 64, edge_hidden). For larger
batches the right upgrade is an explicit-edge formulation that stores
only the |E| ~ 64*27 active edges per board and uses ``index_add_`` to
aggregate. This requires a custom dataloader-side edge index tensor;
deferred until the keep-decision is in.

# Implementation Notes

- Central model code: `src/chess_nn_playground/models/primitives/move_kernel_operator.py`.
- Geometry helper: `src/chess_nn_playground/models/primitives/legal_move_graph.py`
  (the static reach masks are built from the shared geometry tables).
- Idea-local wrapper: `ideas/registry/p033_move_kernel_operator/model.py`.
- Registry key: `move_kernel_operator`.
- Source primitive: `ideas/research/primitives/external_28_sparse_differential_accumulator_move_kernel.md`
  (Section "primitive_mko").

## Inputs

Only the `simple_18` `(B, 18, 8, 8)` current-board tensor is consumed.
The move-type masks are built from chess geometry and are static (do
not depend on piece placement); per-square seed features come from the
12 piece planes plus the side-to-move plane.

CRTK metadata, source labels, verification flags, engine scores, and
principal variations are **not** consulted.

## Stop-gradient contract

The move-type masks are static buffers (built at construction time). Per-
type linear projections, the shared projection, the per-type scalars,
the aggregator, the delta head, and the gate head all carry learned
parameters with full gradient flow. Trunk diagnostics fed to the gate
are detached.

## Output dict contract

The output dict follows the i193 contract, extended with:

- ``logits`` (rebound to ``base_logit + gate * delta``)
- ``base_logit``
- ``primitive_delta`` / ``primitive_delta_raw``
- ``primitive_gate`` / ``primitive_gate_logit`` / ``primitive_gate_entropy``
- ``primitive_contribution`` (gate * delta)
- ``mko_norm_<name>`` for each of the six active move types
- ``trunk_<name>`` for every diagnostic the i193 trunk produced

## Why this is not a `ResearchPacketProbe` scaffold

The model is a bespoke `nn.Module` with its own custom forward pass and
parameter shapes (per-type linear projections + per-type scalars +
static move-type masks). It does not call
``build_research_packet_probe_from_config`` and does not delegate to a
shared CNN / MLP baseline builder.

## Production upgrade path

The current static-mask formulation is dense over the per-square output;
a future optimisation pass could exploit per-row sparsity (knight masks
have <= 8 nonzeros per row; king has <= 8; sliding has <= 14). A custom
gather-scatter kernel would yield wall-clock wins on consumer GPUs.
None of these change the model contract.

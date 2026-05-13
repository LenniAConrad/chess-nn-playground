# Implementation Notes

- Central model code: `src/chess_nn_playground/models/primitives/octilinear_selective_scan.py`.
- Idea-local wrapper: `ideas/registry/p034_octilinear_selective_scan/model.py`.
- Registry key: `octilinear_selective_scan`.
- Source primitive: `ideas/research/primitives/external_29_incremental_move_update_octilinear_scan.md`
  (Section "primitive_oss", "Final Ranking" pos. 1).

## Inputs

Only the `simple_18` `(B, 18, 8, 8)` current-board tensor is consumed.
Per-square seed features come from the 12 piece planes plus the
side-to-move plane. The scan paths are static chess-geometry buffers
registered at construction time.

CRTK metadata, source labels, verification flags, engine scores, and
principal variations are **not** consulted.

## Stop-gradient contract

The scan path indices are static buffers (long tensors). Gradient flow
runs through the i193 trunk (unchanged), the seed-feature projection,
the 8 per-direction ``A_logit_projections`` and ``b_projections``, the
``fixed_a_logits`` parameter (used only in the ``fixed_transition``
ablation), the direction fuser, the delta head, and the gate head.
Trunk diagnostics fed to the gate are detached.

## Output dict contract

The output dict follows the i193 contract, extended with:

- ``logits`` (rebound to ``base_logit + gate * delta``)
- ``base_logit``
- ``primitive_delta`` / ``primitive_delta_raw``
- ``primitive_gate`` / ``primitive_gate_logit`` / ``primitive_gate_entropy``
- ``primitive_contribution`` (gate * delta)
- ``oss_energy_<direction>`` for each of the 8 directions
- ``trunk_<name>`` for every diagnostic the i193 trunk produced

## Ablation modes

See ``ALLOWED_ABLATIONS``. The primary falsifier is
``single_direction``. ``fixed_transition`` is the selectivity
falsifier. ``shuffle_features`` is the rule-feature falsifier.

## Why this is not a `ResearchPacketProbe` scaffold

The model is a bespoke `nn.Module` with eight per-direction selective
scans and a custom scatter / fuse / pool pipeline. It does not call
``build_research_packet_probe_from_config`` and does not delegate to a
shared CNN / MLP baseline builder.

## Production upgrade path

The scan loop is currently a Python `for` over `range(8)` per
direction. The asymptotic Mamba win materialises only with a
``parallel_scan``-style kernel (Triton or CUDA). Until that lands, OSS
should be benchmarked at scout scale only.

Other potential upgrades:

- Use ``torch.compile`` to fuse the per-step Linear + Hadamard. The
  shape is fixed (B, num_tracks, 8, d) so the graph is static.
- Merge cardinal-direction pairs (E/W share scan paths reversed; N/S
  share scan paths reversed; etc.). Could halve the parameter count
  with a "bidirectional Mamba" shape.
- Add a per-track length-mask normalisation to compensate for the
  short-diagonal energy bias.

None of these change the model contract.

# Implementation Notes

- Central model code: `src/chess_nn_playground/models/primitives/reversible_delta_kernel_memory.py`.
- Idea-local wrapper: `ideas/registry/p019_reversible_delta_kernel_memory/model.py`.
- Registry key: `reversible_delta_kernel_memory`.
- Source primitive: `ideas/research/primitives/external_13_reversible_delta_kernel_occlusion_transport.md`
  (rank-1 proposal `primitive_01_reversible_delta_kernel_memory`).

## Inputs

The model only consumes the `simple_18` `(B, 18, 8, 8)` current-board
tensor. The 64 piece tokens are computed in `_build_piece_tokens` from
the 12 piece planes (one one-hot type per occupied square), the
side-to-move plane, and a 64-row square embedding. Tokens on empty
squares are masked to zero before they reach the kernel memory.

## Static vs incremental update API

The spec's primitive is defined as a stateful operator with explicit
`add(u)` / `remove(u)` events that maintain `M, z` in `O(d_h * d_v)`
time per event. The current chess-nn-playground trainer feeds one
static board per sample, so the forward pass computes `M, z` from the
full active-piece set every step. The two paths produce *bitwise-equal*
outputs: the static sum and the streaming insert series have the same
limit by construction.

The incremental API is the speed claim of the primitive, not the lift
claim. Training cost is not the bottleneck for p019 -- the static
forward is `O(64 * h * v)` per sample, comparable to one conv layer at
the same width. The incremental kernel-memory update would only matter
in an engine inference loop with sub-millisecond move/unmove tempo;
that path is deferred behind a real engine integration.

## Stop-gradient contract

- Token construction is fully differentiable (embeddings + linear
  projections + occupancy mask).
- `phi` and `nu` projections receive gradients through `M, z` and the
  query mechanism.
- The gate input is the trunk joint feature *without* an explicit
  `detach()`, so the gate gradient flows back into the trunk's
  exchange/king encoder. This matches the design of i246's
  `promotion_aware_head` gate and lets the trunk learn to recognise
  positions where the kernel-memory signal should fire.

## Output dict contract

The model output is a `dict[str, Tensor]` following the i193 contract,
extended with:

- `logits` (rebound to `base_logit + primitive_delta`)
- `base_logit`
- `primitive_delta` (`primitive_gate * primitive_delta_raw`)
- `primitive_delta_raw`
- `primitive_gate`
- `primitive_gate_logit`
- `primitive_gate_entropy`
- `rdkm_active_count`     -- number of occupied squares per sample
- `rdkm_memory_norm`      -- mean RMS magnitude of `M`
- `rdkm_z_norm`           -- mean RMS magnitude of `z`

All per-sample scalar tensors are emitted in the standard one-column-
per-key shape so the shared trainer copies them into
`predictions_<split>.parquet`.

## Ablation modes

See `ablations.md` and `model.ALLOWED_ABLATIONS`. The primary falsifier
is `shuffle_tokens`: in-batch permutation of the 64-token tensor.

## Deferred internal proposals

The source packet contains four other proposals that are *not*
implemented here:

- `primitive_02_occlusion_scanned_move_transport` -- this batch covers
  two ray formulations as p020 (`blocker_reset_ray_scan`) and p021
  (`occlusion_semiring_ray_scan`).
- `primitive_03_incremental_pair_accumulator` -- sister to p022
  (`event_delta_bilinear_accumulator`).
- `primitive_04_alternating_soft_exchange_scan` -- not in this batch.
- `primitive_05_signed_chess_orbit_norm` -- not in this batch.

## Why this is not a `ResearchPacketProbe` scaffold

The model is a bespoke `nn.Module` that wraps the bespoke i193
`ExchangeThenKingDualStreamNetwork` and adds three new linear
projections plus two MLPs. It does not call
`build_research_packet_probe_from_config`, does not delegate to a
shared CNN / MLP / NNUE / LC0 baseline builder, and has its own
forward pass. The `implementation_kind: bespoke_model` declaration is
consistent with the `audit_implementation_kinds.py` heuristics.

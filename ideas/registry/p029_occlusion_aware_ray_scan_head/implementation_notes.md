# Implementation Notes

- Central model code: `src/chess_nn_playground/models/primitives/occlusion_aware_ray_scan_head.py`.
- Idea-local wrapper: `ideas/registry/p029_occlusion_aware_ray_scan_head/model.py`.
- Registry key: `occlusion_aware_ray_scan_head`.
- Source primitive: `ideas/research/primitives/external_26_delta_update_occlusion_ray_piece_kernels.md`.
- Shared scaffolding: `src/chess_nn_playground/models/primitives/primitive_heads.py`.

## Inputs

The model consumes only the `simple_18` `(B, 18, 8, 8)` current-board
tensor. Both the per-square features and the per-(square, direction)
blocker gate are read from the 12 piece planes only.

## Forward path

1. `trunk(board)` returns the standard i193 diagnostics dict.
2. `features = feature_proj(board[:, :12])` produces a per-square
   feature stack.
3. `gate_value = sigmoid(blocker_gate(features.permute(...)))` is the
   per-(square, direction) blocker gate in (0, 1).
4. For each direction, iterate `max_ray_length` steps: shift the
   running state, multiply by the gate, add the features.
5. Project per-direction states with `direction_proj`, mean-pool over
   the board, stack to `(B, NUM_DIRECTIONS, F)`.
6. Flatten, LayerNorm, concat with trunk diagnostics, run gate /
   delta MLPs.
7. `final_logit = base_logit + sigmoid(gate_logit) *
   primitive_delta_raw`.

## Sequential scan vs parallel scan

The OARS spec frames the operator as a Mamba-style parallel selective
scan; we implement the eager sequential version because:

- The scan length is at most 7 (longest chess ray), so the loop is
  trivially small.
- Sequential gives the cleanest gradient profile under standard
  PyTorch autograd.
- The eager version validates the math before we add CUDA-kernel
  complexity.

A fused parallel-scan implementation would be the natural follow-up
once `p029` survives its falsifier.

## Cost model

| Component | Approximate cost |
|---|---|
| `feature_proj` | 12 → `feature_dim` 1x1 conv on (B, 12, 8, 8) |
| `blocker_gate` | `feature_dim → NUM_DIRECTIONS` linear on (B*64, F) |
| Scan loop | `NUM_DIRECTIONS * max_ray_length` shift + multiply + add |
| `direction_proj` | `feature_dim → feature_dim` linear |
| Fusion MLP | `(NUM_DIRECTIONS * feature_dim + 4) → head_hidden_dim → 1` twice |

At default `feature_dim=16`, the head is comparable to `p026` RayPool
in cost; the dominant difference is the per-step blocker gate matmul.

## Diagnostics surface area

The forward dict adds the following OARS-specific diagnostics on top
of the standard primitive diagnostics:

- `oars_mean_blocker_gate` — average value of the blocker gate
  across squares and directions per sample.
- `oars_dir_energy_mean` — mean ray-feature L2 across 8 directions.
- `oars_dir_energy_max` — peak directional energy per sample.

## Deferred work

- True state-dependent blocker gate (the current implementation reads
  the gate from the raw features for stability; the spec calls for
  reading it from the running state).
- Mamba-style parallel selective scan for batched training.
- Multi-step state with a learned A matrix (the natural cross-over
  to `p030` Ray-SSM).

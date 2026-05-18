# Implementation Notes

- Central model code: `src/chess_nn_playground/models/primitives/efficient_ray_occlusion_scan.py`.
- Shared helpers:
  - `RayGeometry`, `NUM_DIRECTIONS`, `RAY_MAX_LEN`, `SQUARES` from
    `src/chess_nn_playground/models/primitives/ray_geometry.py`.
  - `trunk_joint_features` from
    `src/chess_nn_playground/models/primitives/trunk_features.py`.
- Idea-local wrapper: `ideas/registry/p054_efficient_ray_occlusion_scan/model.py`.
- Registry key: `efficient_ray_occlusion_scan`.
- Source primitive: `ideas/research/primitives/external_49_efficient_ray_occlusion_scan_primitive.md`.

## Inputs

Only the `simple_18` `(B, 18, 8, 8)` current-board tensor is consumed.
The 12 piece-presence planes are split into 6 us / 6 them one-hots; the
occupancy mask is the clamped sum of all 12 planes.

CRTK metadata, source labels, verification flags, engine scores, and
principal variations are **not** consulted.

## Tensor layout

The scan body uses the layout described in the source markdown:

- Geometry buffers `ray_step_index (8, 64, 7)` and `ray_step_mask
  (8, 64, 7)` are registered as non-persistent buffers via
  `RayGeometry.build()`.
- The gathered tensors are `occ_ray (B, 8, 64, 7)` and `feat_ray
  (B, 8, 64, 7, 16)`. The per-square 16-channel `feat` is composed
  inline from the piece planes; no learnable parameter sits inside
  the scan.
- `k = occ_ray.cumsum(dim=-1)` and three equality tests
  (`k_prev == 0`, `k == 1`, `k == 2`, `k_prev == 1`) produce the four
  hard 0/1 masks (`visible`, `first`, `second`, `xray_lane`).
- `first_feat`, `second_feat` are masked reductions over `feat_ray`.

## Stop-gradient contract

The scan body has no learnable parameters and is fully differentiable
through `cumsum`, `gather`, and pointwise ops. Trunk diagnostics fed
to the gate are *not* detached because the gate path needs them; the
joint pool feature is the same one used by p020 / p021 / p046 and
participates in the delta head co-training.

## Output dict contract

The output dict follows the i193 contract, extended with:

- `logits` (rebound to `base_logit + gate * delta`)
- `base_logit`
- `primitive_delta` / `primitive_delta_raw`
- `primitive_gate` / `primitive_gate_applied` / `primitive_gate_logit`
  / `primitive_gate_entropy`
- `primitive_contribution`
- `eros_occupancy_density` -- per-sample mean occupancy
- `eros_mobility_mean` -- mean mobility length over `(direction, source)`
- `eros_xray_pressure_mean` -- mean x-ray pressure over `(direction, source)`
- `eros_visible_density` -- (visible_steps_sum + xray_lane_steps_sum)
  / (D * S * L), a bounded geometry signal used as the
  `mechanism_energy` augmentation
- `eros_first_blocker_rate` -- mean `first_exists` over `(direction, source)`
- `eros_second_blocker_rate` -- mean `second_exists` over `(direction, source)`
- `trunk_<name>` for every diagnostic the i193 trunk produced
- `mechanism_energy` augmented with `eros_visible_density.detach()`
- `proposal_profile_strength` = `|delta| * gate_entropy` (clamped to
  `[0, 20]`)

## Ablation modes

See `ALLOWED_ABLATIONS`. Primary falsifier is `first_only` (drop the
second-blocker / x-ray / discovered / pin channels). Secondary
falsifier is `no_blocker_id` (zero side / value identity channels).
Mask-rule falsifiers are `uniform_occupancy`, `empty_occupancy`,
`shuffle_occupancy`. The gate-load test is `disable_gate`; the
baseline-recovery tests are `zero_delta` / `trunk_only`.

## Numerical notes

- All scan masks are 0 / 1 floats; `cumsum` is over the step axis
  only. No log / exp transform is required.
- `mobility_len = sum_l visible * (1 - occ_ray)` excludes the blocker
  cell itself; `visible_count = sum_l visible` includes it.
- Off-board steps are zeroed via `ray_mask` before the cumsum, so the
  inclusive prefix only counts on-board occupancies.
- `xray_pressure = second_exists * second_value` is bounded by the
  fixed piece values (P=1..K=200) and the binary `second_exists`
  scalar, so it is non-negative.

## Deferred / production notes

- The dense-edge scatter output (`rook_visible`, `bishop_visible`,
  `rook_xray`, `bishop_xray`, `queen_*`) described in the source
  markdown is **deferred**. The implemented primitive operates in the
  compact direction-major layout only. If the future i018 integration
  path needs dense `(B, 64, 64)` edges, add a `return_dense_edges`
  flag and a final `scatter_add_` exactly as outlined in the
  markdown's "PyTorch pseudocode" section.
- A `torch.compile`-wrapped variant of `ray_occlusion_scan` and a
  Triton lowering are mentioned in the source markdown as production
  upgrade paths. Defer behind a benchmark script
  (`scripts/benchmarks/benchmark_ray_occlusion_scan.py`) that compares
  steady-state timing of EROS against p020 / p021 / p026 and the i018
  visibility builder.
- The piece-value table is hard-coded inside the module
  (`US_PIECE_VALUES = THEM_PIECE_VALUES = (1, 3, 3, 5, 9, 200)`). If
  the value scale needs to be learnable, expose it as a parameter
  buffer with `requires_grad=False` and add a config knob; the current
  implementation treats it as a rule-derived constant.

# Architecture

`Blocker-Reset Ray Scan` (p020) is an additive, gated head over the
existing i193 `ExchangeThenKingDualStreamNetwork` trunk. The thesis
(see `math_thesis.md`) is that the i193 trunk's `3x3` convolutions
underfit long-range sliding-piece geometry because their receptive
field grows by depth, not by occupancy. p020 supplies a per-square,
per-direction ray scan whose segment depth is set by the position's
own blockers.

## Mechanism

1. **i193 trunk forward**. The bespoke trunk runs unchanged and emits
   `base_logit` and the joint dual-stream pool feature.

2. **Per-square token construction**. The 12 piece planes plus the
   side-to-move plane are flattened to `(B, 64, 13)` and projected to
   `x_s in R^{B, 64, token_dim}` by a single learned linear layer.

3. **Ray-step gather**. The shared `RayGeometry` lookup
   (`ray_step_index, ray_step_mask` from
   `models/primitives/ray_geometry.py`) maps each `(direction, source
   square, step)` to the visited square and a validity mask. Source
   step (`l = 0`) is prepended so the recurrence starts with the
   source token.

4. **Multi-ray segmented scan**. For each direction `d`, walk along
   the ray:

   ```
   h <- U x_s + (1 - O_s) (.) sigma(lambda_d) (.) h
   ```

   `lambda_d in R^h` is a per-direction learnable parameter; `sigma`
   keeps it in `(0, 1)^h`. The recurrence variable `h` is
   per-(batch, direction, source square), accumulated across the
   `RAY_MAX_LEN + 1 = 8` steps. The mean-pooled hidden state across
   valid steps gives the ray output.

5. **Readout**. Project each ray output through `V` (hidden -> token),
   mean-pool across the 64 squares, concatenate the 8 directions to
   a `(B, 8 * token_dim)` readout vector, and feed through a LayerNorm
   + GELU MLP to produce `primitive_delta_raw`.

6. **Gate**. A second LayerNorm + GELU MLP on the i193 joint feature
   produces a sigmoid gate. The final delta is
   `primitive_delta = gate * primitive_delta_raw`, with
   `gate_init = -2.0` so the head starts effectively closed.

7. **Logit fusion**. `final_logit = base_logit + primitive_delta`.

## Ablations

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `zero_blocker` | Replace `(1 - O)` with `1`, i.e. ignore blockers and run a full-length scan. **The primary falsifier.** If A1 matches `none`, the blocker reset is not load-bearing. |
| A2 | `uniform_blocker` | Replace `(1 - O)` with `0` -- the scan only sees the source token. Tests that the recurrence depth carries signal at all. |
| A3 | `zero_delta` | Hold `primitive_delta = 0`. Recovers i193. |
| A4 | `trunk_only` | Strongest control. |

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
Stockfish scores, principal variations, and any report-only metadata
are *not* consumed.

## Cost

| Stage | Per-sample cost |
|---|---|
| i193 trunk forward | One pass through the dual-stream encoder |
| Token projection | `O(64 * token_dim * 13)` |
| Ray gather | `O(8 * 64 * 7 * token_dim)` |
| Multi-ray scan | `O(8 * 64 * 7 * hidden_dim)` |
| Readout + head | LayerNorm + GELU MLPs over `8 * token_dim` |

The Python-side scan loop runs `RAY_MAX_LEN + 1 = 8` iterations and is
the dominant per-sample cost in the head. A fused CUDA / Triton
segmented-scan kernel is the production speed path (deferred).

## Implementation Binding

- Registered model name: `blocker_reset_ray_scan`.
- Source implementation: `src/chess_nn_playground/models/primitives/blocker_reset_ray_scan.py`.
- Shared ray geometry: `src/chess_nn_playground/models/primitives/ray_geometry.py`.
- Idea-local wrapper: `ideas/registry/p020_blocker_reset_ray_scan/model.py`.
- Training config: `ideas/registry/p020_blocker_reset_ray_scan/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`.

# Architecture — p010 Ray-Occlusion Semiring Scan

p010 is an additive, gated head on top of the i193 trunk; the trunk is
unmodified.

## Mechanism

1. **i193 trunk forward** — unchanged.
2. **Per-square token tower** — Conv1x1 → GELU → Conv1x1 → LayerNorm
   gives `(B, 64, d)`.
3. **Ray-step gather + transmittance** —
   `compute_ray_transmittance(board)` returns `(B, 64, 8, 7)` per-step
   transmittance computed by log-domain prefix sums over
   `(1 - occupancy)`. The geometry tables
   (`ray_step_target`, `ray_step_valid`) live in
   `rule_graph_features.RuleGeometry`.
4. **Weighted directional scan** — for each of 8 directions, weight
   the gathered ray tokens by `transmittance * λ_δ^k * ray_valid` and
   sum over steps to produce a per-direction `(B, 64, d)` vector.
5. **Per-direction linear** — eight `nn.Linear(d, ray_dim)` modules
   (or one shared under `constant_direction`) map each direction's
   sum to `(B, 64, ray_dim)`. The 8 outputs are concatenated.
6. **Delta + gate** — mean over squares, two-layer MLP to scalar
   delta. Gate MLP runs on detached i193 trunk pool.

```text
final_logit = base_logit + primitive_gate * primitive_delta_raw
```

## Inputs not used

CRTK metadata, source labels, verification flags, Stockfish scores, PVs,
and report-only metadata are not consumed. Occupancy is the only board-
derived input; the ray-step geometry tables are content-independent.

## Cost

| Stage | Per-sample cost |
|---|---|
| i193 trunk | One forward pass |
| Token tower | Two Conv1x1 |
| Transmittance | Log-prefix sum over `(B, 64, 8, 7)` |
| Ray gather | `(B, 64, 8, 7, d)` index_select |
| Per-direction linear | 8 × `Linear(d, ray_dim)` |
| Heads | Two small MLPs |

## Implementation Binding

- Registered model name: `ray_occlusion_semiring_scan`.
- Source implementation:
  `src/chess_nn_playground/models/primitives/ray_occlusion_semiring_scan.py`.
- Shared rule-graph helpers (ray geometry + transmittance):
  `src/chess_nn_playground/models/primitives/rule_graph_features.py`.
- Idea-local wrapper:
  `ideas/registry/p010_ray_occlusion_semiring_scan/model.py`.
- Builder entry in `src/chess_nn_playground/models/registry.py`.

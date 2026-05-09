# Architecture

`Pivot Trace Elimination Network` encodes a chess position into a small
square matrix `M(x) in R^{K x K}` over learned group summaries, then runs
a fixed-order differentiable Gaussian elimination and classifies the
puzzle logit from the resulting pivot trace.

- Mechanism family: `linear_algebra`.
- Input: board tensor only (`simple_18`); CRTK / source metadata is
  reporting-only.
- Board trunk: compact convolutional square encoder (`BoardConvStem`)
  over the configured input planes.
- Group summaries: `K = 12` semantic groups (us-pawn, us-minor, us-major,
  us-king-region, them-pawn, them-minor, them-major, them-king-region,
  ranks-line, files-line, center, edge). Each group is a soft-mask
  weighted average of trunk features, mass-normalised so empty groups
  remain stable. The us/them split uses the side-to-move plane and the
  king-region masks come from a 3x3 dilation of each side's king plane.
- Bilinear matrix: `M_ij = (W_L g_i)^T (W_R g_j)`, symmetrised, then
  stabilised by `M += lambda I` with `lambda = 0.1`. No learned pivoting
  is used (per the packet's implementation note); the elimination order
  is the canonical group order.
- Fixed-order elimination: for `t = 0, ..., K-1`,

  ```
  pivot_t       = softplus(M_tt) + eps
  log_pivot_t   = log(pivot_t)
  row_update_t  = M_{t+1:, t} / pivot_t
  M_{t+1:, t+1:} -= row_update_t outer M_{t, t+1:}
  update_norm_t = mean(|row_update_t|)
  residual_t    = ||M_{t+1:, t+1:}||_F / sqrt(K - t - 1 + 1)
  cond_t        = log(running_max_pivot / running_min_pivot)
  ```
- Head: a two-layer MLP over the concatenation of
  `log_pivots`, `update_norms`, `residual_norms`, `cond_ratio`,
  `final_residual`, the log-determinant `sum(log_pivots)`, and the
  per-group masses emits one puzzle logit.
- Diagnostic outputs: `logits`, `log_pivots`, `update_norms`,
  `residual_norms`, `cond_ratio`, `final_residual`, `log_determinant`,
  `matrix`, `group_summaries`, `group_masses`, plus per-batch
  `ablation_*` flags.

## Implementation Binding

- Registered model name: `pivot_trace_elimination_network`.
- Source implementation: `src/chess_nn_playground/models/pivot_trace_elimination_network.py`.
- Idea-local wrapper: `ideas/i142_pivot_trace_elimination_network/model.py`.

The wrapper calls
`build_pivot_trace_elimination_network_from_config` to instantiate the
bespoke `PivotTraceEliminationNetwork` `nn.Module`. Registry build via
`build_model("pivot_trace_elimination_network", ...)` returns the same
class.

## Ablations

The packet's central ablations are exposed via `model.ablation`:

| Name | Effect |
|------|--------|
| `none` | Full architecture above. |
| `raw_matrix_pool` | Pool the constructed `M` directly (skip elimination). |
| `random_elimination_order` | Eliminate under a fixed but non-canonical group permutation (semantic order should matter). |
| `diagonal_matrix_only` | Zero off-diagonal entries before elimination. |
| `determinant_only` | Use only `sum(log_pivots)` (= log-det of `M`) plus group masses. |
| `matrix_pencil_control` | Replace the pivot trace with the symmetric eigenvalues of `M` (matrix-pencil baseline). |

# Architecture

`Ray State-Space Scan Network` realises the line-memory thesis from `math_thesis.md`. Every rank, file, diagonal, and anti-diagonal of the 8x8 board is treated as a short token sequence and processed by a shared linear state-space recurrence with line-type-conditioned parameters. The model returns one puzzle logit plus per-line scan diagnostics.

## Square Encoder

A compact convolutional trunk over the `simple_18` planes produces a `(B, channels, 8, 8)` board feature map. A 1x1 projection emits a `(B, 64, square_dim)` set of square tokens that feed the line scans, while the unprojected map is pooled (mean+max) and fed to the classifier as a global board summary.

## Line Bank

A precomputed buffer enumerates the 38 chess rays as padded token sequences of length up to 8:

- 8 ranks
- 8 files
- 15 diagonals (north-east / south-west)
- 15 anti-diagonals (north-west / south-east)

Each entry stores forward-order positions, reversed positions, a validity mask, and the line type id. The buffer is registered as a non-persistent state so the scan operates on the right square indices for any batch.

## Bidirectional Linear State-Space Scan

The recurrence implements a linear state-space update with line-type-conditioned matrices. For step `t` along line type `r` with token `u_t`:

- `h_t = A_r * h_{t-1} + B_r * u_t + b^h_r`
- `y_t = C_r * h_t + D_r * u_t + b^y_r`

`A_r` is parameterised through a `tanh`-bounded transition (`0.75 * tanh(W_r)`) so the scan stays contractive without explicit eigenvalue clipping; the diagonal of `W_r` is initialised at `0.7` for a slow forgetting prior. Padding tokens preserve the previous hidden state via a mask, so short rays (e.g. corner diagonals of length 2) do not corrupt downstream pooling. The same parameters are applied on the reversed sequence, so each square gets a forward and a backward summary.

## Line Pooling And Diagnostics

Forward and backward outputs are concatenated into per-token line features. The forward pass produces:

- mean and max pools over all valid line tokens (`line_mean`, `line_max`)
- final-state mean across forward and backward endpoints (`endpoint_mean`)
- per-line linear response `r(line) = w^T [y_fwd; y_bwd]`
- per-line-type mean response and energy (`rank/file/diagonal/anti_diagonal_scan_response/energy`)
- `topk_line_response` (mean of the four strongest line responses)
- `king_line_response` (mean response over rays touching either king)
- aggregate `line_state_energy` and `endpoint_state_norm`

These join the global board mean+max pool and are fed to a small two-layer GELU MLP that emits the binary puzzle logit. The model accepts the repo board tensor contract (`(B, 18, 8, 8)`) and returns a dict with `logits` of shape `(B,)` plus the diagnostics above.

## Implementation Binding

- Registered model name: `ray_state_space_scan_network`
- Source implementation file: `src/chess_nn_playground/models/ray_state_space_scan.py`
- Idea-local wrapper: `ideas/i125_ray_state_space_scan_network/model.py`

The wrapper imports `RayStateSpaceScanNetwork` and `build_ray_state_space_scan_network_from_config` and delegates `build_model_from_config` to that builder. The shared `ResearchPacketProbe` scaffold is no longer used.

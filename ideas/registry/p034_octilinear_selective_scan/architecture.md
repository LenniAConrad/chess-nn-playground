# Architecture

`Octilinear Selective Scan` (p034, OSS) is an additive, gated head on
top of the i193 `ExchangeThenKingDualStreamNetwork` trunk. The thesis
(see `math_thesis.md`) is that a Mamba-style selective scan ordered by
chess ray geometry captures long-range piece coordination that the
1-hop legal-mask primitives (p032 DAG, p035 SLMGT) cannot reach in a
single layer.

The model consumes `simple_18` `(B, 18, 8, 8)` and returns one puzzle
logit plus a per-sample diagnostics dict.

## Forward pass

1. **i193 trunk forward**. Emits ``base_logit`` and trunk diagnostics.
2. **Per-square seed feature**. ``X = Linear(13)(piece_planes + stm)``
   shape ``(B, 64, feature_dim)``. Default ``feature_dim = 16``.
3. **Per-direction selective scan**. For each direction ``k`` in
   {E, W, N, S, NE, NW, SE, SW}:
   - Gather the seed features along the precomputed scan paths
     (``(num_tracks, max_len)`` tables stored as buffers).
   - Compute ``A_k = sigmoid(W^A_k @ X)`` and ``B_k = W^B_k @ X``.
   - Run the channelwise SSM ``h_t = A_k * h_{t-1} + B_k * X_t`` along
     the path.
   - Zero out padding positions (variable-length diagonals).
   - Scatter the per-step states back to the (B, 64, feature_dim)
     direction output.
4. **Direction fusion**. Concatenate the 8 direction outputs to
   ``(B, 64, 8 * feature_dim)``. Project through ``LayerNorm +
   Linear + GELU`` to ``head_hidden_dim``.
5. **Pool**. Concatenate own-piece-weighted mean and global mean.
6. **Delta head**. Two-layer MLP -> scalar ``primitive_delta_raw``.
7. **Gate**. MLP over trunk diagnostics + per-direction energy ->
   sigmoid ``primitive_gate``.
8. **Output**. ``final_logit = base_logit + primitive_gate *
   primitive_delta_raw``.

## Ablation modes

| `model.ablation` | What it tests |
|---|---|
| `none` | Full OSS architecture (default). |
| `fixed_transition` | ``A_k`` becomes a data-independent learned parameter (not produced from ``X``). Tests whether data-dependent selectivity is load-bearing. |
| `single_direction` | Run only the E direction; zero the other seven. Tests whether the 8-direction decomposition is load-bearing. |
| `shuffle_features` | In-batch permutation of seed features. Decouples rule features from position. |
| `zero_delta` | Zero primitive delta. Recovers i193 baseline. |
| `trunk_only` | Same as `zero_delta`; semantic alias. |
| `disable_gate` | Pin gate at 1.0. Tests gate load-bearing. |

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
and principal variations are **not** consumed by the model. The scan
paths are static chess-geometry tables; piece presence enters via the
seed feature only.

## Cost

| Stage | Cost |
|---|---|
| i193 trunk | One forward pass through the dual-stream encoder |
| Seed projection | One Linear(13, d) over (B, 64, 13) |
| Per-direction scan | 8 sequential scans, each up to 8 steps; per step is a Linear(d, d) + Hadamard |
| Fuser | LayerNorm + Linear + GELU on (B, 64, 8d) |
| Head / gate | Small MLPs |

The scan loop is implemented in pure Python; the asymptotic Mamba
parallel-scan win is not realised without a Triton kernel.

## Implementation Binding

- Registered model name: `octilinear_selective_scan`.
- Source implementation: `src/chess_nn_playground/models/primitives/octilinear_selective_scan.py`.
- Trunk source: `src/chess_nn_playground/models/trunk/exchange_then_king_dual_stream.py`.
- Idea-local wrapper: `ideas/registry/p034_octilinear_selective_scan/model.py`.
- Training config: `ideas/registry/p034_octilinear_selective_scan/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`:
  `MODEL_BUILDERS["octilinear_selective_scan"] = build_octilinear_selective_scan_from_config`.

# Architecture

`Reversible Delta Kernel Memory` (p019) is an additive, gated head over
the existing i193 `ExchangeThenKingDualStreamNetwork` trunk. The thesis
(see `math_thesis.md`) is that the i193 trunk underfits piece-piece
interaction patterns (king-piece distance, pinned-piece-plus-pinner,
defender-plus-target) because it has to learn them through conv-only
mixing layers. p019 supplies an explicit kernel-attention memory over
the active piece set and produces an additive logit delta.

The model consumes the repository `simple_18` current-board tensor
`(B, 18, 8, 8)` and returns one puzzle logit, plus a per-sample
diagnostic dict mirroring the i248 contract.

## Mechanism

1. **i193 trunk forward**. The bespoke
   `ExchangeThenKingDualStreamNetwork` runs unchanged and emits the
   full diagnostic dict including `logits` (treated here as
   `base_logit`) and the dual-stream joint feature.

2. **Active-piece token construction**. For each square `s in 0..63`,
   we compute one token:

   ```
   u_s = piece_type_embed(t_s) + side_embed(c_s) + square_embed(s)
   ```

   with `t_s in {P, N, B, R, Q, K, empty}` and `c_s in {white, black,
   empty}`. Empty-square tokens are multiplied by an occupancy mask so
   they do not contribute to the memory.

3. **Kernel memory**. Apply `phi = elu + 1` and a value projection
   `nu` to obtain `phi_s = phi(W_phi u_s)` and `nu_s = W_nu u_s`. Sum
   over occupied squares to obtain the memory state:

   ```
   M = sum_s phi_s nu_s^T,    z = sum_s phi_s
   ```

   The shapes are `M in R^{B, h, v}` and `z in R^{B, h}`.

4. **Trunk-conditioned query**. Build `Q in R^{B, K, h}` by applying a
   linear projection to the i193 joint feature, then `phi`. The
   `num_queries = K` queries are distinct so the head can attend to
   different "kernel slots". Each query reads the memory:

   ```
   y_q = (phi(q)^T M) / (phi(q)^T z + epsilon)
   ```

5. **Readout**. Flatten `(K * v)` and feed through a small LayerNorm +
   GELU MLP that produces `primitive_delta_raw`.

6. **Gate**. A second LayerNorm + GELU MLP on the trunk joint feature
   produces a sigmoid gate. The final delta is
   `primitive_delta = gate * primitive_delta_raw`. The gate weights
   are initialised so the head starts effectively closed
   (`gate_init = -2.0`).

7. **Logit fusion**. `final_logit = base_logit + primitive_delta`.

## Ablations

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `shuffle_tokens` | In-batch permutation of the per-square token tensor. The primary falsifier: if A1 matches `none`, the kernel memory carries no signal. |
| A2 | `zero_memory` | Force `M = 0` and `z = 0`. Tests whether the readout learns from queries alone (it should not, structurally). |
| A3 | `uniform_query` | Replace the trunk-derived queries with a uniform tensor. Tests whether the query routing is load-bearing. |
| A4 | `zero_delta` | Hold `primitive_delta = 0`. Recovers i193. |
| A5 | `trunk_only` | Strongest control: zero delta and disable the head. |

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
Stockfish scores, principal variations, and any report-only metadata
are *not* consumed by the model. The active-piece set is rule-derived
from the simple_18 board tensor.

## Cost

| Stage | Per-sample cost |
|---|---|
| i193 trunk forward | One pass through the dual-stream encoder |
| Token construction | `O(64 * token_dim)` |
| Kernel memory `M, z` | `O(64 * memory_heads * memory_value_dim)` |
| Per-query readout | `O(num_queries * memory_heads * memory_value_dim)` |
| Head MLPs | Two small LayerNorm + GELU MLPs |

The static forward is `O(64 * h * v)` per sample, comparable to a
single conv layer at the same width.

## Implementation Binding

- Registered model name: `reversible_delta_kernel_memory`.
- Source implementation: `src/chess_nn_playground/models/primitives/reversible_delta_kernel_memory.py`.
- Idea-local wrapper: `ideas/registry/p019_reversible_delta_kernel_memory/model.py`.
- Training config: `ideas/registry/p019_reversible_delta_kernel_memory/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`.

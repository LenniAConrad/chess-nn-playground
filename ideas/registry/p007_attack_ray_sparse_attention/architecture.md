# Architecture — p007 Attack-Ray Sparse Attention

ARSA is an additive, gated head on top of the i193
`ExchangeThenKingDualStreamNetwork` trunk. The trunk is unmodified.

## Mechanism

1. **i193 trunk forward** — unchanged, emits the i193 diagnostics.
2. **Per-square token tower** — small 1x1 Conv → GELU → 1x1 Conv →
   LayerNorm projects `(B, 18, 8, 8)` to `(B, 64, d)`.
3. **First-blocker lookup** — for every square and each of 8 ray
   directions, `first_blocker_indices` returns the index of the first
   occupied square along the ray. Slots without a blocker are masked.
   A self-edge is appended as slot 8 so the softmax denominator stays
   well-conditioned even when every ray is empty.
4. **Sparse softmax attention** — Q/K/V linear projections produce
   query, key, and value tensors of shape `(B, 64, K, attn_dim/d)` (with
   `K = 9`). Per-slot direction biases are added before softmax.
5. **Pool + delta + gate** — the attended `(B, 64, d)` tensor is mean-
   pooled over squares and projected to the scalar
   `primitive_delta_raw`. The gate runs on the i193 trunk joint pool
   feature (detached) and sigmoids to `primitive_gate`. Final logit:

   ```text
   final_logit = base_logit + primitive_gate * primitive_delta_raw
   ```

## Inputs not used

CRTK metadata, source labels, verification flags, Stockfish scores,
PVs, and report-only metadata are not consumed. The ray-cast key index
is derived from the simple_18 piece planes (occupancy) and the
precomputed `ray_step_target` table.

## Cost

| Stage | Per-sample cost |
|---|---|
| i193 trunk | One forward pass through the dual-stream encoder |
| Token tower | Two 1x1 convs over 18 input channels |
| Ray gather | 9-slot lookup per square (16-bit indices) |
| Attention | `O(64 · K · attn_dim)` with `K = 9` |

ARSA is intentionally cheaper than the i243 chess-decomposed attention
block it would replace in a hybrid; in this head it runs as a side
branch and adds <30% wall-clock to i193 at default widths.

## Implementation Binding

- Registered model name: `attack_ray_sparse_attention`.
- Source implementation:
  `src/chess_nn_playground/models/primitives/attack_ray_sparse_attention.py`.
- Shared rule-graph helpers (first-blocker lookup):
  `src/chess_nn_playground/models/primitives/rule_graph_features.py`.
- Idea-local wrapper:
  `ideas/registry/p007_attack_ray_sparse_attention/model.py`.
- Builder entry in `src/chess_nn_playground/models/registry.py`:
  `MODEL_BUILDERS["attack_ray_sparse_attention"] = build_attack_ray_sparse_attention_from_config`.

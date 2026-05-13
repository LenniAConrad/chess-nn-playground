# Architecture

`Regret Saddlepoint Network` (p002) is an additive, gated head on top
of the existing i193 `ExchangeThenKingDualStreamNetwork` trunk. The
trunk runs once per board; the head adds candidate + reply token
compilers, a learned bilinear payoff table, the entropy-regularized
saddle solver, and a small fusion MLP.

The model consumes the repository `simple_18` current-board tensor
`(B, 18, 8, 8)` and returns one puzzle logit, plus a rich per-sample
diagnostics dict.

## Mechanism

1. **i193 trunk pass.** ``run_i193_trunk_with_spatial`` runs the
   trunk once, returning the base logit, the i193 diagnostic dict,
   the joint pool feature `p_trunk`, and the spatial map
   `H = cat(ex_h, kg_h)` of shape `(B, 2 * channels, 8, 8)`.

2. **Candidate / reply compilers.** Two independent
   `BoardTokenAttention` pools (default 16 candidates, 12 replies,
   token_dim = 48) attend over the 64 spatial cells of `H` to produce
   candidate tokens `c_k in R^{B x K x D}` and reply tokens
   `r_j in R^{B x R x D}`.

3. **Payoff table.** Each token set is linearly projected, then a
   bilinear inner product produces the payoff table:

   ```
   A = scale * (cand_proj(c) @ reply_proj(r)^T) + bias(p_trunk)
   ```

   with `scale = tanh(payoff_scale)` (init at 0 so `A` starts at the
   batch-broadcast bias, keeping the saddle solver stable early in
   training).

4. **RSP operator.** ``regret_saddlepoint`` runs 24 damped soft
   saddle iterations and returns value, both equilibrium strategies,
   attacker / defender regrets, exploitability, and per-side
   entropies. The fallback `pure_max_min` ablation bypasses the
   solver and uses the raw `max_i min_j A_ij` saddle.

5. **Fusion head.** The attacker-pooled candidate token
   `att_pool = p^T c` and defender-pooled reply token
   `def_pool = q^T r` are concatenated with the trunk pool feature
   and the five scalar diagnostics, then fed through a LayerNorm +
   GELU MLP that produces `primitive_delta_raw`. A second MLP over
   `[p_trunk, value, exploitability, attacker_entropy]` produces the
   gate logit. Final contract:

   ```
   primitive_delta = primitive_gate * primitive_delta_raw
   final_logit     = base_logit + primitive_delta
   ```

   The gate bias is initialised at `-2.0` so the primitive starts as
   a near-no-op.

6. **Ablations.** Eight supported modes via `model.ablation`:

   - `none`: full architecture.
   - `row_shuffle_payoff`: permute payoff rows; destroys
     candidate-side game structure but preserves entry distribution.
   - `col_shuffle_payoff`: permute payoff columns; destroys
     reply-side game structure.
   - `uniform_payoff`: collapse the table to per-batch mean —
     completely removes game structure.
   - `pure_max_min`: bypass the solver and use the raw
     `max_i min_j A_ij` saddle; tests whether the regularized solver
     is load-bearing.
   - `disable_gate`: hold the gate at 1.0.
   - `zero_delta`: hold `primitive_delta` at 0.
   - `trunk_only`: zero out features and delta.

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
Stockfish scores, principal variations, and any report-only metadata
are *not* consumed.

## Cost

| Stage | Per-sample cost |
|---|---|
| i193 trunk | One forward pass through the dual-stream encoder |
| Two BoardTokenAttention pools | `O((K + R) * HW * D)` |
| Bilinear payoff | `O(K * R * D)` |
| RSP solver | `O(T * K * R)` |
| Fusion MLPs | small |

For `K = 16`, `R = 12`, `T = 24` the solver does `~4608` ops per
board, dwarfed by the trunk forward.

## Implementation Binding

- Registered model name: `regret_saddlepoint_network`.
- Source implementation:
  `src/chess_nn_playground/models/primitives/regret_saddlepoint_network.py`.
- Shared candidate/reply utilities:
  `src/chess_nn_playground/models/primitives/codex_reply_primitives.py`.
- Idea-local wrapper:
  `ideas/registry/p002_regret_saddlepoint/model.py`.
- Training config:
  `ideas/registry/p002_regret_saddlepoint/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`:
  `MODEL_BUILDERS["regret_saddlepoint_network"] = build_regret_saddlepoint_network_from_config`.

# Architecture

`Reply Channel Capacity Network` (p003) is an additive, gated head on
top of the existing i193 `ExchangeThenKingDualStreamNetwork` trunk.
The trunk runs once per board; the head adds candidate + reply token
compilers, a learned bilinear reply-logit table, the Blahut-Arimoto
channel-capacity reducer, and a small fusion MLP.

The model consumes the repository `simple_18` current-board tensor
`(B, 18, 8, 8)` and returns one puzzle logit, plus a rich per-sample
diagnostics dict.

## Mechanism

1. **i193 trunk pass.** ``run_i193_trunk_with_spatial`` returns the
   base logit, the i193 diagnostic dict, the joint pool feature
   `p_trunk`, and the spatial map `H = cat(ex_h, kg_h)`.

2. **Candidate / reply compilers.** Two independent
   `BoardTokenAttention` pools (default 16 candidates, 12 replies,
   token_dim = 48) attend over the 64 spatial cells of `H` to produce
   candidate tokens `c_k` and reply tokens `r_j`.

3. **Reply logit table.** A bilinear inner product over linear
   projections of the two token sets produces the table:

   ```
   L = (1 + tanh(logit_scale)) * (cand_proj(c) @ reply_proj(r)^T)
   ```

   Initialised so the table starts at the bilinear product without
   distortion.

4. **RCC operator.** ``reply_channel_capacity`` softmaxes each row of
   `L` (per-candidate reply distribution) and runs 24 damped
   Blahut-Arimoto-style iterations to find the capacity-achieving
   prior `q*`, the reply marginal `r`, and the channel capacity in
   nats and bits.

5. **Fusion head.** The candidate-pooled
   `att_pool = q*^T c`, the reply-pooled `def_pool = r^T r_j`, the
   trunk pool feature, and the five scalar diagnostics
   `(capacity_nats, capacity_bits, conditional_entropy,
   output_entropy, capacity_gap)` are concatenated and fed through a
   LayerNorm + GELU MLP that produces `primitive_delta_raw`. A second
   MLP over `[p_trunk, capacity, capacity_gap, conditional_entropy]`
   produces the gate logit. Final contract:

   ```
   primitive_delta = primitive_gate * primitive_delta_raw
   final_logit     = base_logit + primitive_delta
   ```

   The gate bias is initialised at `-2.0` so the primitive starts as
   a near-no-op.

6. **Ablations.** Eight supported modes via `model.ablation`:

   - `none`: full architecture.
   - `row_shuffle_channel`: permute candidate rows; capacity drops.
   - `duplicate_rows`: force all rows to row 0's distribution;
     capacity collapses to zero.
   - `uniform_replies`: collapse each row to uniform; conditional
     entropy is maximised, capacity is zero.
   - `entropy_only`: zero out everything except conditional entropy
     in the fusion head; tests whether full capacity beats entropy.
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
| Bilinear logits | `O(K * R * D)` |
| RCC solver | `O(T * K * R)` |
| Fusion MLPs | small |

## Implementation Binding

- Registered model name: `reply_channel_capacity` (idea slug). The
  legacy registry alias `reply_channel_capacity_network` resolves to
  the same builder for backwards-compatible tests.
- Source implementation:
  `src/chess_nn_playground/models/primitives/reply_channel_capacity_network.py`
  (the module retains its `_network` filename; the class is
  `ReplyChannelCapacityNetwork`).
- Shared candidate/reply utilities:
  `src/chess_nn_playground/models/primitives/codex_reply_primitives.py`.
- Trunk source:
  `src/chess_nn_playground/models/trunk/exchange_then_king_dual_stream.py`
  (the bespoke i193 trunk is wrapped, not reimplemented).
- Idea-local wrapper:
  `ideas/registry/p003_reply_channel_capacity/model.py`.
- Training config:
  `ideas/registry/p003_reply_channel_capacity/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`:
  `MODEL_BUILDERS["reply_channel_capacity"] = build_reply_channel_capacity_network_from_config`
  (with the `reply_channel_capacity_network` alias retained).

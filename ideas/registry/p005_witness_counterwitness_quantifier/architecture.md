# Architecture

`Witness-Counterwitness Quantifier Network` (p005) is an additive,
gated head on top of the existing i193 `ExchangeThenKingDualStreamNetwork`
trunk. The trunk runs once per board; the head adds candidate + reply
token compilers, claim and counter scoring heads, the nested
adversarial quantifier, and a small fusion MLP.

The model consumes the repository `simple_18` current-board tensor
`(B, 18, 8, 8)` and returns one puzzle logit, plus a rich per-sample
diagnostics dict.

## Mechanism

1. **i193 trunk pass.** ``run_i193_trunk_with_spatial`` returns the
   base logit, the i193 diagnostic dict, the joint pool feature
   `p_trunk`, and the spatial map `H = cat(ex_h, kg_h)`.

2. **Witness / counterwitness compilers.** Two independent
   `BoardTokenAttention` pools (default 16 candidates, 12 replies,
   token_dim = 48) attend over the 64 spatial cells of `H` to produce
   candidate (witness) tokens `c_k` and reply (counterwitness) tokens
   `r_j`.

3. **Claim and counter scoring.**

   ```
   ctx       = context_proj(p_trunk)
   c_ctx     = c_k + tanh(ctx)
   claim_i   = claim_head(c_ctx)
   pair_ij   = cand_pair_proj(c_i) * reply_pair_proj(r_j)
   counter_ij = counter_head(pair_ij)
   ```

   `claim_head` is a small LayerNorm + GELU MLP, `counter_head` is a
   similar MLP over the bilinear interaction.

4. **WCQ operator.** ``witness_counterwitness_quantifier`` runs the
   nested `forall_softmax_j` then `exists_softmax_i` reductions with
   independent temperatures (defaults 0.20 / 0.20), returning the
   board-level value, margin, counter envelope, soft witness and
   counter weights, and best-witness / best-counter indices. The
   operator falls back to a zero envelope when a candidate has no
   surviving counterwitness (per the source packet).

5. **Fusion head.** The witness-pooled candidate token
   `att_pool = w^T c`, the counter-pooled reply token
   `def_pool = sum_{i,j} cwt_{ij} r_j`, the trunk pool feature, and
   the four scalar diagnostics
   `(value, max_margin, counter_envelope_max, witness_entropy)` are
   fed through a LayerNorm + GELU MLP that produces
   `primitive_delta_raw`. A second MLP over
   `[p_trunk, value, max_margin, witness_entropy]` produces the gate
   logit. Final contract:

   ```
   primitive_delta = primitive_gate * primitive_delta_raw
   final_logit     = base_logit + primitive_delta
   ```

   The gate bias is initialised at `-2.0` so the primitive starts as
   a near-no-op.

6. **Ablations.** Eight supported modes via `model.ablation`:

   - `none`: full architecture.
   - `max_claim_only`: bypass the counter branch, value = max claim.
   - `mean_counter_penalty`: replace forall-soft with mean per row.
   - `random_counter_assign`: permute counter rows across candidates.
   - `no_counter_branch`: zero out counter scores entirely.
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
| Claim head | `O(K * D * head_hidden_dim)` |
| Counter head | `O(K * R * compat_dim * head_hidden_dim)` |
| WCQ operator | `O(K * R)` logsumexp + `O(K)` outer logsumexp |
| Fusion MLPs | small |

## Implementation Binding

- Registered model name: `witness_counterwitness_quantifier_network`.
- Source implementation:
  `src/chess_nn_playground/models/primitives/witness_counterwitness_quantifier_network.py`.
- Shared candidate/reply utilities:
  `src/chess_nn_playground/models/primitives/codex_reply_primitives.py`.
- Idea-local wrapper:
  `ideas/registry/p005_witness_counterwitness_quantifier/model.py`.
- Training config:
  `ideas/registry/p005_witness_counterwitness_quantifier/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`:
  `MODEL_BUILDERS["witness_counterwitness_quantifier_network"] = build_witness_counterwitness_quantifier_network_from_config`.

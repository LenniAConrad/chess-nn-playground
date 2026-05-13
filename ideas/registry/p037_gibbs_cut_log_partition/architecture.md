# Architecture

`Gibbs Cut Log-Partition Operator` (p037) is an additive, gated head
over the existing i193 trunk. It evaluates a Gibbs log-partition over
the cuts of a small latent grid and pools the resulting per-channel
log-partition values into a scalar delta.

The model consumes the repository `simple_18` current-board tensor
`(B, 18, 8, 8)` and returns one puzzle logit, plus a per-sample
diagnostic dict mirroring the i248 contract.

## Mechanism

1. **i193 trunk forward**. The bespoke
   `ExchangeThenKingDualStreamNetwork` runs unchanged and emits the
   full diagnostic dict including `logits` (`base_logit`) and the
   dual-stream joint feature.

2. **Cut-grid input projection**. Four linear maps project the joint
   feature to the cut graph inputs:

   - `c_h` -- horizontal edge costs `(B, H, W-1, d_cut)`
   - `c_v` -- vertical edge costs `(B, H-1, W, d_cut)`
   - `s`   -- source penalties `(B, H, W, d_cut)`
   - `t`   -- sink penalties `(B, H, W, d_cut)`

   All four are passed through `softplus` so they remain non-negative,
   matching the `c >= 0` contract of the cut log-partition.

3. **State-bit precomputation**. The state bit decomposition
   `bits in {0,1}^{2^W, W}`, the within-row XOR table
   `within_xor in R^{2^W, W-1}`, and the between-row XOR tensor
   `between_xor in R^{2^W, 2^W, W}` are static buffers built once at
   construction time.

4. **Row transfer DP** in log space (see `math_thesis.md`). Per row `r`:

   ```
   within_cost[r, S]                  = sum_j c_h[r, j] * within_xor[S, j] / tau
   cell_cost[r, S]                    = (sum_j s[r, j] * (1 - bits[S, j])
                                          + sum_j t[r, j] * bits[S, j]) / tau
   between_cost[r-1, S_prev, S_curr]  = sum_j c_v[r-1, j] * between_xor[S_prev, S_curr, j] / tau
   log_Z_r[S_curr] = logsumexp_{S_prev}(log_Z_{r-1}[S_prev] - between_cost)
                     - within_cost[r, S_curr] - cell_cost[r, S_curr]
   ```

   Finalisation: `log_Z = logsumexp_{S} log_Z_{H-1}[S]`, then
   `y = -tau * log_Z in R^{B, d_cut}`.

5. **Edge-energy diagnostic**. Per-channel mean of `c_h` + `c_v` gives
   `cut_edge_energy in R^{B, d_cut}`, surfaced as a diagnostic.

6. **Readout**. Concatenate `y` and `cut_edge_energy` to a `(B, 2 d_cut)`
   readout vector and run through a LayerNorm + Linear + GELU + Dropout
   + Linear stack producing `primitive_delta_raw`.

7. **Gate**. A second LayerNorm + Linear + GELU + Linear stack on the
   trunk joint feature produces `gate_logit`. The effective gate is
   `sigmoid(gate_logit)`, initialised near zero by `gate_init = -2.0`.

8. **Logit fusion**. `final_logit = base_logit + gate * delta_raw`.

## Ablations

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `shuffle_logpartition` | In-batch permutation of `y`. Primary falsifier: if A1 matches `none`, the log-partition carries no signal. |
| A2 | `uniform_edges` | Replace `c_h`, `c_v` with all-ones. Tests whether edge costs are load-bearing. |
| A3 | `uniform_sources` | Replace `s`, `t` with all-ones. Tests whether source/sink penalties are load-bearing. |
| A4 | `zero_delta` | Hold `primitive_delta = 0`. Recovers i193. |
| A5 | `trunk_only` | Strongest control: alias of `zero_delta`. |

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
Stockfish scores, principal variations, and any report-only metadata
are *not* consumed by the model. Only the simple_18 board tensor is
input.

## Cost

| Stage | Per-sample cost |
|---|---|
| i193 trunk forward | One pass through the dual-stream encoder |
| Cut input projections | `O(feature_dim * (H*W + (H-1)*W + H*(W-1)) * d_cut)` |
| Row transfer DP | `O(H * 2^(2W) * d_cut)` for between-row + `O(H * 2^W * (W-1 + W) * d_cut)` for within/cell |
| Readout | `O(d_cut + head_hidden_dim)` |

With `H = W = 4`, `d_cut = 4`: `4 * 16 * 16 * 4 = 4096` einsum entries
for between-row, dominated by the projection step.

## Implementation Binding

- Registered model name: `gibbs_cut_log_partition`.
- Source implementation:
  `src/chess_nn_playground/models/primitives/gibbs_cut_log_partition.py`.
- Idea-local wrapper:
  `ideas/registry/p037_gibbs_cut_log_partition/model.py`.
- Training config:
  `ideas/registry/p037_gibbs_cut_log_partition/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`.

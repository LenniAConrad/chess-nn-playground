# Ablations

p003 supports eight ablation modes via `model.ablation`. The primary
falsifier is `entropy_only` — every promotion run must include this
matched control because the main novelty claim is "capacity beats
entropy."

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `entropy_only` | Zero out everything except conditional entropy in the fusion head. **Primary falsifier vs `i192`-style baselines.** If A1 matches the unablated run, full capacity is not load-bearing. |
| A2 | `row_shuffle_channel` | Permute candidate rows; capacity collapses. |
| A3 | `duplicate_rows` | All rows = row 0; capacity is zero. |
| A4 | `uniform_replies` | Uniform reply distribution per row; capacity is zero. |
| A5 | `disable_gate` | Hold `primitive_gate` at 1.0. |
| A6 | `zero_delta` | Zero out `primitive_delta`. Recovers i193 baseline. |
| A7 | `trunk_only` | Zero out features and delta. Strongest control. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p003 >= i193 - 0.005, AND
- matched-recall near-puzzle FP at recall 0.80 improves by at least
  2% versus entropy-only baselines on the same parent, AND
- A1 (`entropy_only`) loses >= 50% of the near-puzzle FP lift, AND
- A2/A3/A4 (capacity-killing ablations) all lose >= 70% of the lift.

Drop especially if A1 matches the unablated run — that means the
full capacity is not load-bearing beyond conditional entropy and the
simpler `i192_latent_reply_entropy_network` architecture is
preferable.

## Out-of-scope ablations (future)

- Replace the unrolled Blahut-Arimoto with implicit-differentiation
  through the fixed point.
- Combine RCC and RSP by computing the saddle on the capacity-
  weighted payoff table.
- Compare against direct mutual information estimators
  (MINE-style) to verify the BA solver beats variational bounds.

Run these only after the primary falsifier (`entropy_only`) passes.

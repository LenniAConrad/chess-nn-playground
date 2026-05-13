# Ablations

p002 supports eight ablation modes via `model.ablation`. The primary
falsifiers are `row_shuffle_payoff` and `col_shuffle_payoff`.

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `row_shuffle_payoff` | Permute payoff rows (per batch). Destroys candidate-side game structure but preserves row entry distributions. **Primary falsifier.** If A1 matches the unablated run, the saddle structure carries no signal. |
| A2 | `col_shuffle_payoff` | Permute payoff columns. Destroys reply-side game structure. |
| A3 | `uniform_payoff` | Collapse the table to per-batch mean. Removes all structure. |
| A4 | `pure_max_min` | Bypass the regularized solver; use raw `max_i min_j A_ij`. Tests whether the entropy-regularized solution beats the pure saddle. |
| A5 | `disable_gate` | Hold `primitive_gate` at 1.0. |
| A6 | `zero_delta` | Zero out `primitive_delta`. Recovers i193 baseline. |
| A7 | `trunk_only` | Zero out features and delta. Strongest control. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p002 >= i193 - 0.005, AND
- matched-recall near-puzzle FP at recall 0.80 improves by at least
  3% versus i193, AND
- A1 (`row_shuffle_payoff`) loses >= 70% of the near-puzzle FP lift,
  AND
- A2 (`col_shuffle_payoff`) loses >= 50% of the lift, AND
- the `crtk_eval_bucket = equal` slice does not regress more than
  the aggregate threshold.

Drop if any condition fails. Drop especially if A4 (`pure_max_min`)
matches the unablated run — that means the regularized solver is not
contributing beyond a hard argmax.

## Out-of-scope ablations (future)

- Replace the unrolled solver with an implicit-differentiation
  fixed-point solver and compare numerical stability on cyclic
  tables.
- Add an auxiliary exploitability-minimisation loss.
- Combine RSP and PAFR by feeding the PAFR frontier summary as a
  candidate prior into the RSP solver.

Run these only after the primary falsifiers pass.

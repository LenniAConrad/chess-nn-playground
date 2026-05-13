# Ablations

p001 supports eight ablation modes via `model.ablation`. The primary
falsifier is `shuffle_channels` — every promotion run must include
this matched control on the same split, seed, and training budget.

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `shuffle_channels` | In-batch permutation of utility channels across candidates. Decouples channels from candidates while keeping marginal channel distributions intact. **The primary falsifier.** If A1 matches the unablated run on the near-puzzle FP slice, the channel structure carries no signal and the primitive is dropped. |
| A2 | `single_channel` | Use only utility channel 0. Collapses the product partial order to a 1-D order. If A2 matches A0, the product partial order is not load-bearing. |
| A3 | `scalar_max` | Collapse utilities to a per-candidate scalar max. Tests whether the partial order beats a learned total order. |
| A4 | `uniform_frontier` | Drop the frontier softmax to a uniform distribution over valid candidates. Tests whether frontier-weighted aggregation matters. |
| A5 | `disable_gate` | Hold `primitive_gate` at 1.0. Tests whether the gate is load-bearing or whether direct fusion works. |
| A6 | `zero_delta` | Zero out `primitive_delta`. Recovers i193 trunk behaviour. Sanity check. |
| A7 | `trunk_only` | Zero out features and delta. Strongest control. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p001 >= i193 - 0.005, AND
- matched-recall near-puzzle FP at recall 0.80 improves by at least
  3% versus i193, AND
- A1 (`shuffle_channels`) loses >= 70% of the near-puzzle FP lift,
  AND
- A2 (`single_channel`) does not match the unablated run.

Drop if any condition fails. Drop especially if A1 matches the
unablated run — that means the head learned to use trunk
diagnostics or the candidate compiler alone without relying on the
partial-order structure.

## Out-of-scope ablations (future)

- Drop individual utility channels at a time to check which axes
  drive the frontier. Requires config-driven feature masking.
- Replace the candidate compiler with a fixed top-K spatial pool to
  remove the set-query attention from the variable.
- Combine with p005 (WCQ) by feeding the PAFR frontier summary into
  the witness branch of WCQ.

Run these only after the primary falsifier (`shuffle_channels`)
passes.

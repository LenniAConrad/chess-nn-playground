# Ablations

p005 supports eight ablation modes via `model.ablation`. The primary
falsifiers are `max_claim_only` and `mean_counter_penalty` — the
main novelty claim is "nested adversarial quantifier beats both
claim-only and mean-penalty baselines."

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `max_claim_only` | Bypass the counter branch entirely; value = max_i claim_i. **Primary falsifier.** If A1 matches the unablated run, the counter branch is not load-bearing. |
| A2 | `mean_counter_penalty` | Replace `tau_forall * logsumexp_j` with `mean_j`. Tests whether the forall structure beats averaging. |
| A3 | `random_counter_assign` | Permute counter rows across candidates so counter scores no longer match. Should destroy the lift. |
| A4 | `no_counter_branch` | Zero out counter scores. Same effect as A1 but without the architectural shortcut. |
| A5 | `disable_gate` | Hold `primitive_gate` at 1.0. |
| A6 | `zero_delta` | Zero out `primitive_delta`. Recovers i193 baseline. |
| A7 | `trunk_only` | Zero out features and delta. Strongest control. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p005 >= i193 - 0.005, AND
- matched-recall near-puzzle FP at recall 0.80 improves by at least
  5% versus i193 and i011 on the promotion / underpromotion and
  mate-in-1 buckets, AND
- A1 (`max_claim_only`) loses >= 70% of the near-FP lift, AND
- A2 (`mean_counter_penalty`) loses >= 50% of the lift.

Drop especially if A1 matches the unablated run — that means claim-
only pooling is sufficient and the nested quantifier is not
load-bearing.

## Out-of-scope ablations (future)

- Anneal `tau_forall, tau_exists` from 0.5 -> 0.15 during training
  and compare against fixed defaults.
- Add a learned compatibility bias per (witness, counterwitness)
  pair via an explicit compatibility MLP.
- Combine WCQ with PAFR by feeding the PAFR frontier summary as a
  candidate prior into the WCQ witness branch.

Run these only after the primary falsifiers (`max_claim_only`,
`mean_counter_penalty`) pass.

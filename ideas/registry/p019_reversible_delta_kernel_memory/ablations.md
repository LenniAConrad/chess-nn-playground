# Ablations

p019 supports five ablation modes via `model.ablation`. The primary
falsifier is `shuffle_tokens` -- every promotion run must include this
matched control on the same split, seed, and training budget.

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `shuffle_tokens` | In-batch permutation of the per-square token tensor. **The primary falsifier.** If A1 matches the unablated run on the declared target slice, the kernel memory carries no signal in this trunk and the primitive is dropped. |
| A2 | `zero_memory` | Force `M = 0` and `z = 0`. Tests whether the kernel-memory readout has any effect. The model should collapse close to i193 here. |
| A3 | `uniform_query` | Replace the trunk-derived queries with a uniform `1/h` tensor (no routing). Tests whether the query mechanism is load-bearing. |
| A4 | `zero_delta` | Hold `primitive_delta = 0`. Recovers i193 trunk behaviour. Sanity check that wrapping the trunk did not regress the baseline. |
| A5 | `trunk_only` | Strongest control: zero out the delta entirely. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p019 >= i193 - 0.005, AND
- declared target slice PR AUC of unablated p019 >= i193 + 0.04, AND
- A1 (`shuffle_tokens`) loses >= 70% of the slice lift over i193, AND
- training throughput drop versus i193 < 25%.

Drop if any condition fails. Drop especially if A1 matches the
unablated run -- that means the head learned to use the gate or trunk
diagnostics without relying on the kernel memory.

## Out-of-scope ablations (future)

- Drop individual `phi`/`nu` projections.
- Replace `(elu + 1)` with an exponential feature map.
- Disable normalisation by `z` (raw inner product).

Run these only after the primary falsifier passes.

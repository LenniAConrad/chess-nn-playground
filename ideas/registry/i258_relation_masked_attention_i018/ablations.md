# Ablations

The i258 ablation grid follows the research packet exactly. Every row is a
one-flag change in `config.yaml` -- `model.relation_attention.*` for the
attention block and `model.scramble_relations` for the i018 falsifier path.
All rows reuse the same trainer, sampler, loss, and seeds, so the
comparison is honest.

| ID | Flag change | Effect | What failure would mean |
|---|---|---|---|
| A0 | `relation_attention.enabled: false` | Disables the attention graft entirely (parent i018 trunk only). | Reference matched-budget i018 baseline. Required floor. |
| A1 | `relation_attention.neighborhood: global` | Top-K neighbor list ignores relation structure (all squares equally favoured). | If A1 >= A3, generic global attention is enough -- the chess constraint is not load-bearing. |
| A2 | `scramble_relations: true` | Degree-preserving random rewiring of the typed relation masks before either diffusion or attention sees them. | If A2 ~= A3, attention is just an extra content mixer riding on top of i018. |
| A3 | `relation_attention.neighborhood: relation` (default) | Primary i258 design. | Reference for keep / drop decision. |
| A4 | `relation_attention.neighborhood: king_zone` | Restricts neighborhoods to king-zone / pin relations. | If A4 << A3 on aggregate but A4 > A3 on tactical/mate slices, the king-zone variant is a specialist worth keeping alongside A3. |
| A5 | `relation_attention.neighborhood: candidate` | Restricts neighborhoods to own-piece outer-product (move-targeted reweighting). | If A5 > A3 on hard / equal slices, "where can this piece act?" beats "which squares are tactically related?" |
| A6 | `relation_attention.force_gate: 0.0` | Forces the gate to zero (graft cannot fire). | Should reproduce A0 closely. Any divergence from A0 indicates leakage of attention params into the readout. |

The four ablations directly requested by the research packet are A0, A1,
A2, A3 (the matched comparisons). A4, A5, A6 are the highest-value
follow-ons if A3 shows even a modest positive signal.

## Expected Ranking (from packet)

```text
A3 (relation)   >    A0 (no attention)  ~>=  A4 (king-zone)   >    A1, A2
```

That expectation is not because global attention is inherently bad, but
because the repo's evidence specifically is against *unconstrained* global
attention at this budget and recipe, while the strongest positive evidence
is for *typed tactical relation structure*.

## Quantitative Decision Rule

The decision rule is stricter than "best single seed wins". Three seeds
(42 / 43 / 44) are required for any keep / drop call.

| Axis | Go | No-go |
|---|---|---|
| Aggregate accuracy | Mean test PR-AUC over seeds improves by `>= 0.003` vs A0 | Lift < 0.002 or regression |
| Mechanism specificity | A3 beats A1 by `>= 0.003` and A2 by `>= 0.010` | A1 ties or wins; A2 ties within 0.01 |
| Hard-slice behavior | No obvious regressions on `hard`, `equal`, `promotion`, `underpromotion` | Any material slice regression without aggregate compensation |
| Matched-recall behavior | Same or better near-puzzle FP at recall `0.80` / `0.85` | Worse on both thresholds |
| Efficiency | Train/inference slowdown stays within `15%` | Slowdown > 20% |
| Stability | Three seeds finish cleanly with no sensitivity spike | High seed variance or unstable optimisation |

Practical summary:

- **Go** if A3 gives a real, repeatable gain at matched budget and clearly
  beats A1 and A2.
- **Soft go** if aggregate PR-AUC is nearly flat but matched-recall
  near-puzzle FP or hard-slice PR-AUC improves measurably *and* A3
  still beats A1 / A2 on at least one slice.
- **No-go** if the gain vanishes under three seeds, if A1 ties or wins,
  or if A3 does not materially depend on real relation masks.

## Loss / sampling ablations (deferred)

| ID | Ablation | What it tests | Status |
|---|---|---|---|
| L0 | `+ lambda_gate * sum_k E[gate_k]` only | Whether the gate-sparsity penalty matters beyond BCE. | Deferred -- requires trainer support for an auxiliary term. |
| L1 | `+ lambda_slice * L_slice` only | Whether slice-restricted ranking lifts the slice PR-AUC numbers. | Deferred -- requires pair-aware trainer batches. |
| L2 | `+ lambda_near * L_near` only | Whether near-puzzle-FP reweighting alone matches the architecture lift. | Deferred -- requires sampler / loss change. |
| L3 | `uniform_sampler` | Whether the chess-explained slice curriculum is worth the complexity. | Deferred -- requires sampler change. |
| L4 | `+ lambda_kd * KL_distill` | Whether teacher-student distillation closes the gap on a smaller student. | Deferred -- requires teacher checkpoint and KD trainer hook. |

These ablations are deferred because they depend on trainer extensions
not bundled with the architecture promotion. The current matched-recipe
run uses BCE-with-logits and the shared sampler so the result is
honestly attributable to the architecture rather than to confounded
loss / sampling changes.

## Keep / Drop Rule

Keep i258 only if all are true:

- A3 beats A0 on mean test PR-AUC by `>= 0.003` over three seeds *or*
  improves matched-recall near-puzzle FP at recall `0.80` / `0.85`
  without aggregate regression.
- A3 beats A1 by `>= 0.003`.
- A3 beats A2 by `>= 0.010`.
- Aggregate PR-AUC stays within `0.005` of the matched i018 baseline
  on the test set (no silent regression from the readout reduction).
- The bounded-residual identity (`h_attn = h0` under
  `relation_attention.force_gate: 0.0`) holds exactly under
  `torch.allclose` with the matching i018 forward.

Drop i258 if A1 ties or beats A3, or if A2 ties A3, or if A3 fails to
beat A0 on either aggregate PR-AUC or matched-recall near-puzzle FP.
If only one neighborhood mode passes, keep only that mode and drop the
other neighborhood flags from the default config.

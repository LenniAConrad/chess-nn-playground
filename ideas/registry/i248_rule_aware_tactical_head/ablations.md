# Ablations

i248 supports five ablation modes via `model.ablation`. The primary
falsifier is `shuffle_tsdp` — every promotion run must include this
matched control on the same split, seed, and training budget.

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `shuffle_tsdp` | In-batch permutation of the 11-d TSDP vector. Decouples rule features from positions. **The primary falsifier.** If A1 matches the unablated run on the mate_in_1 slice, the rule features carry no signal in this trunk and the primitive is dropped. |
| A2 | `disable_gate` | Hold `primitive_gate` at 1.0. Tests whether the learned gate is load-bearing or whether direct fusion works. |
| A3 | `zero_delta` | Zero out `primitive_delta`. Recovers i193 trunk behavior. Sanity check that wrapping the trunk did not regress the baseline. |
| A4 | `zero_features` | Zero out the TSDP vector but keep the head and gate. Tests whether the head learns from trunk diagnostics alone. |
| A5 | `trunk_only` | Zero out features and delta. Strongest control. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated i248 >= i193 - 0.005, AND
- mate_in_1 slice PR AUC of unablated i248 >= 0.85 (i193 ~0.81 + 0.04), AND
- A1 (`shuffle_tsdp`) loses >= 70% of the mate_in_1 slice lift over i193, AND
- the `crtk_eval_bucket = equal` slice does not regress more than the
  aggregate threshold.

Drop if any condition fails. Drop especially if A1 matches the unablated
run — that means the head learned to use the gate / trunk diagnostics
without relying on the rule indicators at all.

## Out-of-scope ablations (future)

The TSDP primitive markdown also lists three ablations that require new
architecture surface area and are out of scope for the first scout run:

- Drop individual indicators (mate-only, check-only, ...): would need
  config-driven feature masking.
- Learned approximation control: replace `_compute_tsdp` with a small CNN
  that predicts `is_checkmate` and compare. Establishes whether
  rule-exactness matters versus learnable approximation.
- Two-ply TSDP: enumerate side-to-move legal moves AND opponent replies,
  classify two-ply terminal states. Order-of-magnitude more expensive;
  only run if the one-ply version passes the falsifier.

Run these only after the primary falsifier (`shuffle_tsdp`) passes.

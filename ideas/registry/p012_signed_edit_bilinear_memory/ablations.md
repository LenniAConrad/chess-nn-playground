# Ablations — Signed-Edit Bilinear Memory (p012)

This idea inherits the shared delta-accumulator ablation set defined on
``DeltaAccumulatorHead`` plus its own primitive-specific falsifiers (see
``signed_edit_bilinear_memory.SignedEditBilinearMemory.DEFAULT_ABLATIONS``).

## Shared baseline ablations

| ID | ``model.ablation`` | What it tests |
|---|---|---|
| B1 | ``zero_delta`` | Forces ``primitive_delta = 0``. Recovers i193 trunk behaviour. Sanity check that wrapping the trunk did not regress the baseline. |
| B2 | ``trunk_only`` | Strongest control — both delta and state zeroed. |
| B3 | ``shuffle_features`` | In-batch permutation of the active-feature index list. Decouples the rule-derived feature set from the position. |
| B4 | ``disable_gate`` | Hold the primitive gate at 1.0. Tests whether the learned gate is load-bearing. |
| B5 | ``zero_state`` | Force the active-feature set to the empty set. Tests whether the head learns from trunk diagnostics alone. |

## Primitive-specific ablations

See ``signed_edit_bilinear_memory.SignedEditBilinearMemory.DEFAULT_ABLATIONS`` for the
complete list; the primitive-specific falsifiers documented in
``ideas/research/primitives/external_01_signed_edit_bilinear_memory_ray_scan.md`` are exposed verbatim.

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of the unablated run ≥ i193 baseline − 0.005, AND
- the declared primitive-specific ablation(s) lose ≥ 50% of the lift on
  the declared target slice (see ``math_thesis.md``).

Drop if any condition fails. The primitive-specific ablations are the
load-bearing falsifiers; failing them means the primitive is not doing
the work documented in the math_thesis.

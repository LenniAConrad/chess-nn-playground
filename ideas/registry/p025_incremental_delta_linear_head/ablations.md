# Ablations

`p025` exposes the shared primitive-head ablation set plus three IDL-
specific controls. The primary falsifier is `shuffle_squares` — every
promotion run must include this matched control on the same split, seed,
and training budget.

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `shuffle_squares` | Random column permutation of the 12-plane indicator tensor before the einsum. Decouples the per-square structure of `E` from the actual board square. **Primary falsifier.** If A1 matches the unablated run on the declared target slice, the per-square factorisation is not load-bearing and the primitive is dropped. |
| A2 | `permute_piece_types` | Row permutation of the indicator tensor. Same logic as A1 but on the piece-type axis. |
| A3 | `zero_accumulator` | Hold `S(x) = 0`. Tests whether the trunk diagnostics in the fusion vector are doing all the work (they should not — the trunk emits them upstream as plain `gate`, `gate_entropy`, etc.). |
| A4 | `zero_delta` | Force `primitive_delta = 0`. Recovers i193 baseline behaviour. Sanity check that wrapping the trunk did not regress the baseline. |
| A5 | `disable_gate` | Hold the gate at 1.0. Tests whether the learned gate is load-bearing or whether unconditional addition works. |
| A6 | `trunk_only` | Force both the gate and the delta to zero. Strongest control. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p025 >= i193 - 0.005, **and**
- the declared target slice (material-count / pawn-structure heavy
  positions) lifts >= +0.02 PR AUC over i193, **and**
- A1 (`shuffle_squares`) loses >= 70% of the target slice lift, **and**
- A2 (`permute_piece_types`) loses >= 50% of the target slice lift.

Drop if any condition fails. Drop especially if A1 matches the
unablated run — that means the head learned to use the embedding-table
sum without relying on the per-square assignment at all.

## Out-of-scope ablations (future)

- Color-symmetric weight tying (the IEL twin from the research file).
- Replacement of the linear sum with a non-linear lift; the ILA primitive
  (`p028`) is the dedicated test of that hypothesis.

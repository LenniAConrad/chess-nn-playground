# Ablations

p055 supports nine ablation modes via `model.ablation`. The primary
falsifiers are `no_replies` and `no_legality_discount`; every
promotion run must include both matched controls on the same split,
seed, and training budget.

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `no_replies` | Zero out `ReplyMass`, `Avail`, `RCI`. **Primary falsifier.** If A1 matches the unablated run, the safe-reply envelope is not load-bearing -- the head reduces to candidate-only scoring. |
| A2 | `no_legality_discount` | Collapse `Disc(m*)` to zero. **Primary falsifier.** If A2 matches the unablated run, the surface-vs-verified gap is not load-bearing. |
| A3 | `concentration_only` | Keep only `Conc` and `Gap12`. Tests whether the entire candidate/reply machinery reduces to a softmax-concentration head. |
| A4 | `shuffle_replies` | In-batch permutation of reply tokens. Decouples replies from position. |
| A5 | `no_overload` | Drop `DOA` from `z`. |
| A6 | `no_king_escape` | Drop `KEP` from `z`. |
| A7 | `zero_delta` | Zero primitive delta. Recovers i193 baseline. |
| A8 | `trunk_only` | Same as A7 (semantic alias). |
| A9 | `disable_gate` | Pin gate at 1.0. Tests gate load-bearing. |

## Keep / drop rule

Promote (keep) only if:

- aggregate test PR AUC of unablated p055 >= i193 - 0.005, AND
- the target near-puzzle FP rate at matched recall 0.80 falls at
  least 3% relative, AND
- the target near-puzzle FP rate at matched recall 0.85 falls at
  least 3% relative, AND
- A1 (`no_replies`) loses >= 50% of that lift, AND
- A2 (`no_legality_discount`) loses >= 50% of that lift, AND
- A3 (`concentration_only`) loses >= 30% of that lift, AND
- the `crtk_eval_bucket = equal`, `crtk_difficulty = hard`, and
  `crtk_difficulty = very_hard` slices do not regress more than the
  aggregate threshold.

Drop if any condition fails.

## Out-of-scope ablations (future)

- *True candidate/reply compiler*: replace the learned attention
  pools with a `python-chess`-driven compiler operating on
  precomputed legal-move parquet tables.
- *Sampler-level near-puzzle mining*: in-trainer pairwise loss with a
  near-puzzle hard-negative replay buffer.
- *FiLM-style trunk conditioning*: use `z(x)` to condition the
  trunk's final residual block.

Run these only after the primary falsifiers (`no_replies`,
`no_legality_discount`) pass.

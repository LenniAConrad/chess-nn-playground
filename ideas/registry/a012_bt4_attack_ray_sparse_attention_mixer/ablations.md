# Ablations

This folder is a controlled architecture study, not a primitive study.
The first-class ablations are *cross-idea* comparisons against the
matched `conv` and `attention` BT4 baselines and against the source
primitive (`p007_attack_ray_sparse_attention`) used as an additive
head rather than a token mixer.

## Ablation switches

| ID | Comparison run | What it changes |
|---|---|---|
| A1 | `bt4_conv_mixer` (sibling idea / baseline) | Replaces the ARSA mixer with the original `conv` mixer, holding every other tower hyperparameter, optimizer setting, and data split fixed. Direct control for "is the ARSA mixer better than a conv?". |
| A2 | `bt4_attention_mixer` (sibling idea / baseline) | Replaces the ARSA mixer with a generic multi-head self-attention over the 64 squares. Direct control for "is the ARSA mixer better than a vanilla token mixer?". |
| A3 | `p007_attack_ray_sparse_attention` (source primitive idea) | Uses ARSA as an additive gated head over the i193 trunk instead of as the per-block spatial mixer. Tests whether the primitive transfers any of its signal through the BT4 tower at all. |
| A4 | Capacity-matched `bt4_conv_mixer` | If the ARSA mixer adds parameters versus the conv mixer, A1 is repeated with conv `channels`/`num_blocks` increased to match the parameter count. Distinguishes "ARSA mixer carries new signal" from "ARSA mixer just adds capacity". |
| A5 | `random_keys` ARSA (replace ray-cast first-blocker key indices with random per-source 8-square draws plus self-edge) | Tests whether the *rule-derived ray geometry* carries signal beyond mere 9-slot sparsity. If the random-keys variant matches the learned-mask variant, the first-blocker structure was not load-bearing. |
| A6 | `uniform_attention` ARSA (replace per-slot softmax with a uniform `1/K` over valid slots) | Tests whether the learned attention scores carry signal beyond fixed-mean pooling over the 9 first-blocker slots. |

## What each ablation tests

- **A1 (vs `conv`)**: the primary architecture-level falsifier. If the
  ARSA mixer does not beat `conv` on at least one CRTK slice without
  regressing aggregate PR AUC, the mixer carries no architecture-level
  signal in this tower.
- **A2 (vs `attention`)**: protects against the trivial conclusion
  "any token mixer beats conv on 64 squares". If `attention` matches
  the ARSA mixer, the win in A1 is generic attention-style mixing,
  not the rule-derived first-blocker 9-slot sparse structure of ARSA.
- **A3 (vs primitive as head)**: tests transferability. The source
  primitive was validated as an additive gated head on the i193
  trunk; A3 tells us whether the same signal survives being
  repurposed as a token mixer.
- **A4 (capacity match)**: distinguishes signal from FLOPs /
  parameter count.
- **A5 (random-keys)**: the primary ARSA falsifier inherited from
  the source primitive. Separates "rule-derived first-blocker key
  set is load-bearing" from "any 9-slot sparse mixer works".
- **A6 (uniform-attention)**: separates "learned softmax over the
  9 first-blocker slots is load-bearing" from "uniform mean over
  the same key set works".

## Falsification criteria

Promote (keep) this idea only if all hold on the held-out test split:

- A1: ARSA mixer beats `conv` on at least one CRTK slice
  (`crtk_eval_bucket`, `crtk_difficulty`, `crtk_phase`, or
  `crtk_tactic_motifs`) by at least the matched-baseline tolerance
  documented in `ideas/docs/BENCHMARK_REPORTING.md`, AND
- aggregate test PR AUC does not regress vs `conv` by more than 0.005,
  AND
- A2: ARSA mixer is not strictly dominated by `attention` on the
  target slice, AND
- A4: the slice-level lift survives the capacity-matched conv
  comparison, AND
- A5: the random-keys ARSA variant loses at least 50% of the target-
  slice lift; otherwise the rule-derived first-blocker geometry is
  not load-bearing and the mixer is just sparse mixing by shape, AND
- A6: the uniform-attention ARSA variant does NOT match the learned
  variant; otherwise the softmax weighting is noise on top of a
  fixed-mean pool.

Drop if any one fails. Drop especially if A4 or A5 close -- A4
closing means the ARSA mixer is buying its win with parameter count,
A5 closing means it is buying its win with arbitrary 9-slot
sparsity rather than the chess-specific first-blocker geometry.

# Ablations

This folder is a controlled architecture study, not a primitive study.
The first-class ablations are *cross-idea* comparisons against the
matched `conv` and `attention` BT4 baselines and against the source
primitive (`p004_tail_copula_concordance`) used as an additive head
rather than a token mixer.

## Ablation switches

| ID | Comparison run | What it changes |
|---|---|---|
| A1 | `bt4_conv_mixer` (sibling idea / baseline) | Replaces the TCC mixer with the original `conv` mixer, holding every other tower hyperparameter, optimizer setting, and data split fixed. Direct control for "is the TCC mixer better than a conv?". |
| A2 | `bt4_attention_mixer` (sibling idea / baseline) | Replaces the TCC mixer with a generic multi-head self-attention over the 64 squares. Direct control for "is the TCC mixer better than a vanilla token mixer?". |
| A3 | `p004_tail_copula_concordance` (source primitive idea) | Uses TCC as an additive head over the i193 trunk instead of as the per-block spatial mixer. Tests whether the primitive transfers any of its signal through the BT4 tower at all. |
| A4 | Capacity-matched `bt4_conv_mixer` | If the TCC mixer adds parameters versus the conv mixer, A1 is repeated with conv `channels`/`num_blocks` increased to match the parameter count. Distinguishes "TCC mixer carries new signal" from "TCC mixer just adds capacity". |

## What each ablation tests

- **A1 (vs `conv`)**: the primary architecture-level falsifier. If the
  TCC mixer does not beat `conv` on at least one CRTK slice without
  regressing aggregate PR AUC, the mixer carries no architecture-level
  signal in this tower.
- **A2 (vs `attention`)**: protects against the trivial conclusion
  "any token mixer beats conv on 64 squares". If `attention` matches
  the TCC mixer, the win in A1 is generic attention-style mixing, not
  chess-aware cross-channel tail-concordance structure.
- **A3 (vs primitive as head)**: tests transferability. The source
  primitive was validated as an additive head on the i193 trunk; A3
  tells us whether the same signal survives being repurposed as a
  token mixer.
- **A4 (capacity match)**: distinguishes signal from FLOPs.

## Falsification criteria

Promote (keep) this idea only if all hold on the held-out test split:

- A1: TCC mixer beats `conv` on at least one CRTK slice
  (`crtk_eval_bucket`, `crtk_difficulty`, `crtk_phase`, or
  `crtk_tactic_motifs`) by at least the matched-baseline tolerance
  documented in `ideas/docs/BENCHMARK_REPORTING.md`, AND
- aggregate test PR AUC does not regress vs `conv` by more than 0.005,
  AND
- A2: TCC mixer is not strictly dominated by `attention` on the
  target slice, AND
- A4: the slice-level lift survives the capacity-matched conv
  comparison.

Drop if any one fails. Drop especially if A4 closes — that means the
TCC mixer is buying its win with parameter count, not with chess-
aware cross-site tail-concordance structure.

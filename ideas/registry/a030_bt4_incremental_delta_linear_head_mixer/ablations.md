# Ablations

This folder is a controlled architecture study, not a primitive
study. The first-class ablations are *cross-idea* comparisons against
the matched `conv` and `attention` BT4 baselines and against the
source primitive (`p025_incremental_delta_linear_head`) used as an
additive head rather than a token mixer.

## Ablation switches

| ID | Comparison run | What it changes |
|---|---|---|
| A1 | `bt4_conv_mixer` (sibling idea / baseline) | Replaces the `incremental_delta_linear_head` mixer with the original `conv` mixer, holding every other tower hyperparameter, optimizer setting, and data split fixed. Direct control for "is the `incremental_delta_linear_head` mixer better than a conv?". |
| A2 | `bt4_attention_mixer` (sibling idea / baseline) | Replaces the `incremental_delta_linear_head` mixer with a generic multi-head self-attention over the 64 squares. Direct control for "is the linear-additive accumulator better than an explicit softmax attention map at matched widths?". |
| A3 | `p025_incremental_delta_linear_head` (source primitive idea) | Uses the primitive as an additive head over the i193 trunk instead of as the per-block spatial mixer. Tests whether the primitive transfers any of its signal through the BT4 tower at all, and isolates the cost of replacing the per-(piece-type, square) embedding table with a channel-agnostic per-square linear map. |
| A4 | Capacity-matched `bt4_conv_mixer` | If the `incremental_delta_linear_head` mixer adds parameters versus the conv mixer, A1 is repeated with conv `channels`/`num_blocks` increased to match the parameter count. Distinguishes "this mixer carries new signal" from "this mixer just adds capacity". |

## What each ablation tests

- **A1 (vs `conv`)**: the primary architecture-level falsifier. If
  the `incremental_delta_linear_head` mixer does not beat `conv` on
  at least one CRTK slice without regressing aggregate PR AUC, the
  mixer carries no architecture-level signal in this tower.
- **A2 (vs `attention`)**: protects against "any all-pairs token
  mixer beats conv on 64 squares". If `attention` matches the mixer,
  the win in A1 is generic global-mixing, not the specific linear-
  additive accumulator structure.
- **A3 (vs primitive as head)**: tests transferability and isolates
  the cost of dropping the discrete per-(piece-type, square)
  embedding axis. The source primitive was designed as an additive
  head on the i193 trunk reading the piece planes directly; A3 tells
  us whether the same signal survives being repurposed as a token
  mixer with a channel-agnostic per-square linear map and a
  broadcast-back per-square fusion.
- **A4 (capacity match)**: distinguishes signal from FLOPs.

## Falsification criteria

Promote (keep) this idea only if all hold on the held-out test split:

- A1: the mixer beats `conv` on at least one CRTK slice
  (`crtk_eval_bucket`, `crtk_difficulty`, `crtk_phase`, or
  `crtk_tactic_motifs`) by at least the matched-baseline tolerance
  documented in `ideas/docs/BENCHMARK_REPORTING.md`, AND
- aggregate test PR AUC does not regress vs `conv` by more than
  0.005, AND
- A2: the mixer is not strictly dominated by `attention` on the
  target slice (or, if it is dominated, it must close the gap at
  lower per-block FLOPs), AND
- A4: the slice-level lift survives the capacity-matched conv
  comparison.

Drop if any one fails. Drop especially if A4 closes -- that means
the mixer is buying its win with parameter count, not with the
linear-additive accumulator structure. Drop also if the source-
primitive `zero_accumulator` ablation (run inside the BT4 tower by
holding the global sum `S = 0`) matches this idea on its declared
target slice, because then the accumulator is not load-bearing and
the mixer degenerates into a per-square own-token MLP.

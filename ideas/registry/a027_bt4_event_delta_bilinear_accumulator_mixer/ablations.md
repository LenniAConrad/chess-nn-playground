# Ablations

This folder is a controlled architecture study, not a primitive
study. The first-class ablations are *cross-idea* comparisons against
the matched `conv` and `attention` BT4 baselines and against the
source primitive (`p022_event_delta_bilinear_accumulator`) used as an
additive head rather than a token mixer.

## Ablation switches

| ID | Comparison run | What it changes |
|---|---|---|
| A1 | `bt4_conv_mixer` (sibling idea / baseline) | Replaces the `event_delta_bilinear_accumulator` mixer with the original `conv` mixer, holding every other tower hyperparameter, optimizer setting, and data split fixed. Direct control for "is the `event_delta_bilinear_accumulator` mixer better than a conv?". |
| A2 | `bt4_attention_mixer` (sibling idea / baseline) | Replaces the `event_delta_bilinear_accumulator` mixer with a generic multi-head self-attention over the 64 squares. Direct control for "is the FM-identity pair-term sum better than an explicit softmax attention map at matched widths?". |
| A3 | `p022_event_delta_bilinear_accumulator` (source primitive idea) | Uses the primitive as an additive head over the i193 trunk instead of as the per-block spatial mixer. Tests whether the primitive transfers any of its signal through the BT4 tower at all, and isolates the cost of swapping the source piece-plane occupancy for the learned soft occupancy used inside the mixer. |
| A4 | Capacity-matched `bt4_conv_mixer` | If the `event_delta_bilinear_accumulator` mixer adds parameters versus the conv mixer, A1 is repeated with conv `channels`/`num_blocks` increased to match the parameter count. Distinguishes "this mixer carries new signal" from "this mixer just adds capacity". |

## What each ablation tests

- **A1 (vs `conv`)**: the primary architecture-level falsifier. If
  the `event_delta_bilinear_accumulator` mixer does not beat `conv`
  on at least one multi-piece-interaction-dependent CRTK slice
  without regressing aggregate PR AUC, the mixer carries no
  architecture-level signal in this tower.
- **A2 (vs `attention`)**: protects against "any all-pairs token
  mixer beats conv on 64 squares". If `attention` matches the mixer,
  the win in A1 is generic global-mixing, not the specific FM-
  identity pair-term reduction.
- **A3 (vs primitive as head)**: tests transferability and isolates
  the cost of the learned soft occupancy. The source primitive was
  designed as an additive head on the i193 trunk with piece-plane
  occupancy; A3 tells us whether the same signal survives being
  repurposed as a token mixer with a learned occupancy indicator and
  a broadcast-back per-square fusion.
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
the mixer is buying its win with parameter count, not with the FM-
identity second-order pair-term sum. Drop also if the source-
primitive `first_order_only` ablation (run inside the BT4 tower by
zeroing the pair term `Q`) matches this idea on its declared target
slice, because then the pair term is not load-bearing and the mixer
degenerates into a first-order accumulator broadcast.

# Ablations

This folder is a controlled architecture study, not a primitive
study. The first-class ablations are *cross-idea* comparisons
against the matched `conv` and `attention` BT4 baselines and
against the source primitive
(`p024_event_symmetric_interaction_accumulator`) used as an
additive head rather than a token mixer.

## Ablation switches

| ID | Comparison run | What it changes |
|---|---|---|
| A1 | `bt4_conv_mixer` (sibling idea / baseline) | Replaces the `event_symmetric_interaction_accumulator` mixer with the original `conv` mixer, holding every other tower hyperparameter, optimizer setting, and data split fixed. Direct control for "is the `event_symmetric_interaction_accumulator` mixer better than a conv?". |
| A2 | `bt4_attention_mixer` (sibling idea / baseline) | Replaces the `event_symmetric_interaction_accumulator` mixer with a generic multi-head self-attention over the 64 squares. Direct control for "is the elementary-symmetric polynomial stack better than an explicit softmax attention map at matched widths?". |
| A3 | `p024_event_symmetric_interaction_accumulator` (source primitive idea) | Uses the primitive as an additive head over the i193 trunk instead of as the per-block spatial mixer. Tests whether the primitive transfers any of its signal through the BT4 tower at all, and isolates the cost of swapping the source piece-plane occupancy for the learned soft occupancy used inside the mixer. |
| A4 | Capacity-matched `bt4_conv_mixer` | If the `event_symmetric_interaction_accumulator` mixer adds parameters versus the conv mixer, A1 is repeated with conv `channels`/`num_blocks` increased to match the parameter count. Distinguishes "this mixer carries new signal" from "this mixer just adds capacity". |
| A5 | `order = 1` vs `order = 2` vs `order = 3` | Sweeps the mixer-local `order` knob. `order = 1` collapses the operator to a first-order EmbeddingBag-equivalent broadcast; `order = 3` enables third-order multiplicative interactions. Isolates which polynomial order is actually load-bearing for chess multi-piece tactics. |

## What each ablation tests

- **A1 (vs `conv`)**: the primary architecture-level falsifier. If
  the `event_symmetric_interaction_accumulator` mixer does not beat
  `conv` on at least one multi-piece-interaction-dependent CRTK
  slice without regressing aggregate PR AUC, the mixer carries no
  architecture-level signal in this tower.
- **A2 (vs `attention`)**: protects against "any all-pairs token
  mixer beats conv on 64 squares". If `attention` matches the mixer,
  the win in A1 is generic global-mixing, not the specific
  elementary-symmetric multiplicative interaction algebra.
- **A3 (vs primitive as head)**: tests transferability and isolates
  the cost of the learned soft occupancy. The source primitive was
  designed as an additive head on the i193 trunk with piece-plane
  occupancy; A3 tells us whether the same signal survives being
  repurposed as a token mixer with a learned occupancy indicator
  and a broadcast-back per-square fusion.
- **A4 (capacity match)**: distinguishes signal from FLOPs.
- **A5 (order sweep)**: distinguishes "the mixer's win comes from
  the second/third-order Hadamard interactions" from "the mixer's
  win comes from the first-order global sum (which conv lacks)". If
  `order = 1` matches `order = 2`/`order = 3` on the declared
  target slice, the higher-order states are not load-bearing and
  the operator degenerates into a global average broadcast.

## Falsification criteria

Promote (keep) this idea only if all hold on the held-out test
split:

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
  comparison, AND
- A5: the slice-level lift survives only at `order >= 2` (i.e.
  drops when `order = 1` is forced).

Drop if any one fails. Drop especially if A4 closes -- that means
the mixer is buying its win with parameter count, not with the
elementary-symmetric multiplicative interactions. Drop also if A5
shows `order = 1` matches `order = 2`/`order = 3` on the target
slice, because then the higher-order polynomial states are not
load-bearing and the mixer degenerates into a first-order
accumulator broadcast.

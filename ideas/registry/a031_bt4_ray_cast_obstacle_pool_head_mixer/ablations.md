# Ablations

This folder is a controlled architecture study, not a primitive
study. The first-class ablations are *cross-idea* comparisons against
the matched `conv` and `attention` BT4 baselines and against the
source primitive (`p026_ray_cast_obstacle_pool_head`) used as a
pooled additive head rather than a token mixer.

## Ablation switches

| ID | Comparison run | What it changes |
|---|---|---|
| A1 | `bt4_conv_mixer` (sibling idea / baseline) | Replaces the `ray_cast_obstacle_pool_head` mixer with the original `conv` mixer, holding every other tower hyperparameter, optimizer setting, and data split fixed. Direct control for "is the ray-pooled mixer better than a conv?". |
| A2 | `bt4_attention_mixer` (sibling idea / baseline) | Replaces the `ray_cast_obstacle_pool_head` mixer with a generic multi-head self-attention over the 64 squares. Direct control for "is the per-direction ray pooling better than an explicit softmax attention map at matched widths?". |
| A3 | `p026_ray_cast_obstacle_pool_head` (source primitive idea) | Uses the primitive as a pooled additive head over the i193 trunk instead of as the per-block spatial mixer. Tests whether the primitive transfers any of its signal through the BT4 tower at all, and isolates the cost of replacing the rule-exact piece-plane occupancy with a soft channel-derived occupancy proxy. |
| A4 | Capacity-matched `bt4_conv_mixer` | If the `ray_cast_obstacle_pool_head` mixer adds parameters versus the conv mixer, A1 is repeated with conv `channels`/`num_blocks` increased to match the parameter count. Distinguishes "this mixer carries new signal" from "this mixer just adds capacity". |
| A5 | `drop_occlusion` (in-mixer ablation) | Hold the soft occupancy `O := 0` so the running unblocked product never collapses; the geometric series runs over the full ray. Tests whether occlusion termination is load-bearing inside the BT4 tower or whether a plain decayed pool suffices. |
| A6 | `shuffle_directions` (in-mixer ablation) | Random per-pass permutation of the 8 direction offsets (with `gamma_d` permuted in lockstep). Tests whether direction-specific learning is load-bearing. |
| A7 | `zero_rays` (in-mixer ablation) | Force the per-direction accumulators to zero so the mixer projection sees only zeros, isolating whether the projection-plus-residual path alone carries the model's predictions. |

## What each ablation tests

- **A1 (vs `conv`)**: the primary architecture-level falsifier. If
  the `ray_cast_obstacle_pool_head` mixer does not beat `conv` on at
  least one CRTK slice without regressing aggregate PR AUC, the
  mixer carries no architecture-level signal in this tower.
- **A2 (vs `attention`)**: protects against "any all-pairs token
  mixer beats conv on 64 squares". If `attention` matches the mixer,
  the win in A1 is generic long-range mixing, not the specific
  per-direction ray-pooled structure.
- **A3 (vs primitive as head)**: tests transferability and isolates
  the cost of dropping the rule-exact piece-plane occupancy. The
  source primitive was designed as a pooled additive head on the
  i193 trunk reading the piece planes directly; A3 tells us whether
  the same signal survives being repurposed as a token mixer with
  a soft content-based occupancy proxy and a per-direction stack
  projection back to `C` channels.
- **A4 (capacity match)**: distinguishes signal from FLOPs.
- **A5 (`drop_occlusion`)**: localises the load-bearing component.
  If the occlusion-terminated geometric series and the unterminated
  geometric series perform identically, the rule-exact "stop at
  first blocker" structure is not what makes RayPool work; the
  win, if any, is in the per-direction decayed pool.
- **A6 (`shuffle_directions`)**: protects against "direction-
  invariance is fine". If shuffled directions match the unablated
  mixer, direction-specific learning is decorative.
- **A7 (`zero_rays`)**: the trivial sanity check.

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
per-direction ray-pooled structure. Drop also if A5
(`drop_occlusion`) matches this idea on its declared target slice,
because then the occlusion-termination is not load-bearing and the
mixer degenerates into a per-direction decayed global pool.

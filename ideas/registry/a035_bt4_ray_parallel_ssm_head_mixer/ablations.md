# Ablations

This folder is a controlled architecture study, not a primitive
study. The first-class ablations are *cross-idea* comparisons against
the matched `conv` and `attention` BT4 baselines and against the
source primitive (`p030_ray_parallel_ssm_head`) used as a pooled
additive head rather than a token mixer.

## Ablation switches

| ID | Comparison run | What it changes |
|---|---|---|
| A1 | `bt4_conv_mixer` (sibling idea / baseline) | Replaces the `ray_parallel_ssm_head` mixer with the original `conv` mixer, holding every other tower hyperparameter, optimizer setting, and data split fixed. Direct control for "is the selective state-space scan along chess rays better than a 3x3 conv pair?". |
| A2 | `bt4_attention_mixer` (sibling idea / baseline) | Replaces the `ray_parallel_ssm_head` mixer with a generic dense multi-head self-attention over the 64 squares. Direct control for "is the 8-direction selective state-space scan with per-channel A/B better than dense all-pairs attention at matched widths?". |
| A3 | `p030_ray_parallel_ssm_head` (source primitive idea) | Uses the primitive as a pooled additive head over the i193 trunk with the original mean-pool to a feature vector and gate / delta MLPs fusing with the four trunk diagnostics, instead of as the per-block spatial mixer with `y_total` returned directly. Tests whether the primitive transfers any of its signal through the BT4 tower at all, and isolates the cost of replacing the pooled feature-vector read-out with a per-square channel read-out. |
| A4 | Capacity-matched `bt4_conv_mixer` | If the `ray_parallel_ssm_head` mixer adds parameters versus the conv mixer (two `Conv2d(C -> NUM_DIRECTIONS * C, 1x1)` projections + the `(NUM_DIRECTIONS, C)` `C` parameter + the output `Conv2d(C -> C, 1x1)`), A1 is repeated with conv `channels`/`num_blocks` increased to match the parameter count. Distinguishes "this mixer carries new signal" from "this mixer just adds capacity". |
| A5 | `disable_selective_A` (in-mixer ablation) | Force `A = 0.5` everywhere (constant, non-input-conditioned). **Primary in-mixer falsifier** -- matches the source primitive's primary falsifier. Tests whether the input-conditioned retention is load-bearing or whether a constant geometric decay along the ray suffices. |
| A6 | `disable_selective_B` (in-mixer ablation) | Force `B = 0.5` everywhere (constant, non-input-conditioned). **Primary in-mixer falsifier** -- matches the source primitive's primary falsifier. Tests whether the input-conditioned injection is load-bearing or whether a constant injection rate suffices. |
| A7 | `no_directional_C` (in-mixer ablation) | Replace each per-direction `C[d]` with the mean across directions (so the 8 directions share one read-out vector). Tests whether direction-specific read-out is load-bearing. |
| A8 | `zero_ssm_features` (in-mixer ablation) | Force the mixer output to zero (`return torch.zeros_like(x)`); the BT4 block degenerates to a SqueezeExcite + residual block. Tests whether the routed-token output is load-bearing or whether the surrounding SqueezeExcite + residual stream is doing all the work in this tower. |
| A9 | `ray_length=1` (in-mixer ablation) | Set `max_ray_length = 1`; the scan reduces to a single step `h_1 = B * x`, equivalent to a per-(direction, square, channel) `B`-gated copy of the per-square features with no long-range accumulation. Tests whether the iterated long-range scan is load-bearing or whether a one-step injection suffices. |

## What each ablation tests

- **A1 (vs `conv`)**: the primary architecture-level falsifier. If
  the `ray_parallel_ssm_head` mixer does not beat `conv` on at
  least one CRTK slice without regressing aggregate PR AUC, the
  mixer carries no architecture-level signal in this tower.
- **A2 (vs `attention`)**: the protective control. If `attention`
  matches or beats the Ray-SSM mixer, the win in A1 is generic
  all-pairs long-range mixing, not the specific 8-direction
  selective state-space prior. Dense attention also moves
  information globally but does so without the chess-ray prior;
  the Ray-SSM mixer bakes the 8 chess directions and per-channel
  retention/injection into the operator.
- **A3 (vs primitive as head with original read-out)**: tests
  transferability and isolates the cost of replacing the pooled
  feature-vector read-out with a per-direction channel read-out
  summed across directions. The source primitive was designed as
  a pooled additive head on the i193 trunk; A3 tells us whether
  the same signal survives being repurposed as a token mixer.
- **A4 (capacity match)**: distinguishes signal from FLOPs. The
  mixer adds the two `Conv2d(C -> 8 * C, 1x1)` projections for A
  and B plus the `(8, C)` `C` parameter plus the output 1x1 conv,
  versus the conv mixer's two 3x3 convs. The conv baseline must
  be sized to match the parameter count before declaring an A1
  win.
- **A5 (`disable_selective_A`)**: the primary in-mixer falsifier
  (matches the source primitive's primary falsifier). If A5
  matches the unablated mixer, the input-conditioned retention is
  decorative and the operator is just a constant geometric prefix
  sum -- equivalent to `bt4_ray_cast_obstacle_pool_head_mixer`
  / `p026` modulated by a constant injection.
- **A6 (`disable_selective_B`)**: the second primary in-mixer
  falsifier. If A6 matches the unablated mixer, the input-
  conditioned injection is decorative.
- **A7 (`no_directional_C`)**: tests whether *direction-specific*
  read-out is load-bearing. If a shared `C` across directions
  matches the unablated mixer, the per-direction projection is
  interchangeable and the 8-direction prior is decorative -- any
  single learned read-out vector would work.
- **A8 (`zero_ssm_features`)**: localises the load-bearing
  component. If A8 matches the unablated mixer, the routed-token
  output is decorative and the surrounding SqueezeExcite +
  residual stream is doing all the work; the BT4 block reduces
  to a parameter-cheap SE-residual block and the whole architecture
  study collapses.
- **A9 (`ray_length=1`)**: tests whether the iterated long-range
  scan is load-bearing. If `max_ray_length = 1` matches the
  unablated mixer, the operator is effectively a one-step injection
  and the long-range ray prior is decorative -- a per-direction
  gated local read would suffice.

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
selective state-space + per-channel A/B prior. Drop also if either
A5 (`disable_selective_A`) or A6 (`disable_selective_B`) matches
this idea on its declared target slice, because then the input-
conditioned A or B -- the load-bearing motivation for Ray-SSM --
is not load-bearing in the mixer adaptation, and the operator
degenerates into a constant-coefficient prefix sum (equivalent to
`p026` RayPool with a constant injection). Drop also if A9
(`ray_length=1`) matches: the long-range scan claim is then
decorative and a one-step injection would have sufficed.

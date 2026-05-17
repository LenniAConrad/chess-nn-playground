# Ablations

This folder is a controlled architecture study, not a primitive
study. The first-class ablations are *cross-idea* comparisons against
the matched `conv` and `attention` BT4 baselines and against the
source primitive (`p029_occlusion_aware_ray_scan_head`) used as a
pooled additive head rather than a token mixer.

## Ablation switches

| ID | Comparison run | What it changes |
|---|---|---|
| A1 | `bt4_conv_mixer` (sibling idea / baseline) | Replaces the `occlusion_aware_ray_scan_head` mixer with the original `conv` mixer, holding every other tower hyperparameter, optimizer setting, and data split fixed. Direct control for "is the selective ray scan better than a 3x3 conv pair?". |
| A2 | `bt4_attention_mixer` (sibling idea / baseline) | Replaces the `occlusion_aware_ray_scan_head` mixer with a generic dense multi-head self-attention over the 64 squares. Direct control for "is the 8-direction selective ray scan with content-dependent termination better than dense all-pairs attention at matched widths?". |
| A3 | `p029_occlusion_aware_ray_scan_head` (source primitive idea) | Uses the primitive as a pooled additive head over the i193 trunk with the original `Conv2d(C -> 1, 1x1)` per-direction read-out and the final mean-pool to a scalar delta logit, instead of as the per-block spatial mixer with per-direction `Conv2d(C -> C, 1x1)` and a sum-across-directions. Tests whether the primitive transfers any of its signal through the BT4 tower at all, and isolates the cost of replacing the pooled scalar read-out with a per-square channel read-out. |
| A4 | Capacity-matched `bt4_conv_mixer` | If the `occlusion_aware_ray_scan_head` mixer adds parameters versus the conv mixer (eight `Conv2d(C -> C, 1x1)` projections + the `Conv2d(C -> 8)` gate), A1 is repeated with conv `channels`/`num_blocks` increased to match the parameter count. Distinguishes "this mixer carries new signal" from "this mixer just adds capacity". |
| A5 | `disable_blocker_gate` (in-mixer ablation) | Clamp `g = 1` everywhere (equivalent to passing the gate through `nn.Identity`); the scan degenerates to a plain geometric prefix sum along each direction. **Primary in-mixer falsifier** -- matches the source primitive's primary falsifier. Tests whether the content-dependent termination is load-bearing or whether the operator is just a directional prefix sum. |
| A6 | `zero_oars_features` (in-mixer ablation) | Force the mixer output to zero (`return torch.zeros_like(x)`); the BT4 block degenerates to a SqueezeExcite + residual block. Tests whether the routed-token output is load-bearing or whether the surrounding SqueezeExcite + residual stream is doing all the work in this tower. |
| A7 | `shuffle_directions` (in-mixer ablation) | Random permutation of the 8 directions before each forward (re-index `RAY_DIRECTIONS` and the per-direction projections). Decouples direction-specific learning; matches the source primitive's `shuffle_directions` falsifier. |
| A8 | `ray_length=1` (in-mixer ablation) | Set `max_ray_length = 1`; the scan reduces to a single shift plus gate, equivalent to a per-(direction, square) gated local operator with no long-range accumulation. Tests whether the iterated long-range scan is load-bearing or whether a one-step gated neighbour read suffices. |

## What each ablation tests

- **A1 (vs `conv`)**: the primary architecture-level falsifier. If
  the `occlusion_aware_ray_scan_head` mixer does not beat `conv` on
  at least one CRTK slice without regressing aggregate PR AUC, the
  mixer carries no architecture-level signal in this tower.
- **A2 (vs `attention`)**: the protective control. If `attention`
  matches or beats the OARS mixer, the win in A1 is generic
  all-pairs long-range mixing, not the specific 8-direction
  selective-scan prior. Dense attention also moves information
  globally but does so without the chess-ray prior; the OARS mixer
  bakes the 8 chess directions and content-dependent termination
  into the operator.
- **A3 (vs primitive as head with original read-out)**: tests
  transferability and isolates the cost of replacing the pooled
  scalar read-out with a per-direction channel read-out summed
  across directions. The source primitive was designed as a pooled
  additive head on the i193 trunk; A3 tells us whether the same
  signal survives being repurposed as a token mixer.
- **A4 (capacity match)**: distinguishes signal from FLOPs. The
  mixer adds the eight per-direction `Conv2d(C -> C, 1x1)`
  projections plus the `Conv2d(C -> 8)` gate versus the conv
  mixer's two 3x3 convs. The conv baseline must be sized to match
  the parameter count before declaring an A1 win.
- **A5 (`disable_blocker_gate`)**: the primary in-mixer falsifier
  (matches the source primitive's primary falsifier). If A5
  matches the unablated mixer, the content-dependent termination
  is decorative and the operator is just a directional prefix
  sum -- equivalent to `bt4_ray_cast_obstacle_pool_head_mixer`
  / `p026` without occupancy gating.
- **A6 (`zero_oars_features`)**: localises the load-bearing
  component. If A6 matches the unablated mixer, the routed-token
  output is decorative and the surrounding SqueezeExcite +
  residual stream is doing all the work; the BT4 block reduces
  to a parameter-cheap SE-residual block and the whole architecture
  study collapses.
- **A7 (`shuffle_directions`)**: tests whether *direction-specific*
  learning is load-bearing. If shuffled directions match the
  unablated mixer, the per-direction projections are
  interchangeable and the 8-direction prior is decorative -- any
  set of 8 learned axial operators would work.
- **A8 (`ray_length=1`)**: tests whether the iterated long-range
  scan is load-bearing. If `max_ray_length = 1` matches the
  unablated mixer, the operator is effectively a one-step gated
  local neighbour read and the long-range ray prior is decorative
  -- a 3x3 conv with eight directional channels would suffice.

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
selective-scan + content-dependent termination prior. Drop also if
A5 (`disable_blocker_gate`) matches this idea on its declared
target slice, because then the content-dependent termination --
the load-bearing motivation for OARS -- is not load-bearing in the
mixer adaptation, and the operator degenerates into a directional
prefix sum (equivalent to `p026` RayPool without occupancy
gating). Drop also if A8 (`ray_length=1`) matches: the long-range
scan claim is then decorative and a one-step gated neighbour read
would have sufficed.

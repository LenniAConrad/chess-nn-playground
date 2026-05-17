# Ablations

This folder is a controlled architecture study, not a primitive
study. The first-class ablations are *cross-idea* comparisons against
the matched `conv` and `attention` BT4 baselines and against the
source primitive (`p033_move_kernel_operator`) used as an
i193-additive head rather than a token mixer.

## Ablation switches

| ID | Comparison run | What it changes |
|---|---|---|
| A1 | `bt4_conv_mixer` (sibling idea / baseline) | Replaces the `move_kernel_operator` mixer with the original `conv` mixer, holding every other tower hyperparameter, optimizer setting, and data split fixed. Direct control for "is the per-move-type masked aggregation better than a 3x3 conv pair?". |
| A2 | `bt4_attention_mixer` (sibling idea / baseline) | Replaces the `move_kernel_operator` mixer with a generic dense multi-head self-attention over the 64 squares. Direct control for "is the chess-rule per-move-type decomposition better than dense all-pairs attention at matched widths?". |
| A3 | `p033_move_kernel_operator` (source primitive idea) | Uses the primitive as an i193-additive head with the original `Linear(13)` per-square seed feature, the pooled feature-vector head, and the gate / delta MLPs over the i193 trunk diagnostics, instead of as the per-block spatial mixer with the per-square channel feature substituted for the seed and `Y` returned directly. Tests whether the primitive transfers any of its signal through the BT4 tower at all, and isolates the cost of replacing the per-square seed projection with the trunk's channel features and the pooled feature-vector read-out with a per-square channel read-out. |
| A4 | Capacity-matched `bt4_conv_mixer` | If the `move_kernel_operator` mixer adds parameters versus the conv mixer (LayerNorm, `T = 6` per-type `Linear(C, C)` projections, and the `out_proj Linear(C, C)`), A1 is repeated with conv `channels`/`num_blocks` increased to match the parameter count. Distinguishes "this mixer carries new signal" from "this mixer just adds capacity". |
| A5 | `shared_kernel` (in-mixer ablation) | Force every `W_t := W` to a single shared projection (the per-type masked aggregations still differ, but they share the same kernel). **Primary in-mixer falsifier** -- matches the source primitive's primary falsifier. If A5 matches the unablated mixer, the per-move-type matrix specialisation is decorative and the load-bearing factor is the static chess-rule reach prior alone -- the operator collapses to a single linear projection applied to the union of static reach geometries. |
| A6 | `scalar_per_type` (in-mixer ablation) | Replace each `W_t` by `w_t * I` (a scalar gain per type). The operator reduces to a content-independent weighted sum of masked aggregations under a shared per-square feature. Tests whether the matrix capacity per type is load-bearing or whether a learned per-type scalar mixture over the static reach geometries suffices. |
| A7 | `shuffle_features` (in-mixer ablation) | In-batch permutation of the seed features so the per-square input is decoupled from the position. Mirrors the source primitive's `shuffle_features` falsifier. If A7 matches the unablated mixer, the rule-derived per-square input carries no signal beyond what a per-type scalar mixture would extract and the operator's wins come from the per-type density structure alone. |
| A8 | `uniform_adjacency` (in-mixer ablation) | Replace every `M_t` with the all-ones (minus identity) adjacency. The chess-rule per-type geometry is erased; the operator becomes a sum of `T` shared linear maps over the full 64-token bag. Tests whether the rule-derived edges matter at all beyond the per-type matrix specialisation. |
| A9 | `zero_mixer_output` (in-mixer ablation) | Force the mixer output to zero (`return torch.zeros_like(x)`); the BT4 block degenerates to a SqueezeExcite + residual block. Tests whether the routed-token output is load-bearing or whether the surrounding SqueezeExcite + residual stream is doing all the work in this tower. |

## What each ablation tests

- **A1 (vs `conv`)**: the primary architecture-level falsifier. If
  the `move_kernel_operator` mixer does not beat `conv` on at least
  one CRTK slice without regressing aggregate PR AUC, the mixer
  carries no architecture-level signal in this tower.
- **A2 (vs `attention`)**: the protective control. If `attention`
  matches or beats the MKO mixer, the win in A1 is generic all-pairs
  long-range mixing, not the specific chess-rule per-move-type
  decomposition prior. Dense attention can express any per-type
  mask pattern as a soft-attention map; the MKO mixer commits to
  the chess-rule per-type partition at construction time.
- **A3 (vs primitive as head with original Linear(13) seed and
  pooled trunk-fusion read-out)**: tests transferability and
  isolates the cost of replacing the per-square `Linear(13)` seed
  projection with the trunk's channel feature vector and replacing
  the pooled feature-vector read-out with a per-square channel
  read-out. The source primitive was designed as an i193-additive
  head; A3 tells us whether the same signal survives being
  repurposed as a token mixer at all, and which of the two
  compromises (seed substitution, per-square read-out) is the
  load-bearing cost.
- **A4 (capacity match)**: distinguishes signal from FLOPs. The
  mixer adds LayerNorm, `T = 6` per-type `Linear(C, C)` projections,
  and the `out_proj Linear(C, C)` versus the conv mixer's two 3x3
  convs. The conv baseline must be sized to match the parameter
  count before declaring an A1 win.
- **A5 (`shared_kernel`)**: the primary in-mixer falsifier (matches
  the source primitive's primary falsifier). If A5 matches the
  unablated mixer, the per-move-type matrix specialisation is
  decorative inside the BT4 tower's residual stack -- the operator
  collapses to a single linear projection `W` applied to the union
  of static chess-rule reach geometries `sum_t M_t`, and the whole
  per-type weight-sharing claim collapses.
- **A6 (`scalar_per_type`)**: the second in-mixer falsifier
  (mirrors the source primitive's `scalar_per_type` falsifier). If
  A6 matches the unablated mixer, the matrix capacity per type is
  decorative and a per-type scalar gain on the masked sum suffices
  -- the operator becomes a learned soft per-type mixture under a
  single shared per-square feature.
- **A7 (`shuffle_features`)**: tests whether the rule-derived per-
  square input carries signal. If A7 matches the unablated mixer,
  the per-square feature input is decorative and the operator's
  wins come from the per-type density structure of the masks
  alone.
- **A8 (`uniform_adjacency`)**: tests whether the chess-rule per-
  type geometry is load-bearing. If A8 matches the unablated mixer,
  the per-type reach prior is decorative -- a sum of `T` shared
  linear maps over the full 64-token bag would suffice, and the
  operator collapses to a learned mixture of generic global mixers.
- **A9 (`zero_mixer_output`)**: localises the load-bearing
  component. If A9 matches the unablated mixer, the routed-token
  output is decorative and the surrounding SqueezeExcite +
  residual stream is doing all the work; the BT4 block reduces to a
  parameter-cheap SE-residual block and the whole architecture
  study collapses.

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
  lower per-block FLOPs -- MKO is roughly `6x` attention's FLOPs,
  so this requires a cheaper MKO variant), AND
- A4: the slice-level lift survives the capacity-matched conv
  comparison.

Drop if any one fails. Drop especially if A4 closes -- that means
the mixer is buying its win with parameter count, not with the
per-move-type matrix specialisation prior. Drop also if A5
(`shared_kernel`) matches this idea on its declared target slice,
because then the per-move-type matrix specialisation -- the
load-bearing motivation for MKO -- is not load-bearing in the mixer
adaptation, and the operator degenerates into a single shared
projection over the union reach geometry. Drop also if A6
(`scalar_per_type`) matches: the per-type matrix capacity is then
decorative and a content-independent per-type scalar mixture
suffices. Drop also if A8 (`uniform_adjacency`) matches: the
chess-rule reach geometry is then decorative and the wins come
from the per-type linear-map mixture over the full 64-token bag.

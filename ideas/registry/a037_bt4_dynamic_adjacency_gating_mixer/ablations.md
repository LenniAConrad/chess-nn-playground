# Ablations

This folder is a controlled architecture study, not a primitive
study. The first-class ablations are *cross-idea* comparisons against
the matched `conv` and `attention` BT4 baselines and against the
source primitive (`p032_dynamic_adjacency_gating`) used as an
i193-additive head rather than a token mixer.

## Ablation switches

| ID | Comparison run | What it changes |
|---|---|---|
| A1 | `bt4_conv_mixer` (sibling idea / baseline) | Replaces the `dynamic_adjacency_gating` mixer with the original `conv` mixer, holding every other tower hyperparameter, optimizer setting, and data split fixed. Direct control for "is the per-move-type masked aggregation better than a 3x3 conv pair?". |
| A2 | `bt4_attention_mixer` (sibling idea / baseline) | Replaces the `dynamic_adjacency_gating` mixer with a generic dense multi-head self-attention over the 64 squares. Direct control for "is the chess-rule per-move-type decomposition better than dense all-pairs attention at matched widths?". |
| A3 | `p032_dynamic_adjacency_gating` (source primitive idea) | Uses the primitive as an i193-additive head with the original blocker-resolved, position-specific binary adjacency, the pawn_push and pawn_capture move-type slots, the pooled feature-vector head, and the gate/delta MLPs over the i193 trunk diagnostics, instead of as the per-block spatial mixer with `Y` returned directly. Tests whether the primitive transfers any of its signal through the BT4 tower at all, and isolates the cost of (a) replacing the occupancy-blocked position-specific adjacency with the static chess-rule reach geometry, (b) dropping the pawn move-type slots, and (c) replacing the pooled feature-vector read-out with a per-square channel read-out. |
| A4 | Capacity-matched `bt4_conv_mixer` | If the `dynamic_adjacency_gating` mixer adds parameters versus the conv mixer (LayerNorm, `T = 6` per-type projections, gate `Linear(C, T)`, `out_proj`), A1 is repeated with conv `channels`/`num_blocks` increased to match the parameter count. Distinguishes "this mixer carries new signal" from "this mixer just adds capacity". |
| A5 | `single_move_type` (in-mixer ablation) | Force every `W_t := W` to a single shared projection (the per-type masked aggregations still differ, but they share the same kernel). **Primary in-mixer falsifier** -- matches the source primitive's primary falsifier. If A5 matches the unablated mixer, the per-move-type kernel specialisation is decorative and the load-bearing factor is the static chess-rule reach prior alone. |
| A6 | `uniform_gate` (in-mixer ablation) | Force `g_t(Z_i) := 1` for all `t, i` (skip the sigmoid gate). The operator reduces to a fixed per-type linear combination of masked aggregations. Tests whether the per-square per-type content gating is load-bearing or whether a constant equal-weight sum across types suffices. |
| A7 | `uniform_adjacency` (in-mixer ablation) | Replace every `M_t` with the all-ones (minus identity) adjacency. The chess-rule per-type geometry is erased; the operator becomes a content-gated mixture of `T` shared linear maps over the full 64-token bag. Tests whether the rule-derived edges matter at all beyond the gating. |
| A8 | `shuffle_adjacency` (in-mixer ablation) | Permute the rows/cols of every `M_t` randomly per batch (a different permutation per type). Decouples the chess-rule reach geometry from the chess-board square indexing while preserving the per-type density. If A8 matches the unablated mixer, the chess-rule reach prior is decorative -- any per-type random graph with comparable density would work. |
| A9 | `zero_mixer_output` (in-mixer ablation) | Force the mixer output to zero (`return torch.zeros_like(x)`); the BT4 block degenerates to a SqueezeExcite + residual block. Tests whether the routed-token output is load-bearing or whether the surrounding SqueezeExcite + residual stream is doing all the work in this tower. |

## What each ablation tests

- **A1 (vs `conv`)**: the primary architecture-level falsifier. If
  the `dynamic_adjacency_gating` mixer does not beat `conv` on at
  least one CRTK slice without regressing aggregate PR AUC, the
  mixer carries no architecture-level signal in this tower.
- **A2 (vs `attention`)**: the protective control. If `attention`
  matches or beats the DAG mixer, the win in A1 is generic all-pairs
  long-range mixing, not the specific chess-rule per-move-type
  decomposition prior. Dense attention can express any per-type
  pattern as a soft-attention map; the DAG mixer commits to the
  chess-rule per-type partition at construction time.
- **A3 (vs primitive as head with original blocker-resolved adjacency
  and pawn types)**: tests transferability and isolates the cost of
  replacing the occupancy-blocked position-specific adjacency with
  the static chess-rule reach geometry, dropping the pawn move-type
  slots, and replacing the pooled feature-vector read-out with a
  per-square channel read-out. The source primitive was designed as
  an i193-additive head; A3 tells us whether the same signal
  survives being repurposed as a token mixer at all, and which of
  the three honest compromises (no occupancy blocking, no pawn
  types, no pooled read-out) is the load-bearing cost.
- **A4 (capacity match)**: distinguishes signal from FLOPs. The
  mixer adds LayerNorm, `T = 6` per-type `Linear(C, C)` projections,
  the gate `Linear(C, T)`, and the `out_proj Linear(C, C)` versus
  the conv mixer's two 3x3 convs. The conv baseline must be sized
  to match the parameter count before declaring an A1 win.
- **A5 (`single_move_type`)**: the primary in-mixer falsifier
  (matches the source primitive's primary falsifier). If A5 matches
  the unablated mixer, the per-move-type kernel specialisation is
  decorative inside the BT4 tower's residual stack -- the operator
  is a content-gated aggregation over the union of static chess-
  rule reach geometries with one shared linear kernel, and the
  whole per-type specialisation claim collapses.
- **A6 (`uniform_gate`)**: the second in-mixer falsifier. If A6
  matches the unablated mixer, the per-square per-type sigmoid gate
  is decorative and the operator reduces to a fixed per-type linear
  combination of masked aggregations -- a content-independent per-
  type 1x1 conv equivalent after the static reach aggregation.
- **A7 (`uniform_adjacency`)**: tests whether the chess-rule per-
  type geometry is load-bearing. If A7 matches the unablated mixer,
  the per-type reach prior is decorative -- a content-gated mixture
  of `T` shared linear maps over the full 64-token bag would
  suffice, and the operator collapses to a learned soft per-type
  mixture of generic global mixers.
- **A8 (`shuffle_adjacency`)**: tests whether the chess-rule reach
  prior is load-bearing in a more conservative way than A7. If A8
  matches the unablated mixer, any per-type random graph with
  comparable density would work and the wins come from the per-
  type density structure rather than the chess-specific reach
  geometry.
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
  lower per-block FLOPs -- DAG is roughly `6x` attention's FLOPs,
  so this requires a cheaper DAG variant), AND
- A4: the slice-level lift survives the capacity-matched conv
  comparison.

Drop if any one fails. Drop especially if A4 closes -- that means
the mixer is buying its win with parameter count, not with the
per-move-type masked aggregation prior. Drop also if A5
(`single_move_type`) matches this idea on its declared target
slice, because then the per-move-type kernel specialisation -- the
load-bearing motivation for DAG -- is not load-bearing in the mixer
adaptation, and the operator degenerates into a content-gated
single-kernel aggregation over the union reach geometry. Drop also
if A6 (`uniform_gate`) matches: the per-square per-type sigmoid gate
is then decorative and the operator reduces to a fixed per-type
linear combination. Drop also if A8 (`shuffle_adjacency`) matches:
the chess-rule reach geometry is then decorative and the wins come
from the per-type density structure of any random graph.

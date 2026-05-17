# Ablations

This folder is a controlled architecture study, not a primitive
study. The first-class ablations are *cross-idea* comparisons against
the matched `conv` and `attention` BT4 baselines and against the
source primitive (`p034_octilinear_selective_scan`) used as a pooled
additive head rather than a token mixer.

## Ablation switches

| ID | Comparison run | What it changes |
|---|---|---|
| A1 | `bt4_conv_mixer` (sibling idea / baseline) | Replaces the `octilinear_selective_scan` mixer with the original `conv` mixer, holding every other tower hyperparameter, optimizer setting, and data split fixed. Direct control for "is the 8-direction Mamba-style selective state-space scan better than a 3x3 conv pair?". |
| A2 | `bt4_attention_mixer` (sibling idea / baseline) | Replaces the `octilinear_selective_scan` mixer with a generic dense multi-head self-attention over the 64 squares. Direct control for "is the 8-direction selective state-space scan with per-channel A/B better than dense all-pairs attention at matched widths?". |
| A3 | `p034_octilinear_selective_scan` (source primitive idea) | Uses the primitive as a pooled additive head over the i193 trunk with the original `Linear(13) -> d` projection of the simple_18 piece planes, own-piece-weighted mean + global mean pool, and gated delta-logit MLP fusing with the i193 base logit, instead of as the per-block spatial mixer with the per-square fused feature returned directly. Tests whether the primitive transfers any of its signal through the BT4 tower at all, and isolates the cost of replacing the piece-plane projection with a generic channel input and the pooled scalar read-out with a per-square channel read-out. |
| A4 | Capacity-matched `bt4_conv_mixer` | If the `octilinear_selective_scan` mixer adds parameters versus the conv mixer (eight pairs of `Linear(C -> C)` projections for `A_k` and `B_k`, plus the `LayerNorm(8*C) + Linear(8*C -> C)` fuse and the input `LayerNorm(C)`), A1 is repeated with conv `channels`/`num_blocks` increased to match the parameter count. Distinguishes "this mixer carries new signal" from "this mixer just adds capacity". |
| A5 | `fixed_transition` (in-mixer ablation) | Force `A_k = 0.5` everywhere per channel (constant, non-input-conditioned, no `sigmoid(A_proj(x))` dependence on the per-square feature). **Primary in-mixer falsifier** -- mirrors the source primitive's `fixed_transition` falsifier. Tests whether the input-conditioned retention is load-bearing or whether a constant per-channel geometric decay along the scan path suffices. |
| A6 | `single_direction` (in-mixer ablation) | Keep only the `E` direction scan and zero the other seven per-direction outputs before the 8 * C -> C fuse (so the fuser sees the seven other directions as identically zero). **Primary in-mixer falsifier** -- mirrors the source primitive's `single_direction` falsifier. Tests whether the 8-direction decomposition is load-bearing or whether a single rightward scan suffices. |
| A7 | `shuffle_features` (in-mixer ablation) | Apply a batch-permutation to the per-square feature tensor seen by `A_proj` and `B_proj` only (the residual stream and the path tables are untouched). Mirrors the source primitive's `shuffle_features` falsifier. Tests whether the gate is content-conditioned (i.e., the selectivity actually reads the per-square feature) or whether it has degenerated into a position-independent constant. |
| A8 | `zero_oss_features` (in-mixer ablation) | Force the mixer output to zero (`return torch.zeros_like(x)`); the BT4 block degenerates to a SqueezeExcite + residual block. Tests whether the routed-token output is load-bearing or whether the surrounding SqueezeExcite + residual stream is doing all the work in this tower. |

## What each ablation tests

- **A1 (vs `conv`)**: the primary architecture-level falsifier. If
  the `octilinear_selective_scan` mixer does not beat `conv` on at
  least one CRTK slice without regressing aggregate PR AUC, the
  mixer carries no architecture-level signal in this tower.
- **A2 (vs `attention`)**: the protective control. If `attention`
  matches or beats the OSS mixer, the win in A1 is generic
  all-pairs long-range mixing, not the specific 8-direction
  selective state-space prior. Dense attention also moves
  information globally but does so without the chess-ray prior;
  the OSS mixer bakes the 8 chess directions and per-channel
  retention/injection into the operator.
- **A3 (vs primitive as head with original read-out)**: tests
  transferability and isolates the cost of (i) replacing the
  piece-plane `Linear(13)` projection with a generic channel input
  and (ii) replacing the pooled scalar read-out with a per-square
  channel read-out. The source primitive was designed as a pooled
  additive head on the i193 trunk; A3 tells us whether the same
  signal survives being repurposed as a token mixer with both
  changes simultaneously.
- **A4 (capacity match)**: distinguishes signal from FLOPs. The
  mixer adds eight pairs of `Linear(C -> C)` projections for `A_k`
  and `B_k`, plus the `LayerNorm(8*C) + Linear(8*C -> C)` fuse and
  an input `LayerNorm(C)`, versus the conv mixer's two 3x3 convs.
  The conv baseline must be sized to match the parameter count
  before declaring an A1 win.
- **A5 (`fixed_transition`)**: the primary in-mixer falsifier
  (matches the source primitive's primary falsifier). If A5
  matches the unablated mixer, the input-conditioned retention is
  decorative and the operator is just a constant geometric prefix
  sum of `B_k(x_t) * x_t` along each scan path -- equivalent to a
  per-direction non-selective injection.
- **A6 (`single_direction`)**: the second primary in-mixer
  falsifier. If A6 matches the unablated mixer, the 8-direction
  decomposition is decorative and a single rightward scan suffices;
  the chess-ray prior is not load-bearing inside the BT4 tower.
- **A7 (`shuffle_features`)**: tests whether the selectivity gate
  reads the per-square feature at all. If A7 matches the unablated
  mixer, the gate has degenerated into a position-independent
  constant (effectively A5 with a fitted constant) and the
  data-dependent selectivity claim is decorative.
- **A8 (`zero_oss_features`)**: localises the load-bearing
  component. If A8 matches the unablated mixer, the routed-token
  output is decorative and the surrounding SqueezeExcite +
  residual stream is doing all the work; the BT4 block reduces to
  a parameter-cheap SE-residual block and the whole architecture
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
  lower per-block FLOPs), AND
- A4: the slice-level lift survives the capacity-matched conv
  comparison.

Drop if any one fails. Drop especially if A4 closes -- that means
the mixer is buying its win with parameter count, not with the
selective state-space + 8-direction chess-ray prior. Drop also if
A5 (`fixed_transition`) matches this idea on its declared target
slice, because then the input-conditioned retention is decorative
and the operator degenerates into a constant geometric prefix sum
of `B_k(x_t) * x_t`. Drop if A6 (`single_direction`) matches: the
8-direction decomposition is then decorative and the chess-ray
prior is not load-bearing. Drop if A7 (`shuffle_features`) matches:
the gate is not reading content and the data-dependent selectivity
claim is decorative.

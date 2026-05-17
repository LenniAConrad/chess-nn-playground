# Ablations

This folder is a controlled architecture study, not a primitive
study. The first-class ablations are *cross-idea* comparisons against
the matched `conv` and `attention` BT4 baselines and against the
source primitive (`p028_incremental_latent_accumulator_head`) used as
a pooled additive head rather than a token mixer.

## Ablation switches

| ID | Comparison run | What it changes |
|---|---|---|
| A1 | `bt4_conv_mixer` (sibling idea / baseline) | Replaces the `incremental_latent_accumulator_head` mixer with the original `conv` mixer, holding every other tower hyperparameter, optimizer setting, and data split fixed. Direct control for "is the king-anchored accumulator better than a 3x3 conv pair?". |
| A2 | `bt4_attention_mixer` (sibling idea / baseline) | Replaces the `incremental_latent_accumulator_head` mixer with a generic dense multi-head self-attention over the 64 squares. Direct control for "is the permutation-structured pooled accumulator + soft-argmax anchor + non-linear lift better than dense all-pairs attention at matched widths?". |
| A3 | `p028_incremental_latent_accumulator_head` (source primitive idea) | Uses the primitive as a pooled additive head over the i193 trunk with the *rule-exact* `(12, 64)` piece-plane indicator and the rule-exact own-king square, instead of as the per-block spatial mixer with a learned soft-argmax anchor and a learned channel-feature accumulator. Tests whether the primitive transfers any of its signal through the BT4 tower at all, and isolates the cost of replacing the rule-exact king-square + piece-type indicator readout with a learned soft anchor over a learned channel-feature accumulator. |
| A4 | Capacity-matched `bt4_conv_mixer` | If the `incremental_latent_accumulator_head` mixer adds parameters versus the conv mixer, A1 is repeated with conv `channels`/`num_blocks` increased to match the parameter count. Distinguishes "this mixer carries new signal" from "this mixer just adds capacity". |
| A5 | `zero_global_accumulator` (in-mixer ablation) | Hold the global accumulator `h_global = 0` (skip the broadcast-back of the global stream). Tests whether the permutation-invariant global sum-pool is load-bearing or whether the king-anchor stream alone carries the signal. |
| A6 | `zero_king_accumulator` (in-mixer ablation) | Hold the king-anchored accumulator `h_king = 0` (skip the broadcast-back of the king stream and the anchor-conditioned table). Tests whether the king-anchor structure is load-bearing or whether the operator is just a non-linear lift over a pooled global summary. **Primary in-mixer falsifier** -- matches the source primitive's primary falsifier. |
| A7 | `linear_only` (in-mixer ablation) | Replace the `phi = LayerNorm -> Linear -> GELU -> Linear` lift with `phi = Linear(2*latent_dim + C -> C)` (drop the GELU non-linearity). Tests whether the non-linear lift is load-bearing or whether a pure linear pooled-then-broadcast operator captures the same signal. |
| A8 | `shuffle_square_order` (in-mixer ablation) | Random column permutation of the per-square bias `global_square` (and the `king_anchor_table`) before each forward. Decouples the per-square learned structure from the actual board squares; matches the source primitive's `shuffle_square_order` falsifier. |

## What each ablation tests

- **A1 (vs `conv`)**: the primary architecture-level falsifier. If
  the `incremental_latent_accumulator_head` mixer does not beat
  `conv` on at least one CRTK slice without regressing aggregate
  PR AUC, the mixer carries no architecture-level signal in this
  tower.
- **A2 (vs `attention`)**: the protective control. If `attention`
  matches or beats the accumulator mixer, the win in A1 is generic
  all-pairs long-range mixing, not the specific permutation-
  structured pooled-accumulate-then-broadcast prior. This is the
  most informative control for this idea, since dense attention
  also pools information globally but does so per-token rather
  than via a shared board-level latent.
- **A3 (vs primitive as head with rule-exact king-square + piece-
  type indicator)**: tests transferability and isolates the cost of
  replacing the rule-exact king-square + per-(piece-type, square)
  embedding table with a learned soft anchor over a learned
  channel-feature accumulator. The source primitive was designed as
  a pooled additive head on the i193 trunk reading the piece planes
  and the own-king square directly to construct
  `h_global, h_king`; A3 tells us whether the same signal survives
  being repurposed as a token mixer with a soft-anchor-plus-learned-
  channel-feature proxy.
- **A4 (capacity match)**: distinguishes signal from FLOPs. The
  mixer adds the `(64, latent_dim)` per-square bias, the `(64,
  latent_dim)` anchor table, the saliency conv, the two
  `Linear(C -> latent_dim)` projections, and the `phi` MLP versus
  the conv mixer's two 3x3 convs. The conv baseline must be sized
  to match the parameter count before declaring an A1 win.
- **A5 (`zero_global_accumulator`)**: localises the load-bearing
  component. If A5 matches the unablated mixer, the global stream
  is decorative and only the king-anchor stream + per-square local
  lift carries signal.
- **A6 (`zero_king_accumulator`)**: the primary in-mixer falsifier
  (matches the source primitive's primary falsifier). If A6 matches
  the unablated mixer, the king-anchor structure is decorative and
  the mixer degenerates into a non-linear lift over a pooled global
  summary -- equivalent to a global-average-pool + broadcast +
  MLP block.
- **A7 (`linear_only`)**: tests whether the `phi` non-linearity is
  load-bearing. If A7 matches the unablated mixer, the operator
  reduces to a per-token linear projection over a pooled global +
  king summary -- a strictly linear pooled-then-broadcast operator
  that the surrounding SqueezeExcite + ReLU may absorb.
- **A8 (`shuffle_square_order`)**: protects against "any per-square
  bias structure works". If shuffled per-square biases match the
  unablated mixer, the *content* of the learned per-square structure
  is decorative -- only the existence of *some* per-square bias
  matters.

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
permutation-structured pooled-accumulate-then-broadcast prior.
Drop also if A6 (`zero_king_accumulator`) matches this idea on its
declared target slice, because then the king-anchor structure --
the load-bearing motivation for ILA -- is not load-bearing in the
mixer adaptation, and the operator degenerates into a global-pool
+ broadcast + MLP block.

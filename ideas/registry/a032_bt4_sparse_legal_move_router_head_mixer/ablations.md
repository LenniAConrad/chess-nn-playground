# Ablations

This folder is a controlled architecture study, not a primitive
study. The first-class ablations are *cross-idea* comparisons against
the matched `conv` and `attention` BT4 baselines and against the
source primitive (`p027_sparse_legal_move_router_head`) used as a
pooled additive head rather than a token mixer.

## Ablation switches

| ID | Comparison run | What it changes |
|---|---|---|
| A1 | `bt4_conv_mixer` (sibling idea / baseline) | Replaces the `sparse_legal_move_router_head` mixer with the original `conv` mixer, holding every other tower hyperparameter, optimizer setting, and data split fixed. Direct control for "is the sparse legal-move router better than a 3x3 conv pair?". |
| A2 | `bt4_attention_mixer` (sibling idea / baseline) | Replaces the `sparse_legal_move_router_head` mixer with a generic dense multi-head self-attention over the 64 squares. Direct control for "is the chess-structured sparse mask better than dense all-pairs attention at matched widths?". |
| A3 | `p027_sparse_legal_move_router_head` (source primitive idea) | Uses the primitive as a pooled additive head over the i193 trunk with the *rule-exact* legal-move adjacency, instead of as the per-block spatial mixer with a fixed-support + learned-gate adjacency. Tests whether the primitive transfers any of its signal through the BT4 tower at all, and isolates the cost of replacing the rule-exact piece-plane adjacency with a learned soft gate over a static chess-geometry support. |
| A4 | Capacity-matched `bt4_conv_mixer` | If the `sparse_legal_move_router_head` mixer adds parameters versus the conv mixer, A1 is repeated with conv `channels`/`num_blocks` increased to match the parameter count. Distinguishes "this mixer carries new signal" from "this mixer just adds capacity". |
| A5 | `full_64x64_mask` (in-mixer ablation) | Drop the chess-geometry support `S` and let masked attention run over the full `(64, 64)` graph (or equivalently set `S := 1`). Tests whether the legal-move sparsity prior is load-bearing or whether dense attention with a learned per-edge gate matches it. |
| A6 | `shuffle_adjacency` (in-mixer ablation) | Random permutation of the rows and columns of the static support `S` (with the per-edge gate `theta` permuted in lockstep). Decouples the chess-move pattern from the actual board squares; tests whether the *content* of the sparse pattern is load-bearing or only its *sparsity level*. |
| A7 | `self_loop_only` (in-mixer ablation) | Force `S := I` so each square attends only to itself. The mixer degenerates into a per-square 1x1-channel-mixing block; isolates whether any spatial routing at all is responsible for the model's predictions. |

## What each ablation tests

- **A1 (vs `conv`)**: the primary architecture-level falsifier. If
  the `sparse_legal_move_router_head` mixer does not beat `conv` on
  at least one CRTK slice without regressing aggregate PR AUC, the
  mixer carries no architecture-level signal in this tower.
- **A2 (vs `attention`)**: the protective control. If `attention`
  matches or beats the sparse router, the win in A1 is generic
  all-pairs long-range mixing, not the specific chess-structured
  sparse-routing prior. This is the most informative control for
  this idea, since sparse and dense attention share the same
  Q/K/V backbone.
- **A3 (vs primitive as head with rule-exact adjacency)**: tests
  transferability and isolates the cost of replacing the rule-exact
  legal-move adjacency with a fixed support plus a learned per-edge
  gate. The source primitive was designed as a pooled additive
  head on the i193 trunk reading the piece planes directly to
  construct a per-batch `(B, 64, 64)` legal-move adjacency; A3
  tells us whether the same signal survives being repurposed as a
  token mixer with a static-support-plus-learned-gate proxy.
- **A4 (capacity match)**: distinguishes signal from FLOPs.
- **A5 (`full_64x64_mask`)**: localises the load-bearing component.
  If the chess-geometry support and the full mask perform
  identically, the legal-move sparsity prior is not what makes
  SLMR work; the win, if any, is in the masked-softmax aggregator
  alone (which is just dense attention).
- **A6 (`shuffle_adjacency`)**: protects against "any sparse mask
  with the same density works". If shuffled supports match the
  unablated mixer, the *content* of the chess-move pattern is
  decorative -- only the sparsity level matters.
- **A7 (`self_loop_only`)**: the trivial sanity check. If a
  self-loop-only routing matches the unablated mixer, the spatial-
  mixing role of the mixer is entirely subsumed by the surrounding
  stem conv and SqueezeExcite.

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
chess-structured sparse-routing prior. Drop also if A5
(`full_64x64_mask`) matches this idea on its declared target slice,
because then the legal-move sparsity is not load-bearing and the
mixer degenerates into a standard dense attention with a learned
per-edge bias.

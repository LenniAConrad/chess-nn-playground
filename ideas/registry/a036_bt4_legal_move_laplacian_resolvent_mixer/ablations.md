# Ablations

This folder is a controlled architecture study, not a primitive
study. The first-class ablations are *cross-idea* comparisons against
the matched `conv` and `attention` BT4 baselines and against the
source primitive (`p031_legal_move_laplacian_resolvent`) used as an
i193-additive head rather than a token mixer.

## Ablation switches

| ID | Comparison run | What it changes |
|---|---|---|
| A1 | `bt4_conv_mixer` (sibling idea / baseline) | Replaces the `legal_move_laplacian_resolvent` mixer with the original `conv` mixer, holding every other tower hyperparameter, optimizer setting, and data split fixed. Direct control for "is the truncated Neumann-series resolvent over the chess legal-move Laplacian better than a 3x3 conv pair?". |
| A2 | `bt4_attention_mixer` (sibling idea / baseline) | Replaces the `legal_move_laplacian_resolvent` mixer with a generic dense multi-head self-attention over the 64 squares. Direct control for "is the chess-rule legal-move Laplacian with multi-hop closure better than dense all-pairs attention at matched widths?". |
| A3 | `p031_legal_move_laplacian_resolvent` (source primitive idea) | Uses the primitive as an i193-additive head with the original occupancy-blocked piece-typed adjacency, the per-piece-type edge weights, the pooled feature-vector head, and the gate/delta MLPs over the i193 trunk diagnostics, instead of as the per-block spatial mixer with `Y` returned directly. Tests whether the primitive transfers any of its signal through the BT4 tower at all, and isolates the cost of (a) replacing the occupancy-blocked piece-typed adjacency with the static chess-rule reach geometry, (b) replacing the per-piece-type weighting with a per-square learned scalar, and (c) replacing the pooled feature-vector read-out with a per-square channel read-out. |
| A4 | Capacity-matched `bt4_conv_mixer` | If the `legal_move_laplacian_resolvent` mixer adds parameters versus the conv mixer (per-square content-weight MLP `Linear(C, C//2 + 1) + Linear(C//2 + 1, 1)`, `alpha_logit` scalar, `Theta = Linear(C, C, bias=False)`, LayerNorm), A1 is repeated with conv `channels`/`num_blocks` increased to match the parameter count. Distinguishes "this mixer carries new signal" from "this mixer just adds capacity". |
| A5 | `k1_gat_rebrand` (in-mixer ablation) | Force `neumann_terms = 1`. The operator becomes `(I + alpha L) X`, a single-hop legal-mask GAT weighted by the per-square scalar `w(x)`. **Primary in-mixer falsifier** -- matches the source primitive's primary falsifier. If A5 matches the unablated mixer, the multi-hop Neumann expansion is decorative and the operator is a single-hop legal-mask GAT. |
| A6 | `zero_alpha` (in-mixer ablation) | Force `alpha = 0`. The operator reduces to `Theta @ X`, a per-square linear projection. Tests whether the resolvent expansion is load-bearing at all. |
| A7 | `uniform_piece_weights` (in-mixer ablation) | Disable the per-square content-weight MLP (`w(x) := 1` everywhere). Tests whether the input-conditioned per-square weighting is load-bearing or whether a constant edge weight suffices. |
| A8 | `shuffle_adjacency` (in-mixer ablation) | Permute the rows/cols of `A_static` randomly per batch. Decouples the chess-rule legal-move geometry from the chess-board square indexing. If A8 matches the unablated mixer, the chess-rule reach prior is decorative -- any random graph with comparable density would work. |
| A9 | `zero_resolvent_features` (in-mixer ablation) | Force the mixer output to zero (`return torch.zeros_like(x)`); the BT4 block degenerates to a SqueezeExcite + residual block. Tests whether the routed-token output is load-bearing or whether the surrounding SqueezeExcite + residual stream is doing all the work in this tower. |

## What each ablation tests

- **A1 (vs `conv`)**: the primary architecture-level falsifier. If
  the `legal_move_laplacian_resolvent` mixer does not beat `conv` on
  at least one CRTK slice without regressing aggregate PR AUC, the
  mixer carries no architecture-level signal in this tower.
- **A2 (vs `attention`)**: the protective control. If `attention`
  matches or beats the LM-LPP mixer, the win in A1 is generic
  all-pairs long-range mixing, not the specific chess-rule legal-
  move-Laplacian + multi-hop Neumann closure prior. Dense attention
  also moves information globally but does so without the chess-rule
  reach prior; the LM-LPP mixer bakes the chess-rule legal-move
  geometry and its multi-hop closure into the operator.
- **A3 (vs primitive as head with original occupancy-blocked
  adjacency)**: tests transferability and isolates the cost of
  replacing the occupancy-blocked piece-typed adjacency with the
  static chess-rule reach geometry, the per-piece weighting with a
  per-square scalar, and the pooled feature-vector read-out with a
  per-square channel read-out. The source primitive was designed as
  an i193-additive head; A3 tells us whether the same signal
  survives being repurposed as a token mixer at all, and which of
  the three honest compromises (no occupancy blocking, no per-piece
  weighting, no pooled read-out) is the load-bearing cost.
- **A4 (capacity match)**: distinguishes signal from FLOPs. The
  mixer adds the per-square content-weight MLP, the `alpha_logit`
  scalar, the `Theta = Linear(C, C, bias=False)` mixing matrix, and
  the LayerNorm versus the conv mixer's two 3x3 convs. The conv
  baseline must be sized to match the parameter count before
  declaring an A1 win.
- **A5 (`k1_gat_rebrand`)**: the primary in-mixer falsifier (matches
  the source primitive's primary falsifier). If A5 matches the
  unablated mixer, the multi-hop Neumann expansion is decorative
  inside the BT4 tower's residual stack -- the operator is a single-
  hop legal-mask GAT weighted by the per-square scalar, and the
  whole multi-hop tactical-influence claim collapses.
- **A6 (`zero_alpha`)**: the second in-mixer falsifier. If A6
  matches the unablated mixer, the resolvent expansion is decorative
  and the operator reduces to a per-square linear projection
  `Theta @ X` -- equivalent to a 1x1 conv on the per-square feature
  vector.
- **A7 (`uniform_piece_weights`)**: tests whether the input-
  conditioned per-square weighting is load-bearing. If A7 matches
  the unablated mixer, the per-square content-weight MLP is
  decorative -- the static chess-rule reach geometry with constant
  edge weights would suffice and the operator reduces to a fixed,
  position-independent Laplacian resolvent.
- **A8 (`shuffle_adjacency`)**: tests whether the chess-rule legal-
  move geometry is load-bearing. If A8 matches the unablated mixer,
  the chess-rule reach prior is decorative -- any random graph with
  comparable density would work, and the mixer's wins come from the
  Neumann-series closure of an arbitrary graph rather than the
  chess-specific reach prior.
- **A9 (`zero_resolvent_features`)**: localises the load-bearing
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
  lower per-block FLOPs -- LM-LPP is roughly `4x` attention's
  FLOPs at `K = 4`, so this requires a cheaper LM-LPP variant), AND
- A4: the slice-level lift survives the capacity-matched conv
  comparison.

Drop if any one fails. Drop especially if A4 closes -- that means
the mixer is buying its win with parameter count, not with the
chess-rule legal-move Laplacian + multi-hop Neumann closure prior.
Drop also if A5 (`k1_gat_rebrand`) matches this idea on its declared
target slice, because then the multi-hop Neumann expansion -- the
load-bearing motivation for LM-LPP -- is not load-bearing in the
mixer adaptation, and the operator degenerates into a single-hop
legal-mask GAT. Drop also if A6 (`zero_alpha`) matches: the
resolvent expansion is then decorative and the operator reduces to
a per-square linear projection. Drop also if A8 (`shuffle_adjacency`)
matches: the chess-rule legal-move geometry is then decorative and
the wins come from the Neumann-series closure of any graph.

# Ablations

This folder is a controlled architecture study, not a primitive
study. The first-class ablations are *cross-idea* comparisons against
the matched `conv` and `attention` BT4 baselines and against the
source primitive (`p035_sparse_legal_graph_transition`) used as a
pooled additive head rather than a token mixer.

## Ablation switches

| ID | Comparison run | What it changes |
|---|---|---|
| A1 | `bt4_conv_mixer` (sibling idea / baseline) | Replaces the `sparse_legal_graph_transition` mixer with the original `conv` mixer, holding every other tower hyperparameter, optimizer setting, and data split fixed. Direct control for "is the joint non-separable edge function on the chess-rule graph better than a 3x3 conv pair?". |
| A2 | `bt4_attention_mixer` (sibling idea / baseline) | Replaces the `sparse_legal_graph_transition` mixer with a generic dense multi-head self-attention over the 64 squares. Direct control for "is the joint edge function with hard-binary chess-rule mask + mean aggregation better than dense softmax all-pairs attention at matched widths?". |
| A3 | `p035_sparse_legal_graph_transition` (source primitive idea) | Uses the primitive as a pooled additive head over the i193 trunk with the *blocker-resolved* per-board legal-move adjacency `A(x)` (built from the `simple_18` piece planes) and a pooled scalar delta-logit MLP fused with the i193 base logit, instead of as the per-block spatial mixer with the static union-of-moves adjacency and per-square channel read-out. Tests whether the primitive transfers any of its signal through the BT4 tower at all, and isolates the cost of replacing the blocker-resolved adjacency with the static union and the pooled scalar read-out with a per-square channel read-out. |
| A4 | Capacity-matched `bt4_conv_mixer` | If the `sparse_legal_graph_transition` mixer adds parameters versus the conv mixer (three `Linear(C -> d_edge)` projections for `W_self`, `W_neighbor`, `W_interact`, plus the `LayerNorm(C)` input norm, the `LayerNorm(d_edge)` per-edge norm, and the `Linear(d_edge -> C)` back-projection), A1 is repeated with conv `channels`/`num_blocks` increased to match the parameter count. Distinguishes "this mixer carries new signal" from "this mixer just adds capacity". |
| A5 | `separable_phi` (in-mixer ablation) | Zero out `W_interact` so `phi(X_i, X_j) = LayerNorm(ReLU(W_self X_i + W_neighbor X_j))` -- a separable additive form. **Primary in-mixer falsifier** -- mirrors the source primitive's `separable_phi` falsifier. Tests whether the Hadamard interaction term `W_interact (X_i (.) X_j)` is load-bearing or whether the operator collapses to a standard separable GAT-style aggregator. |
| A6 | `uniform_adjacency` (in-mixer ablation) | Replace the static union-of-moves adjacency with the all-ones matrix (minus identity); the operator becomes a dense joint-edge aggregator over all 63 other squares per source. **Primary in-mixer falsifier** -- mirrors the source primitive's `uniform_adjacency` falsifier. Tests whether the chess-rule mask is load-bearing or whether the operator works just as well with a dense unmasked neighbourhood. |
| A7 | `shuffle_adjacency` (in-mixer ablation) | Apply a batch-permutation to the adjacency mask (so the rule indicators are decoupled from position; each sample sees a randomly-permuted version of the static graph). Mirrors the source primitive's `shuffle_adjacency` falsifier. Tests whether the chess-rule prior is content-aligned with the per-square features or whether any random binary mask of the same density would do. |
| A8 | `zero_slmgt_features` (in-mixer ablation) | Force the mixer output to zero (`return torch.zeros_like(x)`); the BT4 block degenerates to a SqueezeExcite + residual block. Tests whether the routed-token output is load-bearing or whether the surrounding SqueezeExcite + residual stream is doing all the work in this tower. |

## What each ablation tests

- **A1 (vs `conv`)**: the primary architecture-level falsifier. If
  the `sparse_legal_graph_transition` mixer does not beat `conv` on
  at least one CRTK slice without regressing aggregate PR AUC, the
  mixer carries no architecture-level signal in this tower.
- **A2 (vs `attention`)**: the protective control. If `attention`
  matches or beats the SLMGT mixer, the win in A1 is generic
  all-pairs long-range mixing, not the specific joint edge function
  on the chess-rule graph. Dense softmax attention also moves
  information across all 64 squares but does so without the
  hard-binary chess-rule mask and without the joint non-separable
  edge term; the SLMGT mixer bakes both into the operator.
- **A3 (vs primitive as head with original blocker-resolved
  adjacency)**: tests transferability and isolates the cost of (i)
  replacing the blocker-resolved per-board legal-move adjacency
  `A(x)` with the static union-of-moves adjacency `A` and (ii)
  replacing the pooled scalar read-out with a per-square channel
  read-out. The source primitive was designed as a pooled additive
  head with the per-board legal-move graph; A3 tells us whether
  the same signal survives being repurposed as a token mixer with
  both changes simultaneously.
- **A4 (capacity match)**: distinguishes signal from FLOPs. The
  mixer adds three `Linear(C -> d_edge)` projections for `W_self`,
  `W_neighbor`, `W_interact`, plus the input `LayerNorm(C)`, the
  per-edge `LayerNorm(d_edge)`, and the `Linear(d_edge -> C)`
  back-projection, versus the conv mixer's two 3x3 convs. The conv
  baseline must be sized to match the parameter count before
  declaring an A1 win.
- **A5 (`separable_phi`)**: the primary in-mixer falsifier
  (matches the source primitive's primary falsifier). If A5
  matches the unablated mixer, the Hadamard interaction term is
  decorative and the operator is a standard separable GAT-style
  aggregator `Y[i] = (1 / deg(i)) sum_j A[i, j] ReLU(W_self X_i +
  W_neighbor X_j)` with `LayerNorm` per edge.
- **A6 (`uniform_adjacency`)**: the second primary in-mixer
  falsifier. If A6 matches the unablated mixer, the chess-rule
  mask is decorative and the operator works just as well with a
  dense unmasked neighbourhood (all 63 other squares per source).
  The chess-rule prior is then not load-bearing inside the BT4
  tower.
- **A7 (`shuffle_adjacency`)**: tests whether the chess-rule mask
  is content-aligned with the per-square features. If A7 matches
  the unablated mixer, any random binary mask of the same density
  would do; the *specific* chess-rule pattern is not aligned with
  what the per-square features encode.
- **A8 (`zero_slmgt_features`)**: localises the load-bearing
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
joint edge function on the chess-rule graph. Drop also if A5
(`separable_phi`) matches this idea on its declared target slice,
because then the Hadamard interaction term is decorative and the
operator degenerates into a standard separable GAT-style
aggregator. Drop if A6 (`uniform_adjacency`) matches: the chess-
rule mask is then decorative and the operator works just as well
with a dense unmasked neighbourhood. Drop if A7
(`shuffle_adjacency`) matches: any random binary mask of the same
density suffices and the chess-rule alignment claim is decorative.

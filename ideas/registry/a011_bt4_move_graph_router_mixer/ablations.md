# Ablations

This folder is a controlled architecture study, not a primitive study.
The first-class ablations are *cross-idea* comparisons against the
matched `conv` and `attention` BT4 baselines and against the source
primitive (`p006_move_graph_router`) used as an additive head rather
than a token mixer.

## Ablation switches

| ID | Comparison run | What it changes |
|---|---|---|
| A1 | `bt4_conv_mixer` (sibling idea / baseline) | Replaces the MGR mixer with the original `conv` mixer, holding every other tower hyperparameter, optimizer setting, and data split fixed. Direct control for "is the MGR mixer better than a conv?". |
| A2 | `bt4_attention_mixer` (sibling idea / baseline) | Replaces the MGR mixer with a generic multi-head self-attention over the 64 squares. Direct control for "is the MGR mixer better than a vanilla token mixer?". |
| A3 | `p006_move_graph_router` (source primitive idea) | Uses MGR as an additive head over the i193 trunk instead of as the per-block spatial mixer. Tests whether the primitive transfers any of its signal through the BT4 tower at all. |
| A4 | Capacity-matched `bt4_conv_mixer` | If the MGR mixer adds parameters versus the conv mixer, A1 is repeated with conv `channels`/`num_blocks` increased to match the parameter count. Distinguishes "MGR mixer carries new signal" from "MGR mixer just adds capacity". |
| A5 | `edge_density` sweep (`{0.05, 0.10, 0.25, 0.50, 1.00}`) | Tests whether the sparse stop-grad adjacency is load-bearing. `edge_density = 1.0` collapses MGR to a dense mixer with masked-attention shape; `0.05` starves it of edges. The validated `p006` value `0.25` should be the floor of acceptable performance. |
| A6 | Random-mask MGR (replace `_build_edge_mask` with a per-source uniform Bernoulli mask at the same density) | Tests whether the *content-derived* adjacency carries signal beyond mere sparsity / degree normalisation. If the random-mask variant matches the learned-mask variant, the mask scoring head is not load-bearing. |

## What each ablation tests

- **A1 (vs `conv`)**: the primary architecture-level falsifier. If the
  MGR mixer does not beat `conv` on at least one CRTK slice without
  regressing aggregate PR AUC, the mixer carries no architecture-level
  signal in this tower.
- **A2 (vs `attention`)**: protects against the trivial conclusion
  "any token mixer beats conv on 64 squares". If `attention` matches
  the MGR mixer, the win in A1 is generic attention-style mixing, not
  the gather-scatter-over-sparse-stop-grad-mask structure of MGR.
- **A3 (vs primitive as head)**: tests transferability. The source
  primitive was validated as an additive head on the i193 trunk; A3
  tells us whether the same signal survives being repurposed as a
  token mixer.
- **A4 (capacity match)**: distinguishes signal from FLOPs / parameter
  count.
- **A5 (edge-density sweep)**: separates "sparse stop-grad mask is
  load-bearing" from "any density works".
- **A6 (random-mask)**: separates "learned content-derived mask is
  load-bearing" from "any sparse mask works".

## Falsification criteria

Promote (keep) this idea only if all hold on the held-out test split:

- A1: MGR mixer beats `conv` on at least one CRTK slice
  (`crtk_eval_bucket`, `crtk_difficulty`, `crtk_phase`, or
  `crtk_tactic_motifs`) by at least the matched-baseline tolerance
  documented in `ideas/docs/BENCHMARK_REPORTING.md`, AND
- aggregate test PR AUC does not regress vs `conv` by more than 0.005,
  AND
- A2: MGR mixer is not strictly dominated by `attention` on the
  target slice, AND
- A4: the slice-level lift survives the capacity-matched conv
  comparison, AND
- A6: the random-mask MGR variant loses at least 50% of the target-
  slice lift; otherwise the mask scoring head is not load-bearing
  and the mixer is just sparse-dense-mixing-by-shape.

Drop if any one fails. Drop especially if A4 or A6 close -- A4
closing means the MGR mixer is buying its win with parameter count,
A6 closing means it is buying its win with arbitrary sparsity.

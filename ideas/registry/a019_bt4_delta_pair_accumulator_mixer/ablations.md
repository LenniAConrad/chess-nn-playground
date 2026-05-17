# Ablations

This folder is a controlled architecture study, not a primitive study.
The first-class ablations are *cross-idea* comparisons against the
matched `conv` and `attention` BT4 baselines and against the source
primitive (`p014_delta_pair_accumulator`) used as an additive head
rather than a token mixer.

## Ablation switches

| ID | Comparison run | What it changes |
|---|---|---|
| A1 | `bt4_conv_mixer` (sibling idea / baseline) | Replaces the DPA mixer with the original `conv` mixer, holding every other tower hyperparameter, optimizer setting, and data split fixed. Direct control for "is the DPA mixer better than a conv?". |
| A2 | `bt4_attention_mixer` (sibling idea / baseline) | Replaces the DPA mixer with a generic multi-head self-attention over the 64 squares. Direct control for "is the alignment-restricted pair message better than vanilla content-attention pair messages?". |
| A3 | `p014_delta_pair_accumulator` (source primitive idea) | Uses DPA as an additive head over the i193 trunk instead of as the per-block spatial mixer. Tests whether the primitive transfers any of its signal through the BT4 tower at all. |
| A4 | Capacity-matched `bt4_conv_mixer` | The DPA mixer adds parameters versus the conv mixer (`pair_src`/`pair_dst`/`delta_square_gate`/`out_proj` on `pair_dim`). A1 is repeated with conv `channels`/`num_blocks` increased to match the parameter count. Distinguishes "DPA mixer carries new signal" from "DPA mixer just adds capacity". |
| A5 | DPA-mixer with the alignment mask replaced by the all-pairs mask | Tests whether the rule-derived edge set `E(S)` is the load-bearing inductive bias or whether any low-rank bilinear pair message would do. Implemented by toggling `edge_mask` to `ones - eye` inside the mixer at debug time; not a separate registered idea. |

## What each ablation tests

- **A1 (vs `conv`)**: the primary architecture-level falsifier. If the
  DPA mixer does not beat `conv` on at least one CRTK slice without
  regressing aggregate PR AUC, the mixer carries no architecture-level
  signal in this tower.
- **A2 (vs `attention`)**: protects against the trivial conclusion
  "any pair-message token mixer beats conv on 64 squares". If
  `attention` matches the DPA mixer, the win in A1 is generic
  content-attention-style pair mixing, not the alignment-restricted
  rule-derived edge set.
- **A3 (vs primitive as head)**: tests transferability. The source
  primitive was validated as an additive head on the i193 trunk; A3
  tells us whether the same signal survives being repurposed as a
  token mixer.
- **A4 (capacity match)**: distinguishes signal from FLOPs.
- **A5 (all-pairs mask)**: distinguishes the chess-aware geometry from
  generic low-rank pair mixing.

## Falsification criteria

Promote (keep) this idea only if all hold on the held-out test split:

- A1: DPA mixer beats `conv` on at least one CRTK slice
  (`crtk_eval_bucket`, `crtk_difficulty`, `crtk_phase`, or
  `crtk_tactic_motifs`) by at least the matched-baseline tolerance
  documented in `ideas/docs/BENCHMARK_REPORTING.md`, AND
- aggregate test PR AUC does not regress vs `conv` by more than 0.005,
  AND
- A2: DPA mixer is not strictly dominated by `attention` on the
  target slice, AND
- A4: the slice-level lift survives the capacity-matched conv
  comparison, AND
- A5: replacing the alignment mask with the all-pairs mask closes the
  target-slice gap by at least 50%, i.e. the rule-derived edge set is
  load-bearing rather than the bilinear pair message alone.

Drop if any one fails. Drop especially if A4 closes — that means the
DPA mixer is buying its win with parameter count, not with the
alignment-restricted pair message. Drop if A5 fails — the chess-aware
edge geometry was not actually load-bearing and a generic low-rank
bilinear would have sufficed.

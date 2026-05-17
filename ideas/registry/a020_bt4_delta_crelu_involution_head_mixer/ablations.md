# Ablations

This folder is a controlled architecture study, not a primitive study.
The first-class ablations are *cross-idea* comparisons against the
matched `conv` and `attention` BT4 baselines and against the source
primitive (`p015_delta_crelu_involution_head`) used as an additive head
rather than a token mixer.

## Ablation switches

| ID | Comparison run | What it changes |
|---|---|---|
| A1 | `bt4_conv_mixer` (sibling idea / baseline) | Replaces the DCIH mixer with the original `conv` mixer, holding every other tower hyperparameter, optimizer setting, and data split fixed. Direct control for "is the DCIH mixer better than a conv?". |
| A2 | `bt4_attention_mixer` (sibling idea / baseline) | Replaces the DCIH mixer with a generic multi-head self-attention over the 64 squares. Direct control for "is the DCIH mixer better than a vanilla token mixer?". |
| A3 | `p015_delta_crelu_involution_head` (source primitive idea) | Uses DCIH as an additive head over the i193 trunk instead of as the per-block spatial mixer. Tests whether the primitive transfers any of its signal through the BT4 tower at all. |
| A4 | Capacity-matched `bt4_conv_mixer` | If the DCIH mixer adds parameters versus the conv mixer, A1 is repeated with conv `channels`/`num_blocks` increased to match the parameter count. Distinguishes "DCIH mixer carries new signal" from "DCIH mixer just adds capacity". |
| A5 | `involution_weight=0` ablation of this idea | Zeroes the `(sym, asym)` Reynolds split contribution, leaving only the broadcast ClippedReLU accumulator + per-square clipped state. Tests whether the involution / Reynolds half is load-bearing. |
| A6 | `clip_max=+inf` ablation of this idea | Replaces ClippedReLU with the identity (no saturation tape). Tests whether the saturation-aware part of DCIH is load-bearing. |

## What each ablation tests

- **A1 (vs `conv`)**: the primary architecture-level falsifier. If the
  DCIH mixer does not beat `conv` on at least one CRTK slice without
  regressing aggregate PR AUC, the mixer carries no architecture-level
  signal in this tower.
- **A2 (vs `attention`)**: protects against the trivial conclusion
  "any token mixer beats conv on 64 squares". If `attention` matches
  the DCIH mixer, the win in A1 is generic attention-style mixing, not
  chess-aware structure.
- **A3 (vs primitive as head)**: tests transferability. The source
  primitive was validated as an additive head on the i193 trunk; A3
  tells us whether the same signal survives being repurposed as a
  token mixer.
- **A4 (capacity match)**: distinguishes signal from FLOPs.
- **A5 (involution off)**: isolates the Reynolds equivariance
  contribution from the saturation accumulator.
- **A6 (clipping off)**: isolates the saturation-aware ClippedReLU
  contribution from the involution split.

## Falsification criteria

Promote (keep) this idea only if all hold on the held-out test split:

- A1: DCIH mixer beats `conv` on at least one CRTK slice
  (`crtk_eval_bucket`, `crtk_difficulty`, `crtk_phase`, or
  `crtk_tactic_motifs`) by at least the matched-baseline tolerance
  documented in `ideas/docs/BENCHMARK_REPORTING.md`, AND
- aggregate test PR AUC does not regress vs `conv` by more than 0.005,
  AND
- A2: DCIH mixer is not strictly dominated by `attention` on the
  target slice, AND
- A4: the slice-level lift survives the capacity-matched conv
  comparison, AND
- at least one of A5 or A6 erases the target-slice lift (otherwise
  both load-bearing claims for DCIH have failed and the result is not
  attributable to DCIH structure).

Drop if any one of A1/A2/A4 fails. Drop especially if A4 closes — that
means the DCIH mixer is buying its win with parameter count, not with
chess-aware structure. Drop also if A5 *and* A6 both leave the
target-slice lift intact — the win then has no DCIH-specific
attribution.

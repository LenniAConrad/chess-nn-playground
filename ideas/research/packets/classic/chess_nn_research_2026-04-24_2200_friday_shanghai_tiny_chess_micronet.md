# Codex Research Packet: Tiny Chess MicroNet

## File Metadata

- Filename: `chess_nn_research_2026-04-24_2200_friday_shanghai_tiny_chess_micronet.md`
- Generated at: 2026-04-24 22:00
- Weekday: Friday
- Timezone: Asia/Shanghai
- Intended next consumer: Codex
- Status: full architecture packet, not implemented

## One-Sentence Thesis

Build an intentionally tiny chess puzzle-likeness network, with strict `2k`, `8k`, `25k`, and `50k` parameter tiers, by combining low-rank channel mixing, depthwise board filters, fixed chess line sketches, king-zone pooling, and an INT8-ready descriptor head.

## Why This Packet Exists

Most earlier packets in this archive ask whether a richer mathematical object helps:

- square-pair fields
- Schur line solves
- relation queries
- variational residuals
- transport bottlenecks
- sheaf or graph tension
- attention-style token selection

This packet asks the opposite question:

```text
How much chess signal survives when the model is brutally small?
```

The tiny setting is not only engineering. It is a research constraint. If a model with fewer than `25k` parameters can separate puzzle-like positions from non-puzzles better than material/count controls, then some of the useful signal is low-dimensional and highly compressible.

The architecture should be treated as a new research object, not as a width/depth sweep of an existing CNN.

## Data Contract

Task:

- output `0`: non-puzzle
- output `1`: puzzle-like

Fine labels:

- train on binary labels only
- retain fine labels `0`, `1`, and `2` for diagnostics
- report the fine-label `3 x 2` diagnostic matrix

First implementation:

- input: `simple_18`
- dataset: existing `crtk_sample_3class` splits
- trainer: shared experimental training pipeline

Forbidden model inputs:

- Stockfish scores, PVs, mate scores, node counts, verification metadata, source labels, proposed labels, unresolved candidate status, dataset provenance, or anything derived from them.
- Engine search, forced-line search, legal mate/stalemate oracles, tablebase outcomes, or future game outcomes.

Allowed model inputs:

- Current-board tensor.
- Side-to-move, castling, and en-passant planes already present in `simple_18`.
- Deterministic square coordinates.
- Deterministic line, diagonal, edge, center, and king-zone masks derived from the current board.

## Core Abstraction

Use a small learned board field:

```text
H in R^{B x W x 8 x 8}
```

but forbid the expensive part that often makes small CNNs secretly large:

```text
flatten(H) -> dense MLP
```

Instead, compress `H` through a fixed bank of chess-shaped summaries:

```text
global pools
rank/file/diagonal line sketches
side-relative line sketches
king-zone pools
material/count summaries
```

The model becomes:

```text
simple_18 board
  -> low-rank 1x1 squeeze
  -> repeated tiny depthwise-line blocks
  -> fixed chess sketch bank
  -> tiny descriptor MLP
  -> binary logits
```

The main claim:

```text
A tiny model should spend almost all of its parameters on deciding which chess summaries matter, not on rediscovering board geometry from scratch.
```

## Name

Working name:

```text
TinyChessMicroNet
```

Short tags:

- `TCM-Net`
- `tiny_chess_micronet`
- `micro_line_sketch_net`

## Non-Goals

This packet is not:

- a generic mobile CNN
- a new optimizer schedule
- a pruning recipe for a larger network
- a post-training compression-only experiment
- a small version of attention
- a small version of a residual tower
- a bitboard engine feature extractor

It should be implemented as its own architecture.

## Novelty Relative To Existing Packets

`Multi-Scale Dilated Board Mixer CNN` is practical but still a normal CNN benchmark.

`Piece-Token CNN Hybrid` fuses token and board branches and is a strong baseline, but it is not tiny.

`Schur-Ray Line Algebra Network` uses a high-level line-solve object.

`Bitboard Shift-Algebra Network` uses learned operator polynomials over fixed shift operators.

This packet is different because:

- the parameter budget is the central constraint
- the readout is sketch-based rather than flatten-based
- the architecture is designed for INT8 quantization from the start
- the ablations must prove that chess sketches matter under tiny capacity
- the useful output is not only AUROC but also size/latency/calibration curves

## Architecture Overview

### Step 1: Low-Rank Channel Squeeze

Input:

```text
x: (B, 18, 8, 8)
```

Apply a low-rank `1 x 1` projection:

```text
u = Conv1x1(18 -> R)(x)
h0 = Conv1x1(R -> W)(relu6(u))
```

Recommended:

```text
R = max(4, W / 3)
W in {8, 12, 16, 24}
```

This keeps the input adapter tiny:

```text
params_squeeze = 18R + RW + biases
```

For `W = 16`, `R = 6`, this is only about `204` weights before biases.

### Step 2: Tiny Depthwise-Line Blocks

Each block is:

```text
h = h + alpha * MicroLineBlock(h)
```

where `alpha` is a learned scalar initialized near `0.1`.

The block:

```text
depthwise 3x3 conv
relu6
fixed line-sum residual
low-rank channel mix
relu6
```

Pseudocode:

```python
def block(h):
    local = depthwise_3x3(h)
    line = fixed_line_smooth(h)
    y = local + line
    y = relu6(y)
    y = pointwise_low_rank(y)
    return y
```

The line-sum residual is parameter-light. It uses fixed masks over:

- ranks
- files
- diagonals
- anti-diagonals

For each direction, compute a short average and scatter it back:

```text
line_mean_d(c, line_id) = mean_{s in line_id} h[c, s]
line_back_d(c, s) = line_mean_d(c, line_of_s)
```

Then:

```text
fixed_line_smooth(h) =
  gamma_rank * rank_back(h)
  + gamma_file * file_back(h)
  + gamma_diag * diag_back(h)
  + gamma_anti * anti_back(h)
```

where each `gamma_*` is a learned per-channel scalar:

```text
gamma_* in R^W
```

This costs:

```text
4W parameters
```

and gives the tiny model cheap long-range chess geometry.

### Step 3: Low-Rank Pointwise Mixer

Avoid full `W x W` channel mixing in the tiny tiers.

Use:

```text
pointwise_low_rank(h) = B relu6(A h)
```

with:

```text
A: W -> Rmix
B: Rmix -> W
```

Recommended:

```text
Rmix = 4 for W <= 12
Rmix = 6 for W = 16
Rmix = 8 for W = 24
```

Parameter count per block:

```text
depthwise_3x3: 9W
line gammas: 4W
low_rank_mix: 2 W Rmix
residual scale and biases: O(W)
```

For `W = 16`, `Rmix = 6`:

```text
9W + 4W + 2WRmix = 144 + 64 + 192 = 400
```

before small biases and normalization parameters.

### Step 4: Fixed Chess Sketch Bank

Do not flatten all `64W` hidden activations into a dense head.

Instead compute descriptors:

```text
z = sketch(H)
```

The sketch bank has five groups.

#### A. Global Pools

Per channel:

```text
mean(H_c)
max(H_c)
mean(abs(H_c - mean(H_c)))
```

Dimension:

```text
3W
```

#### B. Fixed Line Basis Sketches

For each direction:

```text
rank
file
diagonal
anti_diagonal
```

use a small bank of fixed basis masks:

```text
constant
edge_heavy
center_heavy
side_relative_forward
side_relative_backward
occupancy_weighted
```

For each basis `k` and channel `c`:

```text
ell[d, k, c] = sum_s mask[d, k, s, x] * H[c, s]
```

Normalize by a fixed or safe denominator:

```text
ell[d, k, c] = ell[d, k, c] / max(sum_s mask[d, k, s, x], 1)
```

Dimension:

```text
4 directions * 6 bases * W = 24W
```

The only current-board-dependent mask here should be `occupancy_weighted`, derived from current occupancy only. It does not use legal moves, attacks, or engine information.

#### C. King-Zone Pools

Decode own and opponent king squares from the current board input.

If the semantic channel mapping is ambiguous, fail closed by disabling this branch and logging it.

For each king:

```text
3x3 zone mean
5x5 ring mean
nearest edge zone mean
```

For two kings:

```text
6W
```

Rationale:

Tiny models should not waste capacity learning that king neighborhoods are special.

#### D. Material And Sparse State Summary

From `simple_18`, derive safe count features:

```text
piece counts by type and side
side to move
castling rights count
en-passant plane occupancy flag
total occupancy
own/opponent occupancy
```

Recommended dimension:

```text
16 to 24
```

These counts are not enough to solve the task. They are included to prevent the tiny network from spending scarce spatial parameters on trivial material summaries.

#### E. Micro Factor Readout

Add a tiny learned low-rank readout over the descriptor vector:

```text
q = relu6(z U)
logits = q V
```

where:

```text
U: D_z -> Hhead
V: Hhead -> 2
```

Recommended:

```text
Hhead = 8, 16, 32, or 48
```

depending on the target parameter tier.

## Parameter Tiers

### Tier 1: `nano_2k`

Purpose:

- sanity-check whether anything beyond counts exists at extreme compression

Suggested settings:

```yaml
width: 8
squeeze_rank: 4
blocks: 1
mix_rank: 3
head_hidden: 8
king_zone: false
line_bases: [constant, center_heavy, side_relative_forward]
quantization_target: int8
expected_params: 1500-2500
```

Expected role:

- should beat `counts_only_mlp`
- probably will not beat a small CNN
- useful as a lower bound

### Tier 2: `micro_8k`

Purpose:

- first serious tiny architecture

Suggested settings:

```yaml
width: 12
squeeze_rank: 4
blocks: 2
mix_rank: 4
head_hidden: 16
king_zone: true
line_bases: all_6
quantization_target: int8
expected_params: 6000-9000
```

Expected role:

- main smallest model
- should be fast on CPU
- should be small enough to run many seeds

### Tier 3: `micro_25k`

Purpose:

- best practical tiny model

Suggested settings:

```yaml
width: 16
squeeze_rank: 6
blocks: 3
mix_rank: 6
head_hidden: 32
king_zone: true
line_bases: all_6
quantization_target: int8
expected_params: 15000-25000
```

Expected role:

- main model for claims
- compare directly to parameter-matched tiny CNN
- run full ablation table

### Tier 4: `tiny_50k`

Purpose:

- upper tiny tier for stronger accuracy without leaving the tiny regime

Suggested settings:

```yaml
width: 24
squeeze_rank: 8
blocks: 4
mix_rank: 8
head_hidden: 48
king_zone: true
line_bases: all_6
quantization_target: int8
expected_params: 35000-50000
```

Expected role:

- test whether the tiny family scales smoothly
- optional teacher distillation target
- should still be far below typical residual CNNs in this archive

## Quantization Rules

The architecture should be quantization-aware from the beginning.

Allowed operations:

- `Conv1x1`
- depthwise `Conv2d`
- integer-friendly sums
- average pooling with fixed denominators
- `ReLU`
- `ReLU6`
- hard clamp
- small dense layers

Avoid:

- GELU
- LayerNorm
- Softmax
- dynamic top-k
- dynamic token lists
- SVD/eigendecomposition
- attention maps
- per-position Python loops
- ragged graph construction

Recommended training path:

1. Train FP32 first.
2. Train with fake quantization.
3. Export an INT8 simulated model.
4. Report FP32 and INT8 validation metrics.

Minimum quantization success criterion:

```text
INT8 AUROC drop <= 0.005 absolute
```

or, if AUROC is unstable:

```text
INT8 class-1 recall drop <= 5% relative at matched fine-label-0 FPR
```

If INT8 fails badly, the architecture is not truly a tiny deployment architecture.

## Why This Could Work

Chess boards are only `8 x 8`.

Many important motifs are low-dimensional:

- material imbalance
- king exposure
- open files
- diagonal alignment
- centralization
- back-rank geometry
- pawn shield asymmetry
- piece activity around king zones
- line tension between occupied squares

A large model can learn these from raw squares. A tiny model cannot afford to rediscover all of them.

The line sketch bank gives cheap inductive bias:

```text
long-range board geometry at almost zero parameter cost
```

The depthwise blocks give local texture:

```text
neighboring pieces, pawn cover, king shell, local congestion
```

The low-rank channel mixers decide which local and line cues combine.

## Why This Could Fail

The dataset may require:

- move legality
- forcing sequence structure
- subtle piece mobility
- long tactical dependencies not visible in coarse line sketches
- resolution that is lost by global sketch pooling

The tiny model may also overuse:

- material counts
- side-to-move priors
- castling/en-passant artifacts
- shallow piece density correlations

That is why the ablations below are mandatory.

## Core Ablation Plan

| Ablation | Change | Claim Tested | Failure Meaning |
|---|---|---|---|
| `counts_only_mlp` | Use only material/state counts with matched head budget | Spatial tiny model beats trivial summaries | If it matches, spatial network adds little. |
| `ordinary_tiny_cnn_matched` | Replace line sketches with ordinary tiny CNN and global pool, same params | Chess sketch beats generic tiny conv capacity | If it matches, use the simpler CNN. |
| `flat_head_same_params` | Flatten `8 x 8` hidden map, shrink width/head to match params | Sketch readout beats location-memorizing dense readout | If it wins, fixed sketches may be too lossy. |
| `no_line_sketch` | Remove rank/file/diag descriptors | Line geometry matters | If unchanged, line sketch is cosmetic. |
| `random_line_basis` | Use random fixed masks with same density and dimension | Chess-shaped masks matter | If unchanged, any random pooling works. |
| `no_king_zone` | Remove own/opponent king-zone pools | King locality matters in tiny setting | If unchanged, king branch is unnecessary. |
| `no_depthwise_local` | Remove depthwise 3x3 blocks | Local texture matters | If unchanged, descriptors dominate. |
| `untied_line_gammas` | Let each line ID have separate learned weights | Parameter sharing matters | If untied wins big, tiny model needed more location specificity. |
| `fp32_no_qat` | Train FP32 only and post-quantize | Quantization-aware design matters | If equal, QAT may be unnecessary. |
| `teacher_distill_off` | Remove optional teacher distillation | Gains are not teacher-only | If distillation is required, report separately. |

Smallest central falsifier:

```text
ordinary_tiny_cnn_matched >= TinyChessMicroNet
```

at the same parameter budget, same training budget, and same random seeds.

If that happens, this packet should not be promoted.

## Baselines To Compare

Required:

- existing smallest `simple_18` CNN baseline
- `counts_only_mlp`
- `ordinary_tiny_cnn_matched`
- `flat_head_same_params`

Recommended:

- `Piece-Token CNN Hybrid` if already implemented
- `Multi-Scale Dilated Board Mixer CNN` if already implemented
- a logistic regression over material/count summaries

Do not compare only against large models. The point is not to beat the strongest architecture. The point is to learn the accuracy/size frontier.

## Metrics

Report:

- parameter count
- serialized FP32 size
- simulated INT8 size
- multiply-add estimate
- validation AUROC
- test AUROC
- class-1 recall at fixed fine-label-0 FPR
- fine-label `3 x 2` diagnostic matrix
- expected calibration error
- latency on CPU for batch size `1`
- throughput for batch size `1024`
- INT8 metric drop from FP32

Optional:

- activation memory
- peak resident memory during inference
- per-position explanation by descriptor group ablation

## Descriptor Group Diagnostics

The head should log learned group norms:

```text
global_pool_norm
line_sketch_norm
king_zone_norm
material_summary_norm
```

For each validation checkpoint, report:

```text
norm(group) / sum_group_norms
```

This does not prove causality, but it helps detect degenerate solutions:

- all weight on material counts
- all weight on side-to-move
- no use of line sketches
- no use of local conv features

## Implementation Sketch

Suggested files:

```text
src/chess_nn_playground/models/trunk/tiny_chess_micronet.py
tests/test_tiny_chess_micronet.py
configs/tiny_chess_micronet_nano2k.yaml
configs/tiny_chess_micronet_micro8k.yaml
configs/tiny_chess_micronet_micro25k.yaml
configs/tiny_chess_micronet_tiny50k.yaml
```

Suggested module structure:

```python
class LowRankConv1x1(nn.Module):
    ...

class MicroLineBlock(nn.Module):
    ...

class ChessSketchBank(nn.Module):
    ...

class TinyChessMicroNet(nn.Module):
    ...
```

Forward contract:

```python
def forward(self, x):
    h = self.squeeze(x)
    for block in self.blocks:
        h = h + block.scale * block(h)
    z = self.sketch(h, x)
    logits = self.head(z)
    return logits
```

The sketch module should be deterministic and differentiable with respect to `h`, but not necessarily with respect to masks derived from `x`.

## Fixed Mask Implementation

Precompute masks as buffers:

```text
rank_masks: (8, 64)
file_masks: (8, 64)
diag_masks: (15, 64)
anti_diag_masks: (15, 64)
center_basis: (64,)
edge_basis: (64,)
side_relative_basis_white: (64,)
side_relative_basis_black: (64,)
```

Use `register_buffer`.

Do not allocate masks inside `forward`.

For speed:

```python
Hf = h.flatten(2)  # (B, W, 64)
summary = torch.einsum("bwc,mc->bwm", Hf, masks)
```

Then reduce line IDs using small fixed basis weights:

```python
line_basis_summary = torch.einsum("bwm,km->bwk", summary, line_basis)
```

Flatten:

```python
z_line = line_basis_summary.flatten(1)
```

## King-Zone Implementation

If `simple_18` has clear king planes:

1. Find own king square.
2. Find opponent king square.
3. Build `3x3` and `5x5` masks by indexing precomputed masks.
4. Pool `h` over those masks.

The model must handle missing or malformed king planes:

```text
if no exact king square:
    use all-zero king-zone descriptor
    increment diagnostic counter
```

No exception should be raised during normal training because corrupted or nonstandard positions should fail closed.

## Optional Teacher Distillation

Distillation is optional and secondary.

Allowed teacher:

- a model trained only on allowed current-board inputs and labels

Suggested teacher:

- `Piece-Token CNN Hybrid`
- `Multi-Scale Dilated Board Mixer CNN`

Forbidden teacher:

- any engine-assisted labeler
- any model that consumed engine scores or future outcomes

Distillation objective:

```text
loss = BCE(logits, y) + lambda * KL(student_logits / T, teacher_logits / T)
```

Recommended:

```text
lambda = 0.1
T = 2.0
```

But the packet's central evidence must come from non-distilled `micro_25k`.

## Minimal Experiment

Run:

```text
micro_8k
micro_25k
counts_only_mlp
ordinary_tiny_cnn_matched
no_line_sketch
no_king_zone
random_line_basis
```

Use:

```text
3 seeds
same train/val/test split
same trainer budget
same binary labels
same fine-label diagnostics
```

Do not tune learning rate separately per ablation unless all variants receive the same grid.

## Promotion Criteria

Promote `TinyChessMicroNet` if:

- `micro_25k` beats `ordinary_tiny_cnn_matched` by at least `0.005` AUROC or a clear recall/calibration improvement
- `micro_25k` beats `counts_only_mlp` decisively
- `random_line_basis` loses meaningful performance
- `no_line_sketch` loses meaningful performance
- INT8 simulation loses little performance
- results are stable over seeds

Soft promotion is allowed if:

- AUROC is tied but INT8 latency/size is much better
- fine-label-1 diagnostics improve even if binary AUROC is similar
- calibration improves at the same parameter budget

## Rejection Criteria

Reject or archive as a low-priority idea if:

- `counts_only_mlp` nearly matches the model
- `ordinary_tiny_cnn_matched` matches or beats it
- `random_line_basis` matches the real line basis
- INT8 loses too much performance
- the head norm concentrates almost entirely on material/count features
- the best results require teacher distillation
- the model is only good after widening beyond `50k` parameters

## Expected Research Value

Even if it fails, this packet gives useful information:

- whether the task has a small low-dimensional signal
- whether chess line sketches are useful under extreme compression
- whether king-zone features matter when capacity is scarce
- whether INT8 quantization hurts this benchmark
- what a credible tiny baseline should be for future larger architectures

If it succeeds, it becomes a strong baseline because every future complex architecture must answer:

```text
Does it beat a 25k-parameter chess-aware tiny model?
```

## Implementation Priority

Recommended priority:

```text
medium-high
```

Reason:

- implementation is simpler than graph, transport, eigenspace, or square-pair models
- runtime is cheap
- many seeds and ablations are affordable
- it creates a useful lower envelope for the accuracy/size frontier

Suggested order:

1. Implement `counts_only_mlp`.
2. Implement `ChessSketchBank` with fixed masks and unit tests.
3. Implement `TinyChessMicroNet`.
4. Verify parameter counts for all tiers.
5. Train `micro_8k` and `micro_25k`.
6. Run ablations.
7. Add QAT/INT8 simulation only after FP32 behavior is credible.

## Continuity Notes

Do not turn this into:

- a bigger CNN
- an attention model
- a post-hoc pruning study
- a generic MobileNet clone

The important object is:

```text
tiny learned local field + fixed chess sketch bank + quantization-ready low-rank head
```

The packet is successful only if the chess-specific tiny structure beats matched generic tiny alternatives.

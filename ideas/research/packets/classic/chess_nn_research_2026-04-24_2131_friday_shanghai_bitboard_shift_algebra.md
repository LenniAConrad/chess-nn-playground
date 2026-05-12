# Codex Research Packet: Bitboard Shift-Algebra Network

## File Metadata

- Filename: `chess_nn_research_2026-04-24_2131_friday_shanghai_bitboard_shift_algebra.md`
- Generated at: 2026-04-24 21:31
- Weekday: Friday
- Timezone: Asia/Shanghai
- Intended next consumer: Codex
- Status: full architecture packet, not implemented

## One-Sentence Thesis

Chess boards can be processed efficiently as bitboard-like sparse shift algebras: use fixed rule-shaped square-shift operators for piece motion directions, learn small polynomial mixtures of those operators, and classify puzzle-likeness from the resulting interaction fields without dense attention, move search, or attack-graph message passing.

## Why This Direction

Most neural chess models treat the board as either:

- an image
- a square-token set
- a graph of attacks
- a move-delta set
- a line/ray language
- an optimal-transport or topology object

Chess engines, however, often exploit a different representation: bitboards plus fast shifts and masks. This packet asks whether a neural architecture can borrow that computational shape without becoming an engine or a handcrafted evaluator.

The key abstraction:

```text
current board planes
+ fixed chess-shaped shift operators
+ learned low-degree operator polynomials
+ gated bitboard interaction fields
+ compact classifier
```

This is intended to be efficient because it avoids dense `64 x 64` interactions. It repeatedly applies sparse fixed shifts over 8x8 planes.

## Data Contract

Task:

- output `0`: non-puzzle
- output `1`: puzzle-like

Fine labels:

- `0`, `1`, and `2` remain diagnostics only
- train on binary labels
- report the fine-label `3 x 2` diagnostic matrix

First implementation:

- input: `simple_18`
- dataset: existing `crtk_sample_3class` splits
- trainer: shared experimental training pipeline

Forbidden model inputs:

- Stockfish scores, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, dataset provenance, unresolved candidate status, or anything derived from them.
- Engine search, forced-line search, legal mate/stalemate oracles, or future game outcomes.

Allowed model inputs:

- Current board occupancy.
- Side-to-move.
- Castling/en-passant planes.
- Deterministic square coordinates.
- Side-relative coordinates.
- Material/count summaries.
- Fixed current-board shift masks derived only from board geometry and piece movement templates.

## Core Object

Define a small set of fixed sparse shift operators over 64 squares:

```text
S_north
S_south
S_east
S_west
S_ne
S_nw
S_se
S_sw
S_knight_1 ... S_knight_8
S_king_1 ... S_king_8
```

Each operator is a masked shift:

```text
S_k: R^64 -> R^64
```

It moves a field in one legal board direction and zeros wraparound squares.

The model learns low-degree operator polynomials:

```text
P_h(S) x = sum_{path p in P_h} alpha_{h,p}(x) S_{p_m} ... S_{p_2} S_{p_1} x
```

where:

- `h` is a head
- `p` is a short shift path, for example rook-like, bishop-like, knight-like, or king-ring-like
- coefficients `alpha` are learned from current board summaries
- the path list is fixed and small

This gives the network a cheap way to propagate information through chess-shaped displacement patterns.

## Why It Is Not A Duplicate

| Existing family | Difference |
|---|---|
| CNN / dilated CNN | Uses fixed chess movement shifts and path polynomials, not learned local image filters. |
| Kinematic commutator bottleneck | Does not pool Lie commutators `[K_i,K_j]`; it learns direct low-degree shift polynomials and interaction fields. |
| Ray-language automaton | Does not parse token strings on lines or use finite automata. |
| Schur-Ray Line Algebra | Does not solve a line-incidence linear system or use Woodbury compression. |
| Attack graph / sheaf / Hodge | Does not build dynamic attack edges, cochains, restrictions, or Laplacians. |
| One-ply move-delta models | Does not enumerate legal moves or board deltas. |
| Tropical circuit | Does not evaluate min-plus clauses; it computes shift-algebra feature fields. |
| TensorSketch / tensor-ring | Does not approximate high-order token products; it applies sparse board operators. |

## Architecture Sketch

### Step 1: Input Projection

Input:

```text
x: (B, 18, 8, 8)
```

Project to a small width:

```text
u0 = Conv2d(18, D, kernel_size=1)(x)
```

Recommended start:

```text
D = 48
```

### Step 2: Fixed Shift Bank

Implement fixed masked shifts as tensor operations:

```text
shift_north(u)
shift_south(u)
shift_east(u)
shift_west(u)
shift_ne(u)
...
shift_knight_1(u)
...
```

Each shift keeps shape:

```text
(B, D, 8, 8) -> (B, D, 8, 8)
```

No learned parameters in the shift bank.

### Step 3: Path Basis

Build a small fixed path basis:

```text
identity
one_step_orthogonal
one_step_diagonal
two_step_rook
two_step_bishop
three_step_rook
three_step_bishop
knight_jump
king_ring
pawn_capture_forward_left_side_relative
pawn_capture_forward_right_side_relative
```

For each basis path, apply the corresponding shift composition to `u0`.

Example:

```text
rook_two = [S_north S_north, S_south S_south, S_east S_east, S_west S_west]
bishop_two = [S_ne S_ne, S_nw S_nw, S_se S_se, S_sw S_sw]
```

The path outputs are summed or concatenated by path family.

### Step 4: Board-Conditioned Polynomial Coefficients

Compute a compact board summary:

```text
summary = pooled_stem(u0, material_counts, side_to_move)
```

Emit head coefficients:

```text
alpha: (B, H, P)
```

where:

```text
H = number of shift-algebra heads
P = number of path families
```

Normalize coefficients with either:

```text
softmax over P
```

or:

```text
tanh followed by scale / sqrt(P)
```

### Step 5: Shift-Algebra Heads

For each head:

```text
v_h = sum_p alpha[h,p] * path_output[p]
```

Then apply a small gated interaction:

```text
gate_h = sigmoid(Conv1x1([u0, v_h]))
w_h = gate_h * v_h + (1 - gate_h) * u0
```

Collect:

```text
mean/max/topk pooled w_h
occupied-square pooled w_h
king-neighborhood pooled w_h
shift residual ||w_h - u0||
```

### Step 6: Classifier

Concatenate:

```text
shift_features
small CNN summary
coefficient diagnostics
```

Classify with a compact MLP:

```text
logits: (B, 2)
```

## Tensor Contract

```text
input:             (B, 18, 8, 8)
u0:                (B, D, 8, 8)
path_outputs:      (B, P, D, 8, 8)
alpha:             (B, H, P)
head_fields:       (B, H, D, 8, 8)
shift_features:    (B, F)
logits:            (B, 2)
```

## Efficient Implementation

The shift bank can be implemented with padding and slicing, not dense matrices.

Example:

```text
shift_north(u) = pad/slice rows by one
shift_east(u)  = pad/slice cols by one, zero wrapped file
```

Knight shifts are just fixed two-axis shifts with edge masking.

Cost is approximately:

```text
O(B * D * 8 * 8 * number_of_paths)
```

There is no:

- `64 x 64` attention matrix
- dynamic edge list
- Cholesky solve
- move enumeration
- legal search

The model should be simple to batch and fast on GPU.

## Chess Meaning

The architecture has access to displacement patterns that match chess motion templates:

- rook-like pressure through repeated orthogonal shifts
- bishop-like pressure through repeated diagonal shifts
- knight jumps through sparse L-shaped shifts
- king rings through adjacent shifts
- side-relative pawn capture templates

It does not decide whether a move is legal and does not check tactical outcomes. It only uses the current board tensor and fixed movement-shaped displacement operators.

The hypothesis is that puzzle-like positions create distinctive responses under these shift polynomials:

- forcing alignments produce high shift residuals near kings
- batteries and pins produce strong repeated same-direction responses
- knight forks produce concentrated knight-shift activations
- quiet positions produce smoother or more diffuse operator responses

## Central Ablations

| Ablation | What it removes | Why it matters | Expected readout |
|---|---|---|---|
| `cnn_only` | Remove shift-algebra branch | Tests whether shift operators add value | If equal, architecture is unnecessary. |
| `random_shift_bank` | Replace chess shifts with random fixed square permutations of matched sparsity | Tests chess displacement semantics | Should degrade if movement templates matter. |
| `orthogonal_only` | Keep rank/file shifts, remove diagonals, knights, pawn captures | Tests non-rook movement value | Should lose bishop/knight/pawn motif signal. |
| `one_step_only` | Remove repeated shift paths | Tests long displacement propagation | Should weaken line and battery patterns. |
| `fixed_alpha` | Use global learned coefficients, not board-conditioned coefficients | Tests dynamic path mixing | Board-conditioned should help varied positions. |
| `no_gate` | Replace gated fusion with simple sum | Tests current-board modulation | Gate should improve blocker/noise handling. |
| `dense_conv_matched` | Replace shift paths with matched-depth separable convolutions | Tests fixed chess operators versus generic local filters | Shift algebra should be competitive if chess geometry matters. |

## Diagnostics

Required shared diagnostics:

- binary accuracy
- AUROC
- PR-AUC
- Brier
- ECE
- fine-label `3 x 2` matrix

Architecture-specific diagnostics:

- learned coefficient distribution by label
- coefficient entropy by label
- top path family per correctly classified puzzle-like position
- shift residual heatmaps
- king-zone shift residual statistics
- random-shift ablation gap
- one-step-only ablation gap
- forward time versus CNN and piece-token hybrid

Useful visualizations:

- head field overlays for rook-like paths
- head field overlays for bishop-like paths
- knight-jump activation around candidate fork squares
- examples where shift branch fixes CNN false negatives

## Expected Positive Result

The idea is promising if:

```text
full model > cnn_only
full model >= dense_conv_matched
random_shift_bank drops clearly
one_step_only drops on line-heavy examples
coefficient diagnostics are not just material buckets
```

The most important falsifier is `random_shift_bank`. If random sparse square permutations match the real chess shifts, the model is probably using extra capacity rather than chess operator semantics.

## Failure Modes

- It may act like a fixed convolution bank and add little beyond a good CNN.
- Repeated shifts may ignore blockers unless gates learn to suppress through occupied squares.
- Board-conditioned coefficients may collapse to a static average.
- The model may learn material/phase shortcuts through the summary that emits `alpha`.
- Knight and pawn shift paths may be too sparse to matter at small width.

## Anti-Shortcut Controls

### Material Bucket Evaluation

Report validation/test metrics inside coarse material buckets. If gains disappear after material bucketing, the path coefficients may be material shortcuts.

### Coefficient Probe

Train a small probe from `alpha` to material counts and phase buckets. If it predicts those too well, report metrics with `alpha` stopped or regularized.

### Geometry Destruction

Use `random_shift_bank` with:

- same number of operators
- same per-operator sparsity
- same channel count
- same path depth

Only the chess displacement semantics should be destroyed.

### Path Family Dropout

During training, randomly drop full path families with small probability. This reduces reliance on a single shortcut path.

## Implementation Notes

Recommended first version:

```text
D = 48
H = 6
P = 12 to 20 path families
path_depth_max = 3
classifier_width = 128
use_small_cnn_summary = true
```

Use side-relative pawn shifts so the model does not have to separately learn whose pawns move upward.

Do not include legal occupancy stop rules in the first version. That would move the design closer to attack generation. Instead, let gates learn current-board modulation from occupancy.

## Possible File Targets

```text
src/chess_nn_playground/models/trunk/bitboard_shift_algebra.py
tests/test_bitboard_shift_algebra.py
configs/model/bitboard_shift_algebra.yaml
```

Unit tests should verify:

- each shift zeroes wraparound squares
- knight shifts match expected square displacement
- side-relative pawn shifts flip correctly by side-to-move
- output logits shape is `(B, 2)`
- random-shift ablation preserves tensor shapes

## Minimal Config

```yaml
model:
  name: bitboard_shift_algebra
  input_channels: 18
  width: 48
  heads: 6
  path_depth_max: 3
  coefficient_mode: tanh_scaled
  use_gated_fusion: true
  use_cnn_summary: true
training:
  loss: cross_entropy
  binary_target: true
diagnostics:
  fine_label_matrix: true
  log_path_coefficients: true
ablations:
  - cnn_only
  - random_shift_bank
  - orthogonal_only
  - one_step_only
  - fixed_alpha
  - no_gate
  - dense_conv_matched
```

## Duplicate Guardrail

Do not repeat this idea as:

- another bitboard shift model with only a different hidden width
- another kinematic motion-operator packet unless it changes the central observable beyond low-degree shifts
- another ray model where the only change is using shifts instead of line strings
- another CNN with fixed directional kernels but no chess-shift falsifier

Revisit only if:

- `random_shift_bank` fails clearly
- shift residual diagnostics show interpretable chess patterns
- the model is materially faster or simpler than attention/token alternatives

## Best Immediate Experiment

Run:

```text
BitboardShiftAlgebraNet
cnn_only ablation
random_shift_bank ablation
dense_conv_matched ablation
```

on the same `simple_18` binary puzzle-likeness setup. If the full model beats the controls with similar parameter count and faster forward time than token attention, this becomes a good practical architecture candidate.


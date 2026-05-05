# Codex Research Packet: Schur-Ray Line Algebra Network

## File Metadata

- Filename: `chess_nn_research_2026-04-24_2127_friday_shanghai_schur_ray_line_algebra.md`
- Generated at: 2026-04-24 21:27
- Weekday: Friday
- Timezone: Asia/Shanghai
- Intended next consumer: Codex
- Status: full architecture packet, not implemented

## One-Sentence Thesis

Chess can be abstracted as a small line-incidence algebra over ranks, files, diagonals, blockers, and king zones; a Schur-complement/Woodbury layer can compute global sliding-line interactions through a small compressed linear solve instead of dense square-square attention or large graph propagation.

## Design Reasoning After A Long Pass

Several chess abstractions are already represented in the research archive:

- board as image-like tensor
- occupied pieces as a set
- pseudo-legal attack graph
- attack sheaf or Hodge tension complex
- one-ply move-delta multiset
- transport between pieces and target measures
- topology of pressure fields
- finite ray languages
- subspace, spectrum, and matrix functional bottlenecks

The strongest new opening is not another attack graph and not another move generator. The chess-specific structure that remains underused is the fact that much tactical causality is line-based:

- rooks and queens operate on ranks and files
- bishops and queens operate on diagonals
- pins and skewers are line constraints through blockers
- batteries are multi-piece line alignments
- king danger often depends on line openings and closures
- quiet non-puzzles often have similar material but less line leverage

This packet treats those lines as a compact linear algebra basis. It uses a current-board-only line incidence matrix and a small Schur complement solve to expose line tension efficiently.

## Data Contract

Task:

- output `0`: non-puzzle
- output `1`: puzzle-like

Fine labels:

- `0`, `1`, and `2` remain diagnostics only
- train the first version on the binary target
- always report the fine-label `3 x 2` diagnostic matrix

First implementation target:

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
- Deterministic transforms of current board tensors.
- Rule-only rank/file/diagonal incidence built from current board occupancy.

## Core Abstraction

Let `N = 64` be the number of board squares.

Build a fixed square-to-line incidence matrix:

```text
B in {0,1}^{N x L}
```

where `L = 46` for:

```text
8 ranks
8 files
15 diagonals
15 anti-diagonals
```

Each square belongs to exactly:

```text
one rank
one file
one diagonal
one anti-diagonal
```

The board tensor produces:

```text
b_h in R^N       source field for head h
d_h in R^N       positive square stiffness for head h
g_h in R^N       blocker/occupancy/king-zone gate for head h
M_h in R^{L x r} compressed line-mode matrix for head h
```

The compressed line incidence operator is:

```text
U_h = diag(g_h) B M_h
```

where `r` is small, for example `r = 8, 12, or 16`.

The model does not learn a dense `64 x 64` attention matrix. It learns a small number of board-conditioned line modes that span rank/file/diagonal interactions.

## Linear Algebra Operator

For each head `h`, define an equilibrium field `z_h` as the solution of:

```text
(D_h + U_h C_h U_h^T) z_h = D_h b_h
```

where:

```text
D_h = diag(d_h) + eps I
C_h = small positive definite r x r line-coupling matrix
```

Equivalent energy:

```text
E_h(z) =
  0.5 * (z - b_h)^T D_h (z - b_h)
  + 0.5 * z^T U_h C_h U_h^T z
```

The solution is the lowest-energy square field after imposing compressed line constraints. Puzzle-like boards are hypothesized to produce distinctive line leverage, residual energy, and Schur complement spectra.

## Woodbury Form

The direct solve would be `64 x 64`. That is already small, but the point is to make the operator scale by line-rank rather than square-pair count.

Using Woodbury:

```text
(D + U C U^T)^-1
= D^-1 - D^-1 U (C^-1 + U^T D^-1 U)^-1 U^T D^-1
```

Since the right side is `D b`, the equilibrium field is:

```text
z = b - D^-1 U S^-1 U^T b
```

with small Schur system:

```text
S = C^-1 + U^T D^-1 U
```

The only dense solve is `r x r`, not `64 x 64` and not `64 x 64` attention.

## Chess Meaning

The term:

```text
U^T z
```

summarizes line-mode activity. Because `U = diag(g) B M`, it is not a generic pooling matrix:

- `B` says which squares share ranks, files, and diagonals.
- `g` says which squares are currently important or blocked.
- `M` learns compressed mixtures of chess lines.
- `C` learns which line modes should suppress, reinforce, or contrast each other.

The Schur solve asks:

```text
How much must the square field change to satisfy the current board's line constraints?
```

That is a plausible puzzle-likeness signal because many tactics are created when line constraints are almost but not quite satisfiable:

- a pinned piece appears to defend locally but fails along a line
- an x-ray line is hidden by one blocker
- a battery becomes relevant if a blocker moves
- a back-rank motif depends on line pressure and escape-space geometry
- an overloaded defender sits at the intersection of several line constraints

The model does not need legal move search to detect the static line algebra that makes those motifs possible.

## Why This Is Efficient

Dense square attention over 64 squares typically constructs an interaction object like:

```text
A in R^{64 x 64}
```

for every layer/head.

This packet instead constructs:

```text
U in R^{64 x r}
S in R^{r x r}
```

with `r << 64`.

Per head approximate cost:

```text
build line sums:       O(N L) or scatter-add over 4 memberships per square
compress to U:         O(N r)
form U^T D^-1 U:       O(N r^2)
solve S:               O(r^3)
recover z:             O(N r)
```

With `N = 64` and `r = 12`, the solve is tiny. The board is small, so wall-clock speedup over a minimal CNN is not guaranteed. The stronger efficiency claim is:

- fewer pairwise interaction parameters
- no dense learned `64 x 64` attention map
- no pseudo-legal edge list explosion
- global line coupling in one compact solve
- stable ablations against full `64 x 64` direct solves

## Why It Is Not A Duplicate

| Existing family | Why this differs |
|---|---|
| Tactical sheaf/Hodge/attack graph | No edge cochains, no sheaf restrictions, no graph message passing, no face/cell complex. |
| Non-backtracking walk | No walk propagation over attack edges. |
| Ray-language automaton | No finite automaton over token strings; line incidence is used as a linear operator. |
| Kinematic commutator | No Lie commutators of move operators. |
| Harmonic potential solver | No grid Laplacian/Poisson solve; the operator is low-rank line incidence plus diagonal square stiffness. |
| Matrix-pencil/Grassmannian/Polar packets | Those use token/subspace matrix geometry; this uses current-board line incidence and Woodbury compression. |
| Tensor-ring square interactions | That models high-order token interactions; this models low-rank line equilibrium. |
| Attention-inspired packets | No query-key-value softmax and no all-square attention matrix. |

## Architecture Sketch

### Step 1: Board Stem

Input:

```text
x: (B, 18, 8, 8)
```

Use a small projection:

```text
stem = Conv2d(18, D, kernel_size=1)
```

Add deterministic coordinate planes before the projection or use them as separate input channels if the repo already supports that.

### Step 2: Head Fields

For `H` heads, emit:

```text
b:         (B, H, 64)
raw_d:     (B, H, 64)
raw_g:     (B, H, 64)
line_feat: (B, L, F)
```

Convert:

```text
d = softplus(raw_d) + eps
g = sigmoid(raw_g)
```

Line features can be built by fixed scatter-add:

```text
line_feat = B^T square_features
```

plus line-type embeddings:

```text
rank, file, diagonal, anti_diagonal
```

### Step 3: Compressed Line Modes

Emit:

```text
M: (B, H, L, r)
```

from line features with a tiny MLP. Normalize columns so line modes do not explode:

```text
M_col = M_col / sqrt(eps + sum_l M_col[l]^2)
```

Build:

```text
U = diag(g) B M
```

Implementation can avoid materializing `B` densely. Each square has four line indices, so `B M` is just a sum of four rows of `M`.

### Step 4: Small Schur Solve

For each batch item and head:

```text
D_inv = 1 / d
G = U^T diag(D_inv) U
C_inv = positive_definite_parameterization()
S = C_inv + G
y = U^T b
a = solve(S, y)
z = b - D_inv * (U a)
```

Where:

```text
a: line-mode correction coefficients
z: equilibrium square field
```

Use Cholesky:

```text
a = cholesky_solve(y, cholesky(S))
```

### Step 5: Readout

For each head collect:

```text
mean(z)
max(z)
std(z)
topk_mean(z)
mean(abs(z - b))
line_correction_norm = ||a||
schur_logdet = logdet(S)
schur_trace = trace(S)
energy_data = (z - b)^T D (z - b)
energy_line = z^T U C U^T z
king_zone_energy
slider_line_energy
```

Concatenate with a small CNN summary and classify:

```text
logits = MLP([schur_features, cnn_summary])
```

## Tensor Contract

```text
input:             (B, 18, 8, 8)
square_features:   (B, D, 8, 8)
flat_features:     (B, 64, D)
b:                 (B, H, 64)
d:                 (B, H, 64)
g:                 (B, H, 64)
B_line:            fixed implicit (64, 46)
M:                 (B, H, 46, r)
U:                 (B, H, 64, r)
S:                 (B, H, r, r)
a:                 (B, H, r)
z:                 (B, H, 64)
features:          (B, Fout)
logits:            (B, 2)
```

## Parameterization Details

### Positive Diagonal Stiffness

Use:

```text
d = softplus(raw_d) + 1e-3
```

Interpretation:

- high `d`: square wants to keep its source value `b`
- low `d`: square is flexible under line constraints

### Positive Line Coupling

Simplest version:

```text
C = diag(softplus(c_raw) + 1e-3)
```

Then:

```text
C_inv = diag(1 / c)
```

Second version:

```text
C = L L^T + beta I
```

where `L` is a learned lower-triangular `r x r` matrix per head, not per example.

Start diagonal. Full `C` is an ablation.

### Blocker Gates

The gate `g` should be learned from current board features, but it should receive explicit safe features:

```text
occupied
own_piece
opponent_piece
king_zone
slider_piece
side_to_move
coordinate planes
```

It should not receive legal checkmate labels, engine pressure maps, or move-search outputs.

## Central Ablations

| Ablation | What it removes | Why it matters | Expected readout |
|---|---|---|---|
| `cnn_only` | Remove Schur-Ray branch | Tests whether the line algebra branch adds signal | If equal, line equilibrium is unnecessary. |
| `dense_attention_matched` | Replace Schur-Ray with small square-token attention of matched params | Tests efficient line basis versus generic interactions | Schur-Ray should match or beat attention if line structure matters. |
| `direct_64_solve` | Solve a full learned `64 x 64` PSD system | Tests low-rank Woodbury approximation | If direct solve greatly improves, rank `r` is too small. |
| `random_line_incidence` | Replace `B` with fixed random square-line incidence with same column counts | Tests chess line geometry | Should degrade if rank/file/diagonal structure matters. |
| `rank_file_only` | Remove diagonal and anti-diagonal columns | Tests bishop/queen diagonal line value | Should lose diagonal tactic signal. |
| `diag_only` | Use only diagonal square stiffness, no `U C U^T` | Tests line coupling | Should reduce to per-square source transform. |
| `fixed_M` | Use learned global `M`, not board-conditioned `M(x)` | Tests dynamic line modes | Dynamic should help varied positions. |
| `no_blocker_gate` | Set `g = 1` for all squares | Tests current occupancy modulation | Should over-smooth through blockers. |
| `large_r` | Increase `r` from 12 to 32 | Tests compression sufficiency | If little gain, small rank is enough. |

## Diagnostics

Required:

- Fine-label `3 x 2` diagnostic matrix for main model and central ablations.
- Binary accuracy, AUROC, PR-AUC, Brier, and ECE.
- Parameter count and average forward time against a matched CNN and matched attention baseline.

Architecture-specific:

- Schur eigenvalue spectrum by label.
- Schur logdet by label.
- Line correction norm `||a||` by label.
- Mean absolute correction `mean(abs(z - b))` by label.
- Energy split:

```text
data_energy
line_energy
king_zone_energy
slider_line_energy
```

- Top contributing line modes for correctly classified puzzle-like positions.
- Random-line ablation gap.
- Rank/file-only versus full-line gap.
- Direct-solve versus Woodbury-compressed gap.

Visualization:

- Board heatmap of `abs(z - b)`.
- Per-line contribution bars for ranks/files/diagonals.
- King-zone line tension heatmap.
- Examples where Schur-Ray corrects CNN false negatives.

## Mathematical Properties

### Proposition 1: Positive Definiteness

If:

```text
D is diagonal with strictly positive entries
C is positive semidefinite
```

then:

```text
A = D + U C U^T
```

is positive definite.

Proof sketch:

```text
v^T A v = v^T D v + v^T U C U^T v
```

The first term is strictly positive for all nonzero `v` because `D` is positive diagonal. The second term is nonnegative because `C` is positive semidefinite. Therefore `A` is positive definite.

### Proposition 2: Woodbury Equivalence

For invertible `D` and `C`, the Woodbury expression computes the same `z` as the direct solve:

```text
z = (D + U C U^T)^-1 D b
```

because the Woodbury identity gives the exact inverse. Approximation enters only through the choice of compressed rank `r`, not through the solve formula.

### Proposition 3: No Dense Pairwise Attention

The induced square interaction matrix:

```text
K = U C U^T
```

has rank at most `r`. Therefore it cannot represent arbitrary square-square attention. This is a constraint, not a bug: the model is forced to explain global interactions through a low-dimensional chess-line basis.

## Expected Signal If The Idea Is Right

The model should improve over a small CNN especially on:

- positions where long sliding lines matter
- back-rank and king-line motifs
- pins and skewers
- batteries
- overloaded defenders on line intersections
- positions where material counts are similar but line geometry differs

The strongest evidence would be:

```text
Schur-Ray > CNN-only
Schur-Ray >= matched dense attention
random_line_incidence drops
rank_file_only drops on diagonal motifs
direct_64_solve does not greatly exceed r=12 Woodbury
```

## Expected Signal If The Idea Is Wrong

Treat the idea as falsified if:

- `cnn_only` matches it across seeds.
- `random_line_incidence` matches it.
- `diag_only` matches it.
- `direct_64_solve` is much better but small-rank variants fail.
- Diagnostics show the branch mainly encodes material or piece count.
- Fine-label class `2` recall does not improve and calibration worsens.

## Implementation Sketch

### New Files

```text
src/chess_nn_playground/models/schur_ray_line_algebra.py
tests/test_schur_ray_line_algebra.py
configs/model/schur_ray_line_algebra.yaml
```

### Fixed Incidence Builder

Create a deterministic helper:

```text
build_square_line_indices()
```

Return four integer tensors per square:

```text
rank_idx[s]
file_idx[s]
diag_idx[s]
anti_diag_idx[s]
```

Use these to gather line-mode rows without dense `B`.

### Forward Pseudocode

```text
def forward(x):
    square = stem(x)                         # (B, D, 8, 8)
    flat = square.flatten_board()            # (B, 64, D)

    b = source_head(flat)                    # (B, H, 64)
    d = softplus(stiffness_head(flat)) + eps # (B, H, 64)
    g = sigmoid(gate_head(flat))             # (B, H, 64)

    line_feat = scatter_square_to_lines(flat)
    M = line_mode_head(line_feat)            # (B, H, 46, r)
    M = normalize_columns(M)

    U_base = gather_four_line_modes_per_square(M)
    U = g[..., None] * U_base                # (B, H, 64, r)

    Dinv = 1.0 / d
    G = einsum("bhnr,bhn,bhns->bhrs", U, Dinv, U)
    S = C_inv[None, :, :, :] + G
    y = einsum("bhnr,bhn->bhr", U, b)
    a = cholesky_solve(S, y)
    Ua = einsum("bhnr,bhr->bhn", U, a)
    z = b - Dinv * Ua

    schur_features = summarize(z, b, d, U, a, S)
    cnn_features = cnn_summary(square)
    return classifier(concat(schur_features, cnn_features))
```

### Numerical Notes

Use:

```text
S = S + jitter * I
jitter = 1e-4
```

Clamp or regularize:

```text
d
g
M column norms
C diagonal
```

If Cholesky fails, log the batch and fall back to:

```text
torch.linalg.solve(S + larger_jitter * I, y)
```

for debugging only.

## Minimal Config

```yaml
model:
  name: schur_ray_line_algebra
  input_channels: 18
  stem_width: 64
  heads: 6
  line_rank: 12
  line_count: 46
  c_parameterization: diagonal
  use_blocker_gate: true
  use_cnn_summary: true
  jitter: 1.0e-4
  eps: 1.0e-3
training:
  loss: cross_entropy
  binary_target: true
diagnostics:
  fine_label_matrix: true
  log_schur_spectrum: true
  log_line_energy: true
ablations:
  - cnn_only
  - dense_attention_matched
  - direct_64_solve
  - random_line_incidence
  - rank_file_only
  - diag_only
  - fixed_M
  - no_blocker_gate
```

## Efficiency Benchmark Plan

For each of these:

```text
small CNN baseline
Piece-Token CNN Hybrid
matched square-token attention
Schur-Ray r=8
Schur-Ray r=12
Schur-Ray r=16
direct 64 solve
```

Report:

```text
validation AUROC
test AUROC
PR-AUC
ECE
parameter count
forward milliseconds per 1024 boards
peak CUDA memory if available
```

The efficiency claim is only meaningful if the model is competitive. If it is fast but weak, archive it as a useful negative result.

## Anti-Shortcut Controls

### Material Control

Train a nuisance probe from Schur features to material counts. If Schur features predict material too well and puzzle-likeness improvement disappears under material-bucket evaluation, the branch is probably a material shortcut.

### Coordinate Control

Run with coordinate planes removed and compare:

```text
full coordinates
side-relative only
no coordinates
```

Line incidence already supplies geometry, so the model should not require absolute coordinates to work at all.

### Random-Line Control

Construct a fake incidence matrix with the same column counts and square degrees as `B`, but shuffled square memberships. This is the central falsifier for chess-line geometry.

### Blocker Control

Compare:

```text
learned blocker gate
occupancy-only fixed blocker gate
all-ones gate
random gate with same mean
```

This distinguishes line geometry from raw line pooling.

## Extensions If First Results Are Good

### Segment-Aware Basis

Instead of full rank/file/diagonal lines, split lines into current-board segments between occupied blockers. Then compress segments back to rank `r` before the Schur solve.

Guardrail:

- keep the solve in compressed `r x r` space
- do not drift into explicit attack-edge graph propagation

### Role-Specific Heads

Use head types:

```text
own_slider_lines
opponent_slider_lines
king_zone_lines
empty_escape_lines
blocker_lines
```

Keep these as head embeddings and gates, not as hard-coded legal-tactic labels.

### Low-Rank Update Across Training Steps

If batching many similar boards, cache fixed `B` gather structures and reuse Cholesky allocation paths. This is an engineering optimization, not a modeling change.

### Hybrid With Piece-Token CNN

Fuse Schur-Ray features with the already promising `Piece-Token CNN Hybrid`. That hybrid supplies piece identity and local pattern recognition; Schur-Ray supplies global line equilibrium.

## Duplicate Guardrail

Do not repeat this idea as:

- another low-rank line solve with only a different rank
- another rank/file/diagonal pooling model without Schur/Woodbury diagnostics
- another attack graph model described using matrix notation
- another ray-language automaton with linear algebra words
- another harmonic potential model with line stencils

Only revisit it if one of these changes:

- the central operator changes from Woodbury line equilibrium to a different falsifiable linear-algebra object
- the direct solve shows clear benefit and motivates a different compression
- diagnostics reveal a specific failure, such as blocker gates or line-mode collapse

## Best Immediate Next Step

Promote this to implementation only after the current practical baseline path is stable. The model is mathematically clean and implementable, but it needs careful diagnostics. A good first implementation would be:

```text
heads = 4
line_rank = 8
C = diagonal
U built from fixed 46-line incidence
small CNN summary included
random-line and diag-only ablations included from day one
```

The central question is simple:

```text
Can low-rank line equilibrium recover most of the useful long-range chess interaction signal more efficiently than dense square attention?
```


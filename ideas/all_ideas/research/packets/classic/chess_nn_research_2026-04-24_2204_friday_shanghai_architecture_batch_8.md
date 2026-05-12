# Codex Research Batch: Additional Architecture Candidates 8

## File Metadata

- Filename: `chess_nn_research_2026-04-24_2204_friday_shanghai_architecture_batch_8.md`
- Generated at: 2026-04-24 22:04
- Weekday: Friday
- Timezone: Asia/Shanghai
- Intended next consumer: Codex
- Status: draft architecture batch, not implemented

## Purpose

This batch adds six more chess neural architecture ideas after the tiny-network packet. The goal is to avoid direct repeats of the already explored families:

- no new sheaf or Hodge graph variant
- no new move-delta landscape
- no new optimal-transport variant
- no new attention wrapper
- no new square-pair tensor-core model
- no new tiny CNN variant
- no new Schur-Ray line solve
- no new bitboard shift polynomial

The emphasis here is on unusual computation objects that still fit the current-board-only rule:

- consistency of learned maps between different board views
- convex support envelopes over piece fields
- differentiable line sorting and majorization
- low-displacement-rank global operators
- submodular tactical coverage
- elimination and pivot traces

These are research candidates, not benchmark results.

## Shared Data Contract

All candidates target the binary puzzle-likeness task:

- output `0`: non-puzzle
- output `1`: puzzle-like

Fine labels `0`, `1`, and `2` remain diagnostics only.

First implementations should use:

- input: `simple_18`
- dataset: existing `crtk_sample_3class` splits
- trainer: shared experimental training pipeline

Forbidden model inputs:

- Stockfish scores, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, dataset provenance, unresolved candidate status, or anything derived from them.
- Engine search, forced-line search, legal mate/stalemate oracles, tablebase outcomes, or future game outcomes.

Allowed model inputs:

- Current board occupancy.
- Side-to-move, castling, and en-passant planes already present in `simple_18`.
- Deterministic square coordinates and side-relative coordinates.
- Deterministic current-board transforms such as fixed masks, fixed line maps, fixed geometric projections, and current occupancy counts.

## Ranked Shortlist

| Rank | Candidate | Main object | Why expand it |
|---|---|---|---|
| 1 | Commutative View-Consistency Network | Learned maps between board, line, piece, and region views | Tests whether puzzle-like positions create cross-view inconsistency or agreement defects. |
| 2 | Support-Function Envelope Network | Convex-geometry support widths of learned piece fields | Compact global shape descriptor with interpretable directional envelopes. |
| 3 | Soft Majorization Line Sorter | Differentiable sorted salience profiles along chess lines | Captures overloaded lines and dominance without ray grammars or attention. |
| 4 | Low-Displacement-Rank Board Operator | Toeplitz/Hankel-like structured global mixing | Efficient long-range mixing through structured linear algebra, distinct from Schur-Ray. |
| 5 | Submodular Coverage Bottleneck | Diminishing-return coverage over learned tactical concepts | Tests whether puzzle evidence behaves like coverage/saturation rather than additive pooling. |
| 6 | Pivot Trace Elimination Network | Fixed Gaussian-elimination-style pivot statistics | Uses algebraic elimination traces as a compact board interaction signature. |

Best next full packet from this batch:

```text
Commutative View-Consistency Network
```

Reason: it can reuse simple deterministic views, has clear ablations, and is less likely to duplicate an existing single mathematical family.

## Candidate 1: Commutative View-Consistency Network

### Thesis

A chess position can be represented through several safe current-board views:

- square grid
- occupied piece set
- rank/file/diagonal line summaries
- king-centered regions
- material and phase summaries

Puzzle-like positions may be recognizable not only by any one view, but by how these views agree or disagree after learned projections into a common latent space. The model classifies from commutator-like consistency defects between view maps.

### Fingerprint

```text
multiple deterministic board views
+ learned small encoders per view
+ learned maps between view latents
+ path-consistency residuals
+ binary puzzle-likeness head
```

### Why It Is Distinct

- Not the kinematic commutator packet: those commutators are between deterministic chess motion operators.
- Not the variational residual packet: no action functional or Euler-Lagrange residual.
- Not attention: no query-key routing between views.
- Not a simple multi-branch CNN: the readout is dominated by map-consistency residuals.

### Core Idea

Build latent views:

```text
z_square
z_piece
z_line
z_region
z_count
```

Each view is encoded from allowed current-board information.

Then learn small linear maps between them:

```text
A_square_to_line
A_line_to_square
A_piece_to_region
A_region_to_piece
A_count_to_square
```

For paths that should describe the same board evidence, compute defects:

```text
d1 = z_line - A_square_to_line z_square
d2 = z_square - A_line_to_square z_line
d3 = A_square_to_region z_square - A_piece_to_region z_piece
d4 = A_region_to_count z_region - A_square_to_count z_square
```

Pool:

```text
norm(d_i)
signed mean(d_i)
max_abs(d_i)
cosine(z_target, A z_source)
```

Then classify from:

```text
[all z_view summaries, all defect summaries]
```

The central bet:

```text
near-tactical positions create unusual cross-view consistency patterns
```

For example, material counts may look ordinary while line and king-region views disagree strongly with the same-count expectation.

### Architecture Sketch

View encoders:

```text
square_encoder: small CNN over (B, 18, 8, 8)
piece_encoder: DeepSets over occupied piece tokens
line_encoder: MLP over rank/file/diagonal summaries
region_encoder: MLP over fixed king/center/edge masks
count_encoder: MLP over safe material/state counts
```

Latent width:

```text
D = 32 or 64
```

Map form:

```text
A_{u->v} = low_rank_linear(D, rank=8)
```

Defect features:

```text
defect_norm = mean(d * d)
defect_l1 = mean(abs(d))
defect_signed = mean(d)
defect_cos = cosine(z_v, A z_u)
```

Head:

```text
MLP(defect_features + view_features) -> logits
```

### Tensor Contract

```text
board:        (B, 18, 8, 8)
z_square:     (B, D)
z_piece:      (B, D)
z_line:       (B, D)
z_region:     (B, D)
z_count:      (B, D)
defects:      (B, K)
logits:       (B, 2)
```

### Central Ablations

| Ablation | Change | Claim Tested | Failure Meaning |
|---|---|---|---|
| `views_only_no_defects` | Remove all consistency residuals | Defects add information beyond multi-view features | If equal, consistency is decorative. |
| `single_square_view` | Use only square CNN with matched params | Multi-view structure matters | If equal, view system is unnecessary. |
| `random_view_maps` | Freeze random maps with matched scale | Learned cross-view maps matter | If equal, residual norms are generic regularizers. |
| `count_to_all_only` | Predict every view only from counts | Detects material shortcut | If strong, report count-stratified metrics. |
| `shuffled_piece_view` | Shuffle piece token squares inside batch | Piece-square consistency matters | Should degrade if piece view contributes real geometry. |

### Diagnostics

- Defect norm by fine label.
- Largest defect path per validation sample.
- Correlation of each defect with material count.
- View ablation attribution.
- Examples where count view is normal but line/region defects are high.

### Failure Modes

- The model may become a larger multi-branch MLP.
- Defect norms may correlate mostly with material imbalance.
- If view encoders are too weak, defects are arbitrary.

### Implementation Notes

Keep the first version small:

```text
D = 32
square_width = 32
piece_width = 32
line_width = 32
map_rank = 8
```

Do not add contrastive pretraining in the first pass. Train end-to-end with the existing binary target and use ablations to decide whether consistency residuals matter.

## Candidate 2: Support-Function Envelope Network

### Thesis

A chess position has geometric envelopes: where the side-to-move has force, where the opponent has force, how far pieces extend toward the king, and how concentrated material is along important directions. A differentiable support-function readout can summarize these envelopes compactly without attention, move generation, or graph construction.

### Fingerprint

```text
learned piece fields
+ fixed direction probes
+ soft support functions
+ width/asymmetry/overlap descriptors
+ binary head
```

### Convex-Geometry Object

For a learned nonnegative field `rho_c(s)` over board squares and a direction vector `u`, compute a soft support value:

```text
h_c(u) = tau * logsumexp_s((dot(u, coord_s) + a_c(s)) / tau)
```

where:

```text
a_c(s) = log(epsilon + rho_c(s))
```

For opposite direction:

```text
h_c(-u)
```

Width:

```text
w_c(u) = h_c(u) + h_c(-u)
```

Center:

```text
m_c(u) = h_c(u) - h_c(-u)
```

Compare own and opponent fields:

```text
overlap_gap(u) = abs(m_own(u) - m_opp(u))
width_ratio(u) = w_own(u) / (epsilon + w_opp(u))
```

### Architecture Sketch

1. Project board to `C=8..16` nonnegative fields:

```text
rho = softplus(Conv1x1_or_small_CNN(x))
```

2. Use fixed directions:

```text
rank
file
two diagonals
knight-like slopes
king-to-king direction
center-to-corner directions
```

3. Compute support descriptors for each field and direction.
4. Add field mass and entropy descriptors.
5. Feed descriptors into a small MLP.

### Why It Is Distinct

- Not topology: no components, Betti curves, or Euler characteristic.
- Not optimal transport: no source-target matching.
- Not harmonic potential: no Laplacian solver.
- Not TinyChessMicroNet: this is support-function geometry, not fixed line sketch pooling.

### Central Ablations

| Ablation | Change | Claim Tested | Failure Meaning |
|---|---|---|---|
| `mean_pool_fields` | Replace support descriptors with mean/max pools | Envelope geometry matters | If equal, support functions are unnecessary. |
| `random_directions` | Use random fixed directions with same count | Chess-relevant directions matter | If equal, direction semantics are weak. |
| `no_opponent_contrast` | Remove own/opponent width gaps | Role contrast matters | If equal, single-field geometry dominates. |
| `hard_max_support` | Use max instead of logsumexp | Soft envelope helps | If equal, simpler max is enough. |
| `counts_plus_envelope_only` | Remove learned fields, use piece count maps | Learned fields matter | If equal, no need for neural field stem. |

### Diagnostics

- Top directions by head weight.
- Own/opponent width ratios by label.
- Support centers relative to kings.
- Sensitivity to `tau`.

### Failure Modes

- The model may just learn center control.
- Support descriptors may be too coarse for tactics.
- Soft support may be unstable if fields become too peaky.

### Implementation Notes

Start with:

```text
fields = 12
directions = 16
tau = 0.25
descriptor_dim about 12 x 16 x 4
```

Use coordinate values normalized to `[-1, 1]`.

## Candidate 3: Soft Majorization Line Sorter

### Thesis

On a tactical line, the exact order and dominance of pieces often matters more than a bag of line pieces. Instead of a ray automaton or line language model, compute differentiable sorted salience profiles along ranks/files/diagonals and classify from majorization-style inequalities, gaps, and concentration.

### Fingerprint

```text
line square scores
+ differentiable sorting per line
+ sorted salience gaps and majorization curves
+ line dominance descriptors
+ binary head
```

### Why It Is Distinct

- Not the ray-language automaton: no token grammar or finite automaton states.
- Not line state-space scans: no recurrence.
- Not Schur-Ray: no line solve.
- Not attention: sorting is over scalar salience profiles, not token routing.

### Architecture Sketch

1. Compute scalar salience fields:

```text
S_k: (B, 8, 8), k = 1..K
```

Suggested salience heads:

```text
own_pressure_like
opponent_pressure_like
king_relevance_like
blocker_like
promotion_lane_like
```

2. Extract line vectors for ranks, files, diagonals, anti-diagonals.
3. Apply differentiable sorting or soft rank approximation:

```text
sorted_scores = softsort(line_scores, temperature=tau)
```

4. Compute majorization curves:

```text
cumsum_top_j = cumulative_sum(sorted_scores)
gap_j = sorted_scores_j - sorted_scores_{j+1}
concentration = top_1 / (epsilon + sum_all)
```

5. Pool descriptors over line types and salience heads.
6. Classify from pooled majorization descriptors and a small board context vector.

### Central Ablations

| Ablation | Change | Claim Tested | Failure Meaning |
|---|---|---|---|
| `unsorted_line_stats` | Use mean/max/sum without sorting | Sorted dominance matters | If equal, majorization is unnecessary. |
| `random_line_assignment` | Preserve line lengths but randomize square membership | Chess lines matter | If equal, line geometry is weak. |
| `hard_sort_stopgrad` | Hard sort with stopped permutation gradients | Soft sorting helps learning | If equal, simpler deterministic sort may suffice. |
| `no_gap_features` | Remove adjacent sorted gaps | Dominance gaps matter | If equal, cumulative curves dominate. |
| `local_cnn_same_params` | Parameter-matched small CNN | Line sorter beats generic capacity | If equal, use CNN. |

### Diagnostics

- Most active line type by label.
- Average top-1 concentration by fine label.
- Sorted gap histograms.
- Temperature sensitivity.

### Failure Modes

- Soft sorting can be numerically fiddly.
- It may collapse to max pooling.
- It may overfit to board orientation unless side-relative coordinates are handled carefully.

### Implementation Notes

Use short lines only. Since chess lines length is at most `8`, soft sorting is affordable:

```text
K salience heads = 8
line_count = 46
line_length <= 8
```

If implementing SoftSort is too much, start with deterministic hard sort and no gradients through ordering. The salience field still receives gradients through sorted values.

## Candidate 4: Low-Displacement-Rank Board Operator

### Thesis

Global square mixing can be parameterized by structured matrices instead of dense attention or convolutions. A low-displacement-rank operator over the flattened board can express long-range interactions with Toeplitz/Hankel-like structure and few parameters.

### Fingerprint

```text
flattened board field
+ structured Toeplitz/Hankel/Cauchy-like operator
+ low-displacement-rank residual
+ operator response statistics
+ binary head
```

### Linear Algebra Object

A matrix `A` has low displacement rank if:

```text
Delta(A) = A - Z A Z^T
```

has low rank for a shift matrix `Z`.

Use a board-specific version:

```text
A = T_rank + T_file + H_diag + H_anti + U V^T
```

where:

- `T_rank` shares Toeplitz offsets along rank order
- `T_file` shares Toeplitz offsets along file order
- `H_diag` shares Hankel-like anti-diagonal offsets
- `H_anti` shares the other diagonal family
- `U V^T` is a small learned low-rank correction

Apply:

```text
y = A h
```

for each channel group.

### Why It Is Distinct

- Not Schur-Ray: no line incidence solve or Woodbury complement.
- Not bitboard shift algebra: no low-degree shift polynomial over discrete move operators.
- Not attention: no data-dependent pair weights.
- Not Tensor-Core Pair Field: no dense square-pair state retained.

### Architecture Sketch

1. Project board to `D=32..64` channels.
2. Flatten squares to `N=64`.
3. Apply `L=2..4` structured operator layers:

```text
h_{t+1} = relu(A_t h_t + pointwise_mix(h_t))
```

4. Pool response descriptors:

```text
mean(y)
max(y)
norm(Ah - h)
energy by operator component
```

5. Classify.

### Central Ablations

| Ablation | Change | Claim Tested | Failure Meaning |
|---|---|---|---|
| `diagonal_only_operator` | Remove Toeplitz/Hankel global terms | Global structured mixing matters | If equal, operator is unnecessary. |
| `random_structured_offsets` | Randomize offset-to-square mapping | Chess-aligned structure matters | If equal, structure is generic. |
| `low_rank_only` | Use only `U V^T` | Displacement structure matters | If equal, low rank is enough. |
| `dense_same_params` | Dense matrix bottleneck with matched params | Structured inductive bias matters | If dense wins, structure is too restrictive. |
| `conv_same_params` | Ordinary CNN with matched params | Operator beats local convolution | If equal, use CNN. |

### Diagnostics

- Energy contributed by rank/file/diagonal/Hankel components.
- Learned offset spectra.
- Response norm by fine label.
- Sensitivity to square flattening order.

### Failure Modes

- The board is tiny, so structured operator overhead may not beat CNN speed.
- Toeplitz assumptions may be too translation-like for chess.
- Low-rank correction may dominate, weakening the research claim.

### Implementation Notes

Keep the operator explicit first:

```text
A_component: (64, 64) buffers built from small parameter vectors
```

Then optimize with indexed gathers only if results are promising.

## Candidate 5: Submodular Coverage Bottleneck

### Thesis

Puzzle evidence may behave like coverage: once a tactical theme is strongly present, another redundant cue adds less value, but a distinct cue adds more. A differentiable submodular coverage layer can force the model to aggregate learned concepts with diminishing returns rather than simple additive pooling.

### Fingerprint

```text
learned concept activations
+ differentiable coverage function
+ saturation and marginal-gain descriptors
+ binary head
```

### Submodular Object

For concept activations:

```text
a_i in [0, 1], i = 1..M
```

and learned nonnegative coverage weights:

```text
W_{i,k} >= 0
```

define covered latent attributes:

```text
c_k = 1 - product_i (1 - a_i W_{i,k})
```

Coverage score:

```text
F(a) = sum_k beta_k c_k
```

The diminishing-return behavior comes from product saturation: repeated activation of the same covered attribute adds less.

### Architecture Sketch

1. Extract local and global concept activations:

```text
a = sigmoid(concept_encoder(x))
```

Concept sources:

- square patches
- line summaries
- king-zone summaries
- material/count summaries

2. Apply coverage layer:

```text
c = 1 - prod(1 - a_i W_i)
```

3. Compute marginal descriptors:

```text
gain_i = F(a) - F(a with a_i removed)
```

4. Classify from:

```text
F(a)
c
top marginal gains
concept entropy
```

### Why It Is Distinct

- Not mixture of experts: no sparse routing to expert modules.
- Not attention: no normalized weighted value pooling.
- Not prototype dictionary: coverage is a set function with diminishing returns.
- Not evidential learning: coverage score is an internal bottleneck, not Dirichlet evidence.

### Central Ablations

| Ablation | Change | Claim Tested | Failure Meaning |
|---|---|---|---|
| `additive_pool` | Replace coverage with linear sum of concepts | Diminishing returns matter | If equal, coverage is unnecessary. |
| `no_marginal_gains` | Remove marginal descriptors | Marginal structure matters | If equal, only covered attributes matter. |
| `unconstrained_W` | Allow signed weights | Nonnegative coverage matters | If better, submodular constraint may be too restrictive. |
| `random_concepts` | Freeze random concept encoder | Learned concepts matter | If equal, coverage head is doing all work. |
| `material_concepts_only` | Use only material/count concepts | Detects shortcut | If strong, spatial concepts are weak. |

### Diagnostics

- Concept coverage saturation by label.
- Top marginal concepts for validation examples.
- Number of active concepts at threshold.
- Redundancy curve: `F(top k concepts)` as `k` increases.

### Failure Modes

- Coverage can collapse to one concept.
- Nonnegative constraints can make optimization slower.
- Marginal-gain calculation may add overhead if implemented naively.

### Implementation Notes

Use stable logs:

```text
log_uncovered_k = sum_i log(clamp(1 - a_i W_i, eps, 1))
c_k = 1 - exp(log_uncovered_k)
```

Start with:

```text
M = 64 concepts
K = 32 covered attributes
```

Compute marginal gains exactly for `M=64`; it is small.

## Candidate 6: Pivot Trace Elimination Network

### Thesis

Gaussian elimination exposes interaction structure through pivot sizes, residual norms, and Schur updates. A chess board can be encoded into a small square matrix, then passed through a fixed-order differentiable elimination procedure. The pivot trace becomes a compact algebraic signature of tactical coupling.

### Fingerprint

```text
learned board matrix
+ fixed-order differentiable elimination
+ pivot/residual/Schur trace descriptors
+ binary head
```

### Core Object

Build a learned matrix:

```text
M(x) in R^{K x K}
```

where rows/columns correspond to learned summaries such as:

- piece-type groups
- side roles
- line groups
- king-region groups
- center/edge groups

Then perform stabilized elimination:

```text
for t in 1..K:
    pivot_t = softplus(M_tt) + eps
    row_update = M_{t+1:, t} / pivot_t
    M_{t+1:, t+1:} = M_{t+1:, t+1:} - row_update outer M_{t, t+1:}
    record pivot_t, update_norm_t, residual_norm_t
```

Read out:

```text
log_pivots
update norms
residual decay curve
final residual norm
condition-like ratio
```

### Why It Is Distinct

- Not Matrix-Pencil: no generalized eigenvalues.
- Not Polar-Procrustes: no orthogonal alignment.
- Not Schur-Ray: no line-incidence Schur solve.
- Not determinant volume: no occupied-token Gram logdet as the main readout.

### Architecture Sketch

1. Encode board into group summaries:

```text
g: (B, K, D)
```

2. Build a symmetric or nonsymmetric matrix:

```text
M_ij = small_bilinear(g_i, g_j)
```

3. Add a diagonal stabilizer:

```text
M = M + lambda I
```

4. Run fixed-order elimination.
5. Classify from pivot trace plus summary features.

### Central Ablations

| Ablation | Change | Claim Tested | Failure Meaning |
|---|---|---|---|
| `raw_matrix_pool` | Pool `M` directly, no elimination | Elimination trace matters | If equal, pivots are unnecessary. |
| `random_elimination_order` | Random fixed group order | Semantic order matters | If equal, order is not meaningful. |
| `diagonal_matrix_only` | Zero off-diagonal entries | Interactions matter | Should degrade if coupling matters. |
| `determinant_only` | Use sum of log pivots only | Full trace matters beyond determinant | If equal, trace can be simplified. |
| `matrix_pencil_control` | Compare to same-size eigen/pencil readout if implemented | Pivot trace is competitive with spectral readout | If spectral wins, prefer pencil route. |

### Diagnostics

- Pivot curves by fine label.
- Off-diagonal update norms.
- Residual decay rates.
- Condition-like ratios.
- Sensitivity to diagonal stabilizer.

### Failure Modes

- Elimination can be numerically unstable.
- The matrix builder may dominate the effect.
- Pivot order may be arbitrary if group semantics are weak.

### Implementation Notes

Start tiny:

```text
K = 12
D = 32
lambda = 0.1
```

Do not use learned pivoting in the first version. Learned pivoting would become routing and make the ablation harder to interpret.

## Cross-Batch Comparison

Best candidates from this batch if implementation time is limited:

1. `Commutative View-Consistency Network`
2. `Submodular Coverage Bottleneck`
3. `Soft Majorization Line Sorter`

Reasoning:

- They have clear central falsifiers.
- They can be implemented without external solvers.
- They produce interpretable diagnostics.
- They are less likely to duplicate already-researched families.

The more speculative candidates:

- `Low-Displacement-Rank Board Operator`
- `Pivot Trace Elimination Network`
- `Support-Function Envelope Network`

These may be worth full packets if the next round asks for more math-heavy or linear-algebra-heavy work.

## Anti-Duplicate Rules For Future Packets

Do not repeat these as mere parameter variants:

| Family | Avoid Near-Duplicate |
|---|---|
| View consistency | Another multi-view branch model unless path defects or commutative residuals remain central. |
| Support envelope | Another directional pooling model unless it uses support/width/asymmetry descriptors with clear direction controls. |
| Soft majorization | Another line pooling model unless differentiable sorting or majorization curves are central. |
| Displacement operator | Another structured matrix mixer unless low-displacement rank is explicitly tested against low-rank-only and CNN controls. |
| Submodular coverage | Another concept bottleneck unless diminishing returns and marginal gains are the point. |
| Pivot trace | Another matrix readout unless elimination/pivot traces, not eigenvalues or determinants, are central. |

## Suggested Implementation Queue

If implementing from this batch, use this order:

1. Full packet for `Commutative View-Consistency Network`.
2. Minimal model and tests for deterministic view construction.
3. `views_only_no_defects` and `single_square_view` ablations.
4. If defects matter, add `Submodular Coverage Bottleneck` as a second independent idea.
5. If line diagnostics suggest strong line order effects, then expand `Soft Majorization Line Sorter`.

The guiding question for this batch:

```text
Can a model gain puzzle-likeness signal from relationships among summaries, not only from larger representations?
```

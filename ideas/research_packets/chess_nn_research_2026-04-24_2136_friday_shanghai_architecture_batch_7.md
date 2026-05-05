# Codex Research Batch: Additional Architecture Candidates 7

## File Metadata

- Filename: `chess_nn_research_2026-04-24_2136_friday_shanghai_architecture_batch_7.md`
- Generated at: 2026-04-24 21:36
- Weekday: Friday
- Timezone: Asia/Shanghai
- Intended next consumer: Codex
- Status: draft architecture batch, not implemented

## Purpose

This batch adds six more chess neural architecture ideas after reviewing nearby archived concepts. The emphasis is on computation objects that are still not overrepresented:

- game-theoretic dynamics over occupied pieces
- differentiable Boolean bitboard algebra
- orthogonal coordinate moments
- legal-board constraint projection residuals
- Zobrist-style random feature kernels
- low-rank signed cut queries over board regions

These are research candidates, not benchmark results.

## Shared Data Contract

All candidates target the binary puzzle-likeness task:

- output `0`: non-puzzle
- output `1`: puzzle-like

Fine labels `0`, `1`, and `2` remain diagnostics only. First implementations should use `simple_18`, the existing `crtk_sample_3class` splits, and the shared trainer.

Forbidden model inputs:

- Stockfish scores, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, dataset provenance, unresolved candidate status, or anything derived from them.
- Engine search, forced-line search, legal mate/stalemate oracles, or future game outcomes.

Allowed model inputs:

- Current board occupancy, side-to-move, castling/en-passant planes, deterministic square coordinates, side-relative coordinates, material/count summaries, and deterministic transforms of current board tensors.

## Ranked Shortlist

| Rank | Candidate | Main object | Why expand it |
|---|---|---|---|
| 1 | Replicator Payoff Piece Dynamics | Learned payoff game over occupied pieces | Chess-like attacker/defender tension with interpretable equilibrium dynamics. |
| 2 | Differentiable Bitboard Boolean Network | Soft AND/OR/XOR/NOT over learned bitboard predicates | Efficient and close to how chess rules are expressed. |
| 3 | Orthogonal Board Moment Network | Legendre/Chebyshev moments of piece fields | Compact global shape descriptor distinct from FFT and CNNs. |
| 4 | Legal-Constraint Projection Residual Network | Projection onto soft legal-board constraint sets | Measures contradiction between learned latent beliefs and legal board structure. |
| 5 | Zobrist Kernel Feature Network | Fixed hash-style random features over piece-square facts | Extremely cheap kernelized board fingerprints with strong controls. |
| 6 | Low-Rank Signed Cut Query Network | Learned board-region cut imbalances | Tests region-separation signals without attention or graph construction. |

Best next full packet from this batch: `Replicator Payoff Piece Dynamics`.

## Candidate 1: Replicator Payoff Piece Dynamics

### Thesis

Puzzle-like positions often feel like unstable games among pieces: one attacker increases pressure, one defender is overloaded, one target becomes strategically dominant. A differentiable payoff game over occupied pieces can model this as a small dynamical system and classify from the equilibrium and instability statistics.

### Fingerprint

```text
occupied piece tokens
+ learned pairwise payoff matrices
+ replicator dynamics
+ equilibrium/instability readout
+ binary puzzle-likeness head
```

### Why It Is Distinct

- Not a determinant volume model: no Gram logdet or eigen-volume bottleneck.
- Not attention: pairwise scores define a game update, not query-key softmax pooling.
- Not a move-delta model: no legal move enumeration or future board states.
- Not transport: no source-target marginal matching.

### Architecture Sketch

1. Extract occupied piece tokens:

```text
token_i = piece_type + color + square + side-relative coordinates + local context
```

2. Build a learned payoff matrix:

```text
P_ij = f_theta(token_i, token_j, relative_geometry_ij)
```

3. Split payoffs into role heads:

```text
attack_like
defense_like
king_pressure_like
blocker_like
```

4. Initialize a population distribution over occupied pieces:

```text
p_i = softmax(init_score_i)
```

5. Run `T=4..8` replicator steps:

```text
fitness_i = (P p)_i
avg_fitness = p^T P p
p_i <- p_i * exp(eta * (fitness_i - avg_fitness))
p <- normalize(p)
```

6. Pool:

```text
final p
entropy(p)
fitness variance
avg payoff
top piece mass
mass on kings/attackers/defenders
change from initial p
```

7. Fuse with a small board encoder and classify.

### Tensor Contract

```text
board:           (B, 18, 8, 8)
tokens:          (B, Pmax, D)
mask:            (B, Pmax)
payoff:          (B, H, Pmax, Pmax)
population:      (B, H, Pmax)
dynamics_stats:  (B, S)
logits:          (B, 2)
```

### Central Ablations

| Ablation | What it removes | Why it matters | Expected readout |
|---|---|---|---|
| `mean_pool_tokens` | Remove game dynamics | Tests whether dynamics add signal | Should drop if equilibrium matters. |
| `one_step_only` | Replace iterative dynamics with one payoff pooling | Tests dynamic convergence | Replicator should help if instability matters. |
| `symmetric_payoff` | Force `P = P^T` | Tests directional roles | Asymmetry should matter for attack/defense. |
| `random_payoff_geometry` | Shuffle relative geometry features | Tests chess geometry | Should degrade if geometry matters. |
| `uniform_initial_population` | Remove learned initial salience | Tests salience initialization | Learned init should help but not dominate. |

### Diagnostics

- Final population entropy by label.
- Top-mass piece type and square by label.
- Fitness variance by label.
- Payoff asymmetry norm.
- Examples where a single overloaded defender receives high mass.

### Failure Modes

- It may learn attention-like pooling under another name.
- Payoff matrix can overfit material and piece types.
- Replicator steps can collapse to one piece too early.

### Implementation Notes

Start conservative:

```text
Pmax = 32
heads = 4
steps = 5
eta = 0.5
token_width = 64
```

Use entropy regularization lightly so the dynamics are sharp but not degenerate.

## Candidate 2: Differentiable Bitboard Boolean Network

### Thesis

Chess rules are often written as bitboard Boolean algebra: masks, shifts, intersections, unions, and complements. A neural model can learn soft bitboard predicates and combine them with differentiable Boolean operations, producing an efficient symbolic-neural board processor without explicit move search.

### Fingerprint

```text
board planes
+ learned predicate bitboards
+ soft Boolean algebra layers
+ predicate truth statistics
+ binary head
```

### Core Operators

Use soft values in `[0, 1]`:

```text
NOT(a) = 1 - a
AND(a,b) = a * b
OR(a,b) = a + b - a*b
XOR(a,b) = a + b - 2*a*b
IMPLIES(a,b) = OR(NOT(a), b)
```

Predicates are board fields:

```text
q_k: (B, 1, 8, 8)
```

### Architecture Sketch

1. Project `simple_18` into `K=32` predicate fields.
2. Apply several Boolean layers:

```text
q_new = mix([
  q_i AND q_j,
  q_i OR q_j,
  q_i XOR q_j,
  q_i IMPLIES q_j,
  NOT q_i
])
```

3. Use learned sparse pairing weights so each layer chooses only a few predicate pairs.
4. Pool truth statistics:

```text
global truth mass
king-zone truth mass
occupied-square truth mass
truth entropy
predicate coactivation
```

5. Fuse with small CNN summary and classify.

### Why It Is Distinct

- Not formal concept closure: no Galois closure over object/attribute incidence.
- Not tropical circuit: no min-plus clause satisfaction.
- Not sparse witness: predicates are soft bitboards, not selected occupied pieces.
- Not a hand-coded rule engine: predicates are learned from current board planes.

### Central Ablations

| Ablation | What it removes | Why it matters | Expected readout |
|---|---|---|---|
| `mlp_predicate_mixer` | Replace Boolean ops with MLP on predicates | Tests Boolean algebra | Boolean ops should help if logic-like structure matters. |
| `and_or_only` | Remove XOR and implication | Tests richer logic | Full ops may help tactical contrast. |
| `random_pairing_frozen` | Freeze random predicate pairings | Tests learned predicate composition | Learned pairing should improve. |
| `truth_marginals_only` | Use only per-predicate means | Tests spatial truth maps | Full pooling should help. |
| `cnn_only` | Remove Boolean branch | Tests branch value | If equal, Boolean algebra is unnecessary. |

### Diagnostics

- Predicate truth maps.
- Operation usage frequencies.
- Boolean layer saturation rates.
- XOR/implication contribution by label.
- Random-pairing ablation gap.

### Failure Modes

- Soft Boolean operations can saturate.
- It may become a generic gated CNN branch.
- Learned predicates may collapse to material planes.

### Implementation Notes

Clamp predicate values away from exact `0` and `1` during early training. Use a temperature schedule only if saturation is severe.

## Candidate 3: Orthogonal Board Moment Network

### Thesis

Puzzle-like positions may differ in global spatial moments of piece fields: centralization, skew, diagonal concentration, king-side imbalance, and high-order shape. Orthogonal polynomial moments provide a compact linear-algebra descriptor that is not convolution, FFT phase coupling, or attention.

### Fingerprint

```text
piece fields
+ fixed orthogonal polynomial basis over board coordinates
+ channel/role moment tensors
+ moment interaction head
+ binary classifier
```

### Moment Basis

Use normalized board coordinates:

```text
u, v in [-1, 1]
```

Build low-order bases:

```text
Legendre: P_0, P_1, P_2, P_3
Chebyshev: T_0, T_1, T_2, T_3
mixed terms: P_i(u) P_j(v)
```

For each learned board field `F_c`, compute:

```text
m_{cij} = sum_{squares} F_c(u,v) P_i(u) P_j(v)
```

### Architecture Sketch

1. Project board tensor into `C=24` learned scalar fields.
2. Compute fixed orthogonal moments up to degree `K=3` or `4`.
3. Split moment families:

```text
low degree: material/center balance
middle degree: side/wing skew
high degree: local concentration
```

4. Feed moment tensor through a small MLP with degree dropout.
5. Fuse with a compact CNN summary.
6. Classify.

### Why It Is Distinct

- Not FFT/bispectrum: moments are coordinate-polynomial integrals, not frequency-domain phase coupling.
- Not topology: no threshold curves or connected components.
- Not harmonic potential: no PDE solve.
- Not normal coordinate planes: the classifier sees structured moment summaries, not raw coordinates only.

### Central Ablations

| Ablation | What it removes | Why it matters | Expected readout |
|---|---|---|---|
| `degree0_only` | Keep only global sums | Tests spatial moments | Higher degrees should matter. |
| `raw_pool_same_dim` | Replace moments with learned global pooling of same dimension | Tests orthogonal basis | Moments should improve if basis matters. |
| `random_orthogonal_basis` | Replace polynomial basis with random orthogonal board basis | Tests coordinate semantics | Polynomial basis should help if geometry matters. |
| `no_cnn_summary` | Use moments only | Tests standalone value | Likely weaker but diagnostic. |
| `degree_shuffle` | Shuffle degree labels before MLP | Tests degree structure | Should hurt if hierarchy matters. |

### Diagnostics

- Moment energy by degree and label.
- Degree-wise ablation table.
- Basis sensitivity.
- Moment-only AUROC.
- False positives with extreme material imbalance.

### Failure Modes

- Low-order moments may be too coarse.
- The model may mostly capture material and center control proxies.
- A CNN may learn equivalent features cheaply.

### Implementation Notes

This is very cheap. It is best as a branch in a practical hybrid model.

## Candidate 4: Legal-Constraint Projection Residual Network

### Thesis

Even when the input board is legal, a learned latent explanation of "why this is puzzle-like" may produce soft piece/square beliefs that violate basic legal-board constraints. Projecting those beliefs back onto a soft legal-board constraint set and reading the residual may expose tactical contradiction or ambiguity.

### Fingerprint

```text
board tensor
+ soft latent board belief planes
+ differentiable projection to legal-board constraints
+ projection residual features
+ binary head
```

### Constraint Set

Use only simple current-board legality constraints:

```text
at most one piece per square
one own king and one opponent king
nonnegative piece probabilities
piece-count upper bounds
pawns not on impossible first/eighth ranks if encoded
side/castling/en-passant consistency only if already represented safely
```

Do not use legal move generation, check status, checkmate, tablebases, or engine information.

### Architecture Sketch

1. Board encoder emits soft latent board beliefs:

```text
Y: (B, C_piece, 8, 8)
```

2. Apply differentiable projection:

```text
Y_proj = argmin_Z ||Z - Y||^2
subject to simple legal-board constraints
```

3. Implement projection approximately with a few alternating steps:

```text
square simplex projection
piece-count clipping
king-count normalization
pawn-rank masking
```

4. Compute residual:

```text
R = Y - Y_proj
```

5. Pool residual norms and spatial residual maps.
6. Fuse with original encoder summary and classify.

### Why It Is Distinct

- Not nuisance projection: this projects soft board beliefs onto legal constraints, not latents away from nuisance features.
- Not score-field denoising: no class-0 prior and no Gaussian repair model.
- Not semantic loss: constraints are internal diagnostic projection, not output-label constraints.
- Not move generation: constraints are static board validity only.

### Central Ablations

| Ablation | What it removes | Why it matters | Expected readout |
|---|---|---|---|
| `no_projection_residual` | Remove residual branch | Tests projection value | Should drop if contradiction signal matters. |
| `random_constraints` | Replace legal constraints with random count-preserving projections | Tests legal semantics | Should degrade if constraints matter. |
| `square_only_projection` | Only enforce one-piece-per-square | Tests richer constraints | Full projection should improve. |
| `residual_norm_only` | Use scalar residual norms only | Tests spatial residual maps | Full maps should help. |
| `encoder_only_matched` | Same encoder capacity, no projection | Tests extra compute/capacity | Projection should beat matched encoder. |

### Diagnostics

- Residual mass by constraint type.
- Residual heatmaps.
- Constraint-specific ablation gaps.
- Projection iteration stability.
- Whether positives show higher localized residuals.

### Failure Modes

- Since real inputs are already legal, residuals may be weak.
- The latent belief decoder may learn to avoid constraints trivially.
- Projection can become an expensive no-op.

### Implementation Notes

Use stop-gradient variants:

```text
Y_proj = stopgrad(project(Y))
```

and fully differentiable variants. Compare both.

## Candidate 5: Zobrist Kernel Feature Network

### Thesis

Zobrist hashing gives chess a compact random fingerprint of piece-square occupancy. A neural model can use many fixed Zobrist-style random feature maps as a cheap kernel approximation, then learn a small classifier over stable board fingerprints.

### Fingerprint

```text
piece-square facts
+ fixed random signed hash tables
+ aggregated random feature vector
+ learned calibration MLP
+ binary classifier
```

### Core Features

For each random table `r`:

```text
z_r = sum_{occupied piece-square facts f} sign_r(f) * scale_r(f)
```

Use several nonlinear expansions:

```text
z
z^2
sign(z)
cos(omega z)
pairwise bucketed interactions
```

Keep the random table fixed and seeded.

### Why It Is Distinct

- Not TensorSketch: no randomized polynomial sketch of learned token embeddings.
- Not masked codec: no reconstruction objective.
- Not kernel mean prototypes: no learned prototype distribution.
- Not material counts: random tables bind piece type and square identity.

### Architecture Sketch

1. Convert board to occupied piece-square facts.
2. Apply `R=256..1024` fixed random signed hash features.
3. Concatenate deterministic safe summaries:

```text
side-to-move
castling/en-passant planes pooled
material counts
```

4. MLP classifier.
5. Optional small CNN fusion branch if random features alone are weak.

### Central Ablations

| Ablation | What it removes | Why it matters | Expected readout |
|---|---|---|---|
| `material_only` | Use only material and side summaries | Tests square binding | Zobrist features should beat it. |
| `random_square_reseed_each_epoch` | Destroy stable square binding | Tests fixed fingerprint semantics | Should degrade if binding matters. |
| `permuted_hash_tables` | Preserve hash distribution but randomize piece-square mapping | Tests chess square semantics | Should degrade if board geometry matters. |
| `linear_only` | Remove nonlinear expansions | Tests kernelization | Nonlinear features should help. |
| `cnn_fusion_off` | Use random features only | Tests standalone value | Likely weaker but cheap. |

### Diagnostics

- Feature count versus AUROC.
- Seed variance across random tables.
- Random-feature-only speed.
- Material-only gap.
- Hash collision sensitivity.

### Failure Modes

- Random features may need too many dimensions.
- It may memorize dataset artifacts if splits are not robust.
- Geometry is implicit and may be weaker than CNNs.

### Implementation Notes

This is a useful cheap baseline even if it is not the most elegant architecture. It can be trained extremely fast and used as a sanity comparison for bigger models.

## Candidate 6: Low-Rank Signed Cut Query Network

### Thesis

Puzzle-like positions may separate the board into tense regions: attacking mass versus defending mass, king-side versus center, blocked wing versus open wing. A model can learn low-rank signed cut queries over board fields and classify from imbalance statistics.

### Fingerprint

```text
board fields
+ learned low-rank signed region masks
+ cut imbalance summaries
+ binary head
```

### Core Object

Learn query masks:

```text
a_k(s), b_k(s) in [-1, 1]
```

over squares, constrained low-rank by coordinate factors:

```text
a_k(rank, file) = r_k(rank) * f_k(file)
```

For board field `F_c`, compute signed cut:

```text
cut_{k,c} = sum_s a_k(s) F_c(s) - sum_s b_k(s) F_c(s)
```

Also compute absolute and squared versions:

```text
abs_cut
cut^2
normalized_cut = cut / (eps + total_mass)
```

### Architecture Sketch

1. Project board tensor into `C=24` fields.
2. Learn `K=32` low-rank signed mask pairs.
3. Compute cut summaries.
4. Add side-relative and king-anchored mask variants.
5. Feed summaries to MLP and fuse with a small CNN.

### Why It Is Distinct

- Not attention: masks are global low-rank region queries, not data-dependent token-token weights.
- Not topology: no connected components or threshold curves.
- Not orbit quotient: no invariance pooling.
- Not nuisance projection: cuts are features, not removed subspaces.

### Central Ablations

| Ablation | What it removes | Why it matters | Expected readout |
|---|---|---|---|
| `global_pool_only` | Remove signed cuts | Tests region imbalance value | Cuts should improve if separation matters. |
| `random_masks_frozen` | Freeze random low-rank masks | Tests learned regions | Learned masks should help. |
| `full_masks` | Use unconstrained 8x8 masks | Tests low-rank sufficiency | If full much better, low-rank too weak. |
| `no_king_anchor` | Remove king-relative mask variants | Tests king-side cuts | Should hurt king-attack motifs. |
| `material_bucket_eval` | Evaluate inside material buckets | Tests material shortcut | Gains should survive. |

### Diagnostics

- Learned mask visualizations.
- Cut magnitude by label.
- Low-rank versus full-mask gap.
- Which cuts activate on false positives.
- King-anchored cut contribution.

### Failure Modes

- Low-rank masks may be too simple.
- It may rediscover material or side imbalance.
- Full masks may overfit square priors.

### Implementation Notes

Keep mask count small and include random-mask controls from the first run.

## Recommended Promotion Order

1. `Replicator Payoff Piece Dynamics`
2. `Differentiable Bitboard Boolean Network`
3. `Orthogonal Board Moment Network`
4. `Low-Rank Signed Cut Query Network`
5. `Legal-Constraint Projection Residual Network`
6. `Zobrist Kernel Feature Network`

## Minimal Benchmark Plan

Use:

```text
dataset: crtk_sample_3class
input: simple_18
target: binary puzzle-like
seeds: 3
metrics: accuracy, AUROC, PR-AUC, Brier, ECE
diagnostics: fine-label confusion, material-bucket metrics where relevant
```

Do not use source or provenance fields as model inputs.

## Duplicate Guardrails For Future Ideation

| Candidate | Do not repeat as |
|---|---|
| Replicator Payoff Piece Dynamics | Another piece payoff model that only changes replicator step count or payoff head count. |
| Differentiable Bitboard Boolean | Another soft Boolean predicate model that only changes operator list or predicate count. |
| Orthogonal Board Moment | Another polynomial moment model with only different degree cutoff. |
| Legal-Constraint Projection Residual | Another static legal-constraint projection model with only different projection iterations. |
| Zobrist Kernel Feature | Another random hash feature classifier with only different random feature count. |
| Low-Rank Signed Cut Query | Another region-cut query model with only different mask rank or mask count. |

## Best Full-Packet Candidate

`Replicator Payoff Piece Dynamics` is the strongest next full packet because it gives a genuinely chess-like abstraction: pieces exert pressure on one another in a small strategic game. It is also easy to falsify:

```text
replicator dynamics must beat token pooling
asymmetric payoffs must beat symmetric payoffs
real relative geometry must beat shuffled geometry
```

If those fail, the family should be archived as a negative result instead of scaled.


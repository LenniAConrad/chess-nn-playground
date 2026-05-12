# Codex Research Synthesis: Top-3 Architecture Derivations

## File Metadata

- Filename: `chess_nn_research_2026-04-24_2143_friday_shanghai_top3_derivations.md`
- Generated at: 2026-04-24 21:43
- Weekday: Friday
- Timezone: Asia/Shanghai
- Intended next consumer: Codex
- Status: synthesis and derivation packet, not implemented

## Purpose

This packet selects the current top three most promising architecture families and derives concrete follow-up variants from each. The goal is not to invent more unrelated ideas. The goal is to make the best ideas deeper, sharper, and more implementation-ready.

## Updated Top Three

The earlier synthesis packet ranked:

```text
Piece-Token CNN Hybrid
Set-Query Attention Bottleneck
Fixed-Point Residual Defect Network
```

That ranking was correct for the state of the archive at 21:13. After the later packets, the updated top three are:

| Rank | Parent idea | Source packet | Why it is top-three now |
|---|---|---|---|
| 1 | Piece-Token CNN Hybrid | `chess_nn_research_2026-04-24_2109_friday_shanghai_piece_token_cnn_hybrid.md` | Best practical baseline and required infrastructure for almost every token/set/role idea. |
| 2 | Schur-Ray Line Algebra Network | `chess_nn_research_2026-04-24_2127_friday_shanghai_schur_ray_line_algebra.md` | Strongest efficient chess-specific linear algebra operator; compact, falsifiable, and line-aware. |
| 3 | Relational Query Algebra Network | `chess_nn_research_2026-04-24_2139_friday_shanghai_relational_query_algebra.md` | Most distinct high-ceiling abstraction: chess as typed current-board fact tables and learned joins. |

Close runners-up:

- `Set-Query Attention Bottleneck`: still useful, but less chess-specific than Schur-Ray and less distinct than relational queries.
- `Replicator Payoff Piece Dynamics`: interesting, but higher risk of becoming attention-like pooling.
- `Ray State-Space Scan Network`: practical, but closer to existing line/ray families.

## Shared Data Contract

All derived candidates target:

- output `0`: non-puzzle
- output `1`: puzzle-like

Fine labels `0`, `1`, and `2` remain diagnostics only.

Use:

```text
input: simple_18
dataset: crtk_sample_3class
trainer: shared experimental training pipeline
```

Forbidden model inputs:

- Stockfish scores, PVs, node counts, mate scores, verification metadata, source labels, proposed labels, dataset provenance, unresolved candidate status, or anything derived from them.
- Engine search, forced-line search, legal mate/stalemate oracles, tablebase outcomes, or future game outcomes.

Allowed model inputs:

- Current board tensor.
- Side-to-move, castling, and en-passant planes already present in the input.
- Deterministic coordinates.
- Deterministic occupied-piece extraction.
- Current-board-only line, square, and relation features.

## Derivation Set Overview

| Parent | Derivation | Main change | Priority |
|---|---|---|---|
| Piece-Token CNN Hybrid | Role-Contrast Piece-Token Hybrid | Adds explicit own/opponent role contrast vectors without attention | High |
| Piece-Token CNN Hybrid | King-Centric Token Fusion Hybrid | Re-centers piece tokens around both kings and fuses king-zone crop summaries | High |
| Piece-Token CNN Hybrid | Material-Conditional Token Adapter | Adds tiny material-conditioned low-rank adapters with strict shortcut controls | Medium |
| Schur-Ray Line Algebra | Segment-Schur Line Algebra | Replaces full lines with blocker-defined line segments before the Schur solve | High |
| Schur-Ray Line Algebra | Dual-Field Schur Tension Network | Solves coupled own/opponent line fields and reads their residual disagreement | High |
| Schur-Ray Line Algebra | Schur-Ray Residual Cascade | Uses successive low-rank line solves as a residual correction cascade | Medium |
| Relational Query Algebra | Semijoin-First Relational Network | Makes piece-square-piece semijoins the central bottleneck | High |
| Relational Query Algebra | Query Dictionary Distillation Network | Learns a small dictionary of reusable relational query templates | Medium |
| Relational Query Algebra | Relational Residual Query Network | Classifies from residual evidence unexplained by independent table pooling | High |

Best immediate derivation to implement:

```text
Role-Contrast Piece-Token Hybrid
```

Best high-upside derivation:

```text
Segment-Schur Line Algebra
```

Most distinct derivation:

```text
Semijoin-First Relational Network
```

## Parent 1: Piece-Token CNN Hybrid

### Why This Parent Is Still Rank 1

The hybrid is not the most novel idea, but it is the most important. It gives the research loop a stronger practical baseline and creates reusable pieces:

- occupied-piece extraction
- token masks
- side-relative token features
- token/CNN fusion
- branch-removal ablations

If this model fails, many higher-level token ideas should be downgraded. If it succeeds, the derived variants below become natural next steps.

## Derivation 1A: Role-Contrast Piece-Token Hybrid

### Thesis

Instead of only pooling all occupied tokens together, explicitly contrast own and opponent piece sets under learned role projections. Puzzle-like positions should produce sharper own/opponent role imbalances than quiet non-puzzles.

### Derived From Parent

Parent:

```text
CNN branch + occupied-piece token branch + late fusion
```

Derivation:

```text
CNN branch
+ own-token pool
+ opponent-token pool
+ learned role projections
+ own-opponent contrast features
+ late fusion
```

### Core Equations

For token embeddings:

```text
t_i in R^D
```

and role projection matrices:

```text
A_r in R^{D x d}
```

compute:

```text
own_r = pool_i own_mask_i * A_r^T t_i
opp_r = pool_i opp_mask_i * A_r^T t_i
contrast_r = [
  own_r,
  opp_r,
  own_r - opp_r,
  own_r * opp_r,
  abs(own_r - opp_r)
]
```

Fuse:

```text
fused = MLP([cnn_pool, token_pool, contrast_1, ..., contrast_R])
```

### Why This Is Promising

It keeps the practical strength of the parent but adds a chess-specific asymmetry. Tactical positions often depend on imbalance:

- attackers versus defenders
- king-side pressure versus shelter
- overloaded defender versus multiple obligations
- active pieces versus passive pieces

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `no_role_contrast` | Revert to parent hybrid | Tests derivation value. |
| `same_side_random_split` | Randomly split tokens into two groups with same sizes | Tests own/opponent semantics. |
| `difference_only` | Remove product and absolute contrast terms | Tests nonlinear contrast. |
| `material_contrast_only` | Use only piece counts in contrast branch | Tests geometry/token content. |

### Implementation Notes

Start with:

```text
roles = 6
role_dim = 32
token_dim = 64
cnn_width = 48
```

Do not add attention in the first version. Keep this as a simple, strong baseline upgrade.

## Derivation 1B: King-Centric Token Fusion Hybrid

### Thesis

The parent token branch treats token coordinates globally. A king-centric derived model expresses every piece twice: relative to the own king and relative to the opponent king. This should improve motifs where piece relevance depends on king geometry.

### Derived From Parent

Parent token features:

```text
piece type, color, rank, file, side-relative coordinates
```

Derived token features:

```text
piece type, color, rank, file,
delta_to_own_king,
delta_to_opp_king,
same_line_as_own_king,
same_line_as_opp_king,
chebyshev_to_kings,
king-zone membership
```

### Architecture

1. Extract occupied tokens.
2. Locate both kings from current board planes.
3. Add king-relative features to every token.
4. Extract small `5x5` king crops through the CNN branch.
5. Fuse:

```text
global_cnn
token_pool
own_king_relative_pool
opp_king_relative_pool
king_crop_summary
```

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `no_king_relative_features` | Revert token coordinates to parent | Tests king-centric token geometry. |
| `random_king_square` | Use random occupied square as king anchor | Tests actual king semantics. |
| `own_king_only` | Remove opponent-king-relative view | Tests asymmetric target geometry. |
| `no_king_crop` | Keep king-relative tokens but remove crop branch | Tests crop value. |

### Implementation Notes

This is probably the best parent-derived branch for immediate puzzle recall, because many puzzles are king-tactical. The main risk is overfitting to checking positions.

## Derivation 1C: Material-Conditional Token Adapter

### Thesis

The right token processing may differ by material phase. Instead of adding a large expert router, use tiny material-conditioned low-rank adapters inside the token MLP.

### Core Adapter

For token hidden state `h`:

```text
h' = h + B(m) A(m) h
```

where `m` is a safe material summary and:

```text
A(m): D -> r
B(m): r -> D
r in {2, 4}
```

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `rank0_adapter` | Adapter disabled | Tests adaptation value. |
| `material_head_only` | Feed material to final MLP only | Tests whether adapter does more than shortcut. |
| `bucket_eval` | Evaluate inside material buckets | Checks material shortcut risk. |
| `random_material_summary` | Shuffle material summaries within coarse buckets | Tests summary semantics. |

### Implementation Notes

This is lower priority than 1A/1B because shortcut risk is high. It becomes valuable after the parent hybrid is stable.

## Parent 2: Schur-Ray Line Algebra

### Why This Parent Is Rank 2

Schur-Ray has the best combination of:

- chess-specific structure
- efficient computation
- linear algebra clarity
- strong falsifiers
- low dependence on hand-coded tactics

The parent uses fixed rank/file/diagonal line incidence and a Woodbury-compressed solve. The derivations below refine its chess abstraction.

## Derivation 2A: Segment-Schur Line Algebra

### Thesis

Full ranks/files/diagonals ignore blockers. A stronger version builds current-board line segments between occupied squares or board edges, then runs the Schur solve over compressed segment modes instead of full-line modes.

### Derived From Parent

Parent basis:

```text
46 full lines = ranks + files + diagonals + anti-diagonals
```

Derived basis:

```text
current-board line segments split by occupied blockers
```

Segment examples:

```text
edge -> blocker
blocker -> blocker
blocker -> edge
king -> nearest blocker
slider -> blocker -> behind segment
```

No legal move generation is needed. This uses only current occupancy and board lines.

### Segment Incidence

Let:

```text
G in {0,1}^{64 x S}
```

where `S` is the number of deterministic line segments, capped and padded:

```text
Smax = 128
```

Then:

```text
U = diag(g) G M
```

and the same Woodbury solve applies:

```text
z = b - D^-1 U (C^-1 + U^T D^-1 U)^-1 U^T b
```

### Why This Derivation Is Better

It keeps the parent linear algebra but adds blocker sensitivity directly to the basis. The parent had to learn blocker effects through `g`; this version gives the solve a better geometry.

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `full_line_parent` | Use original 46-line basis | Tests segment value. |
| `random_segments` | Preserve segment lengths but randomize square memberships | Tests segment geometry. |
| `no_blocker_split` | Build segments independent of occupancy | Tests current-board blocker role. |
| `segment_count_only` | Use segment length/count summaries only | Tests Schur solve value. |

### Implementation Notes

This is the best high-upside derivation in the whole packet. It keeps the parent mechanism but makes the chess geometry more exact.

## Derivation 2B: Dual-Field Schur Tension Network

### Thesis

Puzzle-likeness may appear as disagreement between two line-equilibrium fields: one representing side-to-move force and one representing opponent resistance. Solve both fields with shared line modes, then classify from their residual tension.

### Core System

Build two source fields:

```text
b_own
b_opp
```

and either solve independently:

```text
z_own = solve(D_own + U C_own U^T, D_own b_own)
z_opp = solve(D_opp + U C_opp U^T, D_opp b_opp)
```

or solve a coupled block system:

```text
[A_own   -K   ] [z_own] = [D_own b_own]
[-K^T    A_opp] [z_opp]   [D_opp b_opp]
```

where `K = U C_cross U^T`.

Start with independent solves; block coupling is a later variant.

### Readout

```text
tension = z_own - z_opp
alignment = z_own * z_opp
abs_tension = abs(z_own - z_opp)
line_tension = U^T tension
```

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `single_field_parent` | Use one Schur field | Tests dual-field value. |
| `shared_source` | Set `b_own = b_opp` | Tests side-specific sources. |
| `no_tension_readout` | Use only individual field summaries | Tests disagreement features. |
| `random_color_swap` | Break own/opponent assignment | Tests side semantics. |

### Implementation Notes

This is safer than the coupled block solve. Implement independent solves first, add coupling only if the tension readout is useful.

## Derivation 2C: Schur-Ray Residual Cascade

### Thesis

One Schur solve may capture only first-order line correction. A short cascade can apply line-equilibrium corrections iteratively and classify from the residual decay curve.

### Recurrence

```text
b_0 = source field
for t in 1..T:
    z_t = SchurSolve(b_{t-1})
    r_t = b_{t-1} - z_t
    b_t = b_{t-1} + alpha_t * MLP([z_t, r_t])
```

Readout:

```text
||r_t||
cos(r_t, r_{t-1})
energy_t
line_correction_t
```

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `single_solve` | Parent model | Tests cascade value. |
| `untied_solve_params` | Different solve params per step | Tests shared dynamics. |
| `residual_norm_only` | Remove residual direction features | Tests trajectory geometry. |
| `final_only` | Classify only from final field | Tests residual path. |

### Implementation Notes

This is medium priority. It could become expensive and may duplicate fixed-point residual ideas unless the Schur diagnostics remain central.

## Parent 3: Relational Query Algebra

### Why This Parent Is Rank 3

Relational Query Algebra is the most distinct family in the archive. It treats chess as:

```text
typed current-board fact tables + learned joins
```

That is different from images, graphs, lines, tensors, and move sets. The derivations below make the parent more focused and easier to falsify.

## Derivation 3A: Semijoin-First Relational Network

### Thesis

The parent includes piece-square joins, piece-piece joins, and semijoins. The most chess-like part is the semijoin: a piece, a mediator square, and another piece linked by a relation. Make that the central bottleneck.

### Derived Query Form

Parent:

```text
Piece p
JOIN Square s
JOIN Piece q
```

Derived:

```text
SELECT aggregate(value)
FROM Piece p1, Square m, Piece p2
WHERE R_left(square(p1), m)
  AND R_right(m, square(p2))
  AND learned_predicates(p1, m, p2)
```

This captures mediator evidence:

- blocker-like square
- empty gap square
- king-zone square
- between-line square
- fork center square

without hand-coded tactics.

### Tensor Sketch

Avoid full dense `(Pmax, 64, Pmax)` when possible:

```text
left = relation(square_i, m)
right = relation(m, square_j)
mid_value = square_gate(m) * square_value(m)
score_ij = sum_m left_i_m * mid_value_m * right_m_j
```

Then aggregate over piece pairs.

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `parent_no_semijoin` | Use parent without semijoin | Tests central derivation. |
| `piece_pair_direct_only` | Direct piece-piece relations only | Tests mediator square value. |
| `random_mediator_relation` | Shuffle mediator relation tables | Tests mediator geometry. |
| `mid_square_marginals_only` | Use square gates without joining to pieces | Tests relational binding. |

### Implementation Notes

This is the most distinct derivation. It should be implemented after a smoke-test version of the parent relation extractor exists.

## Derivation 3B: Query Dictionary Distillation Network

### Thesis

The parent uses many independent learned query blocks. A dictionary version learns a small set of reusable query templates, then composes examples from sparse mixtures of those templates. This gives better interpretability and reduces overfitting.

### Query Dictionary

Learn:

```text
K query templates
```

where each template contains:

```text
piece predicate parameters
square predicate parameters
relation mixture
aggregation type
```

For each board, emit sparse mixture weights:

```text
w_k(x) = sparsemax(board_summary)
```

and compute:

```text
summary = sum_k w_k(x) * QueryTemplate_k(board_tables)
```

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `independent_queries_parent` | Parent model | Tests dictionary sharing. |
| `dense_template_mix` | Replace sparsemax with softmax | Tests sparse template use. |
| `random_templates` | Freeze random query templates | Tests learned templates. |
| `template_count_sweep` | Vary K | Tests overfit/underfit. |

### Implementation Notes

This is a second-generation relational idea. It is best after parent query logs reveal that query blocks specialize.

## Derivation 3C: Relational Residual Query Network

### Thesis

Independent table pooling captures easy evidence: material, piece counts, king locations, and broad square occupancy. The useful relational signal is the residual: what learned joins explain beyond independent piece and square summaries.

### Residual Construction

Compute independent summaries:

```text
u_piece = pool(Piece)
u_square = pool(Square)
u_base = MLP([u_piece, u_square])
```

Compute join summaries:

```text
u_join = QueryExecutor(Piece, Square, Relations)
```

Predict the join summary from independent summaries:

```text
u_hat = MLP(u_base)
residual = u_join - u_hat
```

Classify from:

```text
[u_base, residual, abs(residual), residual_norms]
```

### Why This Is Useful

It forces the model to expose whether joins carry information beyond count-like table summaries. This is a cleaner scientific question than simply "does the relational model win?"

### Central Ablations

| Ablation | What it removes | Why it matters |
|---|---|---|
| `join_only` | Remove baseline/residual split | Tests residual framing. |
| `base_only` | Independent table pooling only | Tests relational gain. |
| `stopgrad_base_predictor` | Stop gradient through base predictor | Tests training stability. |
| `residual_norm_only` | Use scalar residual norms only | Tests residual direction. |

### Implementation Notes

This is a high-priority derivation because it makes the parent more falsifiable. It also provides a good report: if residual features do not matter, joins may not be worth the complexity.

## Cross-Derivations

These combine the top-three parents without turning into an untestable mega-model.

## Cross-Derivation A: Piece-Token Relational Adapter

### Idea

Use Piece-Token CNN Hybrid as the backbone, then add a small relational query adapter only over occupied tokens.

```text
Piece-Token CNN Hybrid
+ 4 relational piece-piece query blocks
+ relation_shuffle ablation
```

### Why It Matters

This is the safest path to test relational joins without building the full relational model first.

### Stop Rule

If this adapter does not beat parent hybrid and relation-shuffle does not hurt, delay full relational implementation.

## Cross-Derivation B: Schur Features Into Piece-Token Fusion

### Idea

Run a tiny Schur-Ray branch and inject only diagnostic features into the parent hybrid:

```text
schur_logdet
line_energy
mean_abs_correction
king_zone_line_energy
```

not the full Schur field.

### Why It Matters

This tests whether the Schur signal is useful as a cheap feature branch before implementing a larger Schur model.

### Stop Rule

If line-energy diagnostics do not help over the parent hybrid and random-line incidence matches, do not scale Schur-Ray yet.

## Cross-Derivation C: Relationally Gated Segment-Schur

### Idea

Use relational query gates to choose which line segments enter the Segment-Schur basis:

```text
segment_gate = RelationalQuery(piece, square, relation)
U = diag(segment_gate) G M
```

### Why It Matters

This is high-upside but should be delayed. It combines two complex parents and needs both independent mechanisms to pass their central ablations first.

## Recommended Implementation Queue

| Stage | Implement | Why this order |
|---|---|---|
| 1 | Parent `Piece-Token CNN Hybrid` | Establish strong practical baseline and token extractor. |
| 2 | Derivation `Role-Contrast Piece-Token Hybrid` | Cheap, likely useful, directly tests own/opponent token contrast. |
| 3 | Cross `Piece-Token Relational Adapter` | Tests relational value with minimal infrastructure. |
| 4 | Parent `Schur-Ray Line Algebra` or feature-only Schur branch | Tests line-equilibrium signal against the strong hybrid. |
| 5 | Derivation `Segment-Schur Line Algebra` | Best high-upside line-algebra variant if parent works. |
| 6 | Parent `Relational Query Algebra` | Full distinct model only after adapter evidence. |
| 7 | Derivation `Semijoin-First Relational Network` | Deepens the relational family if parent/adapter passes. |

## Decision Matrix

| Question | If yes | If no |
|---|---|---|
| Does Piece-Token CNN Hybrid beat CNN-only? | Token infrastructure is worth building on. | Downgrade most token-heavy ideas. |
| Does Role-Contrast beat parent hybrid? | Own/opponent contrast is a strong default branch. | Keep parent simple. |
| Does relational adapter beat parent hybrid? | Full relational model is justified. | Delay Relational Query Algebra. |
| Does Schur feature branch help parent hybrid? | Implement full Schur-Ray. | Keep line algebra archived for later. |
| Does Segment-Schur beat full-line Schur? | Segment basis becomes the main Schur variant. | Keep simpler full-line basis. |
| Does semijoin-first beat direct joins? | Relational family should focus on mediator-square evidence. | Keep direct joins or abandon complex semijoins. |

## New Anti-Duplicate Rules

After this packet, do not produce:

- another Piece-Token CNN Hybrid derivation unless it changes the token evidence path or falsifier beyond width/depth.
- another Schur-Ray derivation unless it changes the basis, field coupling, or residual trajectory in a falsifiable way.
- another Relational Query derivation unless it changes the query class or residual question, not just query count.
- a combined mega-model before at least one parent and one child pass central ablations.

## Best Three Derived Ideas

If only three derivations are worth carrying forward, use:

1. `Role-Contrast Piece-Token Hybrid`
2. `Segment-Schur Line Algebra`
3. `Semijoin-First Relational Network`

They preserve the strengths of the top three parents while asking sharper questions:

```text
Do own/opponent token contrasts matter?
Do blocker-defined line segments beat full lines?
Do mediator-square semijoins beat direct pair relations?
```


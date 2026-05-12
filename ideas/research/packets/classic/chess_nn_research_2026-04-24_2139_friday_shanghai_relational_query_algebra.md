# Codex Research Packet: Relational Query Algebra Network

## File Metadata

- Filename: `chess_nn_research_2026-04-24_2139_friday_shanghai_relational_query_algebra.md`
- Generated at: 2026-04-24 21:39
- Weekday: Friday
- Timezone: Asia/Shanghai
- Intended next consumer: Codex
- Status: full architecture packet, not implemented

## One-Sentence Thesis

Treat a chess position as a tiny typed relational database of current-board facts, then classify puzzle-likeness by executing learned differentiable relational-algebra queries with joins, projections, semijoins, and aggregations, instead of using images, attention, move generation, attack graphs, or line scanners.

## Why This Is Deliberately Different

Most archived ideas represent the board as one of these objects:

- image tensor
- occupied-piece set
- line/ray strings
- graph or sheaf over attack relations
- one-ply move-delta set
- transport problem
- topology/field/PDE object
- matrix factorization or spectral summary
- Boolean/tropical clause circuit
- bitboard shift algebra
- learned game/payoff dynamics

This packet uses a different abstraction:

```text
chess position = small database instance
model = learned query executor over typed tables
```

The model asks questions like:

```text
Which learned piece-pair facts survive a same-line join?
Which learned piece-square facts project into a king-zone aggregate?
Which three-table joins create high residual evidence?
Which relations matter after relation-table shuffling controls?
```

It does not hand-code tactical rules. It learns soft query templates and executes them over current-board-only tables.

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
- Fixed current-board-independent relation tables over squares, such as same rank, same file, same diagonal, distance bins, square color, and king-zone membership.
- Current-board piece facts extracted from the input tensor.

## Core Database Schema

The first version should build a tiny in-memory relational instance for each board.

### Piece Table

```text
Piece(pid, square, type, color, side_relative_type, is_king, is_slider, is_pawn)
```

Padded to:

```text
Pmax = 32
```

Tensor form:

```text
piece_features: (B, Pmax, Dp)
piece_mask:     (B, Pmax)
```

### Square Table

```text
Square(sid, rank, file, side_relative_rank, square_color, edge_bin, center_bin)
```

Fixed size:

```text
N = 64
```

Tensor form:

```text
square_features: (B, 64, Ds)
```

Some square features are fixed; others are projected from current board planes.

### Relation Tables

Relation tables are fixed or current-board-safe:

```text
SameRank(s1, s2)
SameFile(s1, s2)
SameDiag(s1, s2)
SameAntiDiag(s1, s2)
KnightOffset(s1, s2)
KingOffset(s1, s2)
DistanceBin_k(s1, s2)
SameSquareColor(s1, s2)
BetweenLine(s_mid, s_a, s_b)
KingZoneOwn(s)
KingZoneOpp(s)
Occupied(s)
Empty(s)
```

Important guardrail:

- `SameRank`, `SameFile`, `SameDiag`, distance bins, and offset relations are board-geometry tables.
- `Occupied` and `Empty` are current-board facts.
- Do not include check, mate, legal move, attacked-square, engine pressure, or best-move information.

### Fact Tables As Soft Weights

Each fact table is represented as a soft tensor:

```text
T_piece:       (B, Pmax, C)
T_square:      (B, 64, C)
R_square2:     (R2, 64, 64)
R_square3:     optional sparse triples for between-line
```

The relation matrices can be sparse or dense. On 64 squares, dense relation matrices are acceptable for a first implementation.

## Main Operator: Learned Differentiable Joins

The model learns a set of query blocks. Each block has:

```text
left selector
right selector
relation selector
join aggregator
projection head
```

### Piece-Square Join

Example tensor expression:

```text
J_q[p, s] =
  piece_gate_q(p)
  * square_gate_q(s)
  * relation_q(piece_square(p), s)
```

Where:

```text
relation_q = weighted mixture of fixed square-square relations
```

Then project:

```text
A_q[p] = aggregate_s J_q[p, s] * square_value_q(s)
```

### Piece-Piece Join

```text
K_q[p_i, p_j] =
  piece_gate_left_q(p_i)
  * piece_gate_right_q(p_j)
  * relation_q(square_i, square_j)
```

Then aggregate:

```text
pair_summary_q =
  mean/max/topk over valid i,j of K_q[p_i,p_j] * pair_value_q(p_i,p_j)
```

### Piece-Square-Piece Semijoin

This is the first genuinely relational block:

```text
Piece p_i
JOIN Square s_mid
JOIN Piece p_j
WHERE BetweenLine(s_mid, square_i, square_j)
AND OccupiedOrEmptyPredicate(s_mid)
```

Soft tensor form:

```text
T_q[i, m, j] =
  left_gate(i)
  * mid_gate(m)
  * right_gate(j)
  * BetweenLine[m, square_i, square_j]
```

Then reduce:

```text
semijoin_summary_q =
  aggregate_{i,m,j} T_q[i,m,j] * value_q(i,m,j)
```

This can detect learned patterns like "some square lies between two selected piece roles" without hard-coding pins or x-rays.

## Why It Is Not Just Logic

This is relational algebra, not a symbolic rule base.

- Queries are soft and learned.
- Relation tables are simple current-board-safe geometry.
- There are no hand-written tactical clauses.
- There is no proof search.
- The output is a supervised neural classifier.

The central claim is not:

```text
we can write chess tactics as rules
```

The central claim is:

```text
learned join structure over typed chess facts is a useful inductive bias
```

## Why It Is Not A Duplicate

| Existing family | Why this differs |
|---|---|
| Formal Concept Closure | FCA computes Galois closure over object-attribute incidence. This packet executes learned typed joins and semijoins over multiple relation tables. |
| Tropical Constraint Circuit | Tropical clauses use min-plus literal costs. This packet uses database-style joins/projections/aggregations, not clause satisfaction. |
| Differentiable Bitboard Boolean | Boolean bitboards combine predicate fields pointwise. This packet performs relational joins across typed tables. |
| Ray-Language Automaton | Ray automata score ordered strings. This packet joins tables and does not run automata over lines. |
| Attack graph/sheaf/Hodge | No dynamic attack edges, cochains, Laplacians, restrictions, or graph message passing. |
| Move-delta packets | No move enumeration or counterfactual board states. |
| Attention | Relation tables constrain interactions before aggregation; there is no learned all-pairs query-key softmax. |
| Sparse Witness | Does not select a fixed top-k piece subset as the bottleneck; it selects soft query evidence over relations. |
| Bitboard Shift-Algebra | No shift-polynomial operator bank. |

## Architecture Sketch

### Step 1: Extract Tables

From `simple_18`, build:

```text
piece_features:   (B, Pmax, Dp)
piece_square_idx: (B, Pmax)
piece_mask:       (B, Pmax)
square_features:  (B, 64, Ds)
occupied:         (B, 64)
empty:            (B, 64)
```

Use pure tensor extraction where possible. A simple padded extractor is enough for the research packet.

### Step 2: Fixed Relation Bank

Precompute:

```text
R2: (R, 64, 64)
```

where `R` includes:

```text
same_rank
same_file
same_diag
same_anti_diag
same_square_color
opposite_square_color
knight_offset
king_offset
manhattan_distance_1
manhattan_distance_2
chebyshev_distance_1
chebyshev_distance_2
same_side_relative_half
same_file_adjacent_rank
```

Optional:

```text
R3_between: sparse list of (mid, a, b) triples
```

For the first version, the ternary relation can be implemented by gathering line-between masks for piece-pair squares.

### Step 3: Query Blocks

Use `Q=16` query blocks.

Each query block emits:

```text
piece gates
square gates
relation mixture weights
value projections
aggregation temperature
```

Example:

```text
rel_q = softmax(w_rel_q) over R relation tables
R_q = sum_r rel_q[r] * R2[r]
```

### Step 4: Join Execution

Execute three families:

```text
piece_square_join
piece_piece_join
piece_square_piece_semijoin
```

For each query, collect:

```text
mean evidence
max evidence
topk mean evidence
soft count
entropy of supporting facts
relation mixture entropy
```

### Step 5: Query Evidence Readout

Concatenate:

```text
query_summaries
relation_usage
support_entropy
small CNN summary
material summary for diagnostics only or controlled fusion
```

Then classify:

```text
MLP -> logits (B, 2)
```

## Tensor Contract

```text
input:                 (B, 18, 8, 8)
piece_features:        (B, Pmax, Dp)
piece_square_idx:      (B, Pmax)
piece_mask:            (B, Pmax)
square_features:       (B, 64, Ds)
relation_bank:         (R, 64, 64)
query_piece_gates:     (B, Q, Pmax)
query_square_gates:    (B, Q, 64)
query_relation_mix:    (Q, R)
piece_square_evidence: (B, Q, Pmax, 64)
piece_piece_evidence:  (B, Q, Pmax, Pmax)
semijoin_evidence:     summarized without storing full dense if needed
query_summary:         (B, Q, S)
logits:                (B, 2)
```

## Efficient Execution Notes

Board size is tiny, so clarity matters more than premature optimization.

Initial dense version:

```text
R_q:       (Q, 64, 64)
PS join:   O(B * Q * Pmax * 64)
PP join:   O(B * Q * Pmax * Pmax)
```

With:

```text
Pmax = 32
Q = 16
R = 16
```

this is small.

Avoid materializing:

```text
(B, Q, Pmax, 64, Pmax)
```

for ternary joins. Use pair summaries plus gathered between-line masks:

```text
for each piece pair (i,j), gather between_mask[square_i, square_j, :]
aggregate over mid squares
```

## Query Block Details

### Gates

Piece gates:

```text
g_piece_q = sigmoid(MLP_q(piece_features))
```

Square gates:

```text
g_square_q = sigmoid(MLP_q(square_features))
```

Value functions:

```text
v_piece_q = Linear(piece_features)
v_square_q = Linear(square_features)
v_pair_q = MLP([piece_i, piece_j, relative_geometry])
```

### Aggregators

Use several aggregators per query:

```text
masked_mean
masked_max
soft_topk_mean
logsumexp_temperature
support_entropy
```

Do not use only max; a pure max can become brittle and hard to calibrate.

### Relation Mixture

Start with global learned relation mixtures per query:

```text
relation_mix_q = softmax(parameter_q)
```

Later variant:

```text
relation_mix_q(x) = softmax(MLP(board_summary))
```

Keep the first version static so relation usage is easier to interpret.

## Mathematical View

A relational query block computes a soft version of:

```text
SELECT AGG(value)
FROM Piece p
JOIN Square s
ON R(p.square, s)
WHERE learned_piece_predicate(p)
  AND learned_square_predicate(s)
```

or:

```text
SELECT AGG(value)
FROM Piece p1
JOIN Piece p2
ON R(p1.square, p2.square)
WHERE learned_left_predicate(p1)
  AND learned_right_predicate(p2)
```

The architecture is invariant to arbitrary ordering of padded piece rows because all piece-table reductions are masked symmetric aggregations.

It is not invariant to square relabeling, because square relations have chess meaning. The central ablation destroys this meaning with relation-table shuffling.

## Central Ablations

| Ablation | What it removes | Why it matters | Expected readout |
|---|---|---|---|
| `no_joins` | Replace all joins with independent piece and square pooling | Tests relational algebra value | Should drop if joins matter. |
| `relation_shuffle` | Replace relation tables with fixed random square-square relations preserving density | Tests chess relation semantics | Should degrade if geometry relations matter. |
| `piece_pair_only` | Remove piece-square and semijoin blocks | Tests need for typed table variety | Full model should improve if square context matters. |
| `no_semijoin` | Remove piece-square-piece semijoins | Tests true multi-table joins | Should drop if between/mediator evidence matters. |
| `static_relation_mix_only` | Remove board-conditioned gates but keep relation mixes | Tests learned predicates | Should degrade if query predicates matter. |
| `mlp_same_params` | Replace query executor with parameter-matched MLP/CNN hybrid | Tests structured executor versus capacity | Query model should beat matched generic model. |
| `fact_table_permutation` | Permute square ids in fact tables but not relation tables | Tests alignment of facts and relations | Should degrade sharply. |

## Required Diagnostics

Shared:

- binary accuracy
- AUROC
- PR-AUC
- Brier
- ECE
- fine-label `3 x 2` diagnostic matrix

Architecture-specific:

- relation mixture weights per query
- relation mixture entropy
- query support entropy
- top contributing query blocks
- top supporting piece/square facts for examples
- relation-shuffle ablation gap
- no-join ablation gap
- semijoin contribution
- piece-row permutation invariance smoke test

## Interpretability Output

For each validation example, optionally log:

```text
query_id
relation_mix top relations
top piece rows
top square rows
top piece-pair rows
semijoin support count
query evidence scalar
```

This gives concrete database-like explanations:

```text
query 7 used same_file + king_zone relations and focused on two heavy pieces plus one empty square
```

The explanation should be treated as diagnostic, not as a proof of a tactic.

## Expected Positive Result

The idea is promising if:

```text
full model > no_joins
full model > relation_shuffle
full model > mlp_same_params
no_semijoin hurts at least some fine-label-2 recall
query support is sparse enough to inspect
```

The strongest result would be that relation joins improve calibration or near-puzzle recall without increasing false positives on fine-label `0`.

## Expected Negative Result

Treat the idea as falsified if:

- `no_joins` matches the full model.
- `relation_shuffle` matches the full model.
- `mlp_same_params` beats the query executor.
- Query evidence collapses to material counts.
- Relation mixture entropy stays high and no query specializes.
- Fine-label diagnostics show no improvement over ordinary CNN/token baselines.

## Failure Modes

- Query blocks may become expensive if semijoins are implemented naively.
- Learned predicates may collapse to material or piece-type gates.
- The architecture can drift toward hand-coded tactics if relation tables become too rich.
- Static relation mixtures may be too rigid; dynamic mixtures may overfit.
- Soft joins may be less expressive than attention while still being more complex than CNNs.

## Implementation Sketch

### New Files

```text
src/chess_nn_playground/models/trunk/relational_query_algebra.py
tests/test_relational_query_algebra.py
configs/model/relational_query_algebra.yaml
```

### Helper Components

```text
PieceTableExtractor
SquareRelationBank
RelationalQueryBlock
MaskedJoinAggregators
RelationalQueryAlgebraNet
```

### Forward Pseudocode

```text
def forward(x):
    pieces = extract_piece_table(x)
    squares = build_square_table(x)
    rel_bank = self.relation_bank

    query_outputs = []
    for query in self.queries:
        gp = query.piece_gate(pieces.features)
        gs = query.square_gate(squares.features)
        rel = query.relation_mix(rel_bank)

        ps = execute_piece_square_join(gp, gs, rel, pieces, squares)
        pp = execute_piece_piece_join(gp, rel, pieces)
        sj = execute_semijoin(gp, gs, rel, pieces, squares)

        query_outputs.append(aggregate(ps, pp, sj))

    z_query = concat(query_outputs)
    z_cnn = self.cnn_summary(x)
    return self.classifier(concat(z_query, z_cnn))
```

### Minimal Config

```yaml
model:
  name: relational_query_algebra
  input_channels: 18
  piece_width: 64
  square_width: 48
  query_count: 16
  relation_count: 16
  use_piece_square_join: true
  use_piece_piece_join: true
  use_semijoin: true
  use_cnn_summary: true
  aggregator: [mean, max, topk_mean, logsumexp, entropy]
training:
  loss: cross_entropy
  binary_target: true
diagnostics:
  fine_label_matrix: true
  log_relation_mixes: true
  log_query_support: true
ablations:
  - no_joins
  - relation_shuffle
  - piece_pair_only
  - no_semijoin
  - static_relation_mix_only
  - mlp_same_params
  - fact_table_permutation
```

## Unit Tests

Required tests:

- Piece table extractor returns at most 32 valid rows and correct masks.
- Piece-row permutation does not change logits within tolerance.
- Relation bank has expected shape and no wraparound relation bugs.
- Relation-shuffle ablation preserves relation densities.
- No-join ablation preserves output shape.
- Semijoin block does not materialize excessive dense tensors.
- Model forward returns finite `(B, 2)` logits.

## Anti-Shortcut Controls

### Material Probe

Train a probe from query summaries to material counts. If query summaries predict material nearly perfectly and gains disappear inside material buckets, the model is using material shortcuts.

### Relation Shuffle

For every relation table:

```text
preserve density
preserve diagonal/non-diagonal status where relevant
preserve symmetry where relevant
destroy chess square semantics
```

If relation-shuffle matches full model, abandon the relational thesis.

### Fact-Relation Misalignment

Permute square ids in fact tables while keeping relation tables fixed. This destroys alignment between pieces and chess geometry while preserving fact counts.

### Query Dropout

Randomly drop query blocks during training with small probability. This discourages one query from memorizing a shortcut.

## Efficiency Benchmark Plan

Compare:

```text
small CNN
Piece-Token CNN Hybrid
RelationalQueryAlgebraNet Q=8
RelationalQueryAlgebraNet Q=16
RelationalQueryAlgebraNet no_semijoin
MLP/CNN matched params
relation_shuffle
```

Report:

```text
AUROC
PR-AUC
ECE
fine-label matrix
parameter count
forward time per 1024 boards
peak memory if available
```

## Duplicate Guardrail

Do not repeat this idea as:

- another learned SQL/query/database board model with only more query blocks
- another differentiable logic model that drops the table/join semantics
- another FCA closure model with renamed tables
- another Boolean bitboard model with relation words added
- another attack graph model described as a database

Only revisit if:

- relation-shuffle fails clearly
- no-join ablation fails clearly
- query support diagnostics show meaningful typed joins
- implementation proves cheaper or more interpretable than square attention

## Best Immediate Experiment

The first implementation should be intentionally small:

```text
Q = 8
relation_count = 12
piece_square_join = true
piece_piece_join = true
semijoin = false for first smoke test
semijoin = true for second run
```

The central test is:

```text
Do learned typed joins over chess fact tables add information beyond independent piece/square pooling and generic CNN capacity?
```

If the answer is no, archive this as a clean negative result. If yes, it becomes one of the most distinct architecture families in the repo.


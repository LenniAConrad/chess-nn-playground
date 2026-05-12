# Mathematical Thesis

## Actual Task And Labels

Let `x` be a legal chess position represented only by current-board features. For the first benchmark, the binary label is:

```text
y = 0 for source class 0 random/non-puzzle and source class 1 verified near-puzzle
y = 1 for source class 2 verified puzzle
```

The model emits one logit `f(x)`. Source labels may be used for stratified reporting and ablations, but not as inference inputs.

Forbidden inputs: Stockfish scores, PVs, node counts, verification metadata, source file identity, source labels, candidate status, or future game outcomes.

## Baseline And Weakness

Closest baselines:

- simple CNN: learns local spatial filters but must discover long-range chess relations indirectly.
- LC0 BT4 tower: strong residual convolutional baseline, still generic spatial computation.
- Bitboard Shift-Algebra Network: uses fixed shifts and low-degree operator polynomials.
- Schur-Ray Line Algebra Network: emphasizes sliding-line global interactions through a structured solve.

Overlap: this idea also uses fixed chess-shaped operators over squares.

Difference: this idea is a general classification block that mixes many relation operators with learned low-rank coefficients at every layer, rather than committing to a polynomial bitboard algebra or a line-specific Schur solve.

## Definitions

Let `H_l in R^{64 x d}` be square embeddings at layer `l`. Define a set of fixed or occupancy-gated sparse operators:

```text
O_0: identity
O_1..O_4: rank/file/diagonal/anti-diagonal ray summaries
O_5: knight-reach adjacency
O_6: king-neighborhood adjacency
O_7..O_8: white/black pawn attack adjacency
O_9: same-color square parity relation
O_10: king-zone mask relation
```

Each operator maps square features to square features:

```text
M_k(H_l) = O_k H_l
```

A layer computes:

```text
U_l = concat_k M_k(H_l)
G_l = low_rank_gate(pool(H_l), side_to_move, material_phase)
H_{l+1} = H_l + phi(sum_k G_{l,k} M_k(H_l) W_{l,k})
```

The final classifier pools square, piece, king-zone, and global summaries.

## Assumptions

- Many chess classification labels depend on a small set of stable rule-shaped relations.
- A learned mixture of relation operators can express useful long-range structure with less data than a generic CNN.
- The benchmark labels are not dominated by source artifacts.

## Claim

Hypothesis: for chess position classification, a sparse operator-basis trunk should beat a size-matched simple CNN and challenge BT4 on PR AUC while reducing near-puzzle false positives, because it exposes nonlocal legal geometry directly.

## Mechanism

CNN filters are translation-shaped. Chess has translation-like local structure, but also rule-specific relations: knight jumps, pawn direction, king-zone adjacency, and line-of-sight. The operator basis makes these relations available as first-class channels. The low-rank gate lets the model choose which relations matter for the current material and side-to-move context.

## Proof Sketch

This does not prove better accuracy. What can be reasoned about:

- Any one-step message along the included operators is represented in one layer.
- Several common tactical relations that need many generic convolution layers can be represented in one or two operator-basis layers.
- Sparse operators give a bounded compute path and clean ablations by operator family.

## Not Proven

- That puzzle labels actually require these relations more than BT4 already captures them.
- That occupancy-gated operators will not overfit to superficial line patterns.
- That the learned gate will use relation families in an interpretable way.

## Counterexamples

- If labels depend mostly on local piece texture, a simple CNN may match it.
- If the data contains source artifacts, a generic model may exploit them equally well or better.
- If the correct concept needs deeper search-like rollouts, current-board operators may be insufficient.

## Falsification Test

Train size-matched `simple_cnn`, `lc0_bt4_classifier`, and `chess_operator_basis_classifier` on the same `puzzle_binary` split. Abandon or revise if:

```text
test PR AUC does not beat the size-matched CNN by at least 0.02
and near-puzzle -> puzzle FP is not lower than the CNN
```

Reject the relation-basis hypothesis if removing the chess-specific operator families has no measurable effect across two seeds.


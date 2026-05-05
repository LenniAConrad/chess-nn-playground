# Mathematical Thesis

## Actual Task And Labels

First benchmark:

```text
source class 0: known non-puzzle / random position -> y = 0
source class 1: verified near-puzzle / hard negative -> y = 0
source class 2: verified puzzle -> y = 1
```

The model emits one puzzle logit. Source classes are diagnostics only.

Allowed inputs:

- current-board tensor
- deterministic legal or pseudo-legal move descriptors generated from the board
- deterministic negative move descriptors for invalid-move rejection
- next-board tensors produced by applying legal moves for auxiliary training only

Forbidden inputs:

- Stockfish scores, PVs, node counts, mate scores
- engine best moves
- verification metadata
- source labels or source file identity
- future game outcomes

## Baseline And Weakness

Closest registered ideas:

- `i001_chess_operator_basis_classifier`: fixed relation operators.
- `i002_response_minimax_classifier`: one-ply response bottleneck.
- `i007_neural_proof_number_search`: bounded proof tree.
- `i009_tactical_equilibrium_network`: attacker/defender matrix game.

Overlap: this idea also uses legal move descriptors and chess rules.

Difference: this idea's central mechanism is self-supervised latent dynamics consistency. The final classifier is encouraged to use a representation that can predict legal consequences, not a handcrafted proof/search/equilibrium score.

## Definitions

Let `x` be a board and `m` a deterministic legal move sampled from `M(x)`. Let:

```text
x' = apply(x, m)
z = E(x)
z' = E(x')
t = T(z, move_descriptor(m))
```

where:

- `E` is the board encoder.
- `T` is a latent transition model.
- `t` is the predicted next latent.

Auxiliary objectives:

```text
L_next = || stopgrad(E(x')) - T(E(x), m) ||^2
L_legal = BCE(legal_head(E(x), move_descriptor), is_legal)
L_reconstruct = CE(piece_head(T(E(x),m)), board_planes(x'))
```

Main puzzle classifier:

```text
f(x) = puzzle_head([E(x), dynamics_summary(x)])
```

where `dynamics_summary` includes legal-move entropy, transition sensitivity, and latent consequence variance over a small sampled move set.

## Assumptions

- Chess puzzle understanding benefits from representing legal consequences.
- Near-puzzles may look tactical statically but differ in consequence structure.
- Self-supervised legal dynamics can improve representations without engine labels.
- Auxiliary legal-move/transition tasks are generated deterministically and do not leak puzzle labels.

## Claim

Hypothesis: a latent dynamics bottleneck should improve puzzle-binary PR AUC and reduce near-puzzle false positives because it forces the model to encode consequence-bearing features that static CNNs can ignore.

## Mechanism

The model must answer:

```text
what moves are legal here?
what latent state would result?
which board features change consequences sharply?
```

Puzzle positions often hinge on consequence-sensitive facts: pins, overloaded defenders, king escape, recapture legality, and line openings. A dynamics-aware representation should make those facts easier for the puzzle head.

## Proof Sketch

What can be reasoned about:

- Auxiliary targets are generated from chess rules, not labels.
- If the model solves legal/transition tasks, its latent must encode some chess legality and consequence structure.
- Comparing with the same trunk without dynamics losses isolates the value of the bottleneck.

This does not prove dynamics pretraining improves puzzle classification. It makes the representation-learning hypothesis testable.

## Not Proven

- That one-ply legal dynamics is enough for puzzle discrimination.
- That auxiliary losses will not distract from the binary target.
- That generated pseudo-legal negatives are representative.
- That transition reconstruction can be learned efficiently from FEN-only data.

## Counterexamples

- Puzzle labels dominated by static motifs that a CNN already learns.
- Near-puzzles whose one-ply legal dynamics is almost identical to puzzles.
- Positions requiring deep search beyond one-ply dynamics.
- Auxiliary task imbalance causing the encoder to underfit puzzle classification.

## Falsification Test

Compare:

- base encoder with puzzle head only
- dynamics network with legal-head only
- dynamics network with next-latent consistency
- dynamics network with all auxiliary heads

Revise or reject if:

```text
all dynamics variants match or underperform the base encoder
or near-puzzle false-positive rate does not improve by >= 0.02 absolute
or dynamics diagnostics do not differ between puzzles and near-puzzles
```

Reject the dynamics hypothesis if legal/transition auxiliary accuracy is high but puzzle metrics do not improve.


# Mathematical Thesis

## Actual Task And Labels

The first target is the corrected puzzle-binary benchmark:

```text
source class 0: known non-puzzle / random position -> y = 0
source class 1: verified near-puzzle / hard negative -> y = 0
source class 2: verified puzzle -> y = 1
```

The model emits one puzzle logit. Source classes are used for diagnostics only.

Allowed inputs:

- current-board tensor
- deterministic rule-bounded edit operators derived from the current board
- current-board legal geometry and piece identities

Forbidden inputs:

- Stockfish scores
- Stockfish PVs or best moves
- node counts
- mate scores
- verification metadata
- source labels
- source file identity
- future game outcomes

## Baseline And Weakness

Closest registered ideas:

- `i005_null_move_contrast_puzzle_network`: tests one specific side-to-move edit.
- `i006_proof_core_set_verifier`: selects a sparse proof core.
- `i007_neural_proof_number_search`: builds a bounded move-continuation proof tree.

Closest research-packet idea:

- Minimal-Edit Puzzle Distance Network.

Overlap: this idea also treats near-puzzles as positions close to puzzlehood.

Difference: this idea uses a constrained unrolled Lagrangian edit solver with both puzzle-making and puzzle-breaking energies, and it requires boundary-distance diagnostics. It is not just a head predicting edit costs, and it does not fabricate edited-board labels.

## Definitions

Let `x` be a chess position. Define a finite edit basis:

```text
E(x) = {e_1, ..., e_m}
```

Each edit is a soft, rule-bounded perturbation over the current-board representation, not an actual new labeled sample. Example edit families:

- flip side-to-move feature
- remove or weaken a defender token
- open or close a slider line blocker
- alter target protection count
- suppress or restore king escape square feature
- change one relation edge in a pinned/overloaded motif

Let `phi(x)` be a board encoder and `C(phi)` be a base classifier. Let edit weights `alpha in [0,1]^m` define a soft edited latent:

```text
z(alpha) = phi(x) + sum_i alpha_i * delta_i(x)
edit_cost(alpha) = sum_i lambda_i * alpha_i + pair_penalty(alpha)
```

Define puzzle-making energy:

```text
E_plus(x) = min_alpha edit_cost(alpha) + beta * softplus(-C(z(alpha)))
```

Define puzzle-breaking energy:

```text
E_minus(x) = min_alpha edit_cost(alpha) + beta * softplus(C(z(alpha)))
```

The final logit uses both:

```text
f(x) = MLP([C(phi(x)), E_minus(x) - E_plus(x), E_plus(x), E_minus(x), edit_stats])
```

Interpretation:

- True puzzles should have low `E_plus` and high `E_minus`.
- Near-puzzles may have low but nonzero `E_plus` and lower `E_minus`.
- Random non-puzzles should have high `E_plus`.

## Assumptions

- Verified near-puzzles are often close to true puzzles under a small chess-shaped edit.
- Verified puzzles are harder to edit into non-puzzles than near-puzzles are to edit into puzzles.
- Rule-bounded soft edits can expose this boundary without leaking labels.
- Binary supervision plus source-class diagnostics can test the geometry.

## Claim

Hypothesis: modeling puzzlehood as a boundary-distance problem should reduce near-puzzle false positives, because the model can represent "almost puzzle" separately from "already puzzle."

## Mechanism

A normal classifier compresses both near-puzzles and true puzzles into one scalar. This architecture explicitly estimates two energies:

```text
how much edit effort makes this position puzzle-like?
how much edit effort destroys its puzzle evidence?
```

The gap between those energies should be more stable than raw puzzle evidence.

## Proof Sketch

What can be reasoned about:

- The edit solver is deterministic given the board and learned parameters.
- No edited board is assigned a fabricated class label.
- If a near-puzzle is truly close to a puzzle under the chosen edit basis, `E_plus` can be low while the final classifier still predicts non-puzzle.
- If a true puzzle depends on a small tactical core, `E_minus` should be higher than for near-puzzles.

This is not a proof that puzzlehood is an edit-distance property. It is a strong, falsifiable boundary hypothesis.

## Not Proven

- That the edit basis covers the important ways near-puzzles differ from puzzles.
- That unrolled optimization will find meaningful edit weights.
- That `E_plus` and `E_minus` will be identifiable from binary labels.
- That the model will not collapse to the base classifier.

## Counterexamples

- Near-puzzles that are not close to any true puzzle under the edit basis.
- True puzzles whose key feature is multi-move search rather than local boundary edits.
- Positions where a single side-to-move flip explains most of the signal, making the richer edit basis unnecessary.
- Data artifacts that make boundary geometry irrelevant.

## Falsification Test

Compare against:

- BT4 baseline
- base classifier without edit solver
- null-move contrast model
- random edit basis
- one-sided `E_plus` only model

Revise or reject if:

```text
test PR AUC <= 0.82
or near-puzzle -> puzzle false-positive rate is not below 0.20
or random edit basis matches legal edit basis
or edit energies do not order as E_plus(random) > E_plus(near) > E_plus(puzzle)
```

Reject the boundary hypothesis if the base classifier matches the full model and edit diagnostics carry no source-class signal.


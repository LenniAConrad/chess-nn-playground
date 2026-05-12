# Mathematical Thesis

## Actual Task And Labels

The first benchmark is:

```text
source class 0: known non-puzzle / random position -> y = 0
source class 1: verified near-puzzle / hard negative -> y = 0
source class 2: verified puzzle -> y = 1
```

The model emits one puzzle logit. Fine source labels are used for metrics and 3x2 diagnostics, not inference.

Allowed inputs:

- current-board tensor
- deterministic piece/square tokens
- deterministic relation features among selected witnesses

Forbidden inputs:

- Stockfish scores, PVs, nodes, mate scores
- verification metadata
- source labels
- source file identity
- engine best moves

## Baseline And Weakness

Closest registered ideas:

- `i003_factor_agreement_classifier`: combines multiple factors.
- `i004_puzzle_obligation_flow_network`: uses obligation/resource allocation.
- `i005_null_move_contrast_puzzle_network`: uses tempo counterfactual contrast.

Closest research-packet ideas:

- Sparse witness-piece bottleneck.
- Forcing-certificate transformer.
- Critical-square budget network.

Overlap: this idea also assumes puzzle evidence can be sparse.

Difference: this idea requires a selected proof core to be both sufficient and deletion-sensitive. It tests whether removing the selected core collapses the score and whether adding irrelevant context does not dominate the verifier.

## Definitions

Let `T(x)` be a set of piece and square tokens from the current position:

```text
T = occupied pieces + king-zone squares + high-value target squares + line-intersection squares
```

A selector predicts witness weights:

```text
w_i = selector(t_i, board_context)
S_k = top_k(T, w, k)
```

A verifier sees only selected tokens and deterministic relations among them:

```text
V(S_k, Rel(S_k)) -> proof_logit
```

Final output:

```text
f(x) = proof_logit + small_global_residual
```

with the residual initialized small and bounded by configuration.

Deletion diagnostic:

```text
f_delete(x) = V(T without S_k)
deletion_gap = f(x) - f_delete(x)
```

For true positives, the hypothesis predicts a larger deletion gap than for near-puzzles.

## Assumptions

- Many real puzzles have a compact causal core of pieces/squares.
- Near-puzzles may activate broad tactical features without a stable compact core.
- A verifier restricted to selected witnesses is less likely to exploit global source artifacts.

## Claim

Hypothesis: a proof-core bottleneck should reduce near-puzzle false positives because the model must find a small sufficient witness set rather than classify from full-board tactical texture.

## Mechanism

The architecture asks:

```text
which small set of pieces/squares makes this a puzzle?
```

If the answer is unstable or diffuse, the verifier should be less confident. If a true puzzle hinges on a fork, pin, overloaded defender, mating net, or promotion tactic, a small proof core should often be enough.

## Proof Sketch

What can be reasoned about:

- The verifier is permutation-invariant over selected witness tokens.
- The bottleneck limits information capacity.
- Deletion tests can measure whether selected witnesses are actually used.
- Random witness ablations test whether selection is meaningful.

This is not a proof that all puzzles have small proof cores.

## Not Proven

- That `k <= 12` is enough.
- That differentiable top-k learns stable witnesses from binary labels.
- That the residual head will not bypass the bottleneck unless constrained.
- That selected cores will align with human tactical explanations.

## Counterexamples

- Long quiet maneuvers requiring broad board context.
- Endgame tablebase tactics.
- Positions with several independent tactical themes.
- Labels with dataset artifacts that can be detected from selected nuisance tokens.

## Falsification Test

Train against a full-board verifier with the same trunk and a random-witness baseline.

Revise or reject if:

```text
learned proof-core verifier does not beat random witnesses
and deletion_gap is not larger for true puzzles than near-puzzles
and near-puzzle false-positive rate does not improve over full-board baseline
```

Reject the core hypothesis if the bounded global residual alone matches the full model.


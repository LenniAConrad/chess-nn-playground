# Mathematical Thesis

## Actual Task And Labels

First benchmark:

```text
source class 0: known non-puzzle / random position -> y = 0
source class 1: verified near-puzzle / hard negative -> y = 0
source class 2: verified puzzle -> y = 1
```

The model emits one puzzle logit. The source class is used for diagnostics only.

Allowed inputs:

- current-board tensor
- deterministic side-to-move swapped tensor
- deterministic null-move view of the same board
- rule-derived current-board metadata already encoded in the board representation

Forbidden inputs:

- Stockfish scores, PVs, node counts, mate scores
- verification metadata
- source labels or source file identity
- engine best moves
- future game outcomes

## Baseline And Weakness

Closest registered ideas:

- `i002_response_minimax_classifier`: uses action/reply candidates.
- `i003_factor_agreement_classifier`: compares multiple views.
- `i004_puzzle_obligation_flow_network`: models defensive resource insufficiency.

Closest research-packet ideas:

- Tempo-odd bottleneck packets.
- Side-to-move intervention packets.

Overlap: this idea also uses side-to-move as a chess-specific signal.

Difference: this idea makes the classification evidence depend on a shared-trunk contrast between current and null-move counterfactual views, rather than a parity decomposition, generic view agreement, or explicit move/reply set.

## Definitions

Let `x` be a legal position and `n(x)` be the deterministic null-move counterfactual: same board occupancy and rights except the side-to-move feature is swapped according to a documented encoder rule.

Let a shared encoder produce:

```text
z_cur = E(x)
z_null = E(n(x))
e_cur = h(z_cur)
e_null = h(z_null)
```

Define contrast features:

```text
delta = e_cur - e_null
abs_delta = |e_cur - e_null|
z_cross = cross(z_cur, z_null)
```

Final logit:

```text
f(x) = MLP([e_cur, e_null, delta, abs_delta, z_cross])
```

Optional positive-only margin, using only true labels:

```text
for y = 1: max(0, margin - (e_cur - e_null))
```

No target is assigned to the null view itself. The null view is a counterfactual feature, not a fabricated training example.

## Assumptions

- Verified puzzles are often immediate side-to-move opportunities.
- Near-puzzles may look tense but lose much less evidence under side-to-move swap.
- The null-move contrast exposes tempo-criticality without engine search.

## Claim

Hypothesis: null-move contrast will reduce near-puzzle false positives because positions that are merely sharp but not tactically forcing will not show the same current-vs-null evidence gap as true puzzles.

## Mechanism

The network is not allowed to only ask "does this board look tactical?" It must also ask:

```text
does the tactical evidence depend on the current player being to move?
```

That question is highly relevant for puzzles, where the existence of a tactic is usually immediate and tempo-bound.

## Proof Sketch

What can be reasoned about:

- The contrast is deterministic and uses no forbidden metadata.
- If a model's signal is side-to-move invariant, the contrast path contributes little.
- If a model's signal depends strongly on tempo, `delta` can express that directly.

This does not prove that every puzzle is null-move sensitive. It only creates a falsifiable test for tempo-critical puzzle signal.

## Not Proven

- That all puzzle labels are immediate-tactic labels.
- That null-move counterfactuals are meaningful in positions where castling, en-passant, or check state complicate side-to-move swaps.
- That near-puzzles are less tempo-critical than puzzles.

## Counterexamples

- Puzzle positions where both sides have tactics.
- Strategic or endgame puzzle positions where the key idea is not a simple tempo swing.
- Positions where the side-to-move swap creates illegal or semantically odd auxiliary state.
- Data source artifacts that correlate with side-to-move distribution.

## Falsification Test

Train against a shared-trunk board-only model with the same parameter count.

Revise or reject if:

```text
null-move contrast does not improve PR AUC by >= 0.015
and does not reduce near-puzzle -> puzzle false positives by >= 0.02 absolute
```

Reject the tempo-contrast hypothesis if:

```text
randomly swapped side-to-move features perform the same as the deterministic null view
or delta features are not larger on true puzzles than on near-puzzles
```


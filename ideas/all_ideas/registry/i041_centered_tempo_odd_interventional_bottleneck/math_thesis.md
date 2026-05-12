# Math Thesis

Centered Tempo-Odd Interventional Bottleneck

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-21_0729_tuesday_pacific_tempo_odd_bottleneck.md`.

## Working Thesis

Puzzle-likeness should be predicted from the board-dependent part of the
position's response to changing only the side to move, not from static
board appearance, source artifacts, or a raw side-to-move prior. The
classifier consumes a side-to-move anti-invariant projection of a shared
encoder's features, after subtracting the encoder's response on a
null-board carrier of the same turn bit.

## Input Space

A simple_18 board tensor is split semantically into a board variable
`b` (the 12 piece planes plus castling/en-passant planes) and a
side-to-move bit `t` carried by channel 12. The model never enumerates
legal moves, never reads engine evaluations, never consumes CRTK source
labels, and never sees verification metadata.

The two deterministic tensor maps are:

- `tau(b, t) = (b, -t)` — only the side-to-move plane is flipped.
- `nu(b, t) = (b_0, t)` — every plane except the side-to-move plane is
  zeroed.

`tau(x)` and `nu(x)` need not be legal FENs; they are deterministic
tensor views used to expose the encoder's structure under turn
intervention.

## Centered Odd Operator

Let `F_theta : X -> R^{D x 8 x 8}` be the shared convolutional encoder.
The model computes

```
O_theta(x)        = 0.5 * (F_theta(x) - F_theta(tau x))
E_theta(x)        = 0.5 * (F_theta(x) + F_theta(tau x))
C_theta(x)        = O_theta(x) - O_theta(nu x)
```

`C_theta(x)` is the centered odd map. Two structural properties hold:

1. `C_theta(tau x) = -C_theta(x)`: the centered representation is
   anti-invariant under turn toggling.
2. If `F` admits a decomposition
   `F(b, t) = a(b) + c + t u + t v(b) + r(b, t)` with `v(b_0) = 0`, then
   the additive board-only term `a(b)`, the constant `c`, and the
   pure-turn term `t u` are exactly removed, leaving only the
   board-turn interaction `t v(b)` and the centered odd part of the
   non-additive residual `r`.

The classifier head reads spatial mean, max and RMS of `C_theta(x)` and
returns puzzle logits.

## Hypotheses

- True puzzle-likeness in this dataset is better captured by board-turn
  interaction than by static board appearance or source artifacts.
- Suppressing static and pure-turn shortcuts via the centered odd
  bottleneck improves fine-label `1` near-puzzle behavior at matched
  fine-label `0` false-positive rate.
- Out-of-distribution counterfactual tensors `tau(x)` and `nu(x)` still
  carry useful representation-learning signal for chess tempo even
  though they are not always legal positions.

## Counterexamples

The mechanism should fail or be flat on:

- Datasets dominated by static source artifacts independent of
  side-to-move.
- Splits where the side-to-move plane itself is spuriously predictive
  and the model recovers it through nonlinear interactions despite
  centering.
- Positions whose puzzle-likeness depends mainly on multi-ply
  consequences invisible to a side-toggle contrast.
- Encodings whose side-to-move semantics are unknown; the deterministic
  adapter fails closed in that case.

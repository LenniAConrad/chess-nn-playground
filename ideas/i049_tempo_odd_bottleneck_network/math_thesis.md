# Math Thesis

Tempo-Odd Bottleneck Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-21_0755_tuesday_los_angeles_tempo_odd_bottleneck.md`.

## Working Thesis

Puzzle-likeness is often a side-to-move interaction property: the
log-odds for a position `x = (B, s, R, E)` are well-approximated by
`eta(x) ~ a(B, R) + s b(B, R)`, where the first term is a side-blind
static context (and any source/material shortcuts) and the second term
is the tempo-dependent tactical interaction. The classifier should
isolate the latent component that changes under a rule-only side-to-move
intervention rather than relying on side-blind static board shortcuts.

## Input Space and Intervention

For the `simple_18` minimal experiment the input tensor has 18 planes:
12 piece occupancy planes (`B`), the side-to-move plane at channel 12
(`S`, encoding `s in {0, 1}`), four castling planes (`R`), and an
en-passant plane (`E`). The deterministic rule-only intervention is

```
tau(B, S, R, E) = (B, 1 - S, R, 0).
```

The en-passant plane is sanitized to `0` in both `x` and `tau(x)`
because en-passant is history-sensitive and can become semantically
inconsistent under a pure side-to-move toggle. `tau` is an involution:
`tau(tau(x)) = x`. No move generation, mate flag, engine input, CRTK
source label, or verification metadata is consulted; for encodings
without declared side-to-move semantics the adapter fails closed.

## Two-Point Walsh Odd/Even Operator

For any shared encoder `h_theta : X -> R^d` define

```
P_+ h(x) = 0.5 * (h(x) + h(tau x))
P_- h(x) = 0.5 * (h(x) - h(tau x)).
```

Two structural properties hold:

1. `P_+ h(tau x) = P_+ h(x)` and `P_- h(tau x) = -P_- h(x)`, so the
   even projection is `tau`-invariant and the odd projection is
   `tau`-anti-invariant.
2. If `h(B, s, R) = u(B, R) + s v(B, R)` for `s in {-1, +1}` then
   `P_+ h = u(B, R)` and `P_- h = s v(B, R)`. The odd projection
   exactly cancels any side-blind term expressible inside the encoder
   and exactly recovers the first-order side-to-move interaction.

The bespoke architecture exposes `P_- h` as the high-capacity
predictive path (signed and magnitude features through a no-bias
projection to `odd_dim`) and exposes `P_+ h` only through a small
context bottleneck of width `even_dim` with stop-gradient applied by
default so label gradients cannot reach the shared encoder via the
side-blind route. The classifier head consumes
`cat([odd_signed, odd_magnitude, even_context])` and returns puzzle
logits.

## Hypotheses

- True puzzle-likeness in this dataset is better captured by tempo-odd
  interaction than by static board appearance or source artifacts.
- Restricting the predictive path to the odd projection improves
  near-puzzle (fine label `1`) recall at matched fine label `0`
  false-positive rate.
- The optional batch-variance floor on `odd_signed` is an anti-collapse
  regularizer, not the central operator; the falsification ablation
  sets `odd_variance_weight = 0`.

## Counterexamples

The mechanism should fail or be flat on:

- Positions whose tactical opportunities are nearly symmetric in side
  to move.
- Puzzles whose label is mostly source-specific or composition-specific
  rather than tempo-specific.
- Static mating nets where toggling side-to-move barely changes the
  relevant pattern.
- Endgame zugzwang-like positions where legal-move availability matters,
  which this model intentionally does not enumerate.
- En-passant-specific tactics, since the safe variant zeroes the
  en-passant plane.

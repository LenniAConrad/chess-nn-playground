# Math Thesis

Support-Function Envelope Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2204_friday_shanghai_architecture_batch_8.md`.

Batch candidate rank: `2`.

## Working thesis

A chess position has geometric envelopes: where the side-to-move has force,
where the opponent has force, how far pieces extend toward the king, and how
concentrated material is along important directions. A differentiable
support-function readout summarises these envelopes compactly without
attention, move generation, or graph construction.

## Object

For a learned nonnegative field `rho_c(s)` over board squares and a unit
direction `u`, define

```
a_c(s) = log(epsilon + rho_c(s))
h_c(u) = tau * logsumexp_s ( (dot(u, coord_s) + a_c(s)) / tau )
```

where `coord_s` is the square coordinate normalised to `[-1, 1]^2`. As
`tau -> 0` this approaches the support function of the field's support set.
For each direction `u`,

```
w_c(u) = h_c(u) + h_c(-u)   # envelope width
m_c(u) = h_c(u) - h_c(-u)   # envelope center along u
```

The direction set is fixed and chess-relevant: rank, file, the two
diagonals, and the four knight slopes (each paired with its negative).

## Side-to-move contrast

Splitting fields into own and opponent halves, the model emits

```
overlap_gap(u) = |m_own(u) - m_opp(u)|
width_ratio(u) = w_own(u) / (epsilon + w_opp(u))
```

per direction. The same contrast is applied to the deterministic per-piece
own/opp planes recovered from the side-to-move flip.

## Head

Per-field descriptors `(h, w, m, mass, entropy, max)` and the own/opp
contrast features are concatenated and passed through a small MLP that
emits the single puzzle logit.

# Math Thesis

Piece-Drop Stability Network

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2118_friday_shanghai_architecture_batch_3.md`.

Batch candidate rank: `6`.

Working thesis: Puzzle-like positions may be less stable under deterministic removal of specific safe current-board evidence groups. Instead of forcing a classifier to use sparse witnesses, measure how a small encoder's latent changes when piece groups are dropped.

## Formal Setting

Let `x ∈ R^{18×8×8}` be the simple_18 board tensor and `f_θ : R^{18×8×8} -> R^D` a shared convolutional encoder producing a latent `z(x) = f_θ(x)`. For each deterministic piece-drop operator `M_m`, the masked input `M_m(x)` zeroes out the piece planes belonging to a fixed semantic group while leaving auxiliary planes (side-to-move, castling, etc.) intact. The stability functional is

```
delta_m(x) = ||z(x) - z(M_m(x))||_2
```

The classifier `g_φ` is trained to predict the puzzle label from the joint feature

```
h(x) = [ z(x) ; delta_1(x) , ... , delta_M(x) ; delta_1(x)/||z(x)|| , ... , delta_M(x)/||z(x)|| ]
```

so that the gradient signal flows back through both the encoder and the deterministic mask operators.

## Why This Should Help

Puzzle positions usually hinge on a small number of specific pieces (the attacker, the pinned defender, the king's escape squares). Deterministic removal of a *semantic* group — own minors, opponent rooks/queens, the king ring, the center — should shift `z(x)` by a non-trivial amount when that group carries the tactical motif, and shift it only marginally when the position is an ordinary middlegame. The per-group stability vector is a low-dimensional fingerprint of *which* groups matter, which is the exact invariant the head reads.

## Group Definitions

For the side to move `s ∈ {0, 1}` decoded from plane 12:

- `own_minor`: piece planes `(N, B)` for the side-to-move color.
- `own_major`: piece planes `(R, Q)` for the side-to-move color.
- `opp_minor`: piece planes `(N, B)` for the opposite color.
- `opp_major`: piece planes `(R, Q)` for the opposite color.
- `center`: per-square mask on the four squares `(rank ∈ {3, 4}) × (file ∈ {3, 4})`.
- `king_neigh`: per-square 3×3 dilation of the union of both king planes.

Only the 12 piece planes are zeroed; planes 12..17 (side-to-move and any auxiliary state) are preserved.

## Implementation

The bespoke implementation lives in
`src/chess_nn_playground/models/trunk/piece_drop_stability_network.py` and is
registered as `piece_drop_stability_network`. The idea-local wrapper at
`ideas/registry/i112_piece_drop_stability_network/model.py` calls
`build_piece_drop_stability_network_from_config` directly. The previous
shared `ResearchPacketProbe` scaffold has been removed.

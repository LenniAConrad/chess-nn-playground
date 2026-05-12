# Math Thesis

Piece Liability Gradient Network

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-25_0109_saturday_shanghai_high_upside_puzzle_batch_4.md`.

Batch candidate rank: `5`.

Working thesis: In many puzzles, one piece is not merely attacked; it is liable. It cannot move, defend, capture, or stay without losing something. Near-puzzles may attack pieces, but the liability does not propagate.

## Formal Sketch

Let `S = 64` squares and let `piece_mask[s] in {0, 1}` indicate that a piece occupies square `s`. For each occupied square the network learns a per-affordance action value `a[i, s]` for `i in {move, defend, capture, stay}`. The *liability* of a square is the soft minimum of these affordances:

```
soft_min(s) = -tau * logsumexp(-a[:, s] / tau)
L_0(s)      = sigmoid(-soft_min(s) / lambda) * piece_mask(s)
```

`L_0(s)` is high when *every* affordance is bad, i.e. the piece cannot move, defend, capture, or stay without losing something. A piece that is attacked but has a good `move` or `defend` value keeps `L_0` low: pure attacks do not by themselves create liability.

Liability of one piece may transfer to another. If a defender is liable, the piece it defends becomes more likely to be liable; if the only escape square is itself contested, liability rises along that retreat. The network captures this with a row-stochastic relation bank `relations[r, :, :]` and a per-round, per-relation gate `gate[t, r] in [0, 1]`:

```
delta(s)        = sum_r gate[t, r] * sum_{s'} relations[r, s, s'] * L_t(s')
L_{t+1}(s)      = L_t(s) + (1 - L_t(s)) * delta(s) * piece_mask(s)
```

The probabilistic-OR step keeps `L_t in [0, 1]` and confines liability to occupied squares. After `K` rounds the difference `L_K - L_0` measures how much liability *propagated*. A real puzzle is expected to admit a high-mass liability gradient (one piece's local trouble lifts neighbouring pieces); a near-puzzle is expected to attack pieces but show little propagation, because the attack does not chain into a forced material loss.

The classifier head reads aggregate descriptors of `L_K` (max, mean over occupied squares, top-`k`) together with the propagation magnitude `mean(L_K - L_0)` and a pooled trunk summary, and emits one puzzle logit.

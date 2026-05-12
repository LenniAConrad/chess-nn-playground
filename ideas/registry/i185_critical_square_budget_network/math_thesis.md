# Math Thesis

Critical-Square Budget Network

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-25_0037_saturday_shanghai_puzzle_architecture_batch_2.md`.

Batch candidate rank: `9`.

Working thesis: Puzzles often hinge on a small number of critical
squares -- king escape squares, line intersections, pinned-piece
squares, promotion squares, or overloaded defender squares -- so the
puzzle logit should be read out of a *budget-constrained* pool that is
allowed to attend to at most a few squares of the board.

Formalisation: with trunk features `f in R^{C x 8 x 8}` and per-square
saliency `s in R^{8 x 8}` the model's mask is

```text
m = K * softmax(s / tau),    sum_{r,c} m[r, c] = K,    m >= 0.
```

The puzzle logit is then `head(sum_{r,c} m[r,c] * f[:, r, c], summary)`
where `summary` reports how much of `m`'s mass falls inside each
prior critical-square region (king zones, promotion ranks, line-piece
intersections, empty squares). `K` is the budget; `tau` controls how
sparse the mask becomes. Lower `tau` and small `K` together enforce
"few critical squares".

The budget `K` is the central knob the packet proposes: the model is
forced to make do with `K` squares' worth of pooled evidence per
position, which is the operational cost of the "critical squares are
few" thesis.

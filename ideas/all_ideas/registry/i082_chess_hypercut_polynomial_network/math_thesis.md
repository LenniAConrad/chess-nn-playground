# Math Thesis

Chess Hypercut Polynomial Network (CHPNet)

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-28_0733_tuesday_new_york_hypercut_poly.md`.

## Working thesis

Let `G = (V, E)` be a deterministic chess hypergraph on the 64 board squares
`V`. Each hyperedge `e in E` is a vertex subset `e subset V` produced from
the current board by chess rules only:

- a sliding ray from a bishop / rook / queen until the first occupied square
  inclusive,
- a knight halo around each knight,
- a king 8-neighbourhood around each king (including a hard king-shell
  alias),
- a pawn stencil over forward push, double push (from the start row), and
  diagonal captures,
- and an occupied line window along any rank, file, or diagonal of size
  `3..max_edge_size` that contains at least two occupied squares.

Let `s in [-1, 1]^V` be a learned probe field over square tokens. The
**masked cut polynomial** on hyperedge `e` is

```
c_e(s) = 1 - prod_{v in e} (1 + s_v) / 2 - prod_{v in e} (1 - s_v) / 2,
```

which is exactly `0` when `s_v` agrees in sign on every `v in e` and `1`
when the signs split across `e`. CHPNet uses `c_e` as a per-edge cut score
and the **exclusive-product derivative**

```
partial c_e / partial s_v
    = -(1/2) prod_{w in e, w != v} (1 + s_w) / 2
      + (1/2) prod_{w in e, w != v} (1 - s_w) / 2
```

as a per-vertex residual, scattered back to square states with degree
normalisation. Stacking masked, masked-active hyperedge cut polynomials
gives a high-order chess-rule mechanism that no purely pairwise CNN or
2-square attention can reproduce.

## Puzzle-binary head

After `L` hypercut blocks, the model pools square states (mean and max),
concatenates the per-block cut moments `(mean, max, std)` of `c_e(s)`
restricted to active edges, and reads out a single puzzle logit through a
`LayerNorm -> Linear -> GELU -> Linear` head. The head consumes only
features derived from the canonicalised board tensor and the deterministic
chess-rule hypergraph; CRTK / engine / source metadata is reporting-only.

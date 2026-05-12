# Math Thesis

`Threat-Topology Betti Bottleneck Network` tests whether the topology of current-board rule pressure fields carries puzzle-binary signal that is not reducible to material, pressure histograms, or ordinary CNN texture.

For the supported `simple_18` encoding, the deterministic branch decodes the current piece planes, side to move, occupancy, and king planes. It computes pseudo-legal attack pressure by color using only current-board movement rules: pawns attack diagonally by color, knights and kings use fixed offsets, and sliding pieces stop at the first occupied square. It does not use legal move generation, checkmate logic, engine output, source metadata, or labels.

Let `A_c(v)` be weighted pseudo-legal attack pressure by color `c` on square `v`, and let `M_c(v)` be the target material-value field. With `m` as the side to move and `bar(m)` as the opponent, the implemented branch builds four scalar fields:

```text
F1 = A_m - A_bar(m) + alpha M_bar(m)
F2 = A_bar(m) - A_m + alpha M_m
F3 = (A_m - A_bar(m)) G_king_bar(m) + alpha M_bar(m)
F4 = (A_bar(m) - A_m) G_king_m + alpha M_m
```

`G_king` is an exponential Chebyshev-distance kernel centered at the current king square, falling back to board center only when a synthetic tensor lacks a king.

For each field `F` and rank budget `k`, the topology branch forms the superlevel set `T_k(F)` containing the top `k` squares under deterministic square-index tie breaking. The cubical complex is the union of the closed unit cells for those squares. The bottleneck records:

```text
B(F,k) = (beta0, beta1, boundary_edges, topk_mean)
```

where `beta0` is the number of 4-neighbor connected cell components, `beta1 = beta0 - V + E - C` is the cubical one-cycle count from vertices, edges, and cells, `boundary_edges` is the exposed cell-side count, and `topk_mean` is the mean pressure over the selected squares.

The proposition implemented by the tests is the packet's histogram counterexample: two fields can share the same sorted values and top-k means while having different `beta0` because the high-rank squares are spatially contiguous in one field and separated in the other. A rank shuffle preserves the scalar values and rank budgets while destroying board adjacency, so it is the central semantics-destroying ablation.

The model learns only the topology MLP, matched CNN stem, and fusion head. The rule-pressure and Betti features are deterministic current-board functions.

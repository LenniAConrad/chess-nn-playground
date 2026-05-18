# Math Thesis

Commutative View-Consistency Network

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2204_friday_shanghai_architecture_batch_8.md`.

Batch candidate rank: `1`.

Working thesis: A chess position can be represented through several safe
current-board views (square grid, occupied piece set, rank/file/diagonal line
summaries, king-centred regions, material/phase summaries). Puzzle-like
positions may be recognizable not only by any one view, but by how these
views agree or disagree after learned projections into a common latent space.
The model learns low-rank maps between latent views and classifies from
commutator-like consistency defects between those maps.

## Formal description

Let `x ∈ R^{18 × 8 × 8}` be a simple_18 board tensor and let `V =
{square, piece, line, region, count}` be the five view names. The model
defines:

- View latents `z_v = E_v(x) ∈ R^D` produced by view-specific encoders
  `E_v` (a small CNN for `square`, DeepSets over occupied piece tokens for
  `piece`, and MLPs over deterministic 30-/8-/25-dimensional summary
  features for `line`, `region`, `count`).
- Cross-view low-rank maps `A_{u → v} : R^D → R^D` with rank at most
  `r = map_rank`, parameterised as `Linear(D, r)` followed by
  `Linear(r, D)`. The active map set is

  ```text
  E = {(square,line), (line,square), (square,region), (piece,region),
       (region,count), (square,count), (count,square), (region,piece)}.
  ```

- Direct cross-view defects `d^{(u,v)} = z_v - A_{u → v}(z_u)` for each of
  six selected edges in `E`.
- Two-step cycle defects
  `c^{(v,m)} = z_v - A_{m → v}(A_{v → m}(z_v))` for three selected loops
  `(v, m) ∈ {(square, line), (piece, region), (square, count)}`.
- Per-defect statistics `phi(d) = (||d||^2/D, ||d||_1/D, mean(d),
  ||d||_∞, cos(target, predicted))`. Stacking over the nine defect vectors
  yields a `(B, 9, 5)` tensor.
- A puzzle logit
  `y = h([z_square, z_piece, z_line, z_region, z_count, vec(phi)])`,
  where `h` is a LayerNorm + GELU MLP head reading the five projected view
  summaries and the 45 per-defect statistics.

The view encoders, the cross-view maps `{A_{u → v}}`, and the head `h` are
the only trainable modules; the line/region/count summary tables, the piece
DeepSets coordinate buffer, and the view registry are non-learnable.

## Decision rule

The packet's promotion criterion is: keep the model if the defect head beats
the `views_only_no_defects` ablation (defects add information beyond
multi-view features) and the `single_square_view` ablation (multi-view
structure matters) on the puzzle_binary benchmark contract, while the
`random_view_maps` ablation strictly underperforms (learned cross-view maps
matter) and the `count_to_all_only` ablation does not catch up to `none`
(the model is not a pure material shortcut). The `shuffled_piece_view`
ablation should also degrade performance if the piece view contributes real
piece-square geometry; otherwise the piece encoder should be simplified.

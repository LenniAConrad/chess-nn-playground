# Math Thesis

Row-File Factor Mixer

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2121_friday_shanghai_architecture_batch_4.md`.

Batch candidate rank: `1`.

Working thesis: chess boards have two privileged axes -- ranks and files --
and a useful inductive bias is to factorize board processing into rank
mixers, file mixers, and piece-channel mixers, then recombine them with a
bilinear interaction between rank-mixed and file-mixed features. This
avoids the quadratic cost of a full square-by-square Transformer while
still exposing axis-aligned structure that a plain CNN cannot easily
recover.

Notation. Let `x in R^{C x H x W}` be a board feature tensor with
`H = W = 8`. Define:

- Rank mixer `R`: a function applied along the rank axis (length `H`),
  shared across files and channels: `(R x)[c, i, j] = sum_k W^R_{ik} x[c, k, j]`.
- File mixer `F`: a function applied along the file axis (length `W`),
  shared across ranks and channels: `(F x)[c, i, j] = sum_k W^F_{jk} x[c, i, k]`.
- Channel mixer `M`: a per-square MLP over the piece-channel axis,
  shared across squares.

Each mixer block computes `R x` and `F x` from a normalized residual
stream, forms the bilinear recombination `B(x) = phi((R x) odot (F x))`
(where `phi` is a linear 1x1 channel projection over the elementwise
Hadamard product), and updates the residual stream as
`x <- x + R x + F x + B(x)`. A subsequent channel mixer adds
`x <- x + M(x)`.

The bilinear `(R x) odot (F x)` term is the "recombine with bilinear
interactions" step from the thesis: rank-axis and file-axis information
are blended multiplicatively before being projected back to channels.
Diagnostics (rank energy, file energy, bilinear energy, rank-file
imbalance) are computed by averaging squared activations over space and
exposed alongside the puzzle logit so that downstream slice reports can
audit which factor the model is leaning on.

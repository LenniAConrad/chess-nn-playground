# Math Thesis

Tensor-Core Square-Pair Field Network tests whether a deliberately dense
linear-algebraic board representation improves puzzle-binary classification.

Instead of extracting sparse chess objects, the model builds the full ordered
square-pair universe:

```text
F in R^(B x H x 64 x 64)
```

Every square can interact with every other square through a learned pair score,
fixed chess-geometry relation planes, and dense batched matrix multiplication. The
scientific claim is that puzzle-like positions may be better separated by
square-pair energy patterns such as same-file alignment, diagonal tension,
knight-offset geometry, occupied-to-empty pressure, and king-zone pair energy than
by ordinary local convolution alone.

The central distinction from attention is that the pair field is not only a routing
probability. The implementation keeps pair scores as features, uses tanh-normalized
dense messages for square updates, and exposes relation-conditioned pair energies to
the classifier and prediction diagnostics.

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2148_friday_shanghai_tensorcore_pairfield.md`.

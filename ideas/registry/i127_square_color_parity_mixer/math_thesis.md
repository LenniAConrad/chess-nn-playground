# Math Thesis

Square-Color Parity Mixer

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2133_friday_shanghai_architecture_batch_6.md`.

Batch candidate rank: `3`.

The chessboard is bipartite by square color. Bishops are constrained to one square color, knights always change square color, pawn captures switch files and square color, and kings and queens mix parity through local or ray movement. Tactical motifs can therefore be represented as coupled signals on two 32-square subspaces rather than as one undifferentiated 64-square field.

Let `D` and `L` be learned token matrices for dark and light squares. A parity-aware mixer should learn same-color operators `A_dark`, `A_light` and a cross-color operator `C_cross`:

```text
[ D' ]   [ A_dark    C_cross ] [ D ]
[ L' ] = [ C_cross^T A_light ] [ L ]
```

This block matrix gives the model a direct way to separate bishop-like within-color persistence from knight-, pawn-capture-, king-, and queen-like color switching. The implemented model makes these coefficients trainable and conditions their use on piece type through per-square gates. The learned gate vector can amplify within-color flow for bishop-occupied squares, cross-color flow for knight-occupied squares, and mixed behavior for pieces whose tactics naturally combine both.

The final classifier sees both pooled board features and parity-specific summaries, including dark/light means, dark/light maxima, parity sums, parity differences, same-color energy, cross-color energy, and piece-type gate diagnostics. The model remains board-only and does not consume CRTK, engine, or source metadata as input.

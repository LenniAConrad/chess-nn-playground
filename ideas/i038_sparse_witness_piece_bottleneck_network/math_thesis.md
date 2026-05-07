# Math Thesis

Sparse Witness-Piece Bottleneck Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-21_0713_tuesday_los_angeles_sparse_witness_bottleneck.md`.

The thesis is that a meaningful share of chess puzzle-likeness should be decidable from a small witness set of occupied piece-squares. Instead of giving the classifier the full board, the architecture learns a fixed-budget selector over occupied squares and classifies only from the selected pieces plus safe global state bits.

Let `P(x) in {0,1}^{12 x 8 x 8}` be the current-board piece planes and let `g(x)` be side-to-move, castling, and en-passant state extracted from `simple_18`. The occupied piece-square set is:

```text
T(x) = {(s_i, p_i)}_{i=1}^{n(x)}
```

where `n(x) <= 32` for ordinary chess positions. For a budget `K`, the witness compression family is:

```text
W_K = { (S, x_S, g(x)) : S subset T(x), |S| <= K }.
```

The selector `q_theta(S | x)` chooses occupied witnesses and the predictor `f_phi` maps the censored board to puzzle logits. The implemented objective is the ordinary supervised puzzle-binary loss through this bottleneck:

```text
min_{theta, phi} E[ BCEWithLogits(f_phi(S_theta(x), x_{S_theta(x)}, g(x)), y) ]
```

where `S_theta(x)` is a hard top-k witness set of size `min(K, n(x))`. During training, the top-k mask uses a straight-through Gumbel relaxation so gradients can reach the scorer; during evaluation, it is deterministic.

If there exists a selector `S*(x)` with `|S*(x)| <= K` such that `Y` is approximately independent of the rest of the board given `(S*(x), x_{S*(x)}, g(x))`, then restricting the classifier to `W_K` should preserve most label signal while reducing diffuse material, source, and full-board shortcuts. The representation support is bounded by:

```text
log |support(W_K)| <= log sum_{j=0}^K binom(32, j) + K log(12 * 64) + dim(g) * b_g
```

which is far smaller than the full board representation when `K << n(x)`.

The implementation does not prove that CRTK puzzle-likeness has such witnesses. It directly tests the falsifiable claim by making the learned hard witness subset the only current-piece information visible to the downstream classifier.

The source packet's first experiment maps fine labels `1` and `2` to positive and uses two logits. This repo idea contract maps only fine label `2` to puzzle and trains a single BCE logit by default. The sparse witness operator is unchanged; only the output head is adapted to the repo's puzzle-binary label contract.

# Math Thesis

Entropic Piece-Target Transport Bottleneck

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-21_0507_tuesday_los_angeles_transport_bottleneck.md`.

Working thesis: a chess position is puzzle-like when side-to-move force can be geometrically coupled to high-value or king-safety target anchors by a low-cost, high-contrast transport plan that is not reducible to material counts, one-ply move bags, or static attack incidence.

For a board tensor `X in {0,1}^{18 x 8 x 8}` we extract the 12 piece planes and the side-to-move scalar, canonicalize colors and ranks so that the side-to-move occupies planes `0..5` of the canonical view, and define six masked source measures `mu_g(X) in Delta_{64}` (us/them sliders, leapers, pawns) by softmaxing per-group salience logits over occupied squares. We define six deterministic target anchors `nu_a(X) in Delta_{64}` for `them_king_zone`, `them_value`, `us_king_zone`, `us_value`, `us_promotion_rank`, and `them_promotion_rank` using empty-board geometry and fixed nominal piece values `Q=9, R=5, B/N=3, P=1`.

The chess-metric cost bank holds seven fixed, normalized empty-board distance matrices `D_r in R_+^{64 x 64}` for `king`, `manhattan`, `rook`, `bishop` (with opposite-color cap), shortest `knight` BFS distance, and pawn-oriented forward-and-file metrics for both colors. Per-group cost matrices are learned mixtures `C_g = softplus(beta_{g,0} + sum_r softplus(alpha_{g,r}) D_r) in R_+^{64 x 64}`. For each pair `(g, a)` in the fixed 12-pair list we compute the entropic transport plan

```
T_epsilon(mu_g, nu_a; C_g) = min_{Pi >= 0, Pi 1 = mu_g, Pi^T 1 = nu_a} <Pi, C_g> + epsilon sum_{ij} Pi_{ij} (log Pi_{ij} - 1).
```

Proposition (product-plan gap). The independent product plan `Pi_prod = mu otimes nu` is feasible for the unregularized transport polytope, so `T_0(mu, nu; C) <= <mu otimes nu, C>` and `G_0(mu, nu; C) = <mu otimes nu, C> - T_0(mu, nu; C) >= 0`. The gap is zero exactly when the marginals already explain the coupled cost; it is positive when geometric mass matching reduces expected cost beyond the marginals.

Proposition (no engine input). All marginals, target anchors, distance matrices, and transport plans are deterministic functions of the current `simple_18` board tensor. No Stockfish scores, principal variations, node counts, verification metadata, source labels, proposed labels, or split identity enter the model.

Hypothesis: the binary puzzle label has useful conditional dependence on the low-dimensional collection of entropic transport observables `tau(X) = { T_epsilon, prod_cost, gap, plan_entropy, sharpness }_{(g,a) in Pairs}`, even after conditioning on a small CNN stem of the same `simple_18` tensor. The central falsifying ablation is the `Pi := mu otimes nu` product-coupling replacement, which preserves marginals, anchors, costs, and downstream feature dimensionality but removes geometric coupling.

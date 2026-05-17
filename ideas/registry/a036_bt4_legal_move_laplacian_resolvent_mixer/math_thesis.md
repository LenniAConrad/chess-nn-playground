# Mathematical Thesis

- Mathematical motivation: The BT4-style residual tower from
  `lc0_bt4_classifier` mixes spatially with a pair of 3x3 convs. This
  idea holds the tower shell (stem -> N residual + SqueezeExcite blocks
  -> value head) fixed and replaces only the per-block spatial-mixing
  operator with the `legal_move_laplacian_resolvent` primitive
  (LM-LPP) from `p031_legal_move_laplacian_resolvent`. Source primitive
  math: `ideas/registry/p031_legal_move_laplacian_resolvent/math_thesis.md`.

  Per BT4 block, given the per-square feature tensor `X in R^{B x 64 x C}`
  (the `(B, C, 8, 8)` input flattened over the 8x8 grid), the operator
  computes a truncated Neumann-series resolvent over a signed graph
  Laplacian `L = D - W`:

  ```
  W(x)        = diag(w(x)) @ A_static          # (B, 64, 64)
  D(x)        = diag(rowsum W(x))
  L(x)        = (D(x) - W(x)) / max(rowsum, 1) # spectral safety scale
  alpha       = alpha_init * tanh(alpha_logit)
  Y           = sum_{k=0..K} alpha^k * L(x)^k * X
  output      = Theta @ Y
  ```

  with `A_static in {0, 1}^{64 x 64}` the position-independent union of
  knight, king, and the four sliding-piece alignments (rank, file, two
  diagonals); per-square scalar `w(x) = softplus(MLP(X))` playing the
  role of the per-piece weight `w(piece(i, x))` in the thesis; and
  `Theta in R^{C x C}` a learned no-bias mixing matrix. The
  load-bearing idea is that `K >= 2` Neumann terms capture multi-hop
  tactical influence (X-rays, batteries, discovered attacks) in a
  single operator application -- standard attention sees only one hop
  per layer.

- Assumptions:
  1. The `legal_move_laplacian_resolvent` primitive is well-defined as
     a shape-preserving operator `(B, C, 8, 8) -> (B, C, 8, 8)` under
     the `chess_nn_playground.models.architecture.bt4_mixers._base.Mixer`
     contract.
  2. The BT4 block wrapper (`mixer -> SqueezeExcite -> +residual ->
     ReLU`) is identical across all `a###_bt4_*_mixer` ideas and
     across the `conv` and `attention` baselines.
  3. The optimizer protocol, data contract (`simple_18`,
     `puzzle_binary`), and training budget are identical across all
     `a###` and baseline runs, so the only experimental variable is
     the mixer.
  4. The source primitive (p031) built a *blocker-resolved,
     piece-typed* adjacency `A(x)` directly from the `simple_18`
     piece planes. The mixer only receives the `(B, C, 8, 8)`
     feature map -- discrete piece planes are not available -- so the
     blocker-resolved, piece-typed adjacency cannot be computed
     here. Honest compromise: the adjacency is the position-
     independent union of knight, king, and the four sliding-piece
     alignments (no occupancy blocking), and content dependence is
     restored via the learned per-square softplus weight `w(x)` that
     plays the role of the per-piece weight in the thesis. The
     Laplacian, the Neumann expansion, the `tanh`-bounded `alpha`,
     and the `Theta` mixing matrix are reproduced exactly.
  5. The truncated Neumann partial sum is bounded for any finite K
     because `L` is scaled by `max(rowsum, 1)` and `alpha` is bounded
     by `alpha_init = 0.25` via the `tanh` envelope, so
     `|alpha * lambda_max(L)| <= alpha_init` after the row-degree
     scaling.

- Claimed advantage: If the `legal_move_laplacian_resolvent`
  primitive carries a load-bearing multi-hop tactical signal beyond
  what conv and dense attention provide, dropping it into the BT4
  block must lift held-out PR AUC (aggregate or on a slice that
  depends on multi-hop tactical influence -- pin / skewer /
  discovered-attack motifs, batteries along files / diagonals / ranks,
  X-ray attacks on the king) versus the two baselines under the same
  tower, optimizer, and data. This is a controlled architecture-level
  test of "is `legal_move_laplacian_resolvent` a better spatial mixer
  than conv or attention inside a fixed BT4 tower shell?", not a new
  primitive claim. The per-block cost is dominated by `K` dense
  `(B, 64, 64) x (B, 64, C)` matmuls (`O(K * 64^2 * C)` per sample)
  plus the per-square content-weight MLP (`O(C^2)` per square) and the
  final `Theta` projection (`O(C^2 * 64)` per sample). The operator is
  asymptotically *comparable* to the dense `attention` baseline
  (`O(64 * 64 * C)` token-pair matmul) for `K ~ 1` and more expensive
  for the default `K = 4`, but substantially more expressive: the
  static chess-rule reach matrix is the inductive prior, and the
  Neumann partial sum is the multi-hop closure of that prior weighted
  by the learned per-square content weight.

- Proof sketch: This is an empirical study, not a theorem. The well-
  definedness of the mixer is enforced at construction time by the
  BT4 block's shape check (raises if `mixer(x).shape != x.shape`).
  The primitive-level math for `legal_move_laplacian_resolvent`
  itself (the Neumann partial sum is bounded for any finite K
  because `L` is rescaled by its max row-degree and `alpha` is
  bounded by the `tanh` envelope; with `alpha = 0` the operator
  reduces to `Theta * X` and the mixer degenerates to a per-square
  linear projection) is proven in the source primitive's math thesis
  and falsified by its own ablation grid (`k1_gat_rebrand`,
  `uniform_piece_weights`, `shuffle_adjacency`, `zero_alpha`). This
  folder inherits that math and tests whether the resulting operator,
  used as a token mixer rather than as an i193-additive head,
  transfers its signal through the BT4 tower.

- What is actually proven: The mixer is shape-preserving on
  `(B, C, 8, 8)` inputs and integrates with the BT4 block via the
  unified mixer registry. A forward + backward smoke test guards the
  mixer at registration time. The truncated Neumann partial sum
  `Y = sum_{k=0..K} alpha^k L^k X` is bounded because `L` is rescaled
  by `max(rowsum, 1)` (so `lambda_max(L) <= 2` for the signed
  Laplacian after rescaling) and `|alpha| <= alpha_init = 0.25` via
  the `tanh` envelope. With `K = 1` the operator reduces to
  `(I + alpha L) X`, which is exactly a single-hop legal-mask GAT
  weighted by the learned per-square scalar `w(x)` -- the
  `k1_gat_rebrand` falsifier from the source primitive. With
  `alpha = 0` the operator reduces to `Theta * X`, a per-square
  linear projection (the `zero_alpha` falsifier).

- What is only hypothesized: That replacing the conv mixer with the
  `legal_move_laplacian_resolvent` mixer lifts PR AUC on at least one
  CRTK slice (most likely multi-hop tactical slices: pin / skewer /
  discovered-attack motifs, batteries on files / diagonals, X-ray
  attacks on the king, queen-on-open-line) without regressing
  aggregate PR AUC by more than the matched-baseline tolerance. The
  hypothesis also covers the higher `crtk_difficulty` band where the
  trunk's local-receptive-field stack and the dense `attention`
  baseline are both likely to be insufficient (the conv stack misses
  multi-hop tactical chains; dense attention has no chess-move-graph
  prior, so it must rediscover the legal-move geometry and its
  multi-hop closure from data).

- Failure cases:
  - The trunk's stem conv plus the surrounding residual + SE blocks
    already encode multi-hop spatial context densely enough (after
    `N` blocks the effective receptive field already covers the 8x8
    board) that adding a single legal-move-Laplacian resolvent layer
    per block buys no marginal signal; the `conv` baseline matches
    the variant within noise.
  - The dense `attention` baseline matches or beats the LM-LPP
    mixer; all-pairs attention can in principle express any legal-
    move adjacency pattern and the explicit chess-rule legal-move
    prior is decorative at the BT4 tower's capacity.
  - The static legal-move adjacency, without occupancy-based blocker
    resolution, over-connects sliding-piece rays (rooks "see"
    through their own piece on the same file). The mixer cannot
    learn to mask occupied squares from the channel features alone
    if the trunk has not encoded piece occupancy in early blocks;
    the resolvent then reduces to multi-hop diffusion over the raw
    chess-rule reach geometry. `uniform_piece_weights` (A2 below)
    should then close the gap.
  - `K = 1` (the single-hop GAT rebrand) matches the unablated
    operator: the multi-hop Neumann expansion is decorative inside
    the BT4 tower's residual stack, since the surrounding blocks
    already chain single-hop spatial mixing. **Primary falsifier.**
  - `alpha = 0` matches the unablated operator: the resolvent
    expansion is decorative and the operator reduces to a per-
    square linear projection `Theta @ X`. The mixer degenerates to
    a 1x1 conv equivalent and the comparison against a per-square
    linear mixer (or a trivially weighted residual) should match.
  - `shuffle_adjacency` (in-batch permutation of the legal-move
    graph) matches the unablated operator: the legal-move geometry
    carries no signal in this trunk, and the resolvent's wins (if
    any) come from the per-square content-weight MLP alone, not
    from the chess-rule reach prior.
  - SqueezeExcite + residual + ReLU absorbs most of the mixer's
    contribution if the routed-token output magnitude is small
    relative to the residual stream; report per-block routed
    output norm statistics alongside the headline number.

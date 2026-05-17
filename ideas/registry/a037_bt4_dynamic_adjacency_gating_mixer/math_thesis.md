# Mathematical Thesis

- Mathematical motivation: The BT4-style residual tower from
  `lc0_bt4_classifier` mixes spatially with a pair of 3x3 convs. This
  idea holds the tower shell (stem -> N residual + SqueezeExcite blocks
  -> value head) fixed and replaces only the per-block spatial-mixing
  operator with the `dynamic_adjacency_gating` primitive (DAG) from
  `p032_dynamic_adjacency_gating`. Source primitive math:
  `ideas/registry/p032_dynamic_adjacency_gating/math_thesis.md`.

  Per BT4 block, given the per-square feature tensor `X in R^{B x 64 x C}`
  (the `(B, C, 8, 8)` input flattened over the 8x8 grid), the operator
  decomposes the chess move geometry into per-move-type binary masks
  `M_t in {0, 1}^{64 x 64}` for `t in {KNIGHT, KING, RANK, FILE, DIAG,
  ANTIDIAG}`, applies one independent linear projection `W_t: R^C ->
  R^C` per type, gates each type's contribution with a content-
  dependent per-square sigmoid `g_t(X_i) = sigmoid(Linear(C, T) X_i)`,
  and sums:

  ```
  Z       = LayerNorm(X)                              # (B, 64, C)
  g(Z)    = sigmoid(Linear_gate(Z))                   # (B, 64, T)
  Y_t     = M_t @ (W_t Z)                             # (B, 64, C)
  Y       = sum_t  g(Z)[:, :, t:t+1] * Y_t            # (B, 64, C)
  output  = Linear_out(Y)                             # (B, 64, C)
  ```

  Each `Y_t` is exactly the source primitive's defining equation
  `y = (G ⊙ Wx)` per move-type slot: the binary mask `M_t` is the hard
  topological constraint, and the gradient of an illegal-edge cell is
  zero by construction. The load-bearing claim is that a *per-move-
  type* kernel can specialise on the move-type class that drives a
  given position (open files vs diagonal pin lattice vs knight
  outpost), while a single shared kernel must average over move types.

- Assumptions:
  1. The `dynamic_adjacency_gating` primitive is well-defined as a
     shape-preserving operator `(B, C, 8, 8) -> (B, C, 8, 8)` under
     the `chess_nn_playground.models.architecture.bt4_mixers._base.Mixer`
     contract.
  2. The BT4 block wrapper (`mixer -> SqueezeExcite -> +residual ->
     ReLU`) is identical across all `a###_bt4_*_mixer` ideas and
     across the `conv` and `attention` baselines.
  3. The optimizer protocol, data contract (`simple_18`,
     `puzzle_binary`), and training budget are identical across all
     `a###` and baseline runs, so the only experimental variable is
     the mixer.
  4. The source primitive (p032) built a position-specific
     blocker-resolved binary adjacency `A(x)` from `simple_18` piece
     planes and intersected it with per-move-type masks. The mixer
     only receives the `(B, C, 8, 8)` feature map -- discrete piece
     planes and blocker occupancy are not available -- so the
     position-specific adjacency cannot be computed here. Honest
     compromise: the per-type adjacency is the position-independent
     static chess-rule reach `M_t in {0, 1}^{64 x 64}` (knight, king,
     and the four sliding-piece alignments). Content dependence is
     restored via the per-square per-type sigmoid gate
     `g_t(X_i) = sigmoid(Linear(C, T) X_i)_t`, mirroring the source
     primitive's `G(x) ⊙ Wx` gating form. The hard binary mask, the
     per-type weight specialisation, and the `out_proj` mixing matrix
     are reproduced exactly. The pawn_push and pawn_capture move
     types from the source primitive (which require side-to-move and
     piece-type discrimination from discrete planes) are dropped;
     `T = 6` here vs `T = 8` in the source primitive's head form.
  5. The per-square per-type gate is bounded in `(0, 1)` by the
     sigmoid, so the masked sum `Y` is bounded whenever `Z` is
     bounded; the LayerNorm on `X` keeps `Z` bounded across blocks
     so the mixer is numerically stable through the residual stack.

- Claimed advantage: If the `dynamic_adjacency_gating` primitive
  carries a load-bearing per-move-type specialisation signal beyond
  what conv and dense attention provide, dropping it into the BT4
  block must lift held-out PR AUC (aggregate or on a slice that
  depends on a single move-type class dominating the position --
  open-file / rook-on-open-file motifs, long-diagonal /
  bishop-pair-on-long-diagonal motifs, knight-outpost / knight-fork
  motifs) versus the two baselines under the same tower, optimizer,
  and data. This is a controlled architecture-level test of "is
  `dynamic_adjacency_gating` a better spatial mixer than conv or
  attention inside a fixed BT4 tower shell?", not a new primitive
  claim. The per-block cost is `T = 6` dense `(B, 64, 64) x
  (B, 64, C)` matmuls (`O(T * 64^2 * C)` per sample, parallel
  across `t`) plus the `T + 2` linear projections (`O((T + 2) *
  C^2 * 64)` per sample). The operator is roughly `T = 6x` more
  expensive than dense attention (`O(64^2 * C)` token-pair matmul
  plus `O(C^2)` projections) and substantially more expensive than
  the conv baseline's two 3x3 convs, but the `T` matmuls are
  parallel across types (one batched bmm per type, or a single
  batched-einsum implementation across all types simultaneously).

- Proof sketch: This is an empirical study, not a theorem. The well-
  definedness of the mixer is enforced at construction time by the
  BT4 block's shape check (raises if `mixer(x).shape != x.shape`).
  The primitive-level math for `dynamic_adjacency_gating` itself
  (the masked per-type aggregation is exactly the source primitive's
  defining equation `y = (G ⊙ Wx)` per move-type slot; the sum of
  the type slots equals applying a single shared kernel `W` to the
  union of move-type adjacencies when all `W_t = W` -- the
  `single_move_type` collapse from the source primitive's
  falsification grid) is proven in the source primitive's math
  thesis. This folder inherits that math and tests whether the
  resulting operator, used as a token mixer rather than as an
  i193-additive head, transfers its signal through the BT4 tower.

- What is actually proven: The mixer is shape-preserving on
  `(B, C, 8, 8)` inputs and integrates with the BT4 block via the
  unified mixer registry. A forward + backward smoke test guards the
  mixer at registration time. The per-type masked aggregation is
  exact: each `Y_t = M_t @ (W_t Z)` is the source primitive's
  `y = (G ⊙ Wx)` form with binary `G = M_t` and `W = W_t`. The
  per-square per-type sigmoid gate `g_t(X_i) in (0, 1)` is the
  content-dependent counterpart of the position-specific binary
  edge weight in the source primitive; with `g_t(X_i) = 1` for all
  `t, i` the operator reduces to a fixed per-move-type linear
  combination (the `uniform_gate` falsifier below). With `M_t`
  replaced by the all-ones (minus identity) adjacency for every `t`,
  the operator reduces to a generic content-gated mixture of `T`
  shared linear maps -- the `uniform_adjacency` falsifier from the
  source primitive.

- What is only hypothesized: That replacing the conv mixer with the
  `dynamic_adjacency_gating` mixer lifts PR AUC on at least one CRTK
  slice (most likely single-move-type-dominated tactical slices:
  open-file / rook-on-open-file motifs, long-diagonal / bishop-pair-
  on-long-diagonal motifs, knight-outpost / knight-fork motifs,
  king-safety / opposing-king motifs) without regressing aggregate
  PR AUC by more than the matched-baseline tolerance. The hypothesis
  also covers positions where one move-type class drives the
  tactic, since a shared kernel must average that class's signal
  with five other types it does not need.

- Failure cases:
  - The trunk's stem conv plus the surrounding residual + SE blocks
    already encode per-move-type spatial context densely enough that
    adding an explicit per-type decomposition per block buys no
    marginal signal; the `conv` baseline matches the variant within
    noise.
  - The dense `attention` baseline matches or beats the DAG mixer;
    all-pairs attention can in principle express any per-type mask
    pattern and the explicit chess-rule per-type prior is decorative
    at the BT4 tower's capacity.
  - The static per-type adjacency, without occupancy-based blocker
    resolution, over-connects sliding-piece rays (rooks "see"
    through their own piece on the same file). The mixer cannot
    learn to mask occupied squares from the channel features alone
    if the trunk has not encoded piece occupancy in early blocks;
    the per-type sigmoid gate then reduces to a generic content
    weighting over an over-connected reach geometry.
  - `single_move_type` (collapse all `W_t` to one shared projection)
    matches the unablated operator: the per-move-type kernel
    specialisation is decorative, the load-bearing factor is the
    chess-rule reach prior itself. **Primary falsifier.**
  - `uniform_gate` (force `g_t = 1` everywhere) matches the
    unablated operator: the per-square per-type gating is
    decorative and the operator reduces to a fixed per-type linear
    combination, equivalent to a 1x1 conv after the static reach
    aggregation.
  - `uniform_adjacency` (replace `M_t` with all-ones for every `t`)
    matches the unablated operator: the chess-rule per-type
    geometry is decorative and the operator collapses to a
    content-gated mixture of `T` shared linear maps over the full
    64-token bag.
  - `shuffle_adjacency` (in-batch permutation of all `M_t`) matches
    the unablated operator: the chess-rule reach geometry carries
    no signal in this trunk, and the wins (if any) come from the
    per-square per-type gate alone.
  - SqueezeExcite + residual + ReLU absorbs most of the mixer's
    contribution if the routed-token output magnitude is small
    relative to the residual stream; report per-block routed
    output norm statistics alongside the headline number.

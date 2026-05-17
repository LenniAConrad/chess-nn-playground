# Mathematical Thesis

- Mathematical motivation: The BT4-style residual tower from
  `lc0_bt4_classifier` mixes spatially with a pair of 3x3 convs. This
  idea holds the tower shell (stem -> N residual + SqueezeExcite blocks
  -> value head) fixed and replaces only the per-block spatial-mixing
  operator with the `move_kernel_operator` primitive (MKO) from
  `p033_move_kernel_operator`. Source primitive math:
  `ideas/registry/p033_move_kernel_operator/math_thesis.md`.

  Per BT4 block, given the per-square feature tensor `X in R^{B x 64 x C}`
  (the `(B, C, 8, 8)` input flattened over the 8x8 grid), the operator
  decomposes the chess move geometry into per-move-type binary reach
  masks `M_t in {0, 1}^{64 x 64}` for `t in {KNIGHT, RANK, FILE, DIAG,
  ANTIDIAG, KING}`, applies one independent matrix-valued linear
  projection `W_t: R^C -> R^C` per type, sums the masked aggregations
  across types, and projects through a final mixing matrix:

  ```
  Z       = LayerNorm(X)                          # (B, 64, C)
  Y_t     = M_t @ (W_t Z)                         # (B, 64, C), one per t
  Y       = sum_t  Y_t                            # (B, 64, C)
  output  = Linear_out(Y)                         # (B, 64, C)
  ```

  Each `Y_t` is exactly the source primitive's defining equation
  `Y_t[i] = sum_{j : M_t[i, j] = 1} (W_t X)[j]`: the binary mask `M_t`
  is the hard topological constraint and the gradient of an illegal
  edge is zero by construction. The load-bearing claim is that *per-
  move-type matrix-valued weight sharing across squares* lets one
  `W_t` learn "what a `t`-neighbour contributes" once and apply it at
  every source square, while Conv2d weights, indexed by spatial offset,
  must relearn identical chess-rule behaviour at every square.

- Assumptions:
  1. The `move_kernel_operator` primitive is well-defined as a
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
  4. The source primitive (p033) is designed with *static*, occlusion-
     free reach masks -- a queen at a1 reaches every square on its
     rays regardless of occupancy. This is by design and matches the
     source primitive's framing; no blocker resolution is required.
     The masks are position-independent and can be registered as
     non-persistent buffers at construction time. No simple_18 piece
     planes are consulted by the mixer; the only adaptation versus
     the source primitive head is cosmetic -- the per-square seed
     feature `X` is the mixer's `C`-channel feature vector rather
     than a `Linear(13)` projection of piece planes.
  5. LayerNorm on the per-square feature vector before the per-type
     projections keeps the per-square feature norms stable across
     blocks; without it the per-type projection outputs can drift
     asymmetrically as the residual stream's magnitude changes
     through the tower.

- Claimed advantage: If the `move_kernel_operator` primitive carries
  a load-bearing per-move-type weight-sharing signal beyond what conv
  and dense attention provide, dropping it into the BT4 block must
  lift held-out PR AUC (aggregate or on a slice that depends on a
  single move-type class dominating the position -- open-file /
  rook-on-open-file motifs, long-diagonal / bishop-pair-on-long-
  diagonal motifs, knight-outpost / knight-fork motifs) versus the
  two baselines under the same tower, optimizer, and data. This is a
  controlled architecture-level test of "is `move_kernel_operator` a
  better spatial mixer than conv or attention inside a fixed BT4
  tower shell?", not a new primitive claim. The per-block cost is
  `T = 6` dense `(B, 64, 64) x (B, 64, C)` matmuls
  (`O(T * 64^2 * C)` per sample) plus the `T + 1` linear projections
  (`O((T + 1) * C^2 * 64)` per sample). The operator is roughly
  `T = 6x` more expensive than dense attention's `O(64^2 * C)` token-
  pair matmul plus `O(C^2)` projections and substantially more
  expensive than the conv baseline's two 3x3 convs, but the `T`
  matmuls are parallel across types (one batched bmm per type, or a
  single batched-einsum across all types simultaneously).

- Proof sketch: This is an empirical study, not a theorem. The well-
  definedness of the mixer is enforced at construction time by the
  BT4 block's shape check (raises if `mixer(x).shape != x.shape`).
  The primitive-level math for `move_kernel_operator` itself (the
  masked per-type aggregation is exactly the source primitive's
  defining equation; the sum of the type slots equals applying a
  single shared kernel `W` to the union of move-type adjacencies
  when all `W_t = W` -- the `shared_kernel` collapse from the source
  primitive's falsification grid; replacing each `W_t` by a scalar
  gain `w_t * I` reduces to a content-independent per-type linear
  combination -- the `scalar_per_type` collapse) is proven in the
  source primitive's math thesis. This folder inherits that math and
  tests whether the resulting operator, used as a token mixer rather
  than as an i193-additive head, transfers its signal through the
  BT4 tower.

- What is actually proven: The mixer is shape-preserving on
  `(B, C, 8, 8)` inputs and integrates with the BT4 block via the
  unified mixer registry. A forward + backward smoke test guards the
  mixer at registration time. The per-type masked aggregation is
  exact: each `Y_t = M_t @ (W_t Z)` is the source primitive's
  defining equation with binary mask `M_t` and per-type matrix
  projection `W_t`. With all `W_t = W` collapsed to one shared
  projection, the operator reduces to `Y = (sum_t M_t) @ (W Z)` --
  a single linear projection applied to the union of static reach
  geometries (the `shared_kernel` falsifier). With each `W_t`
  replaced by a scalar gain `w_t * I`, the operator reduces to a
  content-independent weighted sum of masked aggregations under a
  shared per-square feature (the `scalar_per_type` falsifier).

- What is only hypothesized: That replacing the conv mixer with the
  `move_kernel_operator` mixer lifts PR AUC on at least one CRTK
  slice (most likely single-move-type-dominated tactical slices:
  open-file / rook-on-open-file motifs, long-diagonal / bishop-pair-
  on-long-diagonal motifs, knight-outpost / knight-fork motifs,
  king-safety / opposing-king motifs) without regressing aggregate
  PR AUC by more than the matched-baseline tolerance. The hypothesis
  also covers positions where one move-type class drives the tactic,
  since a shared kernel must average that class's signal with five
  other types it does not need.

- Failure cases:
  - The trunk's stem conv plus the surrounding residual + SE blocks
    already encode per-move-type spatial context densely enough that
    adding an explicit per-type decomposition per block buys no
    marginal signal; the `conv` baseline matches the variant within
    noise.
  - The dense `attention` baseline matches or beats the MKO mixer;
    all-pairs attention can in principle express any per-type mask
    pattern and the explicit chess-rule per-type prior is decorative
    at the BT4 tower's capacity.
  - The static per-type adjacency, without occupancy-based blocker
    resolution, over-connects sliding-piece rays (rooks "see"
    through their own piece on the same file). The mixer cannot
    learn to mask occupied squares from the channel features alone
    if the trunk has not encoded piece occupancy in early blocks;
    the per-type projection then mixes signal from squares behind
    blockers indiscriminately.
  - `shared_kernel` (collapse all `W_t` to one shared projection)
    matches the unablated operator: the per-move-type matrix
    specialisation is decorative, the load-bearing factor is the
    chess-rule reach prior itself. **Primary falsifier.**
  - `scalar_per_type` (replace each `W_t` by `w_t * I`) matches the
    unablated operator: the matrix capacity per type is decorative
    and a per-type scalar gain on the masked sum suffices.
  - `shuffle_features` (in-batch permutation of the seed features so
    the per-square input is decoupled from the board) matches the
    unablated operator: the rule-derived per-square input carries no
    signal beyond what a per-type scalar mixture extracts.
  - SqueezeExcite + residual + ReLU absorbs most of the mixer's
    contribution if the routed-token output magnitude is small
    relative to the residual stream; report per-block routed
    output norm statistics alongside the headline number.

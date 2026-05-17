# Mathematical Thesis

- Mathematical motivation: The BT4-style residual tower from
  `lc0_bt4_classifier` mixes spatially with a pair of 3x3 convs. This
  idea holds the tower shell (stem -> N residual + SqueezeExcite blocks
  -> value head) fixed and replaces only the per-block spatial-mixing
  operator with the `incremental_delta_linear_head` primitive from
  `p025_incremental_delta_linear_head`. Source primitive math:
  `ideas/registry/p025_incremental_delta_linear_head/math_thesis.md`.
  The Incremental Delta-Linear (IDL) operator is the differentiable
  lift of the NNUE accumulator. For per-square tokens `x_s in R^C`
  with a learned per-square linear map `W_s : R^C -> R^d`, the
  primitive forms the sparse-style linear sum

  ```
  S(x) = sum_s W_s x_s + b_s    (linear in the per-square input)
  ```

  Because `S` is linear in the per-square contribution, a chess move
  that changes ``k`` squares (typically 2-4) only requires re-summing
  ``k`` rows: the per-move incremental update is ``O(k)``. The
  global summary `S in R^d` is then broadcast back to every square
  and fused per-square with that square's own token to satisfy the
  `(B, C, 8, 8) -> (B, C, 8, 8)` mixer contract.

- Assumptions:
  1. The `incremental_delta_linear_head` primitive is well-defined as
     a shape-preserving operator
     `(B, C, 8, 8) -> (B, C, 8, 8)` under the
     `chess_nn_playground.models.architecture.bt4_mixers._base.Mixer`
     contract.
  2. The BT4 block wrapper (`mixer -> SqueezeExcite -> +residual ->
     ReLU`) is identical across all `a###_bt4_*_mixer` ideas and
     across the `conv` and `attention` baselines.
  3. The optimizer protocol, data contract (`simple_18`,
     `puzzle_binary`), and training budget are identical across all
     `a###` and baseline runs, so the only experimental variable is
     the mixer.
  4. The source primitive reads piece-plane indicators directly off
     `simple_18` and indexes a per-(piece-type, square) embedding
     table `E in R^{12 x 64 x d}`. The mixer cannot read piece
     planes (it sees only a generic `(B, C, 8, 8)` channel tensor),
     so the per-(piece-type, square) embedding axis is absorbed into
     a per-square linear map `W_s : R^C -> R^d` of the channel
     vector. The per-square structure is preserved; the per-piece-
     type factorisation is replaced by the linear map of the soft
     channel descriptor.

- Claimed advantage: If the `incremental_delta_linear_head` primitive
  carries a load-bearing global linear-additive accumulator signal
  that conv and attention do not, dropping it into the BT4 block
  must lift held-out PR AUC (aggregate or on a slice that depends
  on a stable per-(piece-type, square) statistic -- e.g. simple
  material-count puzzles, rook / back-rank squares, and the lower-
  to-mid `crtk_difficulty` band where the count-style accumulator
  is most informative) versus the two baselines under the same
  tower, optimizer, and data. This is a controlled architecture-
  level test of "is incremental_delta_linear_head a better spatial
  mixer than conv or attention inside a fixed BT4 tower shell?",
  not a new primitive claim. The IDL sum is `O(64 C d)` per block
  for the projection plus `O(C d)` for the broadcast-back fusion;
  it is cheaper per block than a 64x64 attention map and
  comparable to a single 3x3 conv at matched widths.

- Proof sketch: This is an empirical study, not a theorem. The well-
  definedness of the mixer is enforced at construction time by the
  BT4 block's shape check (raises if `mixer(x).shape != x.shape`).
  The primitive-level math for `incremental_delta_linear_head` itself
  (linearity of `S(x)` in the per-square indicator, gradient of the
  embedding, `O(k)` per-move incremental update) is proven in the
  source primitive's math thesis and falsified by its own ablation
  grid (`shuffle_squares`, `permute_piece_types`, `zero_accumulator`).
  This folder inherits that math and tests whether the resulting
  operator, used as a token mixer rather than as an additive head,
  transfers its signal through the BT4 tower.

- What is actually proven: The mixer is shape-preserving on
  `(B, C, 8, 8)` inputs and integrates with the BT4 block via the
  unified mixer registry. A forward + backward smoke test guards
  the mixer at registration time. The per-square linear map plus
  global sum
  `S = sum_s W_s x_s + b_s` is implemented exactly as a single
  `einsum('bsc,sdc->bsd', tokens, square_weight)` followed by a
  per-square bias and reduction; the result is then `LayerNorm`-
  normalised before broadcast-back fusion. Linearity in the per-
  square contribution -- the property that gives the `O(k)` move
  update -- is preserved up to the `LayerNorm` (which is a fixed
  affine transform of `S` and so does not break the additive
  structure).

- What is only hypothesized: That replacing the conv mixer with the
  `incremental_delta_linear_head` mixer lifts PR AUC on at least one
  CRTK slice (most likely slices where a per-(piece-type, square)
  count-style statistic is load-bearing -- simple material puzzles,
  back-rank slices, rook-square slices -- and the lower-to-mid
  `crtk_difficulty` band) without regressing aggregate PR AUC by
  more than the matched-baseline tolerance.

- Failure cases:
  - The trunk's stem conv plus the surrounding residual + SE blocks
    already encode the per-(piece-type, square) statistic densely
    enough that the global linear accumulator adds no marginal
    signal; the `conv` baseline matches the variant within noise.
  - The per-square linear map `W_s` over-fits 64*d*C free
    parameters on the `simple_18` budget; the in-mixer
    `shuffle_squares`-style ablation matches this idea on its
    declared target slice.
  - The broadcast-back fusion `y_s = MLP([x_s; S])` carries only
    the per-square own-token signal `x_s` and ignores the
    broadcast accumulator `S`, meaning the IDL sum is decorative;
    the `zero_accumulator` ablation (hold `S = 0`) closes the gap.
  - SqueezeExcite + residual + ReLU absorbs most of the mixer's
    contribution if `||S||` is small; report per-block `||S||`
    statistics alongside the headline number.

# Mathematical Thesis

- Mathematical motivation: The BT4-style residual tower from
  `lc0_bt4_classifier` mixes spatially with a pair of 3x3 convs. This
  idea holds the tower shell (stem -> N residual + SqueezeExcite blocks
  -> value head) fixed and replaces only the per-block spatial-mixing
  operator with the `incremental_latent_accumulator_head` primitive
  (ILA) from `p028_incremental_latent_accumulator_head`. Source
  primitive math:
  `ideas/registry/p028_incremental_latent_accumulator_head/math_thesis.md`.
  The Incremental Latent Accumulator (ILA) operator factorises into a
  global accumulator (sum of per-(piece-type, square) embeddings) and
  a king-anchored accumulator (the same accumulation indexed by the
  own-king square), followed by a small non-linear lift `phi`:

  ```
  h_global = sum_{(t, s) : x_{t, s} = 1} G_{t, s}
  h_king   = sum_{(t, s) : x_{t, s} = 1} K_{k, t, s}
  z        = LayerNorm(phi([h_global, h_king]))
  ```

  with `G in R^{12 x 64 x global_dim}` and
  `K in R^{65 x 12 x 64 x king_dim}` learned embedding tables and `phi`
  a `LayerNorm -> Linear -> GELU -> Linear` MLP. The load-bearing
  ideas are (1) a permutation-structured accumulation over the board
  and (2) *anchoring* that accumulation on a special "context" square
  (the king) so the same content yields different features depending
  on where the anchor sits.

- Assumptions:
  1. The `incremental_latent_accumulator_head` primitive is well-
     defined as a shape-preserving operator
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
  4. The source primitive (p028) reads the rule-exact own-king square
     and the per-piece-type indicator tensor from the `simple_18`
     piece planes to build `h_global` and `h_king`. The mixer cannot
     read piece planes (it sees only a generic `(B, C, 8, 8)` channel
     tensor) so the king-square anchor is replaced by a *learned*
     soft-argmax saliency square computed from a 1x1 conv over the
     channel features, and the per-(piece-type, square) embedding
     table is replaced by a `Linear(C -> latent_dim)` over the
     channel features plus a learned per-square bias
     `global_square in R^{64 x latent_dim}`. The accumulate-then-
     anchor-then-broadcast structure -- the load-bearing idea -- is
     preserved; only the rule-exact king-square / piece-type-indicator
     readout is replaced by a learned soft-argmax over learned
     channel features.

- Claimed advantage: If the
  `incremental_latent_accumulator_head` primitive carries a load-
  bearing king-anchored permutation-symmetric accumulator signal
  beyond what conv and dense attention provide, dropping it into the
  BT4 block must lift held-out PR AUC (aggregate or on a slice that
  depends on king context / king-zone activity / endgame king
  position, or that depends on global piece-count summaries
  invariant to where the pieces sit -- e.g. material-imbalance
  motifs, king-safety-driven tactics, late-endgame king-and-pawn
  motifs) versus the two baselines under the same tower, optimizer,
  and data. This is a controlled architecture-level test of "is
  `incremental_latent_accumulator_head` a better spatial mixer than
  conv or attention inside a fixed BT4 tower shell?", not a new
  primitive claim. The per-block cost is `O(C * latent_dim + 64 *
  latent_dim)` for the two `Linear(C -> latent_dim)` projections plus
  the per-square bias add and the board-wide sum, plus an `O(C)`
  saliency `Conv2d(C -> 1)` for the anchor and an `O((2 * latent_dim
  + C) * C)` `phi` MLP back to `C` channels. The board-wide sum is
  `O(64 * latent_dim)` per sample; no `O(64^2)` token-pair matmul is
  required, so the operator is asymptotically *cheaper per block*
  than the dense `attention` baseline.

- Proof sketch: This is an empirical study, not a theorem. The well-
  definedness of the mixer is enforced at construction time by the
  BT4 block's shape check (raises if `mixer(x).shape != x.shape`).
  The primitive-level math for
  `incremental_latent_accumulator_head` itself (sum-pool over the 64
  tokens is permutation-invariant by construction; soft-argmax
  saliency is differentiable; the `phi` MLP `LayerNorm -> Linear ->
  GELU -> Linear` lifts the concatenation without changing the
  output rank) is proven in the source primitive's math thesis and
  falsified by its own ablation grid (`zero_global_accumulator`,
  `zero_king_accumulator`, `linear_only`, `shuffle_square_order`).
  This folder inherits that math and tests whether the resulting
  operator, used as a token mixer rather than as a pooled additive
  head, transfers its signal through the BT4 tower.

- What is actually proven: The mixer is shape-preserving on
  `(B, C, 8, 8)` inputs and integrates with the BT4 block via the
  unified mixer registry. A forward + backward smoke test guards
  the mixer at registration time. The global accumulator
  `h_global = sum_s [Linear(C -> latent_dim)(tokens) +
  global_square]` is permutation-equivariant only over the per-square
  bias structure (each square keeps its own bias row), so a column
  shuffle of `global_square` is detectable -- this matches the source
  primitive's `shuffle_square_order` falsifier. The king-anchored
  accumulator `h_king = sum_s Linear(C -> latent_dim)(tokens) +
  einsum(anchor_w, king_anchor_table)` re-uses the soft-argmax anchor
  `anchor_w = softmax(Conv2d(C -> 1)(x))` to pick a `latent_dim`-
  vector from a `(64, latent_dim)` table. The `phi` MLP
  `LayerNorm -> Linear -> GELU -> Linear` is the same lift as in the
  source primitive. With `phi` collapsed to identity and the two
  accumulators set to zero, the mixer reduces to a per-token linear
  projection back to `C` channels (i.e., a 1x1 conv), and the BT4
  block degenerates to a standard SqueezeExcite + residual block --
  this matches the source primitive's `linear_only` ablation.

- What is only hypothesized: That replacing the conv mixer with the
  `incremental_latent_accumulator_head` mixer lifts PR AUC on at
  least one CRTK slice (most likely king-safety / king-zone activity
  / endgame king-position slices, material-imbalance slices, and
  global-piece-count slices where permutation-invariant
  accumulation matters more than local conv windows) without
  regressing aggregate PR AUC by more than the matched-baseline
  tolerance. The hypothesis also covers the higher `crtk_difficulty`
  band where the trunk's local-receptive-field stack and the dense
  `attention` baseline are both likely to be insufficient.

- Failure cases:
  - The trunk's stem conv plus the surrounding residual + SE blocks
    already encode king context and global material features densely
    enough that broadcasting a pooled king-anchored latent back to
    every square adds no marginal signal; the `conv` baseline
    matches the variant within noise.
  - The dense `attention` baseline matches or beats the accumulator
    mixer; the permutation-structured sum-pool is captured by all-
    pairs attention up to a constant factor.
  - The soft-argmax saliency anchor collapses to a near-uniform
    distribution over the 64 squares, so the anchor row is the
    average of all rows of `king_anchor_table` and the king-anchor
    structure carries no information. The `zero_king_accumulator`-
    style ablation (drop the king stream entirely) should then close
    the gap.
  - The non-linear `phi` lift is the only load-bearing component;
    setting `phi = identity` (the `linear_only` ablation) closes
    the gap, so the mixer is just a non-linear MLP over a pooled
    summary and the king-anchor structure is decorative.
  - The `king_anchor_table` is `(64, latent_dim)` (rather than the
    source primitive's `(65, 12, 64, king_dim)`); the mixer cannot
    distinguish piece-type sources of the anchor and so loses the
    HalfKA refinement that motivated the primitive. If A3 (the
    primitive used as a pooled head with the rule-exact king-square
    + piece-type indicator) strictly beats this mixer, the
    HalfKA-style indexing is what made the primitive work, not the
    pooled-accumulate-then-broadcast structure.
  - SqueezeExcite + residual + ReLU absorbs most of the mixer's
    contribution if the routed-token output magnitude is small
    relative to the residual stream; report per-block routed
    output norm statistics alongside the headline number.
  - The per-square sum over the global stream means very late-
    endgame positions (few pieces) collapse `h_global` to the
    same near-zero magnitude as the early-game pooled latent,
    so the mixer cannot distinguish material levels by `h_global`
    norm alone; this should show up as flat performance on the
    `crtk_phase: endgame` slice if the king-anchor stream cannot
    compensate.

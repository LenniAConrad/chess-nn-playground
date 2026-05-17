# Mathematical Thesis

- Mathematical motivation: The BT4-style residual tower from
  `lc0_bt4_classifier` mixes spatially with a pair of 3x3 convs. This
  idea holds the tower shell (stem -> N residual + SqueezeExcite blocks
  -> value head) fixed and replaces only the per-block spatial-mixing
  operator with the `event_delta_bilinear_accumulator` primitive from
  `p022_event_delta_bilinear_accumulator`. Source primitive math:
  `ideas/registry/p022_event_delta_bilinear_accumulator/math_thesis.md`.
  For per-square tokens `x_s` with two learned projections
  `U_s = W_U x_s in R^d` and `V_s = W_V x_s in R^d`, the primitive
  computes the first- and second-order sparse-set accumulator triple
  via the factorisation-machine identity:

  ```
  A = sum_s U_s in R^d
  B = sum_s V_s in R^d
  P = sum_s U_s (.) V_s in R^d
  Q = A (.) B - P                   # the pair term, FM identity
  ```

  so the would-be `O(64^2 d)` pair sum collapses to `O(64 d)`. The
  triple `[A; B; Q]` summarises all first-order and second-order
  Hadamard interactions across the 64 squares in a single permutation-
  invariant readout.

- Assumptions:
  1. The `event_delta_bilinear_accumulator` primitive is well-defined
     as a shape-preserving operator
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
  4. The mixer cannot read the piece planes directly (the BT4 block
     hands it a generic `(B, C, 8, 8)` channel tensor), so the
     occupancy indicator is derived inside the operator as
     `O_s = sigmoid(w . x_s + b)` from the per-square channel vector
     and used to mask `U_s` and `V_s`. The occupancy mask is therefore
     *learned* from features, but it is still generated *inside* the
     operator and never supplied externally, which preserves the
     source thesis's defining property (the FM-identity pair-term
     algebra is computed over a sparse active set).

- Claimed advantage: If the `event_delta_bilinear_accumulator`
  primitive carries a second-order pair-interaction signal across
  active pieces that conv and attention do not, dropping it into the
  BT4 block must lift held-out PR AUC (aggregate or on a slice that
  depends on multi-piece interactions, e.g. fork / discovered-attack /
  battery / pin / skewer / x-ray patterns and the upper
  `crtk_difficulty` tail where two-piece coordination dominates)
  versus the two baselines under the same tower, optimizer, and data.
  This is a controlled architecture-level test of "is
  event_delta_bilinear_accumulator a better spatial mixer than conv or
  attention inside a fixed BT4 tower shell?", not a new primitive
  claim. The FM-identity reduction is `O(64 d + C d)` per block where
  `d = bilinear_dim` and `C` is the channel width, so it is
  asymptotically cheaper than the `O(64^2 d)` naive pair sum and
  cheaper than a full 64x64 attention map.

- Proof sketch: This is an empirical study, not a theorem. The well-
  definedness of the mixer is enforced at construction time by the
  BT4 block's shape check (raises if `mixer(x).shape != x.shape`).
  The primitive-level math for `event_delta_bilinear_accumulator`
  itself (the FM-identity pair-term reduction) is proven in the
  source primitive's math thesis and falsified by its own ablation
  grid (`first_order_only`, `shuffle_pair_term`, `zero_delta`,
  `trunk_only`). This folder inherits that math and tests whether
  the resulting operator, used as a token mixer rather than as an
  additive head, transfers its signal through the BT4 tower.

- What is actually proven: The mixer is shape-preserving on
  `(B, C, 8, 8)` inputs and integrates with the BT4 block via the
  unified mixer registry. A forward + backward smoke test guards the
  mixer at registration time. The FM-identity algebra
  `Q = A (.) B - P` with `A = sum_s U_s`, `B = sum_s V_s`,
  `P = sum_s U_s (.) V_s` is implemented exactly, including the
  scale-invariant normalisation by the soft active count.

- What is only hypothesized: That replacing the conv mixer with the
  `event_delta_bilinear_accumulator` mixer lifts PR AUC on at least
  one CRTK slice (most likely slices where two-piece interactions are
  load-bearing -- fork / discovered-attack / battery / pin / skewer /
  x-ray -- and the upper `crtk_difficulty` tail where multi-piece
  coordination dominates) without regressing aggregate PR AUC by more
  than the matched-baseline tolerance.

- Failure cases:
  - The learned soft occupancy `O_s = sigmoid(w . x_s + b)` fails to
    recover the piece-plane occupancy that the source primitive uses,
    so the pair-term sum is dominated by noise from empty squares;
    the in-mixer `zero_occupancy`-style ablation matches this idea on
    its declared target slice.
  - The `event_delta_bilinear_accumulator` mixer collapses inside the
    BT4 shell because the residual + SqueezeExcite path dominates the
    mixer output; the `conv` baseline matches the variant within
    noise.
  - The broadcast-back fusion `y_s = MLP([x_s; A; B; Q])` carries
    only the per-square own-token signal `x_s` and ignores the
    broadcast context, meaning the pair term is decorative; the
    `first_order_only` ablation (drop `Q`) closes the gap.
  - The pair term `Q` collapses to noise because `U` and `V` learn
    aligned subspaces, making `P ~ A (.) B` and `Q ~ 0`; report
    block-level `||Q||` statistics to catch this.

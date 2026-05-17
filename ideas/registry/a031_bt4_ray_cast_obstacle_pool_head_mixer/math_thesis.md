# Mathematical Thesis

- Mathematical motivation: The BT4-style residual tower from
  `lc0_bt4_classifier` mixes spatially with a pair of 3x3 convs. This
  idea holds the tower shell (stem -> N residual + SqueezeExcite blocks
  -> value head) fixed and replaces only the per-block spatial-mixing
  operator with the `ray_cast_obstacle_pool_head` primitive from
  `p026_ray_cast_obstacle_pool_head`. Source primitive math:
  `ideas/registry/p026_ray_cast_obstacle_pool_head/math_thesis.md`.
  The Ray-Cast Obstacle Pooling (RayPool) operator, for a per-square
  feature stack `X in R^{B x C x 8 x 8}` and a per-square occupancy
  proxy `O in [0, 1]^{B x 8 x 8}`, with per-direction step
  `(dr_d, df_d)` and a learned per-direction decay `gamma_d in [0, 1]`,
  forms

  ```
  Y_{d, i} = sum_{s>=1} gamma_d^s * X_{i + s * (dr_d, df_d)}
                         * prod_{k=1..s-1} (1 - O_{i + k * (dr_d, df_d)})
  ```

  along each of the 8 chess directions `d in {N, NE, E, SE, S, SW, W, NW}`.
  The running product `prod (1 - O)` collapses to zero at the first
  occupied square, terminating the ray at the first blocker. The
  decay `gamma_d` carries a learned per-direction reach.

- Assumptions:
  1. The `ray_cast_obstacle_pool_head` primitive is well-defined as a
     shape-preserving operator `(B, C, 8, 8) -> (B, C, 8, 8)` under
     the
     `chess_nn_playground.models.architecture.bt4_mixers._base.Mixer`
     contract.
  2. The BT4 block wrapper (`mixer -> SqueezeExcite -> +residual ->
     ReLU`) is identical across all `a###_bt4_*_mixer` ideas and
     across the `conv` and `attention` baselines.
  3. The optimizer protocol, data contract (`simple_18`,
     `puzzle_binary`), and training budget are identical across all
     `a###` and baseline runs, so the only experimental variable is
     the mixer.
  4. The source primitive (p026) reads a rule-exact occupancy mask
     directly off the `simple_18` piece planes; the mixer cannot read
     piece planes (it sees only a generic `(B, C, 8, 8)` channel
     tensor) so the occupancy is replaced by a soft proxy
     `O = sigmoid(Conv1x1(X))` learned from the activations. The
     geometric-decay + running-unblocked-product structure -- the
     load-bearing idea of RayPool -- is preserved exactly; occlusion
     termination becomes content-based rather than rule-exact.

- Claimed advantage: If the `ray_cast_obstacle_pool_head` primitive
  carries a load-bearing long-range ray-pooled signal that conv and
  attention do not, dropping it into the BT4 block must lift held-out
  PR AUC (aggregate or on a slice that depends on long-range piece
  influence -- e.g. back-rank pressure, pins, batteries, x-rays, and
  the long-piece tactical motifs that span half the board) versus
  the two baselines under the same tower, optimizer, and data. This
  is a controlled architecture-level test of "is
  ray_cast_obstacle_pool_head a better spatial mixer than conv or
  attention inside a fixed BT4 tower shell?", not a new primitive
  claim. The per-block cost is `O(8 * max_ray_length * C * 8 * 8)`
  for the per-direction shift-and-accumulate, plus an `O(8 C^2)`
  1x1 projection back to `C` channels; for `max_ray_length = 7` this
  is cheaper than a 64x64 attention map and comparable to a small
  conv stack at matched widths.

- Proof sketch: This is an empirical study, not a theorem. The well-
  definedness of the mixer is enforced at construction time by the
  BT4 block's shape check (raises if `mixer(x).shape != x.shape`).
  The primitive-level math for `ray_cast_obstacle_pool_head` itself
  (geometric series terminates at the first occupied square,
  associativity along each direction, bounded magnitude
  `||Y_{d, i}|| <= ||X||_inf * gamma_d / (1 - gamma_d)`) is proven
  in the source primitive's math thesis and falsified by its own
  ablation grid (`drop_occlusion`, `shuffle_directions`,
  `zero_rays`). This folder inherits that math and tests whether the
  resulting operator, used as a token mixer rather than as a pooled
  additive head, transfers its signal through the BT4 tower.

- What is actually proven: The mixer is shape-preserving on
  `(B, C, 8, 8)` inputs and integrates with the BT4 block via the
  unified mixer registry. A forward + backward smoke test guards
  the mixer at registration time. The per-direction accumulator
  `Y_{d, i} = sum_{s>=1} gamma_d^s * X_{i + s * dir_d}
              * prod_{k=1..s-1} (1 - O_{i + k * dir_d})` is
  implemented exactly as a sequential prefix sum of length
  `max_ray_length`, with `gamma_d` clamped to `[0, 1]` for bounded
  magnitude. The running unblocked product
  `prod_{k=1..s-1} (1 - O_{i + k * dir_d})` collapses to zero at
  the first occupied square along the ray, terminating the
  geometric series exactly as the math demands.

- What is only hypothesized: That replacing the conv mixer with the
  `ray_cast_obstacle_pool_head` mixer lifts PR AUC on at least one
  CRTK slice (most likely long-range tactical slices -- back-rank,
  pin / skewer / x-ray, sliding-piece battery, queen-side
  long-diagonal motifs) without regressing aggregate PR AUC by more
  than the matched-baseline tolerance. The hypothesis also covers
  the higher `crtk_difficulty` band where the trunk's local-receptive-
  field stack is most likely to be insufficient.

- Failure cases:
  - The trunk's stem conv plus the surrounding residual + SE blocks
    already encode long-range slider influence densely enough that
    the per-direction ray pooling adds no marginal signal; the
    `conv` baseline matches the variant within noise.
  - The soft occupancy proxy `O = sigmoid(Conv1x1(X))` is not
    discriminative enough to terminate rays at blockers; the
    geometric series runs over the whole board and the mixer
    degenerates into a per-direction global pool weighted by
    `gamma_d`. The `drop_occlusion` style ablation (force
    `O := 0`) should close the gap if so.
  - The per-direction learned decays `gamma_d` collapse to a single
    shared value; the mixer degenerates to an isotropic
    direction-invariant pool. A `shuffle_directions`-style ablation
    matches this idea on its declared target slice if direction-
    specific learning is not load-bearing.
  - SqueezeExcite + residual + ReLU absorbs most of the mixer's
    contribution if the per-direction accumulator magnitude is
    small at high gamma; report per-block ray-output norm
    statistics alongside the headline number.
  - The sequential per-direction prefix scan adds wall-clock cost
    that is not amortised by signal. If
    `train_samples_per_second` falls well below the matched conv
    baseline without a slice-level lift, the mixer fails its cost-
    matched comparison.

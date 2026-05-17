# Mathematical Thesis

- Mathematical motivation: The BT4-style residual tower from
  `lc0_bt4_classifier` mixes spatially with a pair of 3x3 convs. This
  idea holds the tower shell (stem -> N residual + SqueezeExcite blocks
  -> value head) fixed and replaces only the per-block spatial-mixing
  operator with the `ray_parallel_ssm_head` primitive (Ray-SSM) from
  `p030_ray_parallel_ssm_head`. Source primitive math:
  `ideas/registry/p030_ray_parallel_ssm_head/math_thesis.md`.
  The Ray-Parallel SSM operator is a *selective state-space scan*
  along each of the 8 chess directions
  `d in {N, NE, E, SE, S, SW, W, NW}` with diagonal, input-conditioned
  retention A and injection B and a learned per-direction read-out C:

  ```
  h_{i, d, c} = A_{i, d, c} * h_{i - shift_d, d, c} + B_{i, d, c} * x_{i, c}
  y_{i, c}    = sum_d C_{d, c} * h_{i, d, c}
  ```

  with `A_{i, d, c} = sigma(W_A(x_i))_{d, c}` and
  `B_{i, d, c} = sigma(W_B(x_i))_{d, c}` both in (0, 1) per
  (direction, channel), and `C in R^{NUM_DIRECTIONS x C}` a learned
  parameter. The load-bearing idea is that separating *retention*
  (A) from *injection* (B) per channel is the strictly more
  expressive operator in the ray family: RayPool (`p026`) has one
  scalar `gamma` per direction, OARS (`p029`) has a multiplicative-
  only gate, Ray-SSM separately learns A and B per (square,
  direction, channel).

- Assumptions:
  1. The `ray_parallel_ssm_head` primitive is well-defined as
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
  4. The source primitive (p030) was a *pooled head* over the i193
     trunk that mean-pooled `y_total` to a scalar logit fused via
     gate/delta MLPs. The mixer adaptation keeps the operator
     shape-preserving: it returns `y_total` directly (after a 1x1
     output projection) without the terminal mean-pool. The load-
     bearing selective-scan structure (per-(square, direction,
     channel) A and B from learned 1x1 sigmoid projections of the
     per-square features, per-direction `C` read-out) is preserved
     exactly.
  5. `C` is per-direction-only (not per-square); the spec's full
     form `y = sum_d C_{i, d} h_{i, d}` would require an additional
     parameter table conditioned on square. This matches the source
     primitive's stable simplification documented in
     `p030/implementation_notes.md`.

- Claimed advantage: If the `ray_parallel_ssm_head` primitive
  carries a load-bearing selective long-range ray signal beyond
  what conv and dense attention provide, dropping it into the BT4
  block must lift held-out PR AUC (aggregate or on a slice that
  depends on long-range piece interactions along files / diagonals
  / ranks -- e.g. pin/skewer/discovered-attack motifs, batteries
  along files/diagonals, X-ray attacks on the king, rook-on-open-
  file, queen-on-open-line, bishop-pair-on-long-diagonal) versus
  the two baselines under the same tower, optimizer, and data.
  This is a controlled architecture-level test of "is
  `ray_parallel_ssm_head` a better spatial mixer than conv or
  attention inside a fixed BT4 tower shell?", not a new primitive
  claim. The per-block cost is `O(8 * max_ray_length * 64 * C)`
  for the iterated scan, plus `O(2 * 8 * C^2 * 64)` for the
  `A_proj`/`B_proj` per-(direction, channel) projections and
  `O(C^2 * 64)` for the output 1x1 projection; the dominant cost
  scales as `O(C^2)` per token (the per-direction A/B projections
  are `O(NUM_DIRECTIONS * C^2)` each), so the operator is
  asymptotically *cheaper per block* than the dense `attention`
  baseline (`O(64 * 64 * C)` token-pair matmul) but somewhat more
  expensive than the conv baseline's `O(9 * C^2)` 3x3 pair.

- Proof sketch: This is an empirical study, not a theorem. The well-
  definedness of the mixer is enforced at construction time by the
  BT4 block's shape check (raises if `mixer(x).shape != x.shape`).
  The primitive-level math for `ray_parallel_ssm_head` itself (the
  scan is bounded because both A in (0, 1) and B in (0, 1) are
  contractions; with `A = 0` the operator reduces to a per-square
  `B * x` followed by per-direction `C` read-out summed across
  directions -- a trivial baseline; with `B = 0` the state never
  gets injected and the head decays to zero) is proven in the
  source primitive's math thesis and falsified by its own ablation
  grid (`disable_selective_A`, `disable_selective_B`,
  `no_directional_C`, `zero_ssm_features`). This folder inherits
  that math and tests whether the resulting operator, used as a
  token mixer rather than as a pooled additive head, transfers its
  signal through the BT4 tower.

- What is actually proven: The mixer is shape-preserving on
  `(B, C, 8, 8)` inputs and integrates with the BT4 block via the
  unified mixer registry. A forward + backward smoke test guards
  the mixer at registration time. The per-direction scan
  `h_{t+1} = A * shift_d(h_t) + B * x` is bounded because both A
  and B are sigmoid in (0, 1), so the iterated state satisfies
  `||h_t|| <= ||B||_inf * ||x|| * sum_{k=0}^{t-1} ||A||_inf^k`,
  hence `||h_t|| <= ||B||_inf * ||x|| / (1 - ||A||_inf)` for
  `||A||_inf < 1`. With `A` clamped to 0 the operator reduces to
  `h_{1, d} = B * x` (one-step injection only) followed by a
  per-direction `C` read-out summed across directions -- matches
  the source primitive's `disable_selective_A` ablation. With `B`
  clamped to 0 the state never gets injected and `y_total = 0` --
  matches `disable_selective_B`.

- What is only hypothesized: That replacing the conv mixer with the
  `ray_parallel_ssm_head` mixer lifts PR AUC on at least one CRTK
  slice (most likely long-range tactical slices: pins, skewers,
  discovered attacks, batteries on files/diagonals, X-ray attacks
  on the king, rook-on-open-file, bishop-pair-on-long-diagonal)
  without regressing aggregate PR AUC by more than the matched-
  baseline tolerance. The hypothesis also covers the higher
  `crtk_difficulty` band where the trunk's local-receptive-field
  stack and the dense `attention` baseline are both likely to be
  insufficient (the conv stack misses long rays; dense attention
  has no chess-ray prior, so it must rediscover the 8 directions
  and the selective per-channel mix from data).

- Failure cases:
  - The trunk's stem conv plus the surrounding residual + SE blocks
    already encode long-range ray context densely enough (after `N`
    blocks the effective receptive field already covers the 8x8
    board) that adding a single selective-scan layer per block buys
    no marginal signal; the `conv` baseline matches the variant
    within noise.
  - The dense `attention` baseline matches or beats the Ray-SSM
    mixer; all-pairs attention can in principle express any ray
    pattern and the explicit 8-direction selective-scan prior is
    decorative at the BT4 tower's capacity.
  - Both A and B saturate near 0 everywhere, so the iterated state
    collapses to zero and `y_total = 0`. The mixer becomes the
    zero map and the BT4 block degenerates to a SqueezeExcite +
    residual block. `disable_selective_A` and `disable_selective_B`
    (A and B clamped) should then close the gap.
  - A saturates near 1 everywhere, so the scan becomes a plain
    geometric prefix sum equivalent to RayPool / `p026` modulated
    by `B`. The selective-retention story collapses and the
    comparison against `bt4_ray_cast_obstacle_pool_head_mixer`
    (sibling idea / `p026` mixer) should match.
  - Per-direction `C` is per-direction-only (not per-square): the
    spec's full form `y = sum_d C_{i, d} h_{i, d}` is approximated
    by a single learned `C[d]` vector. If A3 (the primitive used as
    a pooled head with the source-primitive's gating) strictly
    beats this mixer, the per-direction-only `C` simplification is
    what killed the signal.
  - SqueezeExcite + residual + ReLU absorbs most of the mixer's
    contribution if the routed-token output magnitude is small
    relative to the residual stream; report per-block routed
    output norm statistics alongside the headline number.
  - The 8-direction sum collapses to a near-zero map if the
    learned per-direction `C[d]` rows learn anti-correlated
    outputs; report per-direction state energy distribution to
    detect this mode.

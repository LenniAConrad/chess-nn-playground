# Mathematical Thesis

- Mathematical motivation: The BT4-style residual tower from
  `lc0_bt4_classifier` mixes spatially with a pair of 3x3 convs. This
  idea holds the tower shell (stem -> N residual + SqueezeExcite blocks
  -> value head) fixed and replaces only the per-block spatial-mixing
  operator with the `occlusion_aware_ray_scan_head` primitive (OARS)
  from `p029_occlusion_aware_ray_scan_head`. Source primitive math:
  `ideas/registry/p029_occlusion_aware_ray_scan_head/math_thesis.md`.
  The Occlusion-Aware Ray Scan operator is a *selective associative
  scan* along each of the 8 chess directions
  `d in {N, NE, E, SE, S, SW, W, NW}`. The associative operator is
  `a (x) b = a + sigma(W_block(.)) * b`, materialised iteratively as:

  ```
  state_{i, d} = features_i + g_{i, d} * shift_d(state_{i, d})
  y_i          = sum_d C_d * state_{i, d}
  ```

  with `g_{i, d} = sigma(Conv2d(C -> 8)(x))_{i, d}` a learned per-
  (square, direction) blocker gate and `C_d` a per-direction
  `Conv2d(C -> C, 1x1)` output projection. The load-bearing idea is
  *content-dependent* termination: the ray is gated by features of the
  per-square channel state, so it can learn to "stop at the first
  hostile piece" rather than at fixed occupancy, which is the
  differentiator from the plain RayPool primitive (`p026`).

- Assumptions:
  1. The `occlusion_aware_ray_scan_head` primitive is well-defined as
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
  4. The source primitive (p029) was a *pooled head* over the i193
     trunk that mean-pooled the per-direction state to a logit. The
     mixer adaptation keeps it shape-preserving: it replaces the
     final mean-pool with a per-direction `Conv2d(C -> C, 1x1)`
     projection summed across the 8 directions, yielding a
     `(B, C, 8, 8)` channel tensor instead of a scalar. The
     load-bearing selective-scan structure (per-direction iterated
     `state = features + g * shifted_state`, with `g` from a learned
     per-(square, direction) sigmoid head on the per-square channel
     features) is preserved exactly.
  5. The state-dependent gate is approximated by a *fixed-feature*
     gate: `g_{i, d} = sigma(Conv2d(C -> 8)(x))` is computed once
     from the raw per-square features rather than re-derived from
     the running state at each step. This matches the source
     primitive's stable simplification for the 8x8 board.

- Claimed advantage: If the `occlusion_aware_ray_scan_head` primitive
  carries a load-bearing selective long-range ray-scan signal beyond
  what conv and dense attention provide, dropping it into the BT4
  block must lift held-out PR AUC (aggregate or on a slice that
  depends on long-range piece interactions terminated by intervening
  pieces -- e.g. pin/skewer/discovered-attack motifs, batteries along
  files/diagonals, X-ray attacks on the king) versus the two baselines
  under the same tower, optimizer, and data. This is a controlled
  architecture-level test of "is `occlusion_aware_ray_scan_head` a
  better spatial mixer than conv or attention inside a fixed BT4 tower
  shell?", not a new primitive claim. The per-block cost is
  `O(8 * max_ray_length * 64 * C)` for the iterated scan, plus
  `O(C * 8)` for the blocker-gate conv and `O(8 * C^2)` for the eight
  `Conv2d(C -> C, 1x1)` per-direction projections; the dominant cost
  scales linearly with `C` per token (the per-direction projections
  are `O(C^2)` per token), so the operator is asymptotically
  *cheaper per block* than the dense `attention` baseline
  (`O(64 * 64 * C)` token-pair matmul) but slightly more expensive
  than the conv baseline's `O(9 * C^2)` 3x3 pair.

- Proof sketch: This is an empirical study, not a theorem. The well-
  definedness of the mixer is enforced at construction time by the
  BT4 block's shape check (raises if `mixer(x).shape != x.shape`).
  The primitive-level math for `occlusion_aware_ray_scan_head` itself
  (the scan is bounded because `sigma in (0, 1)` contracts the
  carried state at each step; with `g = 0` the operator reduces to a
  per-square 1x1 conv followed by directional sum; with `g = 1` the
  operator reduces to a plain geometric prefix sum, equivalent to
  RayPool / `p026` without occupancy-based termination) is proven in
  the source primitive's math thesis and falsified by its own
  ablation grid (`disable_blocker_gate`, `shuffle_directions`,
  `zero_oars_features`). This folder inherits that math and tests
  whether the resulting operator, used as a token mixer rather than
  as a pooled additive head, transfers its signal through the BT4
  tower.

- What is actually proven: The mixer is shape-preserving on
  `(B, C, 8, 8)` inputs and integrates with the BT4 block via the
  unified mixer registry. A forward + backward smoke test guards
  the mixer at registration time. The per-direction scan
  `state_{t+1} = x + g * shift_d(state_t)` is a contraction in the
  shifted-state argument because `g in (0, 1)` (sigmoid), so the
  iterated state is bounded by
  `||state_t|| <= ||x|| * sum_{k=0}^{t-1} ||g||_inf^k`, hence by
  `||x|| / (1 - ||g||_inf)` for `||g||_inf < 1`. With the blocker
  gate clamped to 0 the operator reduces to a per-direction 1x1
  conv summed across directions -- matches the source primitive's
  `disable_blocker_gate` ablation (gate = 1 in source convention,
  gate = 0 in this convention since the iterated state collapses
  to zero contribution). With the blocker gate clamped to 1 the
  operator becomes a plain geometric prefix sum along each
  direction, equivalent to `ray_cast_obstacle_pool_head` (`p026`)
  without occupancy gating.

- What is only hypothesized: That replacing the conv mixer with the
  `occlusion_aware_ray_scan_head` mixer lifts PR AUC on at least one
  CRTK slice (most likely long-range tactical slices: pins, skewers,
  discovered attacks, batteries on files/diagonals, X-ray attacks on
  the king) without regressing aggregate PR AUC by more than the
  matched-baseline tolerance. The hypothesis also covers the higher
  `crtk_difficulty` band where the trunk's local-receptive-field
  stack and the dense `attention` baseline are both likely to be
  insufficient (the conv stack misses long rays; dense attention has
  no chess-ray prior, so it must rediscover the 8 directions and the
  blocker structure from data).

- Failure cases:
  - The trunk's stem conv plus the surrounding residual + SE blocks
    already encode long-range ray context densely enough (after `N`
    blocks the effective receptive field already covers the 8x8
    board) that adding a single ray-scan layer per block buys no
    marginal signal; the `conv` baseline matches the variant within
    noise.
  - The dense `attention` baseline matches or beats the OARS mixer;
    all-pairs attention can in principle express any ray pattern and
    the explicit 8-direction prior is decorative at the BT4 tower's
    capacity.
  - The `sigma(W_block)` gate saturates near 0 everywhere, so the
    iterated state collapses to zero contribution and the mixer
    reduces to the per-direction `Conv2d(C -> C, 1x1)` of zero --
    the operator becomes the zero map and the BT4 block degenerates
    to a SqueezeExcite + residual block. The `disable_blocker_gate`-
    style ablation (gate clamped, no learned termination) should
    then close the gap.
  - The `sigma(W_block)` gate saturates near 1 everywhere, so the
    scan becomes a plain geometric prefix sum equivalent to RayPool
    / `p026` with no content-dependent termination. The whole
    occlusion-aware story collapses, and the comparison against a
    `bt4_ray_cast_obstacle_pool_head_mixer` (sibling idea / `p026`
    mixer) should match.
  - The fixed-feature gate (computed from raw `x` rather than from
    the running state) loses the state-dependent termination claim;
    the operator cannot terminate after passing through a piece
    "discovered" mid-scan. If A3 (the primitive used as a pooled
    head with the source-primitive's gating) strictly beats this
    mixer, the fixed-feature simplification is what killed the
    signal.
  - SqueezeExcite + residual + ReLU absorbs most of the mixer's
    contribution if the routed-token output magnitude is small
    relative to the residual stream; report per-block routed
    output norm statistics alongside the headline number.
  - The 8-direction sum collapses to a near-zero map if the eight
    `Conv2d(C -> C, 1x1)` direction projections learn anti-correlated
    outputs; report per-direction output norm distribution to
    detect this mode.

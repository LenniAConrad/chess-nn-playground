# Mathematical Thesis

- Mathematical motivation: The BT4-style residual tower from
  `lc0_bt4_classifier` mixes spatially with a pair of 3x3 convs. This idea
  holds the tower shell (stem -> N residual + SqueezeExcite blocks ->
  value head) fixed and replaces only the per-block spatial-mixing
  operator with the `blocker_reset_ray_scan` primitive from
  `p020_blocker_reset_ray_scan`. Source primitive math:
  `ideas/registry/p020_blocker_reset_ray_scan/math_thesis.md`.
  For each board square `s`, direction `d in 0..7`, ordered ray cells
  `s_t = ray(s, d, t)` for `t = 0..L_max`, and per-direction learnable
  decay `lambda_d in (0, 1)^h`, the primitive runs the gated recurrence
  `h_{s,d,0} = U x_s`,
  `h_{s,d,t} = U x_{s_t} + (1 - O_{s_t}) (.) lambda_d (.) h_{s,d,t-1}`,
  and mean-pools the per-step states into the directional readout
  `y_{s,d}`. The defining property is the hard reset gate
  `(1 - O_{s_t})`: a blocker at step `t` zeroes the entire carried
  history, so the line behind a blocker cannot see the line in front of
  it. Inside the BT4 block the 8 directional outputs at each source
  square are concatenated and projected back to the channel dimension.

- Assumptions:
  1. The `blocker_reset_ray_scan` primitive is well-defined as a
     shape-preserving operator
     `(B, C, 8, 8) -> (B, C, 8, 8)` under the
     `chess_nn_playground.models.architecture.bt4_mixers._base.Mixer`
     contract.
  2. The BT4 block wrapper (`mixer -> SqueezeExcite -> +residual ->
     ReLU`) is identical across all `a###_bt4_*_mixer` ideas and across
     the `conv` and `attention` baselines.
  3. The optimizer protocol, data contract (`simple_18`,
     `puzzle_binary`), and training budget are identical across all
     `a###` and baseline runs, so the only experimental variable is the
     mixer.
  4. The mixer cannot read the piece planes directly (the BT4 block
     hands it a generic `(B, C, 8, 8)` channel tensor), so the
     occupancy / blocker indicator is derived inside the operator as
     `O_s = sigmoid(w . x_s + b)` from the per-square channel vector.
     The blocker mask is therefore *learned* from features, but it is
     still generated *inside* the operator and never supplied
     externally, which preserves the source thesis's defining property.

- Claimed advantage: If the `blocker_reset_ray_scan` primitive
  carries a sliding-piece signal that conv and attention do not,
  dropping it into the BT4 block must lift held-out PR AUC (aggregate or
  on a sliding-piece-dependent slice such as pin / skewer / discovered
  attack / rook-on-open-file / queen-line-into-king-zone) versus the two
  baselines under the same tower, optimizer, and data. This is a
  controlled architecture-level test of "is blocker_reset_ray_scan a
  better spatial mixer than conv or attention inside a fixed BT4 tower
  shell?", not a new primitive claim. The per-direction segmented
  recurrence is `O(64 * 8 * L * h)` per block where `L <= 7` and `h` is
  the mixer hidden width; this is comparable to a 3x3 conv in FLOPs but
  asymptotically cheaper than a full 64x64 attention map.

- Proof sketch: This is an empirical study, not a theorem. The well-
  definedness of the mixer is enforced at construction time by the BT4
  block's shape check (raises if `mixer(x).shape != x.shape`). The
  primitive-level math for `blocker_reset_ray_scan` itself (the
  segmented gated ray recurrence with content-derived blocker reset) is
  proven in the source primitive's math thesis and falsified by its own
  ablation grid (`zero_blocker`, `uniform_blocker`). This folder
  inherits that math and tests whether the resulting operator, used as
  a token mixer rather than as an additive head, transfers its signal
  through the BT4 tower.

- What is actually proven: The mixer is shape-preserving on
  `(B, C, 8, 8)` inputs and integrates with the BT4 block via the
  unified mixer registry. A forward + backward smoke test guards the
  mixer at registration time. The ray-step index and step-mask tables
  are pure rule-derived constants (queen-style 8 directions, up to 7
  steps, off-board steps masked); the hard reset gate
  `(1 - O_{s_t}) * step_mask` is implemented exactly in the per-step
  recurrence.

- What is only hypothesized: That replacing the conv mixer with the
  `blocker_reset_ray_scan` mixer lifts PR AUC on at least one CRTK
  slice (most likely slices where sliding-piece vision is load-bearing
  -- pin / skewer / discovered attack / rook-on-open-file /
  queen-line-into-king-zone -- and the upper `crtk_difficulty` tail
  where long-range geometry dominates over local conv windows) without
  regressing aggregate PR AUC by more than the matched-baseline
  tolerance.

- Failure cases:
  - The learned soft occupancy `O_s = sigmoid(w . x_s + b)` fails to
    recover the piece-plane occupancy that the source primitive uses,
    so the reset gate is uninformative; the `zero_blocker` ablation
    matches this idea on its declared target slice.
  - The `blocker_reset_ray_scan` mixer collapses inside the BT4 shell
    because the residual + SqueezeExcite path dominates the mixer
    output; the `conv` baseline matches the variant within noise.
  - The 8-way directional fuse `out = Linear(8 * C -> C)` underfits and
    bottlenecks the per-direction signal; the `attention` baseline
    matches or beats this idea.
  - The mixer's per-sample cost (Python-loop scan of depth 8 across
    `L = 8` steps) inflates wall-clock enough that the matched-budget
    comparison is unfair; the baselines train for more effective
    optimizer steps inside the same wall-clock budget.

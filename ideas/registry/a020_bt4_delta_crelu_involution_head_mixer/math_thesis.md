# Mathematical Thesis

- Mathematical motivation: The BT4-style residual tower from
  `lc0_bt4_classifier` mixes spatially with a pair of 3x3 convs. This idea
  holds the tower shell (stem -> N residual + SqueezeExcite blocks ->
  value head) fixed and replaces only the per-block spatial-mixing
  operator with the `delta_crelu_involution_head` (DCIH) primitive from
  `p015_delta_crelu_involution_head`. Source primitive math:
  `ideas/registry/p015_delta_crelu_involution_head/math_thesis.md`.

- Assumptions:
  1. The DCIH primitive is well-defined as a shape-preserving operator
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
  4. The channel-reversal involution is a faithful stand-in for the
     piece-plane colour swap given the channel-agnostic BT4 trunk: the
     rank-flip is exact, and the channel-reversal preserves the
     `iota^2 = id` involution algebra even though no per-channel piece
     semantics is enforced.

- Claimed advantage: If the DCIH primitive carries a saturation-aware
  and colour-equivariant spatial mixing signal that conv and attention do
  not, dropping it into the BT4 block must lift held-out PR AUC
  (aggregate or on a target slice) versus the two baselines under the
  same tower, optimizer, and data. This is a controlled
  architecture-level test of "is DCIH a better spatial mixer than conv
  or attention inside a fixed BT4 tower shell?", not a new primitive
  claim.

- Proof sketch: This is an empirical study, not a theorem. The
  well-definedness of the mixer is enforced at construction time by the
  BT4 block's shape check (raises if `mixer(x).shape != x.shape`). The
  primitive-level math for DCIH itself (saturation-aware ClippedReLU
  accumulator plus the involution Reynolds split that structurally
  enforces colour-flip equivariance) is described in the source
  primitive's math thesis and falsified by its own ablation grid. This
  folder inherits that math and tests whether the resulting operator,
  used as a token mixer rather than as an additive head over the i193
  trunk, transfers its signal through the BT4 tower.

- What is actually proven: The mixer is shape-preserving on
  `(B, C, 8, 8)` inputs and integrates with the BT4 block via the
  unified mixer registry. The channel-reversal + rank-flip involution
  satisfies `iota^2 = id` by construction. A forward + backward smoke
  test guards the mixer at registration time.

- What is only hypothesized: That replacing the conv mixer with the
  DCIH mixer lifts PR AUC on at least one CRTK slice (the saturation
  regime — e.g. high `crtk_difficulty` and `crtk_eval_bucket = equal`
  positions where small per-square deltas matter — or colour-symmetric
  motifs that benefit from the Reynolds split) without regressing
  aggregate PR AUC by more than the matched-baseline tolerance.

- Failure cases:
  - The ClippedReLU saturates immediately because the residual stream
    magnitude dominates and the accumulator pre-activation never lands
    in the saturating band; the saturation-aware part of DCIH carries
    no information and the mixer reduces to a noisy involution
    averager. The `conv` baseline matches within noise.
  - The channel-reversal involution is not a meaningful colour swap on
    the BT4 trunk channels (which carry no piece-plane semantics), so
    the Reynolds split contributes only spurious mixing; `attention`
    matches or beats DCIH.
  - DCIH's per-block embedding + fuse cost inflates wall-clock enough
    that the matched-budget comparison is unfair; the baselines train
    for more effective optimizer steps inside the same wall-clock
    budget.

# Mathematical Thesis

- Mathematical motivation: The BT4-style residual tower from
  `lc0_bt4_classifier` mixes spatially with a pair of 3x3 convs. This idea
  holds the tower shell (stem -> N residual + SqueezeExcite blocks ->
  value head) fixed and replaces only the per-block spatial-mixing
  operator with the `witness_counterwitness_quantifier` (WCQ) primitive
  from `p005_witness_counterwitness_quantifier`. Source primitive math:
  `ideas/registry/p005_witness_counterwitness_quantifier/math_thesis.md`.

- Assumptions:
  1. The WCQ primitive is well-defined as a shape-preserving operator
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
  4. The nested adversarial quantifier `exists candidate (forall reply)`
     remains numerically stable when carried inside a residual mixer
     instead of as an additive head, with the soft-quantifier
     temperatures (`tau_forall=tau_exists=0.2`) acting as smoothers.

- Claimed advantage: If the WCQ primitive carries a spatial mixing
  signal that conv and attention do not, dropping it into the BT4 block
  must lift held-out PR AUC (aggregate or on a target slice) versus the
  two baselines under the same tower, optimizer, and data. This is a
  controlled architecture-level test of "is WCQ a better spatial mixer
  than conv or attention inside a fixed BT4 tower shell?", not a new
  primitive claim.

- Proof sketch: This is an empirical study, not a theorem. The well-
  definedness of the mixer is enforced at construction time by the BT4
  block's shape check (raises if `mixer(x).shape != x.shape`). The
  primitive-level math for WCQ itself (nested `exists / forall`
  reduction over candidate and reply tokens compiled from board
  squares, with the witness weights scattered back onto the 64-square
  grid) is proven in the source primitive's math thesis and falsified
  by its own ablation grid. This folder inherits that math and tests
  whether the resulting operator, used as a token mixer rather than as
  an additive head, transfers its signal through the BT4 tower.

- What is actually proven: The mixer is shape-preserving on
  `(B, C, 8, 8)` inputs and integrates with the BT4 block via the
  unified mixer registry. A forward + backward smoke test guards the
  mixer at registration time. The nested `logsumexp` form is monotone
  in temperature and recovers `max_k [claim_k - max_r counter_{kr}]`
  in the zero-temperature limit, by standard soft-max / soft-min
  arguments.

- What is only hypothesized: That replacing the conv mixer with the
  WCQ mixer lifts PR AUC on at least one CRTK slice (most likely the
  high-`crtk_difficulty` and tactical-motif slices that motivate the
  WCQ primitive's witness/counterwitness framing) without regressing
  aggregate PR AUC by more than the matched-baseline tolerance.

- Failure cases:
  - The WCQ mixer reduces to a noisy attention head inside the BT4
    shell because the residual + SqueezeExcite path dominates the
    mixer output; the `attention` baseline matches the WCQ variant
    within noise.
  - The fixed candidate / reply query counts (`K=R=16`) under-cover
    the chess-relevant witness/counterwitness pairs at the BT4 token
    grid, so the quantifier collapses to a flat reweight; raising
    `K, R` makes the mixer too expensive for the matched-budget
    contract.
  - The soft-quantifier temperatures (`tau_forall, tau_exists`) are
    held constant inside the tower rather than annealed as in the
    p005 head, so gradients through the nested `logsumexp` are too
    smoothed and the WCQ mixer behaves like a generic pooled
    attention. The conv or attention baseline matches it on every
    slice.
  - WCQ's per-block cost inflates wall-clock enough that the
    matched-budget comparison is unfair; the baselines train for
    more effective optimizer steps inside the same wall-clock budget.

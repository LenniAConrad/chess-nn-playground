# Mathematical Thesis

- Mathematical motivation: The BT4-style residual tower from
  `lc0_bt4_classifier` mixes spatially with a pair of 3x3 convs. This idea
  holds the tower shell (stem -> N residual + SqueezeExcite blocks ->
  value head) fixed and replaces only the per-block spatial-mixing
  operator with the `tail_copula_concordance` (TCC) primitive from
  `p004_tail_copula_concordance`. Source primitive math:
  `ideas/registry/p004_tail_copula_concordance/math_thesis.md`.

- Assumptions:
  1. The TCC primitive is well-defined as a shape-preserving operator
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

- Claimed advantage: If the TCC primitive carries a spatial mixing
  signal that conv and attention do not, dropping it into the BT4 block
  must lift held-out PR AUC (aggregate or on a target slice) versus the
  two baselines under the same tower, optimizer, and data. This is a
  controlled architecture-level test of "is TCC a better spatial mixer
  than conv or attention inside a fixed BT4 tower shell?", not a new
  primitive claim.

- Proof sketch: This is an empirical study, not a theorem. The well-
  definedness of the mixer is enforced at construction time by the BT4
  block's shape check (raises if `mixer(x).shape != x.shape`). The
  primitive-level math for TCC itself (per-channel soft-rank Sklar
  copula transform, sigmoid upper-tail membership, and symmetric tail
  concordance `Lambda_{c,d} = sqrt(lambda_{c->d} * lambda_{d->c})`) is
  proven in the source primitive's math thesis and falsified by its
  own ablation grid. This folder inherits that math and tests whether
  the resulting operator, used as a token mixer rather than as an
  additive head, transfers its signal through the BT4 tower.

- What is actually proven: The mixer is shape-preserving on
  `(B, C, 8, 8)` inputs and integrates with the BT4 block via the
  unified mixer registry. A forward + backward smoke test guards the
  mixer at registration time.

- What is only hypothesized: That replacing the conv mixer with the
  TCC mixer lifts PR AUC on at least one CRTK slice (most likely the
  `crtk_difficulty` upper tail where cross-site evidence alignment
  separates real tactics from marginal-matched near-puzzles) without
  regressing aggregate PR AUC by more than the matched-baseline
  tolerance.

- Failure cases:
  - The TCC mixer reduces to a noisy conv inside the BT4 shell
    because the residual + SqueezeExcite path dominates the mixer
    output; the `conv` baseline matches the TCC variant within noise.
  - The simple_18 board tensor lacks the dense per-square evidence
    channels the primitive depends on, so the cross-channel tail
    concordance matrix collapses to near-identity; an `attention`
    baseline matches or beats TCC.
  - TCC's `O(N^2 * C)` soft-rank cost inflates wall-clock enough that
    the matched-budget comparison is unfair; the baselines train for
    more effective optimizer steps inside the same wall-clock budget.

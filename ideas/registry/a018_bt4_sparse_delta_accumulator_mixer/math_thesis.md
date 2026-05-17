# Mathematical Thesis

- Mathematical motivation: The BT4-style residual tower from
  `lc0_bt4_classifier` mixes spatially with a pair of 3x3 convs. This idea
  holds the tower shell (stem -> N residual + SqueezeExcite blocks ->
  value head) fixed and replaces only the per-block spatial-mixing
  operator with the `sparse_delta_accumulator` (SDA) primitive from
  `p013_sparse_delta_accumulator`. Source primitive math:
  `ideas/registry/p013_sparse_delta_accumulator/math_thesis.md`.

- Assumptions:
  1. The SDA primitive is well-defined as a shape-preserving operator
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

- Claimed advantage: If the SDA primitive carries a spatial mixing
  signal that conv and attention do not, dropping it into the BT4 block
  must lift held-out PR AUC (aggregate or on a target slice) versus the
  two baselines under the same tower, optimizer, and data. This is a
  controlled architecture-level test of "is SDA a better spatial mixer
  than conv or attention inside a fixed BT4 tower shell?", not a new
  primitive claim.

- Proof sketch: This is an empirical study, not a theorem. The well-
  definedness of the mixer is enforced at construction time by the BT4
  block's shape check (raises if `mixer(x).shape != x.shape`). The
  primitive-level math for SDA itself (analytical fixed point
  ``h = sum_i W[i]`` over the active piece-square set followed by a
  linear projection and a ClippedReLU, broadcast back to every square
  and fused with a per-square residual term to form an all-to-all
  low-rank spatial mix) is proven in the source primitive's math thesis
  and falsified by its own ablation grid. The make/unmake autograd
  contract that defines SDA at inference time is not expressible in a
  static-batch token mixer and is intentionally not attempted here.
  This folder inherits the analytical-fixed-point math and tests
  whether the resulting operator, used as a token mixer rather than as
  an additive head, transfers its signal through the BT4 tower.

- What is actually proven: The mixer is shape-preserving on
  `(B, C, 8, 8)` inputs and integrates with the BT4 block via the
  unified mixer registry. The accumulator sum followed by the
  ClippedReLU and broadcast-fuse is the only cross-square interaction
  this mixer performs, so the spatial-mix it implements is a closed-
  form all-to-all uniform mix of low rank. A forward + backward smoke
  test guards the mixer at registration time.

- What is only hypothesized: That replacing the conv mixer with the
  SDA mixer lifts PR AUC on at least one CRTK slice (most likely the
  `crtk_eval_bucket = equal` slice and slices where a global,
  accumulator-style summary of board occupancy is load-bearing)
  without regressing aggregate PR AUC by more than the matched-
  baseline tolerance.

- Failure cases:
  - The SDA mixer reduces to a noisy 1x1 conv plus a global mean
    inside the BT4 shell because the residual + SqueezeExcite path
    already absorbs the global-summary signal; the `conv` baseline
    matches the SDA variant within noise.
  - The mixer's all-to-all uniform mix is too low-rank to carry
    chess-relevant cross-square structure once the stem conv has
    compressed the simple_18 board into `C` channels; an `attention`
    baseline matches or beats SDA.
  - The make/unmake autograd path that gives SDA its inference-time
    advantage is absent at training time, so the static fixed-point
    mixer is fundamentally weaker than the source primitive and
    underperforms even when the underlying accumulator idea would help
    at inference.

# Mathematical Thesis

- Mathematical motivation: The BT4-style residual tower from
  `lc0_bt4_classifier` mixes spatially with a pair of 3x3 convs. This idea
  holds the tower shell (stem -> N residual + SqueezeExcite blocks ->
  value head) fixed and replaces only the per-block spatial-mixing
  operator with the `promotion_aware_head` (PAH) primitive from
  `i246_promotion_aware_head` (PFCT). Source primitive math:
  `ideas/registry/i246_promotion_aware_head/math_thesis.md`.

- Assumptions:
  1. The PAH primitive is well-defined as a shape-preserving operator
     `(B, C, 8, 8) -> (B, C, 8, 8)` under the
     `chess_nn_playground.models.architecture.bt4_mixers._base.Mixer`
     contract. In this mixer adaptation the "promotion site" is
     abstracted: every one of the 64 square-tokens is treated as a
     potential promotion site, and the four legal promotion types
     {Q, R, B, N} become four learned per-type affine + GELU transforms,
     selected by a per-token cross-attention head.
  2. The BT4 block wrapper (`mixer -> SqueezeExcite -> +residual ->
     ReLU`) is identical across all `a###_bt4_*_mixer` ideas and across
     the `conv` and `attention` baselines.
  3. The optimizer protocol, data contract (`simple_18`,
     `puzzle_binary`), and training budget are identical across all
     `a###` and baseline runs, so the only experimental variable is the
     mixer.

- Claimed advantage: If the PAH primitive carries a spatial mixing
  signal that conv and attention do not, dropping it into the BT4 block
  must lift held-out PR AUC (aggregate or on a target slice) versus the
  two baselines under the same tower, optimizer, and data. This is a
  controlled architecture-level test of "is PAH a better spatial mixer
  than conv or attention inside a fixed BT4 tower shell?", not a new
  primitive claim.

- Proof sketch: This is an empirical study, not a theorem. The well-
  definedness of the mixer is enforced at construction time by the BT4
  block's shape check (raises if `mixer(x).shape != x.shape`). The
  primitive-level math for PAH itself (deterministic four-way promotion
  fanout `F(p, x) in R^{4xd}` followed by a softmax cross-attention over
  the promoted-type rows) is proven in the source primitive's math
  thesis and falsified by its own shuffled-fanout / copy-baseline
  ablations. This folder inherits that math and tests whether the
  resulting operator, used as a token mixer over abstract per-square
  type-transform tokens rather than as a literal additive head over
  near-promotion pawns, transfers any of its signal through the BT4
  tower.

- What is actually proven: The mixer is shape-preserving on
  `(B, C, 8, 8)` inputs and integrates with the BT4 block via the
  unified mixer registry. A forward + backward smoke test guards the
  mixer at registration time.

- What is only hypothesized: That replacing the conv mixer with the
  PAH mixer lifts PR AUC on at least one CRTK slice (most likely the
  `crtk_tactic_motifs = promotion` slice, which is the largest per-slice
  PR-AUC gap motivating the source primitive) without regressing
  aggregate PR AUC by more than the matched-baseline tolerance.

- Failure cases:
  - The mixer adaptation abstracts away the literal pawn-substitution
    semantics: every square-token receives the same four learned type
    transforms, with no piece-plane gating. The fanout therefore
    collapses to a generic four-branch ensemble per token, and the
    `attention` baseline matches or beats PAH because the per-type
    embeddings carry no chess structure.
  - The promotion slice is only ~6% of positions, so even a perfect
    target-slice fix yields only ~+0.5pp aggregate. Aggregate PR AUC
    may show no detectable lift even when the slice-level claim is
    true; this idea must be judged on the slice, not the aggregate.
  - SqueezeExcite + residual + ReLU absorb most of the mixer output
    because the cross-attention pooled value has small magnitude
    relative to the residual path; the `conv` baseline matches the PAH
    variant within noise.

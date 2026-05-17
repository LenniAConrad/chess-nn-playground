# Mathematical Thesis

- Mathematical motivation: The BT4-style residual tower from
  `lc0_bt4_classifier` mixes spatially with a pair of 3x3 convs. This idea
  holds the tower shell (stem -> N residual + SqueezeExcite blocks ->
  value head) fixed and replaces only the per-block spatial-mixing
  operator with the `complex_amplitude_chess_network` (CAIO) primitive
  from `i247_complex_amplitude_chess_network`. Source primitive math:
  `ideas/registry/i247_complex_amplitude_chess_network/math_thesis.md`.

- Assumptions:
  1. The CAIO primitive is well-defined as a shape-preserving operator
     `(B, C, 8, 8) -> (B, C, 8, 8)` under the
     `chess_nn_playground.models.architecture.bt4_mixers._base.Mixer`
     contract. In this mixer adaptation each of the 64 square-tokens is
     lifted to a complex amplitude `z = rho * exp(i theta)` and the
     per-square interference response is *scattered back* onto the board
     (rather than pooled to a global fingerprint as in the source head).
  2. The four chess relation masks (king-zone adjacency, ray alignment,
     same square colour, file/rank adjacency) are closed-form
     `{0, 1}^{64x64}` buffers and are *not* learned, so any signal
     recovered by the mixer cannot be an artefact of free mask
     parameters.
  3. The BT4 block wrapper (`mixer -> SqueezeExcite -> +residual ->
     ReLU`) is identical across all `a###_bt4_*_mixer` ideas and across
     the `conv` and `attention` baselines.
  4. The optimizer protocol, data contract (`simple_18`,
     `puzzle_binary`), and training budget are identical across all
     `a###` and baseline runs, so the only experimental variable is the
     mixer.

- Claimed advantage: If the CAIO primitive carries a spatial mixing
  signal that conv and attention do not, dropping it into the BT4 block
  must lift held-out PR AUC (aggregate or on a target slice) versus the
  two baselines under the same tower, optimizer, and data. This is a
  controlled architecture-level test of "is CAIO a better spatial mixer
  than conv or attention inside a fixed BT4 tower shell?", not a new
  primitive claim.

- Proof sketch: This is an empirical study, not a theorem. The well-
  definedness of the mixer is enforced at construction time by the BT4
  block's shape check (raises if `mixer(x).shape != x.shape`). The
  primitive-level math for CAIO itself (complex amplitude lift with a
  chess-rule phase prior, masked pairwise interference outer product
  `z_u * conj(z_v) * exp(i beta_r)` under fixed relation masks, and the
  constructive / destructive / curl interference decomposition) is
  proven in the source primitive's math thesis and falsified by its own
  ablation grid. This folder inherits that math and tests whether the
  resulting operator, used as a token mixer with the per-square
  interference response scattered back to the board rather than pooled
  to a global head fingerprint, transfers its signal through the BT4
  tower.

- What is actually proven: The mixer is shape-preserving on
  `(B, C, 8, 8)` inputs and integrates with the BT4 block via the
  unified mixer registry. A forward + backward smoke test guards the
  mixer at registration time.

- What is only hypothesized: That replacing the conv mixer with the
  CAIO mixer lifts PR AUC on at least one CRTK slice (most likely the
  `crtk_eval_bucket = equal` slice and the `crtk_difficulty` upper tail
  where coherence-dependent tactical structure — pins, exchanges,
  mating-net interference between attacker and defender squares — is
  load-bearing) without regressing aggregate PR AUC by more than the
  matched-baseline tolerance.

- Failure cases:
  - The CAIO mixer reduces to a noisy conv inside the BT4 shell
    because the residual + SqueezeExcite path dominates the mixer
    output; the `conv` baseline matches the CAIO variant within noise.
  - The mixer adaptation loses two of the three rule-phase terms
    (piece colour, side-to-move) because the swappable mixer has no
    piece-plane semantics, leaving only the square-colour Z2 indicator.
    The remaining rule prior is too weak to drive any chess-aware
    structure, and the `attention` baseline matches or beats CAIO
    because the complex lift devolves into a generic learned pairwise
    interaction.
  - CAIO's per-call cost (a 64x64 masked complex outer product times
    four relations) inflates wall-clock enough that the matched-budget
    comparison is unfair; the baselines train for more effective
    optimizer steps inside the same wall-clock budget.

# Mathematical Thesis

- Mathematical motivation: A chess tactical position is more than the set
  of pieces and their attack relationships. It is also a *phase-coherent*
  state — pin attacks and defences point along the same line, exchanges
  align tempo with material, mating nets have constructive interference
  between attacker squares and destructive interference at defender
  squares. A real-valued scoring head can recognise *which* facts are
  present, but it cannot natively express *whether they line up*. The
  CAIO primitive lifts the per-square evidence into a complex amplitude
  whose phase carries the chess Z2 symmetry state (piece colour, tempo,
  square colour) and scores constructive vs destructive interference under
  fixed chess relation masks.

- Assumptions:
  1. Let `h_theta : R^(B, 18, 8, 8) -> R^(B, C, 8, 8)` be a learned spatial
     feature encoder operating on the simple_18 board.
  2. The per-square magnitude `rho = softplus(W_r h)` is positive and the
     per-square phase `theta = W_t h + theta_rule(square, side, piece,
     square colour)` is real-valued.
  3. The chess-rule phase contribution `theta_rule` is a learned linear
     combination of three closed-form chess indicators:
     piece colour, side-to-move, and square colour.
  4. Complex amplitudes `z = rho * exp(i theta)` are well-defined real
     differentiable functions of `(rho, theta)`, and complex backward
     propagation through `cos / sin` is supported in modern PyTorch.
  5. The relation masks `M_r in {0, 1}^(64, 64)` (king-zone adjacency,
     ray alignment, square-colour, file-rank adjacency) are closed-form
     and *not* learned, so any signal recovered by the primitive cannot
     be an artefact of free mask parameters.

- Claimed advantage:
  For each relation `r`,

      I_r(u, v) = Re(z_u * conj(z_v) * exp(i beta_r))
      D_r(u, v) = Im(z_u * conj(z_v) * exp(i beta_r))

  decomposes the pairwise amplitude interaction into a real (interference
  intensity) part and an imaginary (phase curl) part. Pooled mass

      constructive_r = sum_{u, v} M_r[u, v] * relu( I_r(u, v))
      destructive_r  = sum_{u, v} M_r[u, v] * relu(-I_r(u, v))
      curl_r         = sum_{u, v} M_r[u, v] * D_r(u, v)

  isolates *coherent* and *destructive* mass per relation. A 1x1 real
  bilinear relation head (the most natural real-valued control) computes
  `sum u^T W v` which collapses both parts into a single real bilinear
  scalar; CAIO retains the sign of the interference plus the phase curl,
  and is automatically Z2-equivariant under colour swap when the encoder
  weights are real.

- Proof sketch:
  1. The chess colour swap acts as complex conjugation on a chess-Z2
     parameterised amplitude with rule phase `pi * piece_colour`. Hence
     when the encoder weights are real, `z(color_flip(x)) =
     z(x).conj()` up to phase-rule fixed parameters. The squared
     `|conj_error| = || z(color_flip(x)) - z(x).conj() ||` is the
     primitive's diagnostic of how close the model is to this
     equivariance.
  2. Two unconstrained real channels `(u, v)` cannot equal a complex
     linear layer `W = X + iY` acting on `z = u + iv` unless the cross
     terms are tied: `(Au + Bv, Cu + Dv) = (Xu - Yv, Xv + Yu)` requires
     `A = D = X` and `B = -C = -Y`. Hence the CAIO complex layer is a
     *constrained* parameterisation that respects the U(1) action on the
     amplitude exactly, and a free-phase or real-only ablation collapses
     to a different operator class.
  3. The prototype `caio_prototype.py` verifies constructive interference
     between two same-colour pieces on adjacent squares (|A|^2 supra
     additive vs single-piece) and destructive interference between
     opposite-colour pieces on adjacent squares (|A|^2 sub additive vs
     same-colour pair). It also verifies that the autograd graph flows
     correctly through the complex `cos / sin` parameterisation.

- What is actually proven: The amplitude lift is autograd-compatible,
  Z2-equivariance under colour swap holds in closed form when the
  rule-phase parameters are at the chess-canonical values, and the
  interference summation produces non-trivial constructive / destructive
  masses on planted same-colour vs opposite-colour configurations. The
  unit tests in `tests/test_complex_amplitude_chess_network.py` exercise
  these properties on toy boards and verify no complex-dtype leakage into
  the trainer pipeline.

- What is only hypothesized: That the encoder + interference summation
  can be trained from puzzle_binary signal alone to recover load-bearing
  phase structure on at least two hard slices, and that the
  `real_only` / `random_phase` / `shuffle_relation_masks` ablations will
  lose most of any near-puzzle FP improvement. These are scout-scale
  empirical predictions; this folder is the implementation that makes
  them falsifiable, not a proof that they hold.

- Failure cases:
  - Phase collapse: the learned phase channel decays to zero, so the
    amplitudes become real and CAIO reduces to a relation-masked real
    bilinear head. The `real_only` ablation tests this directly.
  - Free-phase trivialisation: the chess-rule phase contribution is
    overridden by the learned phase logits. The `free_phase` ablation
    checks whether the rule tying is load-bearing.
  - Mask-shuffled lift: if the gains survive after the relation masks
    are randomly permuted (`shuffle_relation_masks` ablation), the
    primitive is acting as a parametric ensemble rather than as a
    chess-relation-aware interference probe, and should be dropped.

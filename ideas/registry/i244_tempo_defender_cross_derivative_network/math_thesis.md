# Mathematical Thesis

- Mathematical motivation: The puzzle_binary `crtk_eval_bucket = equal` slice
  is the only universally hard cohort across registered ideas. Material is
  balanced by definition, so a model that distinguishes puzzles from
  non-puzzles in this slice must rely on response-asymmetry signals rather
  than static evaluation. The Tempo-Defender Cross-Derivative (TDCD)
  primitive operationalises that response-asymmetry as the *mixed* partial
  of a learned scoring function under two chess-natural Z2 actions: tempo
  flip and per-defender removal.

- Assumptions:
  1. Let `phi_theta: simple_18 -> R^d` be a learned encoder operating on a
     current-board tensor.
  2. The tempo involution `sigma_T` inverts the side-to-move plane; it is
     its own inverse and a closed-form differentiable map on the input
     tensor. No FEN or external rule call is required.
  3. The defender-removal operator `delta_k` zeros the enemy piece planes
     at a square `s_k` selected by a learned saliency head from the same
     simple_18 board; it is a piece-wise deletion intervention and is its
     own inverse on positions where no defender stood at `s_k`.
  4. `sigma_T` and `delta_k` commute on the tensor encoding (each affects a
     disjoint subset of planes), so the cross-derivative is well-defined.

- Claimed advantage: For a true puzzle the tempo asymmetry `g_T = phi(x) -
  phi(sigma_T x)` is supported by *distributed* forcing structure, so
  `||tau_k|| = ||phi(delta_k x) - phi(sigma_T delta_k x)||` remains close to
  `||g_T||` for any single critical-defender removal. For a near-puzzle the
  tactic rests on one critical defender k*, so `||tau_{k*}||` collapses
  toward zero while `||g_T||` is large. The signed mixed partial
  `DeltaDelta_k = ||tau_k|| - ||g_T||` therefore separates the two regimes
  with a sign and magnitude pattern that neither first-order response can
  produce on its own.

- Proof sketch:
  1. By definition of partial derivatives in a finite-difference sense,
     `phi(sigma_T delta_k x) - phi(delta_k x) - phi(sigma_T x) + phi(x)`
     is the second-order discrete partial; in norm form this is exactly
     `tau_k - g_T`, and `DeltaDelta_k` is the signed length gap.
  2. For positions where defender k is necessary for the tactic, removing
     it destroys the forcing initiative under both colours, so
     `phi(delta_{k*} x) approx phi(sigma_T delta_{k*} x)` and
     `||tau_{k*}|| approx 0`, giving `DeltaDelta_{k*} approx -||g_T||`.
  3. For positions where no single defender is critical, removing any one
     defender preserves the tempo asymmetry, so
     `||tau_k|| approx ||g_T||` and `DeltaDelta_k approx 0`.
  4. The discriminator therefore has access to a class-separating signal
     that is invisible to any first-order tempo or defender response.

- What is actually proven: The cross-derivative is well-defined for the
  simple_18 board because the two operators act on disjoint tensor planes
  and commute. The unit tests `test_tempo_flip_is_involution`,
  `test_square_removal_zeros_enemy_planes_only`,
  `test_mixed_partial_zero_for_irrelevant_defender`, and
  `test_mixed_partial_collapses_for_critical_defender` exercise these
  algebraic and behavioural identities on toy boards.

- What is only hypothesized: That the saliency head can be trained from the
  puzzle_binary signal alone to identify the critical defender, and that
  the cross-derivative spectrum lifts equal-slice PR AUC by `>= 0.015`
  without regressing aggregate PR AUC by more than `0.005`. These are
  scout-scale empirical predictions; this folder is the implementation that
  makes them falsifiable, not a proof that they hold.

- Failure cases:
  - Saliency collapses to "always the king" or "always the highest-value
    piece"; the learned head then becomes a global perturbation and TDCD
    reduces to a noisy tempo-only signal.
  - The encoder `phi_theta` is too low-capacity to distinguish
    perturbation-induced changes from input noise; the cross-derivative
    spectrum becomes statistically indistinguishable from the
    main-effects-only spectrum (the `main_effects_only` ablation will
    match the full model, falsifying the central claim).
  - The puzzle_binary dataset has insufficient near-puzzles with a single
    critical defender; the signal exists but is not load-bearing for the
    aggregate metric.

# Ablations

- Ablation switches (selected via `config.model.ablation`):

| id  | config value              | what it changes |
|-----|---------------------------|-----------------|
| A1  | `main_effects_only`       | Zeros the per-defender DeltaDelta spectrum before it reaches the head, leaving the head with main-effect norms only. |
| A1b | `no_mixed_partial`        | Same effect as `main_effects_only`, kept as a named alias to map exactly to the source markdown's A1 falsifier. |
| A2  | `topk: 1 / 3 / 5`         | K sweep for the saliency-selected critical defenders. |
| A3  | not implemented in code   | Closed-form i189 typed mask vs learned saliency. Replicating this requires swapping `SaliencyHead` for the typed-mask path in `counterfactual_defender_dropout.DefenderDropoutMaskBuilder`; documented here as a future ablation rather than half-implemented. |
| A4  | `attacker_perturbation`   | `delta_k` removes the side-to-move's own piece instead of the enemy piece. Tests whether the cross-derivative is colour-asymmetric. |
| A5  | `null_board_perturbation` | Replaces per-defender removal with the global "all enemies removed" board for every k slot. Tests whether localised perturbation matters vs global centring (i041-style). |
| A6  | `skip_cross_derivative`   | Zeros `primitive_delta`'s gate input and forces gate = 0; the model behaves as the i193 baseline. Sanity: should not beat i193 alone. |
| A7  | not gated in code         | Tying `phi_theta` to frozen i193 weights vs training end-to-end. The current implementation uses a *separate* compact TDCD encoder; A7 would replace this with `self.trunk.feature_builder + encoder_clone` and freeze. Documented for future falsification rather than half-implemented. |
| extra | `shared_saliency_uniform` | Replaces learned saliency logits with zero (uniform softmax over enemy squares). Tests whether the learned saliency head is load-bearing. |
| extra | `fixed_zero_gate`         | Hard-codes the discriminator gate to zero. Sanity check: should reproduce i193 base. |

- What each ablation tests:
  - `main_effects_only` / `no_mixed_partial` (A1): is the *mixed* partial
    the source of any win, or does the head only need main-effect norms?
    Promotion threshold: A1 must lose at least `0.005` PR AUC on the
    `crtk_eval_bucket = equal` slice relative to `ablation: none`.
  - `attacker_perturbation` (A4): symmetry check. If A4 matches A1's lift,
    the cross-derivative is not specifically about defenders.
  - `null_board_perturbation` (A5): if global centring matches per-piece
    perturbation, the win is not about localised forcing structure.
  - `skip_cross_derivative` (A6): sanity check that the head cannot beat
    i193 by adding capacity alone.
  - `shared_saliency_uniform`: if uniform saliency matches the learned
    head, the saliency stage is not load-bearing.
  - K sweep (A2): K=1 isolates a single critical defender; K=5 buys more
    coverage at higher cost. Useful for cost/lift trade-off after the
    main-effects-only test passes.

- Falsification criteria (full pass requires all three):
  1. `crtk_eval_bucket = equal` slice PR AUC >= 0.832 (i193 0.817 + 0.015).
  2. Aggregate test PR AUC >= 0.871 (i193 - 0.005).
  3. No slice regresses by more than 0.01 relative to i193.

- Fail criteria (any one disqualifies, drop the primitive):
  - Equal-slice PR AUC improvement < 0.005 vs i193.
  - Aggregate test PR AUC < 0.85.
  - `main_effects_only` ablation matches `ablation: none` within 0.003 PR
    AUC on the equal slice — the cross-derivative is not load-bearing.
  - `shared_saliency_uniform` matches `ablation: none` — the learned
    saliency head is not load-bearing and the primitive reduces to a
    global tempo-only response.

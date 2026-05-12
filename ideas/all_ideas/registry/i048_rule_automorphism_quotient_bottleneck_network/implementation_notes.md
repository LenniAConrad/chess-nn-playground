# Implementation Notes

- Bespoke source: `src/chess_nn_playground/models/rule_automorphism_quotient.py` (`RuleAutomorphismQuotientNet`, builder `build_rule_automorphism_quotient_bottleneck_from_config`).
- Idea-local wrapper: `ideas/all_ideas/registry/i048_rule_automorphism_quotient_bottleneck_network/model.py`.
- Registry key: `rule_automorphism_quotient_bottleneck_network`.
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-21_0751_tuesday_pdt_automorphism_quotient.md`.
- Board-only model: it consumes the `simple_18` current-board tensor; CRTK, engine, verification, and source metadata stay reporting-only.
- The `Simple18AutomorphismOrbit` adapter is parameter-free and deterministic. It always emits `{I, C}` and additionally emits `{H, HC}` only on samples whose four castling planes are zero, exactly matching the rule-safe automorphism groupoid in `math_thesis.md`. The adapter fails closed when the channel schema is anything other than `simple_18`/18 channels unless explicitly opted out.
- The packet's central same-count pseudo-orbit falsifier is wired into the same model class via `pseudo_orbit=True`: it emits `[I, rank_flip, file_flip, rank_file_flip]` with all views unconditionally valid, preserving view count and nuisance statistics while breaking color/side/castling semantics. `use_file_mirror_if_castling_absent=False` recovers the color/turn-only ablation; `use_color_turn_reversal=False` reduces to a single-view augmentation control.
- Auxiliary diagnostics (orbit consistency, view-logit variance, VICReg variance/covariance, latent magnitude) are emitted by the default `forward(x)` so the puzzle-binary trainer can log them, and the un-aggregated tensors plus per-view logits are returned with `forward(x, return_aux=True)` for custom training paths that wire the full `L = L_CE + 0.1 L_orbit + 0.01 L_var + 0.01 L_cov + 0.05 L_rex` objective.

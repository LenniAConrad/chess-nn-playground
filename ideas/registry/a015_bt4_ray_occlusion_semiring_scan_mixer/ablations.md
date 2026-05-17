# Ablations — a015 BT4 Primitive Mixer (ray_occlusion_semiring_scan)

The mixer-internal ablations
(`uniform_transmittance`, `constant_direction`, `no_step_decay`,
`zero_delta`, `disable_gate`, `trunk_only`) are documented in
`ideas/registry/p010_ray_occlusion_semiring_scan/ablations.md` and are
run from the primitive folder. They are reused here only as secondary
evidence about *why* the mixer helps or hurts inside the BT4 tower.

## Switches at the architecture-study level

| Mode | What it tests |
|---|---|
| Mixer = `ray_occlusion_semiring_scan` (this idea) | Replace the conv pair with the occlusion-gated ray-semiring mixer inside the fixed BT4 tower. |
| Mixer = `conv` (sibling baseline) | Original BT4 spatial mixer. The aggregate-metric reference. |
| Mixer = `attention` (sibling baseline) | Generic dense attention mixer. Controls for "attention beats conv" without the chess-aware ray geometry. |
| Tower depth / channel / SE width | **Not ablated here**. Held fixed across the `a###_bt4_*_mixer` sweep. |

## Falsification criteria

Keep `a015` over the conv baseline only if all four hold on
puzzle_binary:

- Aggregate PR AUC delta over `bt4_conv_mixer` >= +0.005.
- CRTK class-1 matched-recall FP rate matches or beats
  `bt4_conv_mixer`.
- Wall-clock per epoch within 1.2x of `bt4_conv_mixer`.
- The lift over `bt4_attention_mixer` is at least as large as the lift
  over `bt4_conv_mixer` (the occlusion-gated ray geometry must be the
  source, not generic "attention beats conv").

Drop `a015` if any of those fail, or if the primitive-level
`uniform_transmittance` ablation matches `none` (the blocker-aware
prefix product was not load-bearing — the operator is just a
depthwise ray-conv with extra steps).

## Out-of-scope

- Re-tuning lr / schedule / SE width per mixer. The comparison is only
  valid if the tower and protocol are fixed.
- New ablation switches inside the mixer itself. Those belong in
  `p010` and would be inherited transitively.

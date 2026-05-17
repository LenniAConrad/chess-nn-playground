# Ablations — a014 BT4 Primitive Mixer (legal_move_graph_delta)

The mixer-internal ablations
(`random_typed_edges`, `shared_weight`, `no_normalization`,
`zero_delta`, `disable_gate`, `trunk_only`) are documented in
`ideas/registry/p009_legal_move_graph_delta/ablations.md` and are run
from the primitive folder. They are reused here only as secondary
evidence about *why* the mixer helps or hurts inside the BT4 tower.

## Switches at the architecture-study level

| Mode | What it tests |
|---|---|
| Mixer = `legal_move_graph_delta` (this idea) | Replace the conv pair with the typed legal-move graph mixer inside the fixed BT4 tower. |
| Mixer = `conv` (sibling baseline) | Original BT4 spatial mixer. The aggregate-metric reference. |
| Mixer = `attention` (sibling baseline) | Generic dense attention mixer. Controls for "attention beats conv" without the chess-aware adjacency. |
| Tower depth / channel / SE width | **Not ablated here**. Held fixed across the `a###_bt4_*_mixer` sweep. |

## Falsification criteria

Keep `a014` over the conv baseline only if all four hold on
puzzle_binary:

- Aggregate PR AUC delta over `bt4_conv_mixer` >= +0.005.
- CRTK class-1 matched-recall FP rate matches or beats
  `bt4_conv_mixer`.
- Wall-clock per epoch within 1.2x of `bt4_conv_mixer`.
- The lift over `bt4_attention_mixer` is at least as large as the lift
  over `bt4_conv_mixer` (typed legal-move adjacency must be the source,
  not generic "attention beats conv").

Drop `a014` if any of those fail, or if the primitive-level
`random_typed_edges` ablation matches `none` (the typed legal-move
geometry was not load-bearing).

## Out-of-scope

- Re-tuning lr / schedule / SE width per mixer. The comparison is only
  valid if the tower and protocol are fixed.
- New ablation switches inside the mixer itself. Those belong in
  `p009` and would be inherited transitively.

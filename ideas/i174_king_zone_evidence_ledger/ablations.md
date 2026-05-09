# Ablations

The packet's required ablations are exposed via `model.ablation`:

- `none` — main model with all three ledger banks and king-relative
  coordinate features.
- `no_king_relative` — drop the five king-relative coordinate
  channels so the gated pool sees only board features. Tests king
  anchoring.
- `random_king_anchor` — replace the real king anchors with
  deterministic per-batch random anchors. Tests real king
  semantics.
- `global_slots_only` — drop the per-king ledgers and feed only the
  global slot ledger to the head. Tests king-specific ledger value.
- `slot_count_sweep` — no-op structural flag for runs that vary
  `model.num_slots`. Detects bottleneck vs capacity (use with
  `num_slots ∈ {1, 2, 3, 5, 8, 16}` for a sweep).

Comparisons:

- LC0 BT4, NNUE, and the strongest registered idea runs on the same
  CRTK split and seeds.

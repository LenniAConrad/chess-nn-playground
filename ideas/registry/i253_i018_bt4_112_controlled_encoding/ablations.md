# Ablations

i253 is a controlled-encoding study, not a new learning hypothesis. The
ablation matrix below is structured around two axes the research markdown
explicitly requires: the encoding axis (`simple_18` vs `lc0_bt4_112`) and
the relation mode axis (`exact` / `confidence` / `hybrid`).

## Primary Comparison (12 cells, 3 seeds each)

| ID  | Encoding       | relation_mode | Other knobs | Hypothesis |
|-----|----------------|---------------|-------------|------------|
| P1  | simple_18      | exact         | -           | Matched-control row; baseline for the entire study. |
| P2  | lc0_bt4_112    | exact         | -           | Does BT4 raw encoding alone help i018 under fixed relation geometry? |
| P3  | simple_18      | confidence    | -           | Does learned edge importance help on simple18 plane content? |
| P4  | lc0_bt4_112    | confidence    | -           | Does BT4 plane content carry useful edge-importance signal? |
| P5  | simple_18      | hybrid        | lambda=0.25 | Does bounded augmentation help on simple18 plane content? |
| P6  | lc0_bt4_112    | hybrid        | lambda=0.25 | Does the full controlled architecture lift on BT4? |

## Falsifier Tranches (18 cells, 3 seeds each)

| ID  | Encoding       | relation_mode | Knob | Hypothesis |
|-----|----------------|---------------|------|------------|
| F1  | simple_18      | exact         | `scramble_exact_relations=true` | Geometry-is-load-bearing falsifier on simple18 (exact). |
| F2  | lc0_bt4_112    | exact         | `scramble_exact_relations=true` | Same falsifier on BT4 (exact). |
| F3  | simple_18      | hybrid        | `scramble_exact_relations=true` | Hybrid falsifier on simple18. |
| F4  | lc0_bt4_112    | hybrid        | `scramble_exact_relations=true` | Hybrid falsifier on BT4. |
| F5  | simple_18      | hybrid        | `augmentation_only=true`        | Is learned augmentation alone enough on simple18? |
| F6  | lc0_bt4_112    | hybrid        | `augmentation_only=true`        | Is learned augmentation alone enough on BT4? |

The exact-support and hybrid scramble rows must each drop PR-AUC by at
least `0.02` versus their intact matched row to count as a passed
falsifier. A drop below `0.01` is treated as rejection of the
load-bearing-geometry thesis for that row.

## Recommended Interpretability Ablations

Beyond the required 36-run base package, four extra ablations
substantially improve interpretability:

| ID  | Switch | What it tests |
|-----|--------|---------------|
| I1  | `simple18_native18` (18-channel raw) vs `simple18_padded112` exact-only row | Confirms the 112-pad trick is inert. If they match, the padded-control comparison is valid. |
| I2  | BT4 exact with currently-dead history/repetition/reserved planes hard-masked to zero at the adapter output | Should be identical to vanilla BT4 exact today; if not, the exporter is leaking signal through dead planes. |
| I3  | Remove signed file coordinates from `SquareTokenEncoder` for one seed per encoding | Checks whether convention drift between simple18 canonicalisation and BT4's already-canonical layout is leaking through coordinate features. |
| I4  | BT4 with only the nontrivial aux planes retained alongside the current-board piece planes (drop the all-zeros aux slots) | Is the BT4 gain coming from a few load-bearing aux symbols or from the 112-plane format itself? |

`I1` is the most important interpretability control. If it fails, the
padded-vs-native difference is itself a confound and the matched-budget
contract has to be revisited.

## Keep / Drop Rule

Keep i253 as the canonical BT4-encoding study if all of these hold:

- Relation scramble drops PR-AUC by at least `0.02` on every matched
  row (the i018 falsifier threshold).
- Hybrid augmentation-only is clearly below intact hybrid (the
  augmentation head must be a residual, not a replacement).
- At least one BT4 row beats its matched simple18 row by `+0.003`
  PR-AUC, or reduces near-puzzle false positives at matched recall by
  `1%`, without slice regressions.

Drop the BT4 conclusion (not the architecture) if:

- BT4 wins are only present in augmentation-only rows. That means the
  learned augmentation is doing chess geometry's job, which contradicts
  the controlled design.
- Confidence rows match exact rows on BT4 but not on simple18 by more
  than seed-to-seed noise. That would imply the relation head is
  consuming metadata-like content of the BT4 aux planes; rerun with the
  dead-plane control (I2) to confirm.
- BT4 wins only show up at a single seed. That is not a controlled
  encoding effect; it is split noise.

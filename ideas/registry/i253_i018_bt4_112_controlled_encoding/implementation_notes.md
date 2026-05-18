# Implementation Notes

- Central code: `src/chess_nn_playground/models/trunk/oriented_tactical_sheaf_controlled_encoding.py`
  (`OrientedTacticalSheafControlledEncodingNet`,
  `BoardStateAdapterControlled`,
  `TacticalIncidenceAugmentationBuilder`,
  `RelationConfidenceHead`,
  `build_i018_bt4_112_controlled_encoding_from_config`).
- Idea-local wrapper: `ideas/registry/i253_i018_bt4_112_controlled_encoding/model.py`
  (`build_model_from_config`).
- Registry key: `i018_bt4_112_controlled_encoding`.
- Parent idea: `i018 oriented_tactical_sheaf_laplacian` (reused intact
  via the imported `BoardStateAdapter`, `TacticalIncidenceBuilder`,
  `SquareTokenEncoder`, `SheafDiffusionBlock`, `TriadDefectPool`,
  `RELATION_NAMES`).

## What is new

- `BoardStateAdapterControlled` standardises the raw square pathway at
  112 channels for both encodings (zero-padded for `simple_18`, native
  for `lc0_bt4_112`). The piece-state path remains exact, so relation
  construction is encoding-aware but never falls back to a learned probe.
- `TacticalIncidenceAugmentationBuilder` wraps i018's exact
  `TacticalIncidenceBuilder` and registers a static
  `(R, 64, 64)` tensor of relation-specific geometric template
  supersets. Slider relations (rook, bishop, queen) get the full
  geometric ray template; attacker/defender relations use the slider
  geometric superset; knight/king/pawn/pin-candidate relations use
  their legal geometric shape.
- `RelationConfidenceHead` produces low-rank source/target codes per
  relation plus a shared pre-projection. Confidence and augmentation
  share the same pre-projection but use independent linear maps. Output
  logits are passed through sigmoid before being multiplied by the exact
  mask (and, in hybrid mode, by the template superset).
- Three relation modes are selected by `model.relation_mode in
  {"exact", "confidence", "hybrid"}`. Two falsifier switches
  (`model.scramble_exact_relations`, `model.augmentation_only`) cover
  the rows the research markdown explicitly requires.

## Parameter Budget Validation

The base-scale parameter counts match the research markdown exactly
(verified by `chess_nn_playground.models.registry.build_model`):

| Variant     | Total params |
|-------------|-------------:|
| `exact`     |       94,371 |
| `confidence`|       99,487 |
| `hybrid`    |      102,763 |

That is the matched-architecture, matched-budget contract the
controlled comparison requires.

## What is reused from i018

The forward pass of i253 is structurally identical to i018 once the
controlled mask is resolved: same encoder fusion, same depth-2 sheaf
diffusion, same triad-defect pool, same readout. The i018 diagnostics
(`mechanism_energy`, `sheaf_tension`, `transport_imbalance`,
`symmetry_residual`, `topology_pressure`, `ray_language_energy`,
`information_surprisal`, `sparse_certificate_energy`,
`rank_file_imbalance`, `king_ring_pressure`, `reply_pressure`,
`defense_gap`, `triad_defect_energy`, `pin_pressure`) are emitted
unchanged. Two new diagnostics are emitted in non-exact modes:
`controlled_confidence_mean` and (hybrid only) `controlled_augmentation_mean`.

## Falsifier wiring

`scramble_exact_relations` applies a per-(batch, relation) random
permutation of mask columns. In hybrid mode the same scramble is applied
to the template supersets so the comparison is apples-to-apples.

`augmentation_only` is only meaningful in hybrid mode. It drops the
exact relation masks and confidence path, leaving only
`clamp(lambda * T_r * sigmoid(A_r), 0, 1)` as the relation weight. The
exact relation masks themselves are still computed (the diagnostics
depend on them) but they are not used to weight the sheaf diffusion.

## Inputs not used

The model does not look at CRTK metadata, source labels, verification
flags, engine evaluations, Stockfish scores, principal variations, or
any report-only metadata. It only sees the board tensor for the
configured encoding, which is the same contract i018 enforces.

## How to run a controlled row

The default `config.yaml` is the `simple18 / exact` control row. The
other 5 primary cells of the research markdown's 36-run base matrix are
single-line config edits:

```
data.encoding: lc0_bt4_112      # BT4 input
model.input_channels: 112       # match encoding
model.relation_mode: confidence # or hybrid
```

Falsifiers (per row):

```
model.scramble_exact_relations: true
model.augmentation_only: true   # hybrid only
```

The shared training header in `config.yaml` already follows the research
markdown (`epochs: 30`, `min_epochs: 15`, `batch_size: 192`,
`early_stopping_patience: 8`, `monitor: pr_auc`, ReduceOnPlateau LR
schedule). Repeat seeds 42 / 43 / 44 are owned by the trainer.

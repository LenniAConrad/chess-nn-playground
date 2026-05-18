# Implementation Notes

- Central code: `src/chess_nn_playground/models/trunk/oriented_tactical_sheaf_efficient_xxl.py`
  (`OrientedTacticalSheafEfficientXXLNet`,
  `EfficientSheafDiffusionBlock`,
  `DEFAULT_RELATION_GROUPS_4`,
  `build_oriented_tactical_sheaf_efficient_xxl_from_config`).
- Idea-local wrapper: `ideas/registry/i254_efficient_i018_scale_xxl/model.py`
  (`build_model_from_config`).
- Registry key: `efficient_i018_scale_xxl`.
- Parent idea: `i018 oriented_tactical_sheaf_laplacian`. The wrapper
  subclasses `OrientedTacticalSheafNet` and only swaps the block
  list; every other module is inherited.

## What is new

- `EfficientSheafDiffusionBlock` supports two restriction modes via
  the `restriction_mode` argument:

  - `full` (default): the parameter shape and forward semantics are
    identical to i018's `SheafDiffusionBlock`. State-dict layout
    matches i018 exactly so i018 weights load via `strict=True`.
  - `grouped_lowrank`: each restriction map is parameterised as
    `I + U_g(r) diag(a_r) V_g(r)^T` with group-shared bases. The
    materialised `(R, s, s)` tensor is recomputed each forward, so
    the rest of the block reduces to the same matrix products as the
    full case.

- `DEFAULT_RELATION_GROUPS_4` is the canonical 4-group partition of
  the 12 typed relations used by the grouped low-rank mode:
  `attack / defense / ray / pin`. Custom partitions can be supplied
  through `model.relation_groups` in the config.

- The block forward wraps the per-relation loop in
  `torch.profiler.record_function("i254/per_relation_loop")` and the
  full forward in `torch.profiler.record_function(
  "i254/efficient_sheaf_block")` so the recommended profile-first
  protocol can see exactly where time is spent. These scopes are
  no-ops outside `torch.profiler.profile`.

- The wrapper exposes `compile_model` and `fuse_incidence` config
  flags for completeness but does not act on them in the default
  forward; they are stored as attributes so a later execution branch
  can read them. The research markdown explicitly requires the first
  XXL benchmark to be a capacity-only run.

## What is reused from i018

The full forward path is inherited from `OrientedTacticalSheafNet`,
which calls `BoardStateAdapter`, `TacticalIncidenceBuilder`,
`SquareTokenEncoder`, the block loop, `TriadDefectPool`, and the
readout head in the same order, with the same diagnostic dict. The
falsifier knob `model.scramble_relations` continues to apply a
degree-preserving relation scramble.

## Parameter Budget Validation

At the recommended first-XXL scale
(`channels=160, hidden_dim=320, depth=4, stalk_dim=8, dropout=0.1`),
the `full` restriction mode has **785,217 parameters**. This exactly
matches the research markdown's static estimate of "about 785k
parameters, roughly 1.66x current scale_xl". The test suite asserts
this exact count.

The grouped-lowrank variant at the same scale with `G=4`, `k=4`
recovers most of those parameters in the readout and width path; the
restriction-map saving is about 928 params per block, ~3.7k total at
`depth=4`. The point of grouped low-rank at `s=8` is *not* the
parameter saving but a safer stalk-scaling path; at `s=12` the
restriction-map saving becomes much more significant.

## i018 Parity Guarantee

The test `test_i254_full_mode_matches_i018_when_loaded` builds i018
and i254 at the same base scale, loads i018 weights into i254 with
`strict=True`, and asserts zero logit / sheaf_tension diff on a fixed
input. This is the explicit, testable form of the i018-thesis-
preserving claim.

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
Stockfish scores, principal variations, and any report-only metadata
are *not* consumed by the model. The contract is identical to i018.

## How to run a controlled row

The default `config.yaml` is the recommended first XXL cell. Other
rows of the i254 ablation matrix are single-line config edits:

```text
# Depth-only diagnostic
model.depth: 6

# Stalk-only diagnostic (full maps)
model.stalk_dim: 12

# Grouped low-rank stalk
model.stalk_dim: 12
model.restriction_mode: grouped_lowrank
model.restriction_rank: 4

# Capacity falsifier (matched i018 scale_xl)
model.channels: 128
model.hidden_dim: 192
model.depth: 4

# Optional execution branch (after the parity ladder passes)
model.compile_model: true
```

The shared training header in `config.yaml` is a slimmed-down i018
paper-grade recipe with `batch_size: 128` (per the research markdown's
memory analysis). Repeat seeds 42 / 43 / 44 are owned by the trainer.

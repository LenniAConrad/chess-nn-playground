# Implementation Notes

- Central code: `src/chess_nn_playground/models/trunk/pin_xray_overload_sheaf.py`
  (`PinXrayOverloadSheafNet`, `TacticalIncidenceBuilderV2`,
  `FastSheafDiffusionBlockV2`, `_make_single_screen_bank`,
  `build_pin_xray_overload_sheaf_from_config`).
- Idea-local wrapper: `ideas/registry/i252_pin_xray_overload_sheaf/model.py`
  (`build_model_from_config`).
- Registry key: `pin_xray_overload_sheaf`.
- Parent ideas: `i018 oriented_tactical_sheaf_laplacian`,
  `i249 oriented_tactical_sheaf_fast`.

## What changed vs i018 / i249

`PinXrayOverloadSheafNet` is a standalone `nn.Module` rather than a
subclass of `OrientedTacticalSheafNet`; it reuses i018's
`BoardStateAdapter`, `SquareTokenEncoder`, and `TriadDefectPool`
unchanged. The reasons for not subclassing:

- the relation count differs (`22` vs `12`), so the diffusion blocks
  and the readout head have different shapes and cannot share weights;
- `TriadDefectPool` accepts an `incidence` whose first four planes are
  `[us_attacks_them_piece, them_attacks_us_piece, us_defends_us_piece,
  them_defends_them_piece]`; both i018 and i252 satisfy that, so a
  shallow `_TriadIncidenceAdapter` is sufficient to feed it without
  inheriting the i018 dataclass shape verbatim.

The `TacticalIncidenceBuilderV2` class *does* subclass i018's
`TacticalIncidenceBuilder` and reuses every i018 buffer
(`rook_ray`, `bishop_ray`, `knight`, `king`, `king_zone`, `our_pawn`,
`their_pawn`, `between`, `rank_one_hot`, `file_one_hot`, `pin_king`,
`pin_blocker`, `pin_slider`, `pin_line`, `pin_clear`). It adds five
new buffers for the single-screen bank plus a piece-value table.

## Single-screen template bank

`_make_single_screen_bank` enumerates every ordered triple
`(s, c, r)` aligned on a rook or bishop ray with `c` strictly between
`s` and `r`. On the 8x8 board this gives exactly `2576` templates
(`1792` rook-aligned + `784` bishop-aligned), matching the count in the
i252 packet. The build is one-shot at module construction; the per-batch
runtime is one matrix-vector clear-product plus four scatter_add passes.

`screen_line` carries the line type per template (`0` = rook, `1` =
bishop). Skewer ordering uses `mu_c > mu_r` with king valued at `100`
so a king never qualifies as a skewered rear piece.

## Side-specific pins

`TacticalIncidenceBuilderV2._side_specific_pin` decomposes i018's
symmetric pin computation into two side-specific scatter passes. The
result is `pin_us` (our slider absolutely pins their piece) and
`pin_them` (their slider absolutely pins our piece). The symmetric
`pin_mask` from i018 is still emitted; the new side-specific masks are
used internally for the pinned-defender mass and the
discovered-attack self-pin filter.

## Numerical guard

`TacticalIncidenceBuilderV2` produces a `relation_masks` tensor whose
first 12 planes are bit-identical to i018's `TacticalIncidenceBuilder`
on the same board (verified by direct comparison on synthetic boards).
This is the contract that justifies inheriting the i018 sign convention
on those 12 planes.

`PinXrayOverloadSheafNet` is *not* a strict numerical extension of i018
at init because the relation count differs. The closest semantic
equivalents are:

- `family_collapse: true`: still uses 22 planes but forces the new ten
  to a generic dependency average, so the model becomes an i018-like
  graph plus a single extra plane duplicated across the new slots.
- `scramble_new_only: true`: every i018 plane is preserved exactly and
  only the new ten are randomized.

## Module shapes and budget

Base scale (`channels=64`, `hidden_dim=96`, `depth=2`, `stalk_dim=8`):

- i018 parent: ~91k parameters.
- i252 (this idea, base scale): about +6k parameters over i018. The
  growth is dominated by the diffusion block's per-relation restriction
  maps (`R * 2 * stalk_dim^2 + R` parameters per block) and a small
  readout widening (`(22 - 12) * 4 = 40` extra readout dims).

## Optional config knobs

- `use_triads` (default true): inherits i018's triad-defect pool.
- `scramble_relations` (default false): i018-style global topology
  scramble.
- `scramble_new_only` (default false): dependency-only scramble (F3).
- `family_collapse` (default false): generic dependency average over
  the 10 new planes (F4).

## Behaviour with falsifier flags

`scramble_relations: true` and `scramble_new_only: true` are mutually
exclusive in spirit but both honored (the global scramble dominates if
both are set). `family_collapse: true` interacts cleanly: after the
mask is collapsed, the diffusion runs on the resulting 22-plane tensor
just as in the default path.

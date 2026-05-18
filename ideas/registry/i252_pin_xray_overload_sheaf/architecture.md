# Architecture

`Pin / X-Ray / Overload Sheaf` keeps i018's side-to-move-oriented
tactical sheaf spine and the i249-style algebraic diffusion path, and
upgrades the relation graph from 12 to 22 typed planes. The new ten
planes encode bounded pairwise tactical dependencies (x-ray, skewer,
discovered attack, attacks against pieces with pinned defenders, and
attacks on overloaded defenders, each in `us` / `them` mirrors).

## Implementation Binding

- Registered model name: `pin_xray_overload_sheaf`
- Source implementation:
  `src/chess_nn_playground/models/trunk/pin_xray_overload_sheaf.py`
- Idea-local wrapper:
  `ideas/registry/i252_pin_xray_overload_sheaf/model.py`
- Parent ideas: `i018 oriented_tactical_sheaf_laplacian`,
  `i249 oriented_tactical_sheaf_fast`.

## Source research

Source packet:
`ideas/research/packets/classic/i252_pin_xray_overload_sheaf.md`. The
packet argues for *targeted* graph enrichment for i018's weakest motif
slices and explicitly avoids changing the sheaf operator class or the
trainer contract.

## Modules

`BoardStateAdapter` and `SquareTokenEncoder` are inherited unchanged
from i018. `TriadDefectPool` is inherited unchanged but wrapped by the
local `_TriadIncidenceAdapter` so it can read `our_attack`,
`them_attack`, `our_piece`, `them_piece`, and the first four planes of
`TacticalIncidenceV2` without depending on i018's `TacticalIncidence`
dataclass shape.

`TacticalIncidenceBuilderV2` (extends i018's `TacticalIncidenceBuilder`)
introduces three new buffers in addition to the inherited i018 geometry:

- `screen_source`, `screen_screen`, `screen_rear`, `screen_line`:
  per-template `(s, c, r, line)` indices of the single-screen template
  bank `T_1`.
- `screen_clear`: the clear mask `Q^1` per template (squares strictly
  between `s` and `r`, excluding the screen `c`).
- `piece_value_table`: heuristic piece values used for skewer ordering
  and overload weighting.

On an 8x8 board the template bank has exactly `2576` ordered triples
(`1792` rook-aligned + `784` bishop-aligned), matching the count in the
i252 packet.

The builder computes the same 12 i018 planes first, derives side-
specific absolute pin masks `P_us` and `P_them` from the existing pin
bank, evaluates the single-screen bank once per batch to produce
template-clear vector `Gamma_m`, and assembles the new planes:

- `us_xray_them_piece`: scatter of `Gamma_m * S^us_{s_m,l_m} *
  occupancy[c_m] * them[r_m]` into `(s_m, r_m)`.
- `us_skewer_them_piece`: scatter of `Gamma_m * S^us_{s_m,l_m} *
  them[c_m] * them[r_m] * 1[mu(c_m) > mu(r_m)]` into `(s_m, r_m)`.
- `us_discovered_attack_candidate`: scatter of `Gamma_m *
  S^us_{s_m,l_m} * U^{us,nonking}_{c_m} * them[r_m] * (1 - pin_us[c_m])`
  into `(c_m, r_m)`.
- `us_attacks_them_piece_with_pinned_defender`: attack-to-target plane
  scaled by `sum_d Def^{them,nonking}_{d,r} * pin_them[d]`.
- `us_attacks_them_overloaded_piece`: attack-to-target plane scaled by
  `sum_d Def^{them,nonking}_{d,r} * second_largest_r (Def^{them}_{d,r}
  * nu_r * 1[any us attacker on r])`.

All `them_*` mirrors are computed by swapping us / them.

`FastSheafDiffusionBlockV2` is the i249 algebraic block parameterised
by `relation_count` and an explicit `relation_signs` tensor (instead of
the 12-literal hardcode). Sign assignment keeps `+1` for the same-side
defense planes (`us_defends_us_piece`, `them_defends_them_piece`) and
`-1` for every other plane, including all ten new dependency planes.

The readout head sizes itself from `len(RELATION_NAMES_V2) = 22` and is
otherwise identical in shape to i018's head.

## Diagnostics

All per-family diagnostic readouts use the `RELATION_FAMILIES` and
`RELATION_INDEX` name lookup, so adding or removing relations only
requires updating the names tuple and the families dict. The forward
emits the i018 diagnostic bundle plus five new pressure scalars:
`xray_pressure`, `skewer_pressure`, `discovered_pressure`,
`pinned_defender_pressure`, `overload_pressure`.

`pin_pressure` now uses the named relation `king_ray_pin_candidate`
rather than literal index `11`; `ray_language_energy` averages over the
named `(bishop_ray_visible, rook_ray_visible, queen_ray_visible)`
indices; `king_ring_pressure` sums the two `*attacks_empty_near_king`
densities. These match the i018 quantities by construction on the same
relation order.

## Optional ablations / falsifiers

Wired via config (no code edits required):

- `scramble_relations: true` -- inherited from i018: degree-preserving
  random column permutation on every plane. This is the topology
  falsifier from the i018 paper-grade comparison.
- `scramble_new_only: true` -- the dependency-only scramble called for
  in the i252 packet: scrambles only the 10 new planes, leaves the
  original 12 untouched.
- `family_collapse: true` -- replaces the 10 new typed planes by a
  single averaged generic-dependency plane duplicated across the new
  slots; tests whether typing the dependency family matters.

Not exposed as config flags but documented in `ablations.md`: no
pinned-defender planes, no overload planes, value-blind skewer/overload,
no self-pin legality filter on discovered planes. Each requires a small
forward edit in `_dependency_planes`.

## Contract

- Input: `(B, 18, 8, 8)` board tensor only; same `simple_18` encoding as
  i018. CRTK / source / verification metadata is reporting-only and
  never enters the model.
- Output: `dict` with `logits` of shape `(B,)` and the i018 diagnostic
  bundle plus five new pressure diagnostics (see above).
- Symmetry: only the side-to-move canonicalization (color swap +
  180-degree rotation) is applied, exactly as in i018.

## Numerical guarantees

Because the relation count differs from i018, the model is *not* a
strict numerical extension of i018 at init -- the readout dimension and
the diffusion blocks have a different shape, so they cannot share
weights. Instead the design guarantees:

- The first 12 planes of `TacticalIncidenceV2.relation_masks` are
  bit-identical to i018's `TacticalIncidenceBuilder` output on the same
  board.
- `family_collapse: true` produces a model that still uses 22 planes but
  with new ones forced to a generic dependency average; this is the
  semantic sibling of "i018 + a single extra dependency plane" used by
  the falsifier study.

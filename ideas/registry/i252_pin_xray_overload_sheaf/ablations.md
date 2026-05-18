# Ablations

i252 is a narrow extension of i018 (and i249) with one new structural
object (a single-screen template bank `T_1` plus side-specific pin
masks) and one new relation family (10 typed dependency planes). The
ablations are designed so any positive aggregate result can still be
traced back to a specific cause.

## Required ablations

| ID | Switch | What it tests | Interpretation |
|---|---|---|---|
| F1 | i018 (`oriented_tactical_sheaf_laplacian`) or i249 fast variant | Parent baseline | Reference point. |
| F2 | `model.scramble_relations: true` | Degree-preserving topology scramble (inherited from i018) | Required topology falsifier; drop of `>= 0.02` PR-AUC keeps the typed-topology claim alive. |
| F3 | `model.scramble_new_only: true` | Scramble only the 10 new dependency planes | If matched within seed noise of full i252, the new family is decorative. |
| F4 | `model.family_collapse: true` | Replace the 10 new planes by one generic averaged dependency plane | If matched within seed noise of full i252, x-ray / skewer / discovered / overload / pinned-defender typing is unnecessary. |

F3 and F4 are mutually exclusive in spirit; pick one per falsifier run.

## Optional plane-family ablations

These zero out individual new families inside `TacticalIncidenceBuilderV2`
by editing the dependency-plane assembly in
`pin_xray_overload_sheaf.py::_dependency_planes`. They are not exposed
as config flags yet.

- No pinned-defender planes: zero out `us_attacks_them_piece_with_pinned_defender`
  and `them_attacks_us_piece_with_pinned_defender`.
- No overload planes: zero out `us_attacks_them_overloaded_piece` and
  `them_attacks_us_overloaded_piece`.
- Value-blind skewer / overload: drop the `mu_c > mu_r` check from the
  skewer formula and the `nu_r` factor from the overload mass.
- No self-pin legality filter on discovered planes: drop the
  `(1 - pin_us_self_c)` factor from the discovered-attack formula.

The matching slice deltas are the falsifier: removing the pinned-
defender planes should mostly hurt `pin` and overload-like slices;
removing the overload planes should mostly hurt the `overload` slice;
removing the self-pin filter on discovered should make discovered fire
on illegal screens and likely hurt `discovered_attack`.

## Keep / drop rule

Treat i252 as a meaningful improvement over i018 / i249 only if all of
the following hold:

- overall PR AUC is not worse than i249-fast by more than `0.003`;
- matched-recall (`0.80` or `0.85`) near-puzzle false positives
  improve, OR mean PR AUC across `pin`, `skewer`, `overload`,
  `discovered_attack` slices rises by `>= 0.010`;
- F2 (topology scramble) still drops test PR-AUC by `>= 0.02`;
- F3 (new-plane scramble) loses most of the dependency-family lift.

Drop i252 if any of the following hold:

- F3 (`scramble_new_only`) matches full i252 within seed noise;
- F4 (`family_collapse`) matches or beats full i252 -- the typed
  dependency family is then unnecessary;
- F2 (topology scramble) drop falls below `0.01` -- the typed
  topology claim has decayed and the family must be re-examined first.

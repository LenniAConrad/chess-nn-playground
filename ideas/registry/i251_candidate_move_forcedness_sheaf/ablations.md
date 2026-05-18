# Ablations

i251 is a narrow extension of i018 with one new structural object
(a deterministic candidate-move bottleneck) and one new fusion rule
(a gated additive delta on the i018 logit). The ablations are designed
so any positive aggregate result can still be traced back to a specific
cause.

## Required ablations

| ID | Switch | What it tests | Interpretation |
|---|---|---|---|
| F1 | i018 (`oriented_tactical_sheaf_laplacian`) | Parent baseline | Reference point. |
| F2 | `model.scramble_relations: true` | Degree-preserving topology scramble (inherited from i018) | Required topology falsifier; drop of `>= 0.02` PR-AUC keeps the topology claim alive. |
| F3 | `model.disable_move_branch: true` | Skip the move branch entirely | If matched within seed noise of full i251, the move branch is not load-bearing. |
| F4 | `model.flat_move_pool: true` | Force uniform pool weights on valid candidates | If matched within seed noise of full i251, the top-k forcedness bottleneck is unnecessary. |
| F5 | `model.top_k: 1` | Hard single-move pool | Tests whether the model needs the full set of forced candidates or just the strongest. |
| F6 | `model.top_k: 32` (or higher) | Looser pool | Tests whether the bottleneck is meaningful or pooling becomes diffuse and helps anyway. |
| F7 | Random move set (replace `score` with `randn`) | Whether the deterministic scoring is what matters | Requires a one-line forward edit; not enabled via config. |
| F8 | No-sheaf-summary (zero `psi_j`) | Whether the i018 local context is what makes the branch work | If matched within seed noise of full i251, the move branch is a side module, not i018-compatible. Requires a small forward change; not enabled via config. |

F7 and F8 are documented here for honesty and reproducibility, but they
are not exposed as config flags in the current implementation. Adding
them would require a small change to `forward` and a dedicated config;
do that only if F3 / F4 already pass (full i251 beats both disabled and
flat).

## Optional feature-level ablations

These zero out individual flag families in `CandidateMoveBuilder`. They
are not exposed as config flags yet; the indices are listed in
`candidate_move_forcedness_sheaf.py` so a focused experiment can mask
them in place.

- No check flag: zero out flag index 0 (`gives_check`).
- No capture flag: zero out flag index 1 (`is_capture`).
- No pin flags: zero out flag indices 2-3 (`source_pinned`,
  `pin_aligned`).
- No king-zone flag: zero out flag index 4 (`enters_their_king_zone`).
- No defended-target flags: zero out flag indices 5-6
  (`target_defended_raw`, `target_defended_unpinned`).
- No promotion flags: zero out flag indices 7-8 (`promotion`,
  `underpromotion`).

The matching slice deltas are the falsifier: removing the check flag
should mostly hurt `mate_in_1`; removing promotion flags should mostly
hurt the promotion / underpromotion CRTK slices; removing pin flags
should mostly hurt overload / deflection-like motifs.

## Keep / drop rule

Treat i251 as a meaningful improvement over i018 only if both are true:

- mean test PR-AUC across seeds 42, 43, 44 is at least `+0.003` over
  i018 at base scale, OR matched-recall (`0.80` or `0.85`) near-puzzle
  false positives are reduced by `>= 1%` absolute without compensating
  regressions on precision or puzzle recall;
- F2 (topology scramble) still drops test PR-AUC by `>= 0.02`.

Drop i251 if any of the following hold:

- F3 (`disable_move_branch`) matches full i251 within seed noise;
- F4 (`flat_move_pool`) matches or beats full i251 -- the bottleneck
  is then unnecessary;
- F2 (topology scramble) drop falls below `0.01` -- the typed
  topology claim has decayed and the family must be re-examined first.

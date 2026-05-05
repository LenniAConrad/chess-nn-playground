# Dykstra-LCP Improvement Audit

Date: 2026-04-29 Asia-Shanghai
Audited run: `results/20260428_162513_idea_i012_dykstra_lcp_lc0bt4`

## Findings

The trained A0 run is valid but needs improvement before another serious attempt. It does not beat the LC0 BT4 baseline or VetoSelect v2, and its main failure is verified near-puzzle false positives.

Two code-level issues were found:

- `decay_violation` was effectively dead because the projector stored the first projection-group residual from each cycle instead of the mean cycle residual. In the trained artifact, `decay_violation` is zero for all 45,000 test rows.
- `_role_masks` assumed the first six piece planes were friendly. That is correct for `lc0_bt4_112`, because the encoding is side-to-move canonical, but it was wrong for `simple_18` smoke/test usage. The mask path now handles `simple_18` with the side-to-move plane.

The larger modeling issue remains:

- Projection diagnostics are meaningful but too weak as a near-puzzle discriminator. On test, `-projection_distance` gets ROC AUC about 0.753 for binary puzzlehood and about 0.693 for near-puzzle-vs-puzzle. The learned `prob_1` is much stronger, so the solver trace helps explain behavior but is not yet strong enough to replace a veto/select mechanism.

## Needed Improvement

The next version should not be another plain Dykstra classifier. The best target is a hybrid:

- keep Dykstra diagnostics as solver features;
- add a VetoSelect-style accepted/rejected positive-evidence head;
- train with self-mined hard-negative decoys, using low projection distance plus high raw puzzle evidence to identify near-puzzle-like negatives;
- increase or rescale the negative projection margin, because the current margin `0.20` is far above observed projection distances and the regularizer weight is too small to shape the representation strongly.

## Patch Applied

- Fixed cycle residual tracking for `decay_violation`.
- Made role masks side-to-move-aware for `simple_18` while preserving the LC0 BT4 board-only benchmark behavior.
- Added a unit test for `simple_18` role-mask side-to-move handling.


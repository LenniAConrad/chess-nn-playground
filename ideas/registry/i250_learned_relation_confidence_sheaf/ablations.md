# Ablations

i250 is a narrow extension of i018 with one new degree of freedom
(intra-relation edge confidence). The ablations are designed so that any
positive aggregate result can still be traced back to a specific cause.

## Required ablations

| ID | Switch | What it tests | Interpretation |
|---|---|---|---|
| F1 | i018 (`oriented_tactical_sheaf_laplacian`) | Parent baseline | Reference point. |
| F2 | `model.scramble_relations: true` | Degree-preserving topology scramble (inherited from i018) | Required topology falsifier; drop of `>= 0.02` PR-AUC keeps the topology claim alive. |
| F3 | `model.flat_confidence: true` | Force `alpha_hat = M` | If matched within seed noise of full i250, the confidence head is unnecessary. |
| F4 | `model.normalize_confidence_within_relation: false` | Confidence absorbs relation-level mass | If results improve only here, the head is duplicating the global gate. |
| F5 | Confidence permutation (offline shuffle of `alpha_hat` over active edges within each relation) | Whether **which edge** gets which weight matters | If matched within seed noise of full i250, the learned weights are not edge-specific in a useful way. Requires a small post-hoc eval-time hook; not enabled via config. |
| F6 | Confidence-in-readout-only | Diffuse with raw i018 masks, but feed confidence summaries into the readout | If matched within seed noise of full i250, the sheaf weighting claim is weak. Requires a one-line forward change; not enabled via config. |

F5 and F6 are documented here for honesty and reproducibility, but they
are not exposed as config flags in the current implementation. Adding
them would require a small change to `forward` and a dedicated config; do
that only if F3 already passes (full i250 beats flat).

## Optional feature-level ablations

These remove individual feature families from `RelationEdgeFeatureBuilder`
to identify which tactical cues are load-bearing. They are not exposed as
config flags yet; the channels are listed by index in
`learned_relation_confidence_sheaf.py` so a focused experiment can mask
them in place.

- No piece value: zero out feature channels 0, 1, 2.
- No distance: zero out channel 3.
- No degree: zero out channels 4, 5.
- No king-zone flag: zero out channel 6.
- No pin flag: zero out channel 7.
- No x-ray flag: zero out channel 8.

## Keep / drop rule

Treat i250 as a meaningful improvement over i018 only if both are true:

- mean test PR-AUC across seeds 42, 43, 44 is at least `+0.003` over i018
  at base scale, OR matched-recall (`0.80` or `0.85`) near-puzzle false
  positives are reduced by `>= 1%` absolute without compensating
  regressions on precision or puzzle recall;
- F2 (topology scramble) still drops test PR-AUC by `>= 0.02`.

Drop i250 if any of the following hold:

- F3 (flat confidence) matches full i250 within seed noise;
- F4 (no normalization) matches or beats full i250 -- the head is then
  just duplicating the global gate;
- F2 (topology scramble) drop falls below `0.01` -- the typed topology
  claim has decayed and the family must be re-examined first.

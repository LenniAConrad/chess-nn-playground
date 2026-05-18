# Implementation Notes

- Central code: `src/chess_nn_playground/models/trunk/learned_relation_confidence_sheaf.py`
  (`LearnedRelationConfidenceSheafNet`, `GroupedRelationConfidence`,
  `RelationEdgeFeatureBuilder`,
  `build_learned_relation_confidence_sheaf_from_config`).
- Idea-local wrapper: `ideas/registry/i250_learned_relation_confidence_sheaf/model.py`
  (`build_model_from_config`).
- Registry key: `learned_relation_confidence_sheaf`.
- Parent idea: `i018 oriented_tactical_sheaf_laplacian`.

## What changed vs i018

`LearnedRelationConfidenceSheafNet` subclasses `OrientedTacticalSheafNet`
and adds two new submodules: a deterministic `RelationEdgeFeatureBuilder`
and a learned `GroupedRelationConfidence` head. Only the `forward` method
is overridden, and only to:

1. compute board-only edge features `phi` on the already-active i018 edges;
2. compute normalized edge confidence `alpha_hat` from `phi` and the
   square tokens `h0`;
3. multiply `alpha_hat` into the i018 relation masks before the sheaf
   blocks run;
4. append five confidence-attribution diagnostics to the standard i018
   diagnostic dictionary.

The adapter, incidence builder, encoder, diffusion block, triad pool, and
readout are inherited unchanged. This makes i250 a narrow extension that
can be falsified directly against i018 without changes to the trainer or
benchmark contract.

## Why this is bespoke, not a probe variant

This is a bespoke architecture extension, not a `ResearchPacketProbe`
wrapper. The new edge feature builder and the grouped confidence MLP are
implemented as their own `nn.Module`s in
`learned_relation_confidence_sheaf.py`. The idea-local `model.py` calls
the new `build_learned_relation_confidence_sheaf_from_config` builder,
not `build_research_packet_probe_from_config`.

## Identity at zero-init

The output linear layers of every group MLP and the relation embedding
are zero-initialized. Combined with relation-wise mean normalization,
this makes `alpha_hat = 1` everywhere at init, so the network is exactly
i018 at init except for FP32 reduction noise from the multiply-by-one
into the relation masks. Local CPU check on a 4-sample batch:

- copying i018 weights into the shared parameters (57 tensors) and
  leaving the new confidence head at its zero init gives a max logit
  difference of about `6e-8`;
- with `flat_confidence: true` the difference is `0.0` exactly.

## Module shapes and budget

Base scale (`channels=64`, `hidden_dim=96`, `depth=2`, `stalk_dim=8`):

- i018 parent: ~91k parameters.
- i250 (this idea, base scale): about 98k parameters (measured by counting
  `model.parameters()`). The added head is small: five tiny MLPs sharing
  the same `(feature_dim + 8 + 3 * context_dim) -> 24 -> 1` shape, plus a
  per-relation bias and a 12 x 8 relation embedding.

## Optional knobs

- `confidence_floor` (default `0.05`): keeps active edges from going to
  zero confidence even after the sigmoid saturates.
- `confidence_context_dim` (default `8`): low-rank node context piped into
  the head.
- `confidence_hidden_dim` (default `24`): head width.
- `confidence_group_count` (default `5`): number of semantic confidence
  groups. Must be at least `max(RELATION_GROUPS) + 1`.
- `normalize_confidence_within_relation` (default `true`): apply the
  per-relation mean normalization. Setting this to `false` is the
  "confidence absorbs global gate" ablation.
- `flat_confidence` (default `false`): force `alpha_hat = M`. Used by the
  i018-equivalence audit; faster than the default head because the new
  feature builder is skipped.

## Behaviour with the `scramble_relations` falsifier

`scramble_relations: true` (inherited from i018) is preserved. When
enabled, the i018 relation masks are randomly column-permuted per
`(batch, relation)` before being multiplied by `alpha_hat`. The
confidence head still scores the *real* (unscrambled) edges via its own
feature builder, so the scrambled-mask falsifier remains a clean test of
whether the typed topology matters, even with i250's new stage.

## Numerical guard

The audit should re-run any time the confidence head is edited:

- shared-weights eval-mode `logits` max abs diff under `flat_confidence`:
  must be exactly `0.0` modulo platform FP semantics;
- shared-weights eval-mode `logits` max abs diff at zero-init with the
  default head: should be under `1e-5` on a small batch.

If either guard fails, the change is no longer a strict extension of
i018 and the implementation_kind / status should be re-evaluated.

# Ablations

The specialist exposes one `ablation` enum (also a config field
`model.ablation`). Each setting corresponds to a falsifiable claim from the
research packet.

| ID | Ablation | What changes | What failure would mean |
|---|---|---|---|
| A0 | `trunk_only` | All specialist branches disabled; only the conv trunk pool feeds the final logit. | Establishes the honest parent baseline. If `none` does not beat `trunk_only`, the specialist is not load-bearing. |
| A1 | `copy_baseline_fanout` | Per-type promotion scores replaced by a uniform repeat of the candidate's mean. | If `promotion` / `underpromotion` slices match `none`, the type-conditioned fanout is decorative -- drop the promotion branch. |
| A2 | `uniform_type_attention` | Per-type scores zeroed so type attention is uniform. | If this ties `none`, selective type weighting is decorative -- simplify the branch. |
| A3 | `zero_under_margin` | Non-queen-vs-queen margin zeroed. | If `underpromotion` slice does not worsen, the margin story is cosmetic -- drop the underpromotion branch. |
| A4 | `no_mate_witness` | Six deterministic king-zone witness scalars zeroed. | If `mate_in_1` slice does not worsen, the king-zone witness path is not load-bearing -- drop or replace with the i248 TSDP cache. |
| A5 | `no_joint_branch` | Joint promotion-mate delta and gate zeroed. | If `none` and `no_joint_branch` tie on `mating_special` examples, the joint branch is decorative. |
| A6 | `disable_gate` | Each gate forced to 1 (still respecting structural masks). | If near-puzzle FP rate rises sharply, the gate is the load-bearing rejection control. |
| A7 | `force_zero_gate` | Every gate zeroed. | The wrapper must recover the trunk baseline closely; any drift exposes leakage of branch parameters into `base_logit`. |

Each ablation is a one-flag change. The same `config.yaml` is reused; only
`model.ablation` is modified.

## Loss-side ablations (deferred)

| ID | Ablation | What it tests | Status |
|---|---|---|---|
| L0 | `lambda_gate * sum_k E[gate_k]` only | Whether the gate-sparsity penalty matters beyond BCE. | Deferred -- requires trainer support for the auxiliary term. |
| L1 | `lambda_slice * L_slice` only | Whether slice-restricted ranking lifts the slice PR-AUC numbers. | Deferred -- requires pair-aware trainer batches. |
| L2 | `lambda_near * L_near` only | Whether near-puzzle-FP reweighting alone matches the architecture lift. | Deferred -- requires sampler / loss change. |
| L3 | `uniform_sampler` | Whether the chess-explained slice curriculum is worth the complexity. | Deferred -- requires sampler change. |
| L4 | `lambda_kd * KL_distill` | Whether a teacher-student distillation step closes the slice gap on a smaller student. | Deferred -- requires teacher checkpoint, KD trainer hook. |

These ablations are intentionally listed as deferred. They depend on
trainer extensions that are not bundled with the architecture promotion.
The current matched-recall run uses BCE-with-logits and the shared sampler
so the result is honestly attributable to the architecture rather than to
confounded loss / sampling changes.

## Keep / Drop Rule

Keep i257 only if all are true:

- `none` beats `trunk_only` on slice PR-AUC for at least one of
  `promotion`, `underpromotion`, `mate_in_1` on the validation set;
- at least one of `copy_baseline_fanout`, `zero_under_margin`,
  `no_mate_witness`, `no_joint_branch` loses most of the gain (a
  chess-semantic ablation falsifies the branch it removes);
- overall test PR-AUC remains within ~`0.005` of the matched i193 parent
  baseline (no aggregate regression);
- matched-recall near-puzzle FP rate at recall `0.80` and `0.85` is no
  worse than the parent baseline;
- the bounded-delta identity `|final - base| <= sum_k Delta_k` holds for
  every batch (guarded by the unit test).

Drop i257 if `trunk_only` ties or beats `none`, or if every ablation
matches `none`, or if any future edit breaks the bounded-delta guarantee.
If only one branch passes its falsifier, keep only that branch (the
primitive-stacking strategy explicitly recommends branch-local drop, not
whole-model drop).

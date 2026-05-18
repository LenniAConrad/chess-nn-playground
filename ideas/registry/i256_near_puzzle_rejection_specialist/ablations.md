# Ablations

The specialist exposes one `ablation` enum (also a config field
`model.ablation`). Each setting corresponds to a falsifiable claim from the
research packet.

| ID | Ablation | What changes | What failure would mean |
|---|---|---|---|
| A0 | `trunk_only` | All specialist heads disabled; only the conv trunk pool feeds the raw claim. | Establishes the honest parent baseline. If `none` does not beat `trunk_only`, the specialist is not load-bearing. |
| A1 | `no_forcedness_gap` | `forcedness_gap` reduced to `claim` only. | If this ties `none`, the model is not really about reply-aware forcedness. |
| A2 | `no_reply_envelope` | The reply MLP is zeroed before computing the gap. | If this ties `none`, reply modelling is not load-bearing and `reply_escape_mass` is decorative. |
| A3 | `no_overload_head` | Overload contribution is zeroed in the veto. | If `equal` / `promotion` slices do not worsen, the obligation-budget story is cosmetic. |
| A4 | `no_king_escape_head` | King-escape contribution is zeroed in the veto. | If `mate_in_1` does not worsen, the king-escape veto is not real. |
| A5 | `no_concentration_head` | Concentration contribution is zeroed in the veto. | If `hard` / `very_hard` do not worsen, candidate concentration is not useful. |

Each ablation is a one-flag change. The same `config.yaml` is reused; only
`model.ablation` is modified. The config also exposes the `compile_model`
/ `inference_autocast_dtype` knobs from i249 conceptually — they are not yet
wired and live in `implementation_notes.md` as planned follow-ups.

## Loss-side ablations (deferred)

| ID | Ablation | What it tests | Status |
|---|---|---|---|
| L0 | `L_gap_rank` only on the gap diagnostic | Whether the pairwise margin term on `forcedness_gap` is needed beyond BCE. | Deferred — requires pair-aware trainer batches. |
| L1 | `L_veto` only on high-`raw_claim` near-puzzles | Whether the focused veto term beats plain BCE. | Deferred — requires custom auxiliary-loss hook. |
| L2 | `threshold_0.5_only` (skip validation calibration) | Whether matched-recall gains are mostly calibration or mostly representation. | Deferred — handled by the reporting pipeline rather than the model. |
| L3 | `uniform_sampler` (no slice curriculum) | Whether the chess-explained near-puzzle curriculum is worth the complexity. | Deferred — requires sampler change. |

These ablations are intentionally listed as deferred. They depend on trainer
extensions that are not bundled with the architecture promotion. The current
matched-recall run uses BCE-with-logits and the shared sampler so the result
is honestly attributable to the architecture rather than to confounded loss /
sampling changes.

## Keep / Drop Rule

Keep i256 only if all are true:

- `none` beats `trunk_only` on matched-recall near-puzzle FP rate at recall
  `0.80` *and* `0.85` on the validation set;
- at least one of `no_reply_envelope`, `no_overload_head`,
  `no_king_escape_head`, `no_concentration_head` loses most of the gain
  (a chess-semantic ablation falsifies the head it removes);
- overall test PR-AUC remains within ~`0.003` of the matched i193 parent
  baseline (no aggregate regression);
- `final_logit <= raw_claim_logit` holds for every batch (rejection identity
  guarded by the unit test).

Drop i256 if `trunk_only` ties or beats `none`, or if every ablation matches
`none`, or if any future edit breaks the `softplus`-only-subtract guarantee.
The only allowed reason for the head to lift `raw_claim` is removing it
entirely.

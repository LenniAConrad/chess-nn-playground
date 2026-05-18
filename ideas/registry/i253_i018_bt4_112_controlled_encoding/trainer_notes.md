# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). The default config is
the matched-control row of the research markdown's 36-run base matrix:
`simple18 padded to 112 raw channels`, exact relations only.

Differences vs `ideas/registry/i018_oriented_tactical_sheaf_laplacian/config.yaml`:

- `model.name = i018_bt4_112_controlled_encoding` (i253 builder).
- `model.input_channels` is `18` for `simple_18` and `112` for
  `lc0_bt4_112`; the raw branch always sees 112 channels internally.
- `model.relation_mode` (`exact` | `confidence` | `hybrid`) chooses
  whether to add the learned relation-confidence head and bounded
  augmentation.
- `model.relation_hidden = 16`, `model.relation_rank = 8`,
  `model.augmentation_lambda = 0.25` match the research markdown.
- Falsifier knobs `model.scramble_exact_relations` and
  `model.augmentation_only` are exposed at config level so each cell of
  the matrix is a one-line edit.
- Training header is stricter than the i018 default:
  `epochs = 30`, `min_epochs = 15`, `min_active_epochs = 15`,
  `batch_size = 192`, `early_stopping_patience = 8`, `monitor =
  pr_auc`. This matches the research-quality regime
  (`reliability_tier: paper_grade`) the markdown requests.

If you change a trunk hyperparameter (`channels`, `hidden_dim`, `depth`,
`stalk_dim`, `dropout`, `use_batchnorm`), change it on i018 too and
re-run both - the encoding comparison is only honest when both nets
share trunk geometry.

## Loss

`bce_with_logits` on the puzzle logit. No i253-specific auxiliary loss.
The i018 mechanism-energy diagnostics are emitted unchanged. Two new
diagnostics appear in non-exact modes (`controlled_confidence_mean`,
`controlled_augmentation_mean`).

## Decision rule

Following the research markdown:

- BT4 earns a real encoding win if, against the matched simple18 row of
  the same `relation_mode`, mean PR-AUC improves by at least about
  `+0.003` or near-puzzle false positives at matched recall drop by at
  least `1%`.
- Confidence and hybrid must not regress on hard, equal, endgame,
  mate-in-1, promotion, or underpromotion slices.
- Relation scramble must keep i018's load-bearing-geometry result:
  scrambled runs should drop PR-AUC by at least `0.02` versus the
  matched intact row.
- Augmentation-only hybrid should collapse versus intact hybrid; if it
  does not, the augmentation head is overpowered and the run should be
  re-tuned (smaller `augmentation_lambda`) before promotion.

## Cost expectation

- Trunk shape, depth, and parameter budget are within ~9% of i018 base.
- Confidence head adds about 5k parameters; hybrid head about 8k.
- Relation augmentation templates are static (no extra runtime cost
  beyond a per-batch broadcast multiplication).
- Wall-clock should be comparable to i018; the augmentation templates
  do not interact with the per-relation Python loop in
  `SheafDiffusionBlock`.

## Benchmark plan

Following the research markdown's 36-run base package:

| Tranche | Cells | Seeds | Runs |
|---|---|---:|---:|
| Primary comparison | `exact`, `confidence`, `hybrid` x `simple18`, `lc0_bt4_112` | 3 | 18 |
| Exact-support falsifier | `exact_scramble` x both encodings | 3 | 6 |
| Hybrid falsifier | `hybrid_scramble_exact_support` x both encodings | 3 | 6 |
| Augmentation-only falsifier | `hybrid_augmentation_only` x both encodings | 3 | 6 |
| **Total** |  |  | **36** |

Use seeds `42, 43, 44` per cell. If wall-clock is a bottleneck, the
research markdown explicitly allows running the entire study on the i249
execution path (`oriented_tactical_sheaf_fast`) instead of the vanilla
i018-style diffusion, but only if every cell uses it. Mixing i018 and
i249 execution paths within the same comparison reintroduces an
avoidable confound; do not do it.

## Reports

Standard idea report (see `report_template.md`). The slice analysis is
inherited from i018's reporting contract and must include
`crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`,
`crtk_tactic_motifs`, and `crtk_tag_families`. The new columns are the
encoding axis (`simple_18` vs `lc0_bt4_112`), the relation_mode axis
(`exact` / `confidence` / `hybrid`), and the falsifier columns
(`scramble`, `augmentation_only`).

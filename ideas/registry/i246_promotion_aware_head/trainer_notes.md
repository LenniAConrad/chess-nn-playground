# Trainer Notes — Promotion-Aware Head (i246)

- The idea-local `train.py` calls `idea_train_cli(__file__)`. No custom
  trainer code is required; the shared puzzle_binary trainer consumes
  `output["logits"]` for the loss and the rest of the dict for diagnostics.
- The trainer guard requires `idea.yaml.implementation_status` to be
  `implemented` (or `tested`), `implementation_kind: bespoke_model`,
  `device: nvidia`, and a registered `model.name`. The config in this
  folder satisfies all four.
- Batch size is set to 128 (vs i193's 256) because the counterfactual
  fanout multiplies the effective per-sample trunk work by up to `K * 4`.
  With `K = max_promotion_pawns = 4`, the worst-case trunk batch shape
  reaches `(B*K*4, 18, 8, 8) = (2048, 18, 8, 8)`. This stays comfortably
  inside a 12 GB GPU at fp16. Drop batch size to 96 if OOM occurs on
  smaller cards.
- Mixed precision is on (`mixed_precision: true`, `matmul_precision:
  high`, `allow_tf32: true`). All operations in the head (LayerNorm,
  Linear, GELU, softmax, einsum) are autocast-safe.
- Reliability tier is `paper_grade`: epochs >= 20, patience >= 5,
  min_epochs >= 10. The shared trainer enforces these.
- `class_weighting: balanced` matches the i193 baseline and TDCD primitive.
- `lr_scheduler: reduce_on_plateau` matches the rest of the primitive
  batch.

## Falsifier protocol

The two head ablations that gate the keep / drop decision are
`copy_baseline_fanout` (the spec's "A1" matched ablation) and
`uniform_attention`. Both can be triggered by setting `model.ablation`
in the config or via CLI override:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/train_model.py \
    --config ideas/registry/i246_promotion_aware_head/config.yaml \
    --override model.ablation=copy_baseline_fanout \
    --override run.name=i246_pfct_copy_baseline_ablation
```

Both ablations leave the trunk weights and gating intact, so the slice
comparison is matched on everything except the substitution / attention
content under test.

## Slice reporting

The promotion / underpromotion slices are tagged in CRTK as
`crtk_tactic_motifs ∈ {promotion, underpromotion}`. The shared slice
report consumes those tags directly; nothing primitive-specific has to
be wired into the reporter.

Additionally the head exports per-sample diagnostics
(`promotion_pawn_count`, `promotion_has_pawn`, `promotion_dominant_type`,
`promotion_attention_entropy`, `promotion_fanout_dispersion`) into the
`predictions_<split>.parquet`. These can be sliced on after the run to
verify that:

- The gate stays near zero on non-promotion positions
  (`mean(primitive_gate | promotion_has_pawn == 0)` should be 0).
- The attention concentrates (`promotion_attention_entropy < log(4)`) on
  positions where one promotion type is clearly best.
- The dominant type distribution matches chess expectations: ~98% Q on
  random near-promotion positions, with non-Q dominance on tagged
  underpromotion / knight-fork puzzles.

# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). The config is paper-grade,
CUDA-required, and matched to the canonical tagged split so the comparison is
honest against the i193 fast conv student parent:

- same train / val / test split (`crtk_sample_3class_unique_crtk_tags`)
- same encoding (`simple_18`)
- same `epochs`, `batch_size`, `class_weighting`, `loss`, early-stopping
  policy, and `lr_scheduler` as i193 and i248
- seeds 42 / 43 / 44 for the reliable protocol

## Loss

`bce_with_logits` on the puzzle logit `final = raw_claim - softplus(veto)`.

The research packet recommends an extended loss

```text
L = L_main + lambda_gap * L_gap_rank + lambda_veto * L_veto
```

with a pairwise margin term on `forcedness_gap` between matched puzzles and
near-puzzles, and a focused BCE veto term on high-`raw_claim` near-puzzles.
Those require a trainer extension (pair-aware batches and a custom auxiliary
loss hook) that is intentionally not bundled with this architecture
promotion. The model already exports `raw_claim_logit`, `reply_veto_logit`,
`max_forcedness_gap`, and per-candidate diagnostics, so plugging the
extension in later is purely additive on the trainer side.

## Sampling

The default config uses the shared sampler — *no* chess-explained near-puzzle
curriculum. Switching to the packet's slice-weighted curriculum is also a
trainer-side change. Until that lands, validation matched-recall remains the
operating-point metric.

## Cost expectation

The default config (channels=32, depth=2, hidden_dim=64, per_square_hidden=48)
is intentionally lighter than the i248 trunk so the first deployment target
(C1 student-full) stays within the research packet's expected
~10–20% GPU / ~15–30% CPU latency envelope versus the i193 parent. If the
matched-recall reliability run shows the specialist is genuinely helpful, the
P3 i018-parent variant can be added by swapping the conv encoder for an i018
trunk without touching the heads.

## Reports

Standard idea report (see `report_template.md`). The slice analysis must
include `crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`,
`crtk_tactic_motifs`, and `crtk_tag_families`. The packet specifies that the
near-puzzle FP rate at `puzzle_recall in {0.80, 0.85}` is the primary
scoreboard, with PR-AUC remaining secondary. Validation-only threshold
selection is therefore mandatory.

The model emits one logit, so the existing artifact pipeline (calibration,
confusion-matrix, slice reports) works without changes. The per-sample
specialist diagnostics (`raw_claim_logit`, `reply_veto_logit`,
`max_forcedness_gap`, `defender_overload`, `king_escape_pressure`,
`concentration_score` via `effective_candidate_count`) all land in the
prediction parquets so the post-hoc reports can attribute slice wins to the
veto path.

## Smoke / CI

CPU smoke is sufficient for compile / registry / forward checks (no GPU is
required at scaffold time). Local `tests/test_idea_i256_near_puzzle_rejection_specialist.py`
covers builder registration, forward shape, ablation safety, gradient flow,
and the rejection identity `final_logit <= raw_claim_logit`.

The reliability run still requires `device: nvidia` per the global idea-config
contract; the guarded trainer will refuse to fall back to CPU silently.

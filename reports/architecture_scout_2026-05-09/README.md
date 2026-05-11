# Architecture Scout — 2026-05-09

A one-pass cheap filter over every bespoke idea architecture. **Not a final
leaderboard.** One seed, one scale, short training budget, fail-fast on OOM.
Used to eliminate weak ideas and pick candidates for full 3-seed promotion.

## Scout settings

| Setting | Value |
|---|---|
| Configs | All 234 bespoke idea configs (no benchmarks) |
| Seed | 42 only |
| Architecture scale | base only |
| Mode | `puzzle_binary` (per each idea's YAML) |
| Checkpoint monitor | **`pr_auc`** (explicit, injected by runner) |
| Epochs (max) | 12 |
| Min epochs | 6 |
| Early-stop patience | 3 |
| Jobs | 1 |
| GPU | 0 |
| Per-task timeout | 60 minutes |
| CPU fallback | **DISABLED** — `CHESS_NN_DISABLE_CPU_FALLBACK=1` |
| Continue on error | yes (runner default) |

## tmux

- **Session name:** `architecture-scout`
- **Attach:** `tmux attach -t architecture-scout`
- **Detach (inside tmux):** Ctrl+B, then D
- **Tee log:** `reports/architecture_scout_2026-05-09/tmux.log`

## Output dirs

- Run artifacts: `results/architecture_scout_2026-05-09/`
- State + logs + reports: `reports/architecture_scout_2026-05-09/`
- Per-task subprocess logs: `reports/architecture_scout_2026-05-09/logs/`
- Generated configs: `reports/architecture_scout_2026-05-09/generated_configs/`
- Resumable state: `reports/architecture_scout_2026-05-09/state.json`
- Event log (JSONL): `reports/architecture_scout_2026-05-09/events.jsonl`

## Exact launch command

```bash
PYTHONDONTWRITEBYTECODE=1 \
CHESS_NN_DISABLE_CPU_FALLBACK=1 \
.venv/bin/python scripts/run_paper_ready_all.py \
  --no-benchmarks \
  --results-dir   results/architecture_scout_2026-05-09 \
  --report-dir    reports/architecture_scout_2026-05-09 \
  --state-path    reports/architecture_scout_2026-05-09/state.json \
  --logs-dir      reports/architecture_scout_2026-05-09/logs \
  --generated-config-dir reports/architecture_scout_2026-05-09/generated_configs \
  --event-log     reports/architecture_scout_2026-05-09/events.jsonl \
  --timeline      reports/architecture_scout_2026-05-09/timeline.md \
  --seeds 42 --scale-variants base:1 \
  --epochs 12 --min-epochs 6 --patience 3 \
  --shorten-training --monitor pr_auc \
  --jobs 1 --gpu-ids 0 --timeout-minutes 60
```

## How to resume after interruption

The runner is fully resumable. State lives in `state.json`; completed tasks are
skipped on the next launch. To resume after crash / power loss / Ctrl+C:

```bash
tmux attach -t architecture-scout 2>/dev/null || \
  tmux new-session -d -s architecture-scout -c "$(pwd)"

# inside the session, paste the same launch command above
```

If the tmux session is gone entirely, just create a new one with the same name
and re-run the same command. Already-completed tasks (those that produced a
`metrics_final.json`) will be skipped automatically.

## Compute budget estimate

234 tasks × ~5 min average per run on this 8 GiB GPU (very rough — bespoke
models vary). Expected wall time: **~15-25 hours**, plus some failed-OOM tasks
that should fail fast (<2 min each since `CPU fallback is disabled`).

## What gets produced

Per-task artifacts in `results/architecture_scout_2026-05-09/<task_id>/`:

- `metrics_final.json` (the canonical "task complete" file; includes
  `test_pr_auc`)
- `predictions_val.parquet` (used downstream for matched-recall and slice
  analysis)
- `checkpoint_best.pt` (PR-AUC-best checkpoint)
- `metrics_history.json`, `metrics_train.csv`, `metrics_val.csv`
- `run_metadata.json` (records `monitor: pr_auc` and `reliability_tier: scout`)

## After the scout finishes

Generate from the artifacts (no new training):

1. **Overall one-seed leaderboard** — by val PR AUC and test PR AUC
2. **League leaderboards** — split by input encoding (`lc0_bt4_112`, `simple_18`)
3. **Matched-recall near-puzzle FP** — at recall 0.80 and 0.85
   (`scripts/analyze_matched_recall_fp.py --results-root results/architecture_scout_2026-05-09 ...`)
4. **Per-class / per-slice leaderboard** —
   (`scripts/analyze_per_class_benchmark.py --results-root results/architecture_scout_2026-05-09 --min-seeds 1 ...`)
   especially `equal` eval bucket, `hard` / `very_hard`, `promotion`,
   `underpromotion`, and `skewer / overload / mate_in_1` motifs.
5. **Failure report** — query `events.jsonl` for OOM, timeout, NaN, missing
   artifacts. Categorize by failure mode.

## Promotion rule (scout → 3-seed runs)

Only promote to full 3-seed × scale_xl runs models that satisfy any of:

- Top 10 by overall test PR AUC
- Top 5 by matched-recall near-puzzle FP
- Top 5 by promotion/underpromotion operating-point performance
- Any `simple_18` model within ~0.005 PR AUC of the LC0 baseline (`bench_lc0_bt4_classifier`)
- Any model with a clear niche win on a hard slice (`equal` eval, `very_hard`,
  promotion, underpromotion)

## Important interpretation

This scout is a **map, not proof**. Single seed at base scale is noisy — within
a ±0.005 PR AUC band, scout ranks should not be trusted. It is good enough to
eliminate clearly-weak ideas and select promotion candidates.

The scout uses each idea's YAML defaults except for the four overrides above
(epochs, min_epochs, patience, monitor) — so per-idea hyperparameters
(learning rate, weight decay, batch size, model-specific hyperparams) remain
the author's intended values.

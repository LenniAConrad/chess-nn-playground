# Paper-Ready Training Status

- Total tasks: `1`
- Completed tasks: `1`
- Remaining tasks: `0`
- ETA: `0s`
- Average observed task time: `8m31s`
- ETA basis: `1` observed task(s), `0` remaining task(s), `1` job(s)
- Dry run: `False`
- Results directory: `results/architecture_scout_2026-05-11_i242`
- Report directory: `reports/architecture_scout_2026-05-11_i242`
- Resume state: `reports/architecture_scout_2026-05-11_i242/state.json`

## Defaults

- Seeds: `42`
- Architecture scales: `base:1`
- Batch-size caps: `base:256,scale_up:192,scale_xl:128`
- Epoch budget: `12`
- Minimum epochs: `6`
- Early-stopping patience: `3`

## Counts

| Status | Count |
|---|---:|
| `completed` | 1 |

| Kind | Count |
|---|---:|
| `idea` | 1 |

| Architecture Scale | Count |
|---|---:|
| `base` | 1 |

## Open First

- Plan: `reports/architecture_scout_2026-05-11_i242/plan.md`
- Status: `reports/architecture_scout_2026-05-11_i242/status.md`
- State JSON: `reports/architecture_scout_2026-05-11_i242/state.json`
- Event log JSONL: `reports/architecture_scout_2026-05-11_i242/events.jsonl`
- Timeline: `reports/architecture_scout_2026-05-11_i242/timeline.md`
- Logs: `reports/architecture_scout_2026-05-11_i242/logs`
- Generated configs: `reports/architecture_scout_2026-05-11_i242/generated_configs`
- Leaderboard: `results/architecture_scout_2026-05-11_i242/leaderboard.md`
- Seed summary: `results/architecture_scout_2026-05-11_i242/leaderboard_seed_summary.md`
- Training dashboard: `reports/architecture_scout_2026-05-11_i242/training/training_dashboard.md`
- Training dashboard HTML: `reports/architecture_scout_2026-05-11_i242/training/training_dashboard.html`
- Paper PDF report: `reports/architecture_scout_2026-05-11_i242/paper_report.pdf`

## Analysis Jobs

| Job | Return Code | Log |
|---|---:|---|
| `build_paper_report` | `0` | `reports/architecture_scout_2026-05-11_i242/logs/analysis_build_paper_report.log` |
| `compare_results` | `0` | `reports/architecture_scout_2026-05-11_i242/logs/analysis_compare_results.log` |
| `plot_training_results` | `0` | `reports/architecture_scout_2026-05-11_i242/logs/analysis_plot_training_results.log` |

## Speed Snapshot

| Task | Scale | Params | Train Samples/s | Val Samples/s | Total Seconds |
|---|---|---:|---:|---:|---:|
| `idea_i242_chess_decomposed_attention_seed42` | `base` | 270791 | 6798.3 | 10436.8 | 355.6 |

## Next Tasks

| Task | Kind | Scale | Seed | Status | Source | Run Dir |
|---|---|---|---:|---|---|---|
| none |  |  |  |  |  |  |

## Resume Command

Rerun the same command after an interruption. Completed tasks stay completed, and unfinished tasks use the same fixed run directories.

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/run_paper_ready_all.py ideas/i242_chess_decomposed_attention/config.yaml --results-dir results/architecture_scout_2026-05-11_i242 --report-dir reports/architecture_scout_2026-05-11_i242 --state-path reports/architecture_scout_2026-05-11_i242/state.json --logs-dir reports/architecture_scout_2026-05-11_i242/logs --generated-config-dir reports/architecture_scout_2026-05-11_i242/generated_configs --seeds 42 --scale-variants base:1 --batch-size-caps base:256,scale_up:192,scale_xl:128 --epochs 12 --min-epochs 6 --patience 3 --jobs 1 --gpu-ids 0 --timeout-minutes 30.0 --no-benchmarks --no-ideas
```

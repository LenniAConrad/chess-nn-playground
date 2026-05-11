# Paper-Ready Training Status

- Total tasks: `1`
- Completed tasks: `1`
- Remaining tasks: `0`
- ETA: `0s`
- Average observed task time: `8m10s`
- ETA basis: `1` observed task(s), `0` remaining task(s), `1` job(s)
- Dry run: `False`
- Results directory: `results/architecture_scout_2026-05-11_i242_ablation`
- Report directory: `reports/architecture_scout_2026-05-11_i242_ablation`
- Resume state: `reports/architecture_scout_2026-05-11_i242_ablation/state.json`

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
| `benchmark` | 1 |

| Architecture Scale | Count |
|---|---:|
| `base` | 1 |

## Open First

- Plan: `reports/architecture_scout_2026-05-11_i242_ablation/plan.md`
- Status: `reports/architecture_scout_2026-05-11_i242_ablation/status.md`
- State JSON: `reports/architecture_scout_2026-05-11_i242_ablation/state.json`
- Event log JSONL: `reports/architecture_scout_2026-05-11_i242_ablation/events.jsonl`
- Timeline: `reports/architecture_scout_2026-05-11_i242_ablation/timeline.md`
- Logs: `reports/architecture_scout_2026-05-11_i242_ablation/logs`
- Generated configs: `reports/architecture_scout_2026-05-11_i242_ablation/generated_configs`
- Leaderboard: `results/architecture_scout_2026-05-11_i242_ablation/leaderboard.md`
- Seed summary: `results/architecture_scout_2026-05-11_i242_ablation/leaderboard_seed_summary.md`
- Training dashboard: `reports/architecture_scout_2026-05-11_i242_ablation/training/training_dashboard.md`
- Training dashboard HTML: `reports/architecture_scout_2026-05-11_i242_ablation/training/training_dashboard.html`
- Paper PDF report: `reports/architecture_scout_2026-05-11_i242_ablation/paper_report.pdf`

## Analysis Jobs

| Job | Return Code | Log |
|---|---:|---|
| `build_paper_report` | `0` | `reports/architecture_scout_2026-05-11_i242_ablation/logs/analysis_build_paper_report.log` |
| `compare_results` | `0` | `reports/architecture_scout_2026-05-11_i242_ablation/logs/analysis_compare_results.log` |
| `plot_training_results` | `0` | `reports/architecture_scout_2026-05-11_i242_ablation/logs/analysis_plot_training_results.log` |

## Speed Snapshot

| Task | Scale | Params | Train Samples/s | Val Samples/s | Total Seconds |
|---|---|---:|---:|---:|---:|
| `benchmark_i242_ablation_noglobal_seed42` | `base` | 203847 | 7911.6 | 10551.4 | 318.7 |

## Next Tasks

| Task | Kind | Scale | Seed | Status | Source | Run Dir |
|---|---|---|---:|---|---|---|
| none |  |  |  |  |  |  |

## Resume Command

Rerun the same command after an interruption. Completed tasks stay completed, and unfinished tasks use the same fixed run directories.

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/run_paper_ready_all.py /tmp/i242_ablation_noglobal.yaml --results-dir results/architecture_scout_2026-05-11_i242_ablation --report-dir reports/architecture_scout_2026-05-11_i242_ablation --state-path reports/architecture_scout_2026-05-11_i242_ablation/state.json --logs-dir reports/architecture_scout_2026-05-11_i242_ablation/logs --generated-config-dir reports/architecture_scout_2026-05-11_i242_ablation/generated_configs --seeds 42 --scale-variants base:1 --batch-size-caps base:256,scale_up:192,scale_xl:128 --epochs 12 --min-epochs 6 --patience 3 --jobs 1 --gpu-ids 0 --timeout-minutes 30.0 --no-benchmarks --no-ideas
```

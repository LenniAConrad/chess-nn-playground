# Paper-Ready Training Status

- Total tasks: `1`
- Completed tasks: `1`
- Remaining tasks: `0`
- ETA: `0s`
- Average observed task time: `7m39s`
- ETA basis: `1` observed task(s), `0` remaining task(s), `1` job(s)
- Dry run: `False`
- Results directory: `results/architecture_scout_2026-05-11_i242_A3_no_exchange`
- Report directory: `reports/architecture_scout_2026-05-11_i242_A3_no_exchange`
- Resume state: `reports/architecture_scout_2026-05-11_i242_A3_no_exchange/state.json`

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

- Plan: `reports/architecture_scout_2026-05-11_i242_A3_no_exchange/plan.md`
- Status: `reports/architecture_scout_2026-05-11_i242_A3_no_exchange/status.md`
- State JSON: `reports/architecture_scout_2026-05-11_i242_A3_no_exchange/state.json`
- Event log JSONL: `reports/architecture_scout_2026-05-11_i242_A3_no_exchange/events.jsonl`
- Timeline: `reports/architecture_scout_2026-05-11_i242_A3_no_exchange/timeline.md`
- Logs: `reports/architecture_scout_2026-05-11_i242_A3_no_exchange/logs`
- Generated configs: `reports/architecture_scout_2026-05-11_i242_A3_no_exchange/generated_configs`
- Leaderboard: `results/architecture_scout_2026-05-11_i242_A3_no_exchange/leaderboard.md`
- Seed summary: `results/architecture_scout_2026-05-11_i242_A3_no_exchange/leaderboard_seed_summary.md`
- Training dashboard: `reports/architecture_scout_2026-05-11_i242_A3_no_exchange/training/training_dashboard.md`
- Training dashboard HTML: `reports/architecture_scout_2026-05-11_i242_A3_no_exchange/training/training_dashboard.html`
- Paper PDF report: `reports/architecture_scout_2026-05-11_i242_A3_no_exchange/paper_report.pdf`

## Analysis Jobs

| Job | Return Code | Log |
|---|---:|---|
| `build_paper_report` | `0` | `reports/architecture_scout_2026-05-11_i242_A3_no_exchange/logs/analysis_build_paper_report.log` |
| `compare_results` | `0` | `reports/architecture_scout_2026-05-11_i242_A3_no_exchange/logs/analysis_compare_results.log` |
| `plot_training_results` | `0` | `reports/architecture_scout_2026-05-11_i242_A3_no_exchange/logs/analysis_plot_training_results.log` |

## Speed Snapshot

| Task | Scale | Params | Train Samples/s | Val Samples/s | Total Seconds |
|---|---|---:|---:|---:|---:|
| `benchmark_A3_no_exchange_seed42` | `base` | 203847 | 8102.5 | 11135.1 | 303.8 |

## Next Tasks

| Task | Kind | Scale | Seed | Status | Source | Run Dir |
|---|---|---|---:|---|---|---|
| none |  |  |  |  |  |  |

## Resume Command

Rerun the same command after an interruption. Completed tasks stay completed, and unfinished tasks use the same fixed run directories.

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/run_paper_ready_all.py /tmp/i242_ablations/A3_no_exchange.yaml --results-dir results/architecture_scout_2026-05-11_i242_A3_no_exchange --report-dir reports/architecture_scout_2026-05-11_i242_A3_no_exchange --state-path reports/architecture_scout_2026-05-11_i242_A3_no_exchange/state.json --logs-dir reports/architecture_scout_2026-05-11_i242_A3_no_exchange/logs --generated-config-dir reports/architecture_scout_2026-05-11_i242_A3_no_exchange/generated_configs --seeds 42 --scale-variants base:1 --batch-size-caps base:256,scale_up:192,scale_xl:128 --epochs 12 --min-epochs 6 --patience 3 --jobs 1 --gpu-ids 0 --timeout-minutes 30.0 --no-benchmarks --no-ideas
```

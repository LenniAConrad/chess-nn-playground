# Paper-Ready Training Status

- Total tasks: `1`
- Completed tasks: `1`
- Remaining tasks: `0`
- ETA: `0s`
- Average observed task time: `6m59s`
- ETA basis: `1` observed task(s), `0` remaining task(s), `1` job(s)
- Dry run: `False`
- Results directory: `results/architecture_scout_2026-05-11_i242_A4_i193_hp`
- Report directory: `reports/architecture_scout_2026-05-11_i242_A4_i193_hp`
- Resume state: `reports/architecture_scout_2026-05-11_i242_A4_i193_hp/state.json`

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

- Plan: `reports/architecture_scout_2026-05-11_i242_A4_i193_hp/plan.md`
- Status: `reports/architecture_scout_2026-05-11_i242_A4_i193_hp/status.md`
- State JSON: `reports/architecture_scout_2026-05-11_i242_A4_i193_hp/state.json`
- Event log JSONL: `reports/architecture_scout_2026-05-11_i242_A4_i193_hp/events.jsonl`
- Timeline: `reports/architecture_scout_2026-05-11_i242_A4_i193_hp/timeline.md`
- Logs: `reports/architecture_scout_2026-05-11_i242_A4_i193_hp/logs`
- Generated configs: `reports/architecture_scout_2026-05-11_i242_A4_i193_hp/generated_configs`
- Leaderboard: `results/architecture_scout_2026-05-11_i242_A4_i193_hp/leaderboard.md`
- Seed summary: `results/architecture_scout_2026-05-11_i242_A4_i193_hp/leaderboard_seed_summary.md`
- Training dashboard: `reports/architecture_scout_2026-05-11_i242_A4_i193_hp/training/training_dashboard.md`
- Training dashboard HTML: `reports/architecture_scout_2026-05-11_i242_A4_i193_hp/training/training_dashboard.html`
- Paper PDF report: `reports/architecture_scout_2026-05-11_i242_A4_i193_hp/paper_report.pdf`

## Analysis Jobs

| Job | Return Code | Log |
|---|---:|---|
| `build_paper_report` | `0` | `reports/architecture_scout_2026-05-11_i242_A4_i193_hp/logs/analysis_build_paper_report.log` |
| `compare_results` | `0` | `reports/architecture_scout_2026-05-11_i242_A4_i193_hp/logs/analysis_compare_results.log` |
| `plot_training_results` | `0` | `reports/architecture_scout_2026-05-11_i242_A4_i193_hp/logs/analysis_plot_training_results.log` |

## Speed Snapshot

| Task | Scale | Params | Train Samples/s | Val Samples/s | Total Seconds |
|---|---|---:|---:|---:|---:|
| `benchmark_A4_i193_hp_seed42` | `base` | 270791 | 9621.1 | 11074.9 | 262.5 |

## Next Tasks

| Task | Kind | Scale | Seed | Status | Source | Run Dir |
|---|---|---|---:|---|---|---|
| none |  |  |  |  |  |  |

## Resume Command

Rerun the same command after an interruption. Completed tasks stay completed, and unfinished tasks use the same fixed run directories.

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/run_paper_ready_all.py /tmp/i242_ablations/A4_i193_hp.yaml --results-dir results/architecture_scout_2026-05-11_i242_A4_i193_hp --report-dir reports/architecture_scout_2026-05-11_i242_A4_i193_hp --state-path reports/architecture_scout_2026-05-11_i242_A4_i193_hp/state.json --logs-dir reports/architecture_scout_2026-05-11_i242_A4_i193_hp/logs --generated-config-dir reports/architecture_scout_2026-05-11_i242_A4_i193_hp/generated_configs --seeds 42 --scale-variants base:1 --batch-size-caps base:256,scale_up:192,scale_xl:128 --epochs 12 --min-epochs 6 --patience 3 --jobs 1 --gpu-ids 0 --timeout-minutes 30.0 --no-benchmarks --no-ideas
```

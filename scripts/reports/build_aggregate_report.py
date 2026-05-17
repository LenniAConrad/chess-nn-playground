"""Build one auto-updating aggregate report across every experiment pipeline.

Scans the results directory of each pipeline -- primitive scout, paper-grade
top-3 trunks, i018 hybrids, i018 falsifier, i249 fast variant, BT4 primitive
mixers, and the LC0 BT4 transformer benchmark -- and writes a single
`reports/aggregate_report.md` with per-experiment tables plus a global
leaderboard.

Designed to be run repeatedly while experiments are in flight: missing or
partial result directories are handled gracefully. It is invoked both by the
self-scheduling monitor (every tick) and by each pipeline launcher on
completion, so the report stays fresh.

Run:
    PYTHONDONTWRITEBYTECODE=1 python -m scripts.reports.build_aggregate_report
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "reports" / "aggregate_report.md"

# (display name, results dir, run-dir glob)
EXPERIMENTS = [
    ("Primitive scout (35 primitives)", "results/primitive_pipeline", "idea_*_seed*"),
    ("Paper-grade: top-3 trunks + BT4-conv", "results/paper_grade_top3", "*_seed*"),
    ("i018 hybrids (sheaf + primitive)", "results/hybrid_i018", "*_seed*"),
    ("i018 falsifier (scrambled relations)", "results/falsifier_i018", "*_seed*"),
    ("i249 fast (i018 speed variant)", "results/i249_fast", "*_seed*"),
    ("BT4 primitive-mixer scout (40 mixers)", "results/bt4_primitive_mixers", "*_seed*"),
    ("LC0 BT4 transformer benchmark", "results/lc0_bt4_transformer", "*_seed*"),
]

SCALE_ORDER = {"base": 0, "scale_up": 1, "scale_xl": 2}


def _scale_of(name: str) -> str:
    if "scale_xl" in name:
        return "scale_xl"
    if "scale_up" in name:
        return "scale_up"
    return "base"


def _variant_key(name: str) -> str:
    """Run-dir name minus the trailing _seedNN, so seeds of one config group."""
    idx = name.rfind("_seed")
    return name[:idx] if idx >= 0 else name


def _load_run(run_dir: Path) -> dict | None:
    metrics = run_dir / "metrics_final.json"
    if not metrics.exists():
        return None
    try:
        m = json.loads(metrics.read_text())
    except Exception:
        return None
    params = None
    meta = run_dir / "run_metadata.json"
    if meta.exists():
        try:
            params = json.loads(meta.read_text()).get("num_params")
        except Exception:
            params = None
    return {
        "name": run_dir.name,
        "test_pr_auc": m.get("test_pr_auc"),
        "test_accuracy": m.get("test_accuracy"),
        "params": params,
        "scale": _scale_of(run_dir.name),
    }


def _collect(results_dir: Path, glob: str) -> list[dict]:
    if not results_dir.exists():
        return []
    runs = []
    for run_dir in sorted(results_dir.glob(glob)):
        if not run_dir.is_dir():
            continue
        info = _load_run(run_dir)
        if info is not None:
            runs.append(info)
    return runs


def _agg(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return float("nan"), float("nan")
    return statistics.fmean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)


def _fmt(x: float | None, p: int = 4) -> str:
    if x is None or (isinstance(x, float) and x != x):  # None or NaN
        return "-"
    return f"{x:.{p}f}"


def main() -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines: list[str] = []
    lines.append("# Aggregate Experiment Report")
    lines.append("")
    lines.append(f"Auto-generated: {now}")
    lines.append("")
    lines.append(
        "Single rolling view across every experiment pipeline. Rebuilt by the "
        "self-scheduling monitor each tick and by each pipeline launcher on "
        "completion. Partial/missing results mean that pipeline is queued or "
        "still running."
    )
    lines.append("")
    lines.append("Reference bars (paper-grade, 3-seed means): "
                 "lc0_bt4_classifier conv tower ~0.859 test PR-AUC; "
                 "i018 oriented_tactical_sheaf base ~0.875, scale_xl ~0.890.")
    lines.append("")

    global_best: list[tuple[float, str, str]] = []  # (pr_auc, experiment, variant)

    for display, rel_dir, glob in EXPERIMENTS:
        results_dir = REPO_ROOT / rel_dir
        runs = _collect(results_dir, glob)
        lines.append(f"## {display}")
        lines.append("")
        lines.append(f"`{rel_dir}` - {len(runs)} completed run(s).")
        lines.append("")
        if not runs:
            lines.append("_No completed runs yet (queued or running)._")
            lines.append("")
            continue

        # group by variant (config minus seed)
        by_variant: dict[str, list[dict]] = defaultdict(list)
        for r in runs:
            by_variant[_variant_key(r["name"])].append(r)

        rows = []
        for variant, vruns in by_variant.items():
            pr_mean, pr_std = _agg([r["test_pr_auc"] for r in vruns])
            acc_mean, _ = _agg([r["test_accuracy"] for r in vruns])
            params = next((r["params"] for r in vruns if r["params"]), None)
            scale = vruns[0]["scale"]
            rows.append((pr_mean, pr_std, acc_mean, params, scale, variant, len(vruns)))
            if pr_mean == pr_mean:  # not NaN
                global_best.append((pr_mean, display, variant))

        rows.sort(key=lambda r: (-(r[0] if r[0] == r[0] else -1)))
        lines.append("| test PR-AUC (mean) | std | test acc | params | scale | seeds | variant |")
        lines.append("|---:|---:|---:|---:|---|---:|---|")
        for pr_mean, pr_std, acc_mean, params, scale, variant, n in rows:
            params_s = f"{params:,}" if params else "-"
            short = variant.replace("idea_", "").replace("benchmark_", "")
            lines.append(
                f"| {_fmt(pr_mean)} | {_fmt(pr_std)} | {_fmt(acc_mean, 3)} | "
                f"{params_s} | {scale} | {n} | {short} |"
            )
        lines.append("")

    # global leaderboard
    lines.append("## Global leaderboard (top 25 by mean test PR-AUC)")
    lines.append("")
    if global_best:
        global_best.sort(reverse=True)
        lines.append("| rank | test PR-AUC | experiment | variant |")
        lines.append("|---:|---:|---|---|")
        for i, (pr, exp, variant) in enumerate(global_best[:25], 1):
            short = variant.replace("idea_", "").replace("benchmark_", "")
            lines.append(f"| {i} | {_fmt(pr)} | {exp} | {short} |")
    else:
        lines.append("_No completed runs anywhere yet._")
    lines.append("")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUTPUT} ({len(global_best)} variants across "
          f"{sum(1 for _, d, _ in EXPERIMENTS if (REPO_ROOT / d).exists())} active pipelines)")


if __name__ == "__main__":
    main()

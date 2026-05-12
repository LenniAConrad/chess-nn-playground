from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chess_nn_playground.utils.paths import utc_timestamp


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_text(path: Path, max_chars: int = 2000) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if len(text) > max_chars:
        return text[:max_chars] + "\n...<truncated>"
    return text


def _count_samples(class_counts: Any) -> int | None:
    if not isinstance(class_counts, dict):
        return None
    total = 0
    for value in class_counts.values():
        try:
            total += int(value)
        except Exception:
            return None
    return total


def _metric(metrics: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metrics:
            return metrics[key]
    return None


def _run_summary(run_dir: Path) -> dict[str, Any]:
    metrics = _load_json(run_dir / "metrics_final.json")
    metadata = _load_json(run_dir / "run_metadata.json")
    class_counts = metadata.get("class_counts", {})
    train_counts = class_counts.get("train", {}) if isinstance(class_counts, dict) else {}
    val_counts = class_counts.get("val", {}) if isinstance(class_counts, dict) else {}
    test_counts = class_counts.get("test", {}) if isinstance(class_counts, dict) else {}
    train_samples = _count_samples(train_counts)
    val_samples = _count_samples(val_counts)
    test_samples = _count_samples(test_counts)
    total_samples = sum(value for value in [train_samples, val_samples, test_samples] if value is not None)
    is_smoke = "smoke" in run_dir.name.lower() or total_samples <= 100

    return {
        "run_dir": str(run_dir),
        "run_name": metadata.get("run_name", run_dir.name),
        "timestamp": metadata.get("timestamp"),
        "mode": metadata.get("mode"),
        "model_name": metadata.get("model_name"),
        "num_params": metadata.get("num_params"),
        "device": metadata.get("device"),
        "dataset_path": metadata.get("dataset_path"),
        "split_paths": metadata.get("split_paths"),
        "class_counts": class_counts,
        "sample_counts": {
            "train": train_samples,
            "val": val_samples,
            "test": test_samples,
            "total": total_samples,
        },
        "is_smoke_or_tiny_run": is_smoke,
        "best_epoch": metrics.get("best_epoch") or metadata.get("best_epoch"),
        "validation": {
            "loss": metrics.get("loss"),
            "accuracy": metrics.get("accuracy"),
            "f1": _metric(metrics, "f1", "macro_f1"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "roc_auc": metrics.get("roc_auc"),
            "pr_auc": metrics.get("pr_auc"),
        },
        "test": {
            "loss": metrics.get("test_loss"),
            "accuracy": metrics.get("test_accuracy"),
            "f1": _metric(metrics, "test_f1", "test_macro_f1"),
            "precision": metrics.get("test_precision"),
            "recall": metrics.get("test_recall"),
            "roc_auc": metrics.get("test_roc_auc"),
            "pr_auc": metrics.get("test_pr_auc"),
        },
        "metric_reasons": {
            "validation": metrics.get("metric_reasons", {}),
            "test": metrics.get("test_metric_reasons", {}),
        },
        "artifacts": {
            "metrics_final": str(run_dir / "metrics_final.json"),
            "run_metadata": str(run_dir / "run_metadata.json"),
            "run_summary": str(run_dir / "run_summary.md"),
            "report_html": str(run_dir / "report.html"),
            "checkpoint_best": metadata.get("checkpoint_best"),
            "checkpoint_last": metadata.get("checkpoint_last"),
        },
        "notes": metadata.get("notes"),
    }


def discover_run_summaries(results_dir: str | Path = "results", max_runs: int | None = None) -> list[dict[str, Any]]:
    root = Path(results_dir)
    runs = []
    for metrics_path in sorted(root.glob("*/metrics_final.json")):
        runs.append(_run_summary(metrics_path.parent))
    runs.sort(key=lambda row: str(row.get("timestamp") or row.get("run_name") or ""), reverse=True)
    if max_runs is not None and max_runs > 0:
        runs = runs[:max_runs]
    return runs


def _read_registry(registry_path: str | Path) -> list[dict[str, Any]]:
    path = Path(registry_path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"_parse_error": line})
    return rows


def build_chatgpt_run_prompt(
    results_dir: str | Path = "results",
    leaderboard_path: str | Path = "results/leaderboard.md",
    registry_path: str | Path = "ideas/all_ideas/registry/registry.jsonl",
    max_runs: int | None = 25,
) -> str:
    runs = discover_run_summaries(results_dir, max_runs=max_runs)
    context = {
        "generated_at": utc_timestamp(),
        "results_dir": str(results_dir),
        "runs_found": len(runs),
        "max_runs_included": max_runs,
        "active_label_policy": {
            "known_non_puzzle": {"coarse_label": 0, "fine_label": 0},
            "candidate_1_or_2_unresolved": {"coarse_label": 1, "fine_label": None},
            "verified_near_puzzle": {"coarse_label": 1, "fine_label": 1},
            "verified_puzzle": {"coarse_label": 1, "fine_label": 2},
        },
        "runs": runs,
        "idea_registry": _read_registry(registry_path),
    }
    leaderboard = _read_text(Path(leaderboard_path), max_chars=5000)

    leaderboard_section = leaderboard or "No leaderboard file is currently available."
    return f"""# ChatGPT Pro Prompt: Tested Runs Context

You are advising the `chess-nn-playground` research project from a pasted run summary.

Your job is to reason from tested evidence only. Do not invent results, do not treat unresolved labels as verified classes, and do not propose a new neural architecture unless the user explicitly asks for idea generation.

## Non-Negotiable Label Rules

- `known_non_puzzle`: `coarse_label = 0`, `fine_label = 0`
- `candidate_1_or_2_unresolved`: `coarse_label = 1`, `fine_label = null`
- `verified_near_puzzle`: `coarse_label = 1`, `fine_label = 1`
- `verified_puzzle`: `coarse_label = 1`, `fine_label = 2`

Unresolved candidate positions are not verified near-puzzles and are not verified real puzzles.

Stockfish scores, PVs, node counts, verification metadata, source labels, and proposed labels must not be used as neural-network input features. They may only be targets, metadata, audit fields, or weak-label proposal evidence.

## Current Tested Run Context

This block was generated automatically from local `results/*/metrics_final.json` and `run_metadata.json` files.

```json
{json.dumps(context, indent=2, sort_keys=True)}
```

## Current Leaderboard

```markdown
{leaderboard_section}
```

## How To Interpret These Runs

- If `is_smoke_or_tiny_run` is true, treat the run only as infrastructure validation.
- Do not make scientific claims from smoke runs.
- A benchmark is meaningful only if it has enough samples, balanced or documented class counts, leakage-safe splits, and a clear dataset source.
- For `coarse_binary`, the task is known non-puzzle versus unresolved candidate pool.
- For `fine_3class`, only use rows where `fine_label` is truly `0`, `1`, or `2`; do not infer class 1 or class 2 from the candidate pool.

## What I Want You To Do

When I paste this prompt into ChatGPT Pro, do the following:

1. Summarize what has actually been tested.
2. Identify which runs are smoke tests and which, if any, are real benchmarks.
3. State the best current result per mode, but only if the run is not tiny.
4. List metric gaps, missing labels, class imbalance, leakage risks, and data-quality blockers.
5. Recommend the next 1-3 practical experiments or data actions.
6. If suggesting a new run, give exact command-line steps and what result would count as success or failure.
7. If suggesting label work, keep verified labels separate from engine-proposed or weak labels.
8. Do not repeat old ideas or rebrand ordinary CNN hyperparameter changes as novel research ideas.

Keep the answer concrete and evidence-bound.
"""


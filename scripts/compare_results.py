#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from _bootstrap import bootstrap

bootstrap()

from chess_nn_playground.evaluation.chatgpt_prompt import build_chatgpt_run_prompt
from chess_nn_playground.evaluation.training_plots import build_global_training_dashboard
from chess_nn_playground.ideas.implementation_kind import detect_idea_implementation_kind


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No result runs found."
    try:
        return df.to_markdown(index=False)
    except Exception:
        columns = list(df.columns)
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for _, row in df.iterrows():
            values = [str(row.get(column, "")) for column in columns]
            values = [value.replace("|", "\\|").replace("\n", " ") for value in values]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _best_val_f1(metrics: dict[str, Any], mode: str | None) -> Any:
    if mode in {"coarse_binary", "puzzle_binary"}:
        return metrics.get("f1")
    return metrics.get("macro_f1")


def _worst_slice(run_dir: Path, split: str) -> dict[str, Any]:
    data = _load_json(run_dir / f"slice_metrics_{split}.json")
    rows = data.get("worst_slices", [])
    if not rows:
        return {}
    return rows[0]


def _seed_group_name(run_name: str) -> str:
    return re.sub(r"_seed\d+(?=$|_)", "", run_name)


def _seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    working = df.copy()
    working["seed_group"] = working["run_name"].map(_seed_group_name)
    group_cols = ["seed_group", "mode", "model_name", "implementation_kind", "architecture_scale"]
    metric_cols = [
        "best_val_f1",
        "best_val_accuracy",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_pr_auc",
        "test_roc_auc",
        "pr_auc",
        "roc_auc",
        "estimated_mflops_per_position",
        "fit_elapsed_seconds",
        "train_samples_per_second",
        "val_samples_per_second",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in working.groupby(group_cols, dropna=False):
        row = {
            "run_group": keys[0],
            "mode": keys[1],
            "model_name": keys[2],
            "implementation_kind": keys[3],
            "architecture_scale": keys[4],
            "seeds": int(group["seed"].nunique()) if "seed" in group else len(group),
            "runs": int(len(group)),
            "report_paths": "; ".join(str(path) for path in group["report_path"].tolist()),
        }
        for col in metric_cols:
            if col in group and group[col].notna().any():
                row[f"{col}_mean"] = float(group[col].mean())
                row[f"{col}_std"] = float(group[col].std(ddof=0))
        rows.append(row)
    summary = pd.DataFrame(rows)
    sort_col = "test_pr_auc_mean" if "test_pr_auc_mean" in summary else "test_f1_mean"
    if sort_col in summary:
        summary = summary.sort_values(sort_col, ascending=False, na_position="last")
    return summary


def _iter_metric_paths(results_dir: Path) -> list[Path]:
    return sorted(results_dir.rglob("metrics_final.json"))


def _load_idea_kind_maps(ideas_root: Path = Path("ideas/all_ideas/registry")) -> tuple[dict[str, str], dict[str, str]]:
    by_id: dict[str, str] = {}
    by_model: dict[str, str] = {}
    for folder in sorted(ideas_root.glob("i[0-9][0-9][0-9]_*")):
        row = detect_idea_implementation_kind(folder)
        by_id[row.idea_id] = row.detected_kind
        if row.model_name:
            by_model[row.model_name] = row.detected_kind
    return by_id, by_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a leaderboard from result directories.")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    rows = []
    idea_kind_by_id, idea_kind_by_model = _load_idea_kind_maps()
    for metrics_path in _iter_metric_paths(Path(args.results_dir)):
        run_dir = metrics_path.parent
        metrics = _load_json(metrics_path)
        metadata = _load_json(run_dir / "run_metadata.json")
        try:
            config = yaml.safe_load((run_dir / "config_resolved.yaml").read_text(encoding="utf-8")) or {}
        except Exception:
            config = {}
        architecture_scale = metadata.get("architecture_scale") or config.get("architecture_scale") or {}
        speed = metadata.get("speed") or metrics.get("speed") or {}
        complexity = metadata.get("complexity") or _load_json(run_dir / "complexity_estimate.json") or {}
        worst_test_slice = _worst_slice(run_dir, "test")
        worst_val_slice = _worst_slice(run_dir, "val")
        class_counts = metadata.get("class_counts", {})
        idea_id = str(config.get("idea_id") or "")
        model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
        model_name = str(metadata.get("model_name") or model_cfg.get("name") or "")
        implementation_kind = idea_kind_by_id.get(idea_id) or idea_kind_by_model.get(model_name)
        row = {
            "run_name": metadata.get("run_name", run_dir.name),
            "created_at": metadata.get("timestamp"),
            "seed": metadata.get("seed"),
            "mode": metadata.get("mode"),
            "model_name": metadata.get("model_name"),
            "implementation_kind": implementation_kind,
            "architecture_scale": (
                architecture_scale.get("variant", "base") if isinstance(architecture_scale, dict) else "base"
            ),
            "scale_multiplier": (
                architecture_scale.get("multiplier") if isinstance(architecture_scale, dict) else None
            ),
            "num_params": metadata.get("num_params"),
            "estimated_flops_per_position": (
                complexity.get("estimated_flops_per_position") if isinstance(complexity, dict) else None
            ),
            "estimated_mflops_per_position": (
                complexity.get("estimated_mflops_per_position") if isinstance(complexity, dict) else None
            ),
            "estimated_macs_per_position": (
                complexity.get("estimated_macs_per_position") if isinstance(complexity, dict) else None
            ),
            "fit_elapsed_seconds": speed.get("fit_elapsed_seconds") if isinstance(speed, dict) else None,
            "train_samples_per_second": (
                speed.get("train_samples_per_second") if isinstance(speed, dict) else None
            ),
            "val_samples_per_second": speed.get("val_samples_per_second") if isinstance(speed, dict) else None,
            "train_samples": sum(class_counts.get("train", {}).values()) if isinstance(class_counts.get("train"), dict) else None,
            "val_samples": sum(class_counts.get("val", {}).values()) if isinstance(class_counts.get("val"), dict) else None,
            "test_samples": sum(class_counts.get("test", {}).values()) if isinstance(class_counts.get("test"), dict) else None,
            "best_val_loss": metrics.get("loss"),
            "best_val_accuracy": metrics.get("accuracy"),
            "best_val_f1": _best_val_f1(metrics, metadata.get("mode")),
            "test_accuracy": metrics.get("test_accuracy"),
            "test_precision": metrics.get("test_precision"),
            "test_recall": metrics.get("test_recall"),
            "test_f1": metrics.get("test_f1") or metrics.get("test_macro_f1"),
            "test_roc_auc": metrics.get("test_roc_auc"),
            "test_pr_auc": metrics.get("test_pr_auc"),
            "roc_auc": metrics.get("roc_auc"),
            "pr_auc": metrics.get("pr_auc"),
            "worst_test_slice": (
                f"{worst_test_slice.get('column')}={worst_test_slice.get('slice')}" if worst_test_slice else None
            ),
            "worst_test_slice_accuracy": worst_test_slice.get("accuracy"),
            "worst_test_slice_rows": worst_test_slice.get("rows"),
            "worst_val_slice": f"{worst_val_slice.get('column')}={worst_val_slice.get('slice')}" if worst_val_slice else None,
            "worst_val_slice_accuracy": worst_val_slice.get("accuracy"),
            "checkpoint_path": metadata.get("checkpoint_best"),
            "report_path": str(run_dir / "run_summary.md"),
            "notes": metadata.get("notes"),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        sort_col = "best_val_f1"
        if "pr_auc" in df.columns and df["pr_auc"].notna().any():
            sort_col = "pr_auc"
        df = df.sort_values(sort_col, ascending=False, na_position="last")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "leaderboard.csv", index=False)
    markdown = _markdown_table(df)
    (results_dir / "leaderboard.md").write_text(markdown + "\n", encoding="utf-8")
    seed_summary = _seed_summary(df)
    seed_summary.to_csv(results_dir / "leaderboard_seed_summary.csv", index=False)
    seed_markdown = _markdown_table(seed_summary)
    (results_dir / "leaderboard_seed_summary.md").write_text(seed_markdown + "\n", encoding="utf-8")
    reports_dir = Path("reports")
    leaderboards_dir = reports_dir / "leaderboards"
    prompts_dir = reports_dir / "prompts"
    leaderboards_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    global_leaderboard_path = leaderboards_dir / "global_leaderboard.md"
    global_seed_summary_path = leaderboards_dir / "global_seed_summary.md"
    prompt_path = prompts_dir / "chatgpt_pro_run_prompt.md"
    global_leaderboard_path.write_text(markdown + "\n", encoding="utf-8")
    global_seed_summary_path.write_text(seed_markdown + "\n", encoding="utf-8")
    prompt = build_chatgpt_run_prompt(
        results_dir=args.results_dir,
        leaderboard_path=results_dir / "leaderboard.md",
    )
    prompt_path.write_text(prompt, encoding="utf-8")
    training_dashboard = build_global_training_dashboard(
        results_dir=args.results_dir,
        output_dir=Path("reports") / "training",
    )
    print(f"Saved {results_dir / 'leaderboard.csv'}")
    print(f"Saved {results_dir / 'leaderboard.md'}")
    print(f"Saved {results_dir / 'leaderboard_seed_summary.csv'}")
    print(f"Saved {results_dir / 'leaderboard_seed_summary.md'}")
    print(f"Saved {global_leaderboard_path}")
    print(f"Saved {global_seed_summary_path}")
    print(f"Saved {prompt_path}")
    print(f"Saved {training_dashboard['markdown']}")
    print(f"Saved {training_dashboard['html']}")
    for path in training_dashboard["plots"]:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()

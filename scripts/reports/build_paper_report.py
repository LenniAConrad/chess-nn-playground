#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _bootstrap import bootstrap

bootstrap()

from matplotlib.backends.backend_pdf import PdfPages

from chess_nn_playground.evaluation.plots import _plt
from chess_nn_playground.ideas.implementation import validate_idea_for_training
from chess_nn_playground.ideas.implementation_kind import detect_idea_implementation_kind
from chess_nn_playground.ideas.registry import list_ideas
from chess_nn_playground.models.complexity import estimate_model_complexity_from_config
from chess_nn_playground.models.registry import available_models
from chess_nn_playground.utils.paths import utc_timestamp


BASELINE_DESCRIPTIONS = {
    "simple_cnn": (
        "A compact spatial convolutional baseline. It learns local piece and square patterns from board tensors, "
        "then pools them into a puzzle logit or class logits."
    ),
    "cnn_baseline": (
        "Alias for the compact spatial convolutional baseline. It is useful as the simplest learned image-like "
        "board model."
    ),
    "residual_cnn": (
        "A deeper convolutional tower with residual connections. It tests whether additional spatial depth and "
        "skip connections improve tactical signal extraction over the plain CNN."
    ),
    "mlp": (
        "A dense baseline over flattened board features. It has no explicit spatial inductive bias, so it is a "
        "useful lower-bound comparison against chess-shaped architectures."
    ),
    "board_mlp": (
        "Alias for the dense board MLP baseline. It tests whether flattened board features are enough without "
        "convolutions or chess-specific structure."
    ),
    "nnue": (
        "A Stockfish-style NNUE baseline over compact board features. It emphasizes efficient accumulated sparse "
        "piece-square interactions rather than image-like convolution."
    ),
    "stockfish_nnue": (
        "A Stockfish-style NNUE baseline over compact board features. It gives a practical chess-specific "
        "reference point for the puzzle classifier."
    ),
    "lc0_bt4": (
        "An LC0 BT4-style residual tower over a 112-plane board tensor. It is the strongest chess-shaped baseline "
        "family and the main reference to beat."
    ),
    "lc0_bt4_classifier": (
        "An LC0 BT4-style residual classifier over the 112-plane tensor. It uses chess-engine-style spatial "
        "planes and a residual tower, but is trained from scratch for puzzle classification."
    ),
}


@dataclass
class RunRecord:
    run_dir: Path
    run_name: str
    key: str
    model_name: str
    mode: str
    seed: Any
    scale_variant: str
    scale_multiplier: Any
    input_encoding: str
    num_params: Any
    complexity: dict[str, Any]
    speed: dict[str, Any]
    metrics: dict[str, Any]
    metadata: dict[str, Any]


@dataclass
class Architecture:
    key: str
    title: str
    kind: str
    model_name: str
    idea_id: str | None = None
    folder: Path | None = None
    target_task: str = ""
    input_representation: str = ""
    implementation_status: str = ""
    implementation_kind: str = ""
    thesis: str = ""
    explanation: str = ""
    config_paths: set[str] = field(default_factory=set)
    planned_tasks: list[dict[str, Any]] = field(default_factory=list)
    runs: list[RunRecord] = field(default_factory=list)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _one_line(text: str, max_len: int = 900) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _first_paragraph(text: str) -> str:
    lines = text.splitlines()
    paragraph: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if paragraph:
                break
            continue
        if stripped.startswith("#") or stripped.startswith("```") or stripped.startswith("|"):
            continue
        paragraph.append(stripped)
        if len(" ".join(paragraph)) > 900:
            break
    return _one_line(" ".join(paragraph))


def _first_paragraph_after_heading(text: str, heading_prefix: str) -> str:
    lines = text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.strip().lower().startswith(heading_prefix.lower()):
            start = idx + 1
            break
    if start is None:
        return ""
    return _first_paragraph("\n".join(lines[start:]))


def _score(metrics: dict[str, Any], mode: str) -> float:
    keys = ["test_pr_auc", "pr_auc", "test_f1", "test_macro_f1", "f1", "macro_f1", "test_accuracy", "accuracy"]
    if mode == "fine_3class":
        keys = ["test_macro_f1", "macro_f1", "test_class2_pr_auc", "class2_pr_auc", "test_accuracy", "accuracy"]
    for key in keys:
        value = metrics.get(key)
        try:
            if value is not None and math.isfinite(float(value)):
                return float(value)
        except Exception:
            pass
    return -math.inf


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(number):
        return "-"
    return f"{number:.{digits}f}"


def _matrix_rate(metrics: dict[str, Any], row_idx: int, col_idx: int, prefix: str = "test_") -> str:
    matrix = metrics.get(f"{prefix}fine_to_binary_confusion_matrix") or metrics.get("fine_to_binary_confusion_matrix")
    if not isinstance(matrix, list) or len(matrix) <= row_idx:
        return "-"
    row = matrix[row_idx]
    if not isinstance(row, list) or len(row) <= col_idx:
        return "-"
    total = sum(float(item) for item in row if item is not None)
    if total <= 0:
        return "-"
    return _fmt(float(row[col_idx]) / total)


def _idea_key(idea_id: str | None) -> str | None:
    if not idea_id:
        return None
    return f"idea:{idea_id}"


def _model_key(model_name: str | None) -> str:
    return f"model:{model_name or 'unknown'}"


def _key_from_config(config: dict[str, Any], fallback_model: str | None = None) -> str:
    idea_id = config.get("idea_id")
    if idea_id:
        return _idea_key(str(idea_id)) or _model_key(fallback_model)
    model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    return _model_key(str(model_cfg.get("name") or fallback_model or "unknown"))


def load_idea_architectures() -> dict[str, Architecture]:
    architectures: dict[str, Architecture] = {}
    for entry in list_ideas():
        idea_id = str(entry.get("idea_id") or "")
        folder = Path(str(entry.get("folder") or ""))
        idea_yaml = _read_yaml(folder / "idea.yaml")
        config = _read_yaml(folder / "config.yaml")
        model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
        architecture_text = _read_text(folder / "architecture.md")
        thesis_text = _read_text(folder / "math_thesis.md")
        explanation = (
            _first_paragraph_after_heading(architecture_text, "## Architecture")
            or _first_paragraph(architecture_text)
            or str(idea_yaml.get("notes") or "")
        )
        thesis = (
            str(idea_yaml.get("short_thesis") or "")
            or _first_paragraph_after_heading(thesis_text, "## Thesis")
            or _first_paragraph(thesis_text)
        )
        key = _idea_key(idea_id)
        if key is None:
            continue
        architectures[key] = Architecture(
            key=key,
            title=str(entry.get("name") or idea_yaml.get("name") or folder.name),
            kind="registered idea",
            model_name=str(model_cfg.get("name") or idea_yaml.get("slug") or entry.get("slug") or ""),
            idea_id=idea_id,
            folder=folder,
            target_task=str(idea_yaml.get("target_task") or entry.get("target_task") or ""),
            input_representation=str(idea_yaml.get("input_representation") or ""),
            implementation_status=str(idea_yaml.get("implementation_status") or entry.get("implementation_status") or ""),
            implementation_kind=detect_idea_implementation_kind(folder).detected_kind,
            thesis=_one_line(thesis),
            explanation=_one_line(explanation),
        )
    return architectures


def load_benchmark_architectures() -> dict[str, Architecture]:
    architectures: dict[str, Architecture] = {}
    for config_path in sorted(Path("configs/benchmarks").rglob("*.yaml")):
        config = _read_yaml(config_path)
        model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
        model_name = str(model_cfg.get("name") or config_path.stem)
        key = _model_key(model_name)
        run_cfg = config.get("run", {}) if isinstance(config.get("run"), dict) else {}
        data_cfg = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
        arch = architectures.get(key)
        if arch is None:
            arch = Architecture(
                key=key,
                title=model_name.replace("_", " ").title(),
                kind="benchmark baseline",
                model_name=model_name,
                target_task=str(config.get("mode") or ""),
                input_representation=str(data_cfg.get("encoding") or ""),
                implementation_status="implemented",
                thesis=str(run_cfg.get("name") or config_path.stem),
                explanation=BASELINE_DESCRIPTIONS.get(
                    model_name,
                    "Benchmark architecture registered through the shared model registry and trained through the common artifact pipeline.",
                ),
            )
            architectures[key] = arch
        arch.config_paths.add(config_path.as_posix())
    return architectures


def _iter_metric_paths(results_dir: Path) -> list[Path]:
    return sorted(results_dir.rglob("metrics_final.json"))


def load_runs(results_dirs: list[Path]) -> list[RunRecord]:
    runs: list[RunRecord] = []
    for results_dir in results_dirs:
        for metrics_path in _iter_metric_paths(results_dir):
            run_dir = metrics_path.parent
            metrics = _read_json(metrics_path)
            metadata = _read_json(run_dir / "run_metadata.json")
            config = _read_yaml(run_dir / "config_resolved.yaml")
            model_name = str(
                metadata.get("model_name")
                or (config.get("model", {}) if isinstance(config.get("model"), dict) else {}).get("name")
                or "unknown"
            )
            key = _key_from_config(config, model_name)
            architecture_scale = metadata.get("architecture_scale") or config.get("architecture_scale") or {}
            speed = metadata.get("speed") or metrics.get("speed") or {}
            complexity = metadata.get("complexity") or _read_json(run_dir / "complexity_estimate.json") or {}
            runs.append(
                RunRecord(
                    run_dir=run_dir,
                    run_name=str(metadata.get("run_name") or run_dir.name),
                    key=key,
                    model_name=model_name,
                    mode=str(metadata.get("mode") or config.get("mode") or ""),
                    seed=metadata.get("seed", config.get("seed")),
                    scale_variant=(
                        str(architecture_scale.get("variant", "base"))
                        if isinstance(architecture_scale, dict)
                        else "base"
                    ),
                    scale_multiplier=(
                        architecture_scale.get("multiplier") if isinstance(architecture_scale, dict) else None
                    ),
                    input_encoding=str(metadata.get("input_encoding") or ""),
                    num_params=metadata.get("num_params"),
                    complexity=complexity if isinstance(complexity, dict) else {},
                    speed=speed if isinstance(speed, dict) else {},
                    metrics=metrics,
                    metadata=metadata,
                )
            )
    return runs


def load_planned_tasks(state_path: Path, generated_config_dir: Path | None = None) -> list[dict[str, Any]]:
    state = _read_json(state_path)
    rows: list[dict[str, Any]] = []
    complexity_cache: dict[str, dict[str, Any]] = {}
    for task_id, row in sorted((state.get("tasks") or {}).items()):
        config_path = Path(str(row.get("generated_config") or ""))
        if not config_path.exists() and generated_config_dir is not None:
            candidate = generated_config_dir / f"{task_id}.yaml"
            if candidate.exists():
                config_path = candidate
        config = _read_yaml(config_path)
        model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
        complexity = estimate_model_complexity_from_config(config, cache=complexity_cache)
        rows.append(
            {
                "task_id": task_id,
                "key": _key_from_config(config, str(model_cfg.get("name") or "")),
                "status": row.get("status"),
                "seed": row.get("seed"),
                "scale_variant": row.get("scale_variant", "base"),
                "scale_multiplier": row.get("scale_multiplier"),
                "complexity": complexity,
                "source_config": row.get("source_config"),
                "run_dir": row.get("run_dir"),
            }
        )
    return rows


def ensure_architectures(
    architectures: dict[str, Architecture],
    runs: list[RunRecord],
    planned: list[dict[str, Any]],
) -> dict[str, Architecture]:
    for run in runs:
        architectures.setdefault(
            run.key,
            Architecture(
                key=run.key,
                title=run.model_name.replace("_", " ").title(),
                kind="observed result",
                model_name=run.model_name,
                implementation_status="implemented",
                explanation="Architecture inferred from completed result metadata.",
            ),
        ).runs.append(run)
    for task in planned:
        architectures.setdefault(
            str(task["key"]),
            Architecture(
                key=str(task["key"]),
                title=str(task["key"]).split(":", 1)[-1].replace("_", " ").title(),
                kind="planned architecture",
                model_name=str(task["key"]).split(":", 1)[-1],
                implementation_status="planned",
                explanation="Architecture inferred from the paper-ready training plan.",
            ),
        ).planned_tasks.append(task)
    for arch in architectures.values():
        arch.runs.sort(key=lambda run: _score(run.metrics, run.mode), reverse=True)
    return architectures


def idea_validation_summary() -> tuple[Counter[str], list[str]]:
    counts: Counter[str] = Counter()
    failures: list[str] = []
    for entry in list_ideas():
        folder = Path(str(entry.get("folder") or ""))
        idea_yaml = _read_yaml(folder / "idea.yaml")
        implementation_status = str(idea_yaml.get("implementation_status") or entry.get("implementation_status") or "")
        if implementation_status not in {"implemented", "tested"}:
            counts["scaffold_only"] += 1
            continue
        report = validate_idea_for_training(folder)
        if report.get("valid"):
            counts["trainable"] += 1
        else:
            counts["invalid"] += 1
            failures.append(f"{entry.get('idea_id')}: " + "; ".join(report.get("issues", [])[:3]))
    return counts, failures


def idea_implementation_kind_summary() -> Counter[str]:
    counts: Counter[str] = Counter()
    for entry in list_ideas():
        folder = Path(str(entry.get("folder") or ""))
        counts[detect_idea_implementation_kind(folder).detected_kind] += 1
    return counts


def _draw_text(ax: Any, text: str, x: float, y: float, width: int = 95, size: float = 9.5, weight: str = "normal") -> float:
    for paragraph in str(text).split("\n"):
        wrapped = textwrap.wrap(paragraph, width=width) or [""]
        for line in wrapped:
            ax.text(x, y, line, ha="left", va="top", fontsize=size, fontweight=weight, family="DejaVu Sans")
            y -= size / 720 * 1.52
        y -= size / 720 * 0.45
    return y


def _new_page(pdf: PdfPages, title: str) -> tuple[Any, Any]:
    plt = _plt()
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.06, 0.05, 0.88, 0.9])
    ax.axis("off")
    ax.text(0.0, 1.035, title, ha="left", va="top", fontsize=18, fontweight="bold", transform=ax.transAxes)
    ax.text(1.0, 1.035, "Chess NN Playground", ha="right", va="top", fontsize=9, transform=ax.transAxes)
    return fig, ax


def _save_page(pdf: PdfPages, fig: Any) -> None:
    pdf.savefig(fig)
    _plt().close(fig)


def _metric_rows(runs: list[RunRecord], limit: int = 6) -> list[list[str]]:
    rows: list[list[str]] = []
    for run in runs[:limit]:
        metrics = run.metrics
        train_sps = run.speed.get("train_samples_per_second") if isinstance(run.speed, dict) else None
        mflops = run.complexity.get("estimated_mflops_per_position") if isinstance(run.complexity, dict) else None
        rows.append(
            [
                run.run_name[:38],
                run.scale_variant,
                str(run.seed) if run.seed is not None else "-",
                _fmt(metrics.get("test_f1") or metrics.get("test_macro_f1")),
                _fmt(metrics.get("test_pr_auc") or metrics.get("pr_auc")),
                _fmt(mflops, digits=2),
                _fmt(train_sps),
                str(run.num_params or "-"),
            ]
        )
    return rows


def _draw_table(ax: Any, headers: list[str], rows: list[list[str]], x: float, y: float, widths: list[float], size: float = 7.5) -> float:
    line_h = size / 720 * 1.9
    ax.text(x, y, " | ".join(headers), ha="left", va="top", fontsize=size, fontweight="bold", family="DejaVu Sans Mono")
    y -= line_h
    ax.text(x, y, "-+-".join("-" * max(3, int(w * 100)) for w in widths), ha="left", va="top", fontsize=size, family="DejaVu Sans Mono")
    y -= line_h
    for row in rows:
        cells = []
        for cell, width in zip(row, widths):
            max_chars = max(4, int(width * 100))
            value = str(cell)
            cells.append(value[: max_chars - 1] if len(value) > max_chars else value.ljust(max_chars))
        ax.text(x, y, " | ".join(cells), ha="left", va="top", fontsize=size, family="DejaVu Sans Mono")
        y -= line_h
    return y - line_h * 0.4


def _draw_image_page(pdf: PdfPages, path: Path, title: str) -> None:
    if not path.exists():
        return
    plt = _plt()
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.88])
    ax.axis("off")
    fig.suptitle(title, fontsize=16, fontweight="bold")
    try:
        image = plt.imread(path)
        ax.imshow(image)
    except Exception:
        ax.text(0.05, 0.8, f"Could not load image: {path}", fontsize=12)
    pdf.savefig(fig)
    plt.close(fig)


def _draw_problem_pages(
    pdf: PdfPages,
    *,
    architectures: dict[str, Architecture],
    runs: list[RunRecord],
    planned: list[dict[str, Any]],
    idea_counts: Counter[str],
    idea_failures: list[str],
    implementation_kind_counts: Counter[str],
) -> None:
    fig, ax = _new_page(pdf, "Problem Statement")
    y = 0.98
    y = _draw_text(
        ax,
        "This report summarizes a large-scale neural architecture benchmark for chess puzzle classification. "
        "The central question is whether a neural network can recognize real puzzle signal from a board position "
        "alone, especially when the negative examples include near-puzzles that look tactically sharp but are not "
        "verified puzzles.",
        0.0,
        y,
        width=94,
        size=10.5,
    )
    y = _draw_text(
        ax,
        "Task contract: source fine label 0 is known non-puzzle, source fine label 1 is verified near-puzzle, "
        "and source fine label 2 is verified puzzle. In the main binary benchmark, labels 0 and 1 map to output 0, "
        "while label 2 maps to output 1. Reports preserve the 3x2 fine-label-to-binary diagnostic matrix so near-puzzle "
        "false positives remain visible.",
        0.0,
        y,
        width=94,
        size=10,
    )
    summary_rows = [
        ["Registered model keys", str(len(available_models()))],
        ["Architecture pages", str(len(architectures))],
        ["Completed result runs loaded", str(len(runs))],
        ["Planned paper-ready tasks", str(len(planned))],
        ["Trainable registered ideas", str(idea_counts.get("trainable", 0))],
        ["Idea validation failures", str(idea_counts.get("invalid", 0))],
        ["Bespoke idea models", str(implementation_kind_counts.get("bespoke_model", 0))],
        ["Shared-probe idea variants", str(implementation_kind_counts.get("shared_probe_variant", 0))],
        ["Other shared-scaffold ideas", str(implementation_kind_counts.get("other_shared_scaffold", 0))],
        ["Unknown implementation kind", str(implementation_kind_counts.get("unknown", 0))],
    ]
    y = _draw_table(ax, ["Quantity", "Value"], summary_rows, 0.0, y, [0.45, 0.12], size=9)
    if idea_failures:
        y = _draw_text(ax, "Idea validation failures:", 0.0, y, width=90, size=10, weight="bold")
        y = _draw_text(ax, "\n".join(f"- {item}" for item in idea_failures[:12]), 0.0, y, width=100, size=8.5)
    y = _draw_text(
        ax,
        "Primary metrics: PR AUC, ROC AUC, F1, recall, precision, near-puzzle false-positive rate, puzzle recall, "
        "calibration error, and worst-slice behavior. Promotion-grade claims should use matched baselines and repeated "
        "seeds; paper-grade claims should include confidence intervals and ablations.",
        0.0,
        y,
        width=94,
        size=9.5,
    )
    _save_page(pdf, fig)

    fig, ax = _new_page(pdf, "Experiment Protocol")
    y = 0.98
    y = _draw_text(
        ax,
        "Reliable runs use the canonical tagged split under data/splits/crtk_sample_3class_unique_crtk_tags/, "
        "device: nvidia, deterministic seeds, the shared trainer, and the common artifact pipeline. Each run should "
        "save metrics, checkpoints, predictions, confusion matrices, slice reports, markdown summaries, and HTML reports.",
        0.0,
        y,
        width=94,
        size=10,
    )
    y = _draw_text(
        ax,
        "The mass-training runner materializes generated configs, writes a resumable state file, uses fixed run "
        "directories, expands each architecture into base, scale_up, and scale_xl variants, estimates inference "
        "FLOPs/MACs from generated configs, records speed and throughput metrics, resumes from checkpoint_last.pt when available, validates artifacts, rebuilds "
        "leaderboards, generates aggregate training plots, and then builds this PDF report.",
        0.0,
        y,
        width=94,
        size=10,
    )
    status_counts = Counter(str(row.get("status") or "unknown") for row in planned)
    rows = [[status, str(count)] for status, count in sorted(status_counts.items())]
    if rows:
        y = _draw_table(ax, ["Planned task status", "Count"], rows, 0.0, y, [0.32, 0.12], size=9)
    y = _draw_text(
        ax,
        "Important interpretation rule: short smoke and triage runs are useful for catching implementation bugs, "
        "but they are not final evidence. A model is not considered better than LC0 BT4, NNUE, VetoSelect, or Dykstra "
        "unless it wins under the same split, seed protocol, convergence budget, and reporting contract. "
        "Implementation kind is separate from trainability: shared-probe variants are ResearchPacketProbe wrappers, "
        "not bespoke architectures.",
        0.0,
        y,
        width=94,
        size=9.5,
    )
    _save_page(pdf, fig)


def _draw_leaderboard_page(pdf: PdfPages, runs: list[RunRecord]) -> None:
    fig, ax = _new_page(pdf, "Results Summary")
    y = 0.98
    if not runs:
        _draw_text(
            ax,
            "No completed result runs were found in the requested results directory. This report still contains the "
            "planned architecture pages, and the same script will populate metrics after training finishes.",
            0.0,
            y,
            width=94,
            size=10,
        )
        _save_page(pdf, fig)
        return
    sorted_runs = sorted(runs, key=lambda run: _score(run.metrics, run.mode), reverse=True)
    rows = _metric_rows(sorted_runs, limit=18)
    y = _draw_text(ax, "Top completed runs by primary score:", 0.0, y, width=90, size=10, weight="bold")
    _draw_table(
        ax,
        ["Run", "Scale", "Seed", "F1", "PR AUC", "MFLOPs", "Train/s", "Params"],
        rows,
        0.0,
        y,
        [0.32, 0.08, 0.05, 0.07, 0.08, 0.07, 0.08, 0.09],
        size=6.8,
    )
    _save_page(pdf, fig)


def _architecture_sort_key(item: Architecture) -> tuple[int, str]:
    if item.kind == "benchmark baseline":
        return (0, item.title)
    if item.idea_id:
        try:
            return (1, f"{int(item.idea_id[1:]):04d}")
        except Exception:
            return (1, item.idea_id)
    return (2, item.title)


def _draw_architecture_page(pdf: PdfPages, arch: Architecture) -> None:
    label = f"{arch.idea_id}: {arch.title}" if arch.idea_id else arch.title
    fig, ax = _new_page(pdf, label[:100])
    y = 0.98
    fields = [
        ["Kind", arch.kind],
        ["Model key", arch.model_name or "-"],
        ["Target task", arch.target_task or "-"],
        ["Input", arch.input_representation or "-"],
        ["Implementation", arch.implementation_status or "-"],
        ["Implementation kind", arch.implementation_kind or "-"],
        ["Completed runs", str(len(arch.runs))],
        ["Planned tasks", str(len(arch.planned_tasks))],
    ]
    y = _draw_table(ax, ["Field", "Value"], fields, 0.0, y, [0.22, 0.58], size=8.5)
    if arch.thesis:
        y = _draw_text(ax, "Thesis", 0.0, y, width=90, size=10, weight="bold")
        y = _draw_text(ax, arch.thesis, 0.0, y, width=94, size=8.8)
    if arch.explanation:
        y = _draw_text(ax, "How It Works", 0.0, y, width=90, size=10, weight="bold")
        y = _draw_text(ax, arch.explanation, 0.0, y, width=94, size=8.8)
    if arch.config_paths:
        y = _draw_text(ax, "Benchmark configs: " + ", ".join(sorted(arch.config_paths)[:4]), 0.0, y, width=110, size=7.5)
    if arch.runs:
        y = _draw_text(ax, "Best Completed Runs", 0.0, y, width=90, size=10, weight="bold")
        y = _draw_table(
            ax,
            ["Run", "Scale", "Seed", "F1", "PR AUC", "MFLOPs", "Train/s", "Params"],
            _metric_rows(arch.runs, limit=5),
            0.0,
            y,
            [0.31, 0.08, 0.05, 0.07, 0.08, 0.07, 0.08, 0.09],
            size=6.4,
        )
        best = arch.runs[0]
        y = _draw_text(ax, f"Best run directory: {best.run_dir.as_posix()}", 0.0, y, width=110, size=7.5)
    else:
        y = _draw_text(
            ax,
            "No completed result is linked yet. This architecture is included because it is registered, benchmarked, "
            "or planned by the paper-ready runner.",
            0.0,
            y,
            width=94,
            size=8.5,
        )
    if arch.planned_tasks:
        counts = Counter(str(row.get("status") or "unknown") for row in arch.planned_tasks)
        text = ", ".join(f"{status}: {count}" for status, count in sorted(counts.items()))
        scale_counts = Counter(str(row.get("scale_variant") or "base") for row in arch.planned_tasks)
        scale_text = ", ".join(f"{scale}: {count}" for scale, count in sorted(scale_counts.items()))
        y = _draw_text(ax, f"Planned task statuses: {text}", 0.0, max(y, 0.1), width=110, size=7.5)
        y = _draw_text(ax, f"Planned architecture scales: {scale_text}", 0.0, y, width=110, size=7.5)
        by_scale: dict[str, dict[str, Any]] = {}
        for task in arch.planned_tasks:
            scale = str(task.get("scale_variant") or "base")
            complexity = task.get("complexity")
            if scale not in by_scale and isinstance(complexity, dict):
                by_scale[scale] = complexity
        if by_scale:
            flop_text = ", ".join(
                f"{scale}: {_fmt(item.get('estimated_mflops_per_position'), digits=2)} MFLOPs/pos"
                for scale, item in sorted(by_scale.items())
            )
            _draw_text(ax, f"Estimated inference cost: {flop_text}", 0.0, y, width=110, size=7.5)
    _save_page(pdf, fig)


def build_report(
    *,
    results_dirs: list[Path],
    state_path: Path,
    generated_config_dir: Path | None,
    output_path: Path,
    training_report_dir: Path,
    max_architectures: int | None,
) -> dict[str, Any]:
    idea_arch = load_idea_architectures()
    benchmark_arch = load_benchmark_architectures()
    architectures = {**benchmark_arch, **idea_arch}
    runs = load_runs(results_dirs)
    planned = load_planned_tasks(state_path, generated_config_dir=generated_config_dir)
    architectures = ensure_architectures(architectures, runs, planned)
    idea_counts, idea_failures = idea_validation_summary()
    implementation_kind_counts = idea_implementation_kind_summary()

    sorted_architectures = sorted(architectures.values(), key=_architecture_sort_key)
    if max_architectures is not None:
        sorted_architectures = sorted_architectures[: max(0, max_architectures)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        metadata = pdf.infodict()
        metadata["Title"] = "Chess NN Playground Paper-Ready Architecture Report"
        metadata["Author"] = "Chess NN Playground"
        metadata["Subject"] = "Mass neural architecture benchmark for chess puzzle classification"
        metadata["Keywords"] = "chess, neural networks, benchmark, puzzle classification"
        metadata["CreationDate"] = None

        _draw_problem_pages(
            pdf,
            architectures=architectures,
            runs=runs,
            planned=planned,
            idea_counts=idea_counts,
            idea_failures=idea_failures,
            implementation_kind_counts=implementation_kind_counts,
        )
        _draw_leaderboard_page(pdf, runs)
        _draw_image_page(pdf, training_report_dir / "training_final_scores.png", "Global Final Scores")
        _draw_image_page(pdf, training_report_dir / "all_training_curves.png", "Global Training Curves")
        for arch in sorted_architectures:
            _draw_architecture_page(pdf, arch)

    summary = {
        "generated_at": utc_timestamp(),
        "output_path": output_path.as_posix(),
        "results_dirs": [path.as_posix() for path in results_dirs],
        "architecture_pages": len(sorted_architectures),
        "completed_runs": len(runs),
        "planned_tasks": len(planned),
        "idea_validation": dict(idea_counts),
        "idea_validation_failures": idea_failures,
        "idea_implementation_kinds": dict(implementation_kind_counts),
    }
    output_path.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a multi-page paper-style PDF report from training results.")
    parser.add_argument("--results-dir", nargs="+", default=["results"], help="One or more result directories to scan.")
    parser.add_argument("--state-path", default="reports/paper_ready_all/state.json")
    parser.add_argument("--generated-config-dir", default="reports/paper_ready_all/generated_configs")
    parser.add_argument("--training-report-dir", default="reports/training")
    parser.add_argument("--output", default="reports/paper_ready_all/paper_report.pdf")
    parser.add_argument("--max-architectures", type=int, default=None, help="Debug limit for architecture pages.")
    args = parser.parse_args()

    summary = build_report(
        results_dirs=[Path(item) for item in args.results_dir],
        state_path=Path(args.state_path),
        generated_config_dir=Path(args.generated_config_dir) if args.generated_config_dir else None,
        output_path=Path(args.output),
        training_report_dir=Path(args.training_report_dir),
        max_architectures=args.max_architectures,
    )
    print(f"Saved {summary['output_path']}")
    print(f"Architecture pages: {summary['architecture_pages']}")
    print(f"Completed runs: {summary['completed_runs']}")
    print(f"Planned tasks: {summary['planned_tasks']}")


if __name__ == "__main__":
    main()

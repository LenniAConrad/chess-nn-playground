#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shlex
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from _bootstrap import bootstrap

bootstrap()

from chess_nn_playground.training.config_validation import validate_training_config
from chess_nn_playground.training.trainer import config_fingerprint
from chess_nn_playground.ideas.implementation import validate_idea_for_training
from chess_nn_playground.ideas.schema import IDEA_FOLDER_GLOB
from chess_nn_playground.utils.config import load_yaml, save_yaml
from chess_nn_playground.utils.paths import utc_timestamp
from scripts.validate_run_artifacts import validate_run_artifacts


TRAINABLE_IDEA_IMPLEMENTATION_STATES = {"implemented", "tested"}

DEFAULT_SUITES = [
    Path("configs/suites/network_signal_benchmark_suite.yaml"),
    Path("configs/suites/network_signal_fine3_benchmark_suite.yaml"),
    Path("configs/suites/experiment_suite.yaml"),
]

DEFAULT_SCALE_VARIANTS_TEXT = "base:1.0,scale_up:1.5,scale_xl:2.0"
DEFAULT_BATCH_SIZE_CAPS_TEXT = "base:256,scale_up:192,scale_xl:128"

WIDTH_SCALE_KEYS = {
    "accumulator_size",
    "board_channels",
    "branch_dim",
    "channels",
    "classifier_hidden",
    "edit_feature_dim",
    "encoder_channels",
    "gate_dim",
    "geom_dim",
    "hidden_dim",
    "latent_dim",
    "move_feature_dim",
    "path_dim",
    "relation_channels",
    "relation_dim",
    "se_channels",
    "square_dim",
    "token_dim",
    "trunk_channels",
    "value_channels",
    "value_hidden",
}

DEPTH_SCALE_KEYS = {
    "blocks",
    "depth",
    "num_blocks",
    "pair_mixer_layers",
    "stem_depth",
    "transition_layers",
}

CAPACITY_SCALE_KEYS = {
    "and_beam",
    "atoms_per_group",
    "max_actions",
    "max_attackers",
    "max_defenders",
    "max_edits",
    "max_invalid",
    "max_moves",
    "max_nodes",
    "max_obligations",
    "max_replies_per_action",
    "max_resources",
    "max_tokens",
    "motif_count",
    "num_atom_groups",
    "or_beam",
    "role_count",
    "selected_k",
    "slack_count",
    "solver_cycles",
    "solver_steps",
}


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("_") or "task"


def _atomic_write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _format_elapsed(seconds: float | int | None) -> str:
    if seconds is None:
        return "-"
    total = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _task_elapsed_seconds(row: dict[str, Any]) -> float | None:
    value = row.get("elapsed_seconds")
    if value is None:
        speed = row.get("speed_summary")
        if isinstance(speed, dict):
            value = speed.get("fit_elapsed_seconds")
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if seconds > 0 else None


def eta_snapshot(tasks: list[dict[str, Any]], jobs: int = 1) -> dict[str, Any]:
    """Estimate remaining wall time from observed task durations."""

    task_count = len(tasks)
    if task_count == 0:
        return {
            "eta_seconds": 0.0,
            "eta": "0s",
            "average_task_seconds": None,
            "average_task_elapsed": "-",
            "processed_tasks": 0,
            "remaining_estimate_tasks": 0,
            "eta_jobs": max(1, int(jobs or 1)),
        }

    elapsed = [
        seconds
        for task in tasks
        if (seconds := _task_elapsed_seconds(task["state"])) is not None
    ]
    eta_jobs = max(1, int(jobs or 1))
    processed = len(elapsed)
    remaining = max(0, task_count - processed)
    if not elapsed:
        return {
            "eta_seconds": None,
            "eta": "not available yet",
            "average_task_seconds": None,
            "average_task_elapsed": "-",
            "processed_tasks": 0,
            "remaining_estimate_tasks": remaining,
            "eta_jobs": eta_jobs,
        }

    average = sum(elapsed) / len(elapsed)
    eta_seconds = average * remaining / eta_jobs
    return {
        "eta_seconds": eta_seconds,
        "eta": _format_elapsed(eta_seconds),
        "average_task_seconds": average,
        "average_task_elapsed": _format_elapsed(average),
        "processed_tasks": processed,
        "remaining_estimate_tasks": remaining,
        "eta_jobs": eta_jobs,
    }


def _event_summary(record: dict[str, Any]) -> str:
    keys = [
        "task_id",
        "status",
        "progress",
        "cuda_visible_devices",
        "batch_size",
        "elapsed",
        "eta",
        "run_dir",
        "log_path",
        "source_config",
    ]
    parts = [f"{key}={record[key]}" for key in keys if record.get(key) not in (None, "")]
    return " | ".join(parts) if parts else "-"


def append_event(args: argparse.Namespace, state: dict[str, Any], event: str, **fields: Any) -> None:
    record = {
        "at": utc_timestamp(),
        "event": event,
        "runner_started_at": state.get("last_started_at"),
        **fields,
    }
    event_log = Path(args.event_log)
    event_log.parent.mkdir(parents=True, exist_ok=True)
    with event_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    timeline = Path(args.timeline)
    if not timeline.exists():
        timeline.parent.mkdir(parents=True, exist_ok=True)
        timeline.write_text("# Paper-Ready Training Timeline\n\n", encoding="utf-8")
    with timeline.open("a", encoding="utf-8") as handle:
        handle.write(f"- `{record['at']}` `{event}` {_event_summary(record)}\n")


def _task_event_fields(row: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    keys = [
        "task_id",
        "kind",
        "seed",
        "scale_variant",
        "scale_multiplier",
        "batch_size",
        "batch_size_cap",
        "source_config",
        "generated_config",
        "run_dir",
        "log_path",
        "cuda_visible_devices",
        "attempts",
        "status",
        "returncode",
        "elapsed_seconds",
        "resume_checkpoint",
        "command",
    ]
    for key in keys:
        value = row.get(key)
        if value is not None:
            fields[key] = value
    if row.get("progress_index") and row.get("progress_total"):
        fields["progress_index"] = row["progress_index"]
        fields["progress_total"] = row["progress_total"]
        fields["progress"] = f"{row['progress_index']}/{row['progress_total']}"
    if row.get("elapsed_seconds") is not None:
        fields["elapsed"] = _format_elapsed(row.get("elapsed_seconds"))
    return fields


def _write_results_container_marker(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    marker = path / "INCOMPLETE_RUN.md"
    marker.write_text(
        "# Paper-Ready All-Run Container\n\n"
        "This directory is a container for per-task training run directories generated by "
        "`scripts/run_paper_ready_all.py`. It is not itself a single trainer run.\n\n"
        "Open `reports/paper_ready_all/status.md` for current task state, logs, generated "
        "configs, and result dashboard links.\n",
        encoding="utf-8",
    )


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "created_at": utc_timestamp(), "tasks": {}, "analysis": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer")
    return seeds


def _parse_scale_variants(value: str) -> list[tuple[str, float]]:
    variants: list[tuple[str, float]] = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" in item:
            raw_name, raw_multiplier = item.split(":", 1)
            name = _safe_name(raw_name.strip())
            multiplier = float(raw_multiplier.strip())
        else:
            multiplier = float(item)
            name = "base" if abs(multiplier - 1.0) < 1e-9 else f"scale_{str(multiplier).replace('.', '_')}"
        if multiplier < 1.0:
            raise SystemExit("--scale-variants multipliers must be >= 1.0")
        if not name:
            raise SystemExit("--scale-variants names must be non-empty")
        variants.append((name, multiplier))
    if not variants:
        raise SystemExit("--scale-variants must contain at least one variant")
    names = [name for name, _ in variants]
    if len(set(names)) != len(names):
        raise SystemExit("--scale-variants names must be unique")
    return variants


def _format_scale_variants(variants: list[tuple[str, float]]) -> str:
    return ",".join(f"{name}:{multiplier:g}" for name, multiplier in variants)


def _parse_batch_size_caps(value: str | None) -> dict[str, int]:
    if value is None:
        return {}
    text = str(value).strip()
    if not text or text.lower() in {"none", "off", "false", "0"}:
        return {}
    caps: dict[str, int] = {}
    for raw_item in text.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise SystemExit("--batch-size-caps entries must be name:max_batch, or use none")
        raw_name, raw_cap = item.split(":", 1)
        name = _safe_name(raw_name.strip())
        if not name:
            raise SystemExit("--batch-size-caps names must be non-empty")
        cap = int(raw_cap.strip())
        if cap <= 0:
            raise SystemExit("--batch-size-caps values must be positive")
        caps[name] = cap
    return caps


def _format_batch_size_caps(caps: dict[str, int]) -> str:
    if not caps:
        return "none"
    return ",".join(f"{name}:{cap}" for name, cap in sorted(caps.items()))


def _suite_configs(path: Path) -> list[Path]:
    if not path.exists():
        return []
    data = load_yaml(path)
    return [Path(item) for item in data.get("configs", [])]


def discover_config_paths(
    *,
    include_benchmarks: bool,
    include_ideas: bool,
    extra_configs: list[str],
) -> list[Path]:
    paths: list[Path] = []
    if include_benchmarks:
        for suite_path in DEFAULT_SUITES:
            paths.extend(_suite_configs(suite_path))
        paths.extend(sorted(Path("configs/benchmarks").rglob("bench_*.yaml")))
    if include_ideas:
        for config_path in sorted(Path("ideas/registry").glob(f"{IDEA_FOLDER_GLOB}/config.yaml")):
            idea_path = config_path.parent / "idea.yaml"
            idea = load_yaml(idea_path) if idea_path.exists() else {}
            if (
                idea.get("implementation_status") in TRAINABLE_IDEA_IMPLEMENTATION_STATES
                and idea.get("implementation_kind") == "bespoke_model"
            ):
                paths.append(config_path)
    paths.extend(Path(item) for item in extra_configs)

    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        normalized = path.as_posix()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(path)
    return unique


def _kind_for_config(path: Path) -> str:
    return "idea" if path.as_posix().startswith("ideas/") else "benchmark"


def _task_id(path: Path, seed: int, scale_variant: str) -> str:
    kind = _kind_for_config(path)
    if kind == "idea":
        base = path.parent.name
    else:
        base = path.stem
    if scale_variant != "base":
        base = f"{base}_{scale_variant}"
    return _safe_name(f"{kind}_{base}_seed{seed}")


def _with_seed_run_name(config: dict[str, Any], seed: int, scale_variant: str = "base") -> str:
    run_cfg = config.setdefault("run", {})
    base_name = str(run_cfg.get("name") or config.get("idea_id") or "run")
    scale_suffix = f"_{scale_variant}" if scale_variant != "base" and not base_name.endswith(f"_{scale_variant}") else ""
    suffix = f"_seed{seed}"
    if scale_suffix and base_name.endswith(suffix):
        base_name = base_name[: -len(suffix)]
    if scale_suffix:
        base_name = f"{base_name}{scale_suffix}"
    if not base_name.endswith(suffix):
        base_name = f"{base_name}{suffix}"
    run_cfg["name"] = base_name
    return base_name


def _round_scaled_int(value: int, multiplier: float, *, multiple: int = 1) -> int:
    if multiplier <= 1.0:
        return int(value)
    scaled = max(int(value) + 1, int(round(float(value) * multiplier)))
    if multiple > 1:
        scaled = int(math.ceil(scaled / multiple) * multiple)
    return scaled


def _scale_depth_int(value: int, multiplier: float) -> int:
    if multiplier <= 1.0:
        return int(value)
    return max(int(value) + 1, int(math.ceil(float(value) + 2.0 * (multiplier - 1.0))))


def _scale_hidden_dims(value: Any, multiplier: float) -> Any:
    if multiplier <= 1.0 or not isinstance(value, list):
        return value
    scaled: list[Any] = []
    for item in value:
        if isinstance(item, int) and not isinstance(item, bool):
            scaled.append(_round_scaled_int(item, multiplier, multiple=8))
        else:
            scaled.append(item)
    return scaled


def apply_architecture_scale(
    base_config: dict[str, Any],
    *,
    scale_variant: str,
    scale_multiplier: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = copy.deepcopy(base_config)
    metadata = {
        "variant": scale_variant,
        "multiplier": float(scale_multiplier),
        "scaled_fields": {},
    }
    if scale_multiplier <= 1.0:
        return config, metadata

    model_cfg = config.setdefault("model", {})
    if not isinstance(model_cfg, dict):
        return config, metadata
    scaled_fields: dict[str, dict[str, Any]] = {}
    for key, value in list(model_cfg.items()):
        if isinstance(value, bool):
            continue
        new_value: Any | None = None
        if key == "hidden_dims":
            new_value = _scale_hidden_dims(value, scale_multiplier)
        elif isinstance(value, int):
            if key in WIDTH_SCALE_KEYS:
                new_value = _round_scaled_int(value, scale_multiplier, multiple=8)
            elif key in DEPTH_SCALE_KEYS:
                new_value = _scale_depth_int(value, scale_multiplier)
            elif key in CAPACITY_SCALE_KEYS:
                new_value = _round_scaled_int(value, scale_multiplier)
        if new_value is not None and new_value != value:
            model_cfg[key] = new_value
            scaled_fields[key] = {"from": value, "to": new_value}
    metadata["scaled_fields"] = scaled_fields
    config["architecture_scale"] = metadata
    return config, metadata


def apply_paper_ready_overrides(
    base_config: dict[str, Any],
    *,
    source_path: Path,
    seed: int,
    task_id: str,
    run_dir: Path,
    epochs: int,
    min_epochs: int,
    patience: int,
    scale_variant: str = "base",
    scale_multiplier: float = 1.0,
    batch_size_caps: dict[str, int] | None = None,
    shorten_training: bool = False,
    monitor: str | None = None,
) -> dict[str, Any]:
    config, scale_metadata = apply_architecture_scale(
        base_config,
        scale_variant=scale_variant,
        scale_multiplier=scale_multiplier,
    )
    config["seed"] = int(seed)
    config["deterministic"] = True
    _with_seed_run_name(config, seed, scale_variant=scale_variant)
    run_cfg = config.setdefault("run", {})
    run_cfg["output_dir"] = str(run_dir.parent)
    run_cfg["run_dir"] = str(run_dir)

    training = config.setdefault("training", {})
    if shorten_training:
        # Scout mode: use CLI values exactly, even if YAML asked for more.
        training["epochs"] = int(epochs)
        training["min_epochs"] = int(min_epochs)
        training["min_active_epochs"] = int(min_epochs)
        training["early_stopping_patience"] = int(patience)
        training["reliability_tier"] = "scout"
    else:
        training["epochs"] = max(int(training.get("epochs", 0) or 0), int(epochs))
        training["min_epochs"] = max(int(training.get("min_epochs", 0) or 0), int(min_epochs))
        training["min_active_epochs"] = max(int(training.get("min_active_epochs", 0) or 0), int(min_epochs))
        training["early_stopping_patience"] = max(int(training.get("early_stopping_patience", 0) or 0), int(patience))
        training["reliability_tier"] = "paper_grade"
    if monitor:
        training["monitor"] = str(monitor)
    caps = batch_size_caps or {}
    batch_cap = caps.get(scale_variant)
    if batch_cap is not None:
        current_batch_size = int(training.get("batch_size", batch_cap) or batch_cap)
        if current_batch_size > batch_cap:
            training["batch_size"] = int(batch_cap)
            training["paper_ready_batch_size_cap"] = {
                "variant": scale_variant,
                "from": current_batch_size,
                "to": int(batch_cap),
                "reason": "RTX3070-safe default cap; override with --batch-size-caps none or custom caps.",
            }
    training.setdefault("mixed_precision", True)
    training.setdefault("allow_tf32", True)
    training.setdefault("matmul_precision", "high")
    training.setdefault("gradient_clip_norm", 1.0)
    scheduler = training.get("lr_scheduler")
    if not isinstance(scheduler, dict) or str(scheduler.get("name", "none")).lower() in {"", "none", "off"}:
        training["lr_scheduler"] = {
            "name": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 2,
            "min_lr": 1.0e-5,
        }

    notes = str(config.get("notes") or "")
    paper_note = (
        f"Paper-ready all-run task {task_id}; source config {source_path.as_posix()}; "
        f"seed {seed}; architecture scale {scale_variant} ({scale_multiplier:g}x); "
        "resumable fixed run directory."
    )
    config["notes"] = f"{notes} {paper_note}".strip()
    if scale_variant != "base":
        config["architecture_scale"] = scale_metadata
    return config


def _resume_checkpoint(run_dir: Path) -> Path | None:
    for name in ["checkpoint_last.pt", "checkpoint_best.pt"]:
        path = run_dir / name
        if path.exists():
            return path
    return None


def _validation_errors(messages: list[str]) -> list[str]:
    return [message for message in messages if message.startswith("ERROR:")]


def _complete_run(run_dir: Path) -> tuple[bool, list[str]]:
    if not (run_dir / "metrics_final.json").exists():
        return False, ["ERROR: missing metrics_final.json"]
    messages = validate_run_artifacts(run_dir)
    return not _validation_errors(messages), messages


def _load_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _record_completed_run_measurements(row: dict[str, Any], run_dir: Path) -> None:
    speed_summary = _load_json_file(run_dir / "speed_summary.json")
    if speed_summary is not None:
        row["speed_summary"] = speed_summary
    metadata = _load_json_file(run_dir / "run_metadata.json")
    if metadata is not None:
        row["num_params"] = metadata.get("num_params")
        row["model_name"] = metadata.get("model_name")
    final_metrics = _load_json_file(run_dir / "metrics_final.json")
    if final_metrics is not None:
        row["best_epoch"] = final_metrics.get("best_epoch")
        row["best_score"] = final_metrics.get("best_score")
        speed = final_metrics.get("speed")
        if isinstance(speed, dict) and "speed_summary" not in row:
            row["speed_summary"] = speed


def build_tasks(args: argparse.Namespace, state: dict[str, Any]) -> list[dict[str, Any]]:
    config_paths = discover_config_paths(
        include_benchmarks=args.include_benchmarks,
        include_ideas=args.include_ideas,
        extra_configs=args.config,
    )
    if args.limit is not None:
        config_paths = config_paths[: int(args.limit)]

    tasks: list[dict[str, Any]] = []
    state_tasks = state.setdefault("tasks", {})
    for source_path in config_paths:
        if not source_path.exists():
            task_id = _safe_name(f"missing_{source_path.as_posix()}")
            row = state_tasks.setdefault(task_id, {})
            row.update(
                {
                    "task_id": task_id,
                    "source_config": source_path.as_posix(),
                    "status": "config_missing",
                    "messages": [f"Config does not exist: {source_path}"],
                }
            )
            continue
        base_config = load_yaml(source_path)
        for scale_variant, scale_multiplier in args.scale_variants:
            for seed in args.seeds:
                task_id = _task_id(source_path, seed, scale_variant)
                run_dir = args.results_dir / task_id
                generated_config = args.generated_config_dir / f"{task_id}.yaml"
                config = apply_paper_ready_overrides(
                    base_config,
                    source_path=source_path,
                    seed=seed,
                    task_id=task_id,
                    run_dir=run_dir,
                    epochs=args.epochs,
                    min_epochs=args.min_epochs,
                    patience=args.patience,
                    scale_variant=scale_variant,
                    scale_multiplier=scale_multiplier,
                    batch_size_caps=args.batch_size_caps,
                    shorten_training=getattr(args, "shorten_training", False),
                    monitor=getattr(args, "monitor", None),
                )
                cfg_hash = config_fingerprint(config)
                row = state_tasks.get(task_id, {})
                if row.get("config_hash") and row.get("config_hash") != cfg_hash:
                    row = {
                        "messages": [
                            f"Reset because config hash changed from {row.get('config_hash')} to {cfg_hash}",
                        ]
                    }
                architecture_scale = config.get("architecture_scale") or {
                    "variant": scale_variant,
                    "multiplier": scale_multiplier,
                    "scaled_fields": {},
                }
                row.update(
                    {
                        "task_id": task_id,
                        "kind": _kind_for_config(source_path),
                        "source_config": source_path.as_posix(),
                        "seed": seed,
                        "scale_variant": scale_variant,
                        "scale_multiplier": scale_multiplier,
                        "architecture_scale": architecture_scale,
                        "batch_size": config.get("training", {}).get("batch_size"),
                        "batch_size_cap": config.get("training", {}).get("paper_ready_batch_size_cap"),
                        "run_dir": run_dir.as_posix(),
                        "generated_config": generated_config.as_posix(),
                        "config_hash": cfg_hash,
                        "base_run_name": config.get("run", {}).get("name"),
                    }
                )
                row.setdefault("attempts", 0)
                row.setdefault("status", "pending")
                row.setdefault("messages", [])
                state_tasks[task_id] = row
                tasks.append({"state": row, "config": config, "source_path": source_path})

    # Order: base scale first (cheapest, fits in VRAM), then scale_up, then scale_xl
    # last (largest models — these are the ones likely to need slow CPU fallback).
    _scale_priority = {"base": 0, "scale_up": 1, "scale_xl": 2}
    tasks.sort(
        key=lambda t: (
            _scale_priority.get(t["state"].get("scale_variant"), 99),
            t["source_path"].as_posix(),
            t["state"].get("seed", 0),
        )
    )
    return tasks


def _write_generated_config(task: dict[str, Any], *, resume: bool) -> None:
    config = copy.deepcopy(task["config"])
    row = task["state"]
    run_dir = Path(row["run_dir"])
    checkpoint = _resume_checkpoint(run_dir)
    training = config.setdefault("training", {})
    training.pop("resume_from", None)
    training.pop("resume_run_dir", None)
    if resume and checkpoint is not None:
        training["resume_run_dir"] = str(run_dir)
        training["resume_from"] = str(checkpoint)
    save_yaml(config, row["generated_config"])


def refresh_task_statuses(tasks: list[dict[str, Any]], state_path: Path, state: dict[str, Any]) -> None:
    statuses_with_existing_attempts = {
        "running",
        "completed",
        "failed",
        "failed_resume_available",
        "timeout",
        "timeout_resume_available",
        "artifact_validation_failed",
        "interrupted_resume_available",
        "interrupted_no_checkpoint",
    }
    for task in tasks:
        row = task["state"]
        run_dir = Path(row["run_dir"])
        status = str(row.get("status") or "pending")
        checkpoint = _resume_checkpoint(run_dir)
        has_metrics = (run_dir / "metrics_final.json").exists()
        should_validate = has_metrics or status in statuses_with_existing_attempts
        complete = False
        if should_validate:
            complete, messages = _complete_run(run_dir)
            row["artifact_validation"] = messages
            if complete:
                row["status"] = "completed"
                row["finished_at"] = row.get("finished_at") or utc_timestamp()
                _record_completed_run_measurements(row, run_dir)
            elif status == "running":
                row["status"] = "interrupted_resume_available" if checkpoint else "interrupted_no_checkpoint"
            elif status == "completed":
                row["status"] = "artifact_validation_failed"
        elif checkpoint is not None:
            row["status"] = "interrupted_resume_available"
            row.pop("artifact_validation", None)
        else:
            row.pop("artifact_validation", None)
        _write_generated_config(task, resume=(not complete and bool(_resume_checkpoint(run_dir))))
    _atomic_write_json(state, state_path)


def _visible_cuda_ids(value: str | None) -> list[str]:
    if value:
        return [item.strip() for item in value.split(",") if item.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [item.strip() for item in visible.split(",") if item.strip()]
    try:
        import torch

        return [str(index) for index in range(int(torch.cuda.device_count()))]
    except Exception:
        return []


def _default_jobs(requested: int | None, gpu_ids: list[str]) -> int:
    if requested is not None:
        return max(1, int(requested))
    return max(1, len(gpu_ids))


def _finish_task(
    task: dict[str, Any],
    process: subprocess.Popen[str],
    log_handle: Any,
    started: float,
) -> bool:
    row = task["state"]
    log_handle.close()
    row["returncode"] = process.returncode
    row["elapsed_seconds"] = time.monotonic() - started
    if process.returncode != 0:
        row["status"] = "failed_resume_available" if _resume_checkpoint(Path(row["run_dir"])) else "failed"
        row.setdefault("messages", []).append(f"Training command failed with return code {process.returncode}")
        return True
    complete, messages = _complete_run(Path(row["run_dir"]))
    row["artifact_validation"] = messages
    if complete:
        row["status"] = "completed"
        row["finished_at"] = utc_timestamp()
        _record_completed_run_measurements(row, Path(row["run_dir"]))
        return False
    row["status"] = "artifact_validation_failed"
    row.setdefault("messages", []).extend(_validation_errors(messages))
    return True


def _timeout_task(task: dict[str, Any], process: subprocess.Popen[str], log_handle: Any, started: float) -> None:
    row = task["state"]
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)
    log_handle.close()
    row["status"] = "timeout_resume_available" if _resume_checkpoint(Path(row["run_dir"])) else "timeout"
    row["returncode"] = None
    row["elapsed_seconds"] = time.monotonic() - started
    row.setdefault("messages", []).append(f"Timed out after {row.get('timeout_minutes')} minutes")


def run_pending_tasks(
    tasks: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    state: dict[str, Any],
    state_path: Path,
) -> bool:
    runnable_statuses = {
        "pending",
        "interrupted_resume_available",
        "interrupted_no_checkpoint",
        "failed_resume_available",
        "failed",
        "timeout_resume_available",
        "timeout",
        "artifact_validation_failed",
        "dry_run_pending",
    }
    pending = [task for task in tasks if task["state"].get("status") in runnable_statuses]
    if args.dry_run:
        append_event(
            args,
            state,
            "dry_run_started",
            pending_tasks=len(pending),
            total_tasks=len(tasks),
            results_dir=args.results_dir.as_posix(),
            report_dir=args.report_dir.as_posix(),
            logs_dir=args.logs_dir.as_posix(),
            generated_config_dir=args.generated_config_dir.as_posix(),
        )
        for index, task in enumerate(pending, start=1):
            row = task["state"]
            row["status"] = "dry_run_pending"
            row["progress_index"] = index
            row["progress_total"] = len(pending)
            print(
                f"[dry-run {index}/{len(pending)}] {row['task_id']} | "
                f"batch={row.get('batch_size')} | config={row['generated_config']} | run={row['run_dir']}",
                flush=True,
            )
            append_event(args, state, "dry_run_task", **_task_event_fields(row))
        _atomic_write_json(state, state_path)
        append_event(args, state, "dry_run_finished", pending_tasks=len(pending), total_tasks=len(tasks))
        return False

    gpu_ids = _visible_cuda_ids(args.gpu_ids)
    max_jobs = _default_jobs(args.jobs, gpu_ids)
    eta = eta_snapshot(pending, max_jobs)
    append_event(
        args,
        state,
        "runner_started",
        pending_tasks=len(pending),
        total_tasks=len(tasks),
        jobs=max_jobs,
        gpu_ids=gpu_ids,
        eta=eta["eta"],
        eta_seconds=eta["eta_seconds"],
        average_task_elapsed=eta["average_task_elapsed"],
        processed_tasks=eta["processed_tasks"],
        cwd=Path.cwd().as_posix(),
        state_path=state_path.as_posix(),
        results_dir=args.results_dir.as_posix(),
        report_dir=args.report_dir.as_posix(),
        logs_dir=args.logs_dir.as_posix(),
        generated_config_dir=args.generated_config_dir.as_posix(),
    )
    print(
        f"Running {len(pending)} pending tasks with jobs={max_jobs}, gpu_ids={gpu_ids or ['cpu/no-cuda-visible']}",
        flush=True,
    )
    print(
        f"ETA: {eta['eta']} from {eta['processed_tasks']} completed/attempted task timing(s); "
        f"avg={eta['average_task_elapsed']}",
        flush=True,
    )
    print(f"Event log: {args.event_log}", flush=True)
    print(f"Timeline: {args.timeline}", flush=True)
    active: list[dict[str, Any]] = []
    next_idx = 0
    next_gpu = 0
    failed = False

    def launch(task: dict[str, Any], progress_index: int) -> None:
        nonlocal next_gpu
        row = task["state"]
        resume_checkpoint = _resume_checkpoint(Path(row["run_dir"]))
        resume = resume_checkpoint is not None
        _write_generated_config(task, resume=resume)
        row["attempts"] = int(row.get("attempts", 0)) + 1
        row["status"] = "running"
        row["started_at"] = utc_timestamp()
        row["timeout_minutes"] = args.timeout_minutes
        row["messages"] = list(row.get("messages", []))
        row["progress_index"] = progress_index
        row["progress_total"] = len(pending)
        log_path = args.logs_dir / f"{row['task_id']}_attempt{row['attempts']}.log"
        row["log_path"] = log_path.as_posix()
        if resume_checkpoint:
            row["resume_checkpoint"] = resume_checkpoint.as_posix()
        else:
            row.pop("resume_checkpoint", None)
        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if gpu_ids:
            assigned_gpu = gpu_ids[next_gpu % len(gpu_ids)]
            next_gpu += 1
            env["CUDA_VISIBLE_DEVICES"] = assigned_gpu
            row["cuda_visible_devices"] = assigned_gpu
        command = [sys.executable, "scripts/train_model.py", "--config", row["generated_config"]]
        row["command"] = command
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("w", encoding="utf-8")
        log.write(f"started_at={row['started_at']}\n")
        log.write(f"cwd={Path.cwd().as_posix()}\n")
        log.write(f"task_id={row['task_id']}\n")
        log.write(f"progress={progress_index}/{len(pending)}\n")
        log.write(f"kind={row.get('kind')}\n")
        log.write(f"seed={row.get('seed')}\n")
        log.write(f"scale_variant={row.get('scale_variant')}\n")
        log.write(f"scale_multiplier={row.get('scale_multiplier')}\n")
        log.write(f"batch_size={row.get('batch_size')}\n")
        if row.get("batch_size_cap"):
            log.write(f"batch_size_cap={json.dumps(row.get('batch_size_cap'), sort_keys=True)}\n")
        log.write(f"source_config={row.get('source_config')}\n")
        log.write(f"generated_config={row.get('generated_config')}\n")
        log.write("$ " + " ".join(command) + "\n")
        log.write(f"run_dir={row['run_dir']}\n")
        log.write(f"state_path={state_path.as_posix()}\n")
        log.write(f"event_log={Path(args.event_log).as_posix()}\n")
        log.write(f"timeline={Path(args.timeline).as_posix()}\n")
        if row.get("cuda_visible_devices") is not None:
            log.write(f"CUDA_VISIBLE_DEVICES={row['cuda_visible_devices']}\n")
        log.write(f"resume={str(resume).lower()}\n")
        if resume_checkpoint:
            log.write(f"resume_checkpoint={resume_checkpoint.as_posix()}\n")
        log.write("\n")
        log.flush()
        _atomic_write_json(state, state_path)
        append_event(args, state, "task_started", **_task_event_fields(row))
        print(
            f"[start {progress_index}/{len(pending)}] {row['task_id']} | "
            f"gpu={row.get('cuda_visible_devices', 'cpu/no-cuda-visible')} | "
            f"scale={row.get('scale_variant', 'base')} | seed={row.get('seed')} | "
            f"batch={row.get('batch_size')} | "
            f"log={log_path} | run={row['run_dir']}",
            flush=True,
        )
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
        started = time.monotonic()
        active.append(
            {
                "task": task,
                "process": process,
                "log": log,
                "started": started,
                "deadline": started + args.timeout_minutes * 60 if args.timeout_minutes else None,
            }
        )

    while next_idx < len(pending) or active:
        while next_idx < len(pending) and len(active) < max_jobs and (args.continue_on_error or not failed):
            launch(pending[next_idx], next_idx + 1)
            next_idx += 1

        for item in list(active):
            process = item["process"]
            deadline = item["deadline"]
            if deadline is not None and time.monotonic() >= deadline and process.poll() is None:
                _timeout_task(item["task"], process, item["log"], item["started"])
                row = item["task"]["state"]
                eta = eta_snapshot(pending, max_jobs)
                append_event(args, state, "task_timeout", **_task_event_fields(row), **eta)
                print(
                    f"[timeout {row.get('progress_index')}/{row.get('progress_total')}] {row['task_id']} | "
                    f"status={row.get('status')} | elapsed={_format_elapsed(row.get('elapsed_seconds'))} | "
                    f"eta={eta['eta']} | avg={eta['average_task_elapsed']} | "
                    f"log={row.get('log_path')} | run={row.get('run_dir')}",
                    flush=True,
                )
                active.remove(item)
                failed = True
                _atomic_write_json(state, state_path)
                continue
            if process.poll() is not None:
                task_failed = _finish_task(item["task"], process, item["log"], item["started"])
                row = item["task"]["state"]
                eta = eta_snapshot(pending, max_jobs)
                append_event(args, state, "task_finished", **_task_event_fields(row), **eta)
                print(
                    f"[finish {row.get('progress_index')}/{row.get('progress_total')}] {row['task_id']} | "
                    f"status={row.get('status')} | rc={row.get('returncode')} | "
                    f"elapsed={_format_elapsed(row.get('elapsed_seconds'))} | "
                    f"eta={eta['eta']} | avg={eta['average_task_elapsed']} | "
                    f"log={row.get('log_path')} | run={row.get('run_dir')}",
                    flush=True,
                )
                failed = task_failed or failed
                active.remove(item)
                _atomic_write_json(state, state_path)

        if active:
            time.sleep(1.0)

    if failed and not args.continue_on_error:
        for task in pending[next_idx:]:
            row = task["state"]
            row["status"] = "not_started_after_failure"
            row.setdefault("messages", []).append("Skipped because a previous task failed")
            append_event(args, state, "task_skipped_after_failure", **_task_event_fields(row))
        _atomic_write_json(state, state_path)
    eta = eta_snapshot(pending, max_jobs)
    append_event(args, state, "runner_finished", failed=failed, pending_tasks=len(pending), total_tasks=len(tasks), **eta)
    return failed


def write_plan_report(tasks: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Paper-Ready All-Run Plan",
        "",
        f"- Tasks: `{len(tasks)}`",
        f"- Idea tasks: `{sum(1 for task in tasks if task['state'].get('kind') == 'idea')}`",
        f"- Benchmark tasks: `{sum(1 for task in tasks if task['state'].get('kind') == 'benchmark')}`",
        "",
        "| Task | Kind | Scale | Seed | Source | Run Dir | Status |",
        "|---|---|---|---:|---|---|---|",
    ]
    for task in tasks:
        row = task["state"]
        lines.append(
            f"| `{row['task_id']}` | `{row.get('kind')}` | `{row.get('scale_variant', 'base')}` | `{row.get('seed')}` | "
            f"`{row.get('source_config')}` | `{row.get('run_dir')}` | `{row.get('status')}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _task_status_counts(tasks: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(task["state"].get("status") or "unknown") for task in tasks)


def _task_kind_counts(tasks: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(task["state"].get("kind") or "unknown") for task in tasks)


def _task_scale_counts(tasks: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(task["state"].get("scale_variant") or "base") for task in tasks)


def _attention_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok_statuses = {"completed", "dry_run_pending"}
    return [task for task in tasks if str(task["state"].get("status") or "") not in ok_statuses]


def _next_tasks(tasks: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    skipped = {"completed"}
    return [task for task in tasks if str(task["state"].get("status") or "") not in skipped][:limit]


def _rel(path: str | Path) -> str:
    return Path(path).as_posix()


def _default_path(path: str) -> Path:
    return Path(path)


def _resume_command(args: argparse.Namespace) -> str:
    command = ["PYTHONDONTWRITEBYTECODE=1", "python", "scripts/run_paper_ready_all.py"]
    command.extend(str(item) for item in args.config)
    defaults = {
        "--results-dir": _default_path("results/paper_ready_all"),
        "--report-dir": _default_path("reports/paper_ready_all"),
        "--state-path": _default_path("reports/paper_ready_all/state.json"),
        "--logs-dir": _default_path("reports/paper_ready_all/logs"),
        "--generated-config-dir": _default_path("reports/paper_ready_all/generated_configs"),
    }
    for flag, default in defaults.items():
        current = getattr(args, flag.lstrip("-").replace("-", "_"))
        if Path(current) != default:
            command.extend([flag, str(current)])
    if Path(args.event_log) != args.report_dir / "events.jsonl":
        command.extend(["--event-log", str(args.event_log)])
    if Path(args.timeline) != args.report_dir / "timeline.md":
        command.extend(["--timeline", str(args.timeline)])
    command.extend(["--seeds", ",".join(str(seed) for seed in args.seeds)])
    command.extend(["--scale-variants", args.scale_variants_text])
    command.extend(["--batch-size-caps", args.batch_size_caps_text])
    command.extend(["--epochs", str(args.epochs)])
    command.extend(["--min-epochs", str(args.min_epochs)])
    command.extend(["--patience", str(args.patience)])
    if args.jobs is not None:
        command.extend(["--jobs", str(args.jobs)])
    if args.gpu_ids:
        command.extend(["--gpu-ids", str(args.gpu_ids)])
    if args.timeout_minutes is not None:
        command.extend(["--timeout-minutes", str(args.timeout_minutes)])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if not args.include_benchmarks:
        command.append("--no-benchmarks")
    if not args.include_ideas:
        command.append("--no-ideas")
    if not args.continue_on_error:
        command.append("--stop-on-error")
    if args.no_analysis:
        command.append("--no-analysis")
    return " ".join(shlex.quote(part) for part in command)


def write_status_report(tasks: list[dict[str, Any]], args: argparse.Namespace, state: dict[str, Any], path: Path) -> None:
    status_counts = _task_status_counts(tasks)
    kind_counts = _task_kind_counts(tasks)
    scale_counts = _task_scale_counts(tasks)
    completed = status_counts.get("completed", 0)
    total = len(tasks)
    attention = _attention_tasks(tasks)
    defaults = state.get("paper_ready_defaults", {})
    analysis = state.get("analysis", {})
    if args.jobs is not None:
        eta_jobs = max(1, int(args.jobs))
    elif args.gpu_ids:
        eta_jobs = max(1, len([item for item in str(args.gpu_ids).split(",") if item.strip()]))
    else:
        eta_jobs = 1
    eta = eta_snapshot(tasks, eta_jobs)

    lines = [
        "# Paper-Ready Training Status",
        "",
        f"- Total tasks: `{total}`",
        f"- Completed tasks: `{completed}`",
        f"- Remaining tasks: `{max(0, total - completed)}`",
        f"- ETA: `{eta['eta']}`",
        f"- Average observed task time: `{eta['average_task_elapsed']}`",
        f"- ETA basis: `{eta['processed_tasks']}` observed task(s), `{eta['remaining_estimate_tasks']}` remaining task(s), `{eta['eta_jobs']}` job(s)",
        f"- Dry run: `{bool(args.dry_run)}`",
        f"- Results directory: `{_rel(args.results_dir)}`",
        f"- Report directory: `{_rel(args.report_dir)}`",
        f"- Resume state: `{_rel(args.state_path)}`",
        "",
        "## Defaults",
        "",
        f"- Seeds: `{', '.join(str(seed) for seed in defaults.get('seeds', []))}`",
        f"- Architecture scales: `{defaults.get('scale_variants')}`",
        f"- Batch-size caps: `{defaults.get('batch_size_caps')}`",
        f"- Epoch budget: `{defaults.get('epochs')}`",
        f"- Minimum epochs: `{defaults.get('min_epochs')}`",
        f"- Early-stopping patience: `{defaults.get('patience')}`",
        "",
        "## Counts",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"| `{status}` | {count} |")
    lines.extend(["", "| Kind | Count |", "|---|---:|"])
    for kind, count in sorted(kind_counts.items()):
        lines.append(f"| `{kind}` | {count} |")
    lines.extend(["", "| Architecture Scale | Count |", "|---|---:|"])
    for scale, count in sorted(scale_counts.items()):
        lines.append(f"| `{scale}` | {count} |")

    lines.extend(
        [
            "",
            "## Open First",
            "",
            f"- Plan: `{_rel(args.report_dir / 'plan.md')}`",
            f"- Status: `{_rel(path)}`",
            f"- State JSON: `{_rel(args.state_path)}`",
            f"- Event log JSONL: `{_rel(args.event_log)}`",
            f"- Timeline: `{_rel(args.timeline)}`",
            f"- Logs: `{_rel(args.logs_dir)}`",
            f"- Generated configs: `{_rel(args.generated_config_dir)}`",
            f"- Leaderboard: `{_rel(args.results_dir / 'leaderboard.md')}`",
            f"- Seed summary: `{_rel(args.results_dir / 'leaderboard_seed_summary.md')}`",
            f"- Training dashboard: `{_rel(args.report_dir / 'training' / 'training_dashboard.md')}`",
            f"- Training dashboard HTML: `{_rel(args.report_dir / 'training' / 'training_dashboard.html')}`",
            f"- Paper PDF report: `{_rel(args.report_dir / 'paper_report.pdf')}`",
            "",
        ]
    )

    if analysis:
        lines.extend(["## Analysis Jobs", "", "| Job | Return Code | Log |", "|---|---:|---|"])
        for name, row in sorted(analysis.items()):
            lines.append(f"| `{name}` | `{row.get('returncode')}` | `{row.get('log_path')}` |")
        lines.append("")

    completed_speed_rows = [
        task for task in tasks if task["state"].get("status") == "completed" and task["state"].get("speed_summary")
    ]
    if completed_speed_rows:
        lines.extend(
            [
                "## Speed Snapshot",
                "",
                "| Task | Scale | Params | Train Samples/s | Val Samples/s | Total Seconds |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for task in completed_speed_rows[:30]:
            row = task["state"]
            speed = row.get("speed_summary") or {}
            lines.append(
                f"| `{row.get('task_id')}` | `{row.get('scale_variant', 'base')}` | "
                f"{row.get('num_params') or '-'} | "
                f"{float(speed.get('train_samples_per_second') or 0):.1f} | "
                f"{float(speed.get('val_samples_per_second') or 0):.1f} | "
                f"{float(speed.get('fit_elapsed_seconds') or row.get('elapsed_seconds') or 0):.1f} |"
            )
        if len(completed_speed_rows) > 30:
            lines.append(f"| ... | ... | ... | ... | ... | `{len(completed_speed_rows) - 30}` more completed runs |")
        lines.append("")

    next_rows = _next_tasks(tasks)
    lines.extend(
        [
            "## Next Tasks",
            "",
            "| Task | Kind | Scale | Seed | Status | Source | Run Dir |",
            "|---|---|---|---:|---|---|---|",
        ]
    )
    if next_rows:
        for task in next_rows:
            row = task["state"]
            lines.append(
                f"| `{row.get('task_id')}` | `{row.get('kind')}` | `{row.get('scale_variant', 'base')}` | "
                f"`{row.get('seed')}` | "
                f"`{row.get('status')}` | `{row.get('source_config')}` | `{row.get('run_dir')}` |"
            )
    else:
        lines.append("| none |  |  |  |  |  |  |")
    lines.append("")

    if attention:
        lines.extend(["## Needs Attention", "", "| Task | Status | Log | Messages |", "|---|---|---|---|"])
        for task in attention[:50]:
            row = task["state"]
            messages = "; ".join(str(message) for message in row.get("messages", [])[-3:])
            lines.append(
                f"| `{row.get('task_id')}` | `{row.get('status')}` | "
                f"`{row.get('log_path') or ''}` | {messages or '-'} |"
            )
        if len(attention) > 50:
            lines.append(f"| ... | ... | ... | `{len(attention) - 50}` more tasks need attention or are pending. |")
        lines.append("")

    lines.extend(
        [
            "## Resume Command",
            "",
            "Rerun the same command after an interruption. Completed tasks stay completed, and unfinished tasks use the same fixed run directories.",
            "",
            "```bash",
            _resume_command(args),
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(args: argparse.Namespace, state: dict[str, Any], state_path: Path) -> bool:
    if args.dry_run or args.no_analysis:
        return False
    failed = False
    commands = [
        [sys.executable, "scripts/compare_results.py", "--results-dir", str(args.results_dir)],
        [
            sys.executable,
            "scripts/reports/plot_training_results.py",
            "--results-dir",
            str(args.results_dir),
            "--output-dir",
            str(args.report_dir / "training"),
            "--max-runs",
            "1000000",
        ],
        [
            sys.executable,
            "scripts/reports/build_paper_report.py",
            "--results-dir",
            str(args.results_dir),
            "--state-path",
            str(args.state_path),
            "--generated-config-dir",
            str(args.generated_config_dir),
            "--training-report-dir",
            str(args.report_dir / "training"),
            "--output",
            str(args.report_dir / "paper_report.pdf"),
        ],
    ]
    analysis = state.setdefault("analysis", {})
    for command in commands:
        name = Path(command[1]).stem
        log_path = args.logs_dir / f"analysis_{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        append_event(args, state, "analysis_started", name=name, command=command, log_path=log_path.as_posix())
        print(f"[analysis start] {name} | log={log_path}", flush=True)
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log_path.write_text(result.stdout, encoding="utf-8")
        analysis[name] = {
            "command": command,
            "log_path": log_path.as_posix(),
            "returncode": result.returncode,
            "finished_at": utc_timestamp(),
        }
        append_event(
            args,
            state,
            "analysis_finished",
            name=name,
            command=command,
            log_path=log_path.as_posix(),
            returncode=result.returncode,
        )
        print(f"[analysis finish] {name} | rc={result.returncode} | log={log_path}", flush=True)
        if result.returncode != 0:
            failed = True
    _atomic_write_json(state, state_path)
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run every registered idea and benchmark config at paper-ready depth with resumable state."
    )
    parser.add_argument("config", nargs="*", help="Extra config YAML files to include in addition to selected defaults.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/paper_ready_all"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/paper_ready_all"))
    parser.add_argument("--state-path", type=Path, default=Path("reports/paper_ready_all/state.json"))
    parser.add_argument("--logs-dir", type=Path, default=Path("reports/paper_ready_all/logs"))
    parser.add_argument(
        "--generated-config-dir",
        type=Path,
        default=Path("reports/paper_ready_all/generated_configs"),
    )
    parser.add_argument("--event-log", type=Path, default=None)
    parser.add_argument("--timeline", type=Path, default=None)
    parser.add_argument("--seeds", type=_parse_seeds, default=_parse_seeds("42,43,44"))
    parser.add_argument(
        "--scale-variants",
        default=DEFAULT_SCALE_VARIANTS_TEXT,
        help=(
            "Comma-separated architecture scale variants as name:multiplier pairs. "
            "Default runs base plus two larger full sweeps."
        ),
    )
    parser.add_argument(
        "--batch-size-caps",
        default=DEFAULT_BATCH_SIZE_CAPS_TEXT,
        help=(
            "Comma-separated scale:max_batch caps applied after scaling. Defaults are conservative for an 8GB RTX 3070. "
            "Use 'none' to preserve config batch sizes."
        ),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--min-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--shorten-training", action="store_true",
                        help="Use --epochs / --min-epochs / --patience exactly (overriding YAML if larger). For scout runs.")
    parser.add_argument("--monitor", default=None,
                        help="Set training.monitor in every generated config (e.g. pr_auc).")
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--gpu-ids", default=None)
    parser.add_argument("--timeout-minutes", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Plan only the first N source configs before seed expansion.")
    parser.add_argument("--no-benchmarks", dest="include_benchmarks", action="store_false")
    parser.add_argument("--no-ideas", dest="include_ideas", action="store_false")
    parser.set_defaults(include_benchmarks=True, include_ideas=True)
    parser.add_argument("--stop-on-error", dest="continue_on_error", action="store_false")
    parser.set_defaults(continue_on_error=True)
    parser.add_argument("--no-analysis", action="store_true")
    args = parser.parse_args()
    if args.event_log is None:
        args.event_log = args.report_dir / "events.jsonl"
    if args.timeline is None:
        args.timeline = args.report_dir / "timeline.md"
    args.scale_variants = _parse_scale_variants(args.scale_variants)
    args.scale_variants_text = _format_scale_variants(args.scale_variants)
    args.batch_size_caps = _parse_batch_size_caps(args.batch_size_caps)
    args.batch_size_caps_text = _format_batch_size_caps(args.batch_size_caps)

    _write_results_container_marker(args.results_dir)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    args.generated_config_dir.mkdir(parents=True, exist_ok=True)

    state = _load_state(args.state_path)
    state["last_started_at"] = utc_timestamp()
    state["paper_ready_defaults"] = {
        "seeds": args.seeds,
        "scale_variants": args.scale_variants_text,
        "batch_size_caps": args.batch_size_caps_text,
        "epochs": args.epochs,
        "min_epochs": args.min_epochs,
        "patience": args.patience,
    }
    tasks = build_tasks(args, state)
    refresh_task_statuses(tasks, args.state_path, state)
    write_plan_report(tasks, args.report_dir / "plan.md")
    write_status_report(tasks, args, state, args.report_dir / "status.md")
    print(f"Planned {len(tasks)} tasks")
    print(f"State: {args.state_path}")
    print(f"Plan: {args.report_dir / 'plan.md'}")
    print(f"Status: {args.report_dir / 'status.md'}")
    print(f"Event log: {args.event_log}")
    print(f"Timeline: {args.timeline}")
    append_event(
        args,
        state,
        "runner_planned",
        total_tasks=len(tasks),
        state_path=args.state_path.as_posix(),
        plan_path=(args.report_dir / "plan.md").as_posix(),
        status_path=(args.report_dir / "status.md").as_posix(),
        results_dir=args.results_dir.as_posix(),
        logs_dir=args.logs_dir.as_posix(),
        generated_config_dir=args.generated_config_dir.as_posix(),
        command=sys.argv,
        cwd=Path.cwd().as_posix(),
    )

    validation_failed = False
    append_event(args, state, "validation_started", total_tasks=len(tasks), dry_run=bool(args.dry_run))
    for task in tasks:
        row = task["state"]
        generated_config_path = Path(row["generated_config"])
        validation = validate_training_config(
            load_yaml(generated_config_path),
            generated_config_path,
            require_device_available=not args.dry_run,
        )
        if row.get("kind") == "idea":
            idea_report = validate_idea_for_training(
                Path(row["source_config"]).parent,
                config_path=generated_config_path,
                require_device_available=not args.dry_run,
            )
            for issue in idea_report["issues"]:
                message = issue if issue.startswith("ERROR:") else f"ERROR: idea guard: {issue}"
                if message not in validation:
                    validation.append(message)
        row["config_validation"] = validation
        if _validation_errors(validation):
            row["status"] = "validation_failed"
            row.setdefault("messages", []).extend(_validation_errors(validation))
            validation_failed = True
    _atomic_write_json(state, args.state_path)
    write_status_report(tasks, args, state, args.report_dir / "status.md")
    append_event(
        args,
        state,
        "validation_finished",
        total_tasks=len(tasks),
        validation_failed=validation_failed,
        validation_failed_tasks=sum(1 for task in tasks if task["state"].get("status") == "validation_failed"),
    )
    if validation_failed and not args.continue_on_error:
        append_event(args, state, "runner_finished", failed=True, reason="validation_failed")
        raise SystemExit(1)

    failed = run_pending_tasks(tasks, args=args, state=state, state_path=args.state_path)
    analysis_failed = run_analysis(args, state, args.state_path)
    write_plan_report(tasks, args.report_dir / "plan.md")
    write_status_report(tasks, args, state, args.report_dir / "status.md")
    append_event(
        args,
        state,
        "runner_exiting",
        failed=failed,
        analysis_failed=analysis_failed,
        validation_failed=validation_failed,
    )
    if failed or analysis_failed or validation_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

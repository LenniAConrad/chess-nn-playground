from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from chess_nn_playground.data.board_features import SIMPLE_18
from chess_nn_playground.data.dataset import BINARY_MODES
from chess_nn_playground.utils.paths import ensure_dir, utc_timestamp


@dataclass(frozen=True)
class RunPaths:
    run_name: str
    run_dir: Path
    train_path: Path
    val_path: Path
    test_path: Path
    input_encoding: str


@dataclass(frozen=True)
class TrainingRuntimeConfig:
    batch_size: int
    num_workers: int
    persistent_workers: bool
    prefetch_factor: int
    epochs: int
    min_epochs: int
    min_active_epochs: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float | None
    use_amp: bool
    pin_memory: bool
    allow_tf32: bool
    matmul_precision: str
    monitor_metric: str
    scheduler_cfg: Any


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "run"


def config_fingerprint(config: dict[str, Any]) -> str:
    normalized = json.loads(json.dumps(config, sort_keys=True, default=str))
    run_cfg = normalized.get("run")
    if isinstance(run_cfg, dict):
        run_cfg.pop("run_dir", None)
    training_cfg = normalized.get("training")
    if isinstance(training_cfg, dict):
        training_cfg.pop("resume_from", None)
        training_cfg.pop("resume_run_dir", None)
    payload = json.dumps(normalized, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def sync_device_for_timing(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def resolve_num_workers(value: Any, device: torch.device) -> int:
    text = str(value).strip().lower() if value is not None else "auto"
    if text in {"", "auto"}:
        if device.type != "cuda":
            return 0
        cpu_count = os.cpu_count() or 2
        return max(1, min(8, max(1, cpu_count // 2)))
    workers = int(value)
    if workers < 0:
        raise ValueError("training.num_workers must be non-negative or auto")
    return workers


def resolve_run_paths(config: dict[str, Any]) -> RunPaths:
    run_cfg = config.get("run", {}) or {}
    data_cfg = config.get("data", {}) or {}
    run_name = safe_name(run_cfg.get("name", "cnn_baseline"))
    fixed_run_dir = run_cfg.get("run_dir")
    if fixed_run_dir:
        run_dir = Path(fixed_run_dir)
    else:
        run_dir = Path(run_cfg.get("output_dir", "results")) / f"{utc_timestamp(compact=True)}_{run_name}"
    ensure_dir(run_dir)
    return RunPaths(
        run_name=run_name,
        run_dir=run_dir,
        train_path=Path(data_cfg.get("train_path", "data/splits/split_train.parquet")),
        val_path=Path(data_cfg.get("val_path", "data/splits/split_val.parquet")),
        test_path=Path(data_cfg.get("test_path", "data/splits/split_test.parquet")),
        input_encoding=str(data_cfg.get("encoding", SIMPLE_18)),
    )


def resolve_training_runtime(
    config: dict[str, Any],
    *,
    device: torch.device,
    mode: str,
) -> TrainingRuntimeConfig:
    training_cfg = config.get("training", {}) or {}
    batch_size = int(training_cfg.get("batch_size", 64))
    num_workers = resolve_num_workers(training_cfg.get("num_workers", "auto"), device)
    persistent_workers = bool(training_cfg.get("persistent_workers", num_workers > 0)) and num_workers > 0
    prefetch_factor = int(training_cfg.get("prefetch_factor", 2))
    epochs = int(training_cfg.get("epochs", 10))
    min_epochs = int(training_cfg.get("min_epochs", 0))
    min_active_epochs = int(training_cfg.get("min_active_epochs", min_epochs))
    learning_rate = float(training_cfg.get("learning_rate", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    clip_value = training_cfg.get("gradient_clip_norm")
    gradient_clip_norm = float(clip_value) if clip_value is not None else None
    mixed_precision_cfg = training_cfg.get("mixed_precision", False)
    if str(mixed_precision_cfg).strip().lower() == "auto":
        use_amp = device.type == "cuda"
    else:
        use_amp = bool(mixed_precision_cfg) and device.type == "cuda"
    pin_memory = bool(training_cfg.get("pin_memory", device.type == "cuda"))
    allow_tf32 = bool(training_cfg.get("allow_tf32", device.type == "cuda"))
    matmul_precision = str(training_cfg.get("matmul_precision", "high" if device.type == "cuda" else "highest"))

    configured_monitor = training_cfg.get("monitor")
    if configured_monitor is None:
        monitor_metric = "pr_auc" if mode in BINARY_MODES else "macro_f1"
    else:
        monitor_metric = str(configured_monitor).strip().lower()

    return TrainingRuntimeConfig(
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        epochs=epochs,
        min_epochs=min_epochs,
        min_active_epochs=min_active_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_clip_norm=gradient_clip_norm,
        use_amp=use_amp,
        pin_memory=pin_memory,
        allow_tf32=allow_tf32,
        matmul_precision=matmul_precision,
        monitor_metric=monitor_metric,
        scheduler_cfg=training_cfg.get("lr_scheduler", {}),
    )


def configure_torch_precision(
    *,
    device: torch.device,
    allow_tf32: bool,
    matmul_precision: str,
) -> None:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(matmul_precision)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

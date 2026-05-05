from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

import torch

from chess_nn_playground.data.board_features import available_encodings, encoding_num_planes
from chess_nn_playground.data.dataset import PUZZLE_BINARY
from chess_nn_playground.models.registry import available_models
from chess_nn_playground.training.device import validate_configured_device


REQUIRED_TOP_LEVEL = {"run", "mode", "data", "model", "training"}
VALID_MODES = {"coarse_binary", PUZZLE_BINARY, "fine_3class"}
CUDA_DEVICE_NAMES = {"gpu", "nvidia", "nvidia_gpu", "cuda_required", "cuda"}


class ParquetSplitInfo(NamedTuple):
    rows: int | None
    columns: frozenset[str]
    error: str | None


def expected_num_classes(mode: str) -> int:
    if mode == PUZZLE_BINARY:
        return 1
    if mode == "coarse_binary":
        return 2
    if mode == "fine_3class":
        return 3
    raise ValueError(f"Unsupported mode: {mode}")


def _is_auto(value: Any) -> bool:
    return value is None or str(value).strip().lower() in {"", "auto"}


def _int_or_error(value: Any, field: str, path_text: str, errors: list[str]) -> int | None:
    try:
        return int(value)
    except Exception:
        errors.append(f"{path_text}: {field} must be an integer")
        return None


def _is_cuda_required_device(device_name: Any) -> bool:
    normalised = str(device_name).strip().lower().replace("-", "_")
    return normalised in CUDA_DEVICE_NAMES or normalised.startswith("cuda:")


@lru_cache(maxsize=512)
def _parquet_split_info(path_text: str) -> ParquetSplitInfo:
    path = Path(path_text)
    try:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        return ParquetSplitInfo(
            rows=int(parquet_file.metadata.num_rows),
            columns=frozenset(str(name) for name in parquet_file.schema_arrow.names),
            error=None,
        )
    except Exception as exc:
        try:
            import pandas as pd

            df = pd.read_parquet(path)
            return ParquetSplitInfo(
                rows=int(len(df)),
                columns=frozenset(str(name) for name in df.columns),
                error=None,
            )
        except Exception as fallback_exc:
            return ParquetSplitInfo(
                rows=None,
                columns=frozenset(),
                error=f"{type(exc).__name__}: {exc}; fallback {type(fallback_exc).__name__}: {fallback_exc}",
            )


def _validate_split_file(
    *,
    config_path_text: str,
    path_text: str,
    field: str,
    mode: str | None,
    errors: list[str],
    warnings: list[str],
    required: bool,
) -> None:
    path = Path(path_text)
    if not path.exists():
        message = f"{config_path_text}: data.{field} does not exist: {path_text}"
        if required:
            errors.append(message)
        else:
            warnings.append(message)
        return

    info = _parquet_split_info(path.as_posix())
    if info.error:
        errors.append(f"{config_path_text}: data.{field} is not a readable parquet split: {path_text} ({info.error})")
        return
    if info.rows is not None and info.rows <= 0:
        errors.append(f"{config_path_text}: data.{field} is empty: {path_text}")
    fen_columns = {"normalized_fen", "fen"}
    if not (info.columns & fen_columns):
        errors.append(f"{config_path_text}: data.{field} must contain normalized_fen or fen: {path_text}")
    if mode == PUZZLE_BINARY:
        required_columns = {"fine_label"}
    elif mode == "fine_3class":
        required_columns = {"fine_label"}
    elif mode == "coarse_binary":
        required_columns = {"coarse_label"}
    else:
        required_columns = set()
    missing = sorted(required_columns - set(info.columns))
    if missing:
        errors.append(f"{config_path_text}: data.{field} is missing required column(s) for mode={mode!r}: {missing}")


def _validate_device_syntax(device_name: Any) -> str | None:
    normalised = str(device_name).strip().lower().replace("-", "_")
    if _is_auto(device_name) or normalised in CUDA_DEVICE_NAMES or normalised.startswith("cuda:"):
        return None
    try:
        torch.device(str(device_name))
    except Exception as exc:
        return f"Invalid device={device_name!r}. Use auto, cpu, cuda, cuda:<index>, or nvidia. ({exc})"
    return None


def validate_training_config(
    config: dict[str, Any],
    config_path: str | Path | None = None,
    *,
    require_device_available: bool = True,
) -> list[str]:
    errors: list[str] = []
    warnings: list[str] = []
    path_text = str(config_path) if config_path else "<config>"

    missing = sorted(REQUIRED_TOP_LEVEL - set(config))
    if missing:
        errors.append(f"{path_text}: missing top-level keys: {missing}")

    mode = config.get("mode")
    if mode not in VALID_MODES:
        errors.append(f"{path_text}: mode must be one of {sorted(VALID_MODES)}, got {mode!r}")
    expected_classes = expected_num_classes(mode) if mode in VALID_MODES else None

    run_cfg = config.get("run", {}) if isinstance(config.get("run"), dict) else {}
    if not run_cfg.get("name"):
        errors.append(f"{path_text}: run.name is required")
    if not run_cfg.get("output_dir"):
        errors.append(f"{path_text}: run.output_dir is required")

    device_name = config.get("device", "auto")
    device_error = (
        validate_configured_device(device_name)
        if require_device_available
        else _validate_device_syntax(device_name)
    )
    if device_error is not None:
        errors.append(f"{path_text}: {device_error}")
    if device_name is None or str(device_name).strip().lower() == "auto":
        warnings.append(
            f"{path_text}: device=auto can fall back to CPU; use device: nvidia to require an NVIDIA CUDA GPU"
        )

    data_cfg = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
    for key in ["train_path", "val_path"]:
        value = data_cfg.get(key)
        if not value:
            errors.append(f"{path_text}: data.{key} is required")
        else:
            _validate_split_file(
                config_path_text=path_text,
                path_text=str(value),
                field=key,
                mode=mode,
                errors=errors,
                warnings=warnings,
                required=True,
            )
    test_path = data_cfg.get("test_path")
    if test_path:
        _validate_split_file(
            config_path_text=path_text,
            path_text=str(test_path),
            field="test_path",
            mode=mode,
            errors=errors,
            warnings=warnings,
            required=False,
        )
    if data_cfg.get("cache_features") is True:
        warnings.append(f"{path_text}: data.cache_features=true can use a lot of RAM for large splits")
    encoding = data_cfg.get("encoding", "simple_18")
    if encoding not in available_encodings():
        errors.append(f"{path_text}: unknown data.encoding {encoding!r}; available: {available_encodings()}")
        expected_input_channels = 18
    else:
        expected_input_channels = encoding_num_planes(encoding)
        if encoding == "lc0_static_112":
            warnings.append(
                f"{path_text}: data.encoding=lc0_static_112 is LC0-style without move history; "
                "history planes beyond the current FEN are zero"
            )
        if encoding == "lc0_bt4_112":
            warnings.append(
                f"{path_text}: data.encoding=lc0_bt4_112 uses the LC0 BT4-style 112-plane layout "
                "from a single FEN; history planes beyond the current FEN are zero"
            )

    model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    model_name = model_cfg.get("name")
    if model_name not in available_models():
        errors.append(f"{path_text}: unknown model {model_name!r}; available: {available_models()}")
    if expected_classes is not None:
        configured_classes = int(model_cfg.get("num_classes", expected_classes))
        if configured_classes != expected_classes:
            errors.append(
                f"{path_text}: model.num_classes={configured_classes} does not match mode={mode} "
                f"(expected {expected_classes})"
            )
    input_channels = int(model_cfg.get("input_channels", expected_input_channels))
    if input_channels != expected_input_channels:
        errors.append(
            f"{path_text}: model.input_channels={input_channels} does not match "
            f"data.encoding={encoding} ({expected_input_channels} planes)"
        )

    training_cfg = config.get("training", {}) if isinstance(config.get("training"), dict) else {}
    for key in ["epochs", "batch_size"]:
        value = _int_or_error(training_cfg.get(key, 0), f"training.{key}", path_text, errors)
        if value is not None and value <= 0:
            errors.append(f"{path_text}: training.{key} must be positive")
    min_epochs = _int_or_error(training_cfg.get("min_epochs", 0), "training.min_epochs", path_text, errors) or 0
    min_active_epochs = (
        _int_or_error(training_cfg.get("min_active_epochs", min_epochs), "training.min_active_epochs", path_text, errors)
        or 0
    )
    epochs = _int_or_error(training_cfg.get("epochs", 0), "training.epochs", path_text, errors) or 0
    if min_epochs < 0:
        errors.append(f"{path_text}: training.min_epochs must be non-negative")
    if min_active_epochs < 0:
        errors.append(f"{path_text}: training.min_active_epochs must be non-negative")
    if epochs > 0 and min_epochs > epochs:
        errors.append(f"{path_text}: training.min_epochs cannot exceed training.epochs")
    if epochs > 0 and min_active_epochs > epochs:
        errors.append(f"{path_text}: training.min_active_epochs cannot exceed training.epochs")
    num_workers_value = training_cfg.get("num_workers", "auto")
    num_workers: int | None
    if _is_auto(num_workers_value):
        num_workers = None
    else:
        num_workers = _int_or_error(num_workers_value, "training.num_workers", path_text, errors)
        if num_workers is not None and num_workers < 0:
            errors.append(f"{path_text}: training.num_workers must be non-negative or auto")
    if data_cfg.get("cache_features") is True and (num_workers is None or num_workers > 0):
        warnings.append(
            f"{path_text}: data.cache_features=true with loader workers can duplicate cached tensors in worker processes"
        )
    if num_workers == 0 and bool(training_cfg.get("persistent_workers", False)):
        errors.append(f"{path_text}: training.persistent_workers=true requires training.num_workers>0 or auto")
    prefetch_factor = training_cfg.get("prefetch_factor")
    if prefetch_factor is not None:
        parsed_prefetch = _int_or_error(prefetch_factor, "training.prefetch_factor", path_text, errors)
        if parsed_prefetch is not None and parsed_prefetch <= 0:
            errors.append(f"{path_text}: training.prefetch_factor must be positive")
    mixed_precision = training_cfg.get("mixed_precision", False)
    if not isinstance(mixed_precision, bool) and str(mixed_precision).strip().lower() != "auto":
        errors.append(f"{path_text}: training.mixed_precision must be true, false, or auto")
    matmul_precision = str(training_cfg.get("matmul_precision", "high")).strip().lower()
    if matmul_precision not in {"highest", "high", "medium"}:
        errors.append(f"{path_text}: training.matmul_precision must be highest, high, or medium")
    if training_cfg.get("class_weighting") not in {None, "none", "balanced"}:
        errors.append(f"{path_text}: training.class_weighting must be none or balanced")
    scheduler_cfg = training_cfg.get("lr_scheduler", {})
    if scheduler_cfg is not None and not isinstance(scheduler_cfg, dict):
        errors.append(f"{path_text}: training.lr_scheduler must be a mapping when provided")
    elif isinstance(scheduler_cfg, dict):
        scheduler_name = str(scheduler_cfg.get("name", "none")).strip().lower()
        if scheduler_name not in {"", "none", "off", "false", "reduce_on_plateau", "cosine"}:
            errors.append(
                f"{path_text}: training.lr_scheduler.name={scheduler_name!r} must be none, reduce_on_plateau, or cosine"
            )
    tier = str(training_cfg.get("reliability_tier", "")).strip().lower()
    if tier in {"paper", "paper_grade", "paper-grade", "publishable"}:
        if not _is_cuda_required_device(device_name):
            errors.append(f"{path_text}: paper-grade training requires device: nvidia or an explicit CUDA device")
        if not test_path:
            errors.append(f"{path_text}: paper-grade training requires data.test_path")
        elif not Path(str(test_path)).exists():
            errors.append(f"{path_text}: paper-grade training requires an existing data.test_path: {test_path}")
        if epochs < 20:
            errors.append(f"{path_text}: paper-grade training requires training.epochs >= 20")
        if int(training_cfg.get("early_stopping_patience", 0)) < 5:
            errors.append(f"{path_text}: paper-grade training requires early_stopping_patience >= 5")
        if min_epochs < 10:
            errors.append(f"{path_text}: paper-grade training requires min_epochs >= 10")
        if mixed_precision is False:
            warnings.append(
                f"{path_text}: paper-grade CUDA benchmark has mixed_precision=false; use true or auto for faster NVIDIA runs"
            )

    return [f"ERROR: {message}" for message in errors] + [f"WARNING: {message}" for message in warnings]

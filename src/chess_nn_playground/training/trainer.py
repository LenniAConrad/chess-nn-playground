from __future__ import annotations

import gc
import json
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from chess_nn_playground.data.dataset import BINARY_MODES, PUZZLE_BINARY, ChessPositionDataset
from chess_nn_playground.evaluation.plots import (
    plot_calibration,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_curves,
    plot_rectangular_confusion_matrix,
)
from chess_nn_playground.evaluation.reports import build_run_report
from chess_nn_playground.evaluation.slices import write_slice_artifacts
from chess_nn_playground.models.trunk.cnn import count_parameters, model_summary_text
from chess_nn_playground.training.callbacks import EarlyStopping
from chess_nn_playground.training.checkpointing import load_checkpoint, save_checkpoint
from chess_nn_playground.training.device import resolve_torch_device
from chess_nn_playground.training.runtime_artifacts import benchmark_inference_forward, speed_totals
from chess_nn_playground.training.runtime_config import (
    config_fingerprint,
    configure_torch_precision,
    resolve_num_workers,
    resolve_run_paths,
    resolve_training_runtime,
    safe_name,
    sync_device_for_timing,
)
from chess_nn_playground.training.runtime_data import (
    binary_pos_weight_tensor,
    build_dataset_bundle,
    build_loader,
    class_weight_tensor,
)
from chess_nn_playground.training.runtime_loss import (
    build_loss_runtime,
    compute_training_loss,
    resolve_loss_runtime_config,
)
from chess_nn_playground.training.runtime_model import build_model_runtime
from chess_nn_playground.training.runtime_outputs import (
    batch_fine_labels,
    fine_to_binary_matrix,
    optional_int,
    primary_logits,
    probabilities_from_logits,
    scalar_output_columns,
)
from chess_nn_playground.training.metrics import compute_metrics
from chess_nn_playground.utils.config import save_yaml
from chess_nn_playground.utils.env import collect_environment
from chess_nn_playground.utils.logging import write_json
from chess_nn_playground.utils.paths import utc_timestamp
from chess_nn_playground.utils.seed import set_seed


_safe_name = safe_name
_probabilities_from_logits = probabilities_from_logits
_primary_logits = primary_logits
_scalar_output_columns = scalar_output_columns
_optional_int = optional_int
_batch_fine_labels = batch_fine_labels
_sync_device_for_timing = sync_device_for_timing
_resolve_num_workers = resolve_num_workers
_fine_to_binary_matrix = fine_to_binary_matrix

CPU_OOM_FALLBACK_LABEL = "cpu_oom_fallback_non_benchmark"
CPU_OOM_FALLBACK_ENV = "CHESS_NN_ALLOW_CPU_OOM_FALLBACK"
CPU_OOM_FALLBACK_DISABLE_ENV = "CHESS_NN_DISABLE_CPU_FALLBACK"


def _bool_from_config(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


class Trainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.seed = int(config.get("seed", 42))
        set_seed(self.seed, deterministic=bool(config.get("deterministic", True)))
        self.mode = config.get("mode", "coarse_binary")
        self.metric_num_classes = 2 if self.mode in BINARY_MODES else 3
        self.requested_device = config.get("device", "auto")
        self.device = resolve_torch_device(self.requested_device)
        paths = resolve_run_paths(config)
        self.run_dir = paths.run_dir
        self.train_path = paths.train_path
        self.val_path = paths.val_path
        self.test_path = paths.test_path
        self.input_encoding = paths.input_encoding

        data_cfg = config.get("data", {}) or {}
        training_cfg = config.get("training", {}) or {}
        runtime_cfg = resolve_training_runtime(config, device=self.device, mode=self.mode)
        self.batch_size = runtime_cfg.batch_size
        self.num_workers = runtime_cfg.num_workers
        self.persistent_workers = runtime_cfg.persistent_workers
        self.prefetch_factor = runtime_cfg.prefetch_factor
        self.epochs = runtime_cfg.epochs
        self.min_epochs = runtime_cfg.min_epochs
        self.min_active_epochs = runtime_cfg.min_active_epochs
        self.learning_rate = runtime_cfg.learning_rate
        self.weight_decay = runtime_cfg.weight_decay
        self.gradient_clip_norm = runtime_cfg.gradient_clip_norm
        self.use_amp = runtime_cfg.use_amp
        self.pin_memory = runtime_cfg.pin_memory
        self.allow_tf32 = runtime_cfg.allow_tf32
        self.matmul_precision = runtime_cfg.matmul_precision
        self.monitor_metric = runtime_cfg.monitor_metric
        configure_torch_precision(
            device=self.device,
            allow_tf32=self.allow_tf32,
            matmul_precision=self.matmul_precision,
        )
        inference_speed_cfg = training_cfg.get("inference_speed_benchmark", {})
        if isinstance(inference_speed_cfg, dict):
            self.inference_speed_enabled = _bool_from_config(inference_speed_cfg.get("enabled"), default=True)
            self.inference_speed_devices = list(inference_speed_cfg.get("devices", ["cpu", "cuda"]))
            self.inference_speed_batch_sizes = inference_speed_cfg.get("batch_sizes")
            self.inference_speed_warmup_iters = int(inference_speed_cfg.get("warmup_iters", 2))
            self.inference_speed_timed_iters = int(inference_speed_cfg.get("timed_iters", 5))
        else:
            self.inference_speed_enabled = _bool_from_config(inference_speed_cfg, default=True)
            self.inference_speed_devices = ["cpu", "cuda"]
            self.inference_speed_batch_sizes = None
            self.inference_speed_warmup_iters = 2
            self.inference_speed_timed_iters = 5

        self.loss_runtime_config = resolve_loss_runtime_config(training_cfg=training_cfg, data_cfg=data_cfg)
        self.class_weighting = self.loss_runtime_config.class_weighting
        self.loss_name = self.loss_runtime_config.loss_name
        self.veto_select_cfg = self.loss_runtime_config.veto_select_cfg
        self.dykstra_lcp_cfg = self.loss_runtime_config.dykstra_lcp_cfg
        self.dykstra_vetoselect_cfg = self.loss_runtime_config.dykstra_vetoselect_cfg
        self.srpa_cfg = self.loss_runtime_config.srpa_cfg
        self.contamination_dro_cfg = self.loss_runtime_config.contamination_dro_cfg
        self.material_locked_dro_cfg = self.loss_runtime_config.material_locked_dro_cfg
        self.soft_sort_order_cfg = self.loss_runtime_config.soft_sort_order_cfg
        self.conditional_surprisal_gate_cfg = self.loss_runtime_config.conditional_surprisal_gate_cfg
        self.veto_select_warmup_epochs = self.loss_runtime_config.veto_select_warmup_epochs
        self.use_rule_texture = self.loss_runtime_config.use_rule_texture
        self.early_stopping = EarlyStopping(
            patience=int(training_cfg.get("early_stopping_patience", 10)),
            mode="max",
        )
        model_runtime = build_model_runtime(
            config,
            mode=self.mode,
            metric_num_classes=self.metric_num_classes,
            device=self.device,
        )
        self.model_name = model_runtime.name
        self.model = model_runtime.model
        self.model_output_classes = model_runtime.output_classes
        self.single_logit_binary = model_runtime.single_logit_binary
        self.model_input_channels = model_runtime.input_channels
        self.model_complexity = model_runtime.complexity
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.scheduler = self._build_scheduler(runtime_cfg.scheduler_cfg)
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except Exception:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.start_epoch = 1
        resume_path = training_cfg.get("resume_from")
        if not resume_path and training_cfg.get("resume_run_dir"):
            candidate = Path(training_cfg["resume_run_dir"]) / "checkpoint_last.pt"
            if candidate.exists():
                resume_path = str(candidate)
        if resume_path:
            checkpoint = load_checkpoint(resume_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if checkpoint.get("optimizer_state_dict"):
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler is not None and checkpoint.get("scheduler_state_dict"):
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.start_epoch = int(checkpoint.get("epoch", 0)) + 1

        datasets = build_dataset_bundle(
            train_path=self.train_path,
            val_path=self.val_path,
            test_path=self.test_path,
            mode=self.mode,
            encoding=self.input_encoding,
            cache_features=bool(data_cfg.get("cache_features", False)),
            include_rule_texture=self.use_rule_texture,
        )
        self.train_dataset = datasets.train
        self.val_dataset = datasets.val
        self.test_dataset = datasets.test
        self.loss_runtime = build_loss_runtime(
            self.loss_runtime_config,
            train_dataset=self.train_dataset,
            metric_num_classes=self.metric_num_classes,
            single_logit_binary=self.single_logit_binary,
            device=self.device,
        )
        self.criterion = self.loss_runtime.criterion
        self.best_score = -math.inf
        self.best_metrics: dict[str, Any] = {}

    def _build_scheduler(self, scheduler_cfg: Any) -> Any | None:
        if not isinstance(scheduler_cfg, dict):
            return None
        name = str(scheduler_cfg.get("name", "none")).strip().lower()
        if name in {"", "none", "off", "false"}:
            return None
        if name == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=str(scheduler_cfg.get("mode", "max")),
                factor=float(scheduler_cfg.get("factor", 0.5)),
                patience=int(scheduler_cfg.get("patience", 2)),
                threshold=float(scheduler_cfg.get("threshold", 1e-4)),
                threshold_mode=str(scheduler_cfg.get("threshold_mode", "rel")),
                cooldown=int(scheduler_cfg.get("cooldown", 0)),
                min_lr=float(scheduler_cfg.get("min_lr", 1e-5)),
            )
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=int(scheduler_cfg.get("t_max", max(1, self.epochs))),
                eta_min=float(scheduler_cfg.get("min_lr", 1e-5)),
            )
        raise ValueError(f"Unsupported lr_scheduler.name={name!r}")

    def _step_scheduler(self, score: float) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(score)
        else:
            self.scheduler.step()

    def _active_epochs_completed(self, epoch: int) -> int:
        if self.loss_name in {"veto_select", "dykstra_vetoselect"}:
            return max(0, epoch - self.veto_select_warmup_epochs)
        return max(0, epoch)

    def _can_stop_early(self, epoch: int) -> bool:
        return epoch >= self.min_epochs and self._active_epochs_completed(epoch) >= self.min_active_epochs

    def _class_names(self) -> list[str]:
        if self.mode == "fine_3class":
            return ["0 non-puzzle", "1 near-puzzle", "2 puzzle"]
        if self.mode == PUZZLE_BINARY:
            return ["0 non-puzzle/near-puzzle", "1 puzzle"]
        return ["0 non-puzzle", "1 puzzle/near-puzzle"]

    def _class_weights(self) -> torch.Tensor:
        return class_weight_tensor(
            self.train_dataset,
            num_classes=self.metric_num_classes,
            device=self.device,
        )

    def _binary_pos_weight(self) -> torch.Tensor:
        return binary_pos_weight_tensor(self.train_dataset, device=self.device)

    def _batch_rule_texture(self, batch: dict[str, Any]) -> torch.Tensor | None:
        texture = batch.get("rule_texture")
        if texture is None:
            return None
        return texture.to(self.device, non_blocking=self.pin_memory)

    def _loader(self, dataset: ChessPositionDataset, shuffle: bool) -> Any:
        return build_loader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
        )

    def _compute_loss(
        self,
        output: torch.Tensor | dict[str, torch.Tensor],
        logits: torch.Tensor,
        y: torch.Tensor,
        epoch: int | None,
        texture: torch.Tensor | None = None,
        fine_label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return compute_training_loss(
            loss_name=self.loss_name,
            criterion=self.criterion,
            output=output,
            logits=logits,
            target=y,
            epoch=epoch,
            veto_select_warmup_epochs=self.veto_select_warmup_epochs,
            single_logit_binary=self.single_logit_binary,
            texture=texture,
            fine_label=fine_label,
        )

    def train_epoch(self, epoch: int) -> dict[str, Any]:
        self.model.train()
        labels: list[int] = []
        probs: list[list[float]] = []
        losses: list[float] = []
        loader = self._loader(self.train_dataset, shuffle=True)
        progress = tqdm(loader, desc=f"epoch {epoch} train", leave=False)
        _sync_device_for_timing(self.device)
        started = time.perf_counter()
        sample_count = 0
        batch_count = 0
        for batch in progress:
            x = batch["x"].to(self.device, non_blocking=self.pin_memory)
            y = batch["y"].to(self.device, non_blocking=self.pin_memory)
            sample_count += int(y.shape[0])
            batch_count += 1
            texture = self._batch_rule_texture(batch)
            fine_label = _batch_fine_labels(batch, self.device, self.pin_memory)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                output = self.model(x)
                logits = _primary_logits(output)
                loss = self._compute_loss(output, logits, y, epoch, texture=texture, fine_label=fine_label)
            self.scaler.scale(loss).backward()
            if self.gradient_clip_norm is not None and self.gradient_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            losses.append(float(loss.detach().cpu()))
            labels.extend(y.detach().cpu().numpy().astype(int).tolist())
            probs.extend(_probabilities_from_logits(logits, self.single_logit_binary).tolist())
            progress.set_postfix(loss=np.mean(losses))
        _sync_device_for_timing(self.device)
        elapsed = max(time.perf_counter() - started, 1e-9)
        metrics = compute_metrics(labels, probs, mode=self.mode)
        metrics["loss"] = float(np.mean(losses)) if losses else None
        metrics["epoch"] = epoch
        metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        metrics["sample_count"] = sample_count
        metrics["batch_count"] = batch_count
        metrics["elapsed_seconds"] = elapsed
        metrics["samples_per_second"] = sample_count / elapsed
        metrics["batches_per_second"] = batch_count / elapsed
        return metrics

    @torch.no_grad()
    def evaluate_dataset(
        self,
        dataset: ChessPositionDataset,
        split_name: str,
        epoch: int | None = None,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        self.model.eval()
        labels: list[int] = []
        probs: list[list[float]] = []
        losses: list[float] = []
        rows: list[dict[str, Any]] = []
        loader = self._loader(dataset, shuffle=False)
        _sync_device_for_timing(self.device)
        started = time.perf_counter()
        sample_count = 0
        batch_count = 0
        for batch in tqdm(loader, desc=f"{split_name} eval", leave=False):
            x = batch["x"].to(self.device, non_blocking=self.pin_memory)
            y = batch["y"].to(self.device, non_blocking=self.pin_memory)
            sample_count += int(y.shape[0])
            batch_count += 1
            texture = self._batch_rule_texture(batch)
            fine_label = _batch_fine_labels(batch, self.device, self.pin_memory)
            output = self.model(x)
            logits = _primary_logits(output)
            loss = self._compute_loss(output, logits, y, epoch, texture=texture, fine_label=fine_label)
            batch_probs = _probabilities_from_logits(logits, self.single_logit_binary)
            batch_pred = batch_probs.argmax(axis=1)
            output_columns = _scalar_output_columns(output)
            losses.append(float(loss.detach().cpu()))
            y_list = y.detach().cpu().numpy().astype(int).tolist()
            labels.extend(y_list)
            probs.extend(batch_probs.tolist())
            for idx, sample_id in enumerate(batch["sample_id"]):
                probability_list = [float(v) for v in batch_probs[idx].tolist()]
                row = {
                    "sample_id": sample_id,
                    "fen": batch["fen"][idx],
                    "true_label": int(y_list[idx]),
                    "true_coarse_label": _optional_int(batch["coarse_label"][idx]),
                    "true_fine_label": _optional_int(batch["fine_label"][idx]),
                    "predicted_label": int(batch_pred[idx]),
                    "probabilities": json.dumps(probability_list),
                    "confidence": float(max(probability_list)),
                    "label_status": batch["label_status"][idx],
                    "correct": int(y_list[idx]) == int(batch_pred[idx]),
                }
                for cls_idx, value in enumerate(probability_list):
                    row[f"prob_{cls_idx}"] = value
                for key, values in output_columns.items():
                    row[key] = values[idx]
                if "rule_texture" in batch:
                    row["rule_texture"] = float(batch["rule_texture"][idx].detach().cpu())
                metadata = batch["metadata"][idx]
                for key, value in metadata.items():
                    if key not in row:
                        row[key] = value
                rows.append(row)
        _sync_device_for_timing(self.device)
        elapsed = max(time.perf_counter() - started, 1e-9)
        metrics = compute_metrics(labels, probs, mode=self.mode)
        metrics["loss"] = float(np.mean(losses)) if losses else None
        metrics["sample_count"] = sample_count
        metrics["batch_count"] = batch_count
        metrics["elapsed_seconds"] = elapsed
        metrics["samples_per_second"] = sample_count / elapsed
        metrics["batches_per_second"] = batch_count / elapsed
        return metrics, pd.DataFrame(rows)

    def _score_metric(self, metrics: dict[str, Any]) -> float:
        value = metrics.get(self.monitor_metric)
        if value is None:
            # Backwards-compatible fallbacks: try the historical default for the
            # mode, then accuracy, then negative loss.
            if self.mode in BINARY_MODES:
                value = metrics.get("f1")
            else:
                value = metrics.get("macro_f1")
        if value is None:
            value = metrics.get("accuracy")
        if value is None:
            value = -metrics.get("loss", math.inf)
        return float(value)

    def _write_run_metadata(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        class_counts = {
            "train": self.train_dataset.label_counts(),
            "val": self.val_dataset.label_counts(),
        }
        source_class_counts = {
            "train": self.train_dataset.value_counts("fine_label"),
            "val": self.val_dataset.value_counts("fine_label"),
        }
        if self.test_dataset is not None:
            class_counts["test"] = self.test_dataset.label_counts()
            source_class_counts["test"] = self.test_dataset.value_counts("fine_label")
        training_for_metadata = self.config.get("training", {})
        if not isinstance(training_for_metadata, dict):
            training_for_metadata = {}
        metadata = {
            "run_name": self.run_dir.name,
            "timestamp": utc_timestamp(),
            "config_hash": config_fingerprint(self.config),
            "git_commit": collect_environment(".").get("git_commit"),
            "python_version": collect_environment(".").get("python_version"),
            "pytorch_version": torch.__version__,
            "requested_device": str(self.requested_device),
            "device": str(self.device),
            "seed": self.seed,
            "model_name": self.model_name,
            "num_params": count_parameters(self.model),
            "complexity": self.model_complexity,
            "input_encoding": self.input_encoding,
            "dataset_path": str(self.train_path.parent),
            "split_paths": {
                "train": str(self.train_path),
                "val": str(self.val_path),
                "test": str(self.test_path),
            },
            "mode": self.mode,
            "architecture_scale": self.config.get("architecture_scale", {"variant": "base", "multiplier": 1.0}),
            "class_counts": class_counts,
            "source_class_counts": source_class_counts,
            "config_path": str(self.run_dir / "config_resolved.yaml"),
            "checkpoint_best": str(self.run_dir / "checkpoint_best.pt"),
            "checkpoint_last": str(self.run_dir / "checkpoint_last.pt"),
            "notes": self.config.get("notes"),
            "benchmark_status": self.config.get("benchmark_status", "benchmark_candidate"),
            "cpu_oom_fallback": self.config.get(
                "cpu_oom_fallback",
                {
                    "used": False,
                    "enabled": bool(training_for_metadata.get("allow_cpu_oom_fallback", False)),
                },
            ),
            "training": {
                "epochs": self.epochs,
                "min_epochs": self.min_epochs,
                "min_active_epochs": self.min_active_epochs,
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "persistent_workers": self.persistent_workers,
                "prefetch_factor": self.prefetch_factor if self.num_workers > 0 else None,
                "pin_memory": self.pin_memory,
                "mixed_precision": self.use_amp,
                "allow_tf32": self.allow_tf32,
                "matmul_precision": self.matmul_precision,
                "early_stopping_patience": self.early_stopping.patience,
                "monitor": self.monitor_metric,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "gradient_clip_norm": self.gradient_clip_norm,
                "lr_scheduler": self.config.get("training", {}).get("lr_scheduler"),
            },
        }
        if extra:
            metadata.update(extra)
        write_json(metadata, self.run_dir / "run_metadata.json")
        return metadata

    def _write_metric_history(
        self,
        train_history: list[dict[str, Any]],
        val_history: list[dict[str, Any]],
    ) -> Path:
        path = self.run_dir / "metrics_history.json"
        write_json({"train": train_history, "val": val_history}, path)
        return path

    @staticmethod
    def _speed_totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return speed_totals(rows)

    def _inference_benchmark_batch_sizes(self) -> list[int]:
        configured = self.inference_speed_batch_sizes
        if configured is None:
            values = [1, self.batch_size]
        elif isinstance(configured, int):
            values = [configured]
        else:
            values = list(configured)
        batch_sizes: list[int] = []
        for value in values:
            batch_size = int(value)
            if batch_size > 0 and batch_size not in batch_sizes:
                batch_sizes.append(batch_size)
        return batch_sizes or [1]

    def _inference_sample_shape(self) -> tuple[int, ...]:
        if len(self.train_dataset) > 0:
            sample = self.train_dataset[0]["x"]
            return tuple(int(value) for value in sample.shape)
        return (int(self.model_input_channels), 8, 8)

    def _run_inference_speed_benchmark(self) -> dict[str, Any]:
        if not self.inference_speed_enabled:
            return {"enabled": False}
        return benchmark_inference_forward(
            self.model,
            sample_shape=self._inference_sample_shape(),
            batch_sizes=self._inference_benchmark_batch_sizes(),
            devices=self.inference_speed_devices,
            warmup_iters=self.inference_speed_warmup_iters,
            timed_iters=self.inference_speed_timed_iters,
        )

    @staticmethod
    def _representative_inference_row(
        inference_benchmark: dict[str, Any],
        *,
        device_key: str,
        batch_size: int,
    ) -> dict[str, Any] | None:
        devices = inference_benchmark.get("devices", {})
        device_result = devices.get(device_key) if isinstance(devices, dict) else None
        if not isinstance(device_result, dict) or not device_result.get("available"):
            return None
        rows = device_result.get("results", [])
        if not isinstance(rows, list):
            return None
        for row in rows:
            if isinstance(row, dict) and int(row.get("batch_size") or 0) == batch_size:
                return row
        return next((row for row in rows if isinstance(row, dict)), None)

    def _write_speed_summary(
        self,
        train_history: list[dict[str, Any]],
        val_history: list[dict[str, Any]],
        split_results: dict[str, Any],
        fit_elapsed_seconds: float,
    ) -> dict[str, Any]:
        train_speed = self._speed_totals(train_history)
        val_speed = self._speed_totals(val_history)
        final_eval: dict[str, Any] = {}
        for split, result in split_results.items():
            metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
            final_eval[split] = {
                "sample_count": metrics.get("sample_count"),
                "batch_count": metrics.get("batch_count"),
                "elapsed_seconds": metrics.get("elapsed_seconds"),
                "samples_per_second": metrics.get("samples_per_second"),
                "batches_per_second": metrics.get("batches_per_second"),
            }
        inference_benchmark = self._run_inference_speed_benchmark()
        cpu_inference = self._representative_inference_row(
            inference_benchmark,
            device_key="cpu",
            batch_size=self.batch_size,
        )
        gpu_inference = self._representative_inference_row(
            inference_benchmark,
            device_key="cuda",
            batch_size=self.batch_size,
        )
        summary = {
            "fit_elapsed_seconds": float(fit_elapsed_seconds),
            "train": train_speed,
            "validation_epochs": val_speed,
            "final_eval": final_eval,
            "inference_forward_benchmark": inference_benchmark,
            "train_samples_per_second": train_speed.get("samples_per_second"),
            "val_samples_per_second": val_speed.get("samples_per_second"),
            "cpu_inference_samples_per_second": (
                cpu_inference.get("samples_per_second") if cpu_inference is not None else None
            ),
            "cpu_inference_ms_per_sample": (
                cpu_inference.get("mean_ms_per_sample") if cpu_inference is not None else None
            ),
            "gpu_inference_samples_per_second": (
                gpu_inference.get("samples_per_second") if gpu_inference is not None else None
            ),
            "gpu_inference_ms_per_sample": (
                gpu_inference.get("mean_ms_per_sample") if gpu_inference is not None else None
            ),
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "mixed_precision": self.use_amp,
            "device": str(self.device),
        }
        write_json(summary, self.run_dir / "speed_summary.json")
        return summary

    def _load_existing_metric_history(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        path = self.run_dir / "metrics_history.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                train = data.get("train", [])
                val = data.get("val", [])
                if isinstance(train, list) and isinstance(val, list):
                    return train, val
            except Exception:
                pass
        train_csv = self.run_dir / "metrics_train.csv"
        val_csv = self.run_dir / "metrics_val.csv"
        if train_csv.exists() and val_csv.exists():
            try:
                train_df = pd.read_csv(train_csv)
                val_df = pd.read_csv(val_csv)
                return train_df.to_dict(orient="records"), val_df.to_dict(orient="records")
            except Exception:
                pass
        return [], []

    def _load_best_checkpoint_state(self) -> tuple[int | None, dict[str, Any] | None]:
        path = self.run_dir / "checkpoint_best.pt"
        if not path.exists():
            return None, None
        checkpoint = load_checkpoint(path, map_location=self.device)
        metrics = dict(checkpoint.get("metrics") or {})
        epoch = int(checkpoint.get("epoch", 0)) or None
        if metrics:
            self.best_metrics = metrics
            self.best_score = self._score_metric(metrics)
        return epoch, metrics

    def _write_final_split_result(
        self,
        split: str,
        dataset: ChessPositionDataset,
        epoch: int | None,
    ) -> dict[str, Any]:
        metrics, predictions = self.evaluate_dataset(dataset, split, epoch=epoch)
        metrics = dict(metrics)
        metrics["split"] = split
        metrics["epoch"] = epoch
        metrics["prediction_rows"] = int(len(predictions))
        prediction_path = self.run_dir / f"predictions_{split}.parquet"
        metrics_path = self.run_dir / f"metrics_{split}_final.json"
        predictions.to_parquet(prediction_path, index=False)
        metrics["predictions_path"] = str(prediction_path)
        write_json(metrics, metrics_path)
        return {
            "split": split,
            "metrics": metrics,
            "metrics_path": str(metrics_path),
            "predictions_path": str(prediction_path),
            "prediction_rows": int(len(predictions)),
        }

    def _run_artifact_pipeline(
        self,
        train_history: list[dict[str, Any]],
        val_history: list[dict[str, Any]],
        final_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        train_df = pd.DataFrame(train_history)
        val_df = pd.DataFrame(val_history)
        artifact_paths: dict[str, Any] = {}

        artifact_paths["metrics_history"] = str(self.run_dir / "metrics_history.json")
        artifact_paths["metrics_by_split"] = str(self.run_dir / "metrics_by_split.json")
        complexity_path = self.run_dir / "complexity_estimate.json"
        if complexity_path.exists():
            artifact_paths["complexity_estimate"] = str(complexity_path)
        speed_summary_path = self.run_dir / "speed_summary.json"
        if speed_summary_path.exists():
            artifact_paths["speed_summary"] = str(speed_summary_path)
        artifact_paths["split_metrics"] = {
            split: str(self.run_dir / f"metrics_{split}_final.json")
            for split in ["train", "val", "test"]
            if (self.run_dir / f"metrics_{split}_final.json").exists()
        }
        artifact_paths["predictions"] = {
            split: str(self.run_dir / f"predictions_{split}.parquet")
            for split in ["train", "val", "test"]
            if (self.run_dir / f"predictions_{split}.parquet").exists()
        }
        artifact_paths["curves"] = plot_curves(train_df, val_df, self.run_dir)
        class_names = self._class_names()
        artifact_paths["confusion_matrix_val"] = plot_confusion_matrix(
            final_metrics.get("confusion_matrix"),
            self.run_dir / "confusion_matrix_val.png",
            class_names,
            title="Validation Confusion Matrix",
        )
        artifact_paths["confusion_matrix_test"] = plot_confusion_matrix(
            final_metrics.get("test_confusion_matrix"),
            self.run_dir / "confusion_matrix_test.png",
            class_names,
            title="Test Confusion Matrix",
        )
        fine_rows = ["0 non-puzzle", "1 near-puzzle", "2 puzzle"]
        binary_cols = ["0 predicted non-puzzle", "1 predicted puzzle"]
        val_pred_path = self.run_dir / "predictions_val.parquet"
        if self.mode in BINARY_MODES and val_pred_path.exists():
            val_preds_for_matrix = pd.read_parquet(
                val_pred_path,
                columns=["true_fine_label", "predicted_label"],
            )
            val_fine_binary = _fine_to_binary_matrix(val_preds_for_matrix)
            if val_fine_binary is not None:
                final_metrics["fine_to_binary_confusion_matrix"] = val_fine_binary
                artifact_paths["fine_to_binary_confusion_matrix_val"] = plot_rectangular_confusion_matrix(
                    val_fine_binary,
                    self.run_dir / "fine_to_binary_confusion_matrix_val.png",
                    fine_rows,
                    binary_cols,
                    title="Validation: Source Class To Binary Prediction",
                    x_label="Model output",
                    y_label="Original source class",
                )
        test_pred_path = self.run_dir / "predictions_test.parquet"
        if self.mode in BINARY_MODES and test_pred_path.exists():
            test_preds_for_matrix = pd.read_parquet(
                test_pred_path,
                columns=["true_fine_label", "predicted_label"],
            )
            test_fine_binary = _fine_to_binary_matrix(test_preds_for_matrix)
            if test_fine_binary is not None:
                final_metrics["test_fine_to_binary_confusion_matrix"] = test_fine_binary
                artifact_paths["fine_to_binary_confusion_matrix_test"] = plot_rectangular_confusion_matrix(
                    test_fine_binary,
                    self.run_dir / "fine_to_binary_confusion_matrix_test.png",
                    fine_rows,
                    binary_cols,
                    title="Test: Source Class To Binary Prediction",
                    x_label="Model output",
                    y_label="Original source class",
                )
        train_counts = {str(k): int(v) for k, v in sorted(self.train_dataset.label_counts().items())}
        artifact_paths["class_distribution"] = plot_class_distribution(
            train_counts,
            self.run_dir / "class_distribution.png",
        )
        pred_path = self.run_dir / "predictions_val.parquet"
        if pred_path.exists():
            preds = pd.read_parquet(pred_path)
            prob_cols = [c for c in preds.columns if c.startswith("prob_")]
            if prob_cols:
                artifact_paths["calibration"] = plot_calibration(
                    preds["true_label"].astype(int).tolist(),
                    preds[prob_cols].to_numpy().tolist(),
                    self.run_dir / "calibration_plot.png",
                    positive_class=1 if self.mode in BINARY_MODES else None,
                )
        slice_artifacts: dict[str, Any] = {}
        for split, split_path in [
            ("train", self.train_path),
            ("val", self.val_path),
            ("test", self.test_path),
        ]:
            pred_path = self.run_dir / f"predictions_{split}.parquet"
            if not pred_path.exists():
                continue
            split_slice = write_slice_artifacts(
                run_dir=self.run_dir,
                split=split,
                pred_path=pred_path,
                split_path=split_path,
            )
            if split_slice is not None:
                slice_artifacts[split] = split_slice
        if slice_artifacts:
            artifact_paths["slice_analysis"] = slice_artifacts
        write_json(artifact_paths, self.run_dir / "artifact_manifest.json")
        return artifact_paths

    def fit(self) -> Path:
        if len(self.train_dataset) == 0:
            raise ValueError("Training dataset is empty")
        fit_started = time.perf_counter()
        save_yaml(self.config, self.run_dir / "config_resolved.yaml")
        write_json(collect_environment("."), self.run_dir / "environment.json")
        (self.run_dir / "model_summary.txt").write_text(model_summary_text(self.model), encoding="utf-8")
        write_json(self.model_complexity, self.run_dir / "complexity_estimate.json")
        self._write_run_metadata()

        if self.start_epoch > 1:
            train_history, val_history = self._load_existing_metric_history()
            best_epoch, _ = self._load_best_checkpoint_state()
        else:
            train_history = []
            val_history = []
            best_epoch = None
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics, val_predictions = self.evaluate_dataset(self.val_dataset, "val", epoch=epoch)
            val_metrics["epoch"] = epoch
            train_history.append(train_metrics)
            val_history.append(val_metrics)

            score = self._score_metric(val_metrics)
            if score > self.best_score:
                self.best_score = score
                self.best_metrics = dict(val_metrics)
                best_epoch = epoch
                save_checkpoint(
                    self.run_dir / "checkpoint_best.pt",
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    val_metrics,
                    self.config,
                )
                val_predictions.to_parquet(self.run_dir / "predictions_val.parquet", index=False)

            save_checkpoint(
                self.run_dir / "checkpoint_last.pt",
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                val_metrics,
                self.config,
            )
            pd.DataFrame(train_history).to_csv(self.run_dir / "metrics_train.csv", index=False)
            pd.DataFrame(val_history).to_csv(self.run_dir / "metrics_val.csv", index=False)
            self._write_metric_history(train_history, val_history)
            self._step_scheduler(score)
            if self.early_stopping.step(score) and self._can_stop_early(epoch):
                break

        checkpoint = load_checkpoint(self.run_dir / "checkpoint_best.pt", map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        split_results: dict[str, Any] = {
            "train": self._write_final_split_result("train", self.train_dataset, best_epoch),
            "val": self._write_final_split_result("val", self.val_dataset, best_epoch),
        }
        if self.test_dataset is not None and len(self.test_dataset) > 0:
            split_results["test"] = self._write_final_split_result("test", self.test_dataset, best_epoch)

        write_json(
            {
                "best_epoch": best_epoch,
                "best_score": self.best_score,
                "splits": split_results,
            },
            self.run_dir / "metrics_by_split.json",
        )

        final_metrics = dict(split_results["val"]["metrics"])
        final_metrics["best_epoch"] = best_epoch
        final_metrics["best_score"] = self.best_score
        final_metrics.update(
            {
                f"train_{key}": value
                for key, value in split_results["train"]["metrics"].items()
                if key != "confusion_matrix"
            }
        )
        if split_results["train"]["metrics"].get("confusion_matrix") is not None:
            final_metrics["train_confusion_matrix"] = split_results["train"]["metrics"]["confusion_matrix"]
        if "test" in split_results:
            final_metrics.update(
                {
                    f"test_{key}": value
                    for key, value in split_results["test"]["metrics"].items()
                    if key != "confusion_matrix"
                }
            )
            if split_results["test"]["metrics"].get("confusion_matrix") is not None:
                final_metrics["test_confusion_matrix"] = split_results["test"]["metrics"]["confusion_matrix"]

        fit_elapsed_seconds = max(time.perf_counter() - fit_started, 1e-9)
        speed_summary = self._write_speed_summary(train_history, val_history, split_results, fit_elapsed_seconds)
        final_metrics["speed"] = speed_summary
        write_json(final_metrics, self.run_dir / "metrics_final.json")
        artifact_paths = self._run_artifact_pipeline(train_history, val_history, final_metrics)
        self._write_run_metadata({"best_epoch": best_epoch, "best_score": self.best_score, "speed": speed_summary})
        final_metrics["artifacts"] = artifact_paths
        write_json(final_metrics, self.run_dir / "metrics_final.json")
        build_run_report(self.run_dir)
        return self.run_dir


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    if isinstance(exc, RuntimeError) and "cuda" in message and "out of memory" in message:
        return True
    return False


def _release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _cpu_oom_fallback_policy(config: dict[str, Any]) -> tuple[bool, str]:
    if _truthy(os.environ.get(CPU_OOM_FALLBACK_DISABLE_ENV)):
        return False, CPU_OOM_FALLBACK_DISABLE_ENV
    training = config.get("training", {}) if isinstance(config.get("training"), dict) else {}
    if _truthy(training.get("allow_cpu_oom_fallback", False)):
        return True, "training.allow_cpu_oom_fallback"
    if _truthy(os.environ.get(CPU_OOM_FALLBACK_ENV)):
        return True, CPU_OOM_FALLBACK_ENV
    return False, "default"


def _append_note(existing: Any, addition: str) -> str:
    text = str(existing or "").strip()
    return f"{text} {addition}".strip() if text else addition


def _force_cpu_config(
    config: dict[str, Any],
    *,
    original_requested_device: Any,
    oom_reason: BaseException,
    enabled_by: str,
) -> dict[str, Any]:
    cpu_config = dict(config)
    cpu_config["device"] = "cpu"
    cpu_config["benchmark_status"] = CPU_OOM_FALLBACK_LABEL
    cpu_config["notes"] = _append_note(
        cpu_config.get("notes"),
        "CPU OOM fallback run: explicitly labeled non-benchmark because execution moved from CUDA to CPU.",
    )
    run_cfg = dict(cpu_config.get("run", {}) or {})
    base_name = str(run_cfg.get("name", "run")).strip() or "run"
    if CPU_OOM_FALLBACK_LABEL not in base_name:
        run_cfg["name"] = f"{base_name}_{CPU_OOM_FALLBACK_LABEL}"
    run_cfg["benchmark_label"] = CPU_OOM_FALLBACK_LABEL
    cpu_config["run"] = run_cfg
    training = dict(cpu_config.get("training") or {})
    training["allow_cpu_oom_fallback"] = True
    training["mixed_precision"] = False
    training["pin_memory"] = False
    training["allow_tf32"] = False
    # Push DataLoader workers up so the CPU trainer isn't bottlenecked on the
    # main process for batch fetching. ~1/4 of logical cores leaves the rest
    # for the model forward/backward.
    cpu_count = os.cpu_count() or 4
    training["num_workers"] = max(2, min(8, cpu_count // 4))
    training["persistent_workers"] = True
    cpu_config["training"] = training
    cpu_config["cpu_oom_fallback"] = {
        "used": True,
        "enabled_by": enabled_by,
        "label": CPU_OOM_FALLBACK_LABEL,
        "original_requested_device": str(original_requested_device),
        "fallback_device": "cpu",
        "reason": str(oom_reason),
    }
    return cpu_config


def _maximize_cpu_threads() -> None:
    """Lift PyTorch's intra-op thread count to use hyperthreads during CPU fallback.

    PyTorch defaults to physical-core count; on an 8c/16t box that leaves half
    the logical CPUs idle. Override (still respects user override via env).
    """
    if os.environ.get("OMP_NUM_THREADS") or os.environ.get("MKL_NUM_THREADS"):
        return
    cpu_count = os.cpu_count() or 1
    try:
        torch.set_num_threads(cpu_count)
    except Exception as exc:
        print(f"[oom-fallback] could not set torch num_threads={cpu_count}: {exc}", file=sys.stderr)
        return
    try:
        torch.set_num_interop_threads(max(2, cpu_count // 4))
    except Exception:
        pass
    print(f"[oom-fallback] torch CPU threads set to {cpu_count} (logical cores)", file=sys.stderr)


def _total_system_ram_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_pages = os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        return None
    return int(page_size) * int(total_pages)


def _apply_cpu_fallback_heap_cap() -> None:
    """Best-effort guard so CPU fallback can't eat the whole desktop's RAM.

    Cap is sized off TOTAL physical RAM (not free RAM at fallback time, which
    is artificially low because CUDA still holds memory). Uses RLIMIT_DATA so
    any lingering CUDA VA reservations don't trip the cap. Floor of 16 GiB so
    we never starve the rebuild more than CUDA already did.
    """
    cap_env = os.environ.get("CHESS_NN_CPU_FALLBACK_RAM_BYTES", "").strip()
    if cap_env:
        try:
            cap_bytes = int(cap_env)
        except ValueError:
            return
    else:
        total = _total_system_ram_bytes()
        if total is None or total <= 0:
            return
        cap_bytes = max(int(total * 0.6), 16 * 1024 * 1024 * 1024)
    if cap_bytes <= 0:
        return
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
    except (ValueError, OSError):
        return
    new_hard = hard if hard != resource.RLIM_INFINITY and hard < cap_bytes else cap_bytes
    try:
        resource.setrlimit(resource.RLIMIT_DATA, (cap_bytes, new_hard))
    except (ValueError, OSError) as exc:
        print(f"[oom-fallback] could not set RLIMIT_DATA={cap_bytes}: {exc}", file=sys.stderr)
        return
    gib = cap_bytes / (1024 ** 3)
    print(f"[oom-fallback] RLIMIT_DATA capped at {cap_bytes} bytes ({gib:.1f} GiB) for CPU fallback", file=sys.stderr)


def train_from_config(config: dict[str, Any]) -> Path:
    requested_device = str(config.get("device", "auto")).strip().lower()
    cpu_only = requested_device == "cpu"
    fallback_enabled, fallback_source = _cpu_oom_fallback_policy(config)
    oom_reason: Exception | None = None
    try:
        return Trainer(config).fit()
    except Exception as exc:
        if cpu_only or not _is_cuda_oom(exc):
            raise
        if not fallback_enabled:
            disabled_reason = (
                f"{CPU_OOM_FALLBACK_DISABLE_ENV} is set"
                if fallback_source == CPU_OOM_FALLBACK_DISABLE_ENV
                else "CPU fallback is disabled by default"
            )
            print(
                "[oom-fallback] CUDA OOM; failing without CPU retry because "
                f"{disabled_reason}. Set training.allow_cpu_oom_fallback: true or "
                f"{CPU_OOM_FALLBACK_ENV}=1 to opt in. Any retry is labeled "
                f"{CPU_OOM_FALLBACK_LABEL}.",
                file=sys.stderr,
                flush=True,
            )
            raise
        print(
            f"[oom-fallback] CUDA out of memory: {exc}. Retrying on CPU as {CPU_OOM_FALLBACK_LABEL} "
            f"(opt-in via {fallback_source}).",
            file=sys.stderr,
            flush=True,
        )
        _release_cuda_memory()
        oom_reason = exc
    _apply_cpu_fallback_heap_cap()
    _maximize_cpu_threads()
    cpu_config = _force_cpu_config(
        config,
        original_requested_device=config.get("device", "auto"),
        oom_reason=oom_reason if oom_reason is not None else RuntimeError("CUDA out of memory"),
        enabled_by=fallback_source,
    )
    return Trainer(cpu_config).fit()

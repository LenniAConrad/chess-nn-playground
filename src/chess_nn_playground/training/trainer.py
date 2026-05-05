from __future__ import annotations

import json
import math
import os
import hashlib
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from chess_nn_playground.data.dataset import BINARY_MODES, PUZZLE_BINARY, ChessPositionDataset, collate_positions
from chess_nn_playground.data.board_features import SIMPLE_18
from chess_nn_playground.evaluation.plots import (
    plot_calibration,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_curves,
    plot_rectangular_confusion_matrix,
)
from chess_nn_playground.evaluation.reports import build_run_report
from chess_nn_playground.evaluation.slices import write_slice_artifacts
from chess_nn_playground.models.cnn import count_parameters, model_summary_text
from chess_nn_playground.models.complexity import estimate_model_complexity
from chess_nn_playground.models.registry import build_model
from chess_nn_playground.training.callbacks import EarlyStopping
from chess_nn_playground.training.checkpointing import load_checkpoint, save_checkpoint
from chess_nn_playground.training.device import resolve_torch_device
from chess_nn_playground.training.losses import (
    ConditionalSurprisalGateLoss,
    ContaminationDROHuberTailLoss,
    DykstraLCPLoss,
    DykstraVetoSelectLoss,
    MaterialLockedDROLoss,
    SRPALoss,
    SoftSortOrderResidualLoss,
    VetoSelectLoss,
    binary_cross_entropy_loss,
    cross_entropy_loss,
)
from chess_nn_playground.training.metrics import compute_metrics
from chess_nn_playground.utils.config import save_yaml
from chess_nn_playground.utils.env import collect_environment
from chess_nn_playground.utils.logging import write_json
from chess_nn_playground.utils.paths import ensure_dir, utc_timestamp
from chess_nn_playground.utils.seed import set_seed


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "run"


def _probabilities_from_logits(logits: torch.Tensor, single_logit_binary: bool) -> np.ndarray:
    if single_logit_binary:
        puzzle_prob = torch.sigmoid(logits.detach().view(-1).cpu())
        return torch.stack([1.0 - puzzle_prob, puzzle_prob], dim=1).numpy()
    return torch.softmax(logits.detach().cpu(), dim=1).numpy()


def _primary_logits(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(output, dict):
        if "selective_puzzle_logit" in output:
            return output["selective_puzzle_logit"]
        if "logits" in output:
            return output["logits"]
        if "puzzle_logit" in output:
            return output["puzzle_logit"]
        raise ValueError(f"Model output dictionary has no usable logits key: {sorted(output)}")
    return output


def _scalar_output_columns(output: torch.Tensor | dict[str, torch.Tensor]) -> dict[str, list[float]]:
    if not isinstance(output, dict):
        return {}
    columns: dict[str, list[float]] = {}
    for key, value in output.items():
        if not isinstance(value, torch.Tensor) or value.ndim == 0:
            continue
        flat = value.detach().cpu().view(value.shape[0], -1)
        if flat.shape[1] == 1:
            columns[key] = [float(item) for item in flat[:, 0].tolist()]
    return columns


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


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return None


def _batch_fine_labels(batch: dict[str, Any], device: torch.device, non_blocking: bool) -> torch.Tensor | None:
    values = batch.get("fine_label")
    if values is None:
        return None
    parsed = [_optional_int(value) for value in values]
    if any(value is None for value in parsed):
        return None
    return torch.tensor([int(value) for value in parsed], dtype=torch.long, device=device)


def _sync_device_for_timing(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _resolve_num_workers(value: Any, device: torch.device) -> int:
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


def _fine_to_binary_matrix(predictions: pd.DataFrame) -> list[list[int]] | None:
    required = {"true_fine_label", "predicted_label"}
    if predictions.empty or not required.issubset(predictions.columns):
        return None
    matrix = np.zeros((3, 2), dtype=int)
    for fine_label, predicted_label in zip(predictions["true_fine_label"], predictions["predicted_label"]):
        fine = _optional_int(fine_label)
        pred = _optional_int(predicted_label)
        if fine in {0, 1, 2} and pred in {0, 1}:
            matrix[fine, pred] += 1
    if matrix.sum() == 0:
        return None
    return matrix.tolist()


class Trainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.seed = int(config.get("seed", 42))
        set_seed(self.seed, deterministic=bool(config.get("deterministic", True)))
        self.mode = config.get("mode", "coarse_binary")
        self.metric_num_classes = 2 if self.mode in BINARY_MODES else 3
        self.requested_device = config.get("device", "auto")
        self.device = resolve_torch_device(self.requested_device)
        run_cfg = config.get("run", {})
        run_name = _safe_name(run_cfg.get("name", "cnn_baseline"))
        fixed_run_dir = run_cfg.get("run_dir")
        if fixed_run_dir:
            self.run_dir = Path(fixed_run_dir)
        else:
            self.run_dir = Path(run_cfg.get("output_dir", "results")) / f"{utc_timestamp(compact=True)}_{run_name}"
        ensure_dir(self.run_dir)

        data_cfg = config.get("data", {})
        self.train_path = Path(data_cfg.get("train_path", "data/splits/split_train.parquet"))
        self.val_path = Path(data_cfg.get("val_path", "data/splits/split_val.parquet"))
        self.test_path = Path(data_cfg.get("test_path", "data/splits/split_test.parquet"))
        self.input_encoding = data_cfg.get("encoding", SIMPLE_18)

        training_cfg = config.get("training", {})
        self.batch_size = int(training_cfg.get("batch_size", 64))
        self.num_workers = _resolve_num_workers(training_cfg.get("num_workers", "auto"), self.device)
        self.persistent_workers = bool(training_cfg.get("persistent_workers", self.num_workers > 0)) and self.num_workers > 0
        self.prefetch_factor = int(training_cfg.get("prefetch_factor", 2))
        self.epochs = int(training_cfg.get("epochs", 10))
        self.min_epochs = int(training_cfg.get("min_epochs", 0))
        self.min_active_epochs = int(training_cfg.get("min_active_epochs", self.min_epochs))
        self.learning_rate = float(training_cfg.get("learning_rate", 1e-3))
        self.weight_decay = float(training_cfg.get("weight_decay", 0.0))
        clip_value = training_cfg.get("gradient_clip_norm")
        self.gradient_clip_norm = float(clip_value) if clip_value is not None else None
        mixed_precision_cfg = training_cfg.get("mixed_precision", False)
        if str(mixed_precision_cfg).strip().lower() == "auto":
            self.use_amp = self.device.type == "cuda"
        else:
            self.use_amp = bool(mixed_precision_cfg) and self.device.type == "cuda"
        self.pin_memory = bool(training_cfg.get("pin_memory", self.device.type == "cuda"))
        self.allow_tf32 = bool(training_cfg.get("allow_tf32", self.device.type == "cuda"))
        self.matmul_precision = str(training_cfg.get("matmul_precision", "high" if self.device.type == "cuda" else "highest"))
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(self.matmul_precision)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
            torch.backends.cudnn.allow_tf32 = self.allow_tf32
        self.class_weighting = training_cfg.get("class_weighting", "none")
        self.loss_name = str(training_cfg.get("loss", "")).strip().lower()
        self.veto_select_cfg = training_cfg.get("veto_select", {})
        if not isinstance(self.veto_select_cfg, dict):
            self.veto_select_cfg = {}
        self.dykstra_lcp_cfg = training_cfg.get("dykstra_lcp", {})
        if not isinstance(self.dykstra_lcp_cfg, dict):
            self.dykstra_lcp_cfg = {}
        self.dykstra_vetoselect_cfg = training_cfg.get("dykstra_vetoselect", {})
        if not isinstance(self.dykstra_vetoselect_cfg, dict):
            self.dykstra_vetoselect_cfg = {}
        self.srpa_cfg = training_cfg.get("srpa", {})
        if not isinstance(self.srpa_cfg, dict):
            self.srpa_cfg = {}
        self.contamination_dro_cfg = training_cfg.get("contamination_dro", {})
        if not isinstance(self.contamination_dro_cfg, dict):
            self.contamination_dro_cfg = {}
        self.material_locked_dro_cfg = training_cfg.get("material_locked_dro", {})
        if not isinstance(self.material_locked_dro_cfg, dict):
            self.material_locked_dro_cfg = {}
        self.soft_sort_order_cfg = training_cfg.get("soft_sort_order", {})
        if not isinstance(self.soft_sort_order_cfg, dict):
            self.soft_sort_order_cfg = {}
        self.conditional_surprisal_gate_cfg = training_cfg.get("conditional_surprisal_gate", {})
        if not isinstance(self.conditional_surprisal_gate_cfg, dict):
            self.conditional_surprisal_gate_cfg = {}
        self.veto_select_warmup_epochs = int(self.veto_select_cfg.get("warmup_epochs", 1))
        self.use_rule_texture = bool(
            self.veto_select_cfg.get("use_rule_texture", data_cfg.get("include_rule_texture", False))
        )
        self.early_stopping = EarlyStopping(
            patience=int(training_cfg.get("early_stopping_patience", 10)),
            mode="max",
        )

        model_cfg = dict(config.get("model", {}))
        model_name = model_cfg.pop("name", "simple_cnn")
        default_model_classes = 1 if self.mode == PUZZLE_BINARY else self.metric_num_classes
        model_cfg.setdefault("num_classes", default_model_classes)
        self.model_output_classes = int(model_cfg.get("num_classes", default_model_classes))
        self.single_logit_binary = self.mode in BINARY_MODES and self.model_output_classes == 1
        self.model_name = model_name
        self.model = build_model(model_name, model_cfg).to(self.device)
        self.model_input_channels = int(model_cfg.get("input_channels", 18))
        try:
            self.model_complexity = estimate_model_complexity(
                self.model,
                input_channels=self.model_input_channels,
                device=self.device,
            )
        except Exception as exc:
            self.model_complexity = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "model_name": self.model_name,
                "input_shape": [1, self.model_input_channels, 8, 8],
            }
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.scheduler = self._build_scheduler(training_cfg.get("lr_scheduler", {}))
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

        self.train_dataset = ChessPositionDataset(
            self.train_path,
            mode=self.mode,
            cache_features=bool(data_cfg.get("cache_features", False)),
            encoding=self.input_encoding,
            include_rule_texture=self.use_rule_texture,
        )
        self.val_dataset = ChessPositionDataset(
            self.val_path,
            mode=self.mode,
            cache_features=bool(data_cfg.get("cache_features", False)),
            encoding=self.input_encoding,
            include_rule_texture=self.use_rule_texture,
        )
        self.test_dataset = None
        if self.test_path.exists():
            self.test_dataset = ChessPositionDataset(
                self.test_path,
                mode=self.mode,
                cache_features=bool(data_cfg.get("cache_features", False)),
                encoding=self.input_encoding,
                include_rule_texture=self.use_rule_texture,
            )
        if self.loss_name == "veto_select":
            if not self.single_logit_binary:
                raise ValueError("training.loss=veto_select requires a single-logit binary model")
            pos_weight = self._binary_pos_weight() if self.class_weighting == "balanced" else None
            self.criterion = VetoSelectLoss(
                pos_weight=pos_weight,
                tau_e=float(self.veto_select_cfg.get("tau_e", 1.5)),
                d_max=float(self.veto_select_cfg.get("d_max", 0.85)),
                gamma_decoy=float(self.veto_select_cfg.get("gamma_decoy", 1.0)),
                lambda_anchor=float(self.veto_select_cfg.get("lambda_anchor", 0.15)),
            )
        elif self.loss_name == "dykstra_lcp":
            if not self.single_logit_binary:
                raise ValueError("training.loss=dykstra_lcp requires a single-logit binary model")
            pos_weight = self._binary_pos_weight() if self.class_weighting == "balanced" else None
            self.criterion = DykstraLCPLoss(
                pos_weight=pos_weight,
                hard_negative_fraction=float(self.dykstra_lcp_cfg.get("hard_negative_fraction", 0.20)),
                hard_negative_weight=float(self.dykstra_lcp_cfg.get("hard_negative_weight", 1.5)),
                lambda_pos_residual=float(self.dykstra_lcp_cfg.get("lambda_pos_residual", 0.03)),
                lambda_neg_margin=float(self.dykstra_lcp_cfg.get("lambda_neg_margin", 0.02)),
                lambda_decay=float(self.dykstra_lcp_cfg.get("lambda_decay", 0.01)),
                negative_projection_margin=float(self.dykstra_lcp_cfg.get("negative_projection_margin", 0.20)),
            )
        elif self.loss_name == "dykstra_vetoselect":
            if not self.single_logit_binary:
                raise ValueError("training.loss=dykstra_vetoselect requires a single-logit binary model")
            pos_weight = self._binary_pos_weight() if self.class_weighting == "balanced" else None
            self.criterion = DykstraVetoSelectLoss(
                pos_weight=pos_weight,
                tau_e=float(self.veto_select_cfg.get("tau_e", 1.5)),
                d_max=float(self.veto_select_cfg.get("d_max", 0.85)),
                gamma_decoy=float(self.veto_select_cfg.get("gamma_decoy", 1.0)),
                lambda_anchor=float(self.veto_select_cfg.get("lambda_anchor", 0.12)),
                projection_temperature=float(self.dykstra_vetoselect_cfg.get("projection_temperature", 0.04)),
                trace_temperature=float(self.dykstra_vetoselect_cfg.get("trace_temperature", 0.006)),
                lambda_pos_residual=float(self.dykstra_vetoselect_cfg.get("lambda_pos_residual", 0.02)),
                lambda_neg_margin=float(self.dykstra_vetoselect_cfg.get("lambda_neg_margin", 0.01)),
                lambda_decay=float(self.dykstra_vetoselect_cfg.get("lambda_decay", 0.01)),
                negative_projection_margin=float(self.dykstra_vetoselect_cfg.get("negative_projection_margin", 0.04)),
            )
        elif self.loss_name == "srpa":
            if not self.single_logit_binary:
                raise ValueError("training.loss=srpa requires a single-logit binary model")
            pos_weight = self._binary_pos_weight() if self.class_weighting == "balanced" else None
            self.criterion = SRPALoss(
                pos_weight=pos_weight,
                lambda_aux=float(self.srpa_cfg.get("lambda_aux", 0.15)),
                lambda_residual=float(self.srpa_cfg.get("lambda_residual", 0.02)),
                lambda_l1=float(self.srpa_cfg.get("lambda_l1", 0.001)),
                lambda_group=float(self.srpa_cfg.get("lambda_group", 0.001)),
                lambda_dictionary_coherence=float(self.srpa_cfg.get("lambda_dictionary_coherence", 0.0005)),
                lambda_branch_separation=float(self.srpa_cfg.get("lambda_branch_separation", 0.0005)),
                lambda_dead_group=float(self.srpa_cfg.get("lambda_dead_group", 0.0001)),
            )
        elif self.loss_name == "contamination_dro_huber":
            if not self.single_logit_binary:
                raise ValueError("training.loss=contamination_dro_huber requires a single-logit binary model")
            pos_weight = self._binary_pos_weight() if self.class_weighting == "balanced" else None
            self.criterion = ContaminationDROHuberTailLoss(
                pos_weight=pos_weight,
                lambda_tail=float(self.contamination_dro_cfg.get("lambda_tail", 0.35)),
                margin=float(self.contamination_dro_cfg.get("margin", 0.25)),
                kappa=float(self.contamination_dro_cfg.get("kappa", 1.0)),
                beta=float(self.contamination_dro_cfg.get("beta", 0.25)),
                min_near_count=int(self.contamination_dro_cfg.get("min_near_count", 4)),
            )
        elif self.loss_name == "material_locked_dro":
            if not self.single_logit_binary:
                raise ValueError("training.loss=material_locked_dro requires a single-logit binary model")
            pos_weight = self._binary_pos_weight() if self.class_weighting == "balanced" else None
            self.criterion = MaterialLockedDROLoss(
                pos_weight=pos_weight,
                gamma_near=float(self.material_locked_dro_cfg.get("gamma_near", 2.0)),
                lambda_robust=float(self.material_locked_dro_cfg.get("lambda_robust", 0.5)),
                lambda_budget=float(self.material_locked_dro_cfg.get("lambda_budget", 0.02)),
            )
        elif self.loss_name == "soft_sort_order":
            if not self.single_logit_binary:
                raise ValueError("training.loss=soft_sort_order requires a single-logit binary model")
            pos_weight = self._binary_pos_weight() if self.class_weighting == "balanced" else None
            self.criterion = SoftSortOrderResidualLoss(
                pos_weight=pos_weight,
                lambda_order=float(self.soft_sort_order_cfg.get("lambda_order", 0.25)),
                tau=float(self.soft_sort_order_cfg.get("tau", 0.25)),
            )
        elif self.loss_name == "conditional_surprisal_gate":
            if not self.single_logit_binary:
                raise ValueError("training.loss=conditional_surprisal_gate requires a single-logit binary model")
            pos_weight = self._binary_pos_weight() if self.class_weighting == "balanced" else None
            self.criterion = ConditionalSurprisalGateLoss(
                pos_weight=pos_weight,
                lambda_kl=float(self.conditional_surprisal_gate_cfg.get("lambda_kl", 0.05)),
                lambda_capacity=float(self.conditional_surprisal_gate_cfg.get("lambda_capacity", 0.05)),
                target_gate_rate=float(self.conditional_surprisal_gate_cfg.get("target_gate_rate", 0.35)),
            )
        elif self.single_logit_binary:
            pos_weight = self._binary_pos_weight() if self.class_weighting == "balanced" else None
            self.criterion = binary_cross_entropy_loss(pos_weight)
        else:
            class_weights = self._class_weights() if self.class_weighting == "balanced" else None
            self.criterion = cross_entropy_loss(class_weights)
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
        labels = self.train_dataset.df[self.train_dataset.label_column].astype(int).to_numpy()
        counts = np.bincount(labels, minlength=self.metric_num_classes)
        total = counts.sum()
        weights = [total / (self.metric_num_classes * count) if count > 0 else 0.0 for count in counts]
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def _binary_pos_weight(self) -> torch.Tensor:
        labels = self.train_dataset.df[self.train_dataset.label_column].astype(int).to_numpy()
        counts = np.bincount(labels, minlength=2)
        negative = float(counts[0])
        positive = float(counts[1])
        value = negative / positive if positive > 0 else 1.0
        return torch.tensor([value], dtype=torch.float32, device=self.device)

    def _batch_rule_texture(self, batch: dict[str, Any]) -> torch.Tensor | None:
        texture = batch.get("rule_texture")
        if texture is None:
            return None
        return texture.to(self.device, non_blocking=self.pin_memory)

    def _loader(self, dataset: ChessPositionDataset, shuffle: bool) -> DataLoader:
        loader_kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "collate_fn": collate_positions,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = self.persistent_workers
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(
            dataset,
            **loader_kwargs,
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
        if self.loss_name == "veto_select":
            if not isinstance(output, dict):
                raise ValueError("training.loss=veto_select requires model output diagnostics")
            enable_decoys = epoch is None or epoch > self.veto_select_warmup_epochs
            return self.criterion(output, y, enable_decoys=enable_decoys, texture=texture)
        if self.loss_name == "dykstra_lcp":
            if not isinstance(output, dict):
                raise ValueError("training.loss=dykstra_lcp requires model output diagnostics")
            return self.criterion(output, y)
        if self.loss_name == "dykstra_vetoselect":
            if not isinstance(output, dict):
                raise ValueError("training.loss=dykstra_vetoselect requires model output diagnostics")
            enable_decoys = epoch is None or epoch > self.veto_select_warmup_epochs
            return self.criterion(output, y, enable_decoys=enable_decoys, texture=texture)
        if self.loss_name == "srpa":
            if not isinstance(output, dict):
                raise ValueError("training.loss=srpa requires model output diagnostics")
            return self.criterion(output, y)
        if self.loss_name == "contamination_dro_huber":
            if not isinstance(output, dict):
                raise ValueError("training.loss=contamination_dro_huber requires model output diagnostics")
            return self.criterion(output, y, fine_label=fine_label)
        if self.loss_name == "material_locked_dro":
            if not isinstance(output, dict):
                raise ValueError("training.loss=material_locked_dro requires model output diagnostics")
            return self.criterion(output, y, fine_label=fine_label)
        if self.loss_name == "soft_sort_order":
            if not isinstance(output, dict):
                raise ValueError("training.loss=soft_sort_order requires model output diagnostics")
            return self.criterion(output, y)
        if self.loss_name == "conditional_surprisal_gate":
            if not isinstance(output, dict):
                raise ValueError("training.loss=conditional_surprisal_gate requires model output diagnostics")
            return self.criterion(output, y)
        if self.single_logit_binary:
            return self.criterion(logits.view(-1), y.float())
        return self.criterion(logits, y)

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
            "train": self.train_dataset.df[self.train_dataset.label_column].value_counts().to_dict(),
            "val": self.val_dataset.df[self.val_dataset.label_column].value_counts().to_dict(),
        }
        source_class_counts = {
            "train": self.train_dataset.df.get("fine_label", pd.Series(dtype=int)).value_counts().sort_index().to_dict(),
            "val": self.val_dataset.df.get("fine_label", pd.Series(dtype=int)).value_counts().sort_index().to_dict(),
        }
        if self.test_dataset is not None:
            class_counts["test"] = self.test_dataset.df[self.test_dataset.label_column].value_counts().to_dict()
            source_class_counts["test"] = (
                self.test_dataset.df.get("fine_label", pd.Series(dtype=int)).value_counts().sort_index().to_dict()
            )
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
        sample_count = sum(int(row.get("sample_count") or 0) for row in rows)
        batch_count = sum(int(row.get("batch_count") or 0) for row in rows)
        elapsed_seconds = sum(float(row.get("elapsed_seconds") or 0.0) for row in rows)
        return {
            "sample_count": sample_count,
            "batch_count": batch_count,
            "elapsed_seconds": elapsed_seconds,
            "samples_per_second": sample_count / elapsed_seconds if elapsed_seconds > 0 else None,
            "batches_per_second": batch_count / elapsed_seconds if elapsed_seconds > 0 else None,
        }

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
        summary = {
            "fit_elapsed_seconds": float(fit_elapsed_seconds),
            "train": train_speed,
            "validation_epochs": val_speed,
            "final_eval": final_eval,
            "train_samples_per_second": train_speed.get("samples_per_second"),
            "val_samples_per_second": val_speed.get("samples_per_second"),
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
        train_counts = {
            str(k): int(v)
            for k, v in self.train_dataset.df[self.train_dataset.label_column].value_counts().sort_index().items()
        }
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


def train_from_config(config: dict[str, Any]) -> Path:
    return Trainer(config).fit()

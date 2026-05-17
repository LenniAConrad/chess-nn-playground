from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.data.dataset import BINARY_MODES, PUZZLE_BINARY
from chess_nn_playground.models.complexity import estimate_model_complexity
from chess_nn_playground.models.registry import build_model


@dataclass(frozen=True)
class ModelRuntime:
    name: str
    model: nn.Module
    output_classes: int
    single_logit_binary: bool
    input_channels: int
    complexity: dict[str, Any]


def build_model_runtime(
    config: dict[str, Any],
    *,
    mode: str,
    metric_num_classes: int,
    device: torch.device,
) -> ModelRuntime:
    model_cfg = dict(config.get("model", {}) or {})
    model_name = model_cfg.pop("name", "simple_cnn")
    default_model_classes = 1 if mode == PUZZLE_BINARY else metric_num_classes
    model_cfg.setdefault("num_classes", default_model_classes)
    output_classes = int(model_cfg.get("num_classes", default_model_classes))
    single_logit_binary = mode in BINARY_MODES and output_classes == 1
    model = build_model(model_name, model_cfg).to(device)
    input_channels = int(model_cfg.get("input_channels", 18))
    try:
        complexity = estimate_model_complexity(
            model,
            input_channels=input_channels,
            device=device,
        )
    except Exception as exc:
        complexity = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "model_name": model_name,
            "input_shape": [1, input_channels, 8, 8],
        }
    return ModelRuntime(
        name=model_name,
        model=model,
        output_classes=output_classes,
        single_logit_binary=single_logit_binary,
        input_channels=input_channels,
        complexity=complexity,
    )

from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.delta_crelu_involution import (
    DeltaCReLUInvolutionHead,
    build_delta_crelu_involution_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> DeltaCReLUInvolutionHead:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_delta_crelu_involution_from_config(model_cfg)

from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.matrix_pencil_generalized_spectrum_bottleneck import (
    MatrixPencilGeneralizedSpectrumNet,
    build_matrix_pencil_generalized_spectrum_bottleneck_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> MatrixPencilGeneralizedSpectrumNet:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_matrix_pencil_generalized_spectrum_bottleneck_from_config(model_cfg)

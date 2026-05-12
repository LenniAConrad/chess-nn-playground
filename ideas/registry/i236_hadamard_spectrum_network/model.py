from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.hadamard_spectrum import HadamardSpectrumNetwork
from chess_nn_playground.models.trunk.hadamard_spectrum import build_hadamard_spectrum_from_config


def build_model_from_config(config: dict[str, Any]) -> HadamardSpectrumNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_hadamard_spectrum_from_config(model_cfg)

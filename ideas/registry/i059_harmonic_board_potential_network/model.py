from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.harmonic_board_potential_network import (
    HarmonicBoardPotentialNet,
    build_harmonic_board_potential_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> HarmonicBoardPotentialNet:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_harmonic_board_potential_network_from_config(model_cfg)

from __future__ import annotations

from typing import Any

from chess_nn_playground.models.bitboard_shift_algebra import (
    BitboardShiftAlgebraNetwork,
    build_bitboard_shift_algebra_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> BitboardShiftAlgebraNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_bitboard_shift_algebra_network_from_config(model_cfg)

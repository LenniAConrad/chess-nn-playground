from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.pawn_skeleton_barrier import PawnSkeletonBarrierNetwork
from chess_nn_playground.models.trunk.pawn_skeleton_barrier import (
    build_pawn_skeleton_barrier_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> PawnSkeletonBarrierNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_pawn_skeleton_barrier_network_from_config(model_cfg)

from __future__ import annotations

from typing import Any

from chess_nn_playground.models.piece_plane_gated_cnn import PiecePlaneGatedCNN
from chess_nn_playground.models.piece_plane_gated_cnn import build_piece_plane_gated_cnn_from_config


def build_model_from_config(config: dict[str, Any]) -> PiecePlaneGatedCNN:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_piece_plane_gated_cnn_from_config(model_cfg)

from __future__ import annotations

from typing import Any

from torch import nn

from chess_nn_playground.models.trunk.masked_codec_interaction_curvature import build_masked_codec_interaction_curvature_network_from_config


def build_model_from_config(config: dict[str, Any]) -> nn.Module:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_masked_codec_interaction_curvature_network_from_config(model_cfg)

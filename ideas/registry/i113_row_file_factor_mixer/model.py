from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.row_file_factor_mixer import (
    RowFileFactorMixerNetwork,
    build_row_file_factor_mixer_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> RowFileFactorMixerNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_row_file_factor_mixer_from_config(model_cfg)

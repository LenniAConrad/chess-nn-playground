from __future__ import annotations

from typing import Any

from chess_nn_playground.models.hall_defect_zeta import HallDefectZetaConvLite
from chess_nn_playground.models.hall_defect_zeta import build_hall_defect_zeta_operator_from_config


def build_model_from_config(config: dict[str, Any]) -> HallDefectZetaConvLite:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_hall_defect_zeta_operator_from_config(model_cfg)

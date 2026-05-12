from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.hall_defect_obligation_matroid import HallDefectObligationMatroidNet
from chess_nn_playground.models.trunk.hall_defect_obligation_matroid import (
    build_hall_defect_obligation_matroid_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> HallDefectObligationMatroidNet:
    return build_hall_defect_obligation_matroid_network_from_config(config)


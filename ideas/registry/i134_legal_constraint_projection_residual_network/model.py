from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.legal_constraint_projection_residual_network import (
    LegalConstraintProjectionResidualNetwork,
    build_legal_constraint_projection_residual_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> LegalConstraintProjectionResidualNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_legal_constraint_projection_residual_network_from_config(model_cfg)

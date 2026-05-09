from __future__ import annotations

from typing import Any

from chess_nn_playground.models.sinkhorn_role_assignment_network import (
    SinkhornRoleAssignmentNetwork,
    build_sinkhorn_role_assignment_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> SinkhornRoleAssignmentNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_sinkhorn_role_assignment_network_from_config(model_cfg)

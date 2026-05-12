from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.krylov_tactical_subspace_network import (
    KrylovTacticalSubspaceNetwork,
    build_krylov_tactical_subspace_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> KrylovTacticalSubspaceNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_krylov_tactical_subspace_network_from_config(model_cfg)

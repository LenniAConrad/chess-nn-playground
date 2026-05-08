from __future__ import annotations

from typing import Any

from chess_nn_playground.models.replicator_payoff_piece_dynamics import (
    ReplicatorPayoffPieceDynamicsNetwork,
    build_replicator_payoff_piece_dynamics_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> ReplicatorPayoffPieceDynamicsNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_replicator_payoff_piece_dynamics_from_config(model_cfg)

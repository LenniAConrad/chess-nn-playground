from __future__ import annotations

from typing import Any

from chess_nn_playground.models.geometry_pseudolikelihood_ratio import GeometryPseudoLikelihoodRatioNet
from chess_nn_playground.models.geometry_pseudolikelihood_ratio import (
    build_geometry_conditioned_board_pseudo_likelihood_ratio_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> GeometryPseudoLikelihoodRatioNet:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_geometry_conditioned_board_pseudo_likelihood_ratio_network_from_config(model_cfg)

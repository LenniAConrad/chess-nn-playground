from __future__ import annotations

from typing import Any

from chess_nn_playground.models.soft_formal_concept_closure import (
    SoftFormalConceptClosureNet,
    build_soft_formal_concept_closure_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> SoftFormalConceptClosureNet:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_soft_formal_concept_closure_network_from_config(model_cfg)

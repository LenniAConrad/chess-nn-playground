from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.ordinal_evidence_ladder import (
    OrdinalEvidenceLadderNet,
    build_ordinal_evidence_ladder_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> OrdinalEvidenceLadderNet:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_ordinal_evidence_ladder_network_from_config(model_cfg)

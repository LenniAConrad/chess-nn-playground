from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.king_zone_evidence_ledger import (
    KingZoneEvidenceLedger,
    build_king_zone_evidence_ledger_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> KingZoneEvidenceLedger:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_king_zone_evidence_ledger_from_config(model_cfg)

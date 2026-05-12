from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.disproof_ledger_puzzle_network import (
    DisproofLedgerPuzzleNetwork,
    build_disproof_ledger_puzzle_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> DisproofLedgerPuzzleNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_disproof_ledger_puzzle_network_from_config(model_cfg)

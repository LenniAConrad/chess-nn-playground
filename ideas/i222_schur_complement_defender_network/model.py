from __future__ import annotations

from typing import Any

from chess_nn_playground.models.research_packet_probe import ResearchPacketProbe
from chess_nn_playground.models.research_packet_probe import build_research_packet_probe_from_config


def build_model_from_config(config: dict[str, Any]) -> ResearchPacketProbe:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_research_packet_probe_from_config(model_cfg)

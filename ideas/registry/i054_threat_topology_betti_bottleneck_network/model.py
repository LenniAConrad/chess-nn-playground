from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.threat_topology_betti import ThreatTopologyBettiNet
from chess_nn_playground.models.trunk.threat_topology_betti import (
    build_threat_topology_betti_bottleneck_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> ThreatTopologyBettiNet:
    return build_threat_topology_betti_bottleneck_network_from_config(config)

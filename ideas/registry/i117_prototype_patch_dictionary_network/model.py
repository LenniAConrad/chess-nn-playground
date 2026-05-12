from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.prototype_patch_dictionary_network import (
    PrototypePatchDictionaryNetwork,
    build_prototype_patch_dictionary_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> PrototypePatchDictionaryNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_prototype_patch_dictionary_network_from_config(model_cfg)

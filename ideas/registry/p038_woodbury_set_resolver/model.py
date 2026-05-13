from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.woodbury_set_resolver import (
    WoodburySetResolver,
    build_woodbury_set_resolver_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> WoodburySetResolver:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_woodbury_set_resolver_from_config(model_cfg)

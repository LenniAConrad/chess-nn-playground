from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.signed_edit_bilinear_memory import (
    SignedEditBilinearMemory,
    build_signed_edit_bilinear_memory_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> SignedEditBilinearMemory:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_signed_edit_bilinear_memory_from_config(model_cfg)

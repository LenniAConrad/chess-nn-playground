from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.octilinear_selective_scan import (
    OctilinearSelectiveScan,
    build_octilinear_selective_scan_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> OctilinearSelectiveScan:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_octilinear_selective_scan_from_config(model_cfg)

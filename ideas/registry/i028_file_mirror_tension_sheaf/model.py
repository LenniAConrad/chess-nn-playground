from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.file_mirror_tension_sheaf import FileMirrorTensionSheafNet
from chess_nn_playground.models.trunk.file_mirror_tension_sheaf import build_file_mirror_tension_sheaf_from_config


def build_model_from_config(config: dict[str, Any]) -> FileMirrorTensionSheafNet:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_file_mirror_tension_sheaf_from_config(model_cfg)

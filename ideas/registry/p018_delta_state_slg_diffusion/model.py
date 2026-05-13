from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.delta_state_slg_diffusion import (
    DeltaStateSLGDiffusionHead,
    build_delta_state_slg_diffusion_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> DeltaStateSLGDiffusionHead:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_delta_state_slg_diffusion_from_config(model_cfg)

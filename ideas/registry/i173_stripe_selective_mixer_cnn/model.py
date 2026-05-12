from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.stripe_selective_mixer_cnn import (
    StripeSelectiveMixerCNN,
    build_stripe_selective_mixer_cnn_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> StripeSelectiveMixerCNN:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_stripe_selective_mixer_cnn_from_config(model_cfg)

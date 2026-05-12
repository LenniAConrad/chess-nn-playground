from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.forcing_certificate_transformer import (
    ForcingCertificateTransformer,
    build_forcing_certificate_transformer_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> ForcingCertificateTransformer:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_forcing_certificate_transformer_from_config(model_cfg)

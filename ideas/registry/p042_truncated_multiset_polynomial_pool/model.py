from __future__ import annotations

from typing import Any

from chess_nn_playground.models.primitives.truncated_multiset_polynomial_pool import (
    TruncatedMultisetPolynomialPool,
    build_truncated_multiset_polynomial_pool_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> TruncatedMultisetPolynomialPool:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_truncated_multiset_polynomial_pool_from_config(model_cfg)

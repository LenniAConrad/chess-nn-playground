from __future__ import annotations

from typing import Any

from chess_nn_playground.models.reply_set_contrastive_transformer import (
    ReplySetContrastiveTransformer,
    build_reply_set_contrastive_transformer_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> ReplySetContrastiveTransformer:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_reply_set_contrastive_transformer_from_config(model_cfg)

from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.multistream_attention_chess_eval import (
    MultistreamAttentionChessEval,
    build_multistream_attention_chess_eval_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> MultistreamAttentionChessEval:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    model_cfg.setdefault("input_channels", 18)
    model_cfg.pop("name", None)
    model_cfg.pop("packet_profile", None)
    model_cfg.pop("mechanism_family", None)
    return build_multistream_attention_chess_eval_from_config(model_cfg)

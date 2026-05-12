from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.typed_hypergraph_motif_grammar import (
    TypedHypergraphMotifGrammarNet,
    build_typed_hypergraph_motif_grammar_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> TypedHypergraphMotifGrammarNet:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_typed_hypergraph_motif_grammar_from_config(model_cfg)

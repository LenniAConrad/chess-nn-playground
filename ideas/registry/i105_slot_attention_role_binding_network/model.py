from __future__ import annotations

from typing import Any

from chess_nn_playground.models.trunk.slot_attention_role_binding_network import (
    SlotAttentionRoleBindingNetwork,
    build_slot_attention_role_binding_network_from_config,
)


def build_model_from_config(config: dict[str, Any]) -> SlotAttentionRoleBindingNetwork:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    return build_slot_attention_role_binding_network_from_config(model_cfg)

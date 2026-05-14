"""Gated-fusion hybrid: i018 oriented tactical sheaf trunk + chosen primitive's logit.

This wires two existing modules in parallel and fuses their logits through a
learned gate. The point is to ask: *given the sheaf trunk is already strong, can
a primitive with non-overlapping math (e.g. an SSM scan or a delta accumulator)
add signal that the sheaf cannot capture on its own?*

Fusion:

    final_logit = sheaf_logit + sigmoid(gate) * primitive_logit

`gate` is a single learned scalar. If sigmoid(gate) trends to 0 during training,
the model is saying the primitive provides no useful complement and the hybrid
collapses to the sheaf alone. If it stays near 0.5-1.0 with improved metrics,
the primitive is contributing real signal.

The sheaf forward dict (diagnostics, energies, gates) is preserved in the output
dict; `logits` is overwritten with the fused logit. We also expose
`sheaf_only_logits`, `primitive_only_logits`, and `hybrid_gate` for diagnostics.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.oriented_tactical_sheaf import (
    build_oriented_tactical_sheaf_from_config,
)


class OrientedSheafPlusPrimitive(nn.Module):
    def __init__(
        self,
        sheaf_config: dict[str, Any],
        primitive_name: str,
        primitive_config: dict[str, Any],
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        from chess_nn_playground.models.registry import build_model

        self.sheaf = build_oriented_tactical_sheaf_from_config(sheaf_config)
        self.primitive = build_model(primitive_name, primitive_config)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.primitive_name = primitive_name

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        sheaf_out = self.sheaf(x)
        prim_out = self.primitive(x)
        sheaf_logits = sheaf_out["logits"]
        prim_logits = prim_out["logits"]
        if prim_logits.dim() > 1 and prim_logits.size(-1) == 1:
            prim_logits = prim_logits.view(-1)
        if sheaf_logits.dim() > 1 and sheaf_logits.size(-1) == 1:
            sheaf_logits = sheaf_logits.view(-1)
        g = torch.sigmoid(self.gate)
        fused = sheaf_logits + g * prim_logits
        out = dict(sheaf_out)
        out["logits"] = fused
        out["hybrid_gate"] = g.expand(x.shape[0])
        out["sheaf_only_logits"] = sheaf_logits
        out["primitive_only_logits"] = prim_logits
        return out


def build_oriented_sheaf_plus_primitive_from_config(
    config: dict[str, Any],
) -> OrientedSheafPlusPrimitive:
    sheaf_cfg = dict(config.get("sheaf", {}))
    prim_holder = dict(config.get("primitive", {}))
    primitive_name = prim_holder.pop("name", None) or config.get("primitive_name")
    if primitive_name is None:
        raise ValueError(
            "oriented_sheaf_plus_primitive config requires primitive.name or primitive_name"
        )
    prim_cfg = prim_holder
    default_inputs = sheaf_cfg.get("input_channels", config.get("input_channels", 18))
    sheaf_cfg.setdefault("input_channels", default_inputs)
    prim_cfg.setdefault("input_channels", default_inputs)
    prim_cfg.setdefault("num_classes", sheaf_cfg.get("num_classes", config.get("num_classes", 1)))
    sheaf_cfg.setdefault("num_classes", prim_cfg["num_classes"])
    gate_init = float(config.get("gate_init", 0.0))
    return OrientedSheafPlusPrimitive(
        sheaf_config=sheaf_cfg,
        primitive_name=primitive_name,
        primitive_config=prim_cfg,
        gate_init=gate_init,
    )

"""Shared trunk-feature extraction for primitives built on the i193 trunk.

This is the same helper that
``src/chess_nn_playground/models/primitives/promotion_aware_head.py``
introduced inline. It is factored out here so the occlusion / blocker /
event-delta primitives (p019..p024) can reuse the i193 dual-stream
pool/joint features without duplicating the trunk-internal wiring.
"""

from __future__ import annotations

import torch

from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    ExchangeThenKingDualStreamNetwork,
)


def trunk_joint_features(
    trunk: ExchangeThenKingDualStreamNetwork,
    board: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replicate i193's joint feature path without re-running the final logit.

    Returns ``(joint, ex_pool, kg_pool)``. The trunk's own forward is still
    called separately for the baseline pass — this helper exists so the
    same pool/joint feature can be reused by primitive heads without
    paying for the trunk's head MLPs again.
    """
    feats = trunk.feature_builder(board)
    if trunk.ablation == "shared_stream_only":
        ex_input = board
        kg_input = board
    else:
        ex_input = torch.cat([board, feats.exchange], dim=1)
        kg_input = torch.cat([board, feats.king], dim=1)
    _, ex_pool = trunk.exchange_encoder(ex_input)
    if trunk.ablation == "shared_stream_only":
        kg_pool = ex_pool
    else:
        _, kg_pool = trunk.king_encoder(kg_input)
    joint = torch.cat([ex_pool, kg_pool, feats.summary], dim=1)
    return joint, ex_pool, kg_pool

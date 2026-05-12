"""Defender Opportunity-Cost Auction Network for idea i210.

Implements the ``defender opportunity-cost auction'' thesis: each
defender token bids on threat tokens; the bid factors in an alternative
duty value (the opportunity cost of being reassigned). A Sinkhorn-style
iterative procedure produces shadow prices and an auction utility that
becomes the puzzle signal. The architecture is materially distinct from
the shared research-packet probe.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from chess_nn_playground.models._packet_bespoke_base import (
    BoardConvStem,
    BoardTensorSpec,
    format_logits,
    require_board_tensor,
)


def _sinkhorn(log_alpha: torch.Tensor, iterations: int) -> torch.Tensor:
    for _ in range(int(iterations)):
        log_alpha = log_alpha - log_alpha.logsumexp(dim=-1, keepdim=True)
        log_alpha = log_alpha - log_alpha.logsumexp(dim=-2, keepdim=True)
    return log_alpha


class OpportunityCostAuction(nn.Module):
    def __init__(self, channels: int, num_threats: int, num_defenders: int, sinkhorn_iters: int) -> None:
        super().__init__()
        self.num_threats = int(num_threats)
        self.num_defenders = int(num_defenders)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.threat_query = nn.Parameter(torch.randn(num_threats, channels) * 0.02)
        self.defender_query = nn.Parameter(torch.randn(num_defenders, channels) * 0.02)
        self.threat_value = nn.Linear(channels, channels)
        self.defender_value = nn.Linear(channels, channels)
        self.alt_duty = nn.Linear(channels, num_defenders)
        self.bid_temp = nn.Parameter(torch.tensor(0.0))

    def _attend(self, tokens: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        attn_logits = torch.einsum("bsc,kc->bks", tokens, queries)
        attn = F.softmax(attn_logits, dim=-1)
        return attn, torch.einsum("bks,bsc->bkc", attn, tokens)

    def forward(self, board_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        threat_attn, threats = self._attend(board_tokens, self.threat_query)
        defender_attn, defenders = self._attend(board_tokens, self.defender_query)
        threat_v = self.threat_value(threats)
        defender_v = self.defender_value(defenders)
        bids = (threat_v.unsqueeze(2) * defender_v.unsqueeze(1)).sum(dim=-1)
        opportunity_cost = self.alt_duty(board_tokens).mean(dim=1)
        cost_matrix = bids - opportunity_cost.unsqueeze(1)
        log_assign = cost_matrix * F.softplus(self.bid_temp)
        log_assign = _sinkhorn(log_assign, self.sinkhorn_iters)
        assignment = log_assign.exp()
        utility_per_threat = (assignment * cost_matrix).sum(dim=-1)
        shadow_prices = log_assign.logsumexp(dim=-1)
        defender_load = assignment.sum(dim=-2)
        auction_total = utility_per_threat.sum(dim=-1)
        unmet = (1.0 - assignment.sum(dim=-1)).clamp_min(0.0).sum(dim=-1)
        return {
            "assignment": assignment,
            "utility_per_threat": utility_per_threat,
            "shadow_prices": shadow_prices,
            "defender_load": defender_load,
            "opportunity_cost": opportunity_cost,
            "auction_total_value": auction_total,
            "unmet_demand": unmet,
            "threat_attention": threat_attn,
            "defender_attention": defender_attn,
        }


class DefenderOpportunityCostAuctionNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        num_threats: int = 6,
        num_defenders: int = 6,
        sinkhorn_iters: int = 4,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("DefenderOpportunityCostAuctionNetwork supports the puzzle_binary one-logit contract")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.stem = BoardConvStem(input_channels, channels, depth=depth, use_batchnorm=use_batchnorm)
        self.auction = OpportunityCostAuction(channels, num_threats, num_defenders, sinkhorn_iters)
        head_in = num_threats + num_defenders + 4
        self.head_norm = nn.LayerNorm(head_in)
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feats = self.stem(x)
        tokens = feats.flatten(2).transpose(1, 2)
        out = self.auction(tokens)
        readout = torch.cat(
            [
                out["utility_per_threat"],
                out["defender_load"],
                out["shadow_prices"].mean(dim=1, keepdim=True),
                out["auction_total_value"].unsqueeze(-1),
                out["unmet_demand"].unsqueeze(-1),
                out["opportunity_cost"].mean(dim=1, keepdim=True),
            ],
            dim=1,
        )
        readout = self.head_norm(readout)
        logits = format_logits(self.head(readout), self.num_classes)
        return {
            "logits": logits,
            "shadow_prices": out["shadow_prices"],
            "auction_total_value": out["auction_total_value"],
            "defender_load": out["defender_load"],
            "unmet_demand": out["unmet_demand"],
            "opportunity_cost_mean": out["opportunity_cost"].mean(dim=1),
            "utility_per_threat": out["utility_per_threat"],
        }


def build_defender_opportunity_cost_auction_network_from_config(config: dict[str, Any]) -> DefenderOpportunityCostAuctionNetwork:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    return DefenderOpportunityCostAuctionNetwork(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        num_threats=int(cfg.get("num_threats", 6)),
        num_defenders=int(cfg.get("num_defenders", 6)),
        sinkhorn_iters=int(cfg.get("sinkhorn_iters", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
    )

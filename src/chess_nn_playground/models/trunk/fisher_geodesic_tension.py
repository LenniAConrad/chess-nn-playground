"""Fisher-Geodesic Tension Network for idea i083."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.view(-1) if num_classes == 1 else logits


def simplex_floor(p: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    n = p.shape[-1]
    return (1.0 - n * eps) * p + eps


def fisher_rao_distance(p: torch.Tensor, q: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    p32 = p.float().clamp_min(eps)
    q32 = q.float().clamp_min(eps)
    coefficient = torch.sqrt(p32 * q32).sum(dim=-1)
    coefficient = coefficient.clamp(min=-1.0 + eps, max=1.0 - eps)
    return 2.0 * torch.acos(coefficient)


def fisher_geodesic_excess(
    p: torch.Tensor,
    hinge: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    d_ph = fisher_rao_distance(p, hinge, eps)
    d_hq = fisher_rao_distance(hinge, q, eps)
    d_pq = fisher_rao_distance(p, q, eps)
    return d_ph + d_hq - d_pq, d_ph, d_hq, d_pq


def sphere_log(base: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    base = F.normalize(base.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    dot = (base * target).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(dot)
    direction = target - dot * base
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(eps)
    return theta * direction


def hinge_turn(p: torch.Tensor, hinge: torch.Tensor, q: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    u_p = torch.sqrt(p.float().clamp_min(eps))
    u_h = torch.sqrt(hinge.float().clamp_min(eps))
    u_q = torch.sqrt(q.float().clamp_min(eps))
    v_hp = sphere_log(u_h, u_p, eps)
    v_hq = sphere_log(u_h, u_q, eps)
    denom = v_hp.norm(dim=-1).clamp_min(eps) * v_hq.norm(dim=-1).clamp_min(eps)
    cos_angle = (v_hp * v_hq).sum(dim=-1) / denom
    angle = torch.acos(cos_angle.clamp(-1.0 + eps, 1.0 - eps))
    return torch.pi - angle


class ResidualBlock(nn.Module):
    def __init__(self, width: int, use_batchnorm: bool = True) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=not use_batchnorm),
            nn.BatchNorm2d(width) if use_batchnorm else nn.GroupNorm(1, width),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=not use_batchnorm),
            nn.BatchNorm2d(width) if use_batchnorm else nn.GroupNorm(1, width),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class FisherGeodesicTensionNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        width: int = 96,
        depth: int = 5,
        routes: int = 8,
        eps: float = 1.0e-6,
        use_angle: bool = True,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("FisherGeodesicTensionNet supports the puzzle_binary one-logit contract")
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.num_classes = int(num_classes)
        self.width = int(width)
        self.routes = int(routes)
        self.eps = float(eps)
        self.use_angle = bool(use_angle)

        self.stem = nn.Sequential(
            nn.Conv2d(int(input_channels), self.width, kernel_size=3, padding=1, bias=not use_batchnorm),
            nn.BatchNorm2d(self.width) if use_batchnorm else nn.GroupNorm(1, self.width),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(self.width, use_batchnorm=use_batchnorm) for _ in range(max(1, int(depth)))]
        )
        self.route_head = nn.Conv2d(self.width, self.routes * 3, kernel_size=1)
        self.route_gate = nn.Sequential(
            nn.LayerNorm(self.width),
            nn.Linear(self.width, self.routes),
        )
        geom_dim = self.routes * 5 + 4
        if self.use_angle:
            geom_dim += self.routes + 2
        self.geom_dim = geom_dim
        self.readout = nn.Sequential(
            nn.LayerNorm(self.width + geom_dim),
            nn.Linear(self.width + geom_dim, self.width),
            nn.SiLU(inplace=True),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(self.width, max(1, self.width // 2)),
            nn.SiLU(inplace=True),
            nn.Linear(max(1, self.width // 2), 1),
        )
        self.geometry_only_head = nn.Sequential(
            nn.LayerNorm(geom_dim),
            nn.Linear(geom_dim, self.width),
            nn.SiLU(inplace=True),
            nn.Linear(self.width, 1),
        )

    def forward(self, x: torch.Tensor, *, return_aux: bool = False) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        features = self.blocks(self.stem(board))
        pooled = features.mean(dim=(2, 3))
        batch = board.shape[0]

        route_logits = self.route_head(features).view(batch, self.routes, 3, 64)
        probs = torch.softmax(route_logits.float(), dim=-1)
        probs = simplex_floor(probs, self.eps)
        source = probs[:, :, 0, :]
        hinge = probs[:, :, 1, :]
        sink = probs[:, :, 2, :]
        excess, d_ph, d_hq, d_pq = fisher_geodesic_excess(source, hinge, sink, self.eps)
        ratio = excess / (d_pq + self.eps)

        route_gate = torch.softmax(self.route_gate(pooled), dim=-1)
        weighted_excess = (route_gate * excess.to(dtype=route_gate.dtype)).sum(dim=-1, keepdim=True)
        max_excess = excess.max(dim=-1, keepdim=True).values
        weighted_ratio = (route_gate * ratio.to(dtype=route_gate.dtype)).sum(dim=-1, keepdim=True)
        max_ratio = ratio.max(dim=-1, keepdim=True).values
        geom_parts = [
            excess,
            ratio,
            weighted_excess.float(),
            max_excess,
            weighted_ratio.float(),
            max_ratio,
            d_ph,
            d_hq,
            d_pq,
        ]

        turn = None
        if self.use_angle:
            turn = hinge_turn(source, hinge, sink, self.eps)
            weighted_turn = (route_gate * turn.to(dtype=route_gate.dtype)).sum(dim=-1, keepdim=True)
            max_turn = turn.max(dim=-1, keepdim=True).values
            geom_parts.extend([turn, weighted_turn.float(), max_turn])
        geom_feat = torch.cat(geom_parts, dim=-1)
        readout_input = torch.cat([pooled, geom_feat.to(dtype=pooled.dtype)], dim=-1)
        logits = _format_logits(self.readout(readout_input), self.num_classes)
        geometry_logits = _format_logits(self.geometry_only_head(geom_feat.to(dtype=pooled.dtype)), self.num_classes)

        output = {
            "logits": logits,
            "geometry_only_logits": geometry_logits,
            "route_gate": route_gate,
            "route_excess": excess,
            "direct_distance": d_pq,
            "route_ratio": ratio,
            "route_probs": probs,
            "geometry_features": geom_feat,
            "weighted_excess": weighted_excess.view(-1),
            "max_excess": max_excess.view(-1),
            "weighted_ratio": weighted_ratio.view(-1),
            "max_ratio": max_ratio.view(-1),
            "fisher_geodesic_tension": weighted_excess.view(-1),
            "information_surprisal": excess.mean(dim=1),
            "mechanism_energy": geom_feat.pow(2).mean(dim=1),
            "proposal_profile_strength": route_gate.max(dim=1).values,
            "proposal_keyword_count": logits.new_full((batch,), 4.0),
        }
        if turn is not None:
            output["hinge_turn"] = turn
            output["weighted_turn"] = (route_gate * turn.to(dtype=route_gate.dtype)).sum(dim=-1)
            output["max_turn"] = turn.max(dim=-1).values
        return output


def build_fisher_geodesic_tension_network_from_config(config: dict[str, Any]) -> FisherGeodesicTensionNet:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    width = int(cfg.get("width", cfg.get("hidden_dim", cfg.get("channels", 96))))
    return FisherGeodesicTensionNet(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        width=width,
        depth=int(cfg.get("fisher_depth", cfg.get("depth", 5))),
        routes=int(cfg.get("routes", 8)),
        eps=float(cfg.get("eps", 1.0e-6)),
        use_angle=bool(cfg.get("use_angle", True)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
    )

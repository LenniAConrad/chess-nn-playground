"""Empty-Square Opportunity Network for idea i162."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


OPPORTUNITY_NAMES: tuple[str, ...] = (
    "escape_like",
    "landing_like",
    "blocker_like",
    "promotion_lane_like",
    "king_zone_empty_like",
)


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


@dataclass(frozen=True)
class EmptySquareOpportunityConfig:
    input_channels: int = 18
    num_classes: int = 1
    trunk_width: int = 64
    branch_width: int = 48
    opportunity_channels: int = 8
    depth: int = 4
    fusion_width: int = 96
    hidden_dim: int = 96
    topk_squares: int = 4
    dropout: float = 0.1
    use_batchnorm: bool = True
    use_coordinate_planes: bool = True
    ablation: str = "none"


class CoordinatePlaneAppender(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        coords = torch.linspace(-1.0, 1.0, 8)
        rank = coords.view(1, 1, 8, 1).expand(1, 1, 8, 8)
        file = coords.view(1, 1, 1, 8).expand(1, 1, 8, 8)
        self.register_buffer("coordinate_planes", torch.cat([rank, file], dim=1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        planes = self.coordinate_planes.to(device=x.device, dtype=x.dtype).expand(x.shape[0], -1, -1, -1)
        return torch.cat([x, planes], dim=1)


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float, use_batchnorm: bool) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvStack(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depth: int, dropout: float, use_batchnorm: bool) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive")
        layers: list[nn.Module] = []
        current_channels = in_channels
        for _ in range(depth):
            layers.append(ConvNormAct(current_channels, out_channels, dropout=dropout, use_batchnorm=use_batchnorm))
            current_channels = out_channels
        self.stack = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype)
    denom = weights.sum(dim=(2, 3)).clamp_min(1.0)
    return (values * weights).sum(dim=(2, 3)) / denom


def _masked_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.to(dtype=torch.bool)
    neg_large = torch.finfo(values.dtype).min / 4.0
    masked = torch.where(valid, values, values.new_full((), neg_large))
    out = masked.amax(dim=(2, 3))
    has_value = valid.flatten(2).any(dim=2)
    return torch.where(has_value, out, torch.zeros_like(out))


def _masked_topk_mean(values: torch.Tensor, mask: torch.Tensor, topk: int) -> torch.Tensor:
    valid = mask.to(dtype=torch.bool).flatten(2)
    flat = values.flatten(2)
    neg_large = torch.finfo(values.dtype).min / 4.0
    masked = torch.where(valid, flat, flat.new_full((), neg_large))
    k = min(max(1, int(topk)), flat.shape[-1])
    top_values = torch.topk(masked, k=k, dim=2).values
    top_values = torch.where(top_values == neg_large, torch.zeros_like(top_values), top_values)
    return top_values.mean(dim=2)


def _masked_pool(values: torch.Tensor, mask: torch.Tensor, topk: int) -> torch.Tensor:
    return torch.cat([_masked_mean(values, mask), _masked_max(values, mask), _masked_topk_mean(values, mask, topk)], dim=1)


class EmptySquareOpportunityNetwork(nn.Module):
    """Dual occupied/empty square opportunity classifier."""

    VALID_ABLATIONS = {
        "none",
        "occupied_only",
        "empty_only",
        "random_empty_mask",
        "no_occ_empty_interaction",
        "cnn_matched_params",
    }

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        trunk_width: int = 64,
        branch_width: int = 48,
        opportunity_channels: int = 8,
        depth: int = 4,
        fusion_width: int = 96,
        hidden_dim: int = 96,
        topk_squares: int = 4,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        use_coordinate_planes: bool = True,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if ablation not in self.VALID_ABLATIONS:
            raise ValueError(f"Unknown EmptySquareOpportunityNetwork ablation: {ablation}")
        if trunk_width < 1 or branch_width < 1 or opportunity_channels < 1:
            raise ValueError("trunk_width, branch_width, and opportunity_channels must be positive")
        if topk_squares < 1 or topk_squares > 64:
            raise ValueError("topk_squares must be between 1 and 64")
        self.config = EmptySquareOpportunityConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            trunk_width=trunk_width,
            branch_width=branch_width,
            opportunity_channels=opportunity_channels,
            depth=depth,
            fusion_width=fusion_width,
            hidden_dim=hidden_dim,
            topk_squares=topk_squares,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            use_coordinate_planes=use_coordinate_planes,
            ablation=ablation,
        )
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.ablation = ablation
        self.topk_squares = int(topk_squares)
        self.coordinate_appender = CoordinatePlaneAppender() if use_coordinate_planes else nn.Identity()
        trunk_input_channels = input_channels + (2 if use_coordinate_planes else 0)
        self.trunk = ConvStack(
            in_channels=trunk_input_channels,
            out_channels=trunk_width,
            depth=depth,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )
        self.occupied_branch = ConvStack(
            in_channels=trunk_width,
            out_channels=branch_width,
            depth=depth,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )
        self.empty_branch = ConvStack(
            in_channels=trunk_width,
            out_channels=branch_width,
            depth=depth,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )
        self.opportunity_head = nn.Conv2d(branch_width, opportunity_channels, kernel_size=1)
        self.occupied_projection = nn.Sequential(
            nn.LayerNorm(branch_width * 3),
            nn.Linear(branch_width * 3, fusion_width),
            nn.GELU(),
        )
        self.empty_projection = nn.Sequential(
            nn.LayerNorm(opportunity_channels * 3),
            nn.Linear(opportunity_channels * 3, fusion_width),
            nn.GELU(),
        )
        self.cnn_projection = nn.Sequential(
            nn.LayerNorm(trunk_width * 3),
            nn.Linear(trunk_width * 3, fusion_width * 2),
            nn.GELU(),
        )
        pair_dim = fusion_width * 2 if ablation in {"no_occ_empty_interaction", "cnn_matched_params"} else fusion_width * 4
        self.classifier = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, max(32, hidden_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(max(32, hidden_dim // 2), num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        occ_mask = self._occupied_mask(board)
        empty_mask = 1.0 - occ_mask
        if self.ablation == "random_empty_mask":
            empty_mask = self._random_same_density_mask(empty_mask)
            occ_mask = 1.0 - empty_mask

        h = self.trunk(self.coordinate_appender(board))
        if self.ablation == "cnn_matched_params":
            all_squares = torch.ones_like(occ_mask)
            cnn_z = self.cnn_projection(_masked_pool(h, all_squares, self.topk_squares))
            aggregate = cnn_z
            h_occ = h.new_zeros(h.shape[0], self.config.branch_width, 8, 8)
            h_empty = torch.zeros_like(h_occ)
            opportunity = h.new_zeros(h.shape[0], self.config.opportunity_channels, 8, 8)
            z_occ = cnn_z[:, : self.config.fusion_width]
            z_empty = cnn_z[:, self.config.fusion_width :]
        else:
            h_occ = self.occupied_branch(h * occ_mask)
            h_empty = self.empty_branch(h * empty_mask)
            opportunity = self.opportunity_head(h_empty) * empty_mask
            z_occ = self.occupied_projection(_masked_pool(h_occ, occ_mask, self.topk_squares))
            z_empty = self.empty_projection(_masked_pool(opportunity, empty_mask, self.topk_squares))
            if self.ablation == "occupied_only":
                z_empty = torch.zeros_like(z_empty)
                opportunity = torch.zeros_like(opportunity)
                h_empty = torch.zeros_like(h_empty)
            elif self.ablation == "empty_only":
                z_occ = torch.zeros_like(z_occ)
                h_occ = torch.zeros_like(h_occ)
            if self.ablation == "no_occ_empty_interaction":
                aggregate = torch.cat([z_occ, z_empty], dim=1)
            else:
                aggregate = torch.cat([z_occ, z_empty, z_occ * z_empty, (z_occ - z_empty).abs()], dim=1)

        raw_logits = self.classifier(aggregate)
        logits = _format_logits(raw_logits, self.num_classes)
        opportunity_energy = opportunity.square().mean(dim=1, keepdim=True)
        top_values, top_indices = self._top_opportunity(opportunity_energy, empty_mask)
        output = {
            "logits": logits,
            "opportunity_maps": opportunity,
            "empty_opportunity_norm": (opportunity.square() * empty_mask).sum(dim=(1, 2, 3))
            / empty_mask.sum(dim=(1, 2, 3)).clamp_min(1.0),
            "occupied_branch_norm": (h_occ.square() * occ_mask).sum(dim=(1, 2, 3))
            / occ_mask.sum(dim=(1, 2, 3)).clamp_min(1.0),
            "empty_branch_norm": (h_empty.square() * empty_mask).sum(dim=(1, 2, 3))
            / empty_mask.sum(dim=(1, 2, 3)).clamp_min(1.0),
            "occ_empty_interaction_energy": (z_occ * z_empty).square().mean(dim=1),
            "occ_empty_gap": (z_occ - z_empty).abs().mean(dim=1),
            "opportunity_top_value": top_values,
            "top_opportunity_square": top_indices.to(dtype=logits.dtype),
            "occupancy_count": occ_mask.sum(dim=(1, 2, 3)),
            "empty_count": empty_mask.sum(dim=(1, 2, 3)),
            "occupancy_fraction": occ_mask.mean(dim=(1, 2, 3)),
            "aggregate_feature_energy": aggregate.square().mean(dim=1),
        }
        output.update(self._named_opportunity_means(opportunity, empty_mask, logits))
        if self.num_classes == 1:
            output["prob"] = torch.sigmoid(logits)
        return output

    def _occupied_mask(self, board: torch.Tensor) -> torch.Tensor:
        piece_planes = board[:, :12].clamp_min(0.0)
        return (piece_planes.amax(dim=1, keepdim=True) >= 0.5).to(dtype=board.dtype)

    def _random_same_density_mask(self, empty_mask: torch.Tensor) -> torch.Tensor:
        batch_size = empty_mask.shape[0]
        flat = empty_mask.new_zeros(batch_size, 1, 64)
        counts = empty_mask.flatten(2).sum(dim=2).round().long().clamp(0, 64)
        for batch_index in range(batch_size):
            count = int(counts[batch_index, 0].item())
            if count:
                indices = torch.randperm(64, device=empty_mask.device)[:count]
                flat[batch_index, 0, indices] = 1.0
        return flat.view_as(empty_mask)

    def _top_opportunity(self, opportunity_energy: torch.Tensor, empty_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = empty_mask.flatten(2).to(dtype=torch.bool)
        flat = opportunity_energy.flatten(2)
        neg_large = torch.finfo(flat.dtype).min / 4.0
        masked = torch.where(valid, flat, flat.new_full((), neg_large))
        values, indices = masked.max(dim=2)
        has_value = valid.any(dim=2)
        values = torch.where(has_value, values, torch.zeros_like(values))
        indices = torch.where(has_value, indices, torch.zeros_like(indices))
        return values.squeeze(1), indices.squeeze(1)

    def _named_opportunity_means(
        self,
        opportunity: torch.Tensor,
        empty_mask: torch.Tensor,
        logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        values: dict[str, torch.Tensor] = {}
        for index, name in enumerate(OPPORTUNITY_NAMES):
            if index < opportunity.shape[1]:
                channel = opportunity[:, index : index + 1]
                values[f"{name}_opportunity"] = (channel * empty_mask).sum(dim=(1, 2, 3)) / empty_mask.sum(
                    dim=(1, 2, 3)
                ).clamp_min(1.0)
            else:
                values[f"{name}_opportunity"] = torch.zeros_like(logits)
        return values


def build_empty_square_opportunity_network_from_config(config: dict[str, Any]) -> EmptySquareOpportunityNetwork:
    trunk_width = int(config.get("trunk_width", config.get("channels", 64)))
    hidden_dim = int(config.get("hidden_dim", 96))
    return EmptySquareOpportunityNetwork(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        trunk_width=trunk_width,
        branch_width=int(config.get("branch_width", max(1, trunk_width * 3 // 4))),
        opportunity_channels=int(config.get("opportunity_channels", 8)),
        depth=int(config.get("depth", 4)),
        fusion_width=int(config.get("fusion_width", hidden_dim)),
        hidden_dim=hidden_dim,
        topk_squares=int(config.get("topk_squares", 4)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        use_coordinate_planes=bool(config.get("use_coordinate_planes", True)),
        ablation=str(config.get("ablation", "none")),
    )

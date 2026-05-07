"""Specialist-Head CNN implementation for idea i147."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


HEAD_NAMES: tuple[str, ...] = ("global", "center", "edge", "king", "material")


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


class SpecialistConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float, use_batchnorm: bool) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpecialistResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float, use_batchnorm: bool) -> None:
        super().__init__()
        self.body = nn.Sequential(
            SpecialistConvBlock(channels, channels, dropout=dropout, use_batchnorm=use_batchnorm),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm),
            nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity(),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


class SpecialistTrunk(nn.Module):
    def __init__(self, input_channels: int, trunk_width: int, trunk_depth: int, dropout: float, use_batchnorm: bool) -> None:
        super().__init__()
        if trunk_depth < 1:
            raise ValueError("trunk_depth must be >= 1")
        layers: list[nn.Module] = [
            SpecialistConvBlock(input_channels, trunk_width, dropout=dropout, use_batchnorm=use_batchnorm)
        ]
        for _idx in range(trunk_depth - 1):
            layers.append(SpecialistResidualBlock(trunk_width, dropout=dropout, use_batchnorm=use_batchnorm))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SpecialistHead(nn.Module):
    def __init__(self, input_dim: int, head_hidden: int, dropout: float) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(head_hidden, head_hidden),
            nn.GELU(),
        )
        self.logit = nn.Linear(head_hidden, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        head_feature = self.feature(features)
        return head_feature, self.logit(head_feature)


def _fixed_region_masks(random_seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    center = torch.zeros(1, 1, 8, 8, dtype=torch.float32)
    center[:, :, 2:6, 2:6] = 1.0
    edge = torch.zeros_like(center)
    edge[:, :, 0, :] = 1.0
    edge[:, :, 7, :] = 1.0
    edge[:, :, :, 0] = 1.0
    edge[:, :, :, 7] = 1.0

    generator = torch.Generator()
    generator.manual_seed(int(random_seed))
    center_random = torch.zeros(64, dtype=torch.float32)
    center_random[torch.randperm(64, generator=generator)[: int(center.sum().item())]] = 1.0
    edge_random = torch.zeros(64, dtype=torch.float32)
    edge_random[torch.randperm(64, generator=generator)[: int(edge.sum().item())]] = 1.0
    return center, edge, center_random.view(1, 1, 8, 8), edge_random.view(1, 1, 8, 8)


@dataclass(frozen=True)
class SpecialistHeadCNNConfig:
    input_channels: int = 18
    num_classes: int = 1
    trunk_width: int = 64
    trunk_depth: int = 4
    head_hidden: int = 32
    fusion_hidden: int = 64
    dropout: float = 0.1
    use_batchnorm: bool = True
    ablation: str = "none"
    random_mask_seed: int = 147


class SpecialistHeadCNN(nn.Module):
    """Shared CNN trunk with fixed-region, king-zone, and material specialist heads."""

    material_feature_dim = 27
    VALID_ABLATIONS = {
        "none",
        "single_global_head",
        "no_king_head",
        "no_material_head",
        "uniform_logit_average",
        "same_region_random_masks",
    }

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        trunk_width: int = 64,
        trunk_depth: int = 4,
        head_hidden: int = 32,
        fusion_hidden: int = 64,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        ablation: str = "none",
        random_mask_seed: int = 147,
    ) -> None:
        super().__init__()
        if trunk_width < 1 or head_hidden < 1 or fusion_hidden < 1:
            raise ValueError("trunk_width, head_hidden, and fusion_hidden must be positive")
        if ablation not in self.VALID_ABLATIONS:
            raise ValueError(f"Unknown SpecialistHeadCNN ablation: {ablation}")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.ablation = ablation
        self.trunk = SpecialistTrunk(
            input_channels=input_channels,
            trunk_width=trunk_width,
            trunk_depth=trunk_depth,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )

        pooled_dim = 2 * trunk_width
        self.global_head = SpecialistHead(pooled_dim, head_hidden=head_hidden, dropout=dropout)
        self.center_head = SpecialistHead(pooled_dim, head_hidden=head_hidden, dropout=dropout)
        self.edge_head = SpecialistHead(pooled_dim, head_hidden=head_hidden, dropout=dropout)
        self.king_head = SpecialistHead(4 * trunk_width, head_hidden=head_hidden, dropout=dropout)
        self.material_head = SpecialistHead(self.material_feature_dim, head_hidden=head_hidden, dropout=dropout)
        fusion_dim = len(HEAD_NAMES) * head_hidden + len(HEAD_NAMES)
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(fusion_hidden, max(16, fusion_hidden // 2)),
            nn.GELU(),
            nn.Linear(max(16, fusion_hidden // 2), num_classes),
        )

        center, edge, random_center, random_edge = _fixed_region_masks(random_mask_seed)
        self.register_buffer("center_mask", center, persistent=False)
        self.register_buffer("edge_mask", edge, persistent=False)
        self.register_buffer("random_center_mask", random_center, persistent=False)
        self.register_buffer("random_edge_mask", random_edge, persistent=False)
        self.register_buffer("king_kernel", torch.ones(1, 1, 3, 3, dtype=torch.float32), persistent=False)
        self.register_buffer("material_weights", torch.tensor([1.0, 3.0, 3.0, 5.0, 9.0, 0.0]), persistent=False)
        self.config = SpecialistHeadCNNConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            trunk_width=trunk_width,
            trunk_depth=trunk_depth,
            head_hidden=head_hidden,
            fusion_hidden=fusion_hidden,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            ablation=ablation,
            random_mask_seed=random_mask_seed,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        trunk = self.trunk(board)
        batch = board.shape[0]
        zero_feature = trunk.new_zeros(batch, self.config.head_hidden)
        zero_logit = trunk.new_zeros(batch, 1)

        global_feature, global_logit = self.global_head(self._mean_max_pool(trunk))
        center_mask, edge_mask = self._region_masks(trunk)
        center_feature, center_logit = self.center_head(self._masked_mean_max_pool(trunk, center_mask))
        edge_feature, edge_logit = self.edge_head(self._masked_mean_max_pool(trunk, edge_mask))

        king_pooled, king_decoded, own_zone, opponent_zone = self._king_zone_features(board, trunk)
        if self.ablation == "no_king_head":
            king_feature, king_logit = zero_feature, zero_logit
            king_active = trunk.new_zeros(batch, 1)
        else:
            king_feature, king_logit = self.king_head(king_pooled)
            king_active = king_decoded.unsqueeze(1)
            king_feature = king_feature * king_active
            king_logit = king_logit * king_active

        material_features = self._material_features(board)
        if self.ablation == "no_material_head":
            material_feature, material_logit = zero_feature, zero_logit
            material_active = trunk.new_zeros(batch, 1)
        else:
            material_feature, material_logit = self.material_head(material_features)
            material_active = trunk.new_ones(batch, 1)

        if self.ablation == "single_global_head":
            center_feature, center_logit = zero_feature, zero_logit
            edge_feature, edge_logit = zero_feature, zero_logit
            king_feature, king_logit = zero_feature, zero_logit
            material_feature, material_logit = zero_feature, zero_logit
            active = torch.cat(
                [
                    trunk.new_ones(batch, 1),
                    trunk.new_zeros(batch, 1),
                    trunk.new_zeros(batch, 1),
                    trunk.new_zeros(batch, 1),
                    trunk.new_zeros(batch, 1),
                ],
                dim=1,
            )
        else:
            active = torch.cat(
                [
                    trunk.new_ones(batch, 1),
                    trunk.new_ones(batch, 1),
                    trunk.new_ones(batch, 1),
                    king_active,
                    material_active,
                ],
                dim=1,
            )

        head_features = torch.cat([global_feature, center_feature, edge_feature, king_feature, material_feature], dim=1)
        head_logits = torch.cat([global_logit, center_logit, edge_logit, king_logit, material_logit], dim=1) * active
        fusion_input = torch.cat([head_features, head_logits], dim=1)
        learned_logits = self.fusion(fusion_input)
        if self.ablation in {"single_global_head", "uniform_logit_average"}:
            logits = (head_logits * active).sum(dim=1, keepdim=True) / active.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            logits = learned_logits

        logit_abs = head_logits.abs() * active
        logit_share = logit_abs / logit_abs.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        return {
            "logits": _format_logits(logits, self.num_classes),
            "learned_fusion_logit": learned_logits.squeeze(-1),
            "uniform_average_logit": (
                (head_logits * active).sum(dim=1, keepdim=True) / active.sum(dim=1, keepdim=True).clamp_min(1.0)
            ).squeeze(-1),
            "global_head_logit": head_logits[:, 0],
            "center_head_logit": head_logits[:, 1],
            "edge_head_logit": head_logits[:, 2],
            "king_head_logit": head_logits[:, 3],
            "material_head_logit": head_logits[:, 4],
            "active_head_count": active.sum(dim=1),
            "specialist_logit_std": head_logits.std(dim=1, unbiased=False),
            "global_logit_share": logit_share[:, 0],
            "center_logit_share": logit_share[:, 1],
            "edge_logit_share": logit_share[:, 2],
            "king_logit_share": logit_share[:, 3],
            "material_logit_share": logit_share[:, 4],
            "trunk_feature_energy": trunk.square().mean(dim=(1, 2, 3)),
            "global_feature_energy": global_feature.square().mean(dim=1),
            "center_region_energy": self._masked_energy(trunk, center_mask),
            "edge_region_energy": self._masked_energy(trunk, edge_mask),
            "king_zone_decoded": king_decoded,
            "own_king_zone_mass": own_zone.sum(dim=(1, 2, 3)),
            "opponent_king_zone_mass": opponent_zone.sum(dim=(1, 2, 3)),
            "material_balance": material_features[:, 19],
            "material_phase": material_features[:, 20],
            "piece_count_total": material_features[:, 18],
        }

    def _region_masks(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.ablation == "same_region_random_masks":
            center = self.random_center_mask
            edge = self.random_edge_mask
        else:
            center = self.center_mask
            edge = self.edge_mask
        return (
            center.to(device=x.device, dtype=x.dtype).expand(x.shape[0], -1, -1, -1),
            edge.to(device=x.device, dtype=x.dtype).expand(x.shape[0], -1, -1, -1),
        )

    @staticmethod
    def _mean_max_pool(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)

    @staticmethod
    def _masked_mean_max_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask / mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        mean = (x * weights).sum(dim=(2, 3))
        masked = x.masked_fill(mask <= 0, torch.finfo(x.dtype).min / 4.0)
        maximum = masked.amax(dim=(2, 3))
        has_region = mask.sum(dim=(2, 3)) > 0
        maximum = torch.where(has_region, maximum, torch.zeros_like(maximum))
        return torch.cat([mean, maximum], dim=1)

    @staticmethod
    def _masked_energy(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask / mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        return (x.square() * weights).sum(dim=(1, 2, 3))

    def _king_zone_features(
        self, board: torch.Tensor, trunk: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if board.shape[1] < 12:
            zeros = trunk.new_zeros(trunk.shape[0], 1, 8, 8)
            return trunk.new_zeros(trunk.shape[0], 4 * trunk.shape[1]), trunk.new_zeros(trunk.shape[0]), zeros, zeros
        white_king = board[:, 5:6].clamp(0.0, 1.0)
        black_king = board[:, 11:12].clamp(0.0, 1.0)
        white_count = white_king.sum(dim=(1, 2, 3))
        black_count = black_king.sum(dim=(1, 2, 3))
        decoded = white_count.eq(1.0) & black_count.eq(1.0)
        if board.shape[1] > 12:
            white_to_move = board[:, 12].mean(dim=(1, 2)) >= 0.5
        else:
            white_to_move = torch.ones(board.shape[0], device=board.device, dtype=torch.bool)
        own_king = torch.where(white_to_move.view(-1, 1, 1, 1), white_king, black_king)
        opponent_king = torch.where(white_to_move.view(-1, 1, 1, 1), black_king, white_king)
        kernel = self.king_kernel.to(device=board.device, dtype=board.dtype)
        valid = decoded.to(dtype=board.dtype).view(-1, 1, 1, 1)
        own_zone = F.conv2d(own_king, kernel, padding=1).clamp(0.0, 1.0) * valid
        opponent_zone = F.conv2d(opponent_king, kernel, padding=1).clamp(0.0, 1.0) * valid
        pooled = torch.cat(
            [
                self._masked_mean_max_pool(trunk, own_zone),
                self._masked_mean_max_pool(trunk, opponent_zone),
            ],
            dim=1,
        )
        return pooled, decoded.to(dtype=trunk.dtype), own_zone, opponent_zone

    def _material_features(self, board: torch.Tensor) -> torch.Tensor:
        piece_planes = board[:, : min(12, board.shape[1])].clamp(0.0, 1.0)
        if piece_planes.shape[1] < 12:
            pad = board.new_zeros(board.shape[0], 12 - piece_planes.shape[1], 8, 8)
            piece_planes = torch.cat([piece_planes, pad], dim=1)
        counts = piece_planes.sum(dim=(2, 3))
        white = counts[:, :6] / 8.0
        black = counts[:, 6:12] / 8.0
        diff = white - black
        weights = self.material_weights.to(device=board.device, dtype=board.dtype)
        piece_total = counts.sum(dim=1, keepdim=True) / 32.0
        material_balance = ((counts[:, :6] - counts[:, 6:12]) * weights).sum(dim=1, keepdim=True) / 39.0
        material_phase = ((counts[:, :6] + counts[:, 6:12]) * weights).sum(dim=1, keepdim=True) / 78.0
        side_to_move = (
            board[:, 12:13].mean(dim=(2, 3)) if board.shape[1] > 12 else board.new_zeros(board.shape[0], 1)
        )
        castling = (
            board[:, 13:17].mean(dim=(2, 3))
            if board.shape[1] >= 17
            else board.new_zeros(board.shape[0], 4)
        )
        en_passant = (
            board[:, 17:18].amax(dim=(2, 3)) if board.shape[1] >= 18 else board.new_zeros(board.shape[0], 1)
        )
        return torch.cat(
            [white, black, diff, piece_total, material_balance, material_phase, side_to_move, castling, en_passant],
            dim=1,
        )


def build_specialist_head_cnn_from_config(config: dict[str, Any]) -> SpecialistHeadCNN:
    trunk_width = int(config.get("trunk_width", config.get("channels", 64)))
    trunk_depth = int(config.get("trunk_depth", config.get("depth", 4)))
    head_hidden = int(config.get("head_hidden", max(16, int(config.get("hidden_dim", 96)) // 3)))
    fusion_hidden = int(config.get("fusion_hidden", config.get("hidden_dim", 64)))
    return SpecialistHeadCNN(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        trunk_width=trunk_width,
        trunk_depth=trunk_depth,
        head_hidden=head_hidden,
        fusion_hidden=fusion_hidden,
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        ablation=str(config.get("ablation", "none")),
        random_mask_seed=int(config.get("random_mask_seed", 147)),
    )

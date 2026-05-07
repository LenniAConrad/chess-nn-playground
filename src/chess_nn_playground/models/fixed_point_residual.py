"""Fixed-Point Residual Defect Network for idea i097."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.view(-1) if num_classes == 1 else logits


class BoardFixedPointEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 64,
        latent_dim: int = 128,
        board_embed_dim: int = 64,
        depth: int = 2,
        dropout: float = 0.0,
        include_coordinates: bool = True,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.include_coordinates = bool(include_coordinates)
        in_channels = int(input_channels) + (2 if self.include_coordinates else 0)
        width = int(channels)
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.GroupNorm(max(1, min(8, width)), width),
            nn.GELU(),
        ]
        for _ in range(max(0, int(depth) - 1)):
            layers.extend(
                [
                    nn.Conv2d(width, width, kernel_size=3, padding=1),
                    nn.GroupNorm(max(1, min(8, width)), width),
                    nn.GELU(),
                    nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity(),
                ]
            )
        self.trunk = nn.Sequential(*layers)
        pooled_dim = width * 2
        self.latent = nn.Sequential(
            nn.Linear(pooled_dim, int(latent_dim)),
            nn.LayerNorm(int(latent_dim)),
            nn.GELU(),
        )
        self.board_embed = nn.Sequential(
            nn.Linear(pooled_dim, int(board_embed_dim)),
            nn.LayerNorm(int(board_embed_dim)),
            nn.GELU(),
        )
        if self.include_coordinates:
            coords = torch.linspace(-1.0, 1.0, 8)
            rank = coords.view(1, 1, 8, 1).expand(1, 1, 8, 8)
            file = coords.view(1, 1, 1, 8).expand(1, 1, 8, 8)
            self.register_buffer("coord_planes", torch.cat([rank, file], dim=1), persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        if self.include_coordinates:
            coords = self.coord_planes.to(device=board.device, dtype=board.dtype).expand(board.shape[0], -1, -1, -1)
            board = torch.cat([board, coords], dim=1)
        features = self.trunk(board)
        pooled = torch.cat([features.mean(dim=(2, 3)), features.amax(dim=(2, 3))], dim=1)
        return self.latent(pooled), self.board_embed(pooled)


class ResidualUpdateBlock(nn.Module):
    def __init__(self, latent_dim: int = 128, board_embed_dim: int = 64, update_hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(latent_dim) + int(board_embed_dim), int(update_hidden)),
            nn.LayerNorm(int(update_hidden)),
            nn.GELU(),
            nn.Linear(int(update_hidden), int(update_hidden)),
            nn.GELU(),
            nn.Linear(int(update_hidden), int(latent_dim)),
        )

    def forward(self, h: torch.Tensor, board_embed: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h, board_embed], dim=1))


class DefectTrajectoryStats(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        steps: int = 6,
        projection_dim: int = 16,
        include_final_latent: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.steps = max(1, int(steps))
        self.projection_dim = int(projection_dim)
        self.include_final_latent = bool(include_final_latent)
        self.projection = nn.Parameter(torch.randn(self.projection_dim, self.latent_dim) * 0.05)
        per_step = 5 + self.projection_dim
        self.full_output_dim = self.steps * per_step + 8 + (self.latent_dim if self.include_final_latent else 0)
        self.norm_only_dim = self.steps * 3 + 4
        self.final_only_dim = self.latent_dim

    def forward(
        self,
        h_path: torch.Tensor,
        r_path: torch.Tensor,
        final_defect: torch.Tensor,
        *,
        mode: str = "none",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h_final = h_path[:, -1]
        r_l2 = r_path.norm(dim=2)
        r_l1 = r_path.abs().mean(dim=2)
        projection = F.normalize(self.projection.to(device=r_path.device, dtype=r_path.dtype), dim=1)
        r_proj = torch.einsum("btd,pd->btp", r_path, projection)

        if r_path.shape[1] > 1:
            cos = F.cosine_similarity(r_path[:, 1:], r_path[:, :-1], dim=2)
            cos = torch.cat([cos.new_zeros(cos.shape[0], 1), cos], dim=1)
            contraction = r_l2[:, 1:] / r_l2[:, :-1].clamp_min(1.0e-6)
            contraction = torch.cat([contraction.new_ones(contraction.shape[0], 1), contraction], dim=1)
        else:
            cos = r_l2.new_zeros(r_l2.shape)
            contraction = r_l2.new_ones(r_l2.shape)

        signed_delta = torch.cat([r_l2.new_zeros(r_l2.shape[0], 1), r_l2[:, 1:] - r_l2[:, :-1]], dim=1)
        oscillation = 1.0 - cos
        final_defect_l2 = final_defect.norm(dim=1)
        final_defect_l1 = final_defect.abs().mean(dim=1)
        path_length = r_l2.sum(dim=1)
        defect_decay = r_l2[:, 0] - r_l2[:, -1]
        defect_stats = torch.stack(
            [
                path_length,
                r_l2.mean(dim=1),
                r_l2.max(dim=1).values,
                r_l2[:, -1],
                final_defect_l2,
                final_defect_l1,
                contraction.mean(dim=1),
                oscillation.mean(dim=1),
            ],
            dim=1,
        )
        per_step_features = torch.cat(
            [
                r_l2.unsqueeze(-1),
                r_l1.unsqueeze(-1),
                cos.unsqueeze(-1),
                contraction.unsqueeze(-1),
                signed_delta.unsqueeze(-1),
                r_proj,
            ],
            dim=2,
        ).flatten(1)
        full_features = [per_step_features, defect_stats]
        if self.include_final_latent:
            full_features.append(h_final)
        full = torch.cat(full_features, dim=1)
        norm_only = torch.cat([r_l2, r_l1, contraction, defect_stats[:, :4]], dim=1)
        final_only = h_final
        if mode == "final_latent_only":
            features = final_only
        elif mode == "defect_norm_only":
            features = norm_only
        else:
            features = full

        diagnostics = {
            "residual_l2": r_l2,
            "residual_l1": r_l1,
            "residual_cosine": cos,
            "contraction_ratio": contraction,
            "residual_projection": r_proj,
            "residual_signed_delta": signed_delta,
            "path_length": path_length,
            "defect_decay": defect_decay,
            "final_defect_l2": final_defect_l2,
            "final_defect_l1": final_defect_l1,
            "oscillation_energy": oscillation.mean(dim=1),
            "defect_stats": defect_stats,
        }
        return features, diagnostics


class FixedPointResidualDefectNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        latent_dim: int = 128,
        board_embed_dim: int = 64,
        depth: int = 2,
        steps: int = 6,
        update_hidden: int = 256,
        alpha: float = 0.5,
        projection_dim: int = 16,
        include_final_latent: bool = True,
        head_hidden: int = 192,
        dropout: float = 0.1,
        mode: str = "none",
        include_coordinates: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("FixedPointResidualDefectNetwork supports the puzzle_binary one-logit contract")
        self.num_classes = int(num_classes)
        self.steps = max(1, int(steps))
        self.alpha = float(alpha)
        self.mode = str(mode)
        self.encoder = BoardFixedPointEncoder(
            input_channels=int(input_channels),
            channels=int(channels),
            latent_dim=int(latent_dim),
            board_embed_dim=int(board_embed_dim),
            depth=int(depth),
            dropout=float(dropout),
            include_coordinates=bool(include_coordinates),
        )
        self.shared_update = ResidualUpdateBlock(
            latent_dim=int(latent_dim),
            board_embed_dim=int(board_embed_dim),
            update_hidden=int(update_hidden),
        )
        self.untied_updates = nn.ModuleList(
            [
                ResidualUpdateBlock(
                    latent_dim=int(latent_dim),
                    board_embed_dim=int(board_embed_dim),
                    update_hidden=int(update_hidden),
                )
                for _ in range(self.steps)
            ]
        )
        if self.mode == "random_update_operator":
            for parameter in self.shared_update.parameters():
                parameter.requires_grad_(False)
        self.stats = DefectTrajectoryStats(
            latent_dim=int(latent_dim),
            steps=self.steps,
            projection_dim=int(projection_dim),
            include_final_latent=bool(include_final_latent),
        )
        self.full_head = self._make_head(self.stats.full_output_dim, int(head_hidden), float(dropout))
        self.norm_head = self._make_head(self.stats.norm_only_dim, int(head_hidden), float(dropout))
        self.final_head = self._make_head(self.stats.final_only_dim, int(head_hidden), float(dropout))

    @staticmethod
    def _make_head(input_dim: int, hidden_dim: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(int(hidden_dim), max(32, int(hidden_dim) // 4)),
            nn.GELU(),
            nn.Linear(max(32, int(hidden_dim) // 4), 1),
        )

    def forward(self, x: torch.Tensor, *, return_path: bool = False) -> dict[str, torch.Tensor]:
        h0, board_embed = self.encoder(x)
        active_steps = 1 if self.mode == "single_step" else self.steps
        h_path_items = [h0]
        r_items: list[torch.Tensor] = []
        h = h0
        for step in range(active_steps):
            update = self.untied_updates[step] if self.mode == "untied_residual_blocks" else self.shared_update
            target = update(h, board_embed)
            r = target - h
            h = h + self.alpha * r
            r_items.append(r)
            h_path_items.append(h)
        while len(r_items) < self.steps:
            r_items.append(h.new_zeros(h.shape))
            h_path_items.append(h)
        h_path = torch.stack(h_path_items[: self.steps + 1], dim=1)
        r_path = torch.stack(r_items[: self.steps], dim=1)
        final_target = self.shared_update(h_path[:, active_steps], board_embed)
        final_defect = final_target - h_path[:, active_steps]
        features, diagnostics = self.stats(h_path, r_path, final_defect, mode=self.mode)
        if self.mode == "final_latent_only":
            logits = _format_logits(self.final_head(features), self.num_classes)
        elif self.mode == "defect_norm_only":
            logits = _format_logits(self.norm_head(features), self.num_classes)
        else:
            logits = _format_logits(self.full_head(features), self.num_classes)

        output = {
            "logits": logits,
            "prob": torch.sigmoid(logits),
            "defect_features": features,
            "h_final": h_path[:, active_steps],
            "residual_l2": diagnostics["residual_l2"],
            "residual_l1": diagnostics["residual_l1"],
            "residual_cosine": diagnostics["residual_cosine"],
            "contraction_ratio": diagnostics["contraction_ratio"],
            "residual_projection": diagnostics["residual_projection"],
            "residual_signed_delta": diagnostics["residual_signed_delta"],
            "path_length": diagnostics["path_length"],
            "defect_decay": diagnostics["defect_decay"],
            "final_defect_l2": diagnostics["final_defect_l2"],
            "final_defect_l1": diagnostics["final_defect_l1"],
            "oscillation_energy": diagnostics["oscillation_energy"],
            "defect_stats": diagnostics["defect_stats"],
            "active_steps": logits.new_full((logits.shape[0],), float(active_steps)),
            "fixed_point_mode": logits.new_full((logits.shape[0],), self._mode_code()),
            "mechanism_energy": diagnostics["final_defect_l2"],
            "proposal_profile_strength": diagnostics["path_length"],
            "proposal_keyword_count": logits.new_full((logits.shape[0],), 4.0),
        }
        if return_path:
            output["h_path"] = h_path
            output["r_path"] = r_path
            output["board_embed"] = board_embed
        return output

    def _mode_code(self) -> float:
        return {
            "none": 0.0,
            "fixed_point": 0.0,
            "final_latent_only": 1.0,
            "untied_residual_blocks": 2.0,
            "random_update_operator": 3.0,
            "defect_norm_only": 4.0,
            "single_step": 5.0,
        }.get(self.mode, 0.0)


def build_fixed_point_residual_defect_network_from_config(config: dict[str, Any]) -> FixedPointResidualDefectNetwork:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    channels = int(cfg.get("channels", cfg.get("hidden_dim", 64)))
    hidden_dim = int(cfg.get("hidden_dim", max(96, channels)))
    return FixedPointResidualDefectNetwork(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        channels=channels,
        latent_dim=int(cfg.get("latent_dim", 128)),
        board_embed_dim=int(cfg.get("board_embed_dim", 64)),
        depth=int(cfg.get("depth", 2)),
        steps=int(cfg.get("steps", 6)),
        update_hidden=int(cfg.get("update_hidden", max(128, hidden_dim * 2))),
        alpha=float(cfg.get("alpha", 0.5)),
        projection_dim=int(cfg.get("projection_dim", 16)),
        include_final_latent=bool(cfg.get("include_final_latent", True)),
        head_hidden=int(cfg.get("head_hidden", max(128, hidden_dim * 2))),
        dropout=float(cfg.get("dropout", 0.1)),
        mode=str(cfg.get("mode", cfg.get("ablation", "none"))),
        include_coordinates=bool(cfg.get("include_coordinates", True)),
    )

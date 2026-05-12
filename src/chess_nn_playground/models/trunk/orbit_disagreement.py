"""Orbit Disagreement Residual Network for idea i218.

Generates exact safe transform views of the board (identity, file flip, rank flip,
180-degree rotation), runs a shared encoder on each view, and classifies from the
invariant mean latent plus disagreement residual statistics.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


class _SharedBoardEncoder(nn.Module):
    def __init__(self, input_channels: int, channels: int, depth: int, dropout: float, use_batchnorm: bool) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = input_channels
        for _ in range(max(1, depth)):
            layers.append(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            in_channels = channels
        self.trunk = nn.Sequential(*layers)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


def _color_flip_planes(x: torch.Tensor) -> torch.Tensor:
    flipped = torch.flip(x, dims=[2])
    if x.shape[1] >= 12:
        white = flipped[:, :6].clone()
        black = flipped[:, 6:12].clone()
        flipped = flipped.clone()
        flipped[:, :6] = black
        flipped[:, 6:12] = white
    if x.shape[1] >= 13:
        side = flipped[:, 12:13]
        flipped = torch.cat([flipped[:, :12], 1.0 - side, flipped[:, 13:]], dim=1)
    return flipped


def _generate_views(x: torch.Tensor) -> torch.Tensor:
    views = [
        x,
        torch.flip(x, dims=[3]),
        torch.flip(x, dims=[2]),
        torch.flip(x, dims=[2, 3]),
        _color_flip_planes(x),
    ]
    return torch.stack(views, dim=1)


class OrbitDisagreementResidualNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        latent_dim: int = 64,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("OrbitDisagreementResidualNetwork supports the puzzle_binary one-logit contract")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.encoder = _SharedBoardEncoder(input_channels, channels, depth, dropout, use_batchnorm)
        self.latent_proj = nn.Sequential(
            nn.Linear(channels * 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )
        self.view_logit = nn.Linear(latent_dim, 1)
        residual_features = latent_dim * 2 + 6
        self.head = nn.Sequential(
            nn.LayerNorm(latent_dim + residual_features),
            nn.Linear(latent_dim + residual_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )
        self.num_views = 5

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        views = _generate_views(x)
        b, g, c, h, w = views.shape
        flat = views.reshape(b * g, c, h, w)
        feat = self.encoder(flat)
        pooled = torch.cat([feat.mean(dim=(2, 3)), feat.amax(dim=(2, 3))], dim=1)
        latents = self.latent_proj(pooled).view(b, g, -1)
        per_view_logit = self.view_logit(latents).view(b, g)

        orbit_mean = latents.mean(dim=1)
        residuals = latents - orbit_mean.unsqueeze(1)
        residual_norm_per_view = residuals.norm(dim=2)
        mean_residual_norm = residual_norm_per_view.mean(dim=1)
        max_residual_norm = residual_norm_per_view.amax(dim=1)
        cov = torch.matmul(residuals.transpose(1, 2), residuals) / float(g)
        diag_cov = cov.diagonal(dim1=1, dim2=2)
        cov_trace = diag_cov.sum(dim=1)
        offdiag = cov - torch.diag_embed(diag_cov)
        cov_offdiag_norm = offdiag.flatten(1).norm(dim=1)
        logit_disagreement = per_view_logit.std(dim=1, unbiased=False)
        logit_range = per_view_logit.amax(dim=1) - per_view_logit.amin(dim=1)
        residual_pool = torch.cat(
            [residuals.mean(dim=1), residuals.std(dim=1, unbiased=False)], dim=1
        )
        scalars = torch.stack(
            [
                mean_residual_norm,
                max_residual_norm,
                cov_trace,
                cov_offdiag_norm,
                logit_disagreement,
                logit_range,
            ],
            dim=1,
        )
        features = torch.cat([orbit_mean, residual_pool, scalars], dim=1)
        logits = self.head(features).view(-1)
        return {
            "logits": logits,
            "orbit_mean_norm": orbit_mean.norm(dim=1),
            "orbit_residual_mean_norm": mean_residual_norm,
            "orbit_residual_max_norm": max_residual_norm,
            "orbit_covariance_trace": cov_trace,
            "orbit_covariance_offdiag_norm": cov_offdiag_norm,
            "view_logit_disagreement": logit_disagreement,
            "view_logit_range": logit_range,
            "per_view_logit": per_view_logit,
            "symmetry_residual": mean_residual_norm,
        }


def build_orbit_disagreement_residual_network_from_config(config: dict[str, Any]) -> OrbitDisagreementResidualNetwork:
    cfg = dict(config)
    return OrbitDisagreementResidualNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        latent_dim=int(cfg.get("latent_dim", cfg.get("channels", 64))),
        num_classes=int(cfg.get("num_classes", 1)),
    )

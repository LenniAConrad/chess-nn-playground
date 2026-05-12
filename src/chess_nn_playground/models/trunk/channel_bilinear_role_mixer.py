"""Channel-Bilinear Role Mixer for idea i166.

Faithful implementation of the markdown thesis: a compact CNN trunk produces
per-square channel features, ``K`` role summaries are pooled with learned
spatial gates and per-role channel projections, and a low-rank bilinear head
explicitly models pairwise role-pair interactions ``r_i^T W_ij r_j`` without
materialising any square-pair tensor or local product convolution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


@dataclass(frozen=True)
class ChannelBilinearRoleMixerConfig:
    input_channels: int = 18
    num_classes: int = 1
    channels: int = 64
    hidden_dim: int = 96
    depth: int = 2
    dropout: float = 0.1
    use_batchnorm: bool = True
    num_roles: int = 8
    role_dim: int = 32
    bilinear_rank: int = 8


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float, use_batchnorm: bool) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=not use_batchnorm,
        )
        self.norm = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.norm(self.conv(x))))


class ChannelBilinearRoleMixer(nn.Module):
    """Channel-bilinear role-pair classifier.

    Pipeline per the thesis:

    1. ``H = Trunk(x)`` -- a stack of ``depth`` ConvBlocks that produces a
       per-square feature map ``H \\in R^{C \\times 8 \\times 8}``.
    2. ``K`` learned role definitions, each consisting of a softmax spatial
       gate ``m_k(s)`` over the 64 squares and a per-role channel projection
       ``W_k \\in R^{D \\times C}`` with bias ``b_k \\in R^D``. The role
       summary is

           ``r_k = LayerNorm( W_k ( sum_s m_k(s) H(s) ) + b_k )`` .

    3. A low-rank bilinear interaction between every ordered role pair
       ``(i, j)`` is obtained by projecting each role summary into two
       rank-``R`` views ``P_k = U r_k`` and ``Q_k = V r_k`` and taking

           ``M_{ij} = (1 / sqrt(R)) * P_i \\cdot Q_j`` .

       The full ``K \\times K`` matrix ``M`` of pairwise role interactions is
       flattened and passed through a small classifier head.

    The bilinear factorisation ``W_{ij} = U^T diag(.) V`` keeps the head
    parameter count at ``O(K * D * R + K * K)`` rather than ``O(K^2 * D^2)``,
    so the head is materially cheaper than a generic K-by-K bilinear form
    while still expressing every ordered role-pair interaction. No
    square-pair tensor or local product convolution is constructed.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        num_roles: int = 8,
        role_dim: int = 32,
        bilinear_rank: int = 8,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if num_roles < 2:
            raise ValueError("num_roles must be >= 2 to form pairwise interactions")
        if role_dim < 1:
            raise ValueError("role_dim must be >= 1")
        if bilinear_rank < 1:
            raise ValueError("bilinear_rank must be >= 1")

        self.config = ChannelBilinearRoleMixerConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            channels=channels,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            num_roles=num_roles,
            role_dim=role_dim,
            bilinear_rank=bilinear_rank,
        )
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.depth = int(depth)
        self.num_roles = int(num_roles)
        self.role_dim = int(role_dim)
        self.bilinear_rank = int(bilinear_rank)

        self.stem = nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.stem_norm = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.stem_activation = nn.GELU()
        self.trunk = nn.Sequential(
            *[
                _ConvBlock(channels, channels, dropout=dropout, use_batchnorm=use_batchnorm)
                for _ in range(depth)
            ]
        )

        # Per-role spatial gates: softmax over the 64 squares so each role
        # forms a proper weighted average rather than an unbounded sum.
        self.role_gate_logits = nn.Parameter(torch.zeros(num_roles, 8 * 8))
        # Per-role channel projection W_k with bias b_k. Stored as packed
        # tensors to support a single einsum over the role index.
        proj_weight = torch.empty(num_roles, role_dim, channels)
        bound = 1.0 / math.sqrt(channels)
        nn.init.uniform_(proj_weight, -bound, bound)
        self.role_proj_weight = nn.Parameter(proj_weight)
        self.role_proj_bias = nn.Parameter(torch.zeros(num_roles, role_dim))
        self.role_norm = nn.LayerNorm(role_dim)

        # Low-rank bilinear projections shared across roles. Two distinct
        # projections give an asymmetric interaction matrix, which is needed
        # so that ``role_i interacts with role_j`` differs from
        # ``role_j interacts with role_i`` (e.g. own-rooks attacking enemy
        # king zone is not the same direction as the converse).
        self.left_proj = nn.Linear(role_dim, bilinear_rank, bias=False)
        self.right_proj = nn.Linear(role_dim, bilinear_rank, bias=False)

        head_in = num_roles * num_roles
        self.classifier = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        h = self.stem_activation(self.stem_norm(self.stem(board)))
        h = self.trunk(h)  # (B, C, 8, 8)
        b, c, height, width = h.shape

        # Per-role spatial gate normalised over the 64 squares.
        gate = F.softmax(self.role_gate_logits, dim=-1)  # (K, 64)
        h_flat = h.view(b, c, height * width)
        # Pool channel features per role: (B, K, C).
        pooled = torch.einsum("ks,bcs->bkc", gate, h_flat)
        # Per-role channel projection W_k r + b_k -> (B, K, D).
        roles_pre = torch.einsum("bkc,kdc->bkd", pooled, self.role_proj_weight) + self.role_proj_bias
        roles = self.role_norm(roles_pre)

        # Low-rank bilinear interaction matrix.
        left = self.left_proj(roles)  # (B, K, R)
        right = self.right_proj(roles)  # (B, K, R)
        scale = 1.0 / math.sqrt(float(self.bilinear_rank))
        interaction = torch.einsum("bir,bjr->bij", left, right) * scale  # (B, K, K)

        head_in = interaction.flatten(1)  # (B, K*K)
        raw_logits = self.classifier(head_in)
        logits = _format_logits(raw_logits, self.num_classes)

        diagnostics: dict[str, torch.Tensor] = {
            "logits": logits,
            "trunk_features": h,
            "role_gates": gate.view(self.num_roles, height, width).detach(),
            "role_summaries": roles,
            "role_summaries_pre_norm": roles_pre,
            "role_pooled_channels": pooled,
            "bilinear_left": left,
            "bilinear_right": right,
            "bilinear_interaction_matrix": interaction,
            "bilinear_diag": torch.diagonal(interaction, dim1=1, dim2=2),
            "role_magnitude": roles.norm(dim=-1),
        }

        # Scalar diagnostics broadcast to (B,) for downstream logging.
        diagnostics["bilinear_energy"] = interaction.pow(2).mean(dim=(1, 2))
        diagnostics["bilinear_off_diag_energy"] = (
            interaction.pow(2).sum(dim=(1, 2)) - interaction.pow(2).diagonal(dim1=1, dim2=2).sum(dim=-1)
        ) / max(self.num_roles * (self.num_roles - 1), 1)
        diagnostics["bilinear_asymmetry"] = (
            (interaction - interaction.transpose(1, 2)).pow(2).mean(dim=(1, 2))
        )
        diagnostics["role_gate_entropy"] = -(gate * gate.clamp(min=1e-12).log()).sum(dim=-1).mean().expand(b)
        diagnostics["depth_levels"] = logits.new_full(logits.shape, float(self.depth))

        if self.num_classes == 1:
            diagnostics["prob"] = torch.sigmoid(logits)
        return diagnostics


def build_channel_bilinear_role_mixer_from_config(
    config: dict[str, Any],
) -> ChannelBilinearRoleMixer:
    return ChannelBilinearRoleMixer(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        depth=int(config.get("depth", 2)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        num_roles=int(config.get("num_roles", 8)),
        role_dim=int(config.get("role_dim", 32)),
        bilinear_rank=int(config.get("bilinear_rank", 8)),
    )

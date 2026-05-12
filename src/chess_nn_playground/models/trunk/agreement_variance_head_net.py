"""Agreement-Variance Head Net for idea i153.

A shared convolutional trunk feeds ``num_heads`` cheap classification
heads, all trained on the same puzzle_binary label. The reported
classification logit is the mean of the per-head logits; the variance
across heads is exposed as an uncertainty diagnostic (and is *not*
backpropagated). This is a lightweight alternative to a full ensemble:
one trunk forward pass produces ``num_heads`` predictions instead of
``num_heads`` independent models.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0, use_batchnorm: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.norm1 = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.norm2 = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.dropout = nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.norm1(self.conv1(x)), inplace=True)
        z = self.dropout(z)
        z = self.norm2(self.conv2(z))
        return F.relu(x + z, inplace=True)


class _CheapHead(nn.Module):
    """A single cheap classifier head: Linear -> ReLU -> Dropout -> Linear."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout(float(dropout)))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled).view(-1)


class AgreementVarianceHeadNet(nn.Module):
    """Shared CNN trunk + ``num_heads`` cheap heads averaged at inference.

    The classification logit is the mean across heads. The cross-head
    variance is exposed as a diagnostic (``head_variance``) and as a
    standard-deviation summary (``head_disagreement``); neither is part
    of the gradient signal, so increasing ``num_heads`` does not bias
    the optimisation toward minimising disagreement.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        num_heads: int = 5,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "AgreementVarianceHeadNet supports the puzzle_binary one-logit contract"
            )
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if num_heads < 2:
            raise ValueError("num_heads must be >= 2 so head variance is well defined")
        if channels < 1 or hidden_dim < 1:
            raise ValueError("channels and hidden_dim must be positive")

        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.dropout_p = float(dropout)

        self.stem = nn.Sequential(
            nn.Conv2d(int(input_channels), int(channels), kernel_size=3, padding=1, bias=not use_batchnorm),
            nn.BatchNorm2d(int(channels)) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.trunk_blocks = nn.ModuleList(
            [
                _ResidualBlock(int(channels), dropout=dropout, use_batchnorm=use_batchnorm)
                for _ in range(int(depth))
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.heads = nn.ModuleList(
            [_CheapHead(int(channels), int(hidden_dim), float(dropout)) for _ in range(int(num_heads))]
        )

        self._init_heads()

    def _init_heads(self) -> None:
        # Independent random init across heads ensures they do not all
        # collapse to the same function, which is what makes the variance
        # diagnostic informative.
        for head in self.heads:
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, a=5 ** 0.5)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def trunk(self, board: torch.Tensor) -> torch.Tensor:
        x = require_board_tensor(board, self.spec)
        h = self.stem(x)
        for block in self.trunk_blocks:
            h = block(h)
        return h

    def forward(self, board: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.trunk(board)
        pooled = self.pool(latent).flatten(1)

        per_head_logits = torch.stack([head(pooled) for head in self.heads], dim=1)
        # Mean logit is the classification output.
        mean_logit = per_head_logits.mean(dim=1)
        # Variance across heads is reported, not optimised.
        head_variance = per_head_logits.detach().var(dim=1, unbiased=False)
        head_disagreement = head_variance.sqrt()

        per_head_probs = torch.sigmoid(per_head_logits.detach())
        mean_prob = torch.sigmoid(mean_logit)
        prob_variance = per_head_probs.var(dim=1, unbiased=False)

        return {
            "logits": mean_logit,
            "logit": mean_logit,
            "prob": mean_prob,
            "per_head_logits": per_head_logits,
            "per_head_probs": per_head_probs,
            "head_variance": head_variance,
            "head_disagreement": head_disagreement,
            "prob_variance": prob_variance,
            "latent": latent,
        }


def build_agreement_variance_head_net_from_config(config: dict[str, Any]) -> AgreementVarianceHeadNet:
    cfg = dict(config)
    cfg.setdefault("input_channels", 18)
    cfg.setdefault("num_classes", 1)
    return AgreementVarianceHeadNet(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        num_heads=int(cfg.get("num_heads", 5)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
    )

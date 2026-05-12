"""Ordinal Evidence Ladder Network (idea i035)."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


def _inverse_softplus_scalar(value: float) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float32)
    return float(torch.log(torch.expm1(tensor.clamp_min(1.0e-4))).item())


class EncodingSafeStem(nn.Module):
    """Learned board stem with explicit input-channel validation."""

    def __init__(self, input_channels: int = 18, stem_width: int = 32, use_batchnorm: bool = True) -> None:
        super().__init__()
        if input_channels < 1:
            raise ValueError("input_channels must be positive")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, stem_width, kernel_size=1, bias=not use_batchnorm),
            nn.BatchNorm2d(stem_width) if use_batchnorm else nn.Identity(),
            nn.GELU(),
        )
        self.output_channels = stem_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(require_board_tensor(x, self.spec))


class ResidualBoardBlock(nn.Module):
    def __init__(self, width: int = 64, dropout: float = 0.0, use_batchnorm: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.norm1 = nn.BatchNorm2d(width) if use_batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.norm2 = nn.BatchNorm2d(width) if use_batchnorm else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        return F.gelu(x + residual)


class TinyBoardBackbone(nn.Module):
    def __init__(
        self,
        input_width: int = 32,
        backbone_width: int = 64,
        embedding_dim: int = 96,
        residual_blocks: int = 2,
        dropout: float = 0.0,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if residual_blocks < 0:
            raise ValueError("residual_blocks must be non-negative")
        self.entry = nn.Sequential(
            nn.Conv2d(input_width, backbone_width, kernel_size=3, padding=1, bias=not use_batchnorm),
            nn.BatchNorm2d(backbone_width) if use_batchnorm else nn.Identity(),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                ResidualBoardBlock(backbone_width, dropout=dropout, use_batchnorm=use_batchnorm)
                for _ in range(residual_blocks)
            ]
        )
        self.exit = nn.Sequential(
            nn.Conv2d(backbone_width, embedding_dim, kernel_size=3, padding=1, bias=not use_batchnorm),
            nn.BatchNorm2d(embedding_dim) if use_batchnorm else nn.Identity(),
            nn.GELU(),
        )
        self.output_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.exit(self.blocks(self.entry(x)))
        return h.mean(dim=(2, 3))


class OrdinalLadderHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 96,
        kappa_min: float = 2.0,
        threshold_gap_init: float = 1.0,
        slope_init: float = 1.0,
        eps: float = 1.0e-4,
        binary_event: str = "ge2",
    ) -> None:
        super().__init__()
        if kappa_min <= 0:
            raise ValueError("kappa_min must be positive")
        if threshold_gap_init <= 0:
            raise ValueError("threshold_gap_init must be positive")
        if slope_init <= 0:
            raise ValueError("slope_init must be positive")
        if binary_event not in {"ge1", "ge2"}:
            raise ValueError("binary_event must be 'ge1' or 'ge2'")
        self.score_head = nn.Linear(embedding_dim, 1)
        self.kappa_head = nn.Linear(embedding_dim, 1)
        self.center = nn.Parameter(torch.zeros(()))
        self.raw_gap = nn.Parameter(torch.tensor(_inverse_softplus_scalar(threshold_gap_init), dtype=torch.float32))
        self.raw_slope = nn.Parameter(torch.tensor(_inverse_softplus_scalar(slope_init), dtype=torch.float32))
        self.kappa_min = float(kappa_min)
        self.eps = float(eps)
        self.binary_event = binary_event

    def thresholds(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gap = F.softplus(self.raw_gap) + self.eps
        slope = F.softplus(self.raw_slope) + self.eps
        tau0 = self.center - 0.5 * gap
        tau1 = self.center + 0.5 * gap
        return tau0, tau1, slope

    def forward(self, h: torch.Tensor, *, num_classes: int = 1) -> dict[str, torch.Tensor]:
        score = self.score_head(h)
        concentration = self.kappa_min + F.softplus(self.kappa_head(h))
        tau0, tau1, slope = self.thresholds()
        ell1 = slope * (score - tau0)
        ell2 = slope * (score - tau1)
        q_ge1 = torch.sigmoid(ell1)
        q_ge2 = torch.sigmoid(ell2)
        fine_probs = torch.cat([1.0 - q_ge1, q_ge1 - q_ge2, q_ge2], dim=1).clamp_min(0.0)
        fine_probs = fine_probs / fine_probs.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        alpha = 1.0 + concentration * fine_probs
        binary_logit = ell1 if self.binary_event == "ge1" else ell2
        if num_classes == 1:
            logits = binary_logit.squeeze(-1)
        elif num_classes == 2:
            logits = torch.cat([torch.zeros_like(binary_logit), binary_logit], dim=1)
        else:
            raise ValueError("OrdinalEvidenceLadderNet supports num_classes 1 or 2")
        threshold_tensor = torch.stack([tau0, tau1]).to(device=h.device, dtype=h.dtype).view(1, 2).expand(h.shape[0], -1)
        alpha_sum = alpha.sum(dim=1)
        return {
            "logits": logits,
            "ordinal_logits": torch.cat([ell1, ell2], dim=1),
            "fine_probs": fine_probs,
            "alpha": alpha,
            "score": score.squeeze(-1),
            "q_ge1": q_ge1.squeeze(-1),
            "q_ge2": q_ge2.squeeze(-1),
            "near_or_puzzle_logit": ell1.squeeze(-1),
            "puzzle_logit": ell2.squeeze(-1),
            "evidence_concentration": concentration.squeeze(-1),
            "thresholds": threshold_tensor,
            "vacuity": 3.0 / alpha_sum.clamp_min(1.0e-8),
            "threshold_gap": (tau1 - tau0).expand(h.shape[0]),
            "slope": slope.expand(h.shape[0]),
        }


class OrdinalEvidenceLadderNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        stem_width: int = 32,
        backbone_width: int = 64,
        embedding_dim: int = 96,
        residual_blocks: int = 2,
        kappa_min: float = 2.0,
        threshold_gap_init: float = 1.0,
        slope_init: float = 1.0,
        binary_event: str = "ge2",
        dropout: float = 0.0,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.stem = EncodingSafeStem(input_channels=input_channels, stem_width=stem_width, use_batchnorm=use_batchnorm)
        self.backbone = TinyBoardBackbone(
            input_width=stem_width,
            backbone_width=backbone_width,
            embedding_dim=embedding_dim,
            residual_blocks=residual_blocks,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )
        self.head = OrdinalLadderHead(
            embedding_dim=embedding_dim,
            kappa_min=kappa_min,
            threshold_gap_init=threshold_gap_init,
            slope_init=slope_init,
            binary_event=binary_event,
        )

    def forward(self, x: torch.Tensor, *, return_aux: bool = False) -> dict[str, torch.Tensor] | torch.Tensor:
        h = self.backbone(self.stem(x))
        output = self.head(h, num_classes=self.num_classes)
        if return_aux:
            return output
        return output


def build_ordinal_evidence_ladder_network_from_config(config: dict[str, Any]) -> OrdinalEvidenceLadderNet:
    return OrdinalEvidenceLadderNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        stem_width=int(config.get("stem_width", 32)),
        backbone_width=int(config.get("backbone_width", config.get("channels", 64))),
        embedding_dim=int(config.get("embedding_dim", config.get("hidden_dim", 96))),
        residual_blocks=int(config.get("residual_blocks", config.get("depth", 2))),
        kappa_min=float(config.get("kappa_min", 2.0)),
        threshold_gap_init=float(config.get("threshold_gap_init", 1.0)),
        slope_init=float(config.get("slope_init", 1.0)),
        binary_event=str(config.get("binary_event", "ge2")),
        dropout=float(config.get("dropout", 0.0)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
    )


build_ordinal_evidence_ladder = build_ordinal_evidence_ladder_network_from_config

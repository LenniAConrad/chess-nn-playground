from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor
from chess_nn_playground.models.trunk.lc0_bt4 import LC0BT4Block


@dataclass
class VetoSelectConfig:
    input_channels: int = 112
    num_classes: int = 1
    channels: int = 64
    num_blocks: int = 4
    value_channels: int = 16
    value_hidden: int = 128
    se_channels: int = 16
    dropout: float = 0.1
    use_batchnorm: bool = True


class VetoSelectPuzzleNet(nn.Module):
    """Positive-claim abstention model for puzzle_binary classification.

    The raw evidence head emits ``puzzle_logit``. The selector head decides
    whether positive evidence should be accepted as a puzzle claim. Metrics use
    ``selective_puzzle_logit`` while prediction artifacts keep all diagnostics.
    """

    def __init__(
        self,
        input_channels: int = 112,
        num_classes: int = 1,
        channels: int = 64,
        num_blocks: int = 4,
        value_channels: int = 16,
        value_hidden: int = 128,
        se_channels: int = 16,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("VetoSelectPuzzleNet supports only a single puzzle_binary output")
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm),
            nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[
                LC0BT4Block(
                    channels=channels,
                    se_channels=se_channels,
                    use_batchnorm=use_batchnorm,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.value_projection = nn.Sequential(
            nn.Conv2d(channels, value_channels, kernel_size=1, bias=not use_batchnorm),
            nn.BatchNorm2d(value_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(value_channels * 8 * 8, value_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.evidence_head = nn.Linear(value_hidden, 1)
        self.selector_head = nn.Linear(value_hidden, 1)
        self.config = VetoSelectConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            channels=channels,
            num_blocks=num_blocks,
            value_channels=value_channels,
            value_hidden=value_hidden,
            se_channels=se_channels,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(require_board_tensor(x, self.spec))
        x = self.blocks(x)
        h = self.value_projection(x)
        z = self.evidence_head(h).squeeze(-1)
        a = self.selector_head(h).squeeze(-1)

        log_pi_n = F.logsigmoid(-z)
        log_pi_r = F.logsigmoid(z) + F.logsigmoid(-a)
        log_pi_p = F.logsigmoid(z) + F.logsigmoid(a)
        log_not_p = torch.logaddexp(log_pi_n, log_pi_r)
        selective_puzzle_logit = log_pi_p - log_not_p

        return {
            "logits": selective_puzzle_logit,
            "puzzle_logit": z,
            "selector_logit": a,
            "log_prob_nonpuzzle": log_pi_n,
            "log_prob_rejected_evidence": log_pi_r,
            "log_prob_accepted_puzzle": log_pi_p,
            "prob_nonpuzzle": log_pi_n.exp(),
            "prob_rejected_evidence": log_pi_r.exp(),
            "prob_accepted_puzzle": log_pi_p.exp(),
            "selective_puzzle_logit": selective_puzzle_logit,
            "reject_positive_logit": log_pi_r - log_pi_n,
        }


def build_vetoselect_from_config(config: dict[str, Any]) -> VetoSelectPuzzleNet:
    return VetoSelectPuzzleNet(
        input_channels=int(config.get("input_channels", 112)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        num_blocks=int(config.get("num_blocks", 4)),
        value_channels=int(config.get("value_channels", 16)),
        value_hidden=int(config.get("value_hidden", 128)),
        se_channels=int(config.get("se_channels", 16)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
    )

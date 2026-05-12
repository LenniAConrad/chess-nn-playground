"""Centered Tempo-Odd Interventional Bottleneck for idea i041.

Implements the markdown thesis: a shared convolutional encoder is applied
to four deterministic views of the same position - the original board
``x``, its side-to-move twin ``tau(x)``, a null board ``nu(x)`` whose
non-turn channels are zeroed, and its toggled twin ``tau(nu(x))``. The
encoder feature maps are split into the C2 odd component
``odd = 0.5 * (h - h_tau)`` and the null-board odd component
``null_odd = 0.5 * (h_null - h_null_tau)``. The centered odd map
``centered_odd = odd - null_odd`` is anti-invariant under turn toggling
and removes the additive board-only term, the constant term and the
pure side-to-move offset. Classifier reads spatial mean, max and RMS of
``centered_odd`` and outputs the puzzle logits.

The architecture is materially distinct from the shared
``ResearchPacketProbe`` scaffold: the encoder is shared by four paired
views, the head consumes a centered odd projection rather than pooled
trunk features, and there are no proposal-profile diagnostics or
mechanism-family embeddings in the forward path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models._packet_bespoke_base import (
    BoardTensorSpec,
    format_logits,
    require_board_tensor,
)


_EPS = 1.0e-6


@dataclass(frozen=True)
class Simple18CenteredTempoSpec:
    encoding: str = "simple_18"
    input_channels: int = 18
    side_to_move_channel: int = 12

    def validate(self, channels: int) -> None:
        if self.encoding != "simple_18":
            raise ValueError(
                "CenteredTempoOddInterventionalBottleneck only validates simple_18 "
                f"semantics; got encoding={self.encoding!r}"
            )
        if channels != self.input_channels:
            raise ValueError(
                "CenteredTempoOddInterventionalBottleneck expects 18 simple_18 channels; "
                f"got channels={channels}"
            )
        if not 0 <= self.side_to_move_channel < channels:
            raise ValueError(
                f"side_to_move_channel={self.side_to_move_channel} is outside {channels} channels"
            )


class CenteredTempoTurnAdapter(nn.Module):
    """Build the ``tau`` and ``nu`` deterministic counterfactual tensors.

    Constructs four paired views along the batch dimension::

        x4 = concat([x, tau(x), nu(x), tau(nu(x))], dim=0)

    where ``tau`` only flips the side-to-move plane and ``nu`` zeros every
    channel except the side-to-move plane. No move generation, mate flag,
    or engine input is consulted.
    """

    def __init__(
        self,
        encoding: str = "simple_18",
        input_channels: int = 18,
        side_to_move_channel: int = 12,
        fail_closed_unknown_channels: bool = True,
    ) -> None:
        super().__init__()
        self.spec = Simple18CenteredTempoSpec(
            encoding=encoding,
            input_channels=input_channels,
            side_to_move_channel=side_to_move_channel,
        )
        self.fail_closed_unknown_channels = bool(fail_closed_unknown_channels)

    def _validate(self, x: torch.Tensor) -> None:
        try:
            self.spec.validate(x.shape[1])
        except ValueError:
            if self.fail_closed_unknown_channels:
                raise

    def toggle_stm(self, x: torch.Tensor) -> torch.Tensor:
        toggled = x.clone()
        c = self.spec.side_to_move_channel
        toggled[:, c : c + 1] = 1.0 - x[:, c : c + 1]
        return toggled

    def null_board(self, x: torch.Tensor) -> torch.Tensor:
        c = self.spec.side_to_move_channel
        null = torch.zeros_like(x)
        null[:, c : c + 1] = x[:, c : c + 1]
        return null

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate(x)
        x_tau = self.toggle_stm(x)
        x_null = self.null_board(x)
        x_null_tau = self.toggle_stm(x_null)
        return torch.cat([x, x_tau, x_null, x_null_tau], dim=0)


class _ResidualBoardBlock(nn.Module):
    def __init__(self, channels: int, dropout: float, use_norm: bool) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_norm),
        ]
        if use_norm:
            layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_norm))
        if use_norm:
            layers.append(nn.BatchNorm2d(channels))
        self.block = nn.Sequential(*layers)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class SharedBoardEncoder(nn.Module):
    """Compact convolutional tower shared across the four paired views.

    Architecture mirrors the markdown spec: ``Conv(C -> stem) -> norm/GELU
    -> Conv(stem -> width) -> norm/GELU`` plus ``encoder_blocks`` residual
    blocks at ``width``. The encoder produces a ``(N, width, 8, 8)`` map
    with no spatial down-sampling so the centered odd map preserves the
    8x8 board grid.
    """

    def __init__(
        self,
        input_channels: int = 18,
        stem_channels: int = 64,
        width: int = 96,
        encoder_blocks: int = 2,
        dropout: float = 0.1,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        if encoder_blocks < 1:
            raise ValueError("encoder_blocks must be >= 1")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.width = int(width)
        stem_layers: list[nn.Module] = [
            nn.Conv2d(input_channels, stem_channels, kernel_size=3, padding=1, bias=not use_norm),
        ]
        if use_norm:
            stem_layers.append(nn.BatchNorm2d(stem_channels))
        stem_layers.append(nn.GELU())
        stem_layers.append(nn.Conv2d(stem_channels, width, kernel_size=3, padding=1, bias=not use_norm))
        if use_norm:
            stem_layers.append(nn.BatchNorm2d(width))
        stem_layers.append(nn.GELU())
        self.stem = nn.Sequential(*stem_layers)
        self.blocks = nn.Sequential(
            *(_ResidualBoardBlock(width, dropout=dropout, use_norm=use_norm) for _ in range(encoder_blocks))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = require_board_tensor(x, self.spec)
        return self.blocks(self.stem(x))


def _spatial_pool_concat(centered_odd: torch.Tensor) -> torch.Tensor:
    avg = centered_odd.mean(dim=(2, 3))
    amax = centered_odd.amax(dim=(2, 3))
    rms = torch.sqrt(centered_odd.pow(2).mean(dim=(2, 3)) + _EPS)
    return torch.cat([avg, amax, rms], dim=1)


class CenteredTempoOddInterventionalBottleneckNet(nn.Module):
    """Bespoke implementation of idea i041's centered tempo-odd bottleneck."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        encoding: str = "simple_18",
        side_to_move_channel: int = 12,
        stem_channels: int = 64,
        encoder_width: int = 96,
        encoder_blocks: int = 2,
        head_hidden_dim: int = 192,
        dropout: float = 0.1,
        use_norm: bool = True,
        encoder_chunk_size: int | None = None,
        fail_closed_unknown_channels: bool = True,
    ) -> None:
        super().__init__()
        if num_classes not in {1, 2}:
            raise ValueError(
                "CenteredTempoOddInterventionalBottleneckNet supports the puzzle_binary "
                "one-logit BCE contract or two-class CE outputs"
            )
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.encoder_chunk_size = int(encoder_chunk_size) if encoder_chunk_size else 0
        self.adapter = CenteredTempoTurnAdapter(
            encoding=encoding,
            input_channels=input_channels,
            side_to_move_channel=side_to_move_channel,
            fail_closed_unknown_channels=fail_closed_unknown_channels,
        )
        self.encoder = SharedBoardEncoder(
            input_channels=input_channels,
            stem_channels=stem_channels,
            width=encoder_width,
            encoder_blocks=encoder_blocks,
            dropout=dropout,
            use_norm=use_norm,
        )
        head_in = 3 * encoder_width
        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(head_hidden_dim, num_classes),
        )

    def _encode(self, x4: torch.Tensor) -> torch.Tensor:
        if self.encoder_chunk_size <= 0 or self.encoder_chunk_size >= x4.shape[0]:
            return self.encoder(x4)
        chunks = [self.encoder(chunk) for chunk in x4.split(self.encoder_chunk_size, dim=0)]
        return torch.cat(chunks, dim=0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        batch = x.shape[0]
        x4 = self.adapter(x)
        h4 = self._encode(x4)
        h, h_tau, h_null, h_null_tau = h4.split(batch, dim=0)

        odd = 0.5 * (h - h_tau)
        even = 0.5 * (h + h_tau)
        null_odd = 0.5 * (h_null - h_null_tau)
        centered_odd = odd - null_odd

        pooled = _spatial_pool_concat(centered_odd)
        head_logits = self.head(pooled)
        logits = format_logits(head_logits, self.num_classes)

        odd_norm = odd.flatten(1).norm(dim=1)
        even_norm = even.flatten(1).norm(dim=1)
        null_odd_norm = null_odd.flatten(1).norm(dim=1)
        centered_odd_norm = centered_odd.flatten(1).norm(dim=1)
        side_intervention_gap = (h - h_tau).flatten(1).norm(dim=1)
        centered_odd_energy = centered_odd.pow(2).mean(dim=(1, 2, 3))

        return {
            "logits": logits,
            "tempo_odd_norm": odd_norm,
            "tempo_even_norm": even_norm,
            "null_odd_norm": null_odd_norm,
            "centered_odd_norm": centered_odd_norm,
            "side_intervention_gap": side_intervention_gap,
            "centered_odd_energy": centered_odd_energy,
            "centered_odd": centered_odd,
        }


def build_centered_tempo_odd_interventional_bottleneck_from_config(
    config: dict[str, Any],
) -> CenteredTempoOddInterventionalBottleneckNet:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    head_hidden_dim = int(
        cfg.get(
            "head_hidden_dim",
            cfg.get("head_hidden", cfg.get("hidden_dim", 192)),
        )
    )
    return CenteredTempoOddInterventionalBottleneckNet(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        encoding=str(cfg.get("encoding", "simple_18")),
        side_to_move_channel=int(cfg.get("side_to_move_channel", cfg.get("stm_channel", 12))),
        stem_channels=int(cfg.get("stem_channels", cfg.get("channels", 64))),
        encoder_width=int(cfg.get("encoder_width", cfg.get("width", 96))),
        encoder_blocks=int(cfg.get("encoder_blocks", cfg.get("depth", 2))),
        head_hidden_dim=head_hidden_dim,
        dropout=float(cfg.get("dropout", 0.1)),
        use_norm=bool(cfg.get("use_norm", cfg.get("use_batchnorm", True))),
        encoder_chunk_size=cfg.get("encoder_chunk_size"),
        fail_closed_unknown_channels=bool(cfg.get("fail_closed_unknown_channels", True)),
    )

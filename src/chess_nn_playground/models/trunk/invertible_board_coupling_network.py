"""Invertible Board Coupling Network (idea i122).

This bespoke model materialises the reversible-encoder thesis from
`ideas/registry/i122_invertible_board_coupling_network/math_thesis.md`. Standard
encoders can discard information early, which makes it hard to know whether
the model latched onto legitimate current-board structure or fragile
shortcuts. This network preserves information by construction with a stack
of invertible affine coupling blocks and invertible 1x1 channel mixings,
then classifies from latent statistics together with per-layer coupling
scale diagnostics and an inverse-reconstruction error.

Following the markdown architecture sketch:

- `simple_18` is projected to width D by zero-padding (a reversible projection).
- Each block applies an invertible 1x1 channel mixing followed by an affine
  coupling `y_a = x_a; y_b = x_b * exp(s(x_a)) + t(x_a)` where the active
  half alternates across blocks.
- Scale outputs are clamped via `s = scale_clamp * tanh(raw_s)` to prevent
  explosion, as called out in the implementation notes of the source packet.
- The diagnostics include `mean_abs_s`, `max_abs_s`, `latent_energy`, and
  `inverse_reconstruction_error`, all of which are returned alongside the
  binary puzzle logit.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.data.board_features import SIMPLE_18
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


class InvertibleConv1x1(nn.Module):
    """Invertible 1x1 channel mixing initialised with a random orthogonal matrix."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels < 2:
            raise ValueError("InvertibleConv1x1 requires channels >= 2")
        weight = torch.empty(channels, channels)
        nn.init.orthogonal_(weight)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.weight.view(self.weight.size(0), self.weight.size(1), 1, 1)
        return F.conv2d(x, kernel)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        weight_inv = torch.linalg.inv(self.weight)
        kernel = weight_inv.view(weight_inv.size(0), weight_inv.size(1), 1, 1)
        return F.conv2d(y, kernel)


class AffineCouplingBlock(nn.Module):
    """Affine coupling `y_b = x_b * exp(s(x_a)) + t(x_a)` with channel split.

    The clamped scale `s = scale_clamp * tanh(raw_s)` keeps the affine
    transform bounded so the inverse is numerically stable.
    """

    def __init__(self, channels: int, hidden: int, swap: bool, scale_clamp: float = 0.8) -> None:
        super().__init__()
        if channels < 2 or channels % 2 != 0:
            raise ValueError("AffineCouplingBlock requires an even channel count")
        if hidden < 2:
            raise ValueError("AffineCouplingBlock requires hidden >= 2")
        self.channels = int(channels)
        self.swap = bool(swap)
        self.scale_clamp = float(scale_clamp)
        half = self.channels // 2
        self.net = nn.Sequential(
            nn.Conv2d(half, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, self.channels, kernel_size=3, padding=1),
        )
        # Initialise the last conv to zero so each block is the identity at init
        # (s=0, t=0 ⇒ y=x). This makes deep stacks stable.
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)

    def _split(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        half = self.channels // 2
        if self.swap:
            return z[:, half:], z[:, :half]
        return z[:, :half], z[:, half:]

    def _join(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        if self.swap:
            return torch.cat([x_b, x_a], dim=1)
        return torch.cat([x_a, x_b], dim=1)

    def _scale_translation(self, x_a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        st = self.net(x_a)
        s_raw, t = st.chunk(2, dim=1)
        s = self.scale_clamp * torch.tanh(s_raw)
        return s, t

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_a, x_b = self._split(z)
        s, t = self._scale_translation(x_a)
        y_b = x_b * torch.exp(s) + t
        return self._join(x_a, y_b), s

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        y_a, y_b = self._split(y)
        s, t = self._scale_translation(y_a)
        x_b = (y_b - t) * torch.exp(-s)
        return self._join(y_a, x_b)


class InvertibleBoardCouplingNetwork(nn.Module):
    """Reversible chess board encoder for the puzzle_binary contract."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 32,
        hidden_dim: int = 96,
        coupling_blocks: int = 6,
        coupling_hidden: int = 64,
        dropout: float = 0.1,
        scale_clamp: float = 0.8,
        encoding_adapter: str = SIMPLE_18,
    ) -> None:
        super().__init__()
        if input_channels != 18 or encoding_adapter != SIMPLE_18:
            raise ValueError("InvertibleBoardCouplingNetwork supports simple_18 with 18 input channels")
        if num_classes != 1:
            raise ValueError("InvertibleBoardCouplingNetwork supports the puzzle_binary one-logit contract")
        if channels < input_channels:
            raise ValueError("channels must be >= input_channels for the zero-padded reversible projection")
        if channels % 2 != 0:
            raise ValueError("channels must be even so the affine coupling can split halves")
        if coupling_blocks < 1:
            raise ValueError("coupling_blocks must be >= 1")

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.input_channels = int(input_channels)
        self.channels = int(channels)
        self.coupling_blocks = int(coupling_blocks)
        self.scale_clamp = float(scale_clamp)
        self.pad_channels = self.channels - self.input_channels

        mixings: list[InvertibleConv1x1] = []
        couplings: list[AffineCouplingBlock] = []
        for index in range(self.coupling_blocks):
            mixings.append(InvertibleConv1x1(self.channels))
            couplings.append(
                AffineCouplingBlock(
                    channels=self.channels,
                    hidden=int(coupling_hidden),
                    swap=bool(index % 2),
                    scale_clamp=self.scale_clamp,
                )
            )
        self.mixings = nn.ModuleList(mixings)
        self.couplings = nn.ModuleList(couplings)
        self.final_mixing = InvertibleConv1x1(self.channels)

        self.diagnostic_dim = 2 * self.coupling_blocks + 3
        latent_pool_dim = 2 * self.channels
        head_in = latent_pool_dim + self.diagnostic_dim

        self.classifier = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )
        # Diagnostic-only branch logit lets the trainer compare against a head
        # that ignores the latent pool, mirroring the markdown's "no_scale_stats"
        # ablation framing.
        self.diagnostic_logit = nn.Linear(self.diagnostic_dim, 1)

    def _pad_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_channels == 0:
            return x
        zeros = x.new_zeros(x.size(0), self.pad_channels, x.size(2), x.size(3))
        return torch.cat([x, zeros], dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        z0 = self._pad_input(x)
        z = z0
        per_block_mean: list[torch.Tensor] = []
        per_block_max: list[torch.Tensor] = []
        for mixing, coupling in zip(self.mixings, self.couplings):
            z = mixing(z)
            z, s = coupling(z)
            per_block_mean.append(s.abs().flatten(1).mean(dim=1))
            per_block_max.append(s.abs().flatten(1).amax(dim=1))
        z = self.final_mixing(z)
        latent = z

        # Frozen-inverse check from the source packet: invert the encoder and
        # measure how far the recovered tensor is from the padded input.
        recon = self.final_mixing.inverse(latent)
        for mixing, coupling in zip(reversed(self.mixings), reversed(self.couplings)):
            recon = coupling.inverse(recon)
            recon = mixing.inverse(recon)
        inverse_error = (recon - z0).abs().flatten(1).mean(dim=1)

        latent_energy = latent.pow(2).flatten(1).mean(dim=1)
        latent_peak = latent.abs().flatten(1).amax(dim=1)

        mean_abs_s = torch.stack(per_block_mean, dim=1)
        max_abs_s = torch.stack(per_block_max, dim=1)
        agg_mean_abs_s = mean_abs_s.mean(dim=1)
        agg_max_abs_s = max_abs_s.amax(dim=1)

        diagnostics = torch.cat(
            [
                mean_abs_s,
                max_abs_s,
                latent_energy.unsqueeze(1),
                latent_peak.unsqueeze(1),
                inverse_error.unsqueeze(1),
            ],
            dim=1,
        )

        latent_pool = torch.cat([latent.mean(dim=(2, 3)), latent.amax(dim=(2, 3))], dim=1)

        logits = self.classifier(torch.cat([latent_pool, diagnostics], dim=1)).squeeze(-1)
        diagnostic_branch_logit = self.diagnostic_logit(diagnostics).squeeze(-1)

        return {
            "logits": logits,
            "diagnostic_branch_logit": diagnostic_branch_logit,
            "latent_energy": latent_energy,
            "latent_peak": latent_peak,
            "inverse_reconstruction_error": inverse_error,
            "mean_abs_coupling_scale": agg_mean_abs_s,
            "max_abs_coupling_scale": agg_max_abs_s,
        }


def build_invertible_board_coupling_network_from_config(
    config: dict[str, Any],
) -> InvertibleBoardCouplingNetwork:
    return InvertibleBoardCouplingNetwork(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 32)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        coupling_blocks=int(config.get("coupling_blocks", 6)),
        coupling_hidden=int(config.get("coupling_hidden", 64)),
        dropout=float(config.get("dropout", 0.1)),
        scale_clamp=float(config.get("scale_clamp", 0.8)),
        encoding_adapter=str(config.get("encoding_adapter", config.get("encoding", SIMPLE_18))),
    )

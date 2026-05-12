from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


_EPS = 1.0e-6


@dataclass(frozen=True)
class Simple18TempoSpec:
    encoding: str = "simple_18"
    input_channels: int = 18
    side_to_move_channel: int = 12
    en_passant_channels: tuple[int, ...] = (17,)

    def validate(self, channels: int) -> None:
        if self.encoding != "simple_18" or channels != self.input_channels:
            raise ValueError(
                "TempoOddBottleneckNet only applies deterministic side-to-move intervention "
                f"to simple_18 with 18 channels, got encoding={self.encoding!r}, channels={channels}"
            )
        if not 0 <= self.side_to_move_channel < channels:
            raise ValueError(f"side_to_move_channel={self.side_to_move_channel} is outside {channels} channels")
        for channel in self.en_passant_channels:
            if not 0 <= channel < channels:
                raise ValueError(f"en_passant channel={channel} is outside {channels} channels")


def _norm2d(channels: int, enabled: bool) -> nn.Module:
    if not enabled:
        return nn.Identity()
    return nn.BatchNorm2d(channels)


class Simple18TempoToggle(nn.Module):
    """Rule-only side-to-move intervention for the repository simple_18 planes."""

    def __init__(
        self,
        encoding: str = "simple_18",
        input_channels: int = 18,
        side_to_move_channel: int = 12,
        en_passant_channels: tuple[int, ...] = (17,),
        zero_en_passant_for_tau: bool = True,
        tau_mode: str = "semantic_toggle",
        fail_closed_unknown_channels: bool = True,
    ) -> None:
        super().__init__()
        self.spec = Simple18TempoSpec(
            encoding=encoding,
            input_channels=input_channels,
            side_to_move_channel=side_to_move_channel,
            en_passant_channels=tuple(en_passant_channels),
        )
        self.zero_en_passant_for_tau = bool(zero_en_passant_for_tau)
        self.tau_mode = str(tau_mode)
        self.fail_closed_unknown_channels = bool(fail_closed_unknown_channels)

    def _validate(self, x: torch.Tensor) -> bool:
        try:
            self.spec.validate(x.shape[1])
        except ValueError:
            if self.fail_closed_unknown_channels:
                raise
            return False
        return True

    def sanitize(self, x: torch.Tensor) -> torch.Tensor:
        if not self._validate(x):
            return x
        sanitized = x.clone()
        if self.zero_en_passant_for_tau:
            for channel in self.spec.en_passant_channels:
                sanitized[:, channel : channel + 1] = 0.0
        return sanitized

    def semantic_toggle(self, x: torch.Tensor) -> torch.Tensor:
        toggled = x.clone()
        channel = self.spec.side_to_move_channel
        toggled[:, channel : channel + 1] = 1.0 - x[:, channel : channel + 1]
        return toggled

    def batch_permuted_side(self, x: torch.Tensor) -> torch.Tensor:
        toggled = x.clone()
        channel = self.spec.side_to_move_channel
        paired_side = x[:, channel : channel + 1].roll(shifts=1, dims=0)
        toggled[:, channel : channel + 1] = 1.0 - paired_side
        return toggled

    def random_side_marginal(self, x: torch.Tensor) -> torch.Tensor:
        toggled = x.clone()
        channel = self.spec.side_to_move_channel
        side_frequency = x[:, channel : channel + 1].mean().clamp(0.0, 1.0)
        random_side = torch.rand_like(x[:, channel : channel + 1]).lt(side_frequency).to(dtype=x.dtype)
        toggled[:, channel : channel + 1] = random_side
        return toggled

    def make_tau(self, x: torch.Tensor) -> torch.Tensor:
        if self.tau_mode == "semantic_toggle":
            return self.semantic_toggle(x)
        if self.tau_mode == "identity":
            return x.clone()
        if self.tau_mode == "batch_permuted_side":
            return self.batch_permuted_side(x)
        if self.tau_mode == "random_side_marginal":
            return self.random_side_marginal(x)
        raise ValueError(
            f"Unsupported tau_mode={self.tau_mode!r}; expected semantic_toggle, identity, "
            "batch_permuted_side, or random_side_marginal"
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x0 = self.sanitize(x)
        return x0, self.make_tau(x0)


class ResidualBoardBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0, use_norm: bool = True) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_norm),
            _norm2d(channels, use_norm),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_norm),
            _norm2d(channels, use_norm),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class SharedBoardEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        encoder_width: int = 64,
        encoder_blocks: int = 4,
        latent_dim: int = 128,
        dropout: float = 0.1,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        if encoder_blocks < 1:
            raise ValueError("encoder_blocks must be >= 1")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.latent_dim = int(latent_dim)
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, encoder_width, kernel_size=3, padding=1, bias=not use_norm),
            _norm2d(encoder_width, use_norm),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *(ResidualBoardBlock(encoder_width, dropout=dropout, use_norm=use_norm) for _ in range(encoder_blocks))
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(
            nn.Linear(encoder_width, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = require_board_tensor(x, self.spec)
        features = self.blocks(self.stem(x))
        return self.projection(self.pool(features).flatten(1))


class OddEvenWalshBottleneck(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        odd_dim: int = 64,
        even_dim: int = 16,
        stopgrad_even_context: bool = True,
    ) -> None:
        super().__init__()
        self.stopgrad_even_context = bool(stopgrad_even_context)
        self.odd_norm = nn.LayerNorm(latent_dim)
        self.even_norm = nn.LayerNorm(latent_dim)
        self.odd_projection = nn.Linear(latent_dim, odd_dim, bias=False)
        self.even_projection = nn.Linear(latent_dim, even_dim)

    def forward(self, h0: torch.Tensor, ht: torch.Tensor) -> dict[str, torch.Tensor]:
        z_even = 0.5 * (h0 + ht)
        z_odd = 0.5 * (h0 - ht)
        odd_signed = self.odd_projection(self.odd_norm(z_odd))
        odd_magnitude = odd_signed.abs()
        even_source = z_even.detach() if self.stopgrad_even_context else z_even
        even_context = self.even_projection(self.even_norm(even_source))
        return {
            "z_even": z_even,
            "z_odd": z_odd,
            "odd_signed": odd_signed,
            "odd_magnitude": odd_magnitude,
            "even_context": even_context,
        }


class TempoOddHead(nn.Module):
    def __init__(self, odd_dim: int = 64, even_dim: int = 16, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        input_dim = odd_dim * 2 + even_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, odd_signed: torch.Tensor, odd_magnitude: torch.Tensor, even_context: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([odd_signed, odd_magnitude, even_context], dim=1))


def _odd_variance_floor(odd_signed: torch.Tensor, floor: float = 0.2) -> torch.Tensor:
    if odd_signed.shape[0] < 2:
        return odd_signed.new_zeros(())
    std = torch.sqrt(odd_signed.var(dim=0, unbiased=False) + _EPS)
    return F.relu(float(floor) - std).pow(2).mean()


class TempoOddBottleneckNet(nn.Module):
    """Two-point Walsh odd/even bottleneck under a side-to-move intervention."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        encoding: str = "simple_18",
        side_to_move_channel: int = 12,
        en_passant_channels: tuple[int, ...] = (17,),
        zero_en_passant_for_tau: bool = True,
        encoder_width: int = 64,
        encoder_blocks: int = 4,
        latent_dim: int = 128,
        odd_dim: int = 64,
        even_dim: int = 16,
        head_hidden_dim: int = 64,
        dropout: float = 0.1,
        stopgrad_even_context: bool = True,
        tau_mode: str = "semantic_toggle",
        odd_variance_floor: float = 0.2,
        fail_closed_unknown_channels: bool = True,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        if num_classes not in {1, 2}:
            raise ValueError("TempoOddBottleneckNet supports one-logit BCE or two-class CE outputs")
        self.num_classes = int(num_classes)
        self.odd_variance_floor = float(odd_variance_floor)
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.adapter = Simple18TempoToggle(
            encoding=encoding,
            input_channels=input_channels,
            side_to_move_channel=side_to_move_channel,
            en_passant_channels=tuple(en_passant_channels),
            zero_en_passant_for_tau=zero_en_passant_for_tau,
            tau_mode=tau_mode,
            fail_closed_unknown_channels=fail_closed_unknown_channels,
        )
        self.encoder = SharedBoardEncoder(
            input_channels=input_channels,
            encoder_width=encoder_width,
            encoder_blocks=encoder_blocks,
            latent_dim=latent_dim,
            dropout=dropout,
            use_norm=use_norm,
        )
        self.bottleneck = OddEvenWalshBottleneck(
            latent_dim=latent_dim,
            odd_dim=odd_dim,
            even_dim=even_dim,
            stopgrad_even_context=stopgrad_even_context,
        )
        self.head = TempoOddHead(odd_dim=odd_dim, even_dim=even_dim, hidden_dim=head_hidden_dim, dropout=dropout)

    def _primary_logits(self, two_class_logits: torch.Tensor) -> torch.Tensor:
        if self.num_classes == 2:
            return two_class_logits
        return two_class_logits[:, 1] - two_class_logits[:, 0]

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        x0, xt = self.adapter(x)
        paired = torch.cat([x0, xt], dim=0)
        encoded = self.encoder(paired)
        h0, ht = encoded.chunk(2, dim=0)
        bottleneck = self.bottleneck(h0, ht)
        two_class_logits = self.head(
            bottleneck["odd_signed"],
            bottleneck["odd_magnitude"],
            bottleneck["even_context"],
        )
        logits = self._primary_logits(two_class_logits)
        odd_signed = bottleneck["odd_signed"]
        z_even = bottleneck["z_even"]
        z_odd = bottleneck["z_odd"]
        odd_energy = odd_signed.pow(2).mean(dim=1)
        even_energy = z_even.pow(2).mean(dim=1)
        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "two_class_logits": two_class_logits,
            "tempo_odd_norm": z_odd.norm(dim=1),
            "tempo_even_norm": z_even.norm(dim=1),
            "odd_energy": odd_energy,
            "even_energy": even_energy,
            "odd_to_even_energy_ratio": odd_energy / even_energy.clamp_min(_EPS),
            "side_intervention_gap": (h0 - ht).norm(dim=1),
            "odd_variance_loss": _odd_variance_floor(odd_signed, floor=self.odd_variance_floor),
            "en_passant_removed": (x[:, self.adapter.spec.en_passant_channels].abs().sum(dim=(1, 2, 3)) > 0).to(x.dtype),
        }
        if return_aux:
            output.update(
                {
                    "x_sanitized": x0,
                    "x_tau": xt,
                    "h_original": h0,
                    "h_tau": ht,
                    "z_even": z_even,
                    "z_odd": z_odd,
                    "odd_signed": odd_signed,
                    "odd_magnitude": bottleneck["odd_magnitude"],
                    "even_context": bottleneck["even_context"],
                }
            )
        return output


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = config.get("model") if isinstance(config.get("model"), dict) else config
    return dict(model_cfg)


def _encoding_from_config(config: dict[str, Any], model_cfg: dict[str, Any]) -> str:
    if "encoding" in model_cfg:
        return str(model_cfg["encoding"])
    if "channel_schema" in model_cfg:
        return str(model_cfg["channel_schema"])
    data_cfg = config.get("data") if isinstance(config.get("data"), dict) else {}
    return str(data_cfg.get("encoding", "simple_18"))


def _en_passant_channels(value: Any) -> tuple[int, ...]:
    if value is None:
        return (17,)
    if isinstance(value, int):
        return (int(value),)
    return tuple(int(item) for item in value)


def build_tempo_odd_bottleneck_from_config(config: dict[str, Any]) -> TempoOddBottleneckNet:
    model_cfg = _model_config(config)
    hidden_dim = int(model_cfg.get("hidden_dim", model_cfg.get("latent_dim", 128)))
    return TempoOddBottleneckNet(
        input_channels=int(model_cfg.get("input_channels", 18)),
        num_classes=int(model_cfg.get("num_classes", 1)),
        encoding=_encoding_from_config(config, model_cfg),
        side_to_move_channel=int(model_cfg.get("side_to_move_channel", 12)),
        en_passant_channels=_en_passant_channels(model_cfg.get("en_passant_channels")),
        zero_en_passant_for_tau=bool(model_cfg.get("zero_en_passant_for_tau", True)),
        encoder_width=int(model_cfg.get("encoder_width", model_cfg.get("channels", 64))),
        encoder_blocks=int(model_cfg.get("encoder_blocks", model_cfg.get("depth", 4))),
        latent_dim=int(model_cfg.get("latent_dim", hidden_dim)),
        odd_dim=int(model_cfg.get("odd_dim", max(8, hidden_dim // 2))),
        even_dim=int(model_cfg.get("even_dim", max(4, hidden_dim // 8))),
        head_hidden_dim=int(model_cfg.get("head_hidden_dim", model_cfg.get("head_hidden", max(16, hidden_dim // 2)))),
        dropout=float(model_cfg.get("dropout", 0.1)),
        stopgrad_even_context=bool(model_cfg.get("stopgrad_even_context", True)),
        tau_mode=str(model_cfg.get("tau_mode", "semantic_toggle")),
        odd_variance_floor=float(model_cfg.get("odd_variance_floor", 0.2)),
        fail_closed_unknown_channels=bool(model_cfg.get("fail_closed_unknown_channels", True)),
        use_norm=bool(model_cfg.get("use_norm", model_cfg.get("use_batchnorm", True))),
    )


def build_tempo_odd_bottleneck(config: dict[str, Any]) -> TempoOddBottleneckNet:
    return build_tempo_odd_bottleneck_from_config(config)

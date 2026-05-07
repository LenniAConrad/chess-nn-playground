from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


_EPS = 1.0e-6


@dataclass(frozen=True)
class RuleExactOrbitConfig:
    input_channels: int = 18
    num_classes: int = 1
    channel_schema: str = "simple_18"
    orbit_group: str = "color_flip"
    pool_mode: str = "probability_mean"
    stem_width: int = 48
    latent_dim: int = 128
    num_blocks: int = 3
    fail_closed_unknown_channels: bool = True
    use_norm: bool = True


def _normalization(channels: int, enabled: bool) -> nn.Module:
    if not enabled:
        return nn.Identity()
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class Simple18ColorFlipAdapter(nn.Module):
    """Exact color-flip orbit adapter for the repo's simple_18 board planes."""

    def __init__(
        self,
        channel_schema: str = "simple_18",
        orbit_group: str = "color_flip",
        fail_closed_unknown_channels: bool = True,
    ) -> None:
        super().__init__()
        self.channel_schema = str(channel_schema)
        self.orbit_group = str(orbit_group)
        self.fail_closed_unknown_channels = bool(fail_closed_unknown_channels)

    def _validate_simple18(self, x: torch.Tensor) -> bool:
        valid = self.channel_schema == "simple_18" and x.shape[1] == 18
        if valid:
            return True
        if self.fail_closed_unknown_channels:
            raise ValueError(
                "RuleExactOrbitBottleneckNet only applies deterministic color flips to "
                f"channel_schema='simple_18' with 18 channels, got schema={self.channel_schema!r} "
                f"and channels={x.shape[1]}"
            )
        return False

    @staticmethod
    def color_flip(x: torch.Tensor) -> torch.Tensor:
        flipped = x.clone()
        rank_mirror = torch.flip(x, dims=[2])
        flipped[:, 0:6] = rank_mirror[:, 6:12]
        flipped[:, 6:12] = rank_mirror[:, 0:6]
        flipped[:, 12:13] = 1.0 - rank_mirror[:, 12:13]
        flipped[:, 13:14] = x[:, 15:16]
        flipped[:, 14:15] = x[:, 16:17]
        flipped[:, 15:16] = x[:, 13:14]
        flipped[:, 16:17] = x[:, 14:15]
        flipped[:, 17:18] = rank_mirror[:, 17:18]
        return flipped

    @staticmethod
    def rank_flip_no_color(x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=[2])

    def make_orbit(self, x: torch.Tensor) -> torch.Tensor:
        if self.orbit_group == "identity":
            return x.unsqueeze(1)
        if not self._validate_simple18(x):
            return x.unsqueeze(1)
        if self.orbit_group == "color_flip":
            transformed = self.color_flip(x)
        elif self.orbit_group == "rank_flip_no_color":
            transformed = self.rank_flip_no_color(x)
        else:
            raise ValueError(
                "Unsupported orbit_group="
                f"{self.orbit_group!r}; expected 'color_flip', 'rank_flip_no_color', or 'identity'"
            )
        return torch.stack([x, transformed], dim=1)


class OrbitAdapterRegistry:
    @staticmethod
    def build(
        channel_schema: str,
        orbit_group: str,
        fail_closed_unknown_channels: bool,
    ) -> Simple18ColorFlipAdapter:
        return Simple18ColorFlipAdapter(
            channel_schema=channel_schema,
            orbit_group=orbit_group,
            fail_closed_unknown_channels=fail_closed_unknown_channels,
        )


class ResidualMicroBlock(nn.Module):
    def __init__(self, channels: int, use_norm: bool = True) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_norm),
            _normalization(channels, use_norm),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_norm),
            _normalization(channels, use_norm),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class TinyBoardStem(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        stem_width: int = 48,
        num_blocks: int = 3,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.width = int(stem_width)
        self.output_dim = self.width * 2
        self.input = nn.Sequential(
            nn.Conv2d(input_channels, self.width, kernel_size=3, padding=1, bias=not use_norm),
            _normalization(self.width, use_norm),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*(ResidualMicroBlock(self.width, use_norm=use_norm) for _ in range(num_blocks)))
        self.projection = nn.Sequential(
            nn.Conv2d(self.width, self.output_dim, kernel_size=1, bias=not use_norm),
            _normalization(self.output_dim, use_norm),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = require_board_tensor(x, self.spec)
        features = self.projection(self.blocks(self.input(x)))
        return self.pool(features).flatten(1)


class RuleExactOrbitBottleneckNet(nn.Module):
    """Reynolds probability-pooling classifier over a rule-exact color-flip orbit."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channel_schema: str = "simple_18",
        orbit_group: str = "color_flip",
        pool_mode: str = "probability_mean",
        stem_width: int = 48,
        latent_dim: int = 128,
        num_blocks: int = 3,
        fail_closed_unknown_channels: bool = True,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        if num_classes not in {1, 2}:
            raise ValueError("RuleExactOrbitBottleneckNet supports one-logit BCE or two-class CE outputs")
        if pool_mode not in {"probability_mean", "logit_mean", "latent_mean"}:
            raise ValueError("pool_mode must be 'probability_mean', 'logit_mean', or 'latent_mean'")
        self.cfg = RuleExactOrbitConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            channel_schema=channel_schema,
            orbit_group=orbit_group,
            pool_mode=pool_mode,
            stem_width=stem_width,
            latent_dim=latent_dim,
            num_blocks=num_blocks,
            fail_closed_unknown_channels=fail_closed_unknown_channels,
            use_norm=use_norm,
        )
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.adapter = OrbitAdapterRegistry.build(channel_schema, orbit_group, fail_closed_unknown_channels)
        self.stem = TinyBoardStem(
            input_channels=input_channels,
            stem_width=stem_width,
            num_blocks=num_blocks,
            use_norm=use_norm,
        )
        self.latent = nn.Sequential(
            nn.Linear(self.stem.output_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(latent_dim, num_classes)

    def _pool_logits(self, z: torch.Tensor, view_logits: torch.Tensor) -> torch.Tensor:
        if self.cfg.pool_mode == "latent_mean":
            logits = self.classifier(z.mean(dim=1))
            return logits.view(-1) if self.cfg.num_classes == 1 else logits
        if self.cfg.pool_mode == "logit_mean":
            logits = view_logits.mean(dim=1)
            return logits.view(-1) if self.cfg.num_classes == 1 else logits
        if self.cfg.num_classes == 1:
            probabilities = torch.sigmoid(view_logits.squeeze(-1)).mean(dim=1).clamp(_EPS, 1.0 - _EPS)
            return torch.logit(probabilities)
        probabilities = F.softmax(view_logits, dim=-1).mean(dim=1).clamp_min(_EPS)
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(_EPS)
        return probabilities.log()

    def _puzzle_probabilities(self, view_logits: torch.Tensor) -> torch.Tensor:
        if self.cfg.num_classes == 1:
            return torch.sigmoid(view_logits.squeeze(-1))
        return F.softmax(view_logits, dim=-1)[..., 1]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        orbit = self.adapter.make_orbit(x)
        batch_size, orbit_size, channels, height, width = orbit.shape
        flat_orbit = orbit.reshape(batch_size * orbit_size, channels, height, width)
        flat_hidden = self.stem(flat_orbit)
        z = self.latent(flat_hidden).reshape(batch_size, orbit_size, self.cfg.latent_dim)
        view_logits = self.classifier(z)
        logits = self._pool_logits(z, view_logits)

        view_probabilities = self._puzzle_probabilities(view_logits)
        transformed_index = 1 if orbit_size > 1 else 0
        identity_probability = view_probabilities[:, 0]
        transformed_probability = view_probabilities[:, transformed_index]
        if self.cfg.num_classes == 1:
            per_view_puzzle_logits = view_logits.squeeze(-1)
        else:
            per_view_puzzle_logits = view_logits[..., 1]
        identity_logit = per_view_puzzle_logits[:, 0]
        transformed_logit = per_view_puzzle_logits[:, transformed_index]
        probability_gap = (identity_probability - transformed_probability).abs()

        return {
            "logits": logits,
            "identity_view_logit": identity_logit,
            "transformed_view_logit": transformed_logit,
            "view_logit_gap": (identity_logit - transformed_logit).abs(),
            "identity_probability": identity_probability,
            "transformed_probability": transformed_probability,
            "mean_view_probability": view_probabilities.mean(dim=1),
            "orbit_probability_gap": probability_gap,
            "symmetry_residual": probability_gap,
            "latent_orbit_variance": z.var(dim=1, unbiased=False).mean(dim=1),
            "mechanism_energy": z.pow(2).mean(dim=(1, 2)),
            "orbit_size": x.new_full((batch_size,), float(orbit_size)),
        }


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = config.get("model") if isinstance(config.get("model"), dict) else config
    return dict(model_cfg)


def build_rule_exact_orbit_bottleneck_from_config(config: dict[str, Any]) -> RuleExactOrbitBottleneckNet:
    model_cfg = _model_config(config)
    return RuleExactOrbitBottleneckNet(
        input_channels=int(model_cfg.get("input_channels", 18)),
        num_classes=int(model_cfg.get("num_classes", 1)),
        channel_schema=str(model_cfg.get("channel_schema", "simple_18")),
        orbit_group=str(model_cfg.get("orbit_group", "color_flip")),
        pool_mode=str(model_cfg.get("pool_mode", "probability_mean")),
        stem_width=int(model_cfg.get("stem_width", model_cfg.get("channels", 48))),
        latent_dim=int(model_cfg.get("latent_dim", model_cfg.get("hidden_dim", 128))),
        num_blocks=int(model_cfg.get("num_blocks", model_cfg.get("depth", 3))),
        fail_closed_unknown_channels=bool(model_cfg.get("fail_closed_unknown_channels", True)),
        use_norm=bool(model_cfg.get("use_norm", model_cfg.get("use_batchnorm", True))),
    )


def build_rule_exact_orbit_bottleneck(config: dict[str, Any]) -> RuleExactOrbitBottleneckNet:
    return build_rule_exact_orbit_bottleneck_from_config(config)

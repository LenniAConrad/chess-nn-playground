from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


_EPS = 1.0e-6


@dataclass(frozen=True)
class Simple18ChannelSpec:
    encoding: str = "simple_18"
    input_channels: int = 18
    piece_planes: tuple[int, ...] = tuple(range(12))
    side_to_move_plane: int = 12
    castling_planes: tuple[int, int, int, int] = (13, 14, 15, 16)
    en_passant_plane: int = 17

    def validate(self, channels: int) -> None:
        if self.encoding != "simple_18" or channels != self.input_channels:
            raise ValueError(
                "RuleAutomorphismQuotientNet only has deterministic rule automorphisms "
                f"for simple_18 with 18 channels, got encoding={self.encoding!r}, channels={channels}"
            )


def _norm2d(channels: int, enabled: bool) -> nn.Module:
    if not enabled:
        return nn.Identity()
    return nn.BatchNorm2d(channels)


class Simple18AutomorphismOrbit(nn.Module):
    """Builds the RAQ-Net orbit [I, C, H, HC] and a per-sample validity mask."""

    def __init__(
        self,
        encoding: str = "simple_18",
        input_channels: int = 18,
        use_color_turn_reversal: bool = True,
        use_file_mirror_if_castling_absent: bool = True,
        fail_closed_unknown_channels: bool = True,
        pseudo_orbit: bool = False,
        castling_absent_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.spec = Simple18ChannelSpec(encoding=encoding, input_channels=input_channels)
        self.use_color_turn_reversal = bool(use_color_turn_reversal)
        self.use_file_mirror_if_castling_absent = bool(use_file_mirror_if_castling_absent)
        self.fail_closed_unknown_channels = bool(fail_closed_unknown_channels)
        self.pseudo_orbit = bool(pseudo_orbit)
        self.castling_absent_threshold = float(castling_absent_threshold)

    def _validate(self, x: torch.Tensor) -> bool:
        try:
            self.spec.validate(x.shape[1])
        except ValueError:
            if self.fail_closed_unknown_channels:
                raise
            return False
        return True

    def castling_absent(self, x: torch.Tensor) -> torch.Tensor:
        self.spec.validate(x.shape[1])
        castling = x[:, 13:17].abs().amax(dim=(1, 2, 3))
        return castling <= self.castling_absent_threshold

    @staticmethod
    def color_turn_reversal(x: torch.Tensor) -> torch.Tensor:
        rank_mirror = torch.flip(x, dims=[2])
        transformed = x.clone()
        transformed[:, 0:6] = rank_mirror[:, 6:12]
        transformed[:, 6:12] = rank_mirror[:, 0:6]
        transformed[:, 12:13] = 1.0 - rank_mirror[:, 12:13]
        transformed[:, 13:14] = x[:, 15:16]
        transformed[:, 14:15] = x[:, 16:17]
        transformed[:, 15:16] = x[:, 13:14]
        transformed[:, 16:17] = x[:, 14:15]
        transformed[:, 17:18] = rank_mirror[:, 17:18]
        return transformed

    @staticmethod
    def file_mirror(x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=[3])

    @staticmethod
    def pseudo_rank_file_orbit(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rank_flip = torch.flip(x, dims=[2])
        file_flip = torch.flip(x, dims=[3])
        rank_file_flip = torch.flip(x, dims=[2, 3])
        return rank_flip, file_flip, rank_file_flip

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._validate(x):
            batch = x.shape[0]
            return x.unsqueeze(1), x.new_ones((batch, 1), dtype=torch.bool)

        batch = x.shape[0]
        identity = x
        valid_identity = torch.ones(batch, dtype=torch.bool, device=x.device)

        if self.pseudo_orbit:
            pseudo_c, pseudo_h, pseudo_hc = self.pseudo_rank_file_orbit(x)
            orbit = torch.stack([identity, pseudo_c, pseudo_h, pseudo_hc], dim=1)
            mask = torch.ones(batch, 4, dtype=torch.bool, device=x.device)
            return orbit, mask

        color_view = self.color_turn_reversal(x) if self.use_color_turn_reversal else x
        valid_color = torch.ones(batch, dtype=torch.bool, device=x.device) if self.use_color_turn_reversal else valid_identity
        mirror_view = self.file_mirror(x)
        composed_view = self.file_mirror(color_view)
        valid_file = self.castling_absent(x) & self.use_file_mirror_if_castling_absent
        valid_composed = valid_file & valid_color
        orbit = torch.stack([identity, color_view, mirror_view, composed_view], dim=1)
        mask = torch.stack([valid_identity, valid_color, valid_file, valid_composed], dim=1)
        return orbit, mask


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
        hidden_channels: int = 64,
        latent_dim: int = 128,
        num_res_blocks: int = 4,
        dropout: float = 0.1,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        if num_res_blocks < 1:
            raise ValueError("num_res_blocks must be >= 1")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.latent_dim = int(latent_dim)
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1, bias=not use_norm),
            _norm2d(hidden_channels, use_norm),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *(ResidualBoardBlock(hidden_channels, dropout=dropout, use_norm=use_norm) for _ in range(num_res_blocks))
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_channels, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = require_board_tensor(x, self.spec)
        features = self.blocks(self.stem(x))
        return self.to_latent(self.pool(features).flatten(1))


class MaskedReynoldsPool(nn.Module):
    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(dtype=z.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (z * weights).sum(dim=1) / denom


class OrbitProjectionHead(nn.Module):
    def __init__(self, latent_dim: int = 128, projection_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(latent_dim, projection_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class QuotientClassifier(nn.Module):
    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim)
        self.classifier = nn.Linear(latent_dim, 2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.norm(z))


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _valid_flat(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return values[mask]


def _vicreg_variance_loss(valid_projection: torch.Tensor) -> torch.Tensor:
    if valid_projection.shape[0] < 2:
        return valid_projection.new_zeros(())
    std = torch.sqrt(valid_projection.var(dim=0, unbiased=False) + _EPS)
    return F.relu(1.0 - std).mean()


def _vicreg_covariance_loss(valid_projection: torch.Tensor) -> torch.Tensor:
    if valid_projection.shape[0] < 2:
        return valid_projection.new_zeros(())
    centered = valid_projection - valid_projection.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / max(valid_projection.shape[0] - 1, 1)
    off_diag = cov - torch.diag_embed(torch.diag(cov))
    return off_diag.pow(2).sum() / valid_projection.shape[1]


class RuleAutomorphismQuotientNet(nn.Module):
    """Masked Reynolds quotient classifier over safe simple_18 chess automorphisms."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        encoding: str = "simple_18",
        hidden_channels: int = 64,
        latent_dim: int = 128,
        projection_dim: int = 64,
        num_res_blocks: int = 4,
        dropout: float = 0.1,
        use_color_turn_reversal: bool = True,
        use_file_mirror_if_castling_absent: bool = True,
        fail_closed_unknown_channels: bool = True,
        pseudo_orbit: bool = False,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        if num_classes not in {1, 2}:
            raise ValueError("RuleAutomorphismQuotientNet supports one-logit BCE or two-class CE outputs")
        self.num_classes = int(num_classes)
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.orbit_adapter = Simple18AutomorphismOrbit(
            encoding=encoding,
            input_channels=input_channels,
            use_color_turn_reversal=use_color_turn_reversal,
            use_file_mirror_if_castling_absent=use_file_mirror_if_castling_absent,
            fail_closed_unknown_channels=fail_closed_unknown_channels,
            pseudo_orbit=pseudo_orbit,
        )
        self.encoder = SharedBoardEncoder(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            use_norm=use_norm,
        )
        self.pool = MaskedReynoldsPool()
        self.projection_head = OrbitProjectionHead(latent_dim=latent_dim, projection_dim=projection_dim, dropout=dropout)
        self.classifier = QuotientClassifier(latent_dim=latent_dim)

    def _primary_logits(self, two_class_logits: torch.Tensor) -> torch.Tensor:
        if self.num_classes == 2:
            return two_class_logits
        return two_class_logits[:, 1] - two_class_logits[:, 0]

    def _view_primary_logits(self, view_two_class_logits: torch.Tensor) -> torch.Tensor:
        if self.num_classes == 2:
            return view_two_class_logits
        return view_two_class_logits[..., 1] - view_two_class_logits[..., 0]

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        orbit, mask = self.orbit_adapter(x)
        batch_size, orbit_size, channels, height, width = orbit.shape
        flat_orbit = orbit.reshape(batch_size * orbit_size, channels, height, width)
        z = self.encoder(flat_orbit).reshape(batch_size, orbit_size, -1)
        z_bar = self.pool(z, mask)
        two_class_logits = self.classifier(z_bar)
        view_two_class_logits = self.classifier(z)
        logits = self._primary_logits(two_class_logits)
        view_logits = self._view_primary_logits(view_two_class_logits)

        projection = self.projection_head(z)
        projection_bar = _masked_mean(projection, mask)
        projection_delta = (projection - projection_bar.unsqueeze(1)).pow(2).mean(dim=-1)
        mask_float = mask.to(dtype=z.dtype)
        orbit_consistency = (projection_delta * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
        valid_projection = _valid_flat(projection, mask)
        view_logit_center = _masked_mean(view_logits.unsqueeze(-1), mask).squeeze(-1)
        view_logit_delta = (view_logits - view_logit_center.unsqueeze(1)).pow(2)
        view_logit_variance = (view_logit_delta * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)

        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "two_class_logits": two_class_logits,
            "valid_view_count": mask_float.sum(dim=1),
            "file_mirror_valid": mask[:, 2].to(dtype=z.dtype) if orbit_size > 2 else z.new_zeros(batch_size),
            "orbit_variance": (z - z_bar.unsqueeze(1)).pow(2).mean(dim=(1, 2)),
            "masked_orbit_variance": ((z - z_bar.unsqueeze(1)).pow(2).mean(dim=2) * mask_float).sum(dim=1)
            / mask_float.sum(dim=1).clamp_min(1.0),
            "view_logit_variance": view_logit_variance,
            "symmetry_residual": view_logit_variance.sqrt(),
            "orbit_consistency": orbit_consistency,
            "reynolds_norm": z_bar.norm(dim=1),
            "projection_norm": projection_bar.norm(dim=1),
            "mechanism_energy": z.pow(2).mean(dim=(1, 2)),
            "risk_variance_proxy": view_logit_variance,
        }
        if return_aux:
            output.update(
                {
                    "orbit_mask": mask,
                    "z": z,
                    "projection": projection,
                    "view_logits": view_logits,
                    "view_two_class_logits": view_two_class_logits,
                    "orbit_consistency_loss": orbit_consistency.mean(),
                    "vicreg_variance_loss": _vicreg_variance_loss(valid_projection),
                    "vicreg_covariance_loss": _vicreg_covariance_loss(valid_projection),
                    "latent_small_loss": z_bar.pow(2).mean(),
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


def build_rule_automorphism_quotient_bottleneck_from_config(config: dict[str, Any]) -> RuleAutomorphismQuotientNet:
    model_cfg = _model_config(config)
    return RuleAutomorphismQuotientNet(
        input_channels=int(model_cfg.get("input_channels", 18)),
        num_classes=int(model_cfg.get("num_classes", 1)),
        encoding=_encoding_from_config(config, model_cfg),
        hidden_channels=int(model_cfg.get("hidden_channels", model_cfg.get("channels", 64))),
        latent_dim=int(model_cfg.get("latent_dim", model_cfg.get("hidden_dim", 128))),
        projection_dim=int(model_cfg.get("projection_dim", max(32, int(model_cfg.get("hidden_dim", 128)) // 2))),
        num_res_blocks=int(model_cfg.get("num_res_blocks", model_cfg.get("depth", 4))),
        dropout=float(model_cfg.get("dropout", 0.1)),
        use_color_turn_reversal=bool(model_cfg.get("use_color_turn_reversal", True)),
        use_file_mirror_if_castling_absent=bool(model_cfg.get("use_file_mirror_if_castling_absent", True)),
        fail_closed_unknown_channels=bool(model_cfg.get("fail_closed_unknown_channels", True)),
        pseudo_orbit=bool(model_cfg.get("pseudo_orbit", False)),
        use_norm=bool(model_cfg.get("use_norm", model_cfg.get("use_batchnorm", True))),
    )


def build_rule_automorphism_quotient_net(config: dict[str, Any]) -> RuleAutomorphismQuotientNet:
    return build_rule_automorphism_quotient_bottleneck_from_config(config)

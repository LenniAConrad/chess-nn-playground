from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


@dataclass(frozen=True)
class EncodingSemanticSpec:
    encoding: str = "simple_18"
    input_channels: int = 18
    piece_planes: tuple[int, ...] = tuple(range(12))
    side_to_move_plane: int = 12
    castling_planes: tuple[int, int, int, int] = (13, 14, 15, 16)
    en_passant_plane: int = 17

    @classmethod
    def simple_18(cls) -> "EncodingSemanticSpec":
        return cls()

    def validate(self, channels: int) -> None:
        if self.encoding != "simple_18" or channels != self.input_channels:
            raise ValueError(
                "ColorFlipOrbitEvidenceNet only has a deterministic semantic adapter for "
                f"simple_18 with 18 channels, got encoding={self.encoding!r}, channels={channels}"
            )


def _norm(channels: int, enabled: bool) -> nn.Module:
    if not enabled:
        return nn.Identity()
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ColorFlipOrbitAdapter(nn.Module):
    def __init__(
        self,
        semantic_spec: EncodingSemanticSpec | None = None,
        orbit_transform: str = "color_flip",
        fail_closed_unknown_channels: bool = True,
    ) -> None:
        super().__init__()
        self.semantic_spec = semantic_spec or EncodingSemanticSpec.simple_18()
        self.orbit_transform = str(orbit_transform)
        self.fail_closed_unknown_channels = bool(fail_closed_unknown_channels)
        self.register_buffer("bad_rank_order", torch.tensor([1, 0, 3, 2, 5, 4, 7, 6], dtype=torch.long))

    def _require_known(self, x: torch.Tensor) -> bool:
        try:
            self.semantic_spec.validate(x.shape[1])
        except ValueError:
            if self.fail_closed_unknown_channels:
                raise
            return False
        return True

    def color_flip(self, x: torch.Tensor) -> torch.Tensor:
        self.semantic_spec.validate(x.shape[1])
        rank_mirror = torch.flip(x, dims=[2])
        flipped = x.clone()
        flipped[:, 0:6] = rank_mirror[:, 6:12]
        flipped[:, 6:12] = rank_mirror[:, 0:6]
        flipped[:, 12:13] = 1.0 - rank_mirror[:, 12:13]
        flipped[:, 13:14] = x[:, 15:16]
        flipped[:, 14:15] = x[:, 16:17]
        flipped[:, 15:16] = x[:, 13:14]
        flipped[:, 16:17] = x[:, 14:15]
        flipped[:, 17:18] = rank_mirror[:, 17:18]
        return flipped

    def bad_rank_color(self, x: torch.Tensor) -> torch.Tensor:
        self.semantic_spec.validate(x.shape[1])
        rank_permuted = x.index_select(2, self.bad_rank_order.to(device=x.device))
        transformed = x.clone()
        transformed[:, 0:6] = rank_permuted[:, 6:12]
        transformed[:, 6:12] = rank_permuted[:, 0:6]
        transformed[:, 12:13] = 1.0 - rank_permuted[:, 12:13]
        transformed[:, 13:14] = x[:, 15:16]
        transformed[:, 14:15] = x[:, 16:17]
        transformed[:, 15:16] = x[:, 13:14]
        transformed[:, 16:17] = x[:, 14:15]
        transformed[:, 17:18] = rank_permuted[:, 17:18]
        return transformed

    def make_orbit(self, x: torch.Tensor) -> torch.Tensor:
        if self.orbit_transform == "identity":
            return torch.stack([x, x], dim=1)
        if not self._require_known(x):
            return torch.stack([x, x], dim=1)
        if self.orbit_transform == "color_flip":
            transformed = self.color_flip(x)
        elif self.orbit_transform in {"bad_rank_color", "bad_rank_color_orbit"}:
            transformed = self.bad_rank_color(x)
        else:
            raise ValueError(
                "Unsupported orbit_transform="
                f"{self.orbit_transform!r}; expected 'color_flip', 'bad_rank_color', or 'identity'"
            )
        return torch.stack([x, transformed], dim=1)


class ConvResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0, use_norm: bool = True) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_norm),
            _norm(channels, use_norm),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_norm),
            _norm(channels, use_norm),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class SharedBoardEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        hidden_channels: int = 96,
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        if num_res_blocks < 1:
            raise ValueError("num_res_blocks must be >= 1")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.hidden_channels = int(hidden_channels)
        mid_channels = max(16, self.hidden_channels // 2)
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, mid_channels, kernel_size=3, padding=1, bias=not use_norm),
            _norm(mid_channels, use_norm),
            nn.GELU(),
            nn.Conv2d(mid_channels, self.hidden_channels, kernel_size=3, padding=1, bias=not use_norm),
            _norm(self.hidden_channels, use_norm),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *(ConvResidualBlock(self.hidden_channels, dropout=dropout, use_norm=use_norm) for _ in range(num_res_blocks))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = require_board_tensor(x, self.spec)
        return self.blocks(self.stem(x))


class OrbitEvidenceIntersectionHead(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 96,
        latent_dim: int = 128,
        dropout: float = 0.1,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.projection = nn.Sequential(
            nn.Linear(hidden_channels, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.evidence_head = nn.Linear(latent_dim, 2)

    def forward(self, feature_map: torch.Tensor, batch_size: int) -> dict[str, torch.Tensor]:
        pooled = feature_map.mean(dim=(2, 3))
        z = self.projection(pooled).reshape(batch_size, 2, -1)
        evidence = F.softplus(self.evidence_head(z)) + self.eps
        e0 = evidence[:, 0, :]
        e1 = evidence[:, 1, :]
        intersection = (2.0 * e0 * e1) / (e0 + e1 + self.eps)
        two_class_logits = torch.log1p(intersection)
        return {
            "z": z,
            "evidence": evidence,
            "intersection": intersection,
            "two_class_logits": two_class_logits,
        }


class ColorFlipOrbitEvidenceNet(nn.Module):
    """Harmonic evidence-intersection classifier over exact color-flip orbit views."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        encoding: str = "simple_18",
        hidden_channels: int = 96,
        latent_dim: int = 128,
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        eps: float = 1.0e-6,
        orbit_transform: str = "color_flip",
        intersection: str = "harmonic",
        fail_closed_unknown_channels: bool = True,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        if num_classes not in {1, 2}:
            raise ValueError("ColorFlipOrbitEvidenceNet supports one-logit BCE or two-class CE outputs")
        if intersection != "harmonic":
            raise ValueError("Only harmonic evidence intersection is implemented for CFOEB")
        self.num_classes = int(num_classes)
        self.eps = float(eps)
        self.spec = BoardTensorSpec(input_channels=input_channels)
        semantic_spec = EncodingSemanticSpec(encoding=encoding, input_channels=input_channels)
        self.adapter = ColorFlipOrbitAdapter(
            semantic_spec=semantic_spec,
            orbit_transform=orbit_transform,
            fail_closed_unknown_channels=fail_closed_unknown_channels,
        )
        self.encoder = SharedBoardEncoder(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            use_norm=use_norm,
        )
        self.head = OrbitEvidenceIntersectionHead(
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            dropout=dropout,
            eps=eps,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        orbit = self.adapter.make_orbit(x)
        batch_size, orbit_size, channels, height, width = orbit.shape
        if orbit_size != 2:
            raise ValueError("ColorFlipOrbitEvidenceNet expects exactly two orbit views")
        flat_orbit = orbit.reshape(batch_size * orbit_size, channels, height, width)
        feature_map = self.encoder(flat_orbit)
        head = self.head(feature_map, batch_size=batch_size)
        two_class_logits = head["two_class_logits"]
        logits = two_class_logits if self.num_classes == 2 else two_class_logits[:, 1] - two_class_logits[:, 0]
        evidence = head["evidence"]
        intersection = head["intersection"]
        z = head["z"]
        view_gap = (evidence[:, 0, :] - evidence[:, 1, :]).abs()
        puzzle_gap = view_gap[:, 1]
        negative_gap = view_gap[:, 0]
        return {
            "logits": logits,
            "negative_evidence": intersection[:, 0],
            "puzzle_evidence": intersection[:, 1],
            "evidence_balance": intersection[:, 1] - intersection[:, 0],
            "view_negative_evidence_gap": negative_gap,
            "view_puzzle_evidence_gap": puzzle_gap,
            "orbit_evidence_residual": view_gap.mean(dim=1),
            "symmetry_residual": puzzle_gap,
            "latent_orbit_variance": z.var(dim=1, unbiased=False).mean(dim=1),
            "identity_puzzle_evidence": evidence[:, 0, 1],
            "flipped_puzzle_evidence": evidence[:, 1, 1],
            "identity_negative_evidence": evidence[:, 0, 0],
            "flipped_negative_evidence": evidence[:, 1, 0],
            "intersection_energy": intersection.pow(2).mean(dim=1),
        }


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = config.get("model") if isinstance(config.get("model"), dict) else config
    return dict(model_cfg)


def build_color_flip_orbit_evidence_bottleneck_from_config(config: dict[str, Any]) -> ColorFlipOrbitEvidenceNet:
    model_cfg = _model_config(config)
    encoding = str(model_cfg.get("encoding", model_cfg.get("channel_schema", "simple_18")))
    return ColorFlipOrbitEvidenceNet(
        input_channels=int(model_cfg.get("input_channels", 18)),
        num_classes=int(model_cfg.get("num_classes", 1)),
        encoding=encoding,
        hidden_channels=int(model_cfg.get("hidden_channels", model_cfg.get("channels", 96))),
        latent_dim=int(model_cfg.get("latent_dim", model_cfg.get("hidden_dim", 128))),
        num_res_blocks=int(model_cfg.get("num_res_blocks", model_cfg.get("depth", 2))),
        dropout=float(model_cfg.get("dropout", 0.1)),
        eps=float(model_cfg.get("eps", 1.0e-6)),
        orbit_transform=str(model_cfg.get("orbit_transform", "color_flip")),
        intersection=str(model_cfg.get("intersection", "harmonic")),
        fail_closed_unknown_channels=bool(model_cfg.get("fail_closed_unknown_channels", True)),
        use_norm=bool(model_cfg.get("use_norm", model_cfg.get("use_batchnorm", True))),
    )


def build_color_flip_orbit_evidence(config: dict[str, Any]) -> ColorFlipOrbitEvidenceNet:
    return build_color_flip_orbit_evidence_bottleneck_from_config(config)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


_EPS = 1.0e-6


@dataclass(frozen=True)
class Simple18RoleSpec:
    encoding: str = "simple_18"
    input_channels: int = 18
    side_to_move_channel: int = 12

    def validate(self, channels: int) -> None:
        if self.encoding != "simple_18" or channels != self.input_channels:
            raise ValueError(
                "KingAnchoredEulerInteractionNet only has deterministic role extraction "
                f"for simple_18 with 18 channels, got encoding={self.encoding!r}, channels={channels}"
            )


def _default_role_pairs() -> tuple[tuple[int, int], ...]:
    return (
        (2, 7),
        (1, 7),
        (0, 7),
        (2, 6),
        (1, 6),
        (6, 3),
        (5, 3),
        (4, 3),
    )


def _directions_king8() -> torch.Tensor:
    directions = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [1.0, 1.0],
            [-1.0, -1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    return directions / directions.abs().sum(dim=1, keepdim=True).clamp_min(1.0)


class Simple18RoleAdapter(nn.Module):
    """Converts simple_18 piece planes into side-relative role masks and safe context."""

    role_names = (
        "own_pawn",
        "own_minor",
        "own_heavy",
        "own_king",
        "opp_pawn",
        "opp_minor",
        "opp_heavy",
        "opp_king",
    )
    context_dim = 29

    def __init__(
        self,
        encoding: str = "simple_18",
        input_channels: int = 18,
        fail_closed_unknown_channels: bool = True,
    ) -> None:
        super().__init__()
        self.spec = Simple18RoleSpec(encoding=encoding, input_channels=input_channels)
        self.fail_closed_unknown_channels = bool(fail_closed_unknown_channels)
        rows = torch.arange(8, dtype=torch.float32)
        cols = torch.arange(8, dtype=torch.float32)
        row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
        self.register_buffer("cell_coords", torch.stack([row_grid, col_grid], dim=-1), persistent=False)

    def _validate(self, x: torch.Tensor) -> bool:
        try:
            self.spec.validate(x.shape[1])
        except ValueError:
            if self.fail_closed_unknown_channels:
                raise
            return False
        return True

    def _king_anchor(self, king_mask: torch.Tensor) -> torch.Tensor:
        weights = king_mask.clamp_min(0.0)
        denom = weights.sum(dim=(1, 2), keepdim=True)
        weighted = (weights.unsqueeze(-1) * self.cell_coords.to(device=king_mask.device, dtype=king_mask.dtype)).sum(
            dim=(1, 2)
        )
        center = king_mask.new_tensor([3.5, 3.5]).expand(king_mask.shape[0], 2)
        return torch.where(denom.view(-1, 1) > _EPS, weighted / denom.view(-1, 1).clamp_min(_EPS), center)

    def anchors(self, roles: torch.Tensor) -> torch.Tensor:
        own_king = self._king_anchor(roles[:, 3])
        opp_king = self._king_anchor(roles[:, 7])
        center = roles.new_tensor([3.5, 3.5]).expand(roles.shape[0], 2)
        return torch.stack([opp_king, own_king, center], dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._validate(x):
            batch = x.shape[0]
            roles = x.new_zeros((batch, 8, 8, 8))
            context = x.new_zeros((batch, self.context_dim))
            anchors = x.new_tensor([3.5, 3.5]).expand(batch, 3, 2)
            return roles, anchors, context

        side_white = x[:, 12:13].mean(dim=(2, 3)).clamp(0.0, 1.0)
        side_black = 1.0 - side_white
        white_pawn = x[:, 0]
        white_minor = (x[:, 1] + x[:, 2]).clamp(0.0, 1.0)
        white_heavy = (x[:, 3] + x[:, 4]).clamp(0.0, 1.0)
        white_king = x[:, 5].clamp(0.0, 1.0)
        black_pawn = x[:, 6]
        black_minor = (x[:, 7] + x[:, 8]).clamp(0.0, 1.0)
        black_heavy = (x[:, 9] + x[:, 10]).clamp(0.0, 1.0)
        black_king = x[:, 11].clamp(0.0, 1.0)

        def side_mix(white: torch.Tensor, black: torch.Tensor) -> torch.Tensor:
            return side_white.view(-1, 1, 1) * white + side_black.view(-1, 1, 1) * black

        def opp_mix(white: torch.Tensor, black: torch.Tensor) -> torch.Tensor:
            return side_black.view(-1, 1, 1) * white + side_white.view(-1, 1, 1) * black

        roles = torch.stack(
            [
                side_mix(white_pawn, black_pawn),
                side_mix(white_minor, black_minor),
                side_mix(white_heavy, black_heavy),
                side_mix(white_king, black_king),
                opp_mix(white_pawn, black_pawn),
                opp_mix(white_minor, black_minor),
                opp_mix(white_heavy, black_heavy),
                opp_mix(white_king, black_king),
            ],
            dim=1,
        ).clamp(0.0, 1.0)
        anchors = self.anchors(roles)

        piece_counts = x[:, :12].sum(dim=(2, 3))
        role_counts = roles.sum(dim=(2, 3))
        castling = x[:, 13:17].mean(dim=(2, 3)).clamp(0.0, 1.0)
        ep_count = x[:, 17:18].sum(dim=(2, 3)).clamp(0.0, 1.0)
        king_delta = (anchors[:, 0] - anchors[:, 1]).abs()
        king_distance = torch.stack(
            [
                king_delta[:, 0] / 7.0,
                king_delta[:, 1] / 7.0,
                king_delta.max(dim=1).values / 7.0,
            ],
            dim=1,
        )
        context = torch.cat([piece_counts, role_counts, side_white, castling, ep_count, king_distance], dim=1)
        return roles, anchors, context


def _cubical_chi(face_masks: torch.Tensor) -> torch.Tensor:
    original_shape = face_masks.shape[:-2]
    faces = face_masks.reshape(-1, 8, 8).clamp(0.0, 1.0)
    face_count = faces.sum(dim=(1, 2))

    horizontal_edges = faces.new_zeros((faces.shape[0], 9, 8))
    horizontal_edges[:, :-1, :] = torch.maximum(horizontal_edges[:, :-1, :], faces)
    horizontal_edges[:, 1:, :] = torch.maximum(horizontal_edges[:, 1:, :], faces)

    vertical_edges = faces.new_zeros((faces.shape[0], 8, 9))
    vertical_edges[:, :, :-1] = torch.maximum(vertical_edges[:, :, :-1], faces)
    vertical_edges[:, :, 1:] = torch.maximum(vertical_edges[:, :, 1:], faces)

    vertices = faces.new_zeros((faces.shape[0], 9, 9))
    vertices[:, :-1, :-1] = torch.maximum(vertices[:, :-1, :-1], faces)
    vertices[:, 1:, :-1] = torch.maximum(vertices[:, 1:, :-1], faces)
    vertices[:, :-1, 1:] = torch.maximum(vertices[:, :-1, 1:], faces)
    vertices[:, 1:, 1:] = torch.maximum(vertices[:, 1:, 1:], faces)

    edge_count = horizontal_edges.sum(dim=(1, 2)) + vertical_edges.sum(dim=(1, 2))
    vertex_count = vertices.sum(dim=(1, 2))
    return (vertex_count - edge_count + face_count).reshape(original_shape)


class CubicalEulerCurveLayer(nn.Module):
    def __init__(self, num_thresholds: int = 15, directions: str = "king8") -> None:
        super().__init__()
        if num_thresholds < 2:
            raise ValueError("num_thresholds must be >= 2")
        if directions != "king8":
            raise ValueError("Only directions='king8' is implemented for king-anchored Euler curves")
        rows = torch.arange(8, dtype=torch.float32)
        cols = torch.arange(8, dtype=torch.float32)
        row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
        self.register_buffer("cell_coords", torch.stack([row_grid, col_grid], dim=-1), persistent=False)
        self.register_buffer("directions", _directions_king8(), persistent=False)
        self.register_buffer("thresholds", torch.linspace(-7.0, 7.0, num_thresholds), persistent=False)

    @property
    def num_directions(self) -> int:
        return int(self.directions.shape[0])

    @property
    def num_thresholds(self) -> int:
        return int(self.thresholds.shape[0])

    def _gates(self, anchors: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        coords = self.cell_coords.to(device=anchors.device, dtype=dtype)
        directions = self.directions.to(device=anchors.device, dtype=dtype)
        thresholds = self.thresholds.to(device=anchors.device, dtype=dtype)
        centered = coords.view(1, 1, 1, 1, 8, 8, 2) - anchors.to(dtype=dtype).view(-1, anchors.shape[1], 1, 1, 1, 1, 2)
        projections = (centered * directions.view(1, 1, -1, 1, 1, 1, 2)).sum(dim=-1)
        return (projections <= thresholds.view(1, 1, 1, -1, 1, 1)).to(dtype=dtype)

    def face_count_curve(self, mask: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        gates = self._gates(anchors, dtype=mask.dtype)
        selected = mask[:, None, None, None] * gates
        return selected.sum(dim=(-1, -2))

    def forward(self, mask: torch.Tensor, anchors: torch.Tensor, mode: str = "euler") -> torch.Tensor:
        gates = self._gates(anchors, dtype=mask.dtype)
        selected = mask[:, None, None, None] * gates
        if mode == "face_count":
            return selected.sum(dim=(-1, -2))
        if mode != "euler":
            raise ValueError(f"Unsupported curve mode {mode!r}")
        return _cubical_chi(selected)


class EulerInteractionFeatureBuilder(nn.Module):
    def __init__(
        self,
        num_thresholds: int = 15,
        directions: str = "king8",
        anchors: tuple[str, ...] = ("opp_king", "own_king", "center"),
        interaction_pairs: tuple[tuple[int, int], ...] | None = None,
        include_first_differences: bool = True,
        include_count_summaries: bool = True,
        include_interactions: bool = True,
        curve_mode: str = "euler",
        curve_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if anchors != ("opp_king", "own_king", "center"):
            raise ValueError("KingAnchoredEulerInteractionNet currently implements anchors opp_king, own_king, center")
        self.curves = CubicalEulerCurveLayer(num_thresholds=num_thresholds, directions=directions)
        self.interaction_pairs = tuple(interaction_pairs or _default_role_pairs())
        self.include_first_differences = bool(include_first_differences)
        self.include_count_summaries = bool(include_count_summaries)
        self.include_interactions = bool(include_interactions)
        self.curve_mode = str(curve_mode)
        self.curve_dropout = float(curve_dropout)
        per_curve = self.curves.num_thresholds + (
            self.curves.num_thresholds - 1 if self.include_first_differences else 0
        )
        curve_families = 8 + (len(self.interaction_pairs) if self.include_interactions else 0)
        self.feature_dim = curve_families * 3 * self.curves.num_directions * per_curve
        if self.include_count_summaries:
            self.feature_dim += Simple18RoleAdapter.context_dim

    @staticmethod
    def _with_delta(curve: torch.Tensor) -> torch.Tensor:
        return torch.cat([curve.flatten(1), curve.diff(dim=-1).flatten(1)], dim=1)

    def _curve_features(self, curve: torch.Tensor) -> torch.Tensor:
        if self.include_first_differences:
            return self._with_delta(curve)
        return curve.flatten(1)

    def forward(self, roles: torch.Tensor, anchors: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        role_curves: list[torch.Tensor] = []
        features: list[torch.Tensor] = []
        for role_index in range(roles.shape[1]):
            curve = self.curves(roles[:, role_index], anchors, mode=self.curve_mode)
            role_curves.append(curve)
            features.append(self._curve_features(curve))

        interaction_values: list[torch.Tensor] = []
        if self.include_interactions:
            for left, right in self.interaction_pairs:
                union = (roles[:, left] + roles[:, right]).clamp(0.0, 1.0)
                union_curve = self.curves(union, anchors, mode=self.curve_mode)
                interaction = union_curve - role_curves[left] - role_curves[right]
                interaction_values.append(interaction)
                features.append(self._curve_features(interaction))

        euler_features = torch.cat(features, dim=1)
        if self.training and self.curve_dropout > 0.0:
            euler_features = F.dropout(euler_features, p=self.curve_dropout, training=True)
        all_features = torch.cat([euler_features, context], dim=1) if self.include_count_summaries else euler_features
        interaction_energy = (
            torch.stack([item.pow(2).mean(dim=(1, 2, 3)) for item in interaction_values], dim=1).mean(dim=1)
            if interaction_values
            else roles.new_zeros(roles.shape[0])
        )
        diagnostics = {
            "euler_feature_energy": euler_features.pow(2).mean(dim=1),
            "euler_interaction_energy": interaction_energy,
            "euler_curve_variation": euler_features.var(dim=1, unbiased=False),
            "role_count_energy": context[:, 12:20].pow(2).mean(dim=1) if context.shape[1] >= 20 else context.pow(2).mean(dim=1),
        }
        return all_features, diagnostics


class EulerFeatureMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, second_hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, second_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(second_hidden_dim, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class KingAnchoredEulerInteractionNet(nn.Module):
    """Classifier over king-anchored cubical Euler curves and additivity interactions."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        encoding: str = "simple_18",
        num_thresholds: int = 15,
        directions: str = "king8",
        anchors: tuple[str, ...] = ("opp_king", "own_king", "center"),
        interaction_pairs: tuple[tuple[int, int], ...] | None = None,
        include_first_differences: bool = True,
        include_count_summaries: bool = True,
        include_interactions: bool = True,
        curve_mode: str = "euler",
        hidden_dim: int = 128,
        second_hidden_dim: int = 64,
        dropout: float = 0.1,
        curve_dropout: float = 0.0,
        fail_closed_unknown_channels: bool = True,
    ) -> None:
        super().__init__()
        if num_classes not in {1, 2}:
            raise ValueError("KingAnchoredEulerInteractionNet supports one-logit BCE or two-class CE outputs")
        self.num_classes = int(num_classes)
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.adapter = Simple18RoleAdapter(
            encoding=encoding,
            input_channels=input_channels,
            fail_closed_unknown_channels=fail_closed_unknown_channels,
        )
        self.feature_builder = EulerInteractionFeatureBuilder(
            num_thresholds=num_thresholds,
            directions=directions,
            anchors=anchors,
            interaction_pairs=interaction_pairs,
            include_first_differences=include_first_differences,
            include_count_summaries=include_count_summaries,
            include_interactions=include_interactions,
            curve_mode=curve_mode,
            curve_dropout=curve_dropout,
        )
        self.head = EulerFeatureMLP(
            input_dim=self.feature_builder.feature_dim,
            hidden_dim=hidden_dim,
            second_hidden_dim=second_hidden_dim,
            dropout=dropout,
        )

    def _primary_logits(self, two_class_logits: torch.Tensor) -> torch.Tensor:
        if self.num_classes == 2:
            return two_class_logits
        return two_class_logits[:, 1] - two_class_logits[:, 0]

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        roles, anchors, context = self.adapter(x)
        features, diagnostics = self.feature_builder(roles, anchors, context)
        two_class_logits = self.head(features)
        logits = self._primary_logits(two_class_logits)
        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "two_class_logits": two_class_logits,
            "topology_pressure": diagnostics["euler_interaction_energy"],
            "euler_feature_energy": diagnostics["euler_feature_energy"],
            "euler_interaction_energy": diagnostics["euler_interaction_energy"],
            "euler_curve_variation": diagnostics["euler_curve_variation"],
            "role_count_energy": diagnostics["role_count_energy"],
            "king_ring_pressure": context[:, -1],
            "mechanism_energy": features.pow(2).mean(dim=1),
        }
        if return_aux:
            output.update(
                {
                    "roles": roles,
                    "anchors": anchors,
                    "euler_features": features,
                    "context_summaries": context,
                }
            )
        return output


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = config.get("model") if isinstance(config.get("model"), dict) else config
    return dict(model_cfg)


def _encoding_from_config(config: dict[str, Any], model_cfg: dict[str, Any]) -> str:
    if "encoding_adapter" in model_cfg:
        return str(model_cfg["encoding_adapter"])
    if "encoding" in model_cfg:
        return str(model_cfg["encoding"])
    data_cfg = config.get("data") if isinstance(config.get("data"), dict) else {}
    return str(data_cfg.get("encoding", "simple_18"))


def _anchors(value: Any) -> tuple[str, ...]:
    if value is None:
        return ("opp_king", "own_king", "center")
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _interaction_pairs(value: Any) -> tuple[tuple[int, int], ...] | None:
    if value is None or value == "default8":
        return None
    return tuple((int(left), int(right)) for left, right in value)


def build_king_anchored_euler_interaction_network_from_config(config: dict[str, Any]) -> KingAnchoredEulerInteractionNet:
    model_cfg = _model_config(config)
    ablation = str(model_cfg.get("ablation", "none"))
    include_interactions = bool(model_cfg.get("include_interactions", ablation not in {"no_euler_interaction", "individual_euler_only"}))
    curve_mode = "face_count" if ablation == "face_count_curves_only" else str(model_cfg.get("curve_mode", "euler"))
    hidden_dim = int(model_cfg.get("hidden_dim", 128))
    return KingAnchoredEulerInteractionNet(
        input_channels=int(model_cfg.get("input_channels", 18)),
        num_classes=int(model_cfg.get("num_classes", 1)),
        encoding=_encoding_from_config(config, model_cfg),
        num_thresholds=int(model_cfg.get("num_thresholds", 15)),
        directions=str(model_cfg.get("directions", "king8")),
        anchors=_anchors(model_cfg.get("anchors")),
        interaction_pairs=_interaction_pairs(model_cfg.get("interaction_pairs")),
        include_first_differences=bool(model_cfg.get("include_first_differences", True)),
        include_count_summaries=bool(model_cfg.get("include_count_summaries", True)),
        include_interactions=include_interactions,
        curve_mode=curve_mode,
        hidden_dim=hidden_dim,
        second_hidden_dim=int(model_cfg.get("second_hidden_dim", max(16, hidden_dim // 2))),
        dropout=float(model_cfg.get("dropout", 0.1)),
        curve_dropout=float(model_cfg.get("curve_dropout", 0.0)),
        fail_closed_unknown_channels=bool(model_cfg.get("fail_closed_unknown_channels", True)),
    )


def build_king_anchored_euler_interaction(config: dict[str, Any]) -> KingAnchoredEulerInteractionNet:
    return build_king_anchored_euler_interaction_network_from_config(config)

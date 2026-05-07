from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


_EPS = 1.0e-6

# simple_18 plane indices
_WHITE_PIECE_PLANES = (0, 1, 2, 3, 4, 5)  # P N B R Q K
_BLACK_PIECE_PLANES = (6, 7, 8, 9, 10, 11)  # p n b r q k
_SIDE_TO_MOVE_CHANNEL = 12
_CASTLING_CHANNELS = (13, 14, 15, 16)
_EN_PASSANT_CHANNEL = 17

# Role indices into the (B, 8, 8, 8) role tensor
ROLE_OWN_PAWN = 0
ROLE_OWN_MINOR = 1
ROLE_OWN_HEAVY = 2
ROLE_OWN_KING = 3
ROLE_OPP_PAWN = 4
ROLE_OPP_MINOR = 5
ROLE_OPP_HEAVY = 6
ROLE_OPP_KING = 7

DEFAULT_INTERACTION_PAIRS: tuple[tuple[int, int], ...] = (
    (ROLE_OWN_HEAVY, ROLE_OPP_KING),
    (ROLE_OWN_MINOR, ROLE_OPP_KING),
    (ROLE_OWN_PAWN, ROLE_OPP_KING),
    (ROLE_OWN_HEAVY, ROLE_OPP_HEAVY),
    (ROLE_OWN_MINOR, ROLE_OPP_HEAVY),
    (ROLE_OPP_HEAVY, ROLE_OWN_KING),
    (ROLE_OPP_MINOR, ROLE_OWN_KING),
    (ROLE_OPP_PAWN, ROLE_OWN_KING),
)

DEFAULT_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (-1, -1),
    (1, -1),
    (-1, 1),
)


@dataclass(frozen=True)
class Simple18EulerSpec:
    encoding: str = "simple_18"
    input_channels: int = 18

    def validate(self, channels: int) -> None:
        if self.encoding != "simple_18" or channels != self.input_channels:
            raise ValueError(
                "KingAnchoredEulerInteractionNet only supports simple_18 with 18 channels, "
                f"got encoding={self.encoding!r}, channels={channels}"
            )


class Simple18RoleAdapter(nn.Module):
    """Map simple_18 piece planes + side-to-move into 8 side-relative role masks."""

    def __init__(
        self,
        encoding: str = "simple_18",
        input_channels: int = 18,
        fail_closed_unknown_channels: bool = True,
    ) -> None:
        super().__init__()
        self.spec = Simple18EulerSpec(encoding=encoding, input_channels=input_channels)
        self.fail_closed_unknown_channels = bool(fail_closed_unknown_channels)

    def _validate(self, x: torch.Tensor) -> None:
        try:
            self.spec.validate(x.shape[1])
        except ValueError:
            if self.fail_closed_unknown_channels:
                raise

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate(x)
        # side-to-move plane is broadcast across all 8x8 cells; use the mean for safety.
        side_to_move = x[:, _SIDE_TO_MOVE_CHANNEL].flatten(1).mean(dim=1).clamp(0.0, 1.0)
        white_to_move = side_to_move.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)

        white_pawn = x[:, 0:1]
        white_minor = x[:, 1:3].sum(dim=1, keepdim=True).clamp(0.0, 1.0)  # N + B
        white_heavy = x[:, 3:5].sum(dim=1, keepdim=True).clamp(0.0, 1.0)  # R + Q
        white_king = x[:, 5:6]
        black_pawn = x[:, 6:7]
        black_minor = x[:, 7:9].sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        black_heavy = x[:, 9:11].sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        black_king = x[:, 11:12]

        own_pawn = white_pawn * white_to_move + black_pawn * (1.0 - white_to_move)
        own_minor = white_minor * white_to_move + black_minor * (1.0 - white_to_move)
        own_heavy = white_heavy * white_to_move + black_heavy * (1.0 - white_to_move)
        own_king = white_king * white_to_move + black_king * (1.0 - white_to_move)
        opp_pawn = black_pawn * white_to_move + white_pawn * (1.0 - white_to_move)
        opp_minor = black_minor * white_to_move + white_minor * (1.0 - white_to_move)
        opp_heavy = black_heavy * white_to_move + white_heavy * (1.0 - white_to_move)
        opp_king = black_king * white_to_move + white_king * (1.0 - white_to_move)

        roles = torch.cat(
            [own_pawn, own_minor, own_heavy, own_king, opp_pawn, opp_minor, opp_heavy, opp_king],
            dim=1,
        )

        # Context features: counts and rule bits.
        counts = roles.flatten(2).sum(dim=2)  # (B, 8)
        material_white = (
            white_pawn.flatten(1).sum(dim=1)
            + 3.0 * white_minor.flatten(1).sum(dim=1)
            + 5.0 * white_heavy.flatten(1).sum(dim=1)
        )
        material_black = (
            black_pawn.flatten(1).sum(dim=1)
            + 3.0 * black_minor.flatten(1).sum(dim=1)
            + 5.0 * black_heavy.flatten(1).sum(dim=1)
        )
        material_diff = material_white - material_black
        white_castling = x[:, list(_CASTLING_CHANNELS[:2])].flatten(1).amax(dim=1)
        black_castling = x[:, list(_CASTLING_CHANNELS[2:])].flatten(1).amax(dim=1)
        en_passant_present = x[:, _EN_PASSANT_CHANNEL].flatten(1).amax(dim=1)
        side_bit = side_to_move

        context = torch.stack(
            [
                material_white,
                material_black,
                material_diff,
                white_castling,
                black_castling,
                en_passant_present,
                side_bit,
            ],
            dim=1,
        )
        context = torch.cat([counts, context], dim=1)  # (B, 8 + 7 = 15)
        return roles, context


def _extract_anchors(roles: torch.Tensor) -> torch.Tensor:
    """Compute (B, A=3, 2) anchor coordinates: opp_king, own_king, board center.

    Coordinates are (row, col) floats. Soft fallback: if a king mask is empty (malformed
    data) we use board center; if it has multiple cells we use the mass-weighted mean.
    """
    batch = roles.shape[0]
    device = roles.device
    dtype = roles.dtype
    rows = torch.arange(8, device=device, dtype=dtype).view(1, 8, 1).expand(1, 8, 8)
    cols = torch.arange(8, device=device, dtype=dtype).view(1, 1, 8).expand(1, 8, 8)

    own_king = roles[:, ROLE_OWN_KING]
    opp_king = roles[:, ROLE_OPP_KING]

    def _centroid(mask: torch.Tensor) -> torch.Tensor:
        weight = mask.flatten(1).sum(dim=1)
        valid = weight > 0
        row_sum = (mask * rows).flatten(1).sum(dim=1)
        col_sum = (mask * cols).flatten(1).sum(dim=1)
        weight_safe = weight.clamp_min(1.0)
        row_c = row_sum / weight_safe
        col_c = col_sum / weight_safe
        center_r = torch.full_like(row_c, 3.5)
        center_c = torch.full_like(col_c, 3.5)
        row_c = torch.where(valid, row_c, center_r)
        col_c = torch.where(valid, col_c, center_c)
        return torch.stack([row_c, col_c], dim=1)

    opp_anchor = _centroid(opp_king)
    own_anchor = _centroid(own_king)
    center = torch.full((batch, 2), 3.5, device=device, dtype=dtype)
    return torch.stack([opp_anchor, own_anchor, center], dim=1)


def _build_sweep_gates(
    anchors: torch.Tensor,
    directions: torch.Tensor,
    thresholds: torch.Tensor,
) -> torch.Tensor:
    """Compute (B, A, U, T, 8, 8) binary gate tensor for half-plane sweeps.

    gate[b, a, u, t, r, c] = 1 if <u, (r, c) - anchor[b, a]> <= thresholds[t].
    """
    device = anchors.device
    dtype = anchors.dtype
    rows = torch.arange(8, device=device, dtype=dtype).view(1, 1, 1, 1, 8, 1)
    cols = torch.arange(8, device=device, dtype=dtype).view(1, 1, 1, 1, 1, 8)
    anchor_r = anchors[..., 0].view(*anchors.shape[:-1], 1, 1, 1, 1)
    anchor_c = anchors[..., 1].view(*anchors.shape[:-1], 1, 1, 1, 1)
    dr = (rows - anchor_r)
    dc = (cols - anchor_c)
    u_r = directions[..., 0].view(1, 1, -1, 1, 1, 1)
    u_c = directions[..., 1].view(1, 1, -1, 1, 1, 1)
    projection = u_r * dr + u_c * dc  # (B, A, U, 1, 8, 8)
    tau = thresholds.view(1, 1, 1, -1, 1, 1)
    gate = (projection <= tau).to(dtype)
    return gate


def _cubical_chi(face_mask: torch.Tensor) -> torch.Tensor:
    """Compute Euler characteristic chi = V - E + F of a cubical complex on the 8x8 grid.

    Input shape: (..., 8, 8) face indicator (binary or in [0, 1]).
    Output shape: (...)
    The complex is the cubical closure: a vertex/edge is present iff at least one
    incident face is present. We hard-binarize via clamp (no-op for already-binary masks).
    """
    f = face_mask.clamp(0.0, 1.0)
    # Faces.
    face_count = f.flatten(-2).sum(dim=-1)
    # Horizontal edges between vertically adjacent faces (rows r, r+1).
    edge_h = torch.maximum(f[..., :-1, :], f[..., 1:, :]).flatten(-2).sum(dim=-1)
    # Vertical edges between horizontally adjacent faces (cols c, c+1).
    edge_v = torch.maximum(f[..., :, :-1], f[..., :, 1:]).flatten(-2).sum(dim=-1)
    # Boundary edges of the 2-cells: each face contributes 4 edges, but shared edges
    # are counted once. The cubical closure has every edge incident to at least one
    # selected face. Count distinct edges.
    # Total edges = boundary_edges (face perimeter, deduped) which equals:
    #   (#faces) * 4 - 2 * (#shared edges between adjacent selected faces)
    # Equivalently: count each grid edge whose at-least-one incident face is selected.
    # Horizontal-grid-edges (between rows r and r+1) above/below: 9 rows of edges (top and bottom of each face).
    # We compute distinct edges directly by indexing edge slots:
    # For an 8x8 face grid, there are 9*8 horizontal grid lines (top edges) and 8*9 vertical grid lines (left edges)
    # But each face has 4 incident grid edges; deduplication yields:
    #   horizontal edges (along x): 9 rows x 8 cols, present iff face above OR face below selected
    #   vertical edges (along y):   8 rows x 9 cols, present iff face left  OR face right selected
    h_above = F.pad(f, (0, 0, 1, 0))  # shape (..., 9, 8): face above (or zero outside)
    h_below = F.pad(f, (0, 0, 0, 1))  # shape (..., 9, 8): face below
    edges_h_total = torch.maximum(h_above, h_below).flatten(-2).sum(dim=-1)
    v_left = F.pad(f, (1, 0, 0, 0))  # (..., 8, 9)
    v_right = F.pad(f, (0, 1, 0, 0))  # (..., 8, 9)
    edges_v_total = torch.maximum(v_left, v_right).flatten(-2).sum(dim=-1)
    edge_count = edges_h_total + edges_v_total
    # Vertices: 9x9 grid; vertex present iff any of (up to 4) incident faces is selected.
    vu_tl = F.pad(f, (1, 0, 1, 0))  # vertex's bottom-right face shifted to position
    vu_tr = F.pad(f, (0, 1, 1, 0))
    vu_bl = F.pad(f, (1, 0, 0, 1))
    vu_br = F.pad(f, (0, 1, 0, 1))
    v_grid = torch.maximum(torch.maximum(vu_tl, vu_tr), torch.maximum(vu_bl, vu_br))
    vertex_count = v_grid.flatten(-2).sum(dim=-1)
    chi = vertex_count - edge_count + face_count
    # Suppress unused intermediate (silences linter / keeps code closer to math).
    del edge_h, edge_v
    return chi


class CubicalEulerCurveLayer(nn.Module):
    """Compute (B, R or P, A, U, T) Euler curves for swept role complexes."""

    def __init__(
        self,
        directions: tuple[tuple[int, int], ...] = DEFAULT_DIRECTIONS,
        thresholds: tuple[float, ...] = tuple(float(t) for t in range(-7, 8)),
    ) -> None:
        super().__init__()
        self.register_buffer(
            "directions",
            torch.tensor(directions, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "thresholds",
            torch.tensor(thresholds, dtype=torch.float32),
            persistent=False,
        )

    @property
    def num_directions(self) -> int:
        return int(self.directions.shape[0])

    @property
    def num_thresholds(self) -> int:
        return int(self.thresholds.shape[0])

    def gates(self, anchors: torch.Tensor) -> torch.Tensor:
        return _build_sweep_gates(anchors, self.directions.to(anchors.dtype), self.thresholds.to(anchors.dtype))

    def sweep_one(self, role_mask: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        """role_mask: (B, 8, 8). gates: (B, A, U, T, 8, 8). Returns (B, A, U, T)."""
        # broadcast: (B, 1, 1, 1, 8, 8) * (B, A, U, T, 8, 8)
        masked = role_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3) * gates
        return _cubical_chi(masked)


class EulerInteractionFeatureBuilder(nn.Module):
    """Stack individual Euler curves and pairwise Euler interaction curves into a feature vector."""

    def __init__(
        self,
        interaction_pairs: tuple[tuple[int, int], ...] = DEFAULT_INTERACTION_PAIRS,
        directions: tuple[tuple[int, int], ...] = DEFAULT_DIRECTIONS,
        thresholds: tuple[float, ...] = tuple(float(t) for t in range(-7, 8)),
        num_roles: int = 8,
        include_first_differences: bool = True,
    ) -> None:
        super().__init__()
        self.curves = CubicalEulerCurveLayer(directions=directions, thresholds=thresholds)
        self.interaction_pairs = tuple((int(r), int(s)) for r, s in interaction_pairs)
        self.num_roles = int(num_roles)
        self.include_first_differences = bool(include_first_differences)

    @property
    def num_anchors(self) -> int:
        return 3

    @property
    def num_directions(self) -> int:
        return self.curves.num_directions

    @property
    def num_thresholds(self) -> int:
        return self.curves.num_thresholds

    def feature_dim(self) -> int:
        per_curve = self.num_anchors * self.num_directions * self.num_thresholds
        per_diff = self.num_anchors * self.num_directions * max(0, self.num_thresholds - 1)
        slabs = self.num_roles + len(self.interaction_pairs)
        if self.include_first_differences:
            return slabs * (per_curve + per_diff)
        return slabs * per_curve

    def forward(
        self,
        roles: torch.Tensor,
        anchors: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        gates = self.curves.gates(anchors)  # (B, A, U, T, 8, 8)
        feature_blocks: list[torch.Tensor] = []
        diff_blocks: list[torch.Tensor] = []

        per_role_curves: list[torch.Tensor] = []
        for r in range(self.num_roles):
            role_mask = roles[:, r]
            curve = self.curves.sweep_one(role_mask, gates)  # (B, A, U, T)
            per_role_curves.append(curve)
            feature_blocks.append(curve.flatten(1))
            if self.include_first_differences:
                diff_blocks.append(curve.diff(dim=-1).flatten(1))

        interaction_curves: list[torch.Tensor] = []
        for r, s in self.interaction_pairs:
            union_mask = torch.clamp(roles[:, r] + roles[:, s], 0.0, 1.0)
            curve_union = self.curves.sweep_one(union_mask, gates)
            interaction = curve_union - per_role_curves[r] - per_role_curves[s]
            interaction_curves.append(interaction)
            feature_blocks.append(interaction.flatten(1))
            if self.include_first_differences:
                diff_blocks.append(interaction.diff(dim=-1).flatten(1))

        features = torch.cat(feature_blocks + diff_blocks, dim=1)

        # Diagnostics summarised per sample.
        role_curve_stack = torch.stack(per_role_curves, dim=1)  # (B, R, A, U, T)
        interaction_stack = torch.stack(interaction_curves, dim=1) if interaction_curves else features.new_zeros(
            features.shape[0], 0
        )

        diagnostics = {
            "role_curves": role_curve_stack,
            "interaction_curves": interaction_stack,
        }
        return {"features": features, "diagnostics": diagnostics}


class EulerFeatureMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        second_hidden_dim: int = 64,
        num_classes: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes not in {1, 2}:
            raise ValueError("num_classes must be 1 (BCE logit) or 2 (CE)")
        self.num_classes = int(num_classes)
        self.norm = nn.LayerNorm(input_dim)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, second_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.head = nn.Linear(second_hidden_dim, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.layers(self.norm(features)))


class KingAnchoredEulerInteractionNet(nn.Module):
    """King-anchored cubical Euler interaction classifier per i050."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        encoding: str = "simple_18",
        num_thresholds: int = 15,
        directions: tuple[tuple[int, int], ...] = DEFAULT_DIRECTIONS,
        interaction_pairs: tuple[tuple[int, int], ...] = DEFAULT_INTERACTION_PAIRS,
        include_first_differences: bool = True,
        include_count_summaries: bool = True,
        hidden_dim: int = 128,
        second_hidden_dim: int = 64,
        dropout: float = 0.1,
        curve_dropout: float = 0.05,
        fail_closed_unknown_channels: bool = True,
    ) -> None:
        super().__init__()
        if num_classes not in {1, 2}:
            raise ValueError("KingAnchoredEulerInteractionNet supports num_classes in {1, 2}")
        if num_thresholds < 2:
            raise ValueError("num_thresholds must be >= 2")
        self.num_classes = int(num_classes)
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.include_count_summaries = bool(include_count_summaries)
        self.include_first_differences = bool(include_first_differences)
        self.curve_dropout = float(curve_dropout)
        thresholds = tuple(
            float(t) for t in torch.linspace(-7.0, 7.0, num_thresholds).tolist()
        )
        self.adapter = Simple18RoleAdapter(
            encoding=encoding,
            input_channels=input_channels,
            fail_closed_unknown_channels=fail_closed_unknown_channels,
        )
        self.feature_builder = EulerInteractionFeatureBuilder(
            interaction_pairs=interaction_pairs,
            directions=directions,
            thresholds=thresholds,
            num_roles=8,
            include_first_differences=include_first_differences,
        )
        feature_dim = self.feature_builder.feature_dim()
        context_dim = 15 if include_count_summaries else 0  # 8 role counts + 7 rule scalars
        self.head = EulerFeatureMLP(
            input_dim=feature_dim + context_dim,
            hidden_dim=hidden_dim,
            second_hidden_dim=second_hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
        self._feature_dim = feature_dim
        self._context_dim = context_dim

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def context_dim(self) -> int:
        return self._context_dim

    def _primary_logits(self, two_class_logits: torch.Tensor) -> torch.Tensor:
        if self.num_classes == 2:
            return two_class_logits
        return two_class_logits[:, 1] - two_class_logits[:, 0]

    def _apply_curve_dropout(self, features: torch.Tensor) -> torch.Tensor:
        if not self.training or self.curve_dropout <= 0.0:
            return features
        # Bernoulli mask over the directions axis for both individual and interaction curves.
        # Reconstruct the layout per slab to drop entire (anchor, direction) tubes uniformly.
        return F.dropout(features, p=self.curve_dropout, training=True)

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        roles, context = self.adapter(x)
        anchors = _extract_anchors(roles)
        feature_pack = self.feature_builder(roles, anchors)
        features = feature_pack["features"]
        diagnostics = feature_pack["diagnostics"]

        features = self._apply_curve_dropout(features)
        if self.include_count_summaries:
            full_features = torch.cat([features, context], dim=1)
        else:
            full_features = features

        two_class_logits = self.head(full_features)
        logits = self._primary_logits(two_class_logits)

        role_curves = diagnostics["role_curves"]  # (B, R, A, U, T)
        interaction_curves = diagnostics["interaction_curves"]  # (B, P, A, U, T)

        role_energy = role_curves.pow(2).flatten(1).mean(dim=1)
        interaction_energy = interaction_curves.pow(2).flatten(1).mean(dim=1) if interaction_curves.numel() > 0 else torch.zeros_like(role_energy)
        # King-ring (anchor=0, opponent king) interaction pressure summary.
        if interaction_curves.numel() > 0:
            king_anchor_interaction = interaction_curves[:, :, 0].abs().flatten(1).mean(dim=1)
            own_king_anchor_interaction = interaction_curves[:, :, 1].abs().flatten(1).mean(dim=1)
            center_anchor_interaction = interaction_curves[:, :, 2].abs().flatten(1).mean(dim=1)
        else:
            king_anchor_interaction = torch.zeros_like(role_energy)
            own_king_anchor_interaction = torch.zeros_like(role_energy)
            center_anchor_interaction = torch.zeros_like(role_energy)

        own_role_count = roles[:, :4].flatten(1).sum(dim=1)
        opp_role_count = roles[:, 4:].flatten(1).sum(dim=1)

        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "two_class_logits": two_class_logits,
            "role_curve_energy": role_energy,
            "interaction_curve_energy": interaction_energy,
            "opp_king_interaction_pressure": king_anchor_interaction,
            "own_king_interaction_pressure": own_king_anchor_interaction,
            "center_interaction_pressure": center_anchor_interaction,
            "own_role_count": own_role_count,
            "opp_role_count": opp_role_count,
        }
        if return_aux:
            output.update(
                {
                    "roles": roles,
                    "anchors": anchors,
                    "context": context,
                    "features": features,
                    "role_curves": role_curves,
                    "interaction_curves": interaction_curves,
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


def _resolve_directions(value: Any) -> tuple[tuple[int, int], ...]:
    if value is None or value == "king8":
        return DEFAULT_DIRECTIONS
    return tuple((int(u[0]), int(u[1])) for u in value)


def _resolve_interaction_pairs(value: Any) -> tuple[tuple[int, int], ...]:
    if value is None or value == "default8":
        return DEFAULT_INTERACTION_PAIRS
    return tuple((int(p[0]), int(p[1])) for p in value)


def build_king_anchored_euler_interaction_network_from_config(
    config: dict[str, Any],
) -> KingAnchoredEulerInteractionNet:
    model_cfg = _model_config(config)
    return KingAnchoredEulerInteractionNet(
        input_channels=int(model_cfg.get("input_channels", 18)),
        num_classes=int(model_cfg.get("num_classes", 1)),
        encoding=_encoding_from_config(config, model_cfg),
        num_thresholds=int(model_cfg.get("num_thresholds", 15)),
        directions=_resolve_directions(model_cfg.get("directions")),
        interaction_pairs=_resolve_interaction_pairs(model_cfg.get("interaction_pairs")),
        include_first_differences=bool(model_cfg.get("include_first_differences", True)),
        include_count_summaries=bool(model_cfg.get("include_count_summaries", True)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        second_hidden_dim=int(model_cfg.get("second_hidden_dim", model_cfg.get("hidden_dim_2", 64))),
        dropout=float(model_cfg.get("dropout", 0.1)),
        curve_dropout=float(model_cfg.get("curve_dropout", 0.05)),
        fail_closed_unknown_channels=bool(model_cfg.get("fail_closed_unknown_channels", True)),
    )


def build_king_anchored_euler_interaction_network(
    config: dict[str, Any],
) -> KingAnchoredEulerInteractionNet:
    return build_king_anchored_euler_interaction_network_from_config(config)

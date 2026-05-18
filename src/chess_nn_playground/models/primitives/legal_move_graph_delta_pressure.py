"""Legal-Move-Graph Pressure-Delta primitive (p053).

Source: ``ideas/research/primitives/external_48_legal_move_graph_delta_primitive.md``.

The source markdown calls for a board-aware **edge-centric** candidate-move
graph head whose load-bearing signal is the **per-edge tactical pressure
delta** along each candidate move, not the per-piece-type adjacency
convolution already implemented by p009 (LMGConv). p053 keeps the same
side-to-move typed legal-move topology that p009 compiles (reusing
``_compute_typed_legal_edges`` so both primitives share a single rule-
exact topology source) and adds a small bank of **per-edge scalar
pressure-delta features** that are absent from p009:

    - ``is_capture``: target has an enemy piece (1 if move captures)
    - ``into_king_zone``: target is in the enemy king's 3x3 neighborhood
    - ``gives_check_proxy``: post-move attack mask from the target
      overlaps the enemy king (gives check after the move)
    - ``enemy_value_at_target``: standard material value of the
      captured piece (Q=9, R=5, B=3, N=3, P=1, 0 for empty / king),
      zero on quiet moves
    - ``pre_opp_attackers_at_target``: enemy attackers on ``t`` *before*
      the move (computed via ``compute_attack_relations``)
    - ``pre_own_defenders_at_target``: own defenders on ``t`` before move
    - ``mover_post_attack_value_from_t``: weighted sum of enemy targets
      the mover (of piece type ``r``) would attack from ``t`` after the
      move, using the precomputed geometric attack table. This is a
      *geometry-only* proxy that ignores the blocker change at the
      source square. It is the load-bearing "pressure delta" signal
      because it tells the head, for each candidate move, what enemy
      material the moving piece will newly press from the arrival
      square; that is precisely the per-edge tactical-pressure
      information p009's per-type SAGE aggregation cannot see.
    - ``mover_post_defender_value_from_t``: same construction over own
      pieces (the mover defending its own material from ``t``)

These per-edge ``(B, R=6, 64s, 64t)`` scalar maps are masked by the
typed legal-move adjacency and then aggregated two ways:

  1. **Per-target aggregation**: sum over source square -> ``(B, R, 64, F)``
     per-arrival-square per-type feature tensor. This is a "messages
     arriving at ``t``" picture, weighted by pressure delta.
  2. **Per-type global summary**: sum + max + mean over (s, t) ->
     ``(B, R, 3 * F)``. These global per-type counts / sums / extremes
     are concatenated as a coarse-grained tactical census.

The per-target tensor is collapsed across piece types with a small
linear stack, the resulting per-square tokens are mean+amax pooled to
a fixed board summary, and the board summary is concatenated with the
per-type global summary and with the i193 trunk joint pool. A two-layer
MLP then projects the concatenation to a scalar ``primitive_delta_raw``.
A small gate MLP over ``(trunk_joint, edge_count_per_type)`` with
``gate_init = -2.0`` keeps the primitive as a learned **additive,
gated** correction on top of the i193 trunk logit:

    final_logit = base_logit + sigmoid(gate_logit) * primitive_delta_raw

This is the same additive-head pattern p009, p011, p052 use, so the
new primitive plugs into the existing trainer / model registry / report
template without bespoke trainer changes.

The topology is compiled under ``torch.no_grad()`` and kept as a
stop-gradient float tensor. No python-chess call enters the forward
path. CRTK metadata, source labels, verification flags, engine
evaluations, and principal variations are *not* consumed.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.primitives.legal_move_graph_delta import (
    PIECE_TYPE_NAMES,
    _compute_typed_legal_edges,
)
from chess_nn_playground.models.primitives.rule_graph_features import (
    BISHOP,
    KING,
    KNIGHT,
    NUM_PIECE_TYPES,
    PAWN,
    QUEEN,
    ROOK,
    SQUARES,
    WHITE,
    BLACK,
    compute_attack_relations,
    piece_planes_flat,
    rule_geometry,
    select_by_side_to_move,
    side_to_move_from_board,
)
from chess_nn_playground.models.primitives.trunk_features import trunk_joint_features
from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    DualStreamFeatureBuilder,
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


SIMPLE_18_PLANES = 18

# Number of per-edge scalar pressure-delta features computed before
# masking by the typed legal-move adjacency.
PER_EDGE_FEATURE_NAMES: tuple[str, ...] = (
    "is_capture",
    "into_king_zone",
    "gives_check_proxy",
    "enemy_value_at_target",
    "pre_opp_attackers_at_target",
    "pre_own_defenders_at_target",
    "mover_post_attack_value_from_t",
    "mover_post_defender_value_from_t",
)
NUM_EDGE_FEATURES = len(PER_EDGE_FEATURE_NAMES)

# Standard material values used by the pressure-delta features. The
# king takes value 0 in the captured-piece column (the king cannot be
# captured) but a non-zero "attack value" in the mover-attack column so
# checks register as pressure. Order matches PAWN, KNIGHT, BISHOP, ROOK,
# QUEEN, KING (see ``rule_graph_features`` piece-type constants).
PIECE_VALUE_CAPTURE: tuple[float, ...] = (1.0, 3.0, 3.0, 5.0, 9.0, 0.0)
PIECE_VALUE_ATTACK: tuple[float, ...] = (1.0, 3.0, 3.0, 5.0, 9.0, 3.0)


ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    "no_pressure_delta",
    "no_capture_value",
    "random_typed_edges",
    "shared_target_pool",
    "zero_delta",
    "disable_gate",
    "trunk_only",
)


def _build_king_zone_template() -> torch.Tensor:
    """``(64, 64)`` king-zone template: ``[src, dst] = 1`` iff ``dst`` is within Chebyshev distance 1 of ``src``."""
    template = torch.zeros(SQUARES, SQUARES, dtype=torch.float32)
    for src in range(SQUARES):
        sr, sf = src // 8, src % 8
        for dr in (-1, 0, 1):
            for df in (-1, 0, 1):
                r, f = sr + dr, sf + df
                if 0 <= r < 8 and 0 <= f < 8:
                    template[src, r * 8 + f] = 1.0
    return template


def _enemy_king_squares_one_hot(
    piece_planes: torch.Tensor,
    stm: torch.Tensor,
) -> torch.Tensor:
    """Return a ``(B, 64)`` one-hot mask of the enemy king square.

    Args:
        piece_planes: ``(B, 12, 64)`` per-color per-type piece planes
            (output of :func:`piece_planes_flat`).
        stm: ``(B,)`` side-to-move scalar (1 = white-to-move).
    """
    white_king = piece_planes[:, 6 + KING]    # opp king when stm=1 (black king plane)
    black_king = piece_planes[:, KING]
    return select_by_side_to_move(white_king, black_king, stm)


def _own_pieces_one_hot(
    piece_planes: torch.Tensor,
    stm: torch.Tensor,
) -> torch.Tensor:
    own_white = piece_planes[:, :6].sum(dim=1).clamp(0.0, 1.0)
    own_black = piece_planes[:, 6:12].sum(dim=1).clamp(0.0, 1.0)
    return select_by_side_to_move(own_white, own_black, stm)


def _enemy_pieces_one_hot(
    piece_planes: torch.Tensor,
    stm: torch.Tensor,
) -> torch.Tensor:
    enemy_white = piece_planes[:, 6:12].sum(dim=1).clamp(0.0, 1.0)
    enemy_black = piece_planes[:, :6].sum(dim=1).clamp(0.0, 1.0)
    return select_by_side_to_move(enemy_white, enemy_black, stm)


def _value_per_square(
    piece_planes: torch.Tensor,
    weights: tuple[float, ...],
    side: str,
    stm: torch.Tensor,
) -> torch.Tensor:
    """Per-square material-value scalar for own / enemy pieces.

    Args:
        piece_planes: ``(B, 12, 64)``.
        weights: 6-tuple of per-piece-type values (P, N, B, R, Q, K).
        side: "own" or "enemy".
        stm: ``(B,)`` side-to-move scalar.
    """
    if side not in {"own", "enemy"}:
        raise ValueError(f"Unknown side={side!r}")
    device = piece_planes.device
    dtype = piece_planes.dtype
    weight_tensor = piece_planes.new_tensor(weights)  # (6,)
    white_values = (piece_planes[:, :6] * weight_tensor.view(1, 6, 1)).sum(dim=1)
    black_values = (piece_planes[:, 6:12] * weight_tensor.view(1, 6, 1)).sum(dim=1)
    if side == "own":
        return select_by_side_to_move(white_values, black_values, stm)
    return select_by_side_to_move(black_values, white_values, stm)


def _own_geometric_attack_tables(
    geometry,
    stm: torch.Tensor,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Per-batch own-side geometric attack tables ``(B, 6, 64, 64)`` (no occlusion).

    The "geometric" tables are the unoccluded per-piece-type attack
    bitboards from ``rule_geometry()``. They are used for the per-edge
    "if my piece were on the target square, what would it attack"
    proxy. Occlusion at the target is *not* re-applied because we are
    pre-aggregating over targets; the proxy intentionally measures the
    geometry of the moved piece's post-move attack set rather than a
    rederived occluded one.
    """
    geom = geometry.geom_attacks.to(device=device, dtype=dtype)
    # geom[piece, color, source, target]
    white_view = geom[:, WHITE].unsqueeze(0).expand(batch, NUM_PIECE_TYPES, SQUARES, SQUARES)
    black_view = geom[:, BLACK].unsqueeze(0).expand(batch, NUM_PIECE_TYPES, SQUARES, SQUARES)
    return select_by_side_to_move(white_view, black_view, stm)


def compute_pressure_delta_edge_features(
    board: torch.Tensor,
    edges: torch.Tensor,
    geometry,
    king_zone_template: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute per-edge scalar pressure-delta features.

    Args:
        board: ``(B, 18, 8, 8)`` simple_18 tensor.
        edges: ``(B, 6, 64, 64)`` typed legal-move adjacency from
            :func:`_compute_typed_legal_edges`. Used both for masking
            and to broadcast per-piece-type features.
        geometry: shared ``RuleGeometry`` singleton.
        king_zone_template: ``(64, 64)`` king-zone Chebyshev-1 template.

    Returns:
        A dict whose values are ``(B, 6, 64, 64)`` per-edge scalar
        feature tensors keyed by :data:`PER_EDGE_FEATURE_NAMES`.
    """
    if board.ndim != 4 or board.shape[1] != SIMPLE_18_PLANES:
        raise ValueError(f"Expected simple_18 board (B, 18, 8, 8), got {tuple(board.shape)}")
    if edges.ndim != 4 or edges.shape[1] != NUM_PIECE_TYPES or edges.shape[-2:] != (SQUARES, SQUARES):
        raise ValueError(f"Expected edges (B, 6, 64, 64), got {tuple(edges.shape)}")
    batch = board.shape[0]
    device = board.device
    dtype = board.dtype

    piece_planes = piece_planes_flat(board)  # (B, 12, 64)
    stm = side_to_move_from_board(board)

    enemy_one_hot = _enemy_pieces_one_hot(piece_planes, stm)  # (B, 64)
    enemy_king_one_hot = _enemy_king_squares_one_hot(piece_planes, stm)  # (B, 64)
    enemy_value_per_sq = _value_per_square(piece_planes, PIECE_VALUE_CAPTURE, "enemy", stm)  # (B, 64)
    enemy_attack_value_per_sq = _value_per_square(piece_planes, PIECE_VALUE_ATTACK, "enemy", stm)
    own_attack_value_per_sq = _value_per_square(piece_planes, PIECE_VALUE_ATTACK, "own", stm)

    attacks_per_color, _rays = compute_attack_relations(board, geometry)
    enemy_attacks = select_by_side_to_move(
        attacks_per_color[:, BLACK],
        attacks_per_color[:, WHITE],
        stm,
    )  # (B, 64, 64) -- src attacks dst, occlusion applied; src side = enemy
    own_attacks = select_by_side_to_move(
        attacks_per_color[:, WHITE],
        attacks_per_color[:, BLACK],
        stm,
    )

    # Pre-move attacker / defender counts at each candidate target square.
    pre_opp_attackers_per_t = enemy_attacks.sum(dim=1).clamp_min(0.0)  # (B, 64)
    pre_own_defenders_per_t = own_attacks.sum(dim=1).clamp_min(0.0)

    # Per-(t) king zone indicator: t is in the enemy king's 3x3 zone.
    king_zone_dt = king_zone_template.to(device=device, dtype=dtype)
    # zone[t] = sum_k enemy_king_one_hot[k] * king_zone_template[k, t]
    into_king_zone_per_t = enemy_king_one_hot @ king_zone_dt  # (B, 64)
    into_king_zone_per_t = into_king_zone_per_t.clamp(0.0, 1.0)

    # Own geometric attack tables (no occlusion).
    own_geom = _own_geometric_attack_tables(geometry, stm, batch, device, dtype)  # (B, 6, 64, 64)

    # Per-(b, r, t): geometric attack value if the mover were placed at t.
    # value = sum_j own_geom[b, r, t, j] * enemy_attack_value[b, j]
    mover_post_attack_value_per_target = torch.einsum(
        "brtj,bj->brt",
        own_geom,
        enemy_attack_value_per_sq,
    )
    mover_post_defender_value_per_target = torch.einsum(
        "brtj,bj->brt",
        own_geom,
        own_attack_value_per_sq,
    )
    gives_check_proxy_per_target = torch.einsum(
        "brtj,bj->brt",
        own_geom,
        enemy_king_one_hot,
    ).clamp(0.0, 1.0)

    # Broadcast per-target scalars to per-edge (s, t) tensors then mask.
    def t_broadcast(per_target: torch.Tensor) -> torch.Tensor:
        # per_target: (B, 64) or (B, 6, 64)
        if per_target.dim() == 2:
            return per_target.view(batch, 1, 1, SQUARES).expand(batch, NUM_PIECE_TYPES, SQUARES, SQUARES)
        return per_target.view(batch, NUM_PIECE_TYPES, 1, SQUARES).expand(
            batch, NUM_PIECE_TYPES, SQUARES, SQUARES
        )

    is_capture_at_t = t_broadcast(enemy_one_hot)
    into_king_zone = t_broadcast(into_king_zone_per_t)
    gives_check_proxy = t_broadcast(gives_check_proxy_per_target)
    enemy_value_at_target = t_broadcast(enemy_value_per_sq)
    pre_opp_attackers = t_broadcast(pre_opp_attackers_per_t)
    pre_own_defenders = t_broadcast(pre_own_defenders_per_t)
    mover_post_attack_value = t_broadcast(mover_post_attack_value_per_target)
    mover_post_defender_value = t_broadcast(mover_post_defender_value_per_target)

    # Mask by candidate edges so non-edge entries do not pollute pools.
    features = {
        "is_capture": is_capture_at_t,
        "into_king_zone": into_king_zone,
        "gives_check_proxy": gives_check_proxy,
        "enemy_value_at_target": enemy_value_at_target,
        "pre_opp_attackers_at_target": pre_opp_attackers,
        "pre_own_defenders_at_target": pre_own_defenders,
        "mover_post_attack_value_from_t": mover_post_attack_value,
        "mover_post_defender_value_from_t": mover_post_defender_value,
    }
    masked = {key: value * edges for key, value in features.items()}
    return masked


def _stack_per_edge_features(
    feature_dict: dict[str, torch.Tensor],
    order: tuple[str, ...] = PER_EDGE_FEATURE_NAMES,
) -> torch.Tensor:
    """Stack a feature dict to a single ``(B, R, 64, 64, F)`` tensor."""
    return torch.stack([feature_dict[name] for name in order], dim=-1)


def aggregate_per_target_features(
    edges: torch.Tensor,
    stacked_features: torch.Tensor,
    include_degree: bool = True,
) -> torch.Tensor:
    """Aggregate stacked per-edge features to per-target tokens.

    For each (batch, piece-type, target square) compute the
    edge-sum of every feature plus optionally a per-target degree
    (the count of candidate edges ending at ``t`` for piece type
    ``r``). Returns ``(B, R, 64, F+1)`` if ``include_degree`` else
    ``(B, R, 64, F)``.
    """
    target_sum = stacked_features.sum(dim=-3)  # sum over source axis
    if include_degree:
        degree = edges.sum(dim=-2).unsqueeze(-1)  # (B, R, 64, 1)
        return torch.cat([target_sum, degree], dim=-1)
    return target_sum


def per_type_global_summary(
    edges: torch.Tensor,
    stacked_features: torch.Tensor,
) -> torch.Tensor:
    """Return ``(B, R, 3*F+1)`` per-type global pooled features.

    For each (batch, piece-type) the head emits ``[sum, mean, max,
    edge_count]`` over the (s, t) axes for each of the F per-edge
    features, where ``mean`` divides ``sum`` by ``edge_count + 1``
    to keep it finite when no edges exist.
    """
    batch, _, _, _, num_feat = stacked_features.shape
    edge_count = edges.sum(dim=(-2, -1)).unsqueeze(-1)  # (B, R, 1)
    summed = stacked_features.sum(dim=(-3, -2))  # (B, R, F)
    mean = summed / edge_count.clamp_min(1.0)
    max_val = stacked_features.amax(dim=(-3, -2))  # (B, R, F)
    return torch.cat([summed, mean, max_val, edge_count], dim=-1)


class LegalMoveGraphDeltaPressure(nn.Module):
    """p053 -- Legal-Move-Graph Pressure-Delta head over the i193 trunk.

    Per-edge pressure-delta features are computed from the simple_18
    board tensor under ``torch.no_grad()``-style stop-gradient
    topology, aggregated per arrival square and per piece type, and
    projected through a small MLP to a scalar logit delta. The delta
    is combined with the i193 trunk's base logit via a sigmoid gate
    initialised at ``gate_init = -2.0`` so the primitive starts as a
    near no-op.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        # Trunk hyper-parameters (i193).
        trunk_channels: int = 64,
        trunk_hidden_dim: int = 96,
        trunk_depth: int = 2,
        trunk_dropout: float = 0.1,
        trunk_use_batchnorm: bool = True,
        trunk_gate_dim: int | None = None,
        trunk_ablation: str = "none",
        # Pressure-delta head hyper-parameters.
        per_type_token_dim: int = 32,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "LegalMoveGraphDeltaPressure supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != SIMPLE_18_PLANES:
            raise ValueError(
                "LegalMoveGraphDeltaPressure requires the simple_18 board tensor"
            )
        if str(ablation) not in ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(ALLOWED_ABLATIONS)}"
            )

        self.num_classes = 1
        self.ablation = str(ablation)
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self._geometry = rule_geometry()

        self.trunk = ExchangeThenKingDualStreamNetwork(
            input_channels=int(input_channels),
            num_classes=1,
            channels=int(trunk_channels),
            hidden_dim=int(trunk_hidden_dim),
            depth=int(trunk_depth),
            dropout=float(trunk_dropout),
            use_batchnorm=bool(trunk_use_batchnorm),
            gate_dim=trunk_gate_dim,
            ablation=str(trunk_ablation),
        )

        self.register_buffer("king_zone_template", _build_king_zone_template())

        # Per-type per-target token tower: F+1 -> per_type_token_dim.
        per_target_dim = NUM_EDGE_FEATURES + 1  # +1 for arrival degree
        if self.ablation == "shared_target_pool":
            self.target_token_proj = nn.Linear(per_target_dim, int(per_type_token_dim))
            self.target_token_per_type = None
        else:
            self.target_token_proj = None
            self.target_token_per_type = nn.ModuleList(
                [nn.Linear(per_target_dim, int(per_type_token_dim)) for _ in range(NUM_PIECE_TYPES)]
            )
        self.target_token_norm = nn.LayerNorm(int(per_type_token_dim))

        # Trunk pool dim.
        self.trunk_feature_dim = (
            2 * self.trunk.exchange_encoder.output_dim
            + DualStreamFeatureBuilder.SUMMARY_DIM
        )

        # Per-type global summary dim: 3 * F + 1.
        self.per_type_global_dim = 3 * NUM_EDGE_FEATURES + 1
        self.global_summary_dim_total = NUM_PIECE_TYPES * self.per_type_global_dim

        board_summary_dim = 2 * int(per_type_token_dim)  # mean + amax over squares
        self.feature_dim_total = board_summary_dim + self.global_summary_dim_total

        head_dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim_total + self.trunk_feature_dim),
            nn.Linear(self.feature_dim_total + self.trunk_feature_dim, int(head_hidden_dim)),
            nn.GELU(),
            head_dropout_module,
            nn.Linear(int(head_hidden_dim), 1),
        )

        # Gate inputs: trunk joint + per-type edge count (R = 6) + total edge count.
        gate_extra_dim = NUM_PIECE_TYPES + 1
        self.gate_head = nn.Sequential(
            nn.LayerNorm(self.trunk_feature_dim + gate_extra_dim),
            nn.Linear(self.trunk_feature_dim + gate_extra_dim, max(16, int(head_hidden_dim) // 2)),
            nn.GELU(),
            nn.Linear(max(16, int(head_hidden_dim) // 2), 1),
        )
        with torch.no_grad():
            final_layer = self.gate_head[-1]
            if isinstance(final_layer, nn.Linear):
                final_layer.bias.fill_(float(gate_init))

    @torch.no_grad()
    def _build_edges(self, board: torch.Tensor) -> torch.Tensor:
        edges = _compute_typed_legal_edges(board, self._geometry)
        if self.ablation == "random_typed_edges":
            density = edges.sum(dim=(2, 3), keepdim=True) / (SQUARES * SQUARES)
            rand = torch.rand_like(edges)
            edges = (rand < density).to(dtype=edges.dtype)
        return edges

    def _maybe_zero_pressure_blocks(
        self,
        feature_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if self.ablation == "no_pressure_delta":
            out = {}
            for name, value in feature_dict.items():
                if name in {
                    "pre_opp_attackers_at_target",
                    "pre_own_defenders_at_target",
                    "mover_post_attack_value_from_t",
                    "mover_post_defender_value_from_t",
                }:
                    out[name] = torch.zeros_like(value)
                else:
                    out[name] = value
            return out
        if self.ablation == "no_capture_value":
            out = {}
            for name, value in feature_dict.items():
                if name in {"enemy_value_at_target", "gives_check_proxy"}:
                    out[name] = torch.zeros_like(value)
                else:
                    out[name] = value
            return out
        return feature_dict

    def _project_per_type_target_tokens(self, per_target_features: torch.Tensor) -> torch.Tensor:
        """Project per-type per-target features to per-type per-square tokens.

        Args:
            per_target_features: ``(B, R, 64, F+1)``.

        Returns:
            ``(B, R, 64, D)`` per-type per-square token tensor.
        """
        if self.ablation == "shared_target_pool":
            assert self.target_token_proj is not None
            return self.target_token_proj(per_target_features)
        assert self.target_token_per_type is not None
        per_type_tokens = []
        for r, linear in enumerate(self.target_token_per_type):
            per_type_tokens.append(linear(per_target_features[:, r]))  # (B, 64, D)
        return torch.stack(per_type_tokens, dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]

        trunk_out = self.trunk(board)
        base_logit = trunk_out["logits"].view(-1)
        joint, _, _ = trunk_joint_features(self.trunk, board)

        edges = self._build_edges(board)
        feature_dict = compute_pressure_delta_edge_features(
            board, edges, self._geometry, self.king_zone_template
        )
        feature_dict = self._maybe_zero_pressure_blocks(feature_dict)
        stacked = _stack_per_edge_features(feature_dict)  # (B, R, 64, 64, F)

        per_target = aggregate_per_target_features(edges, stacked, include_degree=True)
        global_summary = per_type_global_summary(edges, stacked)  # (B, R, 3F+1)

        per_type_tokens = self._project_per_type_target_tokens(per_target)
        # Collapse piece-type axis by summation, then normalise.
        tokens = per_type_tokens.sum(dim=1)  # (B, 64, D)
        tokens = self.target_token_norm(tokens)
        board_summary = torch.cat(
            [tokens.mean(dim=1), tokens.amax(dim=1)],
            dim=-1,
        )
        global_summary_flat = global_summary.reshape(batch, -1)
        feature_vec = torch.cat([board_summary, global_summary_flat], dim=-1)

        delta_input = torch.cat([feature_vec, joint], dim=-1)
        delta_raw = self.delta_head(delta_input).view(-1)

        edge_counts_per_type = edges.sum(dim=(2, 3))  # (B, R)
        total_edge_count = edge_counts_per_type.sum(dim=-1, keepdim=True)  # (B, 1)
        gate_extras = torch.cat([edge_counts_per_type, total_edge_count], dim=-1)
        gate_input = torch.cat([joint, gate_extras], dim=-1)
        gate_logit = self.gate_head(gate_input).view(-1)
        gate = torch.sigmoid(gate_logit)
        if self.ablation == "disable_gate":
            gate = torch.ones_like(gate)

        if self.ablation in {"zero_delta", "trunk_only"}:
            primitive_delta = torch.zeros_like(base_logit)
            gate_applied = torch.zeros_like(gate)
        else:
            primitive_delta = delta_raw
            gate_applied = gate
        contribution = gate_applied * primitive_delta
        logits = base_logit + contribution

        eps = 1.0e-6
        gate_clamped = gate.clamp(eps, 1.0 - eps)
        gate_entropy = -(
            gate_clamped * gate_clamped.log()
            + (1.0 - gate_clamped) * (1.0 - gate_clamped).log()
        )

        # Per-type mean post-attack value: useful diagnostic for the
        # "pressure-delta" claim. Computed from the unmasked-but-edge-
        # weighted sum / edge count.
        per_type_mean_attack_value = (
            stacked[..., PER_EDGE_FEATURE_NAMES.index("mover_post_attack_value_from_t")]
            .sum(dim=(-2, -1))
            / edge_counts_per_type.clamp_min(1.0)
        )
        per_type_mean_capture_value = (
            stacked[..., PER_EDGE_FEATURE_NAMES.index("enemy_value_at_target")]
            .sum(dim=(-2, -1))
            / edge_counts_per_type.clamp_min(1.0)
        )

        out: dict[str, torch.Tensor] = {}
        for key, value in trunk_out.items():
            if key in {"logits", "proposal_profile_strength", "proposal_keyword_count"}:
                continue
            out[f"trunk_{key}"] = value
        out["logits"] = logits
        out["base_logit"] = base_logit
        out["primitive_delta"] = primitive_delta
        out["primitive_delta_raw"] = delta_raw
        out["primitive_gate"] = gate
        out["primitive_gate_applied"] = gate_applied
        out["primitive_gate_logit"] = gate_logit
        out["primitive_gate_entropy"] = gate_entropy
        out["primitive_contribution"] = contribution
        out["lmgdp_total_edge_count"] = total_edge_count.squeeze(-1)
        for r, name in enumerate(PIECE_TYPE_NAMES):
            out[f"lmgdp_edge_count_{name}"] = edge_counts_per_type[:, r]
            out[f"lmgdp_post_attack_value_mean_{name}"] = per_type_mean_attack_value[:, r]
            out[f"lmgdp_capture_value_mean_{name}"] = per_type_mean_capture_value[:, r]
        out["mechanism_energy"] = trunk_out["mechanism_energy"] + total_edge_count.squeeze(-1).detach()
        out["proposal_profile_strength"] = primitive_delta.detach().abs() * gate_entropy
        out["proposal_keyword_count"] = logits.new_full(
            (batch,), float(self.feature_dim_total)
        )
        return out


def build_legal_move_graph_delta_pressure_from_config(
    config: dict[str, Any],
) -> LegalMoveGraphDeltaPressure:
    cfg = dict(config)
    return LegalMoveGraphDeltaPressure(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim")),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        per_type_token_dim=int(cfg.get("per_type_token_dim", 32)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )


__all__ = (
    "ALLOWED_ABLATIONS",
    "PER_EDGE_FEATURE_NAMES",
    "NUM_EDGE_FEATURES",
    "PIECE_VALUE_CAPTURE",
    "PIECE_VALUE_ATTACK",
    "LegalMoveGraphDeltaPressure",
    "build_legal_move_graph_delta_pressure_from_config",
    "compute_pressure_delta_edge_features",
    "aggregate_per_target_features",
    "per_type_global_summary",
)

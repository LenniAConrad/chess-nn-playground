"""Entropic Chess Geometry Transport Network (idea i034)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.data.board_features import SIMPLE_18
from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


TARGET_ROLES: tuple[str, ...] = (
    "king_square",
    "king_ring",
    "heavy_piece",
    "minor_piece",
    "pawn",
    "promotion_anchor",
)
_SOURCE_PRIOR = torch.tensor([1.0, 3.0, 3.0, 5.0, 9.0, 2.0], dtype=torch.float32)
_TARGET_ROLE_PRIOR = torch.tensor([10.0, 3.0, 5.0, 2.5, 1.0, 0.5], dtype=torch.float32)
_CENTER_SQUARE = 27


def _inverse_softplus(values: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(values.clamp_min(1e-4)))


def _rank_file() -> tuple[torch.Tensor, torch.Tensor]:
    square = torch.arange(64, dtype=torch.float32)
    return square // 8, square % 8


def _knight_distance_table(distance_cap: float) -> torch.Tensor:
    moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
    dist = torch.full((64, 64), float(distance_cap), dtype=torch.float32)
    for source in range(64):
        queue = [source]
        dist[source, source] = 0.0
        cursor = 0
        while cursor < len(queue):
            square = queue[cursor]
            cursor += 1
            rank, file = divmod(square, 8)
            for dr, df in moves:
                rr, ff = rank + dr, file + df
                if 0 <= rr < 8 and 0 <= ff < 8:
                    target = rr * 8 + ff
                    if dist[source, target] > dist[source, square] + 1.0:
                        dist[source, target] = dist[source, square] + 1.0
                        queue.append(target)
    return dist.clamp_max(float(distance_cap)) / float(distance_cap)


def _piece_distance_tables(distance_cap: float = 8.0) -> torch.Tensor:
    rank, file = _rank_file()
    src_rank = rank.view(64, 1)
    src_file = file.view(64, 1)
    tgt_rank = rank.view(1, 64)
    tgt_file = file.view(1, 64)
    dr = (src_rank - tgt_rank).abs()
    df = (src_file - tgt_file).abs()
    same_square = (dr == 0) & (df == 0)
    same_rank = dr == 0
    same_file = df == 0
    same_diag = dr == df
    same_color = ((src_rank + src_file - tgt_rank - tgt_file).remainder(2) == 0)
    cap = torch.full((64, 64), float(distance_cap), dtype=torch.float32)

    pawn_forward = src_rank - tgt_rank
    pawn_reachable = (pawn_forward > 0) & (df <= pawn_forward)
    pawn_dist = torch.where(pawn_reachable, pawn_forward + df, cap)
    pawn_dist = torch.where(same_square, torch.zeros_like(pawn_dist), pawn_dist)

    knight_dist = _knight_distance_table(distance_cap) * float(distance_cap)

    bishop_dist = torch.where(same_diag, torch.ones_like(cap), torch.where(same_color, torch.full_like(cap, 2.0), cap))
    bishop_dist = torch.where(same_square, torch.zeros_like(bishop_dist), bishop_dist)

    rook_dist = torch.where(same_rank | same_file, torch.ones_like(cap), torch.full_like(cap, 2.0))
    rook_dist = torch.where(same_square, torch.zeros_like(rook_dist), rook_dist)

    queen_dist = torch.minimum(rook_dist, bishop_dist)
    king_dist = torch.maximum(dr, df)

    return torch.stack([pawn_dist, knight_dist, bishop_dist, rook_dist, queen_dist, king_dist], dim=0).clamp_max(
        float(distance_cap)
    ) / float(distance_cap)


def _manhattan_table() -> torch.Tensor:
    rank, file = _rank_file()
    src_rank = rank.view(64, 1)
    src_file = file.view(64, 1)
    tgt_rank = rank.view(1, 64)
    tgt_file = file.view(1, 64)
    return ((src_rank - tgt_rank).abs() + (src_file - tgt_file).abs()) / 14.0


@dataclass(frozen=True)
class ParsedBoard:
    us: torch.Tensor
    them: torch.Tensor
    white_to_move: torch.Tensor
    castling: torch.Tensor
    en_passant: torch.Tensor


@dataclass(frozen=True)
class TransportAtoms:
    source_square: torch.Tensor
    source_type: torch.Tensor
    source_mask: torch.Tensor
    source_marginal: torch.Tensor
    target_square: torch.Tensor
    target_role: torch.Tensor
    target_mask: torch.Tensor
    target_marginal: torch.Tensor


class EncodingSemanticAdapter(nn.Module):
    """Fail-closed simple_18 parser for deterministic current-board geometry."""

    def __init__(self, input_channels: int = 18, encoding_adapter: str = SIMPLE_18) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=input_channels)
        if encoding_adapter != SIMPLE_18 or input_channels != 18:
            raise ValueError(
                "ChessGeometryTransportNet deterministic atom building supports only simple_18. "
                "LC0 or unknown channel semantics must be registered before use."
            )

    def forward(self, x: torch.Tensor) -> ParsedBoard:
        x = require_board_tensor(x, self.spec)
        white = x[:, 0:6].clamp_min(0.0)
        black = x[:, 6:12].clamp_min(0.0)
        white_to_move = x[:, 12].mean(dim=(1, 2)) >= 0.5
        white_mask = white_to_move.view(-1, 1, 1, 1)
        us_by_color = torch.where(white_mask, white, black)
        them_by_color = torch.where(white_mask, black, white)
        us = torch.where(white_mask, us_by_color, torch.flip(us_by_color, dims=(-2,)))
        them = torch.where(white_mask, them_by_color, torch.flip(them_by_color, dims=(-2,)))
        return ParsedBoard(
            us=us,
            them=them,
            white_to_move=white_to_move,
            castling=x[:, 13:17] if x.shape[1] >= 17 else x.new_zeros(x.shape[0], 4, 8, 8),
            en_passant=x[:, 17:18] if x.shape[1] >= 18 else x.new_zeros(x.shape[0], 1, 8, 8),
        )


class TransportAtomBuilder(nn.Module):
    def __init__(self, max_sources: int = 16, max_targets: int = 40) -> None:
        super().__init__()
        if max_sources < 1:
            raise ValueError("max_sources must be positive")
        if max_targets < 25:
            raise ValueError("max_targets must fit king, king-ring, material, pawn, and promotion atoms")
        self.max_sources = max_sources
        self.max_targets = max_targets
        self.material_slots = max_targets - (1 + 8 + 8)
        if self.material_slots < 1:
            raise ValueError("max_targets must leave at least one material target slot")
        ring_offsets = torch.tensor(
            [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
            dtype=torch.long,
        )
        promotion = torch.arange(8, dtype=torch.long)
        self.register_buffer("king_ring_offsets", ring_offsets, persistent=False)
        self.register_buffer("promotion_squares", promotion, persistent=False)
        self.source_weight_logits = nn.Parameter(_inverse_softplus(_SOURCE_PRIOR.clone()))
        self.target_weight_logits = nn.Parameter(_inverse_softplus(_TARGET_ROLE_PRIOR.clone()))

    def _top_piece_atoms(self, pieces: torch.Tensor, max_atoms: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = pieces.flatten(1)
        order = torch.arange(flat.shape[1], device=pieces.device, dtype=pieces.dtype)
        type_id = (order.long() // 64).float()
        scores = flat * 1000.0 + _SOURCE_PRIOR.to(device=pieces.device, dtype=pieces.dtype)[type_id.long()].view(1, -1)
        scores = scores + (flat.shape[1] - 1 - order).view(1, -1) / 1000.0
        _values, indices = scores.topk(max_atoms, dim=1)
        mask = torch.gather(flat, 1, indices) > 0.5
        source_type = (indices // 64).long()
        source_square = (indices % 64).long()
        no_source = ~mask.any(dim=1)
        first = torch.arange(max_atoms, device=pieces.device).view(1, -1) == 0
        mask = torch.where(no_source[:, None] & first, torch.ones_like(mask), mask)
        source_type = torch.where(no_source[:, None] & first, source_type.new_full(source_type.shape, 5), source_type)
        source_square = torch.where(
            no_source[:, None] & first,
            source_square.new_full(source_square.shape, _CENTER_SQUARE),
            source_square,
        )
        return source_square, source_type, mask

    def _top_subset_squares(self, pieces: torch.Tensor, type_ids: tuple[int, ...], slots: int) -> tuple[torch.Tensor, torch.Tensor]:
        flat_parts = []
        source_indices = []
        for type_id in type_ids:
            squares = torch.arange(64, device=pieces.device)
            flat_parts.append(pieces[:, type_id].flatten(1))
            source_indices.append(squares + type_id * 64)
        flat = torch.cat(flat_parts, dim=1)
        index_map = torch.cat(source_indices, dim=0)
        order = torch.arange(flat.shape[1], device=pieces.device, dtype=pieces.dtype)
        scores = flat * 1000.0 + (flat.shape[1] - 1 - order).view(1, -1) / 1000.0
        _values, indices = scores.topk(slots, dim=1)
        occupied = torch.gather(flat, 1, indices) > 0.5
        original = index_map[indices]
        return (original % 64).long(), occupied

    def _top_material_targets(self, pieces: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        material = pieces[:, 0:5]
        flat = material.flatten(1)
        order = torch.arange(flat.shape[1], device=pieces.device, dtype=pieces.dtype)
        piece_type = (order.long() // 64).clamp(0, 4)
        role_priority = torch.tensor([1.0, 2.5, 2.5, 5.0, 5.5], device=pieces.device, dtype=pieces.dtype)
        scores = flat * 1000.0 + role_priority[piece_type].view(1, -1)
        scores = scores + (flat.shape[1] - 1 - order).view(1, -1) / 1000.0
        _values, indices = scores.topk(self.material_slots, dim=1)
        mask = torch.gather(flat, 1, indices) > 0.5
        target_type = (indices // 64).long()
        square = (indices % 64).long()
        role = torch.where(
            target_type == 0,
            square.new_full(square.shape, 4),
            torch.where(
                (target_type == 1) | (target_type == 2),
                square.new_full(square.shape, 3),
                square.new_full(square.shape, 2),
            ),
        )
        return square, role, mask

    def _king_square(self, pieces: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        king_flat = pieces[:, 5].flatten(1)
        value, square = king_flat.max(dim=1)
        square = torch.where(value > 0.5, square, square.new_full(square.shape, _CENTER_SQUARE))
        return square.long(), value > 0.5

    def _king_ring(self, king_square: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rank = king_square // 8
        file = king_square % 8
        offsets = self.king_ring_offsets.to(device=king_square.device)
        ring_rank = rank[:, None] + offsets[None, :, 0]
        ring_file = file[:, None] + offsets[None, :, 1]
        valid = (ring_rank >= 0) & (ring_rank < 8) & (ring_file >= 0) & (ring_file < 8)
        ring_rank = ring_rank.clamp(0, 7)
        ring_file = ring_file.clamp(0, 7)
        return (ring_rank * 8 + ring_file).long(), valid

    def _normalize(self, raw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raw = raw * mask.float()
        return raw / raw.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def forward(self, board: ParsedBoard) -> TransportAtoms:
        source_square, source_type, source_mask = self._top_piece_atoms(board.us, self.max_sources)
        source_weight = F.softplus(self.source_weight_logits).to(device=board.us.device, dtype=board.us.dtype)
        source_raw = source_weight[source_type.clamp(0, 5)]
        source_marginal = self._normalize(source_raw, source_mask)

        king_square, king_exists = self._king_square(board.them)
        king_ring_square, king_ring_mask = self._king_ring(king_square)
        material_square, material_role, material_mask = self._top_material_targets(board.them)
        promotion_square = self.promotion_squares.to(device=board.us.device).view(1, -1).expand(board.us.shape[0], -1)
        promotion_mask = torch.ones_like(promotion_square, dtype=torch.bool)

        target_square = torch.cat(
            [
                king_square[:, None],
                king_ring_square,
                material_square,
                promotion_square,
            ],
            dim=1,
        )
        target_mask = torch.cat(
            [
                torch.ones_like(king_exists[:, None], dtype=torch.bool),
                king_ring_mask,
                material_mask,
                promotion_mask,
            ],
            dim=1,
        )
        role_chunks = [
            target_square.new_full((target_square.shape[0], 1), 0),
            target_square.new_full(king_ring_square.shape, 1),
            material_role,
            target_square.new_full(promotion_square.shape, 5),
        ]
        target_role = torch.cat(role_chunks, dim=1)
        target_weight = F.softplus(self.target_weight_logits).to(device=board.us.device, dtype=board.us.dtype)
        target_raw = target_weight[target_role.clamp(0, len(TARGET_ROLES) - 1)]
        target_marginal = self._normalize(target_raw, target_mask)
        return TransportAtoms(
            source_square=source_square,
            source_type=source_type,
            source_mask=source_mask,
            source_marginal=source_marginal,
            target_square=target_square,
            target_role=target_role,
            target_mask=target_mask,
            target_marginal=target_marginal,
        )


class ChessDistanceCost(nn.Module):
    def __init__(
        self,
        target_roles: int = len(TARGET_ROLES),
        distance_cap: float = 8.0,
        invalid_cost: float = 1.0e4,
        cost_ablation_mode: str = "none",
    ) -> None:
        super().__init__()
        if cost_ablation_mode not in {"none", "uniform", "random_cost_histogram_preserving"}:
            raise ValueError("unsupported cost_ablation_mode")
        self.target_roles = target_roles
        self.invalid_cost = invalid_cost
        self.cost_ablation_mode = cost_ablation_mode
        self.register_buffer("distance", _piece_distance_tables(distance_cap), persistent=False)
        self.register_buffer("manhattan", _manhattan_table(), persistent=False)
        self.alpha_logits = nn.Parameter(_inverse_softplus(torch.ones(6, target_roles)))
        self.beta = nn.Parameter(torch.zeros(6, target_roles))
        self.coordinate_gamma_logits = nn.Parameter(_inverse_softplus(torch.full((6, target_roles), 0.2)))

    def _gather_pair_table(self, table: torch.Tensor, atoms: TransportAtoms) -> torch.Tensor:
        return table[atoms.source_square.unsqueeze(-1), atoms.target_square.unsqueeze(1)]

    def _randomize_type_role_histograms(self, cost: torch.Tensor, atoms: TransportAtoms, valid: torch.Tensor) -> torch.Tensor:
        randomized = cost.clone()
        flat_randomized = randomized.view(randomized.shape[0], -1)
        flat_cost = cost.view(cost.shape[0], -1)
        flat_valid = valid.view(valid.shape[0], -1)
        pair_source_type = atoms.source_type.unsqueeze(-1).expand_as(cost).reshape(cost.shape[0], -1)
        pair_target_role = atoms.target_role.unsqueeze(1).expand_as(cost).reshape(cost.shape[0], -1)
        for piece_type in range(6):
            for role in range(self.target_roles):
                selected = flat_valid & (pair_source_type == piece_type) & (pair_target_role == role)
                for batch_idx in range(cost.shape[0]):
                    indices = selected[batch_idx].nonzero(as_tuple=False).flatten()
                    if indices.numel() <= 1:
                        continue
                    shift = int((piece_type * 7 + role * 11 + batch_idx * 13) % int(indices.numel()))
                    flat_randomized[batch_idx, indices] = flat_cost[batch_idx, indices.roll(shifts=shift)]
        return randomized

    def forward(self, atoms: TransportAtoms) -> torch.Tensor:
        valid = atoms.source_mask.unsqueeze(-1) & atoms.target_mask.unsqueeze(1)
        distance = self.distance.to(device=atoms.source_square.device)
        piece_distance = distance[
            atoms.source_type.clamp(0, 5).unsqueeze(-1),
            atoms.source_square.unsqueeze(-1),
            atoms.target_square.unsqueeze(1),
        ]
        manhattan = self._gather_pair_table(self.manhattan.to(device=atoms.source_square.device), atoms)
        alpha = F.softplus(self.alpha_logits).to(device=piece_distance.device, dtype=piece_distance.dtype)
        gamma = F.softplus(self.coordinate_gamma_logits).to(device=piece_distance.device, dtype=piece_distance.dtype)
        beta = self.beta.to(device=piece_distance.device, dtype=piece_distance.dtype)
        source_type = atoms.source_type.clamp(0, 5).unsqueeze(-1)
        target_role = atoms.target_role.clamp(0, self.target_roles - 1).unsqueeze(1)
        cost = alpha[source_type, target_role] * piece_distance
        cost = cost + beta[source_type, target_role] + gamma[source_type, target_role] * manhattan
        if self.cost_ablation_mode == "uniform":
            cost = torch.ones_like(cost)
        elif self.cost_ablation_mode == "random_cost_histogram_preserving":
            cost = self._randomize_type_role_histograms(cost, atoms, valid)
        return cost.masked_fill(~valid, self.invalid_cost)


class LogSinkhornTransport(nn.Module):
    def __init__(self, epsilon: float = 0.25, sinkhorn_iters: int = 8) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if sinkhorn_iters < 1:
            raise ValueError("sinkhorn_iters must be >= 1")
        self.epsilon = epsilon
        self.sinkhorn_iters = sinkhorn_iters

    def forward(self, cost: torch.Tensor, atoms: TransportAtoms) -> torch.Tensor:
        valid = atoms.source_mask.unsqueeze(-1) & atoms.target_mask.unsqueeze(1)
        log_a = atoms.source_marginal.float().clamp_min(1e-12).log()
        log_b = atoms.target_marginal.float().clamp_min(1e-12).log()
        log_kernel = (-cost.float() / self.epsilon).masked_fill(~valid, -1.0e9)
        log_u = torch.zeros_like(log_a)
        log_v = torch.zeros_like(log_b)
        for _ in range(self.sinkhorn_iters):
            log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(1), dim=2)
            log_u = log_u.masked_fill(~atoms.source_mask, 0.0)
            log_v = log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(2), dim=1)
            log_v = log_v.masked_fill(~atoms.target_mask, 0.0)
        plan = torch.exp(log_u.unsqueeze(2) + log_kernel + log_v.unsqueeze(1)) * valid.float()
        return plan / plan.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)


class TransportFeatureProjector(nn.Module):
    def __init__(self, max_sources: int = 16, max_targets: int = 40, target_roles: int = len(TARGET_ROLES)) -> None:
        super().__init__()
        self.max_sources = max_sources
        self.max_targets = max_targets
        self.target_roles = target_roles
        self.vector_dim = 6 * target_roles + 6
        self.map_channels = 1 + target_roles

    def forward(self, plan: torch.Tensor, cost: torch.Tensor, atoms: TransportAtoms) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        source_one_hot = F.one_hot(atoms.source_type.clamp(0, 5), num_classes=6).float()
        target_one_hot = F.one_hot(atoms.target_role.clamp(0, self.target_roles - 1), num_classes=self.target_roles).float()
        flow = torch.einsum("bst,bsp,btr->bpr", plan, source_one_hot, target_one_hot)
        expected_cost = (plan * cost.float()).sum(dim=(1, 2))
        entropy = -(plan * plan.clamp_min(1e-12).log()).sum(dim=(1, 2)) / torch.log(
            plan.new_tensor(float(self.max_sources * self.max_targets))
        )
        row_mass = plan.sum(dim=2)
        col_mass = plan.sum(dim=1)
        scalars = torch.stack(
            [
                expected_cost,
                entropy,
                row_mass.amax(dim=1),
                col_mass.amax(dim=1),
                atoms.source_mask.float().sum(dim=1) / float(self.max_sources),
                atoms.target_mask.float().sum(dim=1) / float(self.max_targets),
            ],
            dim=1,
        )
        vector = torch.cat([flow.flatten(1), scalars], dim=1)

        batch = plan.shape[0]
        source_map = plan.new_zeros(batch, 1, 64)
        source_map.scatter_add_(2, atoms.source_square[:, None, :], row_mass[:, None, :])
        target_maps = plan.new_zeros(batch, self.target_roles, 64)
        role_values = col_mass[:, None, :] * target_one_hot.transpose(1, 2).to(dtype=plan.dtype)
        target_maps.scatter_add_(2, atoms.target_square[:, None, :].expand(batch, self.target_roles, -1), role_values)
        maps = torch.cat([source_map, target_maps], dim=1).view(batch, self.map_channels, 8, 8)
        diagnostics = {
            "transport_cost": expected_cost,
            "transport_entropy": entropy,
            "transport_source_concentration": row_mass.square().sum(dim=1),
            "transport_target_concentration": col_mass.square().sum(dim=1),
            "transport_king_flow": flow[:, :, 0].sum(dim=1),
            "transport_role_pressure": flow.sum(dim=1).amax(dim=1),
        }
        return vector, maps, diagnostics


class TransportAugmentedCNN(nn.Module):
    def __init__(self, input_channels: int, map_channels: int, hidden_width: int = 64, depth: int = 3) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers: list[nn.Module] = []
        channels = input_channels + map_channels
        for _ in range(depth):
            layers.extend(
                [
                    nn.Conv2d(channels, hidden_width, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(hidden_width),
                    nn.GELU(),
                ]
            )
            channels = hidden_width
        self.net = nn.Sequential(*layers)
        self.output_dim = hidden_width * 2

    def forward(self, x: torch.Tensor, maps: torch.Tensor) -> torch.Tensor:
        h = self.net(torch.cat([x, maps.to(dtype=x.dtype)], dim=1))
        return torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)


class ChessGeometryTransportNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        encoding_adapter: str = SIMPLE_18,
        max_sources: int = 16,
        max_targets: int = 40,
        epsilon: float = 0.25,
        sinkhorn_iters: int = 8,
        distance_cap: float = 8.0,
        use_pressure_maps: bool = True,
        use_scalar_transport_features: bool = True,
        cost_ablation_mode: str = "none",
        fail_closed_semantic_adapter: bool = True,
        hidden_width: int = 64,
        depth: int = 3,
        transport_hidden_dim: int = 64,
        classifier_hidden_dim: int = 96,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not fail_closed_semantic_adapter:
            raise ValueError("fail_closed_semantic_adapter must remain true for deterministic ECGT atom building")
        self.num_classes = num_classes
        self.use_pressure_maps = use_pressure_maps
        self.use_scalar_transport_features = use_scalar_transport_features
        self.semantic_adapter = EncodingSemanticAdapter(input_channels=input_channels, encoding_adapter=encoding_adapter)
        self.atom_builder = TransportAtomBuilder(max_sources=max_sources, max_targets=max_targets)
        self.cost = ChessDistanceCost(distance_cap=distance_cap, cost_ablation_mode=cost_ablation_mode)
        self.sinkhorn = LogSinkhornTransport(epsilon=epsilon, sinkhorn_iters=sinkhorn_iters)
        self.projector = TransportFeatureProjector(max_sources=max_sources, max_targets=max_targets)
        self.cnn = TransportAugmentedCNN(
            input_channels=input_channels,
            map_channels=self.projector.map_channels,
            hidden_width=hidden_width,
            depth=depth,
        )
        self.transport_mlp = nn.Sequential(
            nn.LayerNorm(self.projector.vector_dim),
            nn.Linear(self.projector.vector_dim, transport_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(transport_hidden_dim, transport_hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.cnn.output_dim + transport_hidden_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, BoardTensorSpec(input_channels=self.semantic_adapter.spec.input_channels))
        parsed = self.semantic_adapter(x)
        atoms = self.atom_builder(parsed)
        cost = self.cost(atoms)
        plan = self.sinkhorn(cost, atoms)
        vector, maps, diagnostics = self.projector(plan, cost, atoms)
        if not self.use_pressure_maps:
            maps = torch.zeros_like(maps)
        if not self.use_scalar_transport_features:
            vector = torch.zeros_like(vector)
        board_embedding = self.cnn(x, maps)
        transport_embedding = self.transport_mlp(vector.to(dtype=board_embedding.dtype))
        logits = self.classifier(torch.cat([board_embedding, transport_embedding], dim=1))
        if self.num_classes == 1:
            logits = logits.squeeze(-1)
        output = {"logits": logits}
        output.update(diagnostics)
        return output


def build_entropic_chess_geometry_transport_network_from_config(config: dict[str, Any]) -> ChessGeometryTransportNet:
    transport_config = dict(config.get("transport", {})) if isinstance(config.get("transport"), dict) else {}
    return ChessGeometryTransportNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        encoding_adapter=str(config.get("encoding_adapter", config.get("encoding", SIMPLE_18))),
        max_sources=int(config.get("max_sources", transport_config.get("max_sources", 16))),
        max_targets=int(config.get("max_targets", transport_config.get("max_targets", 40))),
        epsilon=float(config.get("epsilon", transport_config.get("epsilon", 0.25))),
        sinkhorn_iters=int(config.get("sinkhorn_iters", transport_config.get("sinkhorn_iters", 8))),
        distance_cap=float(config.get("distance_cap", transport_config.get("distance_cap", 8.0))),
        use_pressure_maps=bool(config.get("use_pressure_maps", transport_config.get("use_pressure_maps", True))),
        use_scalar_transport_features=bool(
            config.get("use_scalar_transport_features", transport_config.get("use_scalar_transport_features", True))
        ),
        cost_ablation_mode=str(config.get("cost_ablation_mode", transport_config.get("cost_ablation_mode", "none"))),
        fail_closed_semantic_adapter=bool(
            config.get(
                "fail_closed_semantic_adapter",
                transport_config.get("fail_closed_semantic_adapter", True),
            )
        ),
        hidden_width=int(config.get("hidden_width", config.get("channels", 64))),
        depth=int(config.get("depth", 3)),
        transport_hidden_dim=int(config.get("transport_hidden_dim", 64)),
        classifier_hidden_dim=int(config.get("classifier_hidden_dim", config.get("hidden_dim", 96))),
        dropout=float(config.get("dropout", 0.1)),
    )


build_chess_geometry_transport_net = build_entropic_chess_geometry_transport_network_from_config

"""King-Anchored Material-Null Transport Bottleneck (idea i032)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.data.board_features import SIMPLE_18
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


_CENTER_SQUARE = 27
_SOURCE_ROLE_PRIOR = torch.tensor([1.0, 3.0, 3.0, 5.0, 9.0, 2.0], dtype=torch.float32)
_TARGET_TYPE_PRIOR = torch.tensor([1.0, 3.0, 3.0, 5.0, 9.0, 12.0, 6.0], dtype=torch.float32)


def _inverse_softplus(values: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(values.clamp_min(1e-4)))


def _knight_distance_matrix() -> torch.Tensor:
    moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
    dist = torch.full((64, 64), 7.0, dtype=torch.float32)
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
    return dist / dist.max().clamp_min(1.0)


@dataclass(frozen=True)
class AdaptedBoard:
    pieces: torch.Tensor
    white_to_move: torch.Tensor


@dataclass(frozen=True)
class TransportCandidates:
    source_roles: torch.Tensor
    source_squares: torch.Tensor
    source_mask: torch.Tensor
    target_types: torch.Tensor
    target_squares: torch.Tensor
    target_mask: torch.Tensor
    source_is_white: torch.Tensor


class Simple18PieceAdapter(nn.Module):
    def __init__(self, input_channels: int = 18, encoding_adapter: str = SIMPLE_18) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=input_channels)
        if encoding_adapter != SIMPLE_18 or input_channels != 18:
            raise ValueError(
                "KingAnchoredMaterialNullTransportBottleneck supports deterministic geometry only for simple_18."
            )

    def forward(self, x: torch.Tensor) -> AdaptedBoard:
        x = require_board_tensor(x, self.spec)
        white = x[:, 0:6].clamp_min(0.0)
        black = x[:, 6:12].clamp_min(0.0)
        pieces = torch.stack([white, black], dim=1)
        white_to_move = x[:, 12].mean(dim=(1, 2)) >= 0.5
        return AdaptedBoard(pieces=pieces, white_to_move=white_to_move)


class PieceTargetCandidateBuilder(nn.Module):
    def __init__(self, max_source_candidates: int = 16, max_target_candidates: int = 25) -> None:
        super().__init__()
        if max_source_candidates < 1:
            raise ValueError("max_source_candidates must be positive")
        if max_target_candidates < 10:
            raise ValueError("max_target_candidates must fit at least one piece plus king-zone targets")
        self.max_source_candidates = max_source_candidates
        self.max_piece_targets = max_target_candidates - 9
        self.max_target_candidates = max_target_candidates
        offsets = torch.tensor(
            [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)],
            dtype=torch.long,
        )
        self.register_buffer("king_zone_offsets", offsets, persistent=False)

    def _select_side(self, pieces: torch.Tensor, is_white: torch.Tensor) -> torch.Tensor:
        mask = is_white.view(-1, 1, 1, 1, 1)
        return torch.where(mask, pieces[:, 0:1], pieces[:, 1:2]).squeeze(1)

    def _top_piece_slots(self, side_pieces: torch.Tensor, max_candidates: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = side_pieces.shape[0]
        flat = side_pieces.flatten(1)
        ordering = torch.arange(flat.shape[1], device=flat.device, dtype=flat.dtype)
        scores = flat * 1000.0 + (flat.shape[1] - 1 - ordering).view(1, -1) / 1000.0
        _values, indices = scores.topk(max_candidates, dim=1)
        occupied = torch.gather(flat, 1, indices) > 0.5
        roles = (indices // 64).long()
        squares = (indices % 64).long()
        return roles, squares, occupied

    def _king_zone(self, target_pieces: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = target_pieces.shape[0]
        king_flat = target_pieces[:, 5].flatten(1)
        values, king_square = king_flat.max(dim=1)
        king_square = torch.where(values > 0.5, king_square, king_square.new_full((batch,), _CENTER_SQUARE))
        rank = king_square // 8
        file = king_square % 8
        offsets = self.king_zone_offsets.to(device=target_pieces.device)
        zone_rank = rank[:, None] + offsets[None, :, 0]
        zone_file = file[:, None] + offsets[None, :, 1]
        valid = (zone_rank >= 0) & (zone_rank < 8) & (zone_file >= 0) & (zone_file < 8)
        zone_rank = zone_rank.clamp(0, 7)
        zone_file = zone_file.clamp(0, 7)
        return (zone_rank * 8 + zone_file).long(), valid

    def forward(self, board: AdaptedBoard, *, reverse: bool = False) -> TransportCandidates:
        stm_white = board.white_to_move
        source_is_white = ~stm_white if reverse else stm_white
        target_is_white = ~source_is_white
        source_pieces = self._select_side(board.pieces, source_is_white)
        target_pieces = self._select_side(board.pieces, target_is_white)

        source_roles, source_squares, source_mask = self._top_piece_slots(source_pieces, self.max_source_candidates)
        no_source = ~source_mask.any(dim=1)
        first = torch.arange(self.max_source_candidates, device=source_mask.device).view(1, -1) == 0
        source_mask = torch.where(no_source[:, None] & first, torch.ones_like(source_mask), source_mask)
        source_roles = torch.where(no_source[:, None] & first, source_roles.new_full(source_roles.shape, 5), source_roles)
        source_squares = torch.where(
            no_source[:, None] & first,
            source_squares.new_full(source_squares.shape, _CENTER_SQUARE),
            source_squares,
        )

        target_roles, target_squares, target_mask = self._top_piece_slots(target_pieces, self.max_piece_targets)
        zone_squares, zone_mask = self._king_zone(target_pieces)
        zone_types = target_roles.new_full(zone_squares.shape, 6)
        target_types = torch.cat([target_roles, zone_types], dim=1)
        target_squares = torch.cat([target_squares, zone_squares], dim=1)
        target_mask = torch.cat([target_mask, zone_mask], dim=1)
        return TransportCandidates(
            source_roles=source_roles,
            source_squares=source_squares,
            source_mask=source_mask,
            target_types=target_types,
            target_squares=target_squares,
            target_mask=target_mask,
            source_is_white=source_is_white,
        )


class KingAnchoredMaterialNullSampler(nn.Module):
    def __init__(self, null_samples: int = 4, seed: int = 42) -> None:
        super().__init__()
        if null_samples < 0:
            raise ValueError("null_samples must be non-negative")
        self.null_samples = null_samples
        self.seed = seed

    def _first_fixed_square(self, squares: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
        sentinel = squares.new_full(squares.shape, 64)
        values = torch.where(fixed, squares, sentinel).amin(dim=1)
        return torch.where(values < 64, values, squares[:, 0])

    def forward(self, candidates: TransportCandidates) -> tuple[torch.Tensor, torch.Tensor]:
        batch, ns = candidates.source_squares.shape
        nt = candidates.target_squares.shape[1]
        if self.null_samples == 0:
            return (
                candidates.source_squares[:, None, :].expand(batch, 0, ns),
                candidates.target_squares[:, None, :].expand(batch, 0, nt),
            )
        device = candidates.source_squares.device
        source_fixed = (candidates.source_roles == 5) & candidates.source_mask
        target_fixed = ((candidates.target_types == 5) | (candidates.target_types == 6)) & candidates.target_mask
        fixed_source_square = self._first_fixed_square(candidates.source_squares, source_fixed)
        fixed_target_square = self._first_fixed_square(candidates.target_squares, target_fixed)
        signature = (
            ((candidates.source_roles + 1) * candidates.source_mask.long()).sum(dim=1) * 17
            + ((candidates.target_types + 7) * candidates.target_mask.long()).sum(dim=1) * 31
            + candidates.source_is_white.long() * 43
        ).float()
        square_ids = torch.arange(64, device=device, dtype=torch.float32)
        sample_ids = torch.arange(self.null_samples, device=device, dtype=torch.float32)
        phase = (
            signature[:, None, None] * 0.131
            + sample_ids[None, :, None] * 1.917
            + float(self.seed) * 0.071
        )
        scores = torch.sin((square_ids.view(1, 1, 64) + 1.0) * (phase + 0.37))
        fixed = (square_ids.view(1, 1, 64).long() == fixed_source_square[:, None, None]) | (
            square_ids.view(1, 1, 64).long() == fixed_target_square[:, None, None]
        )
        scores = scores.masked_fill(fixed, -2.0)
        permutation = scores.argsort(dim=-1, descending=True)
        source_assigned = permutation[:, :, :ns]
        target_assigned = permutation[:, :, ns : ns + nt]
        null_source = torch.where(source_fixed[:, None, :], candidates.source_squares[:, None, :], source_assigned)
        null_target = torch.where(target_fixed[:, None, :], candidates.target_squares[:, None, :], target_assigned)
        return null_source.long(), null_target.long()


class ChessGeometryCost(nn.Module):
    def __init__(self, transport_heads: int = 4, cost_floor: float = 1e-4) -> None:
        super().__init__()
        self.transport_heads = transport_heads
        self.cost_floor = cost_floor
        feature_dim = 12
        self.feature_weights = nn.Parameter(torch.randn(transport_heads, feature_dim) * 0.05)
        self.source_role_bias = nn.Parameter(torch.zeros(transport_heads, 6))
        self.target_type_bias = nn.Parameter(torch.zeros(transport_heads, 7))
        self.head_bias = nn.Parameter(torch.full((transport_heads,), -0.5))
        rank = torch.arange(64, dtype=torch.float32) // 8
        file = torch.arange(64, dtype=torch.float32) % 8
        self.register_buffer("rank", rank, persistent=False)
        self.register_buffer("file", file, persistent=False)
        self.register_buffer("knight_distance", _knight_distance_matrix(), persistent=False)

    def _features(
        self,
        source_squares: torch.Tensor,
        source_roles: torch.Tensor,
        target_squares: torch.Tensor,
        target_types: torch.Tensor,
        source_is_white: torch.Tensor,
    ) -> torch.Tensor:
        rank = self.rank.to(device=source_squares.device, dtype=torch.float32)
        file = self.file.to(device=source_squares.device, dtype=torch.float32)
        src_rank = rank[source_squares].unsqueeze(-1)
        src_file = file[source_squares].unsqueeze(-1)
        tgt_rank = rank[target_squares].unsqueeze(-2)
        tgt_file = file[target_squares].unsqueeze(-2)
        dr = (src_rank - tgt_rank).abs()
        df = (src_file - tgt_file).abs()
        manhattan = (dr + df) / 14.0
        chebyshev = torch.maximum(dr, df) / 7.0
        same_file = (df == 0).float()
        same_rank = (dr == 0).float()
        same_diag = (dr == df).float()
        queen_line = ((df == 0) | (dr == 0) | (dr == df)).float()
        knight = self.knight_distance.to(device=source_squares.device)[
            source_squares.unsqueeze(-1), target_squares.unsqueeze(-2)
        ]
        white_forward = (src_rank - tgt_rank) / 7.0
        black_forward = (tgt_rank - src_rank) / 7.0
        forward = torch.where(source_is_white.view(-1, 1, 1), white_forward, black_forward).clamp(-1.0, 1.0)
        pawn_dist = (df / 7.0 + torch.relu(-forward)).clamp(0.0, 1.0)
        bishop_dist = torch.where(same_diag.bool(), chebyshev, torch.ones_like(chebyshev))
        rook_dist = torch.where((same_file.bool() | same_rank.bool()), chebyshev, torch.ones_like(chebyshev))
        queen_dist = torch.where(queen_line.bool(), chebyshev, torch.ones_like(chebyshev))
        role = source_roles.unsqueeze(-1)
        role_dist = torch.where(
            role == 0,
            pawn_dist,
            torch.where(
                role == 1,
                knight,
                torch.where(
                    role == 2,
                    bishop_dist,
                    torch.where(role == 3, rook_dist, torch.where(role == 4, queen_dist, chebyshev)),
                ),
            ),
        )
        king_zone = (target_types.unsqueeze(-2) == 6).float().expand_as(manhattan)
        high_value = (
            ((target_types.unsqueeze(-2) == 3) | (target_types.unsqueeze(-2) == 4) | (target_types.unsqueeze(-2) == 5))
            .float()
            .expand_as(manhattan)
        )
        return torch.stack(
            [
                manhattan,
                chebyshev,
                same_file,
                same_rank,
                same_diag,
                queen_line,
                knight,
                forward,
                forward.abs(),
                role_dist,
                king_zone,
                high_value,
            ],
            dim=-1,
        )

    def forward(
        self,
        source_squares: torch.Tensor,
        source_roles: torch.Tensor,
        target_squares: torch.Tensor,
        target_types: torch.Tensor,
        source_is_white: torch.Tensor,
    ) -> torch.Tensor:
        features = self._features(source_squares, source_roles, target_squares, target_types, source_is_white)
        base = torch.einsum("bnmd,hd->bhnm", features, self.feature_weights)
        src_one_hot = F.one_hot(source_roles.clamp(0, 5), num_classes=6).float()
        tgt_one_hot = F.one_hot(target_types.clamp(0, 6), num_classes=7).float()
        source_bias = torch.einsum("bnr,hr->bhn", src_one_hot, self.source_role_bias).unsqueeze(-1)
        target_bias = torch.einsum("bmt,ht->bhm", tgt_one_hot, self.target_type_bias).unsqueeze(-2)
        logits = base + source_bias + target_bias + self.head_bias.view(1, -1, 1, 1)
        return F.softplus(logits).clamp_max(20.0) + self.cost_floor


class MaskedLogSinkhorn(nn.Module):
    def __init__(self, iterations: int = 12, epsilon: float = 0.08) -> None:
        super().__init__()
        if iterations < 1:
            raise ValueError("sinkhorn iterations must be >= 1")
        if epsilon <= 0:
            raise ValueError("sinkhorn epsilon must be positive")
        self.iterations = iterations
        self.epsilon = epsilon

    def forward(
        self,
        cost: torch.Tensor,
        source_mass: torch.Tensor,
        target_mass: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        cost = cost.float()
        source_mass = source_mass.float() * source_mask.float()
        target_mass = target_mass.float() * target_mask.float()
        source_mass = source_mass / source_mass.sum(dim=1, keepdim=True).clamp_min(1e-8)
        target_mass = target_mass / target_mass.sum(dim=1, keepdim=True).clamp_min(1e-8)
        log_mu = source_mass.clamp_min(1e-12).log().unsqueeze(1).expand(-1, cost.shape[1], -1)
        log_nu = target_mass.clamp_min(1e-12).log().unsqueeze(1).expand(-1, cost.shape[1], -1)
        valid_pair = source_mask[:, None, :, None] & target_mask[:, None, None, :]
        log_kernel = (-cost / self.epsilon).masked_fill(~valid_pair, -1e9)
        log_u = torch.zeros_like(log_mu)
        log_v = torch.zeros_like(log_nu)
        for _ in range(self.iterations):
            log_u = log_mu - torch.logsumexp(log_kernel + log_v.unsqueeze(-2), dim=-1)
            log_u = log_u.masked_fill(~source_mask[:, None, :], 0.0)
            log_v = log_nu - torch.logsumexp(log_kernel + log_u.unsqueeze(-1), dim=-2)
            log_v = log_v.masked_fill(~target_mask[:, None, :], 0.0)
        plan = torch.exp(log_u.unsqueeze(-1) + log_kernel + log_v.unsqueeze(-2)) * valid_pair.float()
        return plan / plan.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)


class TransportDescriptorPool(nn.Module):
    descriptor_dim = 15

    def __init__(self) -> None:
        super().__init__()
        rank = torch.arange(64, dtype=torch.float32) // 8
        file = torch.arange(64, dtype=torch.float32) % 8
        self.register_buffer("rank", rank, persistent=False)
        self.register_buffer("file", file, persistent=False)

    def _pair_geometry(
        self,
        source_squares: torch.Tensor,
        target_squares: torch.Tensor,
        source_is_white: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rank = self.rank.to(device=source_squares.device)
        file = self.file.to(device=source_squares.device)
        src_rank = rank[source_squares].unsqueeze(-1)
        src_file = file[source_squares].unsqueeze(-1)
        tgt_rank = rank[target_squares].unsqueeze(-2)
        tgt_file = file[target_squares].unsqueeze(-2)
        distance = torch.maximum((src_rank - tgt_rank).abs(), (src_file - tgt_file).abs())
        white_forward = src_rank - tgt_rank
        black_forward = tgt_rank - src_rank
        forward = torch.where(source_is_white.view(-1, 1, 1), white_forward, black_forward)
        return distance, forward

    def _target_type_mass(
        self,
        target_mass: torch.Tensor,
        target_types: torch.Tensor,
        target_mask: torch.Tensor,
        type_ids: tuple[int, ...],
    ) -> torch.Tensor:
        selected = torch.zeros_like(target_mask)
        for type_id in type_ids:
            selected = selected | (target_types == type_id)
        return (target_mass * (selected & target_mask).float().unsqueeze(1)).sum(dim=-1)

    def forward(self, plan: torch.Tensor, cost: torch.Tensor, candidates: TransportCandidates) -> torch.Tensor:
        target_plan_mass = plan.sum(dim=-2)
        flat_plan = plan.flatten(2)
        top4 = flat_plan.topk(min(4, flat_plan.shape[-1]), dim=-1).values.sum(dim=-1)
        expected_cost = (plan * cost.float()).sum(dim=(-2, -1))
        entropy = -(plan * plan.clamp_min(1e-12).log()).sum(dim=(-2, -1))
        entropy = entropy / torch.log(plan.new_tensor(float(plan.shape[-2] * plan.shape[-1])))
        distance, forward = self._pair_geometry(
            candidates.source_squares,
            candidates.target_squares,
            candidates.source_is_white,
        )
        distance = distance.unsqueeze(1)
        forward = forward.unsqueeze(1)
        b01 = (plan * (distance <= 1).float()).sum(dim=(-2, -1))
        b2 = (plan * (distance == 2).float()).sum(dim=(-2, -1))
        b3 = (plan * (distance == 3).float()).sum(dim=(-2, -1))
        b4 = (plan * (distance >= 4).float()).sum(dim=(-2, -1))
        forward_mass = (plan * (forward > 0).float()).sum(dim=(-2, -1))
        return torch.stack(
            [
                expected_cost,
                entropy,
                flat_plan.amax(dim=-1),
                top4,
                self._target_type_mass(target_plan_mass, candidates.target_types, candidates.target_mask, (6,)),
                self._target_type_mass(target_plan_mass, candidates.target_types, candidates.target_mask, (4,)),
                self._target_type_mass(target_plan_mass, candidates.target_types, candidates.target_mask, (3,)),
                self._target_type_mass(target_plan_mass, candidates.target_types, candidates.target_mask, (1, 2)),
                self._target_type_mass(target_plan_mass, candidates.target_types, candidates.target_mask, (0,)),
                self._target_type_mass(target_plan_mass, candidates.target_types, candidates.target_mask, (5,)),
                b01,
                b2,
                b3,
                b4,
                forward_mass,
            ],
            dim=-1,
        )


class KingAnchoredMaterialNullTransportBottleneck(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        encoding_adapter: str = SIMPLE_18,
        max_source_candidates: int = 16,
        max_target_candidates: int = 25,
        transport_heads: int = 4,
        null_samples: int = 4,
        sinkhorn_iters: int = 12,
        sinkhorn_epsilon: float = 0.08,
        hidden_dim: int = 128,
        descriptor_dropout: float = 0.05,
        seed: int = 42,
        use_blocker_cost: bool = False,
        use_material_adversary: bool = False,
        fail_closed_unknown_channels: bool = True,
        max_pair_cells_per_chunk: int = 2_000_000,
    ) -> None:
        super().__init__()
        if use_blocker_cost:
            raise ValueError("use_blocker_cost is not part of the minimal KAMN-OTB implementation")
        if use_material_adversary:
            raise ValueError("use_material_adversary requires trainer support and is disabled")
        if not fail_closed_unknown_channels:
            raise ValueError("fail_closed_unknown_channels must remain true for deterministic geometry")
        self.num_classes = num_classes
        self.transport_heads = transport_heads
        self.null_samples = null_samples
        self.max_pair_cells_per_chunk = max_pair_cells_per_chunk
        self.adapter = Simple18PieceAdapter(input_channels=input_channels, encoding_adapter=encoding_adapter)
        self.candidates = PieceTargetCandidateBuilder(max_source_candidates, max_target_candidates)
        self.null_sampler = KingAnchoredMaterialNullSampler(null_samples=null_samples, seed=seed)
        self.cost = ChessGeometryCost(transport_heads=transport_heads)
        self.sinkhorn = MaskedLogSinkhorn(iterations=sinkhorn_iters, epsilon=sinkhorn_epsilon)
        self.pool = TransportDescriptorPool()
        self.source_mass_logits = nn.Parameter(_inverse_softplus(_SOURCE_ROLE_PRIOR.clone()))
        self.target_mass_logits = nn.Parameter(_inverse_softplus(_TARGET_TYPE_PRIOR.clone()))
        descriptor_dim = 3 * transport_heads * self.pool.descriptor_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(descriptor_dim),
            nn.Linear(descriptor_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(descriptor_dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_classes),
        )

    def _masses(self, candidates: TransportCandidates) -> tuple[torch.Tensor, torch.Tensor]:
        source_weights = F.softplus(self.source_mass_logits)[candidates.source_roles.clamp(0, 5)]
        target_weights = F.softplus(self.target_mass_logits)[candidates.target_types.clamp(0, 6)]
        source_weights = source_weights * candidates.source_mask.float()
        target_weights = target_weights * candidates.target_mask.float()
        source_mass = source_weights / source_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        target_mass = target_weights / target_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return source_mass, target_mass

    def _descriptors_for(self, candidates: TransportCandidates) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source_mass, target_mass = self._masses(candidates)
        real_cost = self.cost(
            candidates.source_squares,
            candidates.source_roles,
            candidates.target_squares,
            candidates.target_types,
            candidates.source_is_white,
        )
        real_plan = self.sinkhorn(real_cost, source_mass, target_mass, candidates.source_mask, candidates.target_mask)
        real_desc = self.pool(real_plan, real_cost, candidates)
        if self.null_samples == 0:
            null_mean = torch.zeros_like(real_desc)
        else:
            null_source, null_target = self.null_sampler(candidates)
            null_descs = []
            for sample_idx in range(self.null_samples):
                null_candidates = TransportCandidates(
                    source_roles=candidates.source_roles,
                    source_squares=null_source[:, sample_idx],
                    source_mask=candidates.source_mask,
                    target_types=candidates.target_types,
                    target_squares=null_target[:, sample_idx],
                    target_mask=candidates.target_mask,
                    source_is_white=candidates.source_is_white,
                )
                null_cost = self.cost(
                    null_candidates.source_squares,
                    null_candidates.source_roles,
                    null_candidates.target_squares,
                    null_candidates.target_types,
                    null_candidates.source_is_white,
                )
                null_plan = self.sinkhorn(
                    null_cost,
                    source_mass,
                    target_mass,
                    null_candidates.source_mask,
                    null_candidates.target_mask,
                )
                null_descs.append(self.pool(null_plan, null_cost, null_candidates))
            null_mean = torch.stack(null_descs, dim=1).mean(dim=1)
        return real_desc, null_mean, real_desc - null_mean

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = self.adapter(x)
        forward_candidates = self.candidates(board, reverse=False)
        reverse_candidates = self.candidates(board, reverse=True)
        forward_real, forward_null, forward_resid = self._descriptors_for(forward_candidates)
        reverse_real, reverse_null, reverse_resid = self._descriptors_for(reverse_candidates)
        signed = forward_resid - reverse_resid
        z = torch.cat([forward_resid.flatten(1), reverse_resid.flatten(1), signed.flatten(1)], dim=1)
        logits = self.classifier(z)
        if self.num_classes == 1:
            logits = logits.squeeze(-1)
        return {
            "logits": logits,
            "transport_residual_norm": z.norm(dim=1) / (z.shape[1] ** 0.5),
            "forward_real_cost": forward_real[..., 0].mean(dim=1),
            "forward_null_cost": forward_null[..., 0].mean(dim=1),
            "reverse_real_cost": reverse_real[..., 0].mean(dim=1),
            "reverse_null_cost": reverse_null[..., 0].mean(dim=1),
            "signed_king_zone_residual": signed[..., 4].mean(dim=1),
            "material_null_cost_gap": (forward_resid[..., 0] - reverse_resid[..., 0]).mean(dim=1),
        }


def build_king_anchored_material_null_transport_bottleneck_from_config(
    config: dict[str, Any],
) -> KingAnchoredMaterialNullTransportBottleneck:
    return KingAnchoredMaterialNullTransportBottleneck(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        encoding_adapter=str(config.get("encoding_adapter", config.get("encoding", SIMPLE_18))),
        max_source_candidates=int(config.get("max_source_candidates", 16)),
        max_target_candidates=int(config.get("max_target_candidates", 25)),
        transport_heads=int(config.get("transport_heads", 4)),
        null_samples=int(config.get("null_samples", 4)),
        sinkhorn_iters=int(config.get("sinkhorn_iters", 12)),
        sinkhorn_epsilon=float(config.get("sinkhorn_epsilon", 0.08)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        descriptor_dropout=float(config.get("descriptor_dropout", config.get("dropout", 0.05))),
        seed=int(config.get("seed", 42)),
        use_blocker_cost=bool(config.get("use_blocker_cost", False)),
        use_material_adversary=bool(config.get("use_material_adversary", False)),
        fail_closed_unknown_channels=bool(config.get("fail_closed_unknown_channels", True)),
        max_pair_cells_per_chunk=int(config.get("max_pair_cells_per_chunk", 2_000_000)),
    )

"""Tactical Transport Imbalance Network (idea i031)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.data.board_features import SIMPLE_18
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


_PIECE_VALUE_PRIOR = torch.tensor([1.0, 3.0, 3.0, 5.0, 9.0, 2.0], dtype=torch.float32)
_TARGET_VALUE_PRIOR = torch.tensor([1.0, 3.0, 3.0, 5.0, 9.0, 12.0], dtype=torch.float32)


def _inverse_softplus(values: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(values.clamp_min(1e-4)))


def _knight_distance_matrix() -> torch.Tensor:
    moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
    dist = torch.full((64, 64), 7.0, dtype=torch.float32)
    for source in range(64):
        queue = [source]
        dist[source, source] = 0.0
        head = 0
        while head < len(queue):
            square = queue[head]
            head += 1
            rank, file = divmod(square, 8)
            for dr, df in moves:
                rr, ff = rank + dr, file + df
                if 0 <= rr < 8 and 0 <= ff < 8:
                    target = rr * 8 + ff
                    if dist[source, target] > dist[source, square] + 1.0:
                        dist[source, target] = dist[source, square] + 1.0
                        queue.append(target)
    return dist / dist.max().clamp_min(1.0)


def _chess_geometry_cost_basis() -> torch.Tensor:
    rank = torch.arange(64, dtype=torch.float32) // 8
    file = torch.arange(64, dtype=torch.float32) % 8
    src_rank = rank.view(64, 1)
    src_file = file.view(64, 1)
    tgt_rank = rank.view(1, 64)
    tgt_file = file.view(1, 64)
    dr = (src_rank - tgt_rank).abs()
    df = (src_file - tgt_file).abs()
    same_rank_or_file = ((dr == 0) | (df == 0)).float()
    same_diagonal = (dr == df).float()
    queen_line = ((dr == 0) | (df == 0) | (dr == df)).float()
    src_parity = (src_rank + src_file).remainder(2)
    tgt_parity = (tgt_rank + tgt_file).remainder(2)
    backward_rank = (tgt_rank - src_rank).clamp_min(0.0) / 7.0
    return torch.stack(
        [
            (dr + df) / 14.0,
            torch.maximum(dr, df) / 7.0,
            1.0 - same_rank_or_file,
            1.0 - same_diagonal,
            _knight_distance_matrix(),
            (src_parity != tgt_parity).float(),
            1.0 - queen_line,
            backward_rank,
        ],
        dim=0,
    )


@dataclass(frozen=True)
class SimpleBoardState:
    white_pieces: torch.Tensor
    black_pieces: torch.Tensor
    white_to_move: torch.Tensor


class EncodingAdapter(nn.Module):
    """Extracts current-board simple_18 piece planes and side-to-move metadata."""

    def __init__(self, input_channels: int = 18, encoding: str = SIMPLE_18) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.encoding = encoding
        if encoding != SIMPLE_18 or input_channels != 18:
            raise ValueError(
                "TacticalTransportImbalanceNetwork currently supports only simple_18 "
                "with 18 input channels for deterministic transport geometry."
            )

    def forward(self, x: torch.Tensor) -> SimpleBoardState:
        x = require_board_tensor(x, self.spec)
        white_pieces = x[:, 0:6].clamp_min(0.0)
        black_pieces = x[:, 6:12].clamp_min(0.0)
        white_to_move = x[:, 12].mean(dim=(1, 2)) >= 0.5
        return SimpleBoardState(white_pieces=white_pieces, black_pieces=black_pieces, white_to_move=white_to_move)


class SideToMoveCanonicalizer(nn.Module):
    """Swaps colors and flips ranks so the side to move is always the own side."""

    def forward(self, state: SimpleBoardState) -> tuple[torch.Tensor, torch.Tensor]:
        white_mask = state.white_to_move.view(-1, 1, 1, 1)
        own_by_color = torch.where(white_mask, state.white_pieces, state.black_pieces)
        opp_by_color = torch.where(white_mask, state.black_pieces, state.white_pieces)
        own_flipped = torch.flip(own_by_color, dims=(-2,))
        opp_flipped = torch.flip(opp_by_color, dims=(-2,))
        own = torch.where(white_mask, own_by_color, own_flipped)
        opp = torch.where(white_mask, opp_by_color, opp_flipped)
        pieces = torch.cat([own, opp], dim=1)
        return pieces, pieces.flatten(2)


class LocalResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0, use_batchnorm: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.norm1 = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.norm2 = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        return F.gelu(x + residual)


class LocalBoardEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 64,
        residual_blocks: int = 3,
        dropout: float = 0.0,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm),
            nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity(),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                LocalResidualBlock(channels, dropout=dropout, use_batchnorm=use_batchnorm)
                for _ in range(residual_blocks)
            ]
        )
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=not use_batchnorm),
            nn.BatchNorm2d(channels * 2) if use_batchnorm else nn.Identity(),
            nn.GELU(),
        )
        self.output_dim = channels * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = require_board_tensor(x, self.spec)
        board = self.proj(self.blocks(self.stem(x)))
        return board.mean(dim=(2, 3))


class TransportMassHead(nn.Module):
    def __init__(self, heads: int = 8, mass_floor: float = 1e-4) -> None:
        super().__init__()
        self.heads = heads
        self.mass_floor = mass_floor
        source_init = _inverse_softplus(_PIECE_VALUE_PRIOR).repeat(heads, 1)
        target_init = _inverse_softplus(_TARGET_VALUE_PRIOR).repeat(heads, 1)
        head_offsets = torch.linspace(-0.12, 0.12, heads, dtype=torch.float32).view(heads, 1)
        piece_offsets = torch.linspace(-0.08, 0.08, 6, dtype=torch.float32).view(1, 6)
        self.source_piece_logits = nn.Parameter(source_init + head_offsets * piece_offsets)
        self.target_piece_logits = nn.Parameter(target_init - head_offsets * piece_offsets)
        self.source_square_bias = nn.Parameter(torch.zeros(heads, 64))
        self.target_square_bias = nn.Parameter(torch.zeros(heads, 64))
        self.king_ring_logits = nn.Parameter(_inverse_softplus(torch.full((heads,), 4.0)))
        self.register_buffer("king_ring_kernel", torch.ones(1, 1, 3, 3), persistent=False)

    def _king_ring(self, side_pieces: torch.Tensor) -> torch.Tensor:
        king = side_pieces[:, 5:6]
        ring = F.conv2d(king, self.king_ring_kernel.to(dtype=side_pieces.dtype), padding=1)
        return ring.clamp(0.0, 1.0).flatten(2).squeeze(1)

    def _distribution(self, raw: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        logits = (raw + self.mass_floor).clamp_min(self.mass_floor).log() + bias.unsqueeze(0)
        return torch.softmax(logits, dim=-1)

    def forward(self, canonical_pieces_64: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = canonical_pieces_64.shape[0]
        own_occ = canonical_pieces_64[:, :6]
        opp_occ = canonical_pieces_64[:, 6:]
        own_board = own_occ.view(batch, 6, 8, 8)
        opp_board = opp_occ.view(batch, 6, 8, 8)
        source_weights = F.softplus(self.source_piece_logits).to(dtype=canonical_pieces_64.dtype)
        target_weights = F.softplus(self.target_piece_logits).to(dtype=canonical_pieces_64.dtype)
        own_source_raw = torch.einsum("bps,hp->bhs", own_occ, source_weights)
        opp_source_raw = torch.einsum("bps,hp->bhs", opp_occ, source_weights)
        own_target_raw = torch.einsum("bps,hp->bhs", own_occ, target_weights)
        opp_target_raw = torch.einsum("bps,hp->bhs", opp_occ, target_weights)
        ring_weight = F.softplus(self.king_ring_logits).to(dtype=canonical_pieces_64.dtype).view(1, self.heads, 1)
        own_target_raw = own_target_raw + ring_weight * self._king_ring(own_board).unsqueeze(1)
        opp_target_raw = opp_target_raw + ring_weight * self._king_ring(opp_board).unsqueeze(1)
        source_bias = self.source_square_bias.to(dtype=canonical_pieces_64.dtype)
        target_bias = self.target_square_bias.to(dtype=canonical_pieces_64.dtype)
        return (
            self._distribution(own_source_raw, source_bias),
            self._distribution(opp_target_raw, target_bias),
            self._distribution(opp_source_raw, source_bias),
            self._distribution(own_target_raw, target_bias),
        )


class ChessCostBank(nn.Module):
    def __init__(self, heads: int = 8, cost_floor: float = 1e-3, cost_basis: str = "chess_geometry_v1") -> None:
        super().__init__()
        if cost_basis != "chess_geometry_v1":
            raise ValueError("TacticalTransportImbalanceNetwork supports cost_basis='chess_geometry_v1'")
        self.heads = heads
        self.cost_floor = cost_floor
        self.register_buffer("basis", _chess_geometry_cost_basis(), persistent=False)
        basis_count = int(self.basis.shape[0])
        alpha = torch.full((heads, basis_count), 0.35, dtype=torch.float32)
        alpha += torch.linspace(-0.08, 0.08, heads, dtype=torch.float32).view(heads, 1)
        alpha += torch.linspace(0.06, -0.06, basis_count, dtype=torch.float32).view(1, basis_count)
        self.alpha = nn.Parameter(alpha)
        self.beta = nn.Parameter(torch.full((heads,), -1.0, dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        basis = self.basis.to(dtype=self.alpha.dtype, device=self.alpha.device)
        cost_logits = self.beta[:, None, None] + torch.einsum("hm,mst->hst", self.alpha, basis)
        return F.softplus(cost_logits) + self.cost_floor


class SinkhornTransportBlock(nn.Module):
    def __init__(self, epsilon: float = 0.07, iterations: int = 8) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError("sinkhorn epsilon must be positive")
        if iterations < 1:
            raise ValueError("sinkhorn iterations must be >= 1")
        self.epsilon = epsilon
        self.iterations = iterations

    def forward(self, mu: torch.Tensor, nu: torch.Tensor, cost: torch.Tensor) -> torch.Tensor:
        mu32 = mu.float().clamp_min(1e-8)
        nu32 = nu.float().clamp_min(1e-8)
        mu32 = mu32 / mu32.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        nu32 = nu32 / nu32.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        kernel = torch.exp((-cost.float() / self.epsilon).clamp(min=-80.0, max=0.0)).clamp_min(1e-12)
        v = torch.ones_like(nu32) / float(nu32.shape[-1])
        for _ in range(self.iterations):
            u = mu32 / torch.einsum("hst,bht->bhs", kernel, v).clamp_min(1e-12)
            v = nu32 / torch.einsum("hst,bhs->bht", kernel, u).clamp_min(1e-12)
        plan = u.unsqueeze(-1) * kernel.unsqueeze(0) * v.unsqueeze(-2)
        return plan / plan.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)


class TransportFeaturePool(nn.Module):
    feature_dim_per_head = 10

    def __init__(self) -> None:
        super().__init__()
        rank = (torch.arange(64, dtype=torch.float32) // 8) / 7.0
        file = (torch.arange(64, dtype=torch.float32) % 8) / 7.0
        self.register_buffer("rank", rank, persistent=False)
        self.register_buffer("file", file, persistent=False)

    def _summaries(self, plan: torch.Tensor, cost: torch.Tensor) -> dict[str, torch.Tensor]:
        rank = self.rank.to(device=plan.device, dtype=plan.dtype)
        file = self.file.to(device=plan.device, dtype=plan.dtype)
        source_mass = plan.sum(dim=-1)
        target_mass = plan.sum(dim=-2)
        source_rank = (source_mass * rank.view(1, 1, 64)).sum(dim=-1)
        source_file = (source_mass * file.view(1, 1, 64)).sum(dim=-1)
        target_rank = (target_mass * rank.view(1, 1, 64)).sum(dim=-1)
        target_file = (target_mass * file.view(1, 1, 64)).sum(dim=-1)
        return {
            "cost": (plan * cost.unsqueeze(0).to(dtype=plan.dtype)).sum(dim=(-2, -1)),
            "entropy": -(plan * plan.clamp_min(1e-12).log()).sum(dim=(-2, -1)) / torch.log(
                plan.new_tensor(float(64 * 64))
            ),
            "max_entry": plan.amax(dim=(-2, -1)),
            "l2": plan.square().sum(dim=(-2, -1)),
            "rank_delta": target_rank - source_rank,
            "file_gap": (target_file - source_file).abs(),
        }

    def forward(self, plan_forward: torch.Tensor, plan_reverse: torch.Tensor, cost: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        fwd = self._summaries(plan_forward, cost)
        rev = self._summaries(plan_reverse, cost)
        per_head = torch.stack(
            [
                fwd["cost"],
                rev["cost"],
                rev["cost"] - fwd["cost"],
                fwd["entropy"],
                rev["entropy"],
                rev["entropy"] - fwd["entropy"],
                fwd["max_entry"] - rev["max_entry"],
                fwd["l2"] - rev["l2"],
                fwd["rank_delta"] - rev["rank_delta"],
                rev["file_gap"] - fwd["file_gap"],
            ],
            dim=-1,
        )
        diagnostics = {
            "transport_imbalance": (rev["cost"] - fwd["cost"] + rev["entropy"] - fwd["entropy"]).mean(dim=1),
            "forward_transport_cost": fwd["cost"].mean(dim=1),
            "reverse_transport_cost": rev["cost"].mean(dim=1),
            "transport_entropy_gap": (rev["entropy"] - fwd["entropy"]).mean(dim=1),
            "transport_concentration_gap": (fwd["l2"] - rev["l2"]).mean(dim=1),
            "transport_rank_moment_gap": (fwd["rank_delta"] - rev["rank_delta"]).mean(dim=1),
        }
        return per_head.flatten(1), diagnostics


class TacticalTransportImbalanceNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 128,
        residual_blocks: int = 3,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        encoding: str = SIMPLE_18,
        transport_heads: int = 8,
        sinkhorn_iters: int = 8,
        sinkhorn_epsilon: float = 0.07,
        mass_floor: float = 1e-4,
        cost_floor: float = 1e-3,
        cost_basis: str = "chess_geometry_v1",
        canonicalize_side_to_move: bool = True,
    ) -> None:
        super().__init__()
        if not canonicalize_side_to_move:
            raise ValueError("TacticalTransportImbalanceNetwork requires side-to-move canonicalization")
        self.num_classes = num_classes
        self.adapter = EncodingAdapter(input_channels=input_channels, encoding=encoding)
        self.canonicalizer = SideToMoveCanonicalizer()
        self.local_encoder = LocalBoardEncoder(
            input_channels=input_channels,
            channels=channels,
            residual_blocks=residual_blocks,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )
        self.mass_head = TransportMassHead(heads=transport_heads, mass_floor=mass_floor)
        self.cost_bank = ChessCostBank(heads=transport_heads, cost_floor=cost_floor, cost_basis=cost_basis)
        self.transport = SinkhornTransportBlock(epsilon=sinkhorn_epsilon, iterations=sinkhorn_iters)
        self.feature_pool = TransportFeaturePool()
        transport_dim = transport_heads * self.feature_pool.feature_dim_per_head
        self.classifier = nn.Sequential(
            nn.Linear(self.local_encoder.output_dim + transport_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        state = self.adapter(x)
        _canonical_pieces, canonical_pieces_64 = self.canonicalizer(state)
        z_cnn = self.local_encoder(x)
        mu_own, nu_opp, mu_opp, nu_own = self.mass_head(canonical_pieces_64)
        cost = self.cost_bank()
        plan_forward = self.transport(mu_own, nu_opp, cost)
        plan_reverse = self.transport(mu_opp, nu_own, cost)
        z_transport, diagnostics = self.feature_pool(plan_forward, plan_reverse, cost)
        logits = self.classifier(torch.cat([z_cnn, z_transport.to(dtype=z_cnn.dtype)], dim=1))
        if self.num_classes == 1:
            logits = logits.squeeze(-1)
        output = {"logits": logits}
        output.update(diagnostics)
        return output


def build_tactical_transport_imbalance_network_from_config(config: dict[str, Any]) -> TacticalTransportImbalanceNetwork:
    return TacticalTransportImbalanceNetwork(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        residual_blocks=int(config.get("residual_blocks", config.get("depth", 3))),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        encoding=str(config.get("encoding", SIMPLE_18)),
        transport_heads=int(config.get("transport_heads", 8)),
        sinkhorn_iters=int(config.get("sinkhorn_iters", 8)),
        sinkhorn_epsilon=float(config.get("sinkhorn_epsilon", 0.07)),
        mass_floor=float(config.get("mass_floor", 1e-4)),
        cost_floor=float(config.get("cost_floor", 1e-3)),
        cost_basis=str(config.get("cost_basis", "chess_geometry_v1")),
        canonicalize_side_to_move=bool(config.get("canonicalize_side_to_move", True)),
    )

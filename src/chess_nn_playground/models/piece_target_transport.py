"""Piece-Target Entropic Transport Bottleneck (idea i033)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.data.board_features import SIMPLE_18
from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


_SOURCE_PIECE_PRIOR = torch.tensor([1.0, 3.0, 3.0, 5.0, 9.0, 2.0], dtype=torch.float32)
_TARGET_PIECE_PRIOR = torch.tensor([1.0, 3.0, 3.0, 5.0, 9.0, 12.0], dtype=torch.float32)


def _inverse_softplus(values: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(values.clamp_min(1e-4)))


@dataclass(frozen=True)
class Simple18BoardState:
    white_pieces: torch.Tensor
    black_pieces: torch.Tensor
    white_to_move: torch.Tensor


@dataclass(frozen=True)
class CanonicalPieceState:
    friendly: torch.Tensor
    enemy: torch.Tensor
    friendly_flat: torch.Tensor
    enemy_flat: torch.Tensor


class PiecePlaneAdapter(nn.Module):
    """Extracts current-piece planes and side-to-move from simple_18 tensors."""

    def __init__(self, input_channels: int = 18, encoding_adapter: str = SIMPLE_18) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=input_channels)
        if encoding_adapter != SIMPLE_18 or input_channels != 18:
            raise ValueError(
                "PieceTargetEntropicTransportBottleneck currently supports only simple_18 "
                "with 18 input channels. Unknown deterministic channel maps fail closed."
            )

    def forward(self, x: torch.Tensor) -> Simple18BoardState:
        x = require_board_tensor(x, self.spec)
        return Simple18BoardState(
            white_pieces=x[:, 0:6].clamp_min(0.0),
            black_pieces=x[:, 6:12].clamp_min(0.0),
            white_to_move=x[:, 12].mean(dim=(1, 2)) >= 0.5,
        )


class RelativeBoardCanonicalizer(nn.Module):
    """Makes friendly pieces side-to-move pieces and flips ranks for black-to-move."""

    def forward(self, state: Simple18BoardState) -> CanonicalPieceState:
        white_mask = state.white_to_move.view(-1, 1, 1, 1)
        friendly_by_color = torch.where(white_mask, state.white_pieces, state.black_pieces)
        enemy_by_color = torch.where(white_mask, state.black_pieces, state.white_pieces)
        friendly = torch.where(white_mask, friendly_by_color, torch.flip(friendly_by_color, dims=(-2,)))
        enemy = torch.where(white_mask, enemy_by_color, torch.flip(enemy_by_color, dims=(-2,)))
        return CanonicalPieceState(
            friendly=friendly,
            enemy=enemy,
            friendly_flat=friendly.flatten(2),
            enemy_flat=enemy.flatten(2),
        )


class ConvNormGelu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BoardAdapter(nn.Module):
    def __init__(self, input_channels: int = 18, adapter_width: int = 64, adapter_depth: int = 3) -> None:
        super().__init__()
        if adapter_depth < 1:
            raise ValueError("adapter_depth must be >= 1")
        layers: list[nn.Module] = [ConvNormGelu(input_channels, adapter_width)]
        layers.extend(ConvNormGelu(adapter_width, adapter_width) for _ in range(adapter_depth - 1))
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.net = nn.Sequential(*layers)
        self.output_channels = adapter_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(require_board_tensor(x, self.spec))


class TransportMarginals(nn.Module):
    def __init__(self, transport_heads: int = 4, epsilon_mass: float = 1e-3) -> None:
        super().__init__()
        if transport_heads < 1:
            raise ValueError("transport_heads must be positive")
        if epsilon_mass <= 0:
            raise ValueError("epsilon_mass must be positive")
        self.transport_heads = transport_heads
        self.epsilon_mass = epsilon_mass
        offsets = torch.linspace(-0.12, 0.12, transport_heads, dtype=torch.float32).view(transport_heads, 1)
        piece_offsets = torch.linspace(-0.06, 0.06, 6, dtype=torch.float32).view(1, 6)
        self.source_logits = nn.Parameter(_inverse_softplus(_SOURCE_PIECE_PRIOR).repeat(transport_heads, 1))
        self.target_logits = nn.Parameter(_inverse_softplus(_TARGET_PIECE_PRIOR).repeat(transport_heads, 1))
        with torch.no_grad():
            self.source_logits.add_(offsets * piece_offsets)
            self.target_logits.sub_(offsets * piece_offsets)

    def _distribution(self, piece_flat: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        weights = F.softplus(logits).to(device=piece_flat.device, dtype=piece_flat.dtype)
        raw = torch.einsum("bps,hp->bhs", piece_flat, weights)
        raw = raw + self.epsilon_mass
        return raw / raw.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def forward(self, canonical: CanonicalPieceState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self._distribution(canonical.friendly_flat, self.source_logits),
            self._distribution(canonical.enemy_flat, self.target_logits),
            self._distribution(canonical.enemy_flat, self.source_logits),
            self._distribution(canonical.friendly_flat, self.target_logits),
        )


def _board_geometry_features() -> torch.Tensor:
    square = torch.arange(64, dtype=torch.float32)
    rank = square // 8
    file = square % 8
    src_rank = rank.view(64, 1)
    src_file = file.view(64, 1)
    tgt_rank = rank.view(1, 64)
    tgt_file = file.view(1, 64)
    signed_dr = (tgt_rank - src_rank) / 7.0
    signed_df = (tgt_file - src_file) / 7.0
    abs_dr = signed_dr.abs()
    abs_df = signed_df.abs()
    manhattan = (abs_dr + abs_df) / 2.0
    chebyshev = torch.maximum(abs_dr, abs_df)
    same_file = (src_file == tgt_file).float()
    same_rank = (src_rank == tgt_rank).float()
    same_diag = ((src_rank - tgt_rank).abs() == (src_file - tgt_file).abs()).float()
    same_antidiag = ((src_rank + src_file) == (tgt_rank + tgt_file)).float()
    knight_vector = (
        (((src_rank - tgt_rank).abs() == 1) & ((src_file - tgt_file).abs() == 2))
        | (((src_rank - tgt_rank).abs() == 2) & ((src_file - tgt_file).abs() == 1))
    ).float()
    forward_relation = (src_rank - tgt_rank).clamp_min(0.0) / 7.0
    center_rank = (rank - 3.5).abs() / 3.5
    center_file = (file - 3.5).abs() / 3.5
    source_center = ((center_rank + center_file) / 2.0).view(64, 1).expand(64, 64)
    target_center = ((center_rank + center_file) / 2.0).view(1, 64).expand(64, 64)
    return torch.stack(
        [
            signed_df,
            signed_dr,
            abs_df,
            abs_dr,
            manhattan,
            chebyshev,
            same_file,
            same_rank,
            same_diag,
            same_antidiag,
            knight_vector,
            forward_relation,
            source_center,
            target_center,
        ],
        dim=-1,
    )


class TypeAwareTransportCost(nn.Module):
    def __init__(
        self,
        transport_heads: int = 4,
        transport_type_dim: int = 16,
        transport_cost_hidden: int = 64,
        cost_floor: float = 1e-4,
        transport_ablation: str = "none",
    ) -> None:
        super().__init__()
        if transport_heads < 1:
            raise ValueError("transport_heads must be positive")
        if transport_ablation not in {"none", "cost_semantic_shuffle"}:
            raise ValueError("transport_ablation must be 'none' or 'cost_semantic_shuffle'")
        self.transport_heads = transport_heads
        self.cost_floor = cost_floor
        self.transport_ablation = transport_ablation
        self.piece_embedding = nn.Embedding(7, transport_type_dim)
        self.direction_embedding = nn.Embedding(2, transport_type_dim)
        self.register_buffer("geometry", _board_geometry_features(), persistent=False)
        self.register_buffer("target_permutation", torch.arange(63, -1, -1, dtype=torch.long), persistent=False)
        feature_dim = transport_type_dim * 3 + int(self.geometry.shape[-1])
        self.cost_mlp = nn.Sequential(
            nn.Linear(feature_dim, transport_cost_hidden),
            nn.GELU(),
            nn.Linear(transport_cost_hidden, transport_cost_hidden),
            nn.GELU(),
            nn.Linear(transport_cost_hidden, transport_heads),
        )

    def _square_embeddings(self, piece_flat: torch.Tensor) -> torch.Tensor:
        piece_weights = piece_flat.transpose(1, 2).clamp(0.0, 1.0)
        piece_table = self.piece_embedding.weight[:6].to(dtype=piece_flat.dtype)
        piece_embed = torch.einsum("bsp,pe->bse", piece_weights, piece_table)
        occupancy = piece_weights.sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
        empty_embed = self.piece_embedding.weight[6].to(dtype=piece_flat.dtype).view(1, 1, -1)
        return piece_embed + (1.0 - occupancy) * empty_embed

    def forward(self, source_pieces: torch.Tensor, target_pieces: torch.Tensor, direction_id: int) -> torch.Tensor:
        source_embed = self._square_embeddings(source_pieces)
        target_embed = self._square_embeddings(target_pieces)
        if self.transport_ablation == "cost_semantic_shuffle":
            perm = self.target_permutation.to(device=target_embed.device)
            target_embed = target_embed[:, perm]
        batch = source_embed.shape[0]
        src = source_embed.unsqueeze(2).expand(batch, 64, 64, -1)
        tgt = target_embed.unsqueeze(1).expand(batch, 64, 64, -1)
        direction = self.direction_embedding.weight[direction_id].to(dtype=source_embed.dtype)
        direction = direction.view(1, 1, 1, -1).expand(batch, 64, 64, -1)
        geometry = self.geometry.to(device=source_embed.device, dtype=source_embed.dtype).unsqueeze(0).expand(batch, -1, -1, -1)
        cost_features = torch.cat([src, tgt, direction, geometry], dim=-1)
        logits = self.cost_mlp(cost_features).permute(0, 3, 1, 2)
        return F.softplus(logits).clamp_max(25.0) + self.cost_floor


class LogSinkhornTransport(nn.Module):
    def __init__(self, sinkhorn_iters: int = 16, sinkhorn_tau: float = 0.15) -> None:
        super().__init__()
        if sinkhorn_iters < 1:
            raise ValueError("sinkhorn_iters must be >= 1")
        if sinkhorn_tau <= 0:
            raise ValueError("sinkhorn_tau must be positive")
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_tau = sinkhorn_tau

    def forward(self, cost: torch.Tensor, source_mass: torch.Tensor, target_mass: torch.Tensor) -> torch.Tensor:
        source = source_mass.float().clamp_min(1e-8)
        target = target_mass.float().clamp_min(1e-8)
        source = source / source.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        log_source = source.log()
        log_target = target.log()
        log_kernel = (-cost.float() / self.sinkhorn_tau).clamp(min=-80.0, max=20.0)
        log_u = torch.zeros_like(log_source)
        log_v = torch.zeros_like(log_target)
        for _ in range(self.sinkhorn_iters):
            log_u = log_source - torch.logsumexp(log_kernel + log_v.unsqueeze(-2), dim=-1)
            log_v = log_target - torch.logsumexp(log_kernel + log_u.unsqueeze(-1), dim=-2)
        log_plan = log_u.unsqueeze(-1) + log_kernel + log_v.unsqueeze(-2)
        plan = torch.exp(log_plan)
        return plan / plan.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)


class TransportSummaryProjector(nn.Module):
    stats_per_head = 9
    map_channels_per_head = 4

    def __init__(self) -> None:
        super().__init__()
        geometry = _board_geometry_features()
        self.register_buffer("abs_df", geometry[..., 2], persistent=False)
        self.register_buffer("abs_dr", geometry[..., 3], persistent=False)
        self.register_buffer("manhattan", geometry[..., 4], persistent=False)
        same_line = ((geometry[..., 6] > 0.5) | (geometry[..., 7] > 0.5) | (geometry[..., 8] > 0.5)).float()
        self.register_buffer("same_line", same_line, persistent=False)
        self.register_buffer("knight_vector", geometry[..., 10], persistent=False)

    def forward(self, plan: torch.Tensor, cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = plan.dtype
        abs_df = self.abs_df.to(device=plan.device, dtype=dtype)
        abs_dr = self.abs_dr.to(device=plan.device, dtype=dtype)
        manhattan = self.manhattan.to(device=plan.device, dtype=dtype)
        same_line = self.same_line.to(device=plan.device, dtype=dtype)
        knight_vector = self.knight_vector.to(device=plan.device, dtype=dtype)
        expected_cost = (plan * cost.to(dtype=dtype)).sum(dim=(-2, -1))
        entropy = -(plan * plan.clamp_min(1e-12).log()).sum(dim=(-2, -1))
        entropy = entropy / torch.log(plan.new_tensor(float(64 * 64)))
        l2_concentration = plan.square().sum(dim=(-2, -1))
        mean_abs_df = (plan * abs_df.view(1, 1, 64, 64)).sum(dim=(-2, -1))
        mean_abs_dr = (plan * abs_dr.view(1, 1, 64, 64)).sum(dim=(-2, -1))
        mean_manhattan = (plan * manhattan.view(1, 1, 64, 64)).sum(dim=(-2, -1))
        same_line_mass = (plan * same_line.view(1, 1, 64, 64)).sum(dim=(-2, -1))
        knight_mass = (plan * knight_vector.view(1, 1, 64, 64)).sum(dim=(-2, -1))
        cost_center = cost.to(dtype=dtype).mean(dim=(-2, -1), keepdim=True)
        low_cost_soft = torch.sigmoid((cost_center - cost.to(dtype=dtype)) / 0.1)
        low_cost_mass = (plan * low_cost_soft).sum(dim=(-2, -1))
        stats = torch.stack(
            [
                expected_cost,
                entropy,
                l2_concentration,
                mean_abs_df,
                mean_abs_dr,
                mean_manhattan,
                same_line_mass,
                knight_mass,
                low_cost_mass,
            ],
            dim=-1,
        )

        source_cost_map = (plan * cost.to(dtype=dtype)).sum(dim=-1)
        target_cost_map = (plan * cost.to(dtype=dtype)).sum(dim=-2)
        source_conc_map = plan.square().sum(dim=-1)
        target_conc_map = plan.square().sum(dim=-2)
        maps = torch.cat(
            [source_cost_map, target_cost_map, source_conc_map, target_conc_map],
            dim=1,
        ).view(plan.shape[0], plan.shape[1] * self.map_channels_per_head, 8, 8)
        return stats, maps


class PieceTargetEntropicTransportBottleneck(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        encoding_adapter: str = SIMPLE_18,
        transport_heads: int = 4,
        transport_type_dim: int = 16,
        transport_cost_hidden: int = 64,
        sinkhorn_iters: int = 16,
        sinkhorn_tau: float = 0.15,
        epsilon_mass: float = 1e-3,
        adapter_width: int = 64,
        adapter_depth: int = 3,
        bottleneck_dim: int = 32,
        hidden_dim: int = 96,
        dropout: float = 0.05,
        beta_kl: float = 0.0,
        transport_ablation: str = "none",
        cost_floor: float = 1e-4,
        fail_closed_unknown_channels: bool = True,
    ) -> None:
        super().__init__()
        if not fail_closed_unknown_channels:
            raise ValueError("fail_closed_unknown_channels must remain true for deterministic piece semantics")
        if beta_kl != 0.0:
            raise ValueError("beta_kl is accepted only as 0.0; stochastic KL training is not implemented")
        self.num_classes = num_classes
        self.transport_heads = transport_heads
        self.beta_kl = beta_kl
        self.piece_adapter = PiecePlaneAdapter(input_channels=input_channels, encoding_adapter=encoding_adapter)
        self.canonicalizer = RelativeBoardCanonicalizer()
        self.board_adapter = BoardAdapter(
            input_channels=input_channels,
            adapter_width=adapter_width,
            adapter_depth=adapter_depth,
        )
        self.marginals = TransportMarginals(transport_heads=transport_heads, epsilon_mass=epsilon_mass)
        self.cost = TypeAwareTransportCost(
            transport_heads=transport_heads,
            transport_type_dim=transport_type_dim,
            transport_cost_hidden=transport_cost_hidden,
            cost_floor=cost_floor,
            transport_ablation=transport_ablation,
        )
        self.sinkhorn = LogSinkhornTransport(sinkhorn_iters=sinkhorn_iters, sinkhorn_tau=sinkhorn_tau)
        self.projector = TransportSummaryProjector()
        map_channels = 2 * transport_heads * self.projector.map_channels_per_head
        stats_dim = 2 * transport_heads * self.projector.stats_per_head
        self.fusion = nn.Sequential(
            nn.Conv2d(adapter_width + map_channels, adapter_width, kernel_size=1, bias=False),
            nn.BatchNorm2d(adapter_width),
            nn.GELU(),
            nn.Conv2d(adapter_width, adapter_width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(adapter_width),
            nn.GELU(),
        )
        self.stats_norm = nn.LayerNorm(stats_dim)
        self.bottleneck = nn.Sequential(
            nn.Linear(adapter_width * 2 + stats_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(bottleneck_dim, num_classes)

    def _transport_descriptor(
        self,
        source_pieces: torch.Tensor,
        target_pieces: torch.Tensor,
        source_mass: torch.Tensor,
        target_mass: torch.Tensor,
        direction_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cost = self.cost(source_pieces, target_pieces, direction_id=direction_id)
        plan = self.sinkhorn(cost, source_mass, target_mass)
        stats, maps = self.projector(plan, cost)
        return stats, maps, cost, plan

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board_state = self.piece_adapter(x)
        canonical = self.canonicalizer(board_state)
        board_features = self.board_adapter(x)
        mu_fe, nu_fe, mu_ef, nu_ef = self.marginals(canonical)
        forward_stats, forward_maps, forward_cost, forward_plan = self._transport_descriptor(
            canonical.friendly_flat,
            canonical.enemy_flat,
            mu_fe,
            nu_fe,
            direction_id=0,
        )
        reverse_stats, reverse_maps, reverse_cost, reverse_plan = self._transport_descriptor(
            canonical.enemy_flat,
            canonical.friendly_flat,
            mu_ef,
            nu_ef,
            direction_id=1,
        )
        transport_maps = torch.cat([forward_maps, reverse_maps], dim=1).to(dtype=board_features.dtype)
        fused = self.fusion(torch.cat([board_features, transport_maps], dim=1))
        pooled = torch.cat([fused.mean(dim=(2, 3)), fused.amax(dim=(2, 3))], dim=1)
        transport_stats = torch.cat([forward_stats.flatten(1), reverse_stats.flatten(1)], dim=1).to(dtype=pooled.dtype)
        z = self.bottleneck(torch.cat([pooled, self.stats_norm(transport_stats)], dim=1))
        logits = self.classifier(z)
        if self.num_classes == 1:
            logits = logits.squeeze(-1)
        forward_cost_mean = (forward_plan * forward_cost.float()).sum(dim=(-2, -1)).mean(dim=1)
        reverse_cost_mean = (reverse_plan * reverse_cost.float()).sum(dim=(-2, -1)).mean(dim=1)
        return {
            "logits": logits,
            "transport_cost_forward": forward_cost_mean,
            "transport_cost_reverse": reverse_cost_mean,
            "transport_entropy_forward": forward_stats[..., 1].mean(dim=1),
            "transport_entropy_reverse": reverse_stats[..., 1].mean(dim=1),
            "transport_asymmetry": reverse_cost_mean - forward_cost_mean,
            "transport_low_cost_mass": forward_stats[..., 8].mean(dim=1),
            "transport_bottleneck_norm": z.norm(dim=1) / (z.shape[1] ** 0.5),
        }


def build_piece_target_entropic_transport_bottleneck_from_config(
    config: dict[str, Any],
) -> PieceTargetEntropicTransportBottleneck:
    return PieceTargetEntropicTransportBottleneck(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        encoding_adapter=str(config.get("encoding_adapter", config.get("encoding", SIMPLE_18))),
        transport_heads=int(config.get("transport_heads", 4)),
        transport_type_dim=int(config.get("transport_type_dim", 16)),
        transport_cost_hidden=int(config.get("transport_cost_hidden", 64)),
        sinkhorn_iters=int(config.get("sinkhorn_iters", 16)),
        sinkhorn_tau=float(config.get("sinkhorn_tau", config.get("sinkhorn_epsilon", 0.15))),
        epsilon_mass=float(config.get("epsilon_mass", config.get("mass_floor", 1e-3))),
        adapter_width=int(config.get("adapter_width", config.get("channels", 64))),
        adapter_depth=int(config.get("adapter_depth", config.get("depth", 3))),
        bottleneck_dim=int(config.get("bottleneck_dim", 32)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        dropout=float(config.get("dropout", 0.05)),
        beta_kl=float(config.get("beta_kl", 0.0)),
        transport_ablation=str(config.get("transport_ablation", "none")),
        cost_floor=float(config.get("cost_floor", 1e-4)),
        fail_closed_unknown_channels=bool(config.get("fail_closed_unknown_channels", True)),
    )


build_piece_target_transport_bottleneck = build_piece_target_entropic_transport_bottleneck_from_config

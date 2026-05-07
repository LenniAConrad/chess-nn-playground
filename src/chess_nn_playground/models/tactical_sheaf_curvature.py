"""Tactical Sheaf Curvature Network for idea i019."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


@dataclass(frozen=True)
class RelationComplexSpec:
    include_ray_edges: bool = True
    include_knight_edges: bool = True
    include_king_edges: bool = True
    include_pawn_candidate_edges: bool = True
    tie_file_mirror_relations: bool = True
    tie_north_south_relations: bool = False


def _idx(rank: int, file: int) -> int:
    return rank * 8 + file


def _inside(rank: int, file: int) -> bool:
    return 0 <= rank < 8 and 0 <= file < 8


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


def _distance_bucket(distance: int) -> int:
    return min(max(int(distance), 1), 4) - 1


def _std_pool(x: torch.Tensor, dim: int) -> torch.Tensor:
    return x.var(dim=dim, unbiased=False).clamp_min(0.0).sqrt()


class BoardChannelAdapter(nn.Module):
    def __init__(self, input_channels: int, d_node: int, depth: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers: list[nn.Module] = [
            nn.Conv2d(input_channels, d_node, kernel_size=1),
            nn.GELU(),
        ]
        for _ in range(depth - 1):
            layers.extend(
                [
                    nn.Conv2d(d_node, d_node, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_grid = self.net(x)
        return h_grid.flatten(2).transpose(1, 2)


class TypedRelationComplex(nn.Module):
    def __init__(self, spec: RelationComplexSpec | None = None) -> None:
        super().__init__()
        spec = spec or RelationComplexSpec()
        src: list[int] = []
        dst: list[int] = []
        edge_type: list[int] = []
        group: list[int] = []
        geom: list[list[float]] = []
        relation_type: dict[tuple[str, int], int] = {}
        group_names = {
            "north_ray": 0,
            "south_ray": 1,
            "horizontal_ray": 2,
            "north_diag": 3,
            "south_diag": 4,
            "knight": 5,
            "king": 6,
            "pawn_up": 7,
            "pawn_down": 8,
        }

        def type_id(name: str, bucket: int = 0) -> int:
            key = (name, bucket)
            if key not in relation_type:
                relation_type[key] = len(relation_type)
            return relation_type[key]

        def add_edge(s: int, d: int, name: str, group_name: str, distance: int, bucket: int) -> None:
            sr, sf = divmod(s, 8)
            dr, df = divmod(d, 8)
            src.append(s)
            dst.append(d)
            edge_type.append(type_id(name, bucket))
            group.append(group_names[group_name])
            geom.append(
                [
                    sr / 7.0,
                    sf / 7.0,
                    dr / 7.0,
                    df / 7.0,
                    (dr - sr) / 7.0,
                    (df - sf) / 7.0,
                    float(distance) / 7.0,
                    float(bucket) / 3.0,
                ]
            )

        if spec.include_ray_edges:
            ray_dirs = [
                (-1, 0, "north_ray", "north_ray"),
                (1, 0, "south_ray", "south_ray"),
                (0, -1, "horizontal_ray", "horizontal_ray"),
                (0, 1, "horizontal_ray" if spec.tie_file_mirror_relations else "east_ray", "horizontal_ray"),
                (-1, -1, "north_diag", "north_diag"),
                (-1, 1, "north_diag" if spec.tie_file_mirror_relations else "northeast_diag", "north_diag"),
                (1, -1, "south_diag", "south_diag"),
                (1, 1, "south_diag" if spec.tie_file_mirror_relations else "southeast_diag", "south_diag"),
            ]
            for rank in range(8):
                for file in range(8):
                    s = _idx(rank, file)
                    for d_rank, d_file, rel_name, group_name in ray_dirs:
                        for distance in range(1, 8):
                            rr = rank + d_rank * distance
                            ff = file + d_file * distance
                            if not _inside(rr, ff):
                                break
                            add_edge(s, _idx(rr, ff), rel_name, group_name, distance, _distance_bucket(distance))

        if spec.include_knight_edges:
            knight_dirs = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
            for rank in range(8):
                for file in range(8):
                    s = _idx(rank, file)
                    for d_rank, d_file in knight_dirs:
                        rr = rank + d_rank
                        ff = file + d_file
                        if _inside(rr, ff):
                            name = "knight_north" if d_rank < 0 else "knight_south"
                            if spec.tie_north_south_relations:
                                name = "knight"
                            add_edge(s, _idx(rr, ff), name, "knight", 2, 0)

        if spec.include_king_edges:
            for rank in range(8):
                for file in range(8):
                    s = _idx(rank, file)
                    for d_rank in (-1, 0, 1):
                        for d_file in (-1, 0, 1):
                            if d_rank == 0 and d_file == 0:
                                continue
                            rr = rank + d_rank
                            ff = file + d_file
                            if _inside(rr, ff):
                                add_edge(s, _idx(rr, ff), "king_neighborhood", "king", 1, 0)

        if spec.include_pawn_candidate_edges:
            for rank in range(8):
                for file in range(8):
                    s = _idx(rank, file)
                    for d_rank, name, group_name in ((-1, "pawn_up_capture", "pawn_up"), (1, "pawn_down_capture", "pawn_down")):
                        for d_file in (-1, 1):
                            rr = rank + d_rank
                            ff = file + d_file
                            if _inside(rr, ff):
                                rel_name = name if not spec.tie_north_south_relations else "pawn_capture"
                                add_edge(s, _idx(rr, ff), rel_name, group_name, 1, 0)

        self.group_count = len(group_names)
        self.type_count = len(relation_type)
        self.raw_geom_dim = 8
        self.register_buffer("edge_index", torch.tensor([src, dst], dtype=torch.long), persistent=False)
        self.register_buffer("edge_type", torch.tensor(edge_type, dtype=torch.long), persistent=False)
        self.register_buffer("relation_group", torch.tensor(group, dtype=torch.long), persistent=False)
        self.register_buffer("edge_geom", torch.tensor(geom, dtype=torch.float32), persistent=False)


class SheafRestrictionGenerator(nn.Module):
    def __init__(self, type_count: int, raw_geom_dim: int, d_geom: int, d_stalk: int) -> None:
        super().__init__()
        self.type_embedding = nn.Embedding(type_count, d_geom)
        self.geom_projection = nn.Sequential(
            nn.Linear(raw_geom_dim, d_geom),
            nn.LayerNorm(d_geom),
            nn.GELU(),
        )
        self.restriction = nn.Sequential(
            nn.LayerNorm(d_geom),
            nn.Linear(d_geom, d_geom),
            nn.GELU(),
            nn.Linear(d_geom, d_stalk * 2),
        )

    def forward(self, edge_type: torch.Tensor, edge_geom: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.type_embedding(edge_type) + self.geom_projection(edge_geom)
        raw = self.restriction(q)
        a, b = raw.chunk(2, dim=-1)
        return 1.0 + 0.5 * torch.tanh(a), 1.0 + 0.5 * torch.tanh(b), q


class SheafGate(nn.Module):
    def __init__(self, d_stalk: int, d_geom: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_stalk * 2 + d_geom, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        q_batch = q.unsqueeze(0).expand(x_src.shape[0], -1, -1)
        return torch.sigmoid(self.net(torch.cat([x_src, x_dst, q_batch], dim=-1))).squeeze(-1)


class TacticalSheafLayer(nn.Module):
    def __init__(
        self,
        type_count: int,
        raw_geom_dim: int,
        d_geom: int,
        d_stalk: int,
        gate_hidden: int,
        gate_dropout: float,
        node_dropout: float,
        eta_init: float,
        group_count: int,
    ) -> None:
        super().__init__()
        self.restrictions = SheafRestrictionGenerator(type_count, raw_geom_dim, d_geom, d_stalk)
        self.gate = SheafGate(d_stalk, d_geom, gate_hidden, gate_dropout)
        self.node_mlp = nn.Sequential(
            nn.LayerNorm(d_stalk),
            nn.Linear(d_stalk, d_stalk * 2),
            nn.GELU(),
            nn.Dropout(node_dropout) if node_dropout > 0 else nn.Identity(),
            nn.Linear(d_stalk * 2, d_stalk),
        )
        self.norm = nn.LayerNorm(d_stalk)
        eta = min(max(float(eta_init), 1.0e-4), 0.95)
        self.eta_logit = nn.Parameter(torch.logit(torch.tensor(eta, dtype=torch.float32)))
        self.group_count = int(group_count)

    def _scatter_nodes(self, values: torch.Tensor, index: torch.Tensor, node_count: int) -> torch.Tensor:
        out = values.new_zeros(values.shape[0], node_count, values.shape[-1])
        expanded = index.view(1, -1, 1).expand(values.shape[0], -1, values.shape[-1])
        return out.scatter_add(1, expanded, values)

    def _scatter_scalar(self, values: torch.Tensor, index: torch.Tensor, node_count: int) -> torch.Tensor:
        out = values.new_zeros(values.shape[0], node_count)
        expanded = index.view(1, -1).expand(values.shape[0], -1)
        return out.scatter_add(1, expanded, values)

    def _group_mean(self, values: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
        batch = values.shape[0]
        out = values.new_zeros(batch, self.group_count)
        counts = values.new_zeros(batch, self.group_count)
        group_index = group.view(1, -1).expand(batch, -1)
        out = out.scatter_add(1, group_index, values)
        counts = counts.scatter_add(1, group_index, torch.ones_like(values))
        return out / counts.clamp_min(1.0)

    def _incoming_curvature(
        self,
        claim: torch.Tensor,
        gate: torch.Tensor,
        dst: torch.Tensor,
        node_count: int,
    ) -> torch.Tensor:
        weighted_claim = gate.unsqueeze(-1) * claim
        weight_sum = self._scatter_scalar(gate, dst, node_count)
        claim_sum = self._scatter_nodes(weighted_claim, dst, node_count)
        mean = claim_sum / weight_sum.unsqueeze(-1).clamp_min(1.0e-6)
        mean_edge = mean.index_select(1, dst)
        variance_edge = gate * (claim - mean_edge).square().mean(dim=-1)
        variance_sum = self._scatter_scalar(variance_edge, dst, node_count)
        return variance_sum / weight_sum.clamp_min(1.0e-6)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_geom: torch.Tensor,
        relation_group: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        x_src = x.index_select(1, src)
        x_dst = x.index_select(1, dst)
        a, b, q = self.restrictions(edge_type, edge_geom.to(dtype=x.dtype))
        gate = self.gate(x_src, x_dst, q.to(dtype=x.dtype))
        delta = b.unsqueeze(0) * x_dst - a.unsqueeze(0) * x_src
        edge_energy = gate * delta.square().sum(dim=-1)
        transported_claim = a.unsqueeze(0) * x_src
        dst_update = self._scatter_nodes(b.unsqueeze(0) * gate.unsqueeze(-1) * delta, dst, x.shape[1])
        src_update = self._scatter_nodes(a.unsqueeze(0) * gate.unsqueeze(-1) * delta, src, x.shape[1])
        degree = self._scatter_scalar(gate, src, x.shape[1]) + self._scatter_scalar(gate, dst, x.shape[1])
        lap_update = (dst_update - src_update) / degree.unsqueeze(-1).clamp_min(1.0)
        eta = torch.sigmoid(self.eta_logit)
        x_next = self.norm(x - eta * lap_update + self.node_mlp(x))
        curvature = self._incoming_curvature(transported_claim, gate, dst, x.shape[1])
        gate_entropy = -(gate * gate.clamp_min(1.0e-6).log() + (1.0 - gate) * (1.0 - gate).clamp_min(1.0e-6).log())
        stats = torch.cat(
            [
                edge_energy.mean(dim=1, keepdim=True),
                _std_pool(edge_energy, dim=1).unsqueeze(1),
                edge_energy.amax(dim=1, keepdim=True),
                gate.mean(dim=1, keepdim=True),
                gate_entropy.mean(dim=1, keepdim=True),
                curvature.mean(dim=1, keepdim=True),
                _std_pool(curvature, dim=1).unsqueeze(1),
                curvature.amax(dim=1, keepdim=True),
                self._group_mean(edge_energy, relation_group),
                self._group_mean(gate, relation_group),
            ],
            dim=1,
        )
        return x_next, stats


class CurvatureStatsPool(nn.Module):
    def forward(self, x: torch.Tensor, layer_stats: list[torch.Tensor]) -> torch.Tensor:
        node_pool = torch.cat([x.mean(dim=1), x.amax(dim=1), _std_pool(x, dim=1)], dim=1)
        return torch.cat([node_pool, *layer_stats], dim=1)


class TacticalSheafCurvatureNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        d_node: int = 64,
        d_stalk: int = 48,
        d_geom: int = 24,
        num_layers: int = 3,
        gate_hidden: int = 64,
        classifier_hidden: int = 128,
        adapter_depth: int = 2,
        gate_dropout: float = 0.05,
        node_dropout: float = 0.1,
        eta_init: float = 0.25,
        include_ray_edges: bool = True,
        include_knight_edges: bool = True,
        include_king_edges: bool = True,
        include_pawn_candidate_edges: bool = True,
        tie_file_mirror_relations: bool = True,
        tie_north_south_relations: bool = False,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.adapter = BoardChannelAdapter(input_channels, d_node, depth=adapter_depth, dropout=node_dropout)
        self.stalk_projection = nn.Sequential(
            nn.LayerNorm(d_node),
            nn.Linear(d_node, d_stalk),
        )
        self.relation_complex = TypedRelationComplex(
            RelationComplexSpec(
                include_ray_edges=include_ray_edges,
                include_knight_edges=include_knight_edges,
                include_king_edges=include_king_edges,
                include_pawn_candidate_edges=include_pawn_candidate_edges,
                tie_file_mirror_relations=tie_file_mirror_relations,
                tie_north_south_relations=tie_north_south_relations,
            )
        )
        self.layers = nn.ModuleList(
            [
                TacticalSheafLayer(
                    type_count=self.relation_complex.type_count,
                    raw_geom_dim=self.relation_complex.raw_geom_dim,
                    d_geom=d_geom,
                    d_stalk=d_stalk,
                    gate_hidden=gate_hidden,
                    gate_dropout=gate_dropout,
                    node_dropout=node_dropout,
                    eta_init=eta_init,
                    group_count=self.relation_complex.group_count,
                )
                for _ in range(max(1, int(num_layers)))
            ]
        )
        self.stats_pool = CurvatureStatsPool()
        per_layer_stats = 8 + 2 * self.relation_complex.group_count
        readout_dim = 3 * d_stalk + len(self.layers) * per_layer_stats
        self.classifier = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, classifier_hidden),
            nn.GELU(),
            nn.Dropout(node_dropout) if node_dropout > 0 else nn.Identity(),
            nn.Linear(classifier_hidden, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        h = self.adapter(x)
        stalk = self.stalk_projection(h)
        layer_stats: list[torch.Tensor] = []
        for layer in self.layers:
            stalk, stats = layer(
                stalk,
                self.relation_complex.edge_index,
                self.relation_complex.edge_type,
                self.relation_complex.edge_geom,
                self.relation_complex.relation_group,
            )
            layer_stats.append(stats)
        pooled = self.stats_pool(stalk, layer_stats)
        logits = _format_logits(self.classifier(pooled), self.num_classes)
        stats_stack = torch.stack(layer_stats, dim=1)
        group_energy = stats_stack[:, :, 8 : 8 + self.relation_complex.group_count]
        group_gate = stats_stack[:, :, 8 + self.relation_complex.group_count :]
        diagnostics = {
            "logits": logits,
            "sheaf_frustration": stats_stack[:, :, 0].mean(dim=1),
            "curvature_mean": stats_stack[:, :, 5].mean(dim=1),
            "curvature_max": stats_stack[:, :, 7].amax(dim=1),
            "gate_mean": stats_stack[:, :, 3].mean(dim=1),
            "gate_entropy": stats_stack[:, :, 4].mean(dim=1),
            "ray_energy": group_energy[:, :, :5].mean(dim=(1, 2)),
            "jump_energy": group_energy[:, :, 5:7].mean(dim=(1, 2)),
            "pawn_candidate_energy": group_energy[:, :, 7:9].mean(dim=(1, 2)),
            "relation_gate_pressure": group_gate.mean(dim=(1, 2)),
            "node_stalk_std": _std_pool(stalk, dim=1).mean(dim=1),
        }
        return diagnostics


def build_tactical_sheaf_curvature_from_config(config: dict[str, Any]) -> TacticalSheafCurvatureNet:
    d_node = int(config.get("d_node", config.get("channels", 64)))
    return TacticalSheafCurvatureNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        d_node=d_node,
        d_stalk=int(config.get("d_stalk", 48)),
        d_geom=int(config.get("d_geom", 24)),
        num_layers=int(config.get("num_layers", config.get("sheaf_layers", config.get("depth", 3)))),
        gate_hidden=int(config.get("gate_hidden", config.get("hidden_dim", 64))),
        classifier_hidden=int(config.get("classifier_hidden", config.get("hidden_dim", 128))),
        adapter_depth=int(config.get("adapter_depth", config.get("depth", 2))),
        gate_dropout=float(config.get("gate_dropout", config.get("dropout", 0.05))),
        node_dropout=float(config.get("node_dropout", config.get("dropout", 0.1))),
        eta_init=float(config.get("eta_init", 0.25)),
        include_ray_edges=bool(config.get("include_ray_edges", True)),
        include_knight_edges=bool(config.get("include_knight_edges", True)),
        include_king_edges=bool(config.get("include_king_edges", True)),
        include_pawn_candidate_edges=bool(config.get("include_pawn_candidate_edges", True)),
        tie_file_mirror_relations=bool(config.get("tie_file_mirror_relations", True)),
        tie_north_south_relations=bool(config.get("tie_north_south_relations", False)),
    )

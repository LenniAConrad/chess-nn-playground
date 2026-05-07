"""Attack-Defense Sheaf Energy Network for idea i020."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


GROUP_NAMES: tuple[str, ...] = ("ray", "knight", "king", "pawn_up", "pawn_down")


@dataclass(frozen=True)
class AttackDefenseIncidenceSpec:
    max_ray_length: int = 7
    tie_file_reflection: bool = True
    include_quiet_pawn_pressure: bool = False


def _idx(rank: int, file: int) -> int:
    return rank * 8 + file


def _inside(rank: int, file: int) -> bool:
    return 0 <= rank < 8 and 0 <= file < 8


def _distance_bucket(distance: int) -> int:
    return min(max(int(distance), 1), 4) - 1


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


def _std_pool(x: torch.Tensor, dim: int) -> torch.Tensor:
    return x.var(dim=dim, unbiased=False).clamp_min(0.0).sqrt()


def _square_coordinates() -> torch.Tensor:
    square = torch.arange(64, dtype=torch.float32)
    rank = torch.div(square, 8, rounding_mode="floor")
    file = square.remainder(8)
    center_rank = (rank - 3.5) / 3.5
    center_file = (file - 3.5) / 3.5
    edge_distance = torch.minimum(torch.minimum(rank, 7.0 - rank), torch.minimum(file, 7.0 - file)) / 3.5
    color = ((rank + file).remainder(2.0) * 2.0) - 1.0
    return torch.stack([rank / 7.0, file / 7.0, center_rank, center_file, edge_distance, color], dim=1)


class AttackDefenseIncidence(nn.Module):
    def __init__(self, spec: AttackDefenseIncidenceSpec | None = None) -> None:
        super().__init__()
        spec = spec or AttackDefenseIncidenceSpec()
        max_blockers = max(1, int(spec.max_ray_length) - 1)
        src: list[int] = []
        dst: list[int] = []
        edge_type: list[int] = []
        group: list[int] = []
        is_ray: list[bool] = []
        blockers: list[list[int]] = []
        blocker_mask: list[list[bool]] = []
        edge_geom: list[list[float]] = []
        type_ids: dict[tuple[str, int], int] = {}
        group_ids = {name: index for index, name in enumerate(GROUP_NAMES)}

        def type_id(name: str, bucket: int = 0) -> int:
            key = (name, bucket)
            if key not in type_ids:
                type_ids[key] = len(type_ids)
            return type_ids[key]

        def add_edge(
            source: int,
            target: int,
            relation_name: str,
            group_name: str,
            distance: int,
            ray_edge: bool,
            blocker_squares: list[int] | None = None,
            bucket: int = 0,
        ) -> None:
            blocker_squares = list(blocker_squares or [])[:max_blockers]
            padded = blocker_squares + [-1] * (max_blockers - len(blocker_squares))
            mask = [True] * len(blocker_squares) + [False] * (max_blockers - len(blocker_squares))
            sr, sf = divmod(source, 8)
            tr, tf = divmod(target, 8)
            src.append(source)
            dst.append(target)
            edge_type.append(type_id(relation_name, bucket))
            group.append(group_ids[group_name])
            is_ray.append(ray_edge)
            blockers.append(padded)
            blocker_mask.append(mask)
            edge_geom.append(
                [
                    sr / 7.0,
                    sf / 7.0,
                    tr / 7.0,
                    tf / 7.0,
                    (tr - sr) / 7.0,
                    (tf - sf) / 7.0,
                    float(distance) / 7.0,
                    1.0 if ray_edge else 0.0,
                ]
            )

        ray_dirs = [
            (-1, 0, "north_ray"),
            (1, 0, "south_ray"),
            (0, -1, "horizontal_ray"),
            (0, 1, "horizontal_ray" if spec.tie_file_reflection else "east_ray"),
            (-1, -1, "north_diag"),
            (-1, 1, "north_diag" if spec.tie_file_reflection else "northeast_diag"),
            (1, -1, "south_diag"),
            (1, 1, "south_diag" if spec.tie_file_reflection else "southeast_diag"),
        ]
        for rank in range(8):
            for file in range(8):
                source = _idx(rank, file)
                for d_rank, d_file, relation_name in ray_dirs:
                    blocker_squares: list[int] = []
                    for distance in range(1, int(spec.max_ray_length) + 1):
                        rr = rank + d_rank * distance
                        ff = file + d_file * distance
                        if not _inside(rr, ff):
                            break
                        target = _idx(rr, ff)
                        add_edge(
                            source,
                            target,
                            relation_name,
                            "ray",
                            distance,
                            True,
                            blocker_squares,
                            _distance_bucket(distance),
                        )
                        blocker_squares = [*blocker_squares, target]

        knight_dirs = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        for rank in range(8):
            for file in range(8):
                source = _idx(rank, file)
                for d_rank, d_file in knight_dirs:
                    rr = rank + d_rank
                    ff = file + d_file
                    if _inside(rr, ff):
                        lateral = "wide" if abs(d_file) == 2 else "tall"
                        vertical = "north" if d_rank < 0 else "south"
                        add_edge(source, _idx(rr, ff), f"knight_{vertical}_{lateral}", "knight", 2, False)

        for rank in range(8):
            for file in range(8):
                source = _idx(rank, file)
                for d_rank in (-1, 0, 1):
                    for d_file in (-1, 0, 1):
                        if d_rank == 0 and d_file == 0:
                            continue
                        rr = rank + d_rank
                        ff = file + d_file
                        if _inside(rr, ff):
                            if d_rank == 0:
                                relation_name = "king_horizontal"
                            elif d_file == 0:
                                relation_name = "king_vertical"
                            else:
                                relation_name = "king_diagonal"
                            add_edge(source, _idx(rr, ff), relation_name, "king", 1, False)

        pawn_specs = [(-1, "pawn_up_attack", "pawn_up"), (1, "pawn_down_attack", "pawn_down")]
        if spec.include_quiet_pawn_pressure:
            pawn_specs.extend([(-1, "pawn_up_push_pressure", "pawn_up"), (1, "pawn_down_push_pressure", "pawn_down")])
        for rank in range(8):
            for file in range(8):
                source = _idx(rank, file)
                for d_rank, relation_name, group_name in pawn_specs:
                    target_files = (-1, 1) if "attack" in relation_name else (0,)
                    for d_file in target_files:
                        rr = rank + d_rank
                        ff = file + d_file
                        if _inside(rr, ff):
                            add_edge(source, _idx(rr, ff), relation_name, group_name, 1, False)

        self.type_count = len(type_ids)
        self.group_count = len(GROUP_NAMES)
        self.raw_geom_dim = 8
        self.max_blockers = max_blockers
        self.register_buffer("edge_src", torch.tensor(src, dtype=torch.long), persistent=False)
        self.register_buffer("edge_dst", torch.tensor(dst, dtype=torch.long), persistent=False)
        self.register_buffer("edge_type", torch.tensor(edge_type, dtype=torch.long), persistent=False)
        self.register_buffer("relation_group", torch.tensor(group, dtype=torch.long), persistent=False)
        self.register_buffer("edge_is_ray", torch.tensor(is_ray, dtype=torch.bool), persistent=False)
        self.register_buffer("blocker_index", torch.tensor(blockers, dtype=torch.long), persistent=False)
        self.register_buffer("blocker_mask", torch.tensor(blocker_mask, dtype=torch.bool), persistent=False)
        self.register_buffer("edge_geom", torch.tensor(edge_geom, dtype=torch.float32), persistent=False)
        self.register_buffer("square_coords", _square_coordinates(), persistent=False)


class SquareAdapter(nn.Module):
    def __init__(self, input_channels: int, d_model: int, coord_dim: int = 6, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = max(int(d_model), int(input_channels) + int(coord_dim))
        self.net = nn.Sequential(
            nn.LayerNorm(input_channels + coord_dim),
            nn.Linear(input_channels + coord_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, board: torch.Tensor, square_coords: torch.Tensor) -> torch.Tensor:
        square_raw = board.flatten(2).transpose(1, 2)
        coords = square_coords.to(device=board.device, dtype=board.dtype).unsqueeze(0).expand(board.shape[0], -1, -1)
        return self.net(torch.cat([square_raw, coords], dim=-1))


class EdgeGate(nn.Module):
    def __init__(self, d_model: int, type_emb_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(2 * d_model + type_emb_dim + 1),
            nn.Linear(2 * d_model + type_emb_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h_src: torch.Tensor,
        h_dst: torch.Tensor,
        type_features: torch.Tensor,
        ray_visibility: torch.Tensor,
    ) -> torch.Tensor:
        type_batch = type_features.unsqueeze(0).expand(h_src.shape[0], -1, -1)
        gate_input = torch.cat([h_src, h_dst, type_batch, ray_visibility.unsqueeze(-1)], dim=-1)
        return torch.sigmoid(self.net(gate_input)).squeeze(-1)


class SheafDiffusionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        type_count: int,
        type_emb_dim: int,
        sheaf_rank: int,
        hidden_dim: int,
        dropout: float,
        edge_dropout: float,
        eta_init: float,
    ) -> None:
        super().__init__()
        self.type_embedding = nn.Embedding(type_count, type_emb_dim)
        self.gate = EdgeGate(d_model, type_emb_dim, hidden_dim, dropout)
        scale = float(d_model) ** -0.5
        self.r_src = nn.Parameter(torch.randn(type_count, sheaf_rank, d_model) * scale)
        self.r_dst = nn.Parameter(torch.randn(type_count, sheaf_rank, d_model) * scale)
        self.node_ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        eta = min(max(float(eta_init), 1.0e-4), 0.95)
        self.eta_logit = nn.Parameter(torch.logit(torch.tensor(eta, dtype=torch.float32)))
        self.edge_dropout = float(edge_dropout)

    def _scatter_nodes(self, values: torch.Tensor, index: torch.Tensor, node_count: int) -> torch.Tensor:
        out = values.new_zeros(values.shape[0], node_count, values.shape[-1])
        expanded = index.view(1, -1, 1).expand(values.shape[0], -1, values.shape[-1])
        return out.scatter_add(1, expanded, values)

    def _scatter_scalar(self, values: torch.Tensor, index: torch.Tensor, node_count: int) -> torch.Tensor:
        out = values.new_zeros(values.shape[0], node_count)
        expanded = index.view(1, -1).expand(values.shape[0], -1)
        return out.scatter_add(1, expanded, values)

    def forward(
        self,
        h: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_type: torch.Tensor,
        ray_visibility: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h_src = h.index_select(1, edge_src)
        h_dst = h.index_select(1, edge_dst)
        type_features = self.type_embedding(edge_type).to(dtype=h.dtype)
        gate = self.gate(h_src, h_dst, type_features, ray_visibility.to(dtype=h.dtype))
        if self.training and self.edge_dropout > 0:
            keep_prob = 1.0 - self.edge_dropout
            keep = torch.empty_like(gate).bernoulli_(keep_prob) / keep_prob
            gate = gate * keep

        r_src = self.r_src.index_select(0, edge_type).to(dtype=h.dtype)
        r_dst = self.r_dst.index_select(0, edge_type).to(dtype=h.dtype)
        src_claim = torch.einsum("bed,erd->ber", h_src, r_src)
        dst_claim = torch.einsum("bed,erd->ber", h_dst, r_dst)
        residual = src_claim - dst_claim
        weighted_residual = gate.unsqueeze(-1) * residual
        src_msg = torch.einsum("ber,erd->bed", weighted_residual, r_src)
        dst_msg = torch.einsum("ber,erd->bed", weighted_residual, r_dst)
        node_delta = self._scatter_nodes(src_msg, edge_src, h.shape[1]) - self._scatter_nodes(dst_msg, edge_dst, h.shape[1])
        degree = self._scatter_scalar(gate, edge_src, h.shape[1]) + self._scatter_scalar(gate, edge_dst, h.shape[1])
        node_delta = node_delta / degree.unsqueeze(-1).clamp_min(1.0)
        edge_energy = gate * residual.square().sum(dim=-1)
        eta = torch.sigmoid(self.eta_logit)
        h_next = self.norm(h - eta * node_delta + self.node_ffn(h))
        return h_next, edge_energy, gate, residual


class AttackDefenseSheafNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        d_model: int = 64,
        sheaf_rank: int = 16,
        num_blocks: int = 2,
        type_emb_dim: int = 16,
        hidden_dim: int = 96,
        classifier_hidden: int = 128,
        dropout: float = 0.1,
        edge_dropout: float = 0.05,
        eta_init: float = 0.2,
        max_ray_length: int = 7,
        tie_file_reflection: bool = True,
        include_quiet_pawn_pressure: bool = False,
        topk_edges: int = 32,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.incidence = AttackDefenseIncidence(
            AttackDefenseIncidenceSpec(
                max_ray_length=max_ray_length,
                tie_file_reflection=tie_file_reflection,
                include_quiet_pawn_pressure=include_quiet_pawn_pressure,
            )
        )
        self.adapter = SquareAdapter(input_channels, d_model, dropout=dropout)
        self.occupancy_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.blocks = nn.ModuleList(
            [
                SheafDiffusionBlock(
                    d_model=d_model,
                    type_count=self.incidence.type_count,
                    type_emb_dim=type_emb_dim,
                    sheaf_rank=sheaf_rank,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    edge_dropout=edge_dropout,
                    eta_init=eta_init,
                )
                for _ in range(max(1, int(num_blocks)))
            ]
        )
        self.topk_edges = max(1, int(topk_edges))
        self.group_count = self.incidence.group_count
        per_block_stats = 7 + self.group_count
        convergence_dim = 10
        readout_dim = 3 * d_model + len(self.blocks) * per_block_stats + convergence_dim + 4
        self.classifier = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(classifier_hidden, self.num_classes),
        )

    def _ray_visibility(self, occupancy: torch.Tensor) -> torch.Tensor:
        safe_index = self.incidence.blocker_index.clamp_min(0)
        flat_index = safe_index.reshape(-1)
        blocker_values = occupancy.index_select(1, flat_index).view(
            occupancy.shape[0], self.incidence.blocker_index.shape[0], self.incidence.blocker_index.shape[1]
        )
        mask = self.incidence.blocker_mask.to(device=occupancy.device).unsqueeze(0)
        clear_terms = torch.where(mask, 1.0 - blocker_values.clamp(0.0, 1.0) + 1.0e-4, torch.ones_like(blocker_values))
        visibility = clear_terms.prod(dim=-1).clamp(0.0, 1.0)
        ray_mask = self.incidence.edge_is_ray.to(device=occupancy.device).view(1, -1)
        return torch.where(ray_mask, visibility, torch.ones_like(visibility))

    def _scatter_scalar(self, values: torch.Tensor, index: torch.Tensor, node_count: int) -> torch.Tensor:
        out = values.new_zeros(values.shape[0], node_count)
        expanded = index.view(1, -1).expand(values.shape[0], -1)
        return out.scatter_add(1, expanded, values)

    def _group_mean(self, values: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
        out = values.new_zeros(values.shape[0], self.group_count)
        counts = values.new_zeros(values.shape[0], self.group_count)
        expanded_group = group.view(1, -1).expand(values.shape[0], -1)
        out = out.scatter_add(1, expanded_group, values)
        counts = counts.scatter_add(1, expanded_group, torch.ones_like(values))
        return out / counts.clamp_min(1.0)

    def _block_stats(
        self,
        edge_energy: torch.Tensor,
        gate: torch.Tensor,
        ray_visibility: torch.Tensor,
    ) -> torch.Tensor:
        k = min(self.topk_edges, edge_energy.shape[1])
        group_energy = self._group_mean(edge_energy, self.incidence.relation_group)
        return torch.cat(
            [
                edge_energy.mean(dim=1, keepdim=True),
                _std_pool(edge_energy, dim=1).unsqueeze(1),
                edge_energy.amax(dim=1, keepdim=True),
                edge_energy.topk(k, dim=1).values.mean(dim=1, keepdim=True),
                gate.mean(dim=1, keepdim=True),
                _std_pool(gate, dim=1).unsqueeze(1),
                ray_visibility.mean(dim=1, keepdim=True),
                group_energy,
            ],
            dim=1,
        )

    def _convergence_features(self, edge_energy: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        incoming_energy = self._scatter_scalar(edge_energy, self.incidence.edge_dst, 64)
        incoming_gate = self._scatter_scalar(gate, self.incidence.edge_dst, 64)
        mean_incoming = incoming_energy / incoming_gate.clamp_min(1.0e-6)
        outgoing_energy = self._scatter_scalar(edge_energy, self.incidence.edge_src, 64)
        net_tension = (incoming_energy - outgoing_energy).abs()
        return torch.cat(
            [
                mean_incoming.mean(dim=1, keepdim=True),
                mean_incoming.amax(dim=1, keepdim=True),
                _std_pool(mean_incoming, dim=1).unsqueeze(1),
                incoming_energy.mean(dim=1, keepdim=True),
                incoming_energy.amax(dim=1, keepdim=True),
                incoming_gate.mean(dim=1, keepdim=True),
                incoming_gate.amax(dim=1, keepdim=True),
                net_tension.mean(dim=1, keepdim=True),
                net_tension.amax(dim=1, keepdim=True),
                (incoming_energy.topk(min(8, incoming_energy.shape[1]), dim=1).values.sum(dim=1, keepdim=True)
                 / incoming_energy.sum(dim=1, keepdim=True).clamp_min(1.0e-6)),
            ],
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        h = self.adapter(x, self.incidence.square_coords)
        occupancy = torch.sigmoid(self.occupancy_head(h)).squeeze(-1)
        ray_visibility = self._ray_visibility(occupancy)
        block_stats: list[torch.Tensor] = []
        edge_energies: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        for block in self.blocks:
            h, edge_energy, gate, _residual = block(
                h,
                self.incidence.edge_src,
                self.incidence.edge_dst,
                self.incidence.edge_type,
                ray_visibility,
            )
            edge_energies.append(edge_energy)
            gates.append(gate)
            block_stats.append(self._block_stats(edge_energy, gate, ray_visibility))

        last_energy = edge_energies[-1]
        last_gate = gates[-1]
        pooled = torch.cat(
            [
                h.mean(dim=1),
                h.amax(dim=1),
                _std_pool(h, dim=1),
                *block_stats,
                self._convergence_features(last_energy, last_gate),
                torch.stack(
                    [
                        occupancy.mean(dim=1),
                        _std_pool(occupancy, dim=1),
                        occupancy.amax(dim=1),
                        occupancy.amin(dim=1),
                    ],
                    dim=1,
                ),
            ],
            dim=1,
        )
        logits = _format_logits(self.classifier(pooled), self.num_classes)
        energy_stack = torch.stack(edge_energies, dim=1)
        gate_stack = torch.stack(gates, dim=1)
        group_energy = self._group_mean(last_energy, self.incidence.relation_group)
        ray_energy = group_energy[:, 0]
        knight_energy = group_energy[:, 1]
        king_energy = group_energy[:, 2]
        pawn_energy = 0.5 * (group_energy[:, 3] + group_energy[:, 4])
        incoming_energy = self._scatter_scalar(last_energy, self.incidence.edge_dst, 64)
        outgoing_energy = self._scatter_scalar(last_energy, self.incidence.edge_src, 64)
        diagnostics = {
            "logits": logits,
            "mechanism_energy": torch.log1p(energy_stack.mean(dim=(1, 2))),
            "proposal_profile_strength": gate_stack.mean(dim=(1, 2)),
            "proposal_keyword_count": logits.new_full((x.shape[0],), 5.0),
            "sheaf_tension": energy_stack.mean(dim=(1, 2)),
            "ray_visibility_mean": ray_visibility.mean(dim=1),
            "gate_mean": gate_stack.mean(dim=(1, 2)),
            "edge_energy_mean": energy_stack.mean(dim=(1, 2)),
            "ray_energy": ray_energy,
            "knight_energy": knight_energy,
            "king_energy": king_energy,
            "pawn_energy": pawn_energy,
            "convergence_tension": incoming_energy.mean(dim=1),
            "defense_gap": (incoming_energy - outgoing_energy).abs().mean(dim=1),
            "top_edge_tension": last_energy.topk(min(self.topk_edges, last_energy.shape[1]), dim=1).values.mean(dim=1),
            "occupancy_proxy_mean": occupancy.mean(dim=1),
        }
        return diagnostics


def build_attack_defense_sheaf_from_config(config: dict[str, Any]) -> AttackDefenseSheafNet:
    d_model = int(config.get("d_model", config.get("channels", 64)))
    hidden_dim = int(config.get("hidden_dim", max(96, d_model)))
    return AttackDefenseSheafNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        d_model=d_model,
        sheaf_rank=int(config.get("sheaf_rank", config.get("stalk_dim", 16))),
        num_blocks=int(config.get("num_blocks", config.get("sheaf_layers", config.get("depth", 2)))),
        type_emb_dim=int(config.get("type_emb_dim", min(16, d_model))),
        hidden_dim=hidden_dim,
        classifier_hidden=int(config.get("classifier_hidden", config.get("readout_hidden", hidden_dim))),
        dropout=float(config.get("dropout", 0.1)),
        edge_dropout=float(config.get("edge_dropout", min(0.1, float(config.get("dropout", 0.05))))),
        eta_init=float(config.get("eta_init", 0.2)),
        max_ray_length=int(config.get("max_ray_length", 7)),
        tie_file_reflection=bool(config.get("tie_file_reflection", True)),
        include_quiet_pawn_pressure=bool(config.get("include_quiet_pawn_pressure", False)),
        topk_edges=int(config.get("topk_edges", 32)),
    )

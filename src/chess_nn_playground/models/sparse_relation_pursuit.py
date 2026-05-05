from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


_RAY_DIRECTIONS = [
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
]
_KNIGHT_DIRECTIONS = [
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
]
_PAWN_DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


@dataclass
class SparseRelationPursuitConfig:
    input_channels: int = 112
    num_classes: int = 1
    square_dim: int = 64
    stem_depth: int = 3
    relation_dim: int = 64
    geom_dim: int = 16
    path_dim: int = 16
    num_atom_groups: int = 12
    atoms_per_group: int = 4
    pursuit_steps: int = 4
    classifier_hidden: int = 128
    dropout: float = 0.1
    max_ray_distance: int = 7
    edge_chunk_size: int = 256
    active_threshold: float = 1e-3
    use_batchnorm: bool = True


def _square_index(rank: int, file: int) -> int:
    return rank * 8 + file


def _inside_board(rank: int, file: int) -> bool:
    return 0 <= rank < 8 and 0 <= file < 8


def _inverse_sigmoid(value: float) -> float:
    value = min(max(value, 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(max(value, 1e-8)))


def _ordered_relation_edges(max_ray_distance: int) -> dict[str, torch.Tensor]:
    if max_ray_distance < 1 or max_ray_distance > 7:
        raise ValueError("max_ray_distance must be in [1, 7]")
    max_path_len = max(1, max_ray_distance - 1)
    src_indices: list[int] = []
    dst_indices: list[int] = []
    type_ids: list[int] = []
    direction_ids: list[int] = []
    distances: list[int] = []
    path_indices: list[list[int]] = []
    path_masks: list[list[bool]] = []

    def add_edge(
        src: int,
        dst: int,
        type_id: int,
        direction_id: int,
        distance: int,
        path: list[int] | None = None,
    ) -> None:
        path = path or []
        padded = path[:max_path_len] + [0] * max(0, max_path_len - len(path))
        mask = [True] * min(len(path), max_path_len) + [False] * max(0, max_path_len - len(path))
        src_indices.append(src)
        dst_indices.append(dst)
        type_ids.append(type_id)
        direction_ids.append(direction_id)
        distances.append(distance)
        path_indices.append(padded)
        path_masks.append(mask)

    for rank in range(8):
        for file in range(8):
            src = _square_index(rank, file)
            for direction_id, (dr, df) in enumerate(_RAY_DIRECTIONS):
                path: list[int] = []
                for distance in range(1, max_ray_distance + 1):
                    dst_rank = rank + dr * distance
                    dst_file = file + df * distance
                    if not _inside_board(dst_rank, dst_file):
                        break
                    dst = _square_index(dst_rank, dst_file)
                    add_edge(src, dst, 0, direction_id, distance, path)
                    path.append(dst)
            for knight_id, (dr, df) in enumerate(_KNIGHT_DIRECTIONS):
                dst_rank = rank + dr
                dst_file = file + df
                if _inside_board(dst_rank, dst_file):
                    add_edge(src, _square_index(dst_rank, dst_file), 1, 8 + knight_id, 2)
            for direction_id, (dr, df) in enumerate(_RAY_DIRECTIONS):
                dst_rank = rank + dr
                dst_file = file + df
                if _inside_board(dst_rank, dst_file):
                    add_edge(src, _square_index(dst_rank, dst_file), 2, direction_id, 1)
            for pawn_id, (dr, df) in enumerate(_PAWN_DIAGONALS):
                dst_rank = rank + dr
                dst_file = file + df
                if _inside_board(dst_rank, dst_file):
                    add_edge(src, _square_index(dst_rank, dst_file), 3, 16 + pawn_id // 2, 1)

    return {
        "src": torch.tensor(src_indices, dtype=torch.long),
        "dst": torch.tensor(dst_indices, dtype=torch.long),
        "type": torch.tensor(type_ids, dtype=torch.long),
        "direction": torch.tensor(direction_ids, dtype=torch.long),
        "distance": torch.tensor(distances, dtype=torch.long),
        "path_indices": torch.tensor(path_indices, dtype=torch.long),
        "path_mask": torch.tensor(path_masks, dtype=torch.bool),
    }


class SquareRelationStem(nn.Module):
    def __init__(
        self,
        input_channels: int,
        square_dim: int,
        stem_depth: int,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if stem_depth < 1:
            raise ValueError("stem_depth must be >= 1")
        layers: list[nn.Module] = []
        in_channels = input_channels
        for idx in range(stem_depth):
            kernel_size = 1 if idx == 0 else 3
            padding = kernel_size // 2
            layers.append(
                nn.Conv2d(
                    in_channels,
                    square_dim,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=not use_batchnorm,
                )
            )
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(square_dim))
            layers.append(nn.GELU())
            in_channels = square_dim
        self.layers = nn.Sequential(*layers)
        self.output_channels = square_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class GroupSparsePursuit(nn.Module):
    """Unrolled LISTA-style pursuit with atom groups and an explicit decoder."""

    def __init__(
        self,
        relation_dim: int,
        num_atom_groups: int,
        atoms_per_group: int,
        pursuit_steps: int,
        active_threshold: float = 1e-3,
    ) -> None:
        super().__init__()
        if relation_dim < 1:
            raise ValueError("relation_dim must be positive")
        if num_atom_groups < 1:
            raise ValueError("num_atom_groups must be positive")
        if atoms_per_group < 1:
            raise ValueError("atoms_per_group must be positive")
        if pursuit_steps < 1:
            raise ValueError("pursuit_steps must be positive")
        self.relation_dim = relation_dim
        self.num_atom_groups = num_atom_groups
        self.atoms_per_group = atoms_per_group
        self.num_atoms = num_atom_groups * atoms_per_group
        self.pursuit_steps = pursuit_steps
        self.active_threshold = float(active_threshold)

        dictionary = torch.empty(self.num_atoms, relation_dim)
        nn.init.orthogonal_(dictionary)
        self.dictionary = nn.Parameter(dictionary)
        self.raw_step = nn.Parameter(torch.full((pursuit_steps,), _inverse_sigmoid(0.70)))
        self.raw_l1 = nn.Parameter(torch.full((pursuit_steps, self.num_atoms), _inverse_softplus(0.015)))
        self.raw_group = nn.Parameter(torch.full((pursuit_steps, num_atom_groups), _inverse_softplus(0.025)))

    def normalized_dictionary(self) -> torch.Tensor:
        return F.normalize(self.dictionary, dim=1, eps=1e-8)

    def dictionary_coherence(self) -> torch.Tensor:
        dictionary = self.normalized_dictionary()
        gram = torch.matmul(dictionary, dictionary.transpose(0, 1))
        eye = torch.eye(self.num_atoms, device=gram.device, dtype=gram.dtype)
        off_diag = gram - eye
        denom = max(self.num_atoms * (self.num_atoms - 1), 1)
        return off_diag.pow(2).sum() / float(denom)

    def forward(self, relation_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        if relation_tokens.ndim != 3:
            raise ValueError(f"Expected relation tokens with shape (batch, edges, dim), got {tuple(relation_tokens.shape)}")
        batch, edge_count, dim = relation_tokens.shape
        if dim != self.relation_dim:
            raise ValueError(f"Expected relation dim {self.relation_dim}, got {dim}")

        dictionary = self.normalized_dictionary().to(device=relation_tokens.device, dtype=relation_tokens.dtype)
        codes = relation_tokens.new_zeros(batch, edge_count, self.num_atoms)
        residual_trace: list[torch.Tensor] = []

        for step_idx in range(self.pursuit_steps):
            reconstruction = torch.einsum("bek,kd->bed", codes, dictionary)
            gradient = torch.einsum("bed,kd->bek", reconstruction - relation_tokens, dictionary)
            step_size = 0.05 + 0.95 * torch.sigmoid(self.raw_step[step_idx]).to(dtype=relation_tokens.dtype)
            proposal = codes - step_size * gradient

            l1_threshold = F.softplus(self.raw_l1[step_idx]).to(device=relation_tokens.device, dtype=relation_tokens.dtype)
            proposal = torch.sign(proposal) * F.relu(proposal.abs() - l1_threshold.view(1, 1, -1))
            grouped = proposal.view(batch, edge_count, self.num_atom_groups, self.atoms_per_group)
            group_norm = grouped.norm(p=2, dim=3, keepdim=True)
            group_threshold = F.softplus(self.raw_group[step_idx]).to(
                device=relation_tokens.device,
                dtype=relation_tokens.dtype,
            )
            shrink = F.relu(1.0 - group_threshold.view(1, 1, self.num_atom_groups, 1) / group_norm.clamp_min(1e-8))
            codes = (grouped * shrink).reshape(batch, edge_count, self.num_atoms)

            reconstruction = torch.einsum("bek,kd->bed", codes, dictionary)
            residual = relation_tokens - reconstruction
            residual_trace.append(residual.pow(2).mean(dim=2))

        residual_by_step = torch.stack(residual_trace, dim=1)
        final_residual = residual_by_step[:, -1]
        grouped_codes = codes.view(batch, edge_count, self.num_atom_groups, self.atoms_per_group)
        group_norm = grouped_codes.norm(p=2, dim=3)
        group_energy = group_norm.pow(2).mean(dim=1)
        active_atoms = (codes.abs() > self.active_threshold).to(relation_tokens.dtype).mean(dim=(1, 2))
        active_groups = (group_norm > self.active_threshold).to(relation_tokens.dtype).mean(dim=(1, 2))

        return {
            "residual_by_step": residual_by_step,
            "final_residual": final_residual,
            "group_energy": group_energy,
            "active_atom_fraction": active_atoms,
            "active_group_fraction": active_groups,
            "mean_abs_code": codes.abs().mean(dim=(1, 2)),
            "mean_group_norm": group_norm.mean(dim=(1, 2)),
        }


class SparseRelationPursuitClassifier(nn.Module):
    """Sparse Relation Pursuit Asymmetry classifier.

    The classifier intentionally receives only sparse-code residuals, group
    energies, and activity statistics. Dense board or relation embeddings do not
    bypass the two equal-capacity pursuit branches.
    """

    def __init__(
        self,
        input_channels: int = 112,
        num_classes: int = 1,
        square_dim: int = 64,
        stem_depth: int = 3,
        relation_dim: int = 64,
        geom_dim: int = 16,
        path_dim: int = 16,
        num_atom_groups: int = 12,
        atoms_per_group: int = 4,
        pursuit_steps: int = 4,
        classifier_hidden: int = 128,
        dropout: float = 0.1,
        max_ray_distance: int = 7,
        edge_chunk_size: int = 256,
        active_threshold: float = 1e-3,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        if edge_chunk_size < 1:
            raise ValueError("edge_chunk_size must be positive")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = num_classes
        self.square_dim = square_dim
        self.relation_dim = relation_dim
        self.num_atom_groups = num_atom_groups
        self.pursuit_steps = pursuit_steps
        self.edge_chunk_size = edge_chunk_size

        self.stem = SquareRelationStem(
            input_channels=input_channels,
            square_dim=square_dim,
            stem_depth=stem_depth,
            use_batchnorm=use_batchnorm,
        )
        edges = _ordered_relation_edges(max_ray_distance=max_ray_distance)
        for name, tensor in edges.items():
            self.register_buffer(f"edge_{name}", tensor)
        self.edge_count = int(edges["src"].numel())

        self.type_embedding = nn.Embedding(4, geom_dim)
        self.direction_embedding = nn.Embedding(18, geom_dim)
        self.distance_embedding = nn.Embedding(max(max_ray_distance, 2) + 1, geom_dim)
        self.geom_norm = nn.LayerNorm(geom_dim)
        self.path_projection = nn.Sequential(
            nn.Linear(square_dim, path_dim),
            nn.GELU(),
            nn.LayerNorm(path_dim),
        )
        relation_input_dim = square_dim * 4 + geom_dim + path_dim
        self.relation_projection = nn.Sequential(
            nn.Linear(relation_input_dim, relation_dim),
            nn.GELU(),
            nn.LayerNorm(relation_dim),
            nn.Linear(relation_dim, relation_dim),
            nn.GELU(),
            nn.LayerNorm(relation_dim),
        )
        self.background_pursuit = GroupSparsePursuit(
            relation_dim=relation_dim,
            num_atom_groups=num_atom_groups,
            atoms_per_group=atoms_per_group,
            pursuit_steps=pursuit_steps,
            active_threshold=active_threshold,
        )
        self.tactical_pursuit = GroupSparsePursuit(
            relation_dim=relation_dim,
            num_atom_groups=num_atom_groups,
            atoms_per_group=atoms_per_group,
            pursuit_steps=pursuit_steps,
            active_threshold=active_threshold,
        )

        self.sparse_descriptor_dim = 3 * pursuit_steps + 3 * num_atom_groups + 17
        self.descriptor_norm = nn.LayerNorm(self.sparse_descriptor_dim)
        self.head = nn.Sequential(
            self.descriptor_norm,
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(self.sparse_descriptor_dim, classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(classifier_hidden, num_classes),
        )

    def _relation_tokens(self, squares: torch.Tensor) -> torch.Tensor:
        src = squares.index_select(1, self.edge_src)
        dst = squares.index_select(1, self.edge_dst)
        path_flat = self.edge_path_indices.reshape(-1)
        path_squares = squares.index_select(1, path_flat).reshape(
            squares.shape[0],
            self.edge_count,
            self.edge_path_indices.shape[1],
            squares.shape[2],
        )
        path_mask = self.edge_path_mask.to(device=squares.device, dtype=squares.dtype).view(
            1,
            self.edge_count,
            self.edge_path_indices.shape[1],
            1,
        )
        path_count = path_mask.sum(dim=2).clamp_min(1.0)
        path_mean = (path_squares * path_mask).sum(dim=2) / path_count
        path_features = self.path_projection(path_mean)

        geom = (
            self.type_embedding(self.edge_type)
            + self.direction_embedding(self.edge_direction)
            + self.distance_embedding(self.edge_distance)
        )
        geom = self.geom_norm(geom).unsqueeze(0).expand(squares.shape[0], -1, -1)

        relation_features = torch.cat(
            [
                src,
                dst,
                src * dst,
                (src - dst).abs(),
                geom.to(dtype=squares.dtype),
                path_features,
            ],
            dim=2,
        )
        return self.relation_projection(relation_features)

    def _branch_stats(self, pursuit: GroupSparsePursuit, relation_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = relation_tokens.shape[0]
        dtype = relation_tokens.dtype
        device = relation_tokens.device
        total_edges = relation_tokens.shape[1]
        residual_sum = relation_tokens.new_zeros(batch, self.pursuit_steps)
        final_sum = relation_tokens.new_zeros(batch)
        final_sumsq = relation_tokens.new_zeros(batch)
        group_energy_sum = relation_tokens.new_zeros(batch, self.num_atom_groups)
        active_atom_sum = relation_tokens.new_zeros(batch)
        active_group_sum = relation_tokens.new_zeros(batch)
        mean_abs_sum = relation_tokens.new_zeros(batch)
        mean_group_sum = relation_tokens.new_zeros(batch)

        for start in range(0, total_edges, self.edge_chunk_size):
            stop = min(start + self.edge_chunk_size, total_edges)
            edge_weight = float(stop - start)
            stats = pursuit(relation_tokens[:, start:stop])
            residual = stats["residual_by_step"]
            final = stats["final_residual"]
            residual_sum = residual_sum + residual.sum(dim=2)
            final_sum = final_sum + final.sum(dim=1)
            final_sumsq = final_sumsq + final.pow(2).sum(dim=1)
            group_energy_sum = group_energy_sum + stats["group_energy"] * edge_weight
            active_atom_sum = active_atom_sum + stats["active_atom_fraction"] * edge_weight
            active_group_sum = active_group_sum + stats["active_group_fraction"] * edge_weight
            mean_abs_sum = mean_abs_sum + stats["mean_abs_code"] * edge_weight
            mean_group_sum = mean_group_sum + stats["mean_group_norm"] * edge_weight

        denom = float(max(total_edges, 1))
        residual_steps = residual_sum / denom
        final_mean = final_sum / denom
        final_var = (final_sumsq / denom - final_mean.pow(2)).clamp_min(0.0)
        group_energy = group_energy_sum / denom
        group_prob = group_energy / group_energy.sum(dim=1, keepdim=True).clamp_min(1e-8)
        group_entropy = -(group_prob * group_prob.clamp_min(1e-8).log()).sum(dim=1) / math.log(
            max(self.num_atom_groups, 2)
        )

        return {
            "residual_steps": residual_steps,
            "final_residual": final_mean,
            "final_residual_std": final_var.sqrt(),
            "group_energy": group_energy,
            "active_atom_fraction": active_atom_sum / denom,
            "active_group_fraction": active_group_sum / denom,
            "mean_abs_code": mean_abs_sum / denom,
            "mean_group_norm": mean_group_sum / denom,
            "group_entropy": group_entropy.to(device=device, dtype=dtype),
        }

    def _descriptor(
        self,
        background: dict[str, torch.Tensor],
        tactical: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eps = 1e-6
        bg_steps = torch.log1p(background["residual_steps"])
        tac_steps = torch.log1p(tactical["residual_steps"])
        residual_step_gap = bg_steps - tac_steps
        bg_group = torch.log1p(background["group_energy"])
        tac_group = torch.log1p(tactical["group_energy"])
        group_gap = bg_group - tac_group

        bg_final = background["final_residual"]
        tac_final = tactical["final_residual"]
        asymmetry = torch.log(bg_final + eps) - torch.log(tac_final + eps)
        bg_drop = bg_steps[:, 0] - torch.log1p(bg_final)
        tac_drop = tac_steps[:, 0] - torch.log1p(tac_final)
        scalars = torch.stack(
            [
                torch.log1p(bg_final),
                torch.log1p(tac_final),
                asymmetry,
                torch.log1p(background["final_residual_std"]),
                torch.log1p(tactical["final_residual_std"]),
                bg_drop,
                tac_drop,
                background["active_atom_fraction"],
                tactical["active_atom_fraction"],
                background["active_group_fraction"],
                tactical["active_group_fraction"],
                background["group_entropy"],
                tactical["group_entropy"],
                background["mean_abs_code"],
                tactical["mean_abs_code"],
                background["mean_group_norm"],
                tactical["mean_group_norm"],
            ],
            dim=1,
        )
        descriptor = torch.cat([bg_steps, tac_steps, residual_step_gap, bg_group, tac_group, group_gap, scalars], dim=1)
        return descriptor, asymmetry

    def dictionary_coherence(self) -> torch.Tensor:
        return 0.5 * (
            self.background_pursuit.dictionary_coherence() + self.tactical_pursuit.dictionary_coherence()
        )

    def branch_separation(self) -> torch.Tensor:
        background = self.background_pursuit.normalized_dictionary()
        tactical = self.tactical_pursuit.normalized_dictionary()
        cosine = torch.matmul(background, tactical.transpose(0, 1))
        return cosine.pow(2).mean()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        square_map = self.stem(x)
        squares = square_map.flatten(2).transpose(1, 2)
        relation_tokens = self._relation_tokens(squares)
        background = self._branch_stats(self.background_pursuit, relation_tokens)
        tactical = self._branch_stats(self.tactical_pursuit, relation_tokens)
        descriptor, asymmetry = self._descriptor(background, tactical)
        logits = self.head(descriptor)
        if self.num_classes == 1:
            logits = logits.view(-1)
        aux_logit = asymmetry

        combined_group_energy = 0.5 * (background["group_energy"] + tactical["group_energy"])
        dead_group_penalty = F.relu(1e-4 - combined_group_energy.mean(dim=0)).mean()
        mean_abs_code = 0.5 * (background["mean_abs_code"] + tactical["mean_abs_code"])
        mean_group_norm = 0.5 * (background["mean_group_norm"] + tactical["mean_group_norm"])

        return {
            "logits": logits,
            "aux_logit": aux_logit,
            "bg_final_residual": background["final_residual"],
            "tac_final_residual": tactical["final_residual"],
            "residual_asymmetry": asymmetry,
            "bg_active_atom_fraction": background["active_atom_fraction"],
            "tac_active_atom_fraction": tactical["active_atom_fraction"],
            "bg_active_group_fraction": background["active_group_fraction"],
            "tac_active_group_fraction": tactical["active_group_fraction"],
            "bg_group_entropy": background["group_entropy"],
            "tac_group_entropy": tactical["group_entropy"],
            "mean_abs_code": mean_abs_code,
            "mean_group_norm": mean_group_norm,
            "dictionary_coherence": self.dictionary_coherence(),
            "branch_separation": self.branch_separation(),
            "dead_group_penalty": dead_group_penalty,
            "bg_group_energy": background["group_energy"],
            "tac_group_energy": tactical["group_energy"],
        }


def build_sparse_relation_pursuit_from_config(config: dict[str, Any]) -> SparseRelationPursuitClassifier:
    cfg = SparseRelationPursuitConfig(
        input_channels=int(config.get("input_channels", SparseRelationPursuitConfig.input_channels)),
        num_classes=int(config.get("num_classes", SparseRelationPursuitConfig.num_classes)),
        square_dim=int(config.get("square_dim", SparseRelationPursuitConfig.square_dim)),
        stem_depth=int(config.get("stem_depth", SparseRelationPursuitConfig.stem_depth)),
        relation_dim=int(config.get("relation_dim", SparseRelationPursuitConfig.relation_dim)),
        geom_dim=int(config.get("geom_dim", SparseRelationPursuitConfig.geom_dim)),
        path_dim=int(config.get("path_dim", SparseRelationPursuitConfig.path_dim)),
        num_atom_groups=int(config.get("num_atom_groups", SparseRelationPursuitConfig.num_atom_groups)),
        atoms_per_group=int(config.get("atoms_per_group", SparseRelationPursuitConfig.atoms_per_group)),
        pursuit_steps=int(config.get("pursuit_steps", SparseRelationPursuitConfig.pursuit_steps)),
        classifier_hidden=int(config.get("classifier_hidden", SparseRelationPursuitConfig.classifier_hidden)),
        dropout=float(config.get("dropout", SparseRelationPursuitConfig.dropout)),
        max_ray_distance=int(config.get("max_ray_distance", SparseRelationPursuitConfig.max_ray_distance)),
        edge_chunk_size=int(config.get("edge_chunk_size", SparseRelationPursuitConfig.edge_chunk_size)),
        active_threshold=float(config.get("active_threshold", SparseRelationPursuitConfig.active_threshold)),
        use_batchnorm=bool(config.get("use_batchnorm", SparseRelationPursuitConfig.use_batchnorm)),
    )
    return SparseRelationPursuitClassifier(**cfg.__dict__)

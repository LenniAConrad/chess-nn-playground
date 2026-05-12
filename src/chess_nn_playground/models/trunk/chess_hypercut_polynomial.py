"""Chess Hypercut Polynomial Network for idea i082.

CHPNet builds deterministic current-board hyperedges over board squares, applies
masked high-order cut polynomials on learned square probes, and scatters the
exclusive-product derivative residuals back to square states.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


DEFAULT_MAX_EDGES = 1024
DEFAULT_MAX_EDGE_SIZE = 9
MIN_EDGE_SIZE = 3
PIECE_CHANNELS = {
    "white_pawn": 0,
    "white_knight": 1,
    "white_bishop": 2,
    "white_rook": 3,
    "white_queen": 4,
    "white_king": 5,
    "black_pawn": 6,
    "black_knight": 7,
    "black_bishop": 8,
    "black_rook": 9,
    "black_queen": 10,
    "black_king": 11,
}
EPS = 1e-6


@dataclass(frozen=True)
class HyperedgeBatch:
    edge_index: torch.Tensor
    edge_mask: torch.Tensor
    edge_active: torch.Tensor
    edge_size: torch.Tensor


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.view(-1) if num_classes == 1 else logits


def _vertex(row: int, file: int) -> int:
    return row * 8 + file


def _inside(row: int, file: int) -> bool:
    return 0 <= row < 8 and 0 <= file < 8


def _row_file(vertex_id: int) -> tuple[int, int]:
    return vertex_id // 8, vertex_id % 8


def _normalize_simple18_side_to_move(sample: torch.Tensor) -> torch.Tensor:
    if sample.shape[0] < 18:
        return sample
    white_to_move = bool(sample[12].mean().item() >= 0.5)
    if white_to_move:
        normalized = sample.clone()
        normalized[12].fill_(1.0)
        return normalized
    rotated = torch.rot90(sample, k=2, dims=(-2, -1))
    normalized = rotated.clone()
    normalized[:6] = rotated[6:12]
    normalized[6:12] = rotated[:6]
    normalized[12].fill_(1.0)
    normalized[13] = rotated[15]
    normalized[14] = rotated[16]
    normalized[15] = rotated[13]
    normalized[16] = rotated[14]
    normalized[17] = rotated[17]
    return normalized


def normalize_side_to_move_tensor(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] < 18:
        return x
    rotated = torch.rot90(x, k=2, dims=(-2, -1))
    black_normalized = rotated.clone()
    black_normalized[:, :6] = rotated[:, 6:12]
    black_normalized[:, 6:12] = rotated[:, :6]
    black_normalized[:, 12] = 1.0
    black_normalized[:, 13] = rotated[:, 15]
    black_normalized[:, 14] = rotated[:, 16]
    black_normalized[:, 15] = rotated[:, 13]
    black_normalized[:, 16] = rotated[:, 14]
    black_normalized[:, 17] = rotated[:, 17]
    white_selector = (x[:, 12].mean(dim=(1, 2)) >= 0.5).to(dtype=x.dtype).view(-1, 1, 1, 1)
    white_normalized = x.clone()
    white_normalized[:, 12] = 1.0
    return white_selector * white_normalized + (1.0 - white_selector) * black_normalized


def exclusive_prod(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x.transpose(dim, -1)
    prefix = torch.cumprod(x, dim=-1)
    suffix = torch.cumprod(torch.flip(x, dims=[-1]), dim=-1)
    suffix = torch.flip(suffix, dims=[-1])
    ones = torch.ones_like(x[..., :1])
    left = torch.cat([ones, prefix[..., :-1]], dim=-1)
    right = torch.cat([suffix[..., 1:], ones], dim=-1)
    return (left * right).transpose(dim, -1)


class ChessHyperedgeBuilder:
    def __init__(
        self,
        max_edges: int = DEFAULT_MAX_EDGES,
        max_edge_size: int = DEFAULT_MAX_EDGE_SIZE,
        cache_size: int = 4096,
    ) -> None:
        self.max_edges = int(max_edges)
        self.max_edge_size = int(max_edge_size)
        self.cache_size = int(cache_size)
        self._cache: dict[bytes, HyperedgeBatch] = {}

    def build(self, x: torch.Tensor) -> HyperedgeBatch:
        x_cpu = x.detach().to(device="cpu", dtype=torch.float32).contiguous()
        batches = [self._build_one(sample) for sample in x_cpu]
        return HyperedgeBatch(
            edge_index=torch.stack([item.edge_index for item in batches], dim=0).to(device=x.device),
            edge_mask=torch.stack([item.edge_mask for item in batches], dim=0).to(device=x.device),
            edge_active=torch.stack([item.edge_active for item in batches], dim=0).to(device=x.device),
            edge_size=torch.stack([item.edge_size for item in batches], dim=0).to(device=x.device, dtype=x.dtype),
        )

    def _build_one(self, sample: torch.Tensor) -> HyperedgeBatch:
        key = sample.numpy().tobytes()
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        normalized = _normalize_simple18_side_to_move(sample)
        occupancy = (normalized[:12].sum(dim=0) > 0.5)
        occupied_flat = [bool(value) for value in occupancy.reshape(-1).tolist()]
        piece_vertices = self._piece_vertices(normalized)
        king_vertices = set(piece_vertices["white_king"] + piece_vertices["black_king"])
        edges: set[tuple[int, ...]] = set()

        self._add_sliding_ray_edges(edges, occupancy, piece_vertices)
        self._add_piece_stencil_edges(edges, piece_vertices)
        self._add_line_window_edges(edges, occupied_flat)
        self._add_king_shell_edges(edges, piece_vertices)

        def priority(edge: tuple[int, ...]) -> tuple[bool, int, int, tuple[int, ...]]:
            touches_king = any(vertex_id in king_vertices for vertex_id in edge)
            occupied_count = sum(1 for vertex_id in edge if occupied_flat[vertex_id])
            return (not touches_king, -occupied_count, -len(edge), edge)

        sorted_edges = sorted(edges, key=priority)[: self.max_edges]
        edge_index = torch.zeros(self.max_edges, self.max_edge_size, dtype=torch.long)
        edge_mask = torch.zeros(self.max_edges, self.max_edge_size, dtype=torch.bool)
        edge_active = torch.zeros(self.max_edges, dtype=torch.bool)
        edge_size = torch.zeros(self.max_edges, dtype=torch.float32)
        for row, edge in enumerate(sorted_edges):
            edge_active[row] = True
            edge_size[row] = float(len(edge))
            for slot, vertex_id in enumerate(edge):
                edge_index[row, slot] = int(vertex_id)
                edge_mask[row, slot] = True

        result = HyperedgeBatch(
            edge_index=edge_index,
            edge_mask=edge_mask,
            edge_active=edge_active,
            edge_size=edge_size,
        )
        if self.cache_size > 0:
            if len(self._cache) >= self.cache_size:
                self._cache.clear()
            self._cache[key] = result
        return result

    @staticmethod
    def _piece_vertices(sample: torch.Tensor) -> dict[str, list[int]]:
        vertices: dict[str, list[int]] = {}
        for name, channel in PIECE_CHANNELS.items():
            positions = torch.nonzero(sample[channel] > 0.5, as_tuple=False)
            vertices[name] = [_vertex(int(row), int(file)) for row, file in positions.tolist()]
        return vertices

    def _keep_edge(self, edges: set[tuple[int, ...]], vertices: list[int]) -> None:
        edge = tuple(sorted(set(vertices)))
        if MIN_EDGE_SIZE <= len(edge) <= self.max_edge_size:
            edges.add(edge)

    def _add_sliding_ray_edges(
        self,
        edges: set[tuple[int, ...]],
        occupancy: torch.Tensor,
        piece_vertices: dict[str, list[int]],
    ) -> None:
        bishop_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        rook_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ray_specs = [
            ("white_bishop", bishop_dirs),
            ("black_bishop", bishop_dirs),
            ("white_rook", rook_dirs),
            ("black_rook", rook_dirs),
            ("white_queen", bishop_dirs + rook_dirs),
            ("black_queen", bishop_dirs + rook_dirs),
        ]
        for name, directions in ray_specs:
            for source in piece_vertices[name]:
                source_row, source_file = _row_file(source)
                for dr, df in directions:
                    ray = [source]
                    row = source_row + dr
                    file = source_file + df
                    while _inside(row, file):
                        vertex_id = _vertex(row, file)
                        ray.append(vertex_id)
                        if bool(occupancy[row, file]):
                            break
                        row += dr
                        file += df
                    self._keep_edge(edges, ray)

    def _add_piece_stencil_edges(self, edges: set[tuple[int, ...]], piece_vertices: dict[str, list[int]]) -> None:
        knight_offsets = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        king_offsets = [(dr, df) for dr in (-1, 0, 1) for df in (-1, 0, 1) if dr != 0 or df != 0]
        for name in ("white_knight", "black_knight"):
            for source in piece_vertices[name]:
                row, file = _row_file(source)
                vertices = [source]
                for dr, df in knight_offsets:
                    rr, ff = row + dr, file + df
                    if _inside(rr, ff):
                        vertices.append(_vertex(rr, ff))
                self._keep_edge(edges, vertices)
        for name in ("white_king", "black_king"):
            for source in piece_vertices[name]:
                row, file = _row_file(source)
                vertices = [source]
                for dr, df in king_offsets:
                    rr, ff = row + dr, file + df
                    if _inside(rr, ff):
                        vertices.append(_vertex(rr, ff))
                self._keep_edge(edges, vertices)
        pawn_specs = [("white_pawn", -1, 6), ("black_pawn", 1, 1)]
        for name, direction, start_row in pawn_specs:
            for source in piece_vertices[name]:
                row, file = _row_file(source)
                vertices = [source]
                for rr, ff in (
                    (row + direction, file),
                    (row + 2 * direction, file) if row == start_row else (-1, -1),
                    (row + direction, file - 1),
                    (row + direction, file + 1),
                ):
                    if _inside(rr, ff):
                        vertices.append(_vertex(rr, ff))
                self._keep_edge(edges, vertices)

    def _add_line_window_edges(self, edges: set[tuple[int, ...]], occupied_flat: list[bool]) -> None:
        lines: list[list[int]] = []
        lines.extend([[_vertex(row, file) for file in range(8)] for row in range(8)])
        lines.extend([[_vertex(row, file) for row in range(8)] for file in range(8)])
        for start_file in range(8):
            line = []
            row, file = 0, start_file
            while _inside(row, file):
                line.append(_vertex(row, file))
                row += 1
                file += 1
            if len(line) >= MIN_EDGE_SIZE:
                lines.append(line)
        for start_row in range(1, 8):
            line = []
            row, file = start_row, 0
            while _inside(row, file):
                line.append(_vertex(row, file))
                row += 1
                file += 1
            if len(line) >= MIN_EDGE_SIZE:
                lines.append(line)
        for start_file in range(8):
            line = []
            row, file = 0, start_file
            while _inside(row, file):
                line.append(_vertex(row, file))
                row += 1
                file -= 1
            if len(line) >= MIN_EDGE_SIZE:
                lines.append(line)
        for start_row in range(1, 8):
            line = []
            row, file = start_row, 7
            while _inside(row, file):
                line.append(_vertex(row, file))
                row += 1
                file -= 1
            if len(line) >= MIN_EDGE_SIZE:
                lines.append(line)

        for line in lines:
            max_window = min(8, len(line), self.max_edge_size)
            for window_size in range(MIN_EDGE_SIZE, max_window + 1):
                for start in range(0, len(line) - window_size + 1):
                    window = line[start : start + window_size]
                    if sum(1 for vertex_id in window if occupied_flat[vertex_id]) >= 2:
                        self._keep_edge(edges, window)

    def _add_king_shell_edges(self, edges: set[tuple[int, ...]], piece_vertices: dict[str, list[int]]) -> None:
        for name in ("white_king", "black_king"):
            for source in piece_vertices[name]:
                row, file = _row_file(source)
                shell = [source]
                for dr in (-1, 0, 1):
                    for df in (-1, 0, 1):
                        if dr == 0 and df == 0:
                            continue
                        rr, ff = row + dr, file + df
                        if _inside(rr, ff):
                            shell.append(_vertex(rr, ff))
                self._keep_edge(edges, shell)


def gather_vertices(values: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    batch, edge_count, edge_size = edge_index.shape
    dim = values.shape[-1]
    expanded = values.unsqueeze(1).expand(batch, edge_count, values.shape[1], dim)
    gather_index = edge_index.clamp_min(0).clamp_max(values.shape[1] - 1).unsqueeze(-1).expand(-1, -1, -1, dim)
    return torch.gather(expanded, 2, gather_index)


def scatter_add_vertices(
    values: torch.Tensor,
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    vertex_count: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, edge_count, edge_size, dim = values.shape
    source = values * edge_mask.to(dtype=values.dtype).unsqueeze(-1)
    flat_source = source.reshape(batch, edge_count * edge_size, dim)
    flat_index = edge_index.reshape(batch, edge_count * edge_size, 1).clamp_min(0).clamp_max(vertex_count - 1)
    out = values.new_zeros(batch, vertex_count, dim)
    out.scatter_add_(1, flat_index.expand(-1, -1, dim), flat_source)

    flat_count = edge_mask.to(dtype=values.dtype).reshape(batch, edge_count * edge_size, 1)
    incidence = values.new_zeros(batch, vertex_count, 1)
    incidence.scatter_add_(1, flat_index, flat_count)
    return out, incidence


def _masked_moments(cut: torch.Tensor, edge_active: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    active = edge_active.to(dtype=cut.dtype).unsqueeze(-1)
    count = active.sum(dim=1).clamp_min(EPS)
    mean = (cut * active).sum(dim=1) / count
    masked_cut = cut.masked_fill(~edge_active.unsqueeze(-1), -1.0e9)
    max_values = masked_cut.max(dim=1).values
    max_values = torch.where(edge_active.any(dim=1, keepdim=True), max_values, torch.zeros_like(max_values))
    variance = (((cut - mean.unsqueeze(1)) ** 2) * active).sum(dim=1) / count
    return mean, max_values, torch.sqrt(variance.clamp_min(0.0))


class HypercutBlock(nn.Module):
    def __init__(self, dim: int, probes: int, feedforward_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.probe = nn.Linear(dim, probes)
        self.out_weight = nn.Parameter(torch.empty(probes, dim))
        nn.init.normal_(self.out_weight, mean=0.0, std=0.02)
        self.norm_residual = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(feedforward_dim, dim),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
        edge_active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probe_state = torch.tanh(self.probe(h))
        gathered = gather_vertices(probe_state, edge_index).float()
        slot_mask = edge_mask.unsqueeze(-1)
        active_mask = edge_active.unsqueeze(-1)
        plus = torch.where(slot_mask, 0.5 * (1.0 + gathered), torch.ones_like(gathered))
        minus = torch.where(slot_mask, 0.5 * (1.0 - gathered), torch.ones_like(gathered))
        all_plus = plus.prod(dim=2)
        all_minus = minus.prod(dim=2)
        cut = (1.0 - all_plus - all_minus) * active_mask.to(dtype=gathered.dtype)

        plus_excl = exclusive_prod(plus, dim=2)
        minus_excl = exclusive_prod(minus, dim=2)
        derivative = (-0.5 * plus_excl + 0.5 * minus_excl) * slot_mask.to(dtype=gathered.dtype)
        derivative = derivative * edge_active[:, :, None, None].to(dtype=gathered.dtype)
        edge_delta = torch.einsum("bmkr,rd->bmkd", derivative.to(dtype=h.dtype), self.out_weight.to(dtype=h.dtype))
        delta, incidence = scatter_add_vertices(edge_delta, edge_index, edge_mask & edge_active.unsqueeze(-1))
        delta = delta / torch.sqrt(1.0 + incidence).clamp_min(EPS)

        h = self.norm_residual(h + self.dropout(delta))
        h = self.norm_ffn(h + self.dropout(self.ffn(h)))
        mean, max_values, std = _masked_moments(cut.to(dtype=h.dtype), edge_active)
        summary = torch.cat([mean, max_values, std], dim=-1)
        return h, summary


class ChessHypercutPolynomialNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 128,
        cut_probes: int = 32,
        hypercut_blocks: int = 4,
        feedforward_dim: int | None = None,
        head_dim: int | None = None,
        max_edges: int = DEFAULT_MAX_EDGES,
        max_edge_size: int = DEFAULT_MAX_EDGE_SIZE,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("ChessHypercutPolynomialNet supports the puzzle_binary one-logit contract")
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.cut_probes = int(cut_probes)
        self.hypercut_blocks = int(hypercut_blocks)
        self.edge_builder = ChessHyperedgeBuilder(max_edges=int(max_edges), max_edge_size=int(max_edge_size))
        feedforward_dim = int(feedforward_dim or max(2 * self.hidden_dim, 128))
        head_dim = int(head_dim or max(2 * self.hidden_dim, 128))

        self.stem = nn.Sequential(
            nn.Conv2d(int(input_channels), int(channels), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(channels), self.hidden_dim, kernel_size=1),
        )
        self.position = nn.Parameter(torch.zeros(64, self.hidden_dim))
        nn.init.normal_(self.position, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList(
            [
                HypercutBlock(self.hidden_dim, self.cut_probes, feedforward_dim, dropout=float(dropout))
                for _ in range(self.hypercut_blocks)
            ]
        )
        readout_dim = 2 * self.hidden_dim + self.hypercut_blocks * 3 * self.cut_probes
        self.head = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, head_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(head_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor | None = None,
        *,
        board: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
        edge_active: torch.Tensor | None = None,
        edge_size: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        board_tensor = board if board is not None else x
        if board_tensor is None:
            raise ValueError("forward requires a board tensor")
        board_tensor = require_board_tensor(board_tensor, self.spec)
        normalized = normalize_side_to_move_tensor(board_tensor)
        if edge_index is None or edge_mask is None or edge_active is None:
            built = self.edge_builder.build(board_tensor)
            edge_index = built.edge_index
            edge_mask = built.edge_mask
            edge_active = built.edge_active
            edge_size = built.edge_size
        else:
            edge_index = edge_index.to(device=board_tensor.device, dtype=torch.long)
            edge_mask = edge_mask.to(device=board_tensor.device, dtype=torch.bool)
            edge_active = edge_active.to(device=board_tensor.device, dtype=torch.bool)
            if edge_size is None:
                edge_size = edge_mask.to(dtype=board_tensor.dtype).sum(dim=-1)
            else:
                edge_size = edge_size.to(device=board_tensor.device, dtype=board_tensor.dtype)

        h = self.stem(normalized).flatten(2).transpose(1, 2)
        h = h + self.position.to(device=h.device, dtype=h.dtype).unsqueeze(0)
        summaries: list[torch.Tensor] = []
        for block in self.blocks:
            h, summary = block(h, edge_index, edge_mask, edge_active)
            summaries.append(summary)
        vertex_mean = h.mean(dim=1)
        vertex_max = h.max(dim=1).values
        cut_summary = torch.cat(summaries, dim=-1) if summaries else h.new_zeros(h.shape[0], 0)
        readout = torch.cat([vertex_mean, vertex_max, cut_summary], dim=-1)
        logits = _format_logits(self.head(readout), self.num_classes)

        active_count = edge_active.to(dtype=h.dtype).sum(dim=1)
        size_mean = (edge_size * edge_active.to(dtype=edge_size.dtype)).sum(dim=1) / active_count.clamp_min(1.0)
        cut_energy = cut_summary.pow(2).mean(dim=1) if cut_summary.numel() else logits.new_zeros(logits.shape[0])
        final_summary = summaries[-1] if summaries else logits.new_zeros(logits.shape[0], 3 * self.cut_probes)
        return {
            "logits": logits,
            "hyperedge_count": active_count,
            "hyperedge_size_mean": size_mean,
            "hypercut_energy": cut_energy,
            "hypercut_mean": final_summary[:, : self.cut_probes].mean(dim=1),
            "hypercut_max": final_summary[:, self.cut_probes : 2 * self.cut_probes].mean(dim=1),
            "hypercut_std": final_summary[:, 2 * self.cut_probes :].mean(dim=1),
            "higher_order_residual_energy": h.pow(2).mean(dim=(1, 2)),
            "mechanism_energy": cut_energy,
            "proposal_profile_strength": active_count / float(max(1, self.edge_builder.max_edges)),
            "proposal_keyword_count": logits.new_full((logits.shape[0],), 5.0),
        }


def build_chess_hypercut_polynomial_network_from_config(config: dict[str, Any]) -> ChessHypercutPolynomialNet:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    cfg.setdefault("channels", 64)
    return ChessHypercutPolynomialNet(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        cut_probes=int(cfg.get("cut_probes", 32)),
        hypercut_blocks=int(cfg.get("hypercut_blocks", 4)),
        feedforward_dim=int(cfg.get("feedforward_dim", max(2 * int(cfg.get("hidden_dim", 128)), 128))),
        head_dim=int(cfg.get("head_dim", max(2 * int(cfg.get("hidden_dim", 128)), 128))),
        max_edges=int(cfg.get("max_edges", DEFAULT_MAX_EDGES)),
        max_edge_size=int(cfg.get("max_edge_size", DEFAULT_MAX_EDGE_SIZE)),
        dropout=float(cfg.get("dropout", 0.1)),
    )

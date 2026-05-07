"""Bitboard shift-algebra network for idea i069."""
from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.data.board_features import SIMPLE_18
from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


MATERIAL_SUMMARY_DIM = 20
SHIFT_NAMES = (
    "north",
    "south",
    "east",
    "west",
    "ne",
    "nw",
    "se",
    "sw",
    "knight_nne",
    "knight_nee",
    "knight_see",
    "knight_sse",
    "knight_ssw",
    "knight_sww",
    "knight_nww",
    "knight_nnw",
)
SHIFT_DELTAS = {
    "north": (-1, 0),
    "south": (1, 0),
    "east": (0, 1),
    "west": (0, -1),
    "ne": (-1, 1),
    "nw": (-1, -1),
    "se": (1, 1),
    "sw": (1, -1),
    "knight_nne": (-2, 1),
    "knight_nee": (-1, 2),
    "knight_see": (1, 2),
    "knight_sse": (2, 1),
    "knight_ssw": (2, -1),
    "knight_sww": (1, -2),
    "knight_nww": (-1, -2),
    "knight_nnw": (-2, -1),
}
PATH_NAMES = (
    "identity",
    "orthogonal_one",
    "diagonal_one",
    "rook_two",
    "bishop_two",
    "rook_three",
    "bishop_three",
    "knight_jump",
    "king_ring",
    "pawn_capture_left",
    "pawn_capture_right",
    "knight_king_ring",
)


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


def _shift_map(delta_rank: int, delta_file: int) -> torch.Tensor:
    dest_to_source = torch.full((64,), -1, dtype=torch.long)
    for rank in range(8):
        for file in range(8):
            dest_rank = rank + delta_rank
            dest_file = file + delta_file
            if 0 <= dest_rank < 8 and 0 <= dest_file < 8:
                dest_to_source[dest_rank * 8 + dest_file] = rank * 8 + file
    return dest_to_source


def build_shift_maps() -> torch.Tensor:
    return torch.stack([_shift_map(*SHIFT_DELTAS[name]) for name in SHIFT_NAMES], dim=0)


def build_random_shift_maps(base_maps: torch.Tensor, seed: int = 69069) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    random_maps = torch.full_like(base_maps, -1)
    square_count = base_maps.shape[1]
    for op_idx, base in enumerate(base_maps):
        valid_count = int((base >= 0).sum().item())
        sources = torch.randperm(square_count, generator=generator)[:valid_count]
        destinations = torch.randperm(square_count, generator=generator)[:valid_count]
        random_maps[op_idx, destinations] = sources
    return random_maps


def _king_zone_matrix() -> torch.Tensor:
    matrix = torch.zeros(64, 64, dtype=torch.float32)
    for source in range(64):
        rank = source // 8
        file = source % 8
        for dr in (-1, 0, 1):
            for df in (-1, 0, 1):
                rr = rank + dr
                ff = file + df
                if 0 <= rr < 8 and 0 <= ff < 8:
                    matrix[source, rr * 8 + ff] = 1.0
    return matrix


def _material_summary(x: torch.Tensor) -> torch.Tensor:
    piece_planes = x[:, :12].clamp(0.0, 1.0)
    white_counts = piece_planes[:, :6].sum(dim=(2, 3))
    black_counts = piece_planes[:, 6:12].sum(dim=(2, 3))
    white_to_move = x[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0).view(-1, 1)
    own_counts = white_to_move * white_counts + (1.0 - white_to_move) * black_counts
    opp_counts = white_to_move * black_counts + (1.0 - white_to_move) * white_counts
    count_delta = own_counts - opp_counts
    values = x.new_tensor([1.0, 3.0, 3.0, 5.0, 9.0, 0.0])
    own_material = (own_counts * values).sum(dim=1, keepdim=True)
    opp_material = (opp_counts * values).sum(dim=1, keepdim=True)
    total_count = (own_counts + opp_counts).sum(dim=1, keepdim=True)
    material_balance = (own_material - opp_material) / 39.0
    return torch.cat(
        [
            own_counts / 8.0,
            opp_counts / 8.0,
            count_delta / 8.0,
            total_count / 32.0,
            material_balance,
        ],
        dim=1,
    )


@dataclass(frozen=True)
class ShiftAlgebraBatch:
    features: torch.Tensor
    alpha: torch.Tensor
    head_fields: torch.Tensor
    path_outputs: torch.Tensor
    shift_residual: torch.Tensor
    king_zone_residual: torch.Tensor
    occupied_energy: torch.Tensor


class BitboardStem(nn.Module):
    """Pointwise input projection with deterministic coordinate planes."""

    def __init__(
        self,
        input_channels: int = 18,
        width: int = 48,
        depth: int = 2,
        dropout: float = 0.0,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        rank = torch.linspace(0.0, 1.0, 8).view(1, 1, 8, 1).expand(1, 1, 8, 8)
        file = torch.linspace(0.0, 1.0, 8).view(1, 1, 1, 8).expand(1, 1, 8, 8)
        center = torch.sqrt((rank - 0.5).square() + (file - 0.5).square()) / (0.5 * 2.0**0.5)
        self.register_buffer("rank_plane", rank, persistent=False)
        self.register_buffer("file_plane", file, persistent=False)
        self.register_buffer("center_plane", center, persistent=False)

        layers: list[nn.Module] = []
        in_channels = input_channels + 4
        for block_idx in range(depth):
            kernel_size = 1 if block_idx == 0 else 3
            padding = 0 if kernel_size == 1 else 1
            layers.append(nn.Conv2d(in_channels, width, kernel_size=kernel_size, padding=padding, bias=not use_batchnorm))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(width))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            in_channels = width
        self.layers = nn.Sequential(*layers)
        self.output_channels = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = require_board_tensor(x, self.spec)
        batch = x.shape[0]
        rank = self.rank_plane.to(device=x.device, dtype=x.dtype).expand(batch, -1, -1, -1)
        file = self.file_plane.to(device=x.device, dtype=x.dtype).expand(batch, -1, -1, -1)
        center = self.center_plane.to(device=x.device, dtype=x.dtype).expand(batch, -1, -1, -1)
        white_to_move = x[:, 12:13].mean(dim=(2, 3), keepdim=True).clamp(0.0, 1.0)
        forward_rank = white_to_move * (1.0 - rank) + (1.0 - white_to_move) * rank
        return self.layers(torch.cat([x, rank, file, center, forward_rank], dim=1))


class CoefficientEmitter(nn.Module):
    """Emits board-conditioned polynomial path coefficients."""

    def __init__(
        self,
        width: int,
        heads: int,
        path_count: int,
        hidden_dim: int = 96,
        coefficient_mode: str = "tanh_scaled",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if coefficient_mode not in {"tanh_scaled", "softmax"}:
            raise ValueError("coefficient_mode must be 'tanh_scaled' or 'softmax'")
        self.heads = int(heads)
        self.path_count = int(path_count)
        self.coefficient_mode = coefficient_mode
        self.net = nn.Sequential(
            nn.LayerNorm(width * 2 + MATERIAL_SUMMARY_DIM),
            nn.Linear(width * 2 + MATERIAL_SUMMARY_DIM, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, self.heads * self.path_count),
        )
        self.fixed_alpha = nn.Parameter(torch.zeros(self.heads, self.path_count))

    def forward(self, summary: torch.Tensor, fixed_alpha: bool = False) -> torch.Tensor:
        if fixed_alpha:
            raw = self.fixed_alpha.unsqueeze(0).expand(summary.shape[0], -1, -1)
        else:
            raw = self.net(summary).view(summary.shape[0], self.heads, self.path_count)
        if self.coefficient_mode == "softmax":
            return F.softmax(raw, dim=-1)
        return torch.tanh(raw) / float(self.path_count) ** 0.5


class BitboardShiftAlgebraNetwork(nn.Module):
    """Sparse chess-shift polynomial classifier for puzzle_binary."""

    VALID_ABLATIONS = {
        "none",
        "cnn_only",
        "random_shift_bank",
        "orthogonal_only",
        "one_step_only",
        "fixed_alpha",
        "no_gate",
        "dense_conv_matched",
    }

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        width: int = 48,
        channels: int | None = None,
        heads: int = 6,
        path_depth_max: int = 3,
        hidden_dim: int = 96,
        classifier_width: int = 128,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        coefficient_mode: str = "tanh_scaled",
        use_gated_fusion: bool = True,
        use_cnn_summary: bool = True,
        ablation: str = "none",
        encoding: str = SIMPLE_18,
    ) -> None:
        super().__init__()
        if encoding != SIMPLE_18 or input_channels != 18:
            raise ValueError("BitboardShiftAlgebraNetwork currently implements the simple_18 board contract only")
        if ablation not in self.VALID_ABLATIONS:
            raise ValueError(f"Unknown bitboard shift-algebra ablation: {ablation}")
        if path_depth_max < 1 or path_depth_max > 3:
            raise ValueError("path_depth_max must be between 1 and 3")
        width = int(channels if channels is not None else width)
        self.num_classes = int(num_classes)
        self.width = width
        self.heads = int(heads)
        self.path_depth_max = int(path_depth_max)
        self.ablation = ablation
        self.use_gated_fusion = bool(use_gated_fusion)
        self.use_cnn_summary = bool(use_cnn_summary)
        self.path_names = PATH_NAMES
        self.shift_name_to_index = {name: idx for idx, name in enumerate(SHIFT_NAMES)}

        shift_maps = build_shift_maps()
        self.register_buffer("shift_maps", shift_maps, persistent=False)
        self.register_buffer("random_shift_maps", build_random_shift_maps(shift_maps), persistent=False)
        self.register_buffer("king_zone_matrix", _king_zone_matrix(), persistent=False)
        self.register_buffer("orthogonal_path_mask", self._path_mask({"identity", "orthogonal_one", "rook_two", "rook_three"}), persistent=False)
        self.register_buffer(
            "one_step_path_mask",
            self._path_mask(
                {
                    "identity",
                    "orthogonal_one",
                    "diagonal_one",
                    "knight_jump",
                    "king_ring",
                    "pawn_capture_left",
                    "pawn_capture_right",
                }
            ),
            persistent=False,
        )

        self.stem = BitboardStem(
            input_channels=input_channels,
            width=width,
            depth=depth,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )
        self.coefficients = CoefficientEmitter(
            width=width,
            heads=heads,
            path_count=len(self.path_names),
            hidden_dim=hidden_dim,
            coefficient_mode=coefficient_mode,
            dropout=dropout,
        )
        self.gate = nn.Sequential(
            nn.Conv2d(width * 2, width, kernel_size=1),
            nn.Sigmoid(),
        )
        self.dense_path_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(width, width, kernel_size=3, padding=1, groups=width),
                    nn.GELU(),
                    nn.Conv2d(width, width, kernel_size=1),
                )
                for _ in self.path_names
            ]
        )

        per_head_dim = width * 5 + 3
        shift_dim = per_head_dim * heads
        cnn_dim = width * 2 if self.use_cnn_summary else 0
        classifier_dim = shift_dim + cnn_dim + MATERIAL_SUMMARY_DIM + heads * 3
        mid_dim = max(32, classifier_width // 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_dim),
            nn.Linear(classifier_dim, classifier_width),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(classifier_width, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(mid_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, BoardTensorSpec(input_channels=18))
        u0 = self.stem(x)
        material = _material_summary(x)
        shift_batch = self._shift_algebra_features(x, u0, material)
        cnn_summary = torch.cat([u0.mean(dim=(2, 3)), u0.amax(dim=(2, 3))], dim=1)
        shift_features = shift_batch.features
        if self.ablation == "cnn_only":
            shift_features = torch.zeros_like(shift_features)

        alpha_abs = shift_batch.alpha.abs()
        alpha_probs = alpha_abs / alpha_abs.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        entropy = -(alpha_probs * alpha_probs.clamp_min(1.0e-8).log()).sum(dim=-1) / log(len(self.path_names))
        coefficient_summary = torch.cat(
            [
                entropy,
                alpha_abs.mean(dim=-1),
                alpha_abs.amax(dim=-1),
            ],
            dim=1,
        )

        fused = [shift_features, material, coefficient_summary]
        if self.use_cnn_summary:
            fused.insert(1, cnn_summary)
        logits = self.classifier(torch.cat(fused, dim=1))
        return {
            "logits": _format_logits(logits, self.num_classes),
            "coefficient_entropy": entropy.mean(dim=1),
            "coefficient_abs_mean": alpha_abs.mean(dim=(1, 2)),
            "top_path_strength": alpha_abs.amax(dim=-1).mean(dim=1),
            "shift_residual": shift_batch.shift_residual.mean(dim=1),
            "king_zone_shift_residual": shift_batch.king_zone_residual.mean(dim=1),
            "occupied_shift_energy": shift_batch.occupied_energy.mean(dim=1),
            "path_output_energy": shift_batch.path_outputs.square().mean(dim=(1, 2, 3, 4)),
            "head_field_energy": shift_batch.head_fields.square().mean(dim=(1, 2, 3, 4)),
            "cnn_energy": u0.square().mean(dim=(1, 2, 3)),
            "material_balance": material[:, -1],
            "piece_count": material[:, -2] * 32.0,
        }

    def _shift_algebra_features(self, x: torch.Tensor, u0: torch.Tensor, material: torch.Tensor) -> ShiftAlgebraBatch:
        board_summary = torch.cat([u0.mean(dim=(2, 3)), u0.amax(dim=(2, 3)), material], dim=1)
        alpha = self.coefficients(board_summary, fixed_alpha=self.ablation == "fixed_alpha")
        path_outputs = self._path_outputs(x, u0)
        if self.ablation == "orthogonal_only":
            path_outputs = path_outputs * self.orthogonal_path_mask.to(device=u0.device, dtype=u0.dtype).view(1, -1, 1, 1, 1)
        elif self.ablation == "one_step_only":
            path_outputs = path_outputs * self.one_step_path_mask.to(device=u0.device, dtype=u0.dtype).view(1, -1, 1, 1, 1)

        head_values = torch.einsum("bhp,bpcij->bhcij", alpha, path_outputs)
        u0_expanded = u0.unsqueeze(1).expand(-1, self.heads, -1, -1, -1)
        if self.ablation == "no_gate" or not self.use_gated_fusion:
            head_fields = u0_expanded + head_values
        else:
            gate_in = torch.cat([u0_expanded, head_values], dim=2).flatten(0, 1)
            gate = self.gate(gate_in).view(u0.shape[0], self.heads, self.width, 8, 8)
            head_fields = gate * head_values + (1.0 - gate) * u0_expanded

        occupied_mask = x[:, :12].sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        king_mask = self._king_zone_mask(x).view(x.shape[0], 1, 1, 8, 8)
        residual = (head_fields - u0_expanded).square().mean(dim=(2, 3, 4))
        king_residual = self._masked_field_mean((head_fields - u0_expanded).abs(), king_mask)
        occupied_energy = self._masked_field_mean(head_fields.abs(), occupied_mask.unsqueeze(1))

        mean_pool = head_fields.mean(dim=(3, 4))
        max_pool = head_fields.amax(dim=(3, 4))
        topk_pool = head_fields.flatten(3).topk(k=8, dim=-1).values.mean(dim=-1)
        occupied_pool = self._masked_channel_pool(head_fields, occupied_mask.unsqueeze(1))
        king_pool = self._masked_channel_pool(head_fields, king_mask)
        per_head = torch.cat(
            [
                mean_pool,
                max_pool,
                topk_pool,
                occupied_pool,
                king_pool,
                residual.unsqueeze(-1),
                king_residual.unsqueeze(-1),
                occupied_energy.unsqueeze(-1),
            ],
            dim=-1,
        )
        return ShiftAlgebraBatch(
            features=per_head.flatten(1),
            alpha=alpha,
            head_fields=head_fields,
            path_outputs=path_outputs,
            shift_residual=residual,
            king_zone_residual=king_residual,
            occupied_energy=occupied_energy,
        )

    def _path_outputs(self, x: torch.Tensor, u0: torch.Tensor) -> torch.Tensor:
        if self.ablation == "dense_conv_matched":
            return torch.stack([conv(u0) for conv in self.dense_path_convs], dim=1)
        outputs = [
            u0,
            self._sum_sequences(u0, [("north",), ("south",), ("east",), ("west",)]),
            self._sum_sequences(u0, [("ne",), ("nw",), ("se",), ("sw",)]),
            self._sum_sequences(u0, [("north", "north"), ("south", "south"), ("east", "east"), ("west", "west")]),
            self._sum_sequences(u0, [("ne", "ne"), ("nw", "nw"), ("se", "se"), ("sw", "sw")]),
            self._sum_sequences(
                u0,
                [
                    ("north", "north", "north"),
                    ("south", "south", "south"),
                    ("east", "east", "east"),
                    ("west", "west", "west"),
                ],
            ),
            self._sum_sequences(u0, [("ne", "ne", "ne"), ("nw", "nw", "nw"), ("se", "se", "se"), ("sw", "sw", "sw")]),
            self._sum_sequences(u0, [(name,) for name in SHIFT_NAMES if name.startswith("knight_")]),
            self._sum_sequences(u0, [(name,) for name in SHIFT_NAMES[:8]]),
            self._side_relative_pawn_capture(x, u0, left=True),
            self._side_relative_pawn_capture(x, u0, left=False),
            self._knight_king_ring(u0),
        ]
        if self.path_depth_max < 3:
            outputs[5] = torch.zeros_like(outputs[5])
            outputs[6] = torch.zeros_like(outputs[6])
        if self.path_depth_max < 2:
            outputs[3] = torch.zeros_like(outputs[3])
            outputs[4] = torch.zeros_like(outputs[4])
            outputs[11] = torch.zeros_like(outputs[11])
        return torch.stack(outputs, dim=1)

    def _sum_sequences(self, u: torch.Tensor, sequences: list[tuple[str, ...]]) -> torch.Tensor:
        total = torch.zeros_like(u)
        for sequence in sequences:
            shifted = u
            for name in sequence:
                shifted = self._apply_shift(shifted, name)
            total = total + shifted
        return total / float(len(sequences)) ** 0.5

    def _knight_king_ring(self, u: torch.Tensor) -> torch.Tensor:
        sequences = [
            (knight_name, king_name)
            for knight_name in SHIFT_NAMES
            if knight_name.startswith("knight_")
            for king_name in SHIFT_NAMES[:8]
        ]
        return self._sum_sequences(u, sequences)

    def _side_relative_pawn_capture(self, x: torch.Tensor, u: torch.Tensor, left: bool) -> torch.Tensor:
        white_to_move = x[:, 12:13].mean(dim=(2, 3), keepdim=True).clamp(0.0, 1.0)
        if left:
            white_shift = self._apply_shift(u, "nw")
            black_shift = self._apply_shift(u, "se")
        else:
            white_shift = self._apply_shift(u, "ne")
            black_shift = self._apply_shift(u, "sw")
        return white_to_move * white_shift + (1.0 - white_to_move) * black_shift

    def _apply_shift(self, u: torch.Tensor, name: str) -> torch.Tensor:
        maps = self.random_shift_maps if self.ablation == "random_shift_bank" else self.shift_maps
        op_idx = self.shift_name_to_index[name]
        dest_to_source = maps[op_idx].to(device=u.device)
        valid = (dest_to_source >= 0).to(device=u.device, dtype=u.dtype).view(1, 1, 64)
        gather_idx = dest_to_source.clamp_min(0).view(1, 1, 64).expand(u.shape[0], u.shape[1], -1)
        flat = u.flatten(2)
        shifted = flat.gather(2, gather_idx) * valid
        return shifted.view_as(u)

    def _king_zone_mask(self, x: torch.Tensor) -> torch.Tensor:
        king_occ = (x[:, 5] + x[:, 11]).clamp(0.0, 1.0).flatten(1)
        matrix = self.king_zone_matrix.to(device=x.device, dtype=x.dtype)
        return king_occ.matmul(matrix).clamp(0.0, 1.0)

    @staticmethod
    def _masked_field_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(device=values.device, dtype=values.dtype)
        denom = weights.sum(dim=(-1, -2)).clamp_min(1.0)
        return (values * weights).sum(dim=(2, 3, 4)) / denom.squeeze(1)

    @staticmethod
    def _masked_channel_pool(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(device=values.device, dtype=values.dtype)
        denom = weights.sum(dim=(-1, -2), keepdim=True).clamp_min(1.0)
        return (values * weights).sum(dim=(-1, -2)) / denom.squeeze(-1).squeeze(-1)

    def _path_mask(self, keep: set[str]) -> torch.Tensor:
        return torch.tensor([1.0 if name in keep else 0.0 for name in self.path_names], dtype=torch.float32)


def build_bitboard_shift_algebra_network_from_config(config: dict[str, Any]) -> BitboardShiftAlgebraNetwork:
    width = int(config.get("width", config.get("channels", 48)))
    return BitboardShiftAlgebraNetwork(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        width=width,
        heads=int(config.get("heads", config.get("num_heads", 6))),
        path_depth_max=int(config.get("path_depth_max", 3)),
        hidden_dim=int(config.get("coefficient_hidden_dim", config.get("hidden_dim", 96))),
        classifier_width=int(config.get("classifier_width", config.get("head_hidden", 128))),
        depth=int(config.get("depth", 2)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        coefficient_mode=str(config.get("coefficient_mode", "tanh_scaled")),
        use_gated_fusion=bool(config.get("use_gated_fusion", True)),
        use_cnn_summary=bool(config.get("use_cnn_summary", True)),
        ablation=str(config.get("ablation", "none")),
        encoding=str(config.get("encoding", SIMPLE_18)),
    )

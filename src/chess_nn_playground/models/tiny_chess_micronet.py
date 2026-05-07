"""Tiny Chess MicroNet for idea i073."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


LINE_BASE_NAMES: tuple[str, ...] = (
    "constant",
    "edge_heavy",
    "center_heavy",
    "side_relative_forward",
    "side_relative_backward",
    "occupancy_weighted",
)
MICRONET_ABLATIONS = {
    "none",
    "counts_only_mlp",
    "ordinary_tiny_cnn_matched",
    "flat_head_same_params",
    "no_line_sketch",
    "random_line_basis",
    "no_king_zone",
    "no_depthwise_local",
}


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


def _normalize_weights(weights: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1.0e-6)


def _line_ids() -> dict[str, torch.Tensor]:
    square = torch.arange(64)
    rank = torch.div(square, 8, rounding_mode="floor")
    file = square.remainder(8)
    return {
        "rank": rank,
        "file": file,
        "diag": rank - file + 7,
        "anti": rank + file,
    }


def _line_back_matrices() -> torch.Tensor:
    ids_by_direction = _line_ids()
    matrices: list[torch.Tensor] = []
    for ids in ids_by_direction.values():
        same_line = ids.view(64, 1) == ids.view(1, 64)
        denom = same_line.sum(dim=1, keepdim=True).clamp_min(1)
        matrices.append(same_line.to(dtype=torch.float32) / denom.to(dtype=torch.float32))
    return torch.stack(matrices, dim=0)


def _line_memberships() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ids_by_direction = _line_ids()
    max_lines = 15
    memberships = torch.zeros(4, max_lines, 64)
    valid = torch.zeros(4, max_lines)
    line_position = torch.zeros(4, max_lines)
    line_edge = torch.zeros(4, max_lines)
    for direction, ids in enumerate(ids_by_direction.values()):
        unique_ids = torch.arange(int(ids.max().item()) + 1)
        for line_id in unique_ids:
            mask = ids == line_id
            memberships[direction, line_id, mask] = 1.0
            valid[direction, line_id] = 1.0
        count = max(1, int(unique_ids.numel()) - 1)
        line_position[direction, : unique_ids.numel()] = unique_ids.to(dtype=torch.float32) / float(count)
        length = memberships[direction].sum(dim=1)
        position = line_position[direction]
        line_edge[direction] = torch.maximum((position - 0.5).abs() * 2.0, 1.0 - (length / length.max().clamp_min(1.0)))
    return memberships, valid, line_position, line_edge


def _king_zone_masks() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    square = torch.arange(64)
    rank = torch.div(square, 8, rounding_mode="floor")
    file = square.remainder(8)
    rank_i = rank.view(64, 1)
    file_i = file.view(64, 1)
    rank_j = rank.view(1, 64)
    file_j = file.view(1, 64)
    chebyshev = torch.maximum((rank_i - rank_j).abs(), (file_i - file_j).abs())
    zone3 = (chebyshev <= 1).to(dtype=torch.float32)
    zone5 = (chebyshev <= 2).to(dtype=torch.float32)
    ring5 = (zone5 - zone3).clamp_min(0.0)
    edge_rank = torch.where(rank_i <= 3, torch.zeros_like(rank_i), torch.full_like(rank_i, 7))
    edge_file = torch.where(file_i <= 3, torch.zeros_like(file_i), torch.full_like(file_i, 7))
    rank_edge_mask = (rank_j == edge_rank) & ((file_j - file_i).abs() <= 2)
    file_edge_mask = (file_j == edge_file) & ((rank_j - rank_i).abs() <= 2)
    edge_zone = (rank_edge_mask | file_edge_mask).to(dtype=torch.float32)
    return zone3, ring5, edge_zone


class LowRankConv1x1(nn.Module):
    def __init__(self, input_channels: int, width: int, squeeze_rank: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, squeeze_rank, kernel_size=1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(squeeze_rank, width, kernel_size=1),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MicroLineBlock(nn.Module):
    def __init__(self, width: int, mix_rank: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(width, width, kernel_size=3, padding=1, groups=width)
        self.mix_down = nn.Conv2d(width, mix_rank, kernel_size=1)
        self.mix_up = nn.Conv2d(mix_rank, width, kernel_size=1)
        self.line_gamma = nn.Parameter(torch.zeros(4, width))
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.register_buffer("line_back", _line_back_matrices(), persistent=False)
        nn.init.normal_(self.line_gamma, mean=0.0, std=0.02)

    def fixed_line_smooth(self, h: torch.Tensor) -> torch.Tensor:
        flat = h.flatten(2)
        backed = torch.einsum("bws,dts->bdwt", flat, self.line_back.to(device=h.device, dtype=h.dtype))
        line = (backed * self.line_gamma.to(dtype=h.dtype).view(1, 4, -1, 1)).sum(dim=1)
        return line.view_as(h)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(h) + self.fixed_line_smooth(h)
        y = F.relu6(y)
        y = self.mix_up(F.relu6(self.mix_down(y)))
        return F.relu6(y)


@dataclass(frozen=True)
class SketchGroups:
    global_pool: torch.Tensor
    line_sketch: torch.Tensor
    king_zone: torch.Tensor
    material: torch.Tensor
    malformed_king_count: torch.Tensor


class ChessSketchBank(nn.Module):
    def __init__(self, width: int, line_bases: tuple[str, ...] = LINE_BASE_NAMES, king_zone: bool = True) -> None:
        super().__init__()
        self.width = int(width)
        self.line_bases = tuple(line_bases)
        self.king_zone = bool(king_zone)
        unknown = set(self.line_bases) - set(LINE_BASE_NAMES)
        if unknown:
            raise ValueError(f"Unknown line bases: {sorted(unknown)}")
        memberships, valid, line_position, line_edge = _line_memberships()
        line_length = memberships.sum(dim=2)
        center = _normalize_weights(line_length, dim=1)
        edge = _normalize_weights(line_edge * valid, dim=1)
        constant = _normalize_weights(valid, dim=1)
        self.register_buffer("memberships", memberships, persistent=False)
        self.register_buffer("valid_lines", valid, persistent=False)
        self.register_buffer("line_position", line_position, persistent=False)
        self.register_buffer("basis_constant", constant, persistent=False)
        self.register_buffer("basis_edge", edge, persistent=False)
        self.register_buffer("basis_center", center, persistent=False)
        generator = torch.Generator().manual_seed(73073)
        random_basis = torch.rand(4, len(self.line_bases), 15, generator=generator) * valid.unsqueeze(1)
        self.register_buffer("random_basis", _normalize_weights(random_basis, dim=2), persistent=False)
        zone3, ring5, edge_zone = _king_zone_masks()
        self.register_buffer("king_zone3", zone3, persistent=False)
        self.register_buffer("king_ring5", ring5, persistent=False)
        self.register_buffer("king_edge_zone", edge_zone, persistent=False)
        self.global_dim = 3 * width
        self.line_dim = 4 * len(self.line_bases) * width
        self.king_dim = 6 * width
        self.material_dim = 18
        self.output_dim = self.global_dim + self.line_dim + self.king_dim + self.material_dim

    def _global_pool(self, h_flat: torch.Tensor) -> torch.Tensor:
        mean = h_flat.mean(dim=2)
        max_value = h_flat.amax(dim=2)
        mad = (h_flat - mean.unsqueeze(-1)).abs().mean(dim=2)
        return torch.cat([mean, max_value, mad], dim=1)

    def _line_sketch(
        self,
        h_flat: torch.Tensor,
        x: torch.Tensor,
        *,
        random_basis: bool = False,
        disabled: bool = False,
    ) -> torch.Tensor:
        batch = h_flat.shape[0]
        if disabled or not self.line_bases:
            return h_flat.new_zeros(batch, self.line_dim)
        memberships = self.memberships.to(device=h_flat.device, dtype=h_flat.dtype)
        denom = memberships.sum(dim=2).clamp_min(1.0)
        line_mean = torch.einsum("bws,dls->bwdl", h_flat, memberships) / denom.view(1, 1, 4, 15)
        if random_basis:
            basis = self.random_basis.to(device=h_flat.device, dtype=h_flat.dtype)
            descriptors = torch.einsum("bwdl,dkl->bwdk", line_mean, basis)
            return descriptors.permute(0, 2, 3, 1).reshape(batch, -1)

        side = x[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0).view(batch, 1, 1) if x.shape[1] > 12 else x.new_ones(batch, 1, 1)
        line_pos = self.line_position.to(device=h_flat.device, dtype=h_flat.dtype)
        valid = self.valid_lines.to(device=h_flat.device, dtype=h_flat.dtype)
        occupancy = x[:, :12].clamp(0.0, 1.0).sum(dim=1).clamp(0.0, 1.0).flatten(1) if x.shape[1] >= 12 else x.new_zeros(batch, 64)
        occ_by_line = torch.einsum("bs,dls->bdl", occupancy, memberships)
        occ_basis = occ_by_line / occ_by_line.sum(dim=2, keepdim=True).clamp_min(1.0)
        forward = _normalize_weights((side * line_pos + (1.0 - side) * (1.0 - line_pos)) * valid, dim=2)
        backward = _normalize_weights(((1.0 - side) * line_pos + side * (1.0 - line_pos)) * valid, dim=2)

        fixed_basis = {
            "constant": self.basis_constant.to(device=h_flat.device, dtype=h_flat.dtype).unsqueeze(0).expand(batch, -1, -1),
            "edge_heavy": self.basis_edge.to(device=h_flat.device, dtype=h_flat.dtype).unsqueeze(0).expand(batch, -1, -1),
            "center_heavy": self.basis_center.to(device=h_flat.device, dtype=h_flat.dtype).unsqueeze(0).expand(batch, -1, -1),
            "side_relative_forward": forward,
            "side_relative_backward": backward,
            "occupancy_weighted": occ_basis,
        }
        parts = []
        for base_name in self.line_bases:
            weights = fixed_basis[base_name]
            parts.append(torch.einsum("bwdl,bdl->bwd", line_mean, weights))
        descriptors = torch.stack(parts, dim=3)
        return descriptors.permute(0, 2, 3, 1).reshape(batch, -1)

    def _king_pools(self, h_flat: torch.Tensor, x: torch.Tensor, *, disabled: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        batch = h_flat.shape[0]
        if disabled or not self.king_zone or x.shape[1] < 12:
            return h_flat.new_zeros(batch, self.king_dim), h_flat.new_ones(batch)
        white_king = x[:, 5].clamp(0.0, 1.0).flatten(1)
        black_king = x[:, 11].clamp(0.0, 1.0).flatten(1)
        white_valid = (white_king.sum(dim=1) == 1.0).to(dtype=h_flat.dtype)
        black_valid = (black_king.sum(dim=1) == 1.0).to(dtype=h_flat.dtype)
        zone_bank = torch.stack([self.king_zone3, self.king_ring5, self.king_edge_zone], dim=0).to(
            device=h_flat.device, dtype=h_flat.dtype
        )

        def select(king: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            selected = torch.einsum("bs,zst->bzt", king.to(dtype=h_flat.dtype), zone_bank)
            return selected * valid.view(batch, 1, 1)

        white_masks = select(white_king, white_valid)
        black_masks = select(black_king, black_valid)
        side = x[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0).view(batch, 1, 1) if x.shape[1] > 12 else x.new_ones(batch, 1, 1)
        own_masks = side * white_masks + (1.0 - side) * black_masks
        opponent_masks = side * black_masks + (1.0 - side) * white_masks

        def pool(masks: torch.Tensor) -> torch.Tensor:
            denom = masks.sum(dim=2).clamp_min(1.0)
            pooled = torch.einsum("bws,bzs->bwz", h_flat, masks) / denom.unsqueeze(1)
            return pooled.transpose(1, 2).reshape(batch, -1)

        malformed = 2.0 - white_valid - black_valid
        return torch.cat([pool(own_masks), pool(opponent_masks)], dim=1), malformed

    def _material_summary(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        if x.shape[1] < 12:
            return x.new_zeros(batch, self.material_dim)
        piece_counts = x[:, :12].clamp(0.0, 1.0).sum(dim=(2, 3)) / 16.0
        side = x[:, 12].mean(dim=(1, 2), keepdim=False).unsqueeze(1) if x.shape[1] > 12 else x.new_ones(batch, 1)
        castling = x[:, 13:17].mean(dim=(2, 3)).sum(dim=1, keepdim=True) / 4.0 if x.shape[1] >= 17 else x.new_zeros(batch, 1)
        en_passant = x[:, 17].amax(dim=(1, 2), keepdim=False).unsqueeze(1) if x.shape[1] > 17 else x.new_zeros(batch, 1)
        white_occ = x[:, :6].clamp(0.0, 1.0).sum(dim=(1, 2, 3), keepdim=False).unsqueeze(1) / 16.0
        black_occ = x[:, 6:12].clamp(0.0, 1.0).sum(dim=(1, 2, 3), keepdim=False).unsqueeze(1) / 16.0
        total_occ = white_occ + black_occ
        own_occ = side * white_occ + (1.0 - side) * black_occ
        opponent_occ = side * black_occ + (1.0 - side) * white_occ
        return torch.cat([piece_counts, side, castling, en_passant, total_occ, own_occ, opponent_occ], dim=1)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        *,
        no_line_sketch: bool = False,
        random_line_basis: bool = False,
        no_king_zone: bool = False,
    ) -> SketchGroups:
        h_flat = h.flatten(2)
        global_pool = self._global_pool(h_flat)
        line_sketch = self._line_sketch(h_flat, x, random_basis=random_line_basis, disabled=no_line_sketch)
        king_zone, malformed = self._king_pools(h_flat, x, disabled=no_king_zone)
        material = self._material_summary(x)
        return SketchGroups(
            global_pool=global_pool,
            line_sketch=line_sketch,
            king_zone=king_zone,
            material=material,
            malformed_king_count=malformed,
        )


class MicroDescriptorHead(nn.Module):
    def __init__(
        self,
        group_dims: dict[str, int],
        head_hidden: int,
        num_classes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.group_dims = dict(group_dims)
        self.group_order = ("global_pool", "line_sketch", "king_zone", "material")
        descriptor_dim = sum(self.group_dims[name] for name in self.group_order)
        self.fc1 = nn.Linear(descriptor_dim, head_hidden)
        self.fc2 = nn.Linear(head_hidden, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, groups: SketchGroups) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z = torch.cat([getattr(groups, name) for name in self.group_order], dim=1)
        hidden = self.dropout(F.relu6(self.fc1(z)))
        logits = self.fc2(hidden)
        diagnostics = {}
        offset = 0
        norms = []
        for name in self.group_order:
            dim = self.group_dims[name]
            weight = self.fc1.weight[:, offset : offset + dim]
            norm = weight.norm()
            diagnostics[f"{name}_norm"] = norm
            norms.append(norm)
            offset += dim
        total = torch.stack(norms).sum().clamp_min(1.0e-6)
        for name in self.group_order:
            diagnostics[f"{name}_norm_fraction"] = diagnostics[f"{name}_norm"] / total
        return logits, diagnostics


class TinyChessMicroNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        width: int = 16,
        squeeze_rank: int | None = None,
        blocks: int = 3,
        mix_rank: int | None = None,
        head_hidden: int = 32,
        dropout: float = 0.1,
        line_bases: str | list[str] | tuple[str, ...] = "all_6",
        king_zone: bool = True,
        quantization_target: str = "int8",
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if ablation not in MICRONET_ABLATIONS:
            raise ValueError(f"Unsupported ablation {ablation!r}; expected one of {sorted(MICRONET_ABLATIONS)}")
        if width < 1 or blocks < 0:
            raise ValueError("width must be positive and blocks must be non-negative")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.width = int(width)
        self.ablation = ablation
        self.quantization_target = str(quantization_target)
        squeeze_rank = int(squeeze_rank or max(4, round(width / 3)))
        mix_rank = int(mix_rank or (4 if width <= 12 else 6 if width <= 16 else 8))
        if line_bases == "all_6":
            line_basis_names = LINE_BASE_NAMES
        else:
            line_basis_names = tuple(line_bases)
        self.squeeze = LowRankConv1x1(input_channels, width, squeeze_rank)
        self.blocks = nn.ModuleList([MicroLineBlock(width, mix_rank) for _ in range(blocks)])
        self.sketch = ChessSketchBank(width, line_bases=line_basis_names, king_zone=king_zone)
        self.flat_descriptor = (
            nn.Sequential(
                nn.Linear(width * 64, head_hidden),
                nn.ReLU6(inplace=True),
                nn.Linear(head_hidden, self.sketch.output_dim),
                nn.ReLU6(inplace=True),
            )
            if ablation == "flat_head_same_params"
            else None
        )
        self.cnn_control = nn.Sequential(
            nn.Conv2d(input_channels, width, kernel_size=3, padding=1, groups=1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1, groups=width),
            nn.ReLU6(inplace=True),
        )
        group_dims = {
            "global_pool": self.sketch.global_dim,
            "line_sketch": self.sketch.line_dim,
            "king_zone": self.sketch.king_dim,
            "material": self.sketch.material_dim,
        }
        self.head = MicroDescriptorHead(group_dims, head_hidden=head_hidden, num_classes=num_classes, dropout=dropout)

    def _field(self, x: torch.Tensor) -> torch.Tensor:
        if self.ablation == "ordinary_tiny_cnn_matched":
            return self.cnn_control(x)
        h = self.squeeze(x)
        if self.ablation != "no_depthwise_local":
            for block in self.blocks:
                h = F.relu6(h + block.scale * block(h))
        return h

    def _apply_ablation(self, groups: SketchGroups, h: torch.Tensor) -> SketchGroups:
        if self.ablation == "counts_only_mlp":
            return SketchGroups(
                global_pool=torch.zeros_like(groups.global_pool),
                line_sketch=torch.zeros_like(groups.line_sketch),
                king_zone=torch.zeros_like(groups.king_zone),
                material=groups.material,
                malformed_king_count=groups.malformed_king_count,
            )
        if self.ablation == "ordinary_tiny_cnn_matched":
            return SketchGroups(
                global_pool=groups.global_pool,
                line_sketch=torch.zeros_like(groups.line_sketch),
                king_zone=torch.zeros_like(groups.king_zone),
                material=groups.material,
                malformed_king_count=groups.malformed_king_count,
            )
        if self.ablation == "flat_head_same_params":
            if self.flat_descriptor is None:
                raise RuntimeError("flat_head_same_params requires a flat descriptor module")
            flat = self.flat_descriptor(h.flatten(1))
            sizes = [
                self.sketch.global_dim,
                self.sketch.line_dim,
                self.sketch.king_dim,
                self.sketch.material_dim,
            ]
            global_pool, line_sketch, king_zone, material_like = flat.split(sizes, dim=1)
            return SketchGroups(
                global_pool=global_pool,
                line_sketch=line_sketch,
                king_zone=king_zone,
                material=material_like + groups.material,
                malformed_king_count=groups.malformed_king_count,
            )
        return groups

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        h = self._field(x)
        groups = self.sketch(
            h,
            x,
            no_line_sketch=self.ablation == "no_line_sketch",
            random_line_basis=self.ablation == "random_line_basis",
            no_king_zone=self.ablation == "no_king_zone",
        )
        groups = self._apply_ablation(groups, h)
        logits_raw, head_diagnostics = self.head(groups)
        logits = _format_logits(logits_raw, self.num_classes)
        batch = x.shape[0]
        param_count = sum(parameter.numel() for parameter in self.parameters())
        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "hidden_field_energy": h.square().mean(dim=(1, 2, 3)),
            "global_descriptor_energy": groups.global_pool.square().mean(dim=1),
            "line_sketch_energy": groups.line_sketch.square().mean(dim=1),
            "king_zone_energy": groups.king_zone.square().mean(dim=1),
            "material_summary_energy": groups.material.square().mean(dim=1),
            "malformed_king_count": groups.malformed_king_count,
            "parameter_count": logits.new_full((batch,), float(param_count)),
            "fp32_size_bytes": logits.new_full((batch,), float(param_count * 4)),
            "simulated_int8_size_bytes": logits.new_full((batch,), float(param_count)),
            "line_sketch_active": logits.new_full((batch,), 0.0 if self.ablation == "no_line_sketch" else 1.0),
            "king_zone_active": logits.new_full((batch,), 0.0 if self.ablation == "no_king_zone" else 1.0),
            "quantization_target_int8": logits.new_full((batch,), 1.0 if self.quantization_target == "int8" else 0.0),
        }
        for key, value in head_diagnostics.items():
            output[key] = value.to(dtype=logits.dtype).expand(batch)
        return output


def build_tiny_chess_micronet_from_config(config: dict[str, Any]) -> TinyChessMicroNet:
    cfg = dict(config)
    cfg.pop("name", None)
    cfg.pop("mechanism_family", None)
    cfg.pop("packet_profile", None)
    cfg.pop("channels", None)
    cfg.pop("hidden_dim", None)
    cfg.pop("depth", None)
    cfg.pop("tier", None)
    cfg.pop("use_batchnorm", None)
    return TinyChessMicroNet(
        input_channels=int(cfg.pop("input_channels", 18)),
        num_classes=int(cfg.pop("num_classes", 1)),
        width=int(cfg.pop("width", 16)),
        squeeze_rank=int(cfg.pop("squeeze_rank", 6)) if cfg.get("squeeze_rank") is not None else None,
        blocks=int(cfg.pop("blocks", 3)),
        mix_rank=int(cfg.pop("mix_rank", 6)) if cfg.get("mix_rank") is not None else None,
        head_hidden=int(cfg.pop("head_hidden", 32)),
        dropout=float(cfg.pop("dropout", 0.1)),
        line_bases=cfg.pop("line_bases", "all_6"),
        king_zone=bool(cfg.pop("king_zone", True)),
        quantization_target=str(cfg.pop("quantization_target", "int8")),
        ablation=str(cfg.pop("ablation", "none")),
    )

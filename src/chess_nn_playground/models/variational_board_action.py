"""Variational Board Action Network for idea i071."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardConvStem, BoardTensorSpec, require_board_tensor


SUMMARY_NAMES = (
    "action_value",
    "potential_energy",
    "gradient_energy",
    "field_energy",
    "stiffness_mean",
    "stiffness_anisotropy",
    "boundary_flux",
    "residual_l1",
    "residual_l2",
    "residual_max",
    "king_zone_residual",
    "occupied_square_residual",
    "empty_square_residual",
    "own_piece_residual",
    "opponent_piece_residual",
    "center_residual",
    "edge_residual",
    "residual_localization",
)
ACTION_SUMMARY_COUNT = 7


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


def _norm_act(width: int, use_batchnorm: bool) -> list[nn.Module]:
    layers: list[nn.Module] = []
    if use_batchnorm:
        layers.append(nn.BatchNorm2d(width))
    layers.append(nn.GELU())
    return layers


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weighted = values * weights
    denom = weights.sum(dim=(1, 2, 3)).clamp_min(1.0) * values.shape[1]
    return weighted.sum(dim=(1, 2, 3)) / denom


@dataclass(frozen=True)
class BoardMasks:
    occupied: torch.Tensor
    empty: torch.Tensor
    own_piece: torch.Tensor
    opponent_piece: torch.Tensor
    king_zone: torch.Tensor
    center: torch.Tensor
    edge: torch.Tensor
    boundary: torch.Tensor


@dataclass(frozen=True)
class FieldBatch:
    fields: torch.Tensor
    context: torch.Tensor


@dataclass(frozen=True)
class VariationalTerms:
    dx: torch.Tensor
    dy: torch.Tensor
    gx: torch.Tensor
    gy: torch.Tensor
    potential_force: torch.Tensor
    potential_density: torch.Tensor
    residual: torch.Tensor
    gradient_density: torch.Tensor
    action_density: torch.Tensor


class BoardMaskBuilder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        rank = torch.arange(8, dtype=torch.float32).view(1, 1, 8, 1)
        file = torch.arange(8, dtype=torch.float32).view(1, 1, 1, 8)
        center = ((rank >= 2) & (rank <= 5) & (file >= 2) & (file <= 5)).to(torch.float32)
        edge = ((rank == 0) | (rank == 7) | (file == 0) | (file == 7)).to(torch.float32)
        boundary = edge.clone()
        self.register_buffer("center_mask", center, persistent=False)
        self.register_buffer("edge_mask", edge, persistent=False)
        self.register_buffer("boundary_mask", boundary, persistent=False)

    def forward(self, x: torch.Tensor) -> BoardMasks:
        batch = x.shape[0]
        dtype = x.dtype
        device = x.device
        if x.shape[1] >= 12:
            white = x[:, :6].clamp(0.0, 1.0)
            black = x[:, 6:12].clamp(0.0, 1.0)
            side = x[:, 12:13].clamp(0.0, 1.0) if x.shape[1] > 12 else x.new_ones(batch, 1, 8, 8)
            occupied = (white.sum(dim=1, keepdim=True) + black.sum(dim=1, keepdim=True)).clamp(0.0, 1.0)
            own_piece = (side * white.sum(dim=1, keepdim=True) + (1.0 - side) * black.sum(dim=1, keepdim=True)).clamp(
                0.0, 1.0
            )
            opponent_piece = (
                side * black.sum(dim=1, keepdim=True) + (1.0 - side) * white.sum(dim=1, keepdim=True)
            ).clamp(0.0, 1.0)
            kings = (x[:, 5:6].clamp(0.0, 1.0) + x[:, 11:12].clamp(0.0, 1.0)).clamp(0.0, 1.0)
            king_zone = F.max_pool2d(kings, kernel_size=3, stride=1, padding=1).clamp(0.0, 1.0)
        else:
            occupied = x.new_zeros(batch, 1, 8, 8)
            own_piece = occupied
            opponent_piece = occupied
            king_zone = occupied
        center = self.center_mask.to(device=device, dtype=dtype).expand(batch, -1, -1, -1)
        edge = self.edge_mask.to(device=device, dtype=dtype).expand(batch, -1, -1, -1)
        boundary = self.boundary_mask.to(device=device, dtype=dtype).expand(batch, -1, -1, -1)
        return BoardMasks(
            occupied=occupied,
            empty=1.0 - occupied,
            own_piece=own_piece,
            opponent_piece=opponent_piece,
            king_zone=king_zone,
            center=center,
            edge=edge,
            boundary=boundary,
        )


class BoardFieldEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        field_channels: int = 12,
        context_width: int = 48,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.context = nn.Sequential(
            nn.Conv2d(input_channels, context_width, kernel_size=1, bias=not use_batchnorm),
            *_norm_act(context_width, use_batchnorm),
            nn.Conv2d(context_width, context_width, kernel_size=3, padding=1, bias=not use_batchnorm),
            *_norm_act(context_width, use_batchnorm),
            nn.Conv2d(context_width, context_width, kernel_size=1, bias=not use_batchnorm),
            *_norm_act(context_width, use_batchnorm),
        )
        self.field_head = nn.Conv2d(context_width, field_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> FieldBatch:
        x = require_board_tensor(x, self.spec)
        context = self.context(x)
        fields = torch.tanh(self.field_head(context))
        return FieldBatch(fields=fields, context=context)


class FiniteDifferenceLayer(nn.Module):
    def __init__(self, field_channels: int, boundary_mode: str = "reflect", random_seed: int = 71071) -> None:
        super().__init__()
        if boundary_mode not in {"reflect", "zero"}:
            raise ValueError("boundary_mode must be 'reflect' or 'zero'")
        self.field_channels = int(field_channels)
        self.boundary_mode = boundary_mode
        generator = torch.Generator().manual_seed(random_seed)
        random_x = torch.randn(self.field_channels, 1, 3, 3, generator=generator)
        random_y = torch.randn(self.field_channels, 1, 3, 3, generator=generator)
        random_x = random_x - random_x.mean(dim=(2, 3), keepdim=True)
        random_y = random_y - random_y.mean(dim=(2, 3), keepdim=True)
        random_x = random_x / random_x.flatten(1).norm(dim=1).view(self.field_channels, 1, 1, 1).clamp_min(1.0e-6)
        random_y = random_y / random_y.flatten(1).norm(dim=1).view(self.field_channels, 1, 1, 1).clamp_min(1.0e-6)
        self.register_buffer("random_x", random_x, persistent=False)
        self.register_buffer("random_y", random_y, persistent=False)

    def dx(self, u: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(u)
        out[:, :, :, :-1] = u[:, :, :, 1:] - u[:, :, :, :-1]
        if self.boundary_mode == "zero":
            out[:, :, :, -1] = -u[:, :, :, -1]
        return out

    def dy(self, u: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(u)
        out[:, :, :-1, :] = u[:, :, 1:, :] - u[:, :, :-1, :]
        if self.boundary_mode == "zero":
            out[:, :, -1, :] = -u[:, :, -1, :]
        return out

    def adjoint_x(self, flux: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(flux)
        active = flux if self.boundary_mode == "zero" else flux[:, :, :, :-1]
        if self.boundary_mode == "zero":
            out[:, :, :, :-1] -= active[:, :, :, :-1]
            out[:, :, :, 1:] += active[:, :, :, :-1]
            out[:, :, :, -1] -= active[:, :, :, -1]
        else:
            out[:, :, :, :-1] -= active
            out[:, :, :, 1:] += active
        return out

    def adjoint_y(self, flux: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(flux)
        active = flux if self.boundary_mode == "zero" else flux[:, :, :-1, :]
        if self.boundary_mode == "zero":
            out[:, :, :-1, :] -= active[:, :, :-1, :]
            out[:, :, 1:, :] += active[:, :, :-1, :]
            out[:, :, -1, :] -= active[:, :, -1, :]
        else:
            out[:, :, :-1, :] -= active
            out[:, :, 1:, :] += active
        return out

    def random_dx(self, u: torch.Tensor) -> torch.Tensor:
        return F.conv2d(u, self.random_x.to(device=u.device, dtype=u.dtype), padding=1, groups=self.field_channels)

    def random_dy(self, u: torch.Tensor) -> torch.Tensor:
        return F.conv2d(u, self.random_y.to(device=u.device, dtype=u.dtype), padding=1, groups=self.field_channels)

    def random_adjoint_x(self, flux: torch.Tensor) -> torch.Tensor:
        weight = self.random_x.flip(-1).flip(-2).to(device=flux.device, dtype=flux.dtype)
        return F.conv2d(flux, weight, padding=1, groups=self.field_channels)

    def random_adjoint_y(self, flux: torch.Tensor) -> torch.Tensor:
        weight = self.random_y.flip(-1).flip(-2).to(device=flux.device, dtype=flux.dtype)
        return F.conv2d(flux, weight, padding=1, groups=self.field_channels)

    def forward(self, u: torch.Tensor, *, random_operators: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if random_operators:
            return self.random_dx(u), self.random_dy(u)
        return self.dx(u), self.dy(u)

    def adjoint(self, flux_x: torch.Tensor, flux_y: torch.Tensor, *, random_operators: bool = False) -> torch.Tensor:
        if random_operators:
            return self.random_adjoint_x(flux_x) + self.random_adjoint_y(flux_y)
        return self.adjoint_x(flux_x) + self.adjoint_y(flux_y)


class LagrangianHeads(nn.Module):
    def __init__(self, field_channels: int, context_width: int, eps: float = 1.0e-4) -> None:
        super().__init__()
        self.eps = float(eps)
        self.gx_head = nn.Conv2d(context_width, field_channels, kernel_size=1)
        self.gy_head = nn.Conv2d(context_width, field_channels, kernel_size=1)
        self.potential_force = nn.Sequential(
            nn.Conv2d(context_width + field_channels, context_width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(context_width, field_channels, kernel_size=1),
        )
        self.potential_density = nn.Sequential(
            nn.Conv2d(context_width + field_channels, context_width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(context_width, field_channels, kernel_size=1),
        )
        self.direct_residual = nn.Sequential(
            nn.Conv2d(context_width, context_width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(context_width, field_channels, kernel_size=1),
        )

    def forward(self, fields: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        joint = torch.cat([fields, context], dim=1)
        gx = F.softplus(self.gx_head(context)) + self.eps
        gy = F.softplus(self.gy_head(context)) + self.eps
        force = self.potential_force(joint)
        density = self.potential_density(joint)
        return gx, gy, force, density

    def direct_force_residual(self, context: torch.Tensor) -> torch.Tensor:
        return self.direct_residual(context)


class EulerLagrangeResidualLayer(nn.Module):
    def __init__(
        self,
        field_channels: int,
        boundary_mode: str = "reflect",
        eps: float = 1.0e-4,
        random_seed: int = 71071,
    ) -> None:
        super().__init__()
        self.field_channels = int(field_channels)
        self.finite_difference = FiniteDifferenceLayer(field_channels, boundary_mode, random_seed=random_seed)
        self.eps = float(eps)

    def forward(
        self,
        fields: torch.Tensor,
        gx: torch.Tensor,
        gy: torch.Tensor,
        potential_force: torch.Tensor,
        potential_density: torch.Tensor,
        *,
        ablation: str = "none",
        direct_residual: torch.Tensor | None = None,
    ) -> VariationalTerms:
        random_ops = ablation == "random_difference_operators"
        dx, dy = self.finite_difference(fields, random_operators=random_ops)
        if ablation == "no_gradient_terms":
            dx = torch.zeros_like(dx)
            dy = torch.zeros_like(dy)
        gradient_density = 0.5 * (gx * dx.square() + gy * dy.square())
        if ablation == "force_head_only" and direct_residual is not None:
            residual = direct_residual
        elif ablation == "harmonic_control":
            ones = torch.ones_like(gx)
            residual = -self.finite_difference.adjoint(ones * dx, ones * dy, random_operators=False)
        else:
            div = self.finite_difference.adjoint(gx * dx, gy * dy, random_operators=random_ops)
            residual = potential_force - div
        action_density = potential_density + gradient_density
        return VariationalTerms(
            dx=dx,
            dy=dy,
            gx=gx,
            gy=gy,
            potential_force=potential_force,
            potential_density=potential_density,
            residual=residual,
            gradient_density=gradient_density,
            action_density=action_density,
        )


class VariationalSummaryHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, fields: torch.Tensor, terms: VariationalTerms, masks: BoardMasks) -> torch.Tensor:
        abs_residual = terms.residual.abs()
        residual_l1 = abs_residual.mean(dim=(1, 2, 3))
        residual_l2 = terms.residual.square().mean(dim=(1, 2, 3))
        residual_max = abs_residual.flatten(1).amax(dim=1)
        residual_localization = residual_max / residual_l1.clamp_min(1.0e-6)
        flux = terms.gx * terms.dx.abs() + terms.gy * terms.dy.abs()
        summary = torch.stack(
            [
                terms.action_density.mean(dim=(1, 2, 3)),
                terms.potential_density.abs().mean(dim=(1, 2, 3)),
                terms.gradient_density.mean(dim=(1, 2, 3)),
                fields.square().mean(dim=(1, 2, 3)),
                0.5 * (terms.gx.mean(dim=(1, 2, 3)) + terms.gy.mean(dim=(1, 2, 3))),
                (terms.gx - terms.gy).abs().mean(dim=(1, 2, 3)),
                _weighted_mean(flux, masks.boundary),
                residual_l1,
                residual_l2,
                residual_max,
                _weighted_mean(abs_residual, masks.king_zone),
                _weighted_mean(abs_residual, masks.occupied),
                _weighted_mean(abs_residual, masks.empty),
                _weighted_mean(abs_residual, masks.own_piece),
                _weighted_mean(abs_residual, masks.opponent_piece),
                _weighted_mean(abs_residual, masks.center),
                _weighted_mean(abs_residual, masks.edge),
                residual_localization,
            ],
            dim=1,
        )
        return summary


class VariationalBoardActionNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        field_channels: int = 12,
        context_width: int = 48,
        channels: int = 64,
        depth: int = 2,
        hidden_dim: int = 96,
        residual_map_width: int = 32,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        boundary_mode: str = "reflect",
        include_residual_map_cnn: bool = True,
        include_board_cnn_summary: bool = True,
        use_force_head_approximation: bool = True,
        eps: float = 1.0e-4,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if field_channels < 1:
            raise ValueError("field_channels must be positive")
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.field_channels = int(field_channels)
        self.context_width = int(context_width)
        self.ablation = str(ablation)
        self.include_residual_map_cnn = bool(include_residual_map_cnn)
        self.include_board_cnn_summary = bool(include_board_cnn_summary)
        self.use_force_head_approximation = bool(use_force_head_approximation)
        self.residual_map_width = int(residual_map_width)
        self.mask_builder = BoardMaskBuilder()
        self.field_encoder = BoardFieldEncoder(input_channels, field_channels, context_width, use_batchnorm)
        self.lagrangian_heads = LagrangianHeads(field_channels, context_width, eps)
        self.residual_layer = EulerLagrangeResidualLayer(field_channels, boundary_mode, eps)
        self.summary_builder = VariationalSummaryHead()
        self.residual_map_encoder = nn.Sequential(
            nn.Conv2d(field_channels, residual_map_width, kernel_size=3, padding=1, bias=not use_batchnorm),
            *_norm_act(residual_map_width, use_batchnorm),
            nn.Conv2d(residual_map_width, residual_map_width, kernel_size=3, padding=1, bias=not use_batchnorm),
            *_norm_act(residual_map_width, use_batchnorm),
        )
        self.board_cnn = BoardConvStem(input_channels, channels=channels, depth=depth, use_batchnorm=use_batchnorm)
        summary_dim = len(SUMMARY_NAMES)
        residual_dim = residual_map_width if include_residual_map_cnn else 0
        board_dim = channels * 2 if include_board_cnn_summary else 0
        head_in = summary_dim + residual_dim + board_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_classes),
        )

    @property
    def finite_difference(self) -> FiniteDifferenceLayer:
        return self.residual_layer.finite_difference

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, BoardTensorSpec(self.input_channels))
        masks = self.mask_builder(x)
        fields_batch = self.field_encoder(x)
        gx, gy, force, density = self.lagrangian_heads(fields_batch.fields, fields_batch.context)
        direct_residual = self.lagrangian_heads.direct_force_residual(fields_batch.context)
        terms = self.residual_layer(
            fields_batch.fields,
            gx,
            gy,
            force,
            density,
            ablation=self.ablation,
            direct_residual=direct_residual,
        )
        summary_raw = self.summary_builder(fields_batch.fields, terms, masks)
        summary = self.summary_for_head(summary_raw)
        pieces: list[torch.Tensor] = [summary]

        if self.include_residual_map_cnn:
            if self.ablation in {"action_only", "residual_norm_only", "cnn_only_matched"}:
                residual_vec = summary.new_zeros(summary.shape[0], self.residual_map_width)
            else:
                residual_map = self.residual_map_encoder(terms.residual)
                residual_vec = F.adaptive_avg_pool2d(residual_map, (1, 1)).flatten(1)
            pieces.append(residual_vec)

        if self.include_board_cnn_summary:
            board_map = self.board_cnn(x)
            board_vec = torch.cat(
                [
                    F.adaptive_avg_pool2d(board_map, (1, 1)).flatten(1),
                    F.adaptive_max_pool2d(board_map, (1, 1)).flatten(1),
                ],
                dim=1,
            )
            pieces.append(board_vec)
        else:
            board_map = x.new_zeros(x.shape[0], 1, 8, 8)

        features = torch.cat(pieces, dim=1)
        logits = self.classifier(features)
        output = {name: value for name, value in zip(SUMMARY_NAMES, summary_raw.unbind(dim=1))}
        output.update(
            {
                "logits": _format_logits(logits, self.num_classes),
                "residual_map_energy": terms.residual.square().mean(dim=(1, 2, 3)),
                "dx_energy": terms.dx.square().mean(dim=(1, 2, 3)),
                "dy_energy": terms.dy.square().mean(dim=(1, 2, 3)),
                "force_head_approximation": torch.full(
                    (x.shape[0],), float(self.use_force_head_approximation), device=x.device, dtype=x.dtype
                ),
                "board_cnn_energy": board_map.square().mean(dim=(1, 2, 3)),
            }
        )
        return output

    def summary_for_head(self, summary: torch.Tensor) -> torch.Tensor:
        if self.ablation == "cnn_only_matched":
            return torch.zeros_like(summary)
        if self.ablation == "action_only":
            out = torch.zeros_like(summary)
            out[:, :ACTION_SUMMARY_COUNT] = summary[:, :ACTION_SUMMARY_COUNT]
            return out
        return summary


def build_variational_board_action_network_from_config(config: dict[str, Any]) -> VariationalBoardActionNetwork:
    return VariationalBoardActionNetwork(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        field_channels=int(config.get("field_channels", 12)),
        context_width=int(config.get("context_width", config.get("channels", 48))),
        channels=int(config.get("channels", 64)),
        depth=int(config.get("depth", 2)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        residual_map_width=int(config.get("residual_map_width", max(16, int(config.get("hidden_dim", 96)) // 3))),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        boundary_mode=str(config.get("boundary_mode", "reflect")),
        include_residual_map_cnn=bool(config.get("include_residual_map_cnn", True)),
        include_board_cnn_summary=bool(config.get("include_board_cnn_summary", True)),
        use_force_head_approximation=bool(config.get("use_force_head_approximation", True)),
        eps=float(config.get("eps", 1.0e-4)),
        ablation=str(config.get("ablation", "none")),
    )

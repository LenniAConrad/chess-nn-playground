"""Multiplicative Conjunction ConvNet for idea i161."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


@dataclass(frozen=True)
class MultiplicativeConjunctionConfig:
    input_channels: int = 18
    num_classes: int = 1
    width: int = 64
    depth: int = 5
    branch_width: int = 32
    hidden_dim: int = 128
    dropout: float = 0.1
    use_batchnorm: bool = True
    use_coordinate_planes: bool = True
    ablation: str = "none"


class CoordinatePlaneAppender(nn.Module):
    """Appends fixed rank/file coordinate planes to an 8x8 board tensor."""

    def __init__(self) -> None:
        super().__init__()
        coords = torch.linspace(-1.0, 1.0, 8)
        rank = coords.view(1, 1, 8, 1).expand(1, 1, 8, 8)
        file = coords.view(1, 1, 1, 8).expand(1, 1, 8, 8)
        self.register_buffer("coordinate_planes", torch.cat([rank, file], dim=1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        planes = self.coordinate_planes.to(dtype=x.dtype, device=x.device).expand(x.shape[0], -1, -1, -1)
        return torch.cat([x, planes], dim=1)


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.GELU())
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MultiplicativeConjunctionBlock(nn.Module):
    """Residual block with paired branch factors and an explicit product feature."""

    PRODUCT_MODES = {"none", "additive_only", "gate_only_no_product", "late_product_only"}

    def __init__(
        self,
        width: int,
        branch_width: int,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        product_mode: str = "none",
    ) -> None:
        super().__init__()
        if branch_width < 1:
            raise ValueError("branch_width must be positive")
        if product_mode not in self.PRODUCT_MODES:
            raise ValueError(f"Unknown product mode: {product_mode}")
        self.product_mode = product_mode
        self.branch_a = nn.Conv2d(width, branch_width, kernel_size=3, padding=1)
        self.branch_b = nn.Conv2d(width, branch_width, kernel_size=3, padding=1)
        self.gate = nn.Conv2d(width, branch_width, kernel_size=1)
        self.fusion = nn.Conv2d(branch_width * 4, width, kernel_size=1, bias=not use_batchnorm)
        self.norm = nn.BatchNorm2d(width) if use_batchnorm else nn.Identity()
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        a = self.branch_a(h)
        b = self.branch_b(h)
        gate = torch.sigmoid(self.gate(h))
        raw_product = a * b
        if self.product_mode == "additive_only":
            product_feature = a + b
        elif self.product_mode in {"gate_only_no_product", "late_product_only"}:
            product_feature = torch.zeros_like(raw_product)
        else:
            product_feature = raw_product

        gated_a = gate * a
        fused = self.fusion(torch.cat([a, b, product_feature, gated_a], dim=1))
        fused = self.dropout(self.activation(self.norm(fused)))
        out = h + fused
        diagnostics = {
            "product_feature": product_feature,
            "raw_product": raw_product,
            "gate": gate,
            "branch_a": a,
            "branch_b": b,
            "fused": fused,
        }
        return out, diagnostics


class PlainResidualBlock(nn.Module):
    """Plain CNN residual block used for capacity-control ablations."""

    def __init__(self, width: int, dropout: float = 0.1, use_batchnorm: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=not use_batchnorm),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(width))
        layers.append(nn.GELU())
        layers.append(nn.Conv2d(width, width, kernel_size=3, padding=1, bias=not use_batchnorm))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(width))
        self.body = nn.Sequential(*layers)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        fused = self.dropout(self.activation(self.body(h)))
        out = h + fused
        zeros = h.new_zeros(h.shape[0])
        diagnostics = {
            "product_feature_norm": zeros,
            "raw_product_norm": zeros,
            "gate_mean": zeros,
            "gate_saturation": zeros,
            "branch_a_norm": zeros,
            "branch_b_norm": zeros,
            "fusion_norm": fused.square().mean(dim=(1, 2, 3)),
        }
        return out, diagnostics


class MultiplicativeConjunctionConvNet(nn.Module):
    """CNN classifier with explicit local multiplicative conjunction channels."""

    VALID_ABLATIONS = {
        "none",
        "additive_only",
        "gate_only_no_product",
        "single_branch_matched",
        "late_product_only",
        "cnn_matched_params",
    }

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        width: int = 64,
        depth: int = 5,
        branch_width: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        use_coordinate_planes: bool = True,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if width < 1:
            raise ValueError("width must be positive")
        if depth < 1:
            raise ValueError("depth must be positive")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if ablation not in self.VALID_ABLATIONS:
            raise ValueError(f"Unknown MultiplicativeConjunctionConvNet ablation: {ablation}")

        self.config = MultiplicativeConjunctionConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            width=width,
            depth=depth,
            branch_width=branch_width,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            use_coordinate_planes=use_coordinate_planes,
            ablation=ablation,
        )
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.ablation = ablation
        self.coordinate_appender = CoordinatePlaneAppender() if use_coordinate_planes else nn.Identity()
        stem_input_channels = input_channels + (2 if use_coordinate_planes else 0)
        self.stem = nn.Sequential(
            ConvNormAct(stem_input_channels, width, use_batchnorm=use_batchnorm),
            ConvNormAct(width, width, use_batchnorm=use_batchnorm),
        )

        product_mode = ablation if ablation in {"additive_only", "gate_only_no_product", "late_product_only"} else "none"
        plain_blocks = ablation in {"single_branch_matched", "cnn_matched_params"}
        blocks: list[nn.Module] = []
        for _ in range(depth):
            if plain_blocks:
                blocks.append(PlainResidualBlock(width, dropout=dropout, use_batchnorm=use_batchnorm))
            else:
                blocks.append(
                    MultiplicativeConjunctionBlock(
                        width=width,
                        branch_width=branch_width,
                        dropout=dropout,
                        use_batchnorm=use_batchnorm,
                        product_mode=product_mode,
                    )
                )
        self.blocks = nn.ModuleList(blocks)

        aggregate_dim = width * 2
        if ablation == "late_product_only":
            aggregate_dim += width // 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(aggregate_dim),
            nn.Linear(aggregate_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, max(32, hidden_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(max(32, hidden_dim // 2), num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        h = self.stem(self.coordinate_appender(board))

        product_norms: list[torch.Tensor] = []
        raw_product_norms: list[torch.Tensor] = []
        gate_means: list[torch.Tensor] = []
        gate_saturations: list[torch.Tensor] = []
        branch_a_norms: list[torch.Tensor] = []
        branch_b_norms: list[torch.Tensor] = []
        fusion_norms: list[torch.Tensor] = []

        for block in self.blocks:
            h, diagnostics = block(h)
            if "product_feature" in diagnostics:
                product_norms.append(diagnostics["product_feature"].square().mean(dim=(1, 2, 3)))
                raw_product_norms.append(diagnostics["raw_product"].square().mean(dim=(1, 2, 3)))
                gate = diagnostics["gate"]
                gate_means.append(gate.mean(dim=(1, 2, 3)))
                gate_saturations.append((2.0 * (gate - 0.5).abs()).mean(dim=(1, 2, 3)))
                branch_a_norms.append(diagnostics["branch_a"].square().mean(dim=(1, 2, 3)))
                branch_b_norms.append(diagnostics["branch_b"].square().mean(dim=(1, 2, 3)))
                fusion_norms.append(diagnostics["fused"].square().mean(dim=(1, 2, 3)))
            else:
                product_norms.append(diagnostics["product_feature_norm"])
                raw_product_norms.append(diagnostics["raw_product_norm"])
                gate_means.append(diagnostics["gate_mean"])
                gate_saturations.append(diagnostics["gate_saturation"])
                branch_a_norms.append(diagnostics["branch_a_norm"])
                branch_b_norms.append(diagnostics["branch_b_norm"])
                fusion_norms.append(diagnostics["fusion_norm"])

        aggregate_parts = [h.mean(dim=(2, 3)), h.amax(dim=(2, 3))]
        if self.ablation == "late_product_only":
            aggregate_parts.append(self._late_product_pool(h))
        aggregate = torch.cat(aggregate_parts, dim=1)
        raw_logits = self.classifier(aggregate)
        logits = _format_logits(raw_logits, self.num_classes)

        product_by_layer = torch.stack(product_norms, dim=1)
        raw_product_by_layer = torch.stack(raw_product_norms, dim=1)
        gate_mean_by_layer = torch.stack(gate_means, dim=1)
        gate_saturation_by_layer = torch.stack(gate_saturations, dim=1)
        branch_a_by_layer = torch.stack(branch_a_norms, dim=1)
        branch_b_by_layer = torch.stack(branch_b_norms, dim=1)
        fusion_by_layer = torch.stack(fusion_norms, dim=1)
        output = {
            "logits": logits,
            "product_norm_by_layer": product_by_layer,
            "raw_product_norm_by_layer": raw_product_by_layer,
            "gate_mean_by_layer": gate_mean_by_layer,
            "gate_saturation_by_layer": gate_saturation_by_layer,
            "product_branch_norm": product_by_layer.mean(dim=1),
            "raw_product_branch_norm": raw_product_by_layer.mean(dim=1),
            "gate_mean": gate_mean_by_layer.mean(dim=1),
            "gate_saturation": gate_saturation_by_layer.mean(dim=1),
            "branch_balance": (branch_a_by_layer.mean(dim=1) - branch_b_by_layer.mean(dim=1)).abs(),
            "fusion_energy": fusion_by_layer.mean(dim=1),
            "feature_energy": h.square().mean(dim=(1, 2, 3)),
            "aggregate_feature_energy": aggregate.square().mean(dim=1),
        }
        if self.num_classes == 1:
            output["prob"] = torch.sigmoid(logits)
        return output

    def _late_product_pool(self, h: torch.Tensor) -> torch.Tensor:
        left, right = torch.chunk(h, chunks=2, dim=1)
        channels = min(left.shape[1], right.shape[1])
        return (left[:, :channels] * right[:, :channels]).mean(dim=(2, 3))


def build_multiplicative_conjunction_convnet_from_config(
    config: dict[str, Any],
) -> MultiplicativeConjunctionConvNet:
    width = int(config.get("width", config.get("channels", 64)))
    return MultiplicativeConjunctionConvNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        width=width,
        depth=int(config.get("depth", 5)),
        branch_width=int(config.get("branch_width", max(1, width // 2))),
        hidden_dim=int(config.get("hidden_dim", 128)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        use_coordinate_planes=bool(config.get("use_coordinate_planes", True)),
        ablation=str(config.get("ablation", "none")),
    )

"""Material-Phase Low-Rank Adapter Network (idea i130)."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.data.board_features import SIMPLE_18
from chess_nn_playground.models.idea_blocks import BoardConvStem, BoardTensorSpec, require_board_tensor


_PIECE_VALUES = (1.0, 3.0, 3.0, 5.0, 9.0, 0.0)


class MaterialPhaseSummary(nn.Module):
    """Compute a permutation-free material/phase summary from the simple_18 tensor."""

    summary_dim = 23

    def __init__(self) -> None:
        super().__init__()
        values = torch.tensor(_PIECE_VALUES, dtype=torch.float32)
        self.register_buffer("piece_values", values, persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        own_counts = x[:, 0:6].sum(dim=(2, 3))
        opp_counts = x[:, 6:12].sum(dim=(2, 3))
        side_to_move = x[:, 12, :1, :1].squeeze(-1).squeeze(-1)
        white_kingside = x[:, 13, 0, 0]
        white_queenside = x[:, 14, 0, 0]
        black_kingside = x[:, 15, 0, 0]
        black_queenside = x[:, 16, 0, 0]
        en_passant_active = x[:, 17].sum(dim=(1, 2)).clamp_max(1.0)

        total_pieces = own_counts.sum(dim=1) + opp_counts.sum(dim=1)
        own_material = (own_counts * self.piece_values).sum(dim=1)
        opp_material = (opp_counts * self.piece_values).sum(dim=1)
        material_signed = own_material - opp_material
        major = own_counts[:, 3] + own_counts[:, 4] + opp_counts[:, 3] + opp_counts[:, 4]
        minor = own_counts[:, 1] + own_counts[:, 2] + opp_counts[:, 1] + opp_counts[:, 2]
        pawns_total = own_counts[:, 0] + opp_counts[:, 0]
        # Smooth phase coordinate in [0, 1]; 1.0 ~ opening, 0.0 ~ endgame.
        phase = ((major * 4.0 + minor * 2.0 + pawns_total) / 60.0).clamp(0.0, 1.0)

        scaled_own = own_counts / 8.0
        scaled_opp = opp_counts / 8.0

        summary = torch.cat(
            [
                scaled_own,
                scaled_opp,
                (side_to_move - 0.5).unsqueeze(-1),
                (material_signed / 39.0).unsqueeze(-1),
                ((own_material + opp_material) / 78.0).unsqueeze(-1),
                (total_pieces / 32.0).unsqueeze(-1),
                phase.unsqueeze(-1),
                white_kingside.unsqueeze(-1),
                white_queenside.unsqueeze(-1),
                black_kingside.unsqueeze(-1),
                black_queenside.unsqueeze(-1),
                en_passant_active.unsqueeze(-1),
                (1.0 - phase).unsqueeze(-1),
            ],
            dim=1,
        )

        diagnostics = {
            "side_to_move": side_to_move,
            "material_signed": material_signed,
            "own_material": own_material,
            "opponent_material": opp_material,
            "own_pawn_count": own_counts[:, 0],
            "opponent_pawn_count": opp_counts[:, 0],
            "own_minor_count": own_counts[:, 1] + own_counts[:, 2],
            "opponent_minor_count": opp_counts[:, 1] + opp_counts[:, 2],
            "own_major_count": own_counts[:, 3] + own_counts[:, 4],
            "opponent_major_count": opp_counts[:, 3] + opp_counts[:, 4],
            "phase": phase,
            "endgame_score": 1.0 - phase,
            "total_piece_count": total_pieces,
            "castling_available": (
                white_kingside + white_queenside + black_kingside + black_queenside
            ),
            "en_passant_active": en_passant_active,
        }
        return summary, diagnostics


class LowRankAdaptedLinear(nn.Module):
    """Shared weight ``Linear`` augmented by a per-sample low-rank LoRA-style update.

    The shared weight ``W`` and bias ``b`` are not conditioned on input. The update
    ``Delta W(s) = B(s) A(s)`` is generated from the material/phase summary and
    has rank ``r``, so the effective transform is

    ``y = W h + b + scale * (B(s) (A(s) h))``.

    ``A(s)`` and ``B(s)`` are produced by linear generators from the phase
    embedding. ``A(s)`` is initialized so the network starts as the shared
    backbone and the adapter contributes only after training.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        phase_dim: int,
        rank: int,
        scale: float | None = None,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scale = float(scale) if scale is not None else 1.0 / float(rank)
        self.shared = nn.Linear(in_features, out_features)
        self.a_generator = nn.Linear(phase_dim, rank * in_features)
        self.b_generator = nn.Linear(phase_dim, out_features * rank)
        nn.init.zeros_(self.b_generator.weight)
        nn.init.zeros_(self.b_generator.bias)

    def forward(
        self, hidden: torch.Tensor, phase: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared_out = self.shared(hidden)
        a_flat = self.a_generator(phase).view(-1, self.rank, self.in_features)
        b_flat = self.b_generator(phase).view(-1, self.out_features, self.rank)
        compressed = torch.bmm(a_flat, hidden.unsqueeze(-1))
        delta = torch.bmm(b_flat, compressed).squeeze(-1)
        adapter_output = self.scale * delta
        return shared_out + adapter_output, adapter_output


class MaterialPhaseLowRankAdapterNetwork(nn.Module):
    """Shared CNN backbone with a stack of material/phase-conditioned low-rank adapter blocks."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        adapter_rank: int = 4,
        adapter_blocks: int | None = None,
        phase_embed_dim: int = 32,
        adapter_scale: float | None = None,
        encoding_adapter: str = SIMPLE_18,
    ) -> None:
        super().__init__()
        if input_channels != 18 or encoding_adapter != SIMPLE_18:
            raise ValueError(
                "MaterialPhaseLowRankAdapterNetwork currently supports simple_18 with 18 input channels"
            )
        if num_classes != 1:
            raise ValueError(
                "MaterialPhaseLowRankAdapterNetwork supports the puzzle_binary one-logit contract"
            )
        if adapter_rank < 1:
            raise ValueError("adapter_rank must be >= 1")

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.adapter_rank = adapter_rank
        self.adapter_scale = (
            float(adapter_scale) if adapter_scale is not None else 1.0 / float(adapter_rank)
        )
        num_adapter_blocks = int(adapter_blocks) if adapter_blocks is not None else max(1, depth)

        self.backbone = BoardConvStem(
            input_channels=input_channels,
            channels=channels,
            depth=max(1, depth),
            use_batchnorm=use_batchnorm,
        )
        self.summary = MaterialPhaseSummary()
        self.phase_encoder = nn.Sequential(
            nn.Linear(self.summary.summary_dim, phase_embed_dim),
            nn.GELU(),
            nn.Linear(phase_embed_dim, phase_embed_dim),
            nn.GELU(),
        )

        backbone_pool_dim = channels * 2  # mean + max pooled
        self.input_projection = nn.Linear(backbone_pool_dim, hidden_dim)

        self.adapter_blocks = nn.ModuleList(
            [
                LowRankAdaptedLinear(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    phase_dim=phase_embed_dim,
                    rank=adapter_rank,
                    scale=self.adapter_scale,
                )
                for _ in range(num_adapter_blocks)
            ]
        )
        self.block_activation = nn.GELU()
        self.block_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.block_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_adapter_blocks)])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + self.summary.summary_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        backbone_features = self.backbone(x)
        pooled = torch.cat(
            [backbone_features.mean(dim=(2, 3)), backbone_features.amax(dim=(2, 3))], dim=1
        )
        backbone_norm = backbone_features.flatten(1).norm(dim=1)

        summary, diagnostics = self.summary(x)
        phase = self.phase_encoder(summary)
        hidden = self.input_projection(pooled)
        adapter_norms: list[torch.Tensor] = []
        for block, norm in zip(self.adapter_blocks, self.block_norms):
            updated, adapter_delta = block(hidden, phase)
            adapter_norms.append(adapter_delta.norm(dim=1))
            updated = norm(updated)
            updated = self.block_activation(updated)
            updated = self.block_dropout(updated)
            hidden = hidden + updated

        logits = self.classifier(torch.cat([hidden, summary], dim=1)).squeeze(-1)

        adapter_norm_tensor = torch.stack(adapter_norms, dim=1)
        mean_adapter_norm = adapter_norm_tensor.mean(dim=1)
        max_adapter_norm = adapter_norm_tensor.amax(dim=1)

        diagnostics_out: dict[str, torch.Tensor] = {
            "logits": logits,
            "mean_adapter_norm": mean_adapter_norm,
            "max_adapter_norm": max_adapter_norm,
            "backbone_feature_norm": backbone_norm,
            "phase_summary_norm": summary.norm(dim=1),
        }
        for index, value in enumerate(adapter_norms):
            diagnostics_out[f"adapter_block_{index}_norm"] = value
        diagnostics_out.update(diagnostics)
        return diagnostics_out


def build_material_phase_low_rank_adapter_network_from_config(
    config: dict[str, Any],
) -> MaterialPhaseLowRankAdapterNetwork:
    return MaterialPhaseLowRankAdapterNetwork(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        depth=int(config.get("depth", 2)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        adapter_rank=int(config.get("adapter_rank", 4)),
        adapter_blocks=int(config["adapter_blocks"]) if "adapter_blocks" in config else None,
        phase_embed_dim=int(config.get("phase_embed_dim", 32)),
        adapter_scale=float(config["adapter_scale"]) if "adapter_scale" in config else None,
        encoding_adapter=str(config.get("encoding_adapter", config.get("encoding", SIMPLE_18))),
    )

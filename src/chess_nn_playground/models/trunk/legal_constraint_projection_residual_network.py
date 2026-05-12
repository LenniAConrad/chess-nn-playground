"""Legal-Constraint Projection Residual Network (idea i134).

Bespoke implementation of the projection-residual architecture promoted from
``ideas/research/packets/classic/chess_nn_research_2026-04-24_2136_friday_shanghai_architecture_batch_7.md``
(candidate 4).

A small CNN trunk turns the simple_18 board tensor into a soft latent belief map
``Y`` over 13 per-square classes (12 piece classes plus an "empty" class). A
differentiable alternating projection pushes ``Y`` onto a soft set of basic
current-board legality constraints:

* per-square simplex (non-negative probabilities, sum to 1);
* per-piece-class count caps (kings <= 1, queens <= 9, rooks/bishops/knights
  <= 10, pawns <= 8 per colour);
* king-count normalisation (own/opponent king total expectation = 1);
* pawn-rank masking (pawns cannot sit on the first/eighth rank).

The projection residual ``R = Y - Y_proj`` exposes how strongly the latent
belief decoder violated each constraint family. Channel-wise residual norms,
per-constraint diagnostic energies, and a tiny residual-map CNN summary are
fused with the encoder summary and classified into one ``puzzle_binary``
logit.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from chess_nn_playground.data.board_features import SIMPLE_18
from chess_nn_playground.models.trunk.idea_blocks import BoardConvStem, BoardTensorSpec, require_board_tensor


_PIECE_CLASSES = ("P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k")
_NUM_PIECE_CLASSES = len(_PIECE_CLASSES)
_NUM_BELIEF_CLASSES = _NUM_PIECE_CLASSES + 1  # +1 for explicit empty class
_EMPTY_CLASS_INDEX = _NUM_PIECE_CLASSES

# Maximum count caps per piece class. Promotions can in principle push minor
# and major counts above the starting count, so we use the standard
# promotion-aware upper bounds (max 8 promotions per side).
_PIECE_CAPS = (
    8.0,   # P pawns
    10.0,  # N knights (2 + up to 8 promoted)
    10.0,  # B bishops
    10.0,  # R rooks
    9.0,   # Q queens (1 + up to 8 promoted)
    1.0,   # K kings
    8.0,   # p
    10.0,  # n
    10.0,  # b
    10.0,  # r
    9.0,   # q
    1.0,   # k
)

_KING_INDICES = (5, 11)  # K, k
_PAWN_RANK_FORBIDDEN = {0: (0, 7), 6: (0, 7)}  # white pawn / black pawn forbidden ranks


def _build_pawn_rank_mask() -> torch.Tensor:
    """Mask is 0 on forbidden (class, rank) entries and 1 elsewhere."""
    mask = torch.ones(_NUM_BELIEF_CLASSES, 8)
    for class_idx, ranks in _PAWN_RANK_FORBIDDEN.items():
        for r in ranks:
            mask[class_idx, r] = 0.0
    return mask


class LegalConstraintProjectionResidualNetwork(nn.Module):
    """Bespoke implementation of the legal-constraint projection residual idea."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        projection_iters: int = 3,
        residual_pool_channels: int = 16,
        stop_gradient_projection: bool = True,
        encoding_adapter: str = SIMPLE_18,
    ) -> None:
        super().__init__()
        if input_channels != 18 or encoding_adapter != SIMPLE_18:
            raise ValueError(
                "LegalConstraintProjectionResidualNetwork currently supports simple_18 with 18 input channels"
            )
        if num_classes != 1:
            raise ValueError(
                "LegalConstraintProjectionResidualNetwork supports the puzzle_binary one-logit contract"
            )
        if projection_iters < 1:
            raise ValueError("projection_iters must be >= 1")

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.projection_iters = int(projection_iters)
        self.stop_gradient_projection = bool(stop_gradient_projection)

        self.backbone = BoardConvStem(
            input_channels=input_channels,
            channels=channels,
            depth=max(1, depth),
            use_batchnorm=use_batchnorm,
        )
        self.belief_head = nn.Conv2d(channels, _NUM_BELIEF_CLASSES, kernel_size=1)

        # Constants used inside the projection. Buffers so they move with the
        # module (and survive .to(device)).
        self.register_buffer(
            "_piece_caps",
            torch.tensor(_PIECE_CAPS, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_pawn_rank_mask",
            _build_pawn_rank_mask(),
            persistent=False,
        )
        self.register_buffer(
            "_king_indices",
            torch.tensor(_KING_INDICES, dtype=torch.long),
            persistent=False,
        )

        # Tiny CNN over the spatial residual magnitude map: input is
        # (B, _NUM_BELIEF_CLASSES, 8, 8), output is (B, residual_pool_channels).
        residual_pool_channels = int(residual_pool_channels)
        self.residual_map_cnn = nn.Sequential(
            nn.Conv2d(_NUM_BELIEF_CLASSES, residual_pool_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(residual_pool_channels, residual_pool_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # Diagnostic feature dimensions:
        # - per-class residual norms: _NUM_BELIEF_CLASSES
        # - per-constraint residual energies (simplex, piece_count, king_count, pawn_rank): 4
        # - residual map summary: residual_pool_channels
        # - encoder summary (mean + max pool): 2 * channels
        # - belief entropy and total residual norm: 2
        diagnostic_dim = _NUM_BELIEF_CLASSES + 4 + residual_pool_channels + 2 * channels + 2

        head_layers: list[nn.Module] = [nn.Linear(diagnostic_dim, hidden_dim), nn.GELU()]
        if dropout > 0:
            head_layers.append(nn.Dropout(dropout))
        head_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        if dropout > 0:
            head_layers.append(nn.Dropout(dropout))
        head_layers.append(nn.Linear(hidden_dim, 1))
        self.classifier = nn.Sequential(*head_layers)

    # ------------------------------------------------------------------
    # Projection primitives
    # ------------------------------------------------------------------

    def _project_piece_count(self, probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cap the expected count of every piece class and dump excess into empty."""
        # probs: (B, 13, 8, 8). Sum the 12 piece classes only; do not touch empty.
        piece_probs = probs[:, :_NUM_PIECE_CLASSES]
        counts = piece_probs.sum(dim=(2, 3))  # (B, 12)
        caps = self._piece_caps.unsqueeze(0)  # (1, 12)
        scale = torch.where(
            counts > caps,
            caps / counts.clamp_min(1.0e-6),
            torch.ones_like(counts),
        )
        scale_4d = scale.unsqueeze(-1).unsqueeze(-1)  # (B, 12, 1, 1)
        scaled_pieces = piece_probs * scale_4d
        excess_per_square = (piece_probs - scaled_pieces).sum(dim=1, keepdim=True)  # (B, 1, 8, 8)
        empty = probs[:, _EMPTY_CLASS_INDEX:_EMPTY_CLASS_INDEX + 1] + excess_per_square
        residual = (piece_probs - scaled_pieces).pow(2).flatten(1).sum(dim=1)  # (B,)
        new_probs = torch.cat([scaled_pieces, empty], dim=1)
        return new_probs, residual

    def _project_king_count(self, probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Force expected king total per side to 1 (or scale down if > 1)."""
        residual = torch.zeros(probs.shape[0], device=probs.device, dtype=probs.dtype)
        out = probs
        for king_idx in _KING_INDICES:
            king_plane = out[:, king_idx]
            king_total = king_plane.sum(dim=(1, 2))  # (B,)
            scale = torch.where(
                king_total > 1.0,
                1.0 / king_total.clamp_min(1.0e-6),
                torch.ones_like(king_total),
            )
            scaled = king_plane * scale.unsqueeze(-1).unsqueeze(-1)
            excess_per_square = king_plane - scaled  # (B, 8, 8)
            residual = residual + excess_per_square.pow(2).flatten(1).sum(dim=1)
            empty = out[:, _EMPTY_CLASS_INDEX] + excess_per_square
            new_planes = list(torch.unbind(out, dim=1))
            new_planes[king_idx] = scaled
            new_planes[_EMPTY_CLASS_INDEX] = empty
            out = torch.stack(new_planes, dim=1)
        return out, residual

    def _project_pawn_rank(self, probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero pawn probability on the impossible first/eighth ranks."""
        # _pawn_rank_mask: (13, 8) — apply along the rank axis (dim=2).
        mask = self._pawn_rank_mask.unsqueeze(0).unsqueeze(-1)  # (1, 13, 8, 1)
        masked = probs * mask
        residual_tensor = probs - masked
        residual = residual_tensor.pow(2).flatten(1).sum(dim=1)
        # The probability that was zeroed must go somewhere; redistribute it
        # back into empty so the simplex-projection pass below is gentle.
        excess_per_square = residual_tensor.sum(dim=1, keepdim=True)
        masked = masked.clone()
        masked[:, _EMPTY_CLASS_INDEX:_EMPTY_CLASS_INDEX + 1] = (
            masked[:, _EMPTY_CLASS_INDEX:_EMPTY_CLASS_INDEX + 1] + excess_per_square
        )
        return masked, residual

    def _project_simplex(self, probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Renormalise per-square so the 13 classes form a probability simplex."""
        # Clamp to non-negative first.
        non_negative = probs.clamp_min(0.0)
        residual_neg = (probs - non_negative).pow(2).flatten(1).sum(dim=1)
        totals = non_negative.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        renormalised = non_negative / totals
        residual_norm = (non_negative - renormalised).pow(2).flatten(1).sum(dim=1)
        return renormalised, residual_neg + residual_norm

    # ------------------------------------------------------------------
    # Full projection
    # ------------------------------------------------------------------

    def _project_legal(self, probs: torch.Tensor) -> dict[str, torch.Tensor]:
        residual_simplex = torch.zeros(probs.shape[0], device=probs.device, dtype=probs.dtype)
        residual_piece = torch.zeros_like(residual_simplex)
        residual_king = torch.zeros_like(residual_simplex)
        residual_pawn = torch.zeros_like(residual_simplex)

        current = probs
        for _ in range(self.projection_iters):
            current, r_piece = self._project_piece_count(current)
            residual_piece = residual_piece + r_piece
            current, r_king = self._project_king_count(current)
            residual_king = residual_king + r_king
            current, r_pawn = self._project_pawn_rank(current)
            residual_pawn = residual_pawn + r_pawn
            current, r_simplex = self._project_simplex(current)
            residual_simplex = residual_simplex + r_simplex

        return {
            "projected": current,
            "residual_simplex": residual_simplex,
            "residual_piece_count": residual_piece,
            "residual_king_count": residual_king,
            "residual_pawn_rank": residual_pawn,
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        backbone = self.backbone(x)
        logits_belief = self.belief_head(backbone)  # (B, 13, 8, 8)
        beliefs = F.softmax(logits_belief, dim=1)

        projection_input = beliefs.detach() if self.stop_gradient_projection else beliefs
        projection = self._project_legal(projection_input)
        projected = projection["projected"]
        residual = beliefs - projected

        # Per-class residual norms across all squares — these summarise which
        # belief planes drift hardest from the legal-board constraint set.
        per_class_residual_norm = residual.flatten(2).norm(dim=2)  # (B, 13)

        # Map summary: tiny CNN over signed residual planes.
        residual_map_summary = self.residual_map_cnn(residual)

        # Encoder summary: mean + max pool of the backbone features.
        encoder_summary = torch.cat(
            [backbone.mean(dim=(2, 3)), backbone.amax(dim=(2, 3))], dim=1
        )

        residual_total_norm = residual.flatten(1).norm(dim=1)
        belief_entropy = -(beliefs.clamp_min(1.0e-9).log() * beliefs).sum(dim=1).mean(dim=(1, 2))

        per_constraint_residuals = torch.stack(
            [
                projection["residual_simplex"],
                projection["residual_piece_count"],
                projection["residual_king_count"],
                projection["residual_pawn_rank"],
            ],
            dim=1,
        )

        fusion = torch.cat(
            [
                per_class_residual_norm,
                per_constraint_residuals,
                residual_map_summary,
                encoder_summary,
                residual_total_norm.unsqueeze(-1),
                belief_entropy.unsqueeze(-1),
            ],
            dim=1,
        )

        logits = self.classifier(fusion).squeeze(-1)

        diagnostics: dict[str, torch.Tensor] = {"logits": logits}
        diagnostics["residual_total_norm"] = residual_total_norm
        diagnostics["residual_simplex_energy"] = projection["residual_simplex"]
        diagnostics["residual_piece_count_energy"] = projection["residual_piece_count"]
        diagnostics["residual_king_count_energy"] = projection["residual_king_count"]
        diagnostics["residual_pawn_rank_energy"] = projection["residual_pawn_rank"]
        diagnostics["belief_entropy"] = belief_entropy
        diagnostics["belief_empty_mass"] = beliefs[:, _EMPTY_CLASS_INDEX].flatten(1).sum(dim=1)
        diagnostics["projected_empty_mass"] = projected[:, _EMPTY_CLASS_INDEX].flatten(1).sum(dim=1)
        # Useful per-class residual probes for ablations.
        for idx, name in enumerate(_PIECE_CLASSES):
            diagnostics[f"residual_norm_{name}"] = per_class_residual_norm[:, idx]
        diagnostics["residual_norm_empty"] = per_class_residual_norm[:, _EMPTY_CLASS_INDEX]
        diagnostics["white_king_belief_total"] = beliefs[:, 5].flatten(1).sum(dim=1)
        diagnostics["black_king_belief_total"] = beliefs[:, 11].flatten(1).sum(dim=1)
        diagnostics["white_king_projected_total"] = projected[:, 5].flatten(1).sum(dim=1)
        diagnostics["black_king_projected_total"] = projected[:, 11].flatten(1).sum(dim=1)
        diagnostics["backbone_feature_norm"] = backbone.flatten(1).norm(dim=1)
        diagnostics["encoder_summary_norm"] = encoder_summary.norm(dim=1)
        diagnostics["residual_map_summary_norm"] = residual_map_summary.norm(dim=1)
        return diagnostics


def build_legal_constraint_projection_residual_network_from_config(
    config: dict[str, Any],
) -> LegalConstraintProjectionResidualNetwork:
    return LegalConstraintProjectionResidualNetwork(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        depth=int(config.get("depth", 2)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        projection_iters=int(config.get("projection_iters", 3)),
        residual_pool_channels=int(config.get("residual_pool_channels", 16)),
        stop_gradient_projection=bool(config.get("stop_gradient_projection", True)),
        encoding_adapter=str(config.get("encoding_adapter", config.get("encoding", SIMPLE_18))),
    )

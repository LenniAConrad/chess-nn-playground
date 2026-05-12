"""Low-Rank Signed Cut Query Network (idea i136).

Bespoke implementation of the architecture promoted from
``ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2136_friday_shanghai_architecture_batch_7.md``
(candidate 6).

Thesis: puzzle-like positions often separate the board into tense regions
(attacking vs defending mass, king-side vs centre, blocked wing vs open wing).
The model learns *low-rank signed cut queries* over learned board fields and
classifies puzzle-likeness from per-region imbalance statistics.

Architecture summary (faithful to the markdown):

1. Project the simple_18 board tensor into ``C`` learned board fields ``F`` via
   a ``1x1`` convolution.  No nonlinearity is applied so the fields stay linear
   reductions of the input planes, matching the "board fields" abstraction in
   the paper.
2. Build ``K`` *low-rank signed mask pairs*.  Each mask ``a_k(s)`` is the outer
   product of two coordinate factors,
   ``a_k(rank, file) = tanh(r^a_k(rank)) * tanh(f^a_k(file))`` (and similarly
   for the second mask ``b_k`` of the pair), so each mask is rank-1 in
   ``8x8`` and bounded in ``[-1, 1]``.
3. For every pair ``k`` and every field ``c`` compute the signed cut
   ``cut_{k,c} = sum_s a_k(s) F_c(s) - sum_s b_k(s) F_c(s)`` together with
   ``|cut|``, ``cut^2`` and a normalised cut
   ``cut / (eps + sum_s |F_c(s)|)``.
4. Add a *king-anchored* variant: a separate set of low-rank mask pairs that
   are translated so they are centred on the white king and on the black
   king of each board.  The translation is implemented by indexing into the
   ``64`` rolls of each mask using the king square argmax of piece planes
   ``5`` (white king) and ``11`` (black king); when no king is present the
   masks fall back to the central anchor.
5. A small convolutional trunk reads the same projected fields and global
   pools to a feature vector.  The cut summaries (mean/abs/squared/normalised
   plus king-anchored variants) and the trunk vector are concatenated and
   fed to an MLP head that emits one ``puzzle_binary`` logit.

The forward pass returns a dict whose ``logits`` tensor has shape ``(batch,)``
alongside per-batch diagnostics consumed by the prediction artefacts.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.data.board_features import SIMPLE_18
from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


_NUM_SQUARES = 64
_BOARD_SIDE = 8
_WHITE_KING_PLANE = 5
_BLACK_KING_PLANE = 11
_NUM_PIECE_PLANES = 12


def _build_roll_table(masks: torch.Tensor) -> torch.Tensor:
    """Return a ``(64, K, 8, 8)`` tensor of every (rank, file) roll of ``masks``.

    Index ``r * 8 + c`` stores ``torch.roll(masks, shifts=(r-3, c-3), dims=(-2, -1))``,
    so indexing by the king square of a board produces the mask "centred on
    that king" (with the canonical centre at ``(3, 3)``).
    """
    rolls: list[torch.Tensor] = []
    for r in range(_BOARD_SIDE):
        for c in range(_BOARD_SIDE):
            rolls.append(torch.roll(masks, shifts=(r - 3, c - 3), dims=(-2, -1)))
    return torch.stack(rolls, dim=0)


class LowRankSignedCutQueryNetwork(nn.Module):
    """Bespoke implementation of the Low-Rank Signed Cut Query Network idea."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        num_fields: int = 24,
        num_query_pairs: int = 32,
        num_king_pairs: int = 16,
        cnn_channels: int = 32,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        use_king_anchor: bool = True,
        eps: float = 1.0e-6,
        encoding_adapter: str = SIMPLE_18,
    ) -> None:
        super().__init__()
        if input_channels != 18 or encoding_adapter != SIMPLE_18:
            raise ValueError(
                "LowRankSignedCutQueryNetwork currently supports simple_18 with 18 input channels"
            )
        if num_classes != 1:
            raise ValueError(
                "LowRankSignedCutQueryNetwork supports the puzzle_binary one-logit contract"
            )
        if num_fields < 1:
            raise ValueError("num_fields must be >= 1")
        if num_query_pairs < 1:
            raise ValueError("num_query_pairs must be >= 1")
        if num_king_pairs < 0:
            raise ValueError("num_king_pairs must be >= 0")

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_fields = int(num_fields)
        self.num_query_pairs = int(num_query_pairs)
        self.num_king_pairs = int(num_king_pairs) if use_king_anchor else 0
        self.use_king_anchor = bool(use_king_anchor) and self.num_king_pairs > 0
        self.eps = float(eps)

        # 1x1 board-field projection: C learned linear fields F_c(s) over the
        # simple_18 planes. No nonlinearity -- the paper's "board fields" are
        # linear reductions, with all nonlinearity sitting in the cut head.
        self.field_projection = nn.Conv2d(input_channels, self.num_fields, kernel_size=1, bias=True)

        # Low-rank coordinate factors r_k, f_k for the global signed mask pairs.
        # tanh(r) * tanh(f) gives a rank-1 mask in [-1, 1] per the paper's
        # constraint ``a_k(rank, file) = r_k(rank) * f_k(file)``.
        self.rank_factor_a = nn.Parameter(torch.randn(self.num_query_pairs, _BOARD_SIDE) * 0.3)
        self.file_factor_a = nn.Parameter(torch.randn(self.num_query_pairs, _BOARD_SIDE) * 0.3)
        self.rank_factor_b = nn.Parameter(torch.randn(self.num_query_pairs, _BOARD_SIDE) * 0.3)
        self.file_factor_b = nn.Parameter(torch.randn(self.num_query_pairs, _BOARD_SIDE) * 0.3)

        if self.use_king_anchor:
            self.king_rank_factor_a = nn.Parameter(
                torch.randn(self.num_king_pairs, _BOARD_SIDE) * 0.3
            )
            self.king_file_factor_a = nn.Parameter(
                torch.randn(self.num_king_pairs, _BOARD_SIDE) * 0.3
            )
            self.king_rank_factor_b = nn.Parameter(
                torch.randn(self.num_king_pairs, _BOARD_SIDE) * 0.3
            )
            self.king_file_factor_b = nn.Parameter(
                torch.randn(self.num_king_pairs, _BOARD_SIDE) * 0.3
            )

        # Small convolutional trunk over the same fields -- the architecture
        # sketch step 5 explicitly fuses cut summaries with a small CNN.
        self.cnn_trunk = nn.Sequential(
            nn.Conv2d(self.num_fields, cnn_channels, kernel_size=3, padding=1, bias=not use_batchnorm),
            nn.BatchNorm2d(cnn_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1, bias=not use_batchnorm),
            nn.BatchNorm2d(cnn_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        # Per-pair-and-field summaries: signed cut, |cut|, cut^2, normalised
        # cut. Four scalars per (k, c).
        global_summary_dim = 4 * self.num_query_pairs * self.num_fields
        # Two king-anchored variants (white king, black king) reuse the same
        # four scalar summaries per (king_pair, field).
        king_summary_dim = (
            2 * 4 * self.num_king_pairs * self.num_fields if self.use_king_anchor else 0
        )
        # Fold in a small set of global field statistics for stability.
        global_field_dim = 3 * self.num_fields  # mean, abs-mean, squared-mean
        # Plus the CNN trunk feature vector.
        cnn_feature_dim = cnn_channels

        fusion_dim = global_summary_dim + king_summary_dim + global_field_dim + cnn_feature_dim

        layers: list[nn.Module] = []
        in_dim = fusion_dim
        head_depth = max(1, int(depth))
        for _ in range(head_depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.classifier = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Mask construction
    # ------------------------------------------------------------------

    def _global_mask(self, rank_factor: torch.Tensor, file_factor: torch.Tensor) -> torch.Tensor:
        # tanh-bounded rank-1 outer product: a_k(r, c) = tanh(r_k(r)) * tanh(f_k(c)).
        rk = torch.tanh(rank_factor).unsqueeze(-1)  # (K, 8, 1)
        fk = torch.tanh(file_factor).unsqueeze(-2)  # (K, 1, 8)
        return rk * fk  # (K, 8, 8)

    def _king_anchored_mask(
        self,
        mask: torch.Tensor,
        king_plane: torch.Tensor,
    ) -> torch.Tensor:
        """Return king-anchored masks of shape ``(B, K_king, 8, 8)``.

        ``king_plane`` is a one-hot ``(B, 8, 8)`` plane (white or black king).
        For each batch element we look up the king's square index ``i`` and
        translate the mask so that its canonical centre ``(3, 3)`` lies on
        that square. When no king is present (e.g. partial test fixtures) we
        fall back to the centred mask.
        """
        rolls = _build_roll_table(mask)  # (64, K_king, 8, 8)
        flat = king_plane.flatten(1)  # (B, 64)
        present = flat.sum(dim=1) > 0
        king_idx = flat.argmax(dim=1)  # (B,)
        center_idx = 3 * _BOARD_SIDE + 3
        idx = torch.where(present, king_idx, torch.full_like(king_idx, center_idx))
        return rolls[idx]  # (B, K_king, 8, 8)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _signed_cut(
        self,
        fields: torch.Tensor,
        mask_a: torch.Tensor,
        mask_b: torch.Tensor,
        field_mass: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # fields: (B, C, 8, 8); mask_a/mask_b: (K, 8, 8) or (B, K, 8, 8).
        if mask_a.dim() == 3:
            cut_pos = torch.einsum("bchw,khw->bkc", fields, mask_a)
            cut_neg = torch.einsum("bchw,khw->bkc", fields, mask_b)
        else:
            cut_pos = torch.einsum("bchw,bkhw->bkc", fields, mask_a)
            cut_neg = torch.einsum("bchw,bkhw->bkc", fields, mask_b)
        cut = cut_pos - cut_neg
        abs_cut = cut.abs()
        cut_sq = cut * cut
        # field_mass: (B, C) -> broadcast to (B, K, C).
        norm = cut / (self.eps + field_mass.unsqueeze(1))
        return cut, abs_cut, cut_sq, norm

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        batch = x.shape[0]

        fields = self.field_projection(x)  # (B, C, 8, 8)

        # Mass per field for normalised cuts.
        field_mass = fields.abs().flatten(2).sum(dim=2)  # (B, C)

        # Global signed cuts.
        mask_a = self._global_mask(self.rank_factor_a, self.file_factor_a)
        mask_b = self._global_mask(self.rank_factor_b, self.file_factor_b)
        cut, abs_cut, cut_sq, norm_cut = self._signed_cut(fields, mask_a, mask_b, field_mass)
        global_summary = torch.cat(
            [cut.flatten(1), abs_cut.flatten(1), cut_sq.flatten(1), norm_cut.flatten(1)],
            dim=1,
        )

        diagnostics: dict[str, torch.Tensor] = {}

        if self.use_king_anchor:
            piece_planes = x[:, :_NUM_PIECE_PLANES]
            white_king = piece_planes[:, _WHITE_KING_PLANE]  # (B, 8, 8)
            black_king = piece_planes[:, _BLACK_KING_PLANE]
            king_mask_a_global = self._global_mask(self.king_rank_factor_a, self.king_file_factor_a)
            king_mask_b_global = self._global_mask(self.king_rank_factor_b, self.king_file_factor_b)
            white_mask_a = self._king_anchored_mask(king_mask_a_global, white_king)
            white_mask_b = self._king_anchored_mask(king_mask_b_global, white_king)
            black_mask_a = self._king_anchored_mask(king_mask_a_global, black_king)
            black_mask_b = self._king_anchored_mask(king_mask_b_global, black_king)
            wk_cut, wk_abs, wk_sq, wk_norm = self._signed_cut(
                fields, white_mask_a, white_mask_b, field_mass
            )
            bk_cut, bk_abs, bk_sq, bk_norm = self._signed_cut(
                fields, black_mask_a, black_mask_b, field_mass
            )
            king_summary = torch.cat(
                [
                    wk_cut.flatten(1), wk_abs.flatten(1), wk_sq.flatten(1), wk_norm.flatten(1),
                    bk_cut.flatten(1), bk_abs.flatten(1), bk_sq.flatten(1), bk_norm.flatten(1),
                ],
                dim=1,
            )
            diagnostics["king_anchored_cut_mean"] = (
                wk_cut.flatten(1).mean(dim=1) + bk_cut.flatten(1).mean(dim=1)
            ) * 0.5
            diagnostics["king_anchored_abs_cut_mean"] = (
                wk_abs.flatten(1).mean(dim=1) + bk_abs.flatten(1).mean(dim=1)
            ) * 0.5
            diagnostics["white_king_cut_energy"] = wk_sq.flatten(1).sum(dim=1)
            diagnostics["black_king_cut_energy"] = bk_sq.flatten(1).sum(dim=1)
        else:
            king_summary = torch.zeros(batch, 0, device=fields.device, dtype=fields.dtype)
            zeros = torch.zeros(batch, device=fields.device, dtype=fields.dtype)
            diagnostics["king_anchored_cut_mean"] = zeros
            diagnostics["king_anchored_abs_cut_mean"] = zeros
            diagnostics["white_king_cut_energy"] = zeros
            diagnostics["black_king_cut_energy"] = zeros

        # Global field statistics.
        flat_fields = fields.flatten(2)
        field_mean = flat_fields.mean(dim=2)
        field_abs_mean = flat_fields.abs().mean(dim=2)
        field_sq_mean = (flat_fields * flat_fields).mean(dim=2)
        field_stats = torch.cat([field_mean, field_abs_mean, field_sq_mean], dim=1)

        # CNN trunk fused with cut summaries.
        trunk = self.cnn_trunk(fields)
        trunk_vec = trunk.flatten(2).mean(dim=2)  # global avg pool

        fusion = torch.cat([global_summary, king_summary, field_stats, trunk_vec], dim=1)
        logits = self.classifier(fusion).squeeze(-1)

        diagnostics["logits"] = logits
        diagnostics["cut_signed_mean"] = cut.flatten(1).mean(dim=1)
        diagnostics["cut_abs_mean"] = abs_cut.flatten(1).mean(dim=1)
        diagnostics["cut_squared_mean"] = cut_sq.flatten(1).mean(dim=1)
        diagnostics["normalised_cut_mean"] = norm_cut.flatten(1).mean(dim=1)
        diagnostics["cut_abs_max"] = abs_cut.flatten(1).max(dim=1).values
        diagnostics["field_total_energy"] = field_mass.sum(dim=1)
        diagnostics["field_max_abs"] = flat_fields.abs().flatten(1).max(dim=1).values
        # Rank-file imbalance: contrast row vs column factor masses, summed
        # over all global pairs, surfaces the directionality of the learned
        # masks per board.
        rf_imbalance = (
            torch.einsum("bchw,kh->bkc", fields, torch.tanh(self.rank_factor_a)).abs().mean(dim=(1, 2))
            - torch.einsum("bchw,kw->bkc", fields, torch.tanh(self.file_factor_a)).abs().mean(dim=(1, 2))
        )
        diagnostics["rank_file_imbalance"] = rf_imbalance
        return diagnostics


def build_low_rank_signed_cut_query_network_from_config(
    config: dict[str, Any],
) -> LowRankSignedCutQueryNetwork:
    return LowRankSignedCutQueryNetwork(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        num_fields=int(config.get("num_fields", 24)),
        num_query_pairs=int(config.get("num_query_pairs", 32)),
        num_king_pairs=int(config.get("num_king_pairs", 16)),
        cnn_channels=int(config.get("cnn_channels", config.get("channels", 32))),
        hidden_dim=int(config.get("hidden_dim", 96)),
        depth=int(config.get("depth", 2)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        use_king_anchor=bool(config.get("use_king_anchor", True)),
        eps=float(config.get("eps", 1.0e-6)),
        encoding_adapter=str(config.get("encoding_adapter", config.get("encoding", SIMPLE_18))),
    )

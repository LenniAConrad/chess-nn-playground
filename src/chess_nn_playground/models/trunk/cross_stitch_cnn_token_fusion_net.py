"""Cross-Stitch CNN-Token Fusion Net for idea i157.

A bespoke architecture for the ``puzzle_binary`` contract. Two branches
read the same simple_18 board: a convolutional board branch over the
``(B, C, 8, 8)`` board map and an occupied-piece token branch over the
up-to-32 occupied-square tokens produced by
:class:`Simple18PieceTokenExtractor`. At several intermediate
"cross-stitch" stages each branch is summarised, the two summaries are
mixed by a learned ``2x2`` (or per-group ``G x 2 x 2``) cross-stitch
matrix, and the mixed summaries are injected back into the branches
through small adapters. A final head reads the pooled board feature,
the pooled token feature, and the per-stage cross-stitch diagnostics
and produces one puzzle logit.

This is the markdown thesis' "cross-stitch network exchanges
information at multiple depths through learned linear mixing": the
exchange happens at every stage rather than only at a final concat.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.data.board_features import SIMPLE_18
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor
from chess_nn_playground.models.trunk.piece_token_cnn_hybrid import (
    MATERIAL_SUMMARY_DIM,
    TOKEN_FEATURE_DIM,
    Simple18PieceTokenExtractor,
    _masked_max,
    _masked_mean,
)


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.norm = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.conv(x))))


class _TokenBlock(nn.Module):
    """A single residual token-mixer block: per-token MLP + masked summary gate."""

    def __init__(self, token_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(token_dim)
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.summary_norm = nn.LayerNorm(token_dim * 2)
        self.summary_gate = nn.Linear(token_dim * 2, token_dim)
        self.summary_proj = nn.Linear(token_dim * 2, token_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        token_mask = mask.to(dtype=tokens.dtype).unsqueeze(-1)
        residual = self.mlp(self.norm(tokens)) * token_mask
        tokens = tokens + residual
        summary = torch.cat(
            [_masked_mean(tokens, mask), _masked_max(tokens, mask)],
            dim=1,
        )
        summary = self.summary_norm(summary)
        gate = torch.sigmoid(self.summary_gate(summary)).unsqueeze(1)
        update = torch.tanh(self.summary_proj(summary)).unsqueeze(1)
        return (tokens + gate * update) * token_mask


class CrossStitchUnit(nn.Module):
    """Per-group ``2x2`` cross-stitch mixing matrix.

    Given branch-channel summaries ``b in R^{B x C}`` and
    ``p in R^{B x C}`` (the board and token branches share the channel
    width ``C``), split each summary into ``G = num_groups`` groups
    ``b_g, p_g in R^{B x C/G}`` and apply a per-group ``2x2`` matrix:

    ::

        [b'_g]   [a_g  c_g] [b_g]
        [p'_g] = [d_g  e_g] [p_g]

    The matrix entries are stored as a learnable
    ``(num_groups, 2, 2)`` parameter initialised to the identity so the
    unit starts as a no-op. ``diagonal_only`` zeros the off-diagonal
    entries to disable cross-branch exchange (the ``diagonal_stitch``
    ablation).
    """

    def __init__(self, channels: int, num_groups: int, diagonal_only: bool = False) -> None:
        super().__init__()
        if channels % num_groups != 0:
            raise ValueError("channels must be divisible by num_groups")
        self.channels = int(channels)
        self.num_groups = int(num_groups)
        self.group_width = self.channels // self.num_groups
        self.diagonal_only = bool(diagonal_only)
        # Initialise to the identity so the unit is a no-op at start.
        init = torch.eye(2).unsqueeze(0).repeat(self.num_groups, 1, 1)
        self.matrix = nn.Parameter(init.clone())
        if self.diagonal_only:
            mask = torch.eye(2).unsqueeze(0).repeat(self.num_groups, 1, 1)
            self.register_buffer("diag_mask", mask, persistent=False)
        else:
            self.diag_mask = None

    def effective_matrix(self) -> torch.Tensor:
        if self.diagonal_only and self.diag_mask is not None:
            return self.matrix * self.diag_mask
        return self.matrix

    def forward(self, b: torch.Tensor, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = b.shape[0]
        b_g = b.view(batch, self.num_groups, self.group_width)
        p_g = p.view(batch, self.num_groups, self.group_width)
        stacked = torch.stack([b_g, p_g], dim=2)  # (B, G, 2, w)
        matrix = self.effective_matrix().unsqueeze(0)  # (1, G, 2, 2)
        mixed = torch.matmul(matrix, stacked)  # (B, G, 2, w)
        b_new = mixed[:, :, 0, :].reshape(batch, self.channels)
        p_new = mixed[:, :, 1, :].reshape(batch, self.channels)
        return b_new, p_new


class CrossStitchCNNTokenFusionNet(nn.Module):
    """Cross-stitch CNN + piece-token fusion network for puzzle_binary.

    Pipeline:

    1. **Stem.** ``input_channels -> board_width`` ``3x3`` conv (with
       optional batch-norm and GELU) lifts the board to the trunk
       width. ``Simple18PieceTokenExtractor`` extracts up to
       ``max_piece_tokens`` occupied-piece tokens. A small
       ``Linear(TOKEN_FEATURE_DIM, token_width)`` lifts each token to
       the token branch width. Both branches share the same hidden
       width ``C = board_width = token_width`` so the cross-stitch unit
       can mix per-channel summaries.
    2. **Cross-stitch stages.** For each of ``num_stages`` stages:

       - Run one residual conv block on the board branch.
       - Run one residual token-mixer block on the token branch.
       - Pool each branch into a ``(B, C)`` summary
         (board: average over spatial grid; tokens: masked mean).
       - Mix the two summaries through a per-group ``2x2``
         :class:`CrossStitchUnit` (initialised to identity).
       - Inject the mixed summary back into each branch through a
         learned adapter (``Linear`` + broadcast for board,
         ``Linear`` + broadcast over tokens for tokens).

    3. **Final fusion.** The final pooled summaries
       ``(b_final, p_final)``, the material summary, and the
       cross-stitch diagnostics (per-stage off-diagonal energy,
       branch norms before/after stitching) are concatenated and run
       through a small MLP head that produces the puzzle logit.

    Diagnostics expose the learned cross-stitch matrices, the
    board-to-token and token-to-board transfer magnitudes per stage,
    and branch norms before and after stitching.
    """

    VALID_ABLATIONS = {
        "none",
        "late_fusion_only",
        "board_only",
        "token_only",
        "diagonal_stitch",
        "random_token_coords",
    }

    COORD_SLICE = slice(10, 14)

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        board_width: int = 64,
        token_width: int = 64,
        num_stages: int = 3,
        cross_stitch_groups: int = 8,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        max_piece_tokens: int = 32,
        ablation: str = "none",
        encoding: str = SIMPLE_18,
    ) -> None:
        super().__init__()
        if encoding != SIMPLE_18 or input_channels != 18:
            raise ValueError(
                "CrossStitchCNNTokenFusionNet currently implements the simple_18 board contract only"
            )
        if num_classes != 1:
            raise ValueError(
                "CrossStitchCNNTokenFusionNet supports the puzzle_binary one-logit contract"
            )
        if num_stages < 1:
            raise ValueError("num_stages must be >= 1")
        if board_width != token_width:
            raise ValueError("board_width must equal token_width to share the cross-stitch channel width")
        if board_width % cross_stitch_groups != 0:
            raise ValueError("board_width must be divisible by cross_stitch_groups")
        if ablation not in self.VALID_ABLATIONS:
            raise ValueError(f"Unknown CrossStitchCNNTokenFusionNet ablation: {ablation}")

        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.channels = int(board_width)
        self.num_stages = int(num_stages)
        self.cross_stitch_groups = int(cross_stitch_groups)
        self.hidden_dim = int(hidden_dim)
        self.dropout_p = float(dropout)
        self.use_batchnorm = bool(use_batchnorm)
        self.max_piece_tokens = int(max_piece_tokens)
        self.ablation = ablation
        self.spec = BoardTensorSpec(input_channels=self.input_channels)

        self.extractor = Simple18PieceTokenExtractor(
            input_channels=self.input_channels, max_tokens=self.max_piece_tokens
        )
        self.token_embed = nn.Sequential(
            nn.Linear(TOKEN_FEATURE_DIM, self.channels),
            nn.GELU(),
            nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity(),
            nn.Linear(self.channels, self.channels),
        )

        self.stem = _ConvBlock(self.input_channels, self.channels, use_batchnorm, dropout=self.dropout_p)

        self.board_blocks = nn.ModuleList(
            [_ConvBlock(self.channels, self.channels, use_batchnorm, dropout=self.dropout_p) for _ in range(self.num_stages)]
        )
        self.token_blocks = nn.ModuleList(
            [_TokenBlock(self.channels, dropout=self.dropout_p) for _ in range(self.num_stages)]
        )

        diagonal_only = self.ablation == "diagonal_stitch"
        # late_fusion_only disables intermediate cross-stitch entirely.
        self.use_intermediate_cross_stitch = self.ablation != "late_fusion_only"
        self.cross_stitch = nn.ModuleList(
            [
                CrossStitchUnit(self.channels, self.cross_stitch_groups, diagonal_only=diagonal_only)
                for _ in range(self.num_stages)
            ]
        )
        # Adapters inject the cross-stitch summary back into each branch.
        self.board_adapters = nn.ModuleList(
            [nn.Linear(self.channels, self.channels) for _ in range(self.num_stages)]
        )
        self.token_adapters = nn.ModuleList(
            [nn.Linear(self.channels, self.channels) for _ in range(self.num_stages)]
        )

        # Final head reads pooled board, pooled tokens, material summary,
        # and cross-stitch diagnostics (per-stage off-diagonal energy =
        # one scalar per stage).
        diag_dim = self.num_stages
        fused_dim = self.channels * 2 + MATERIAL_SUMMARY_DIM + diag_dim
        mid = max(32, self.hidden_dim // 2)
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity(),
            nn.Linear(self.hidden_dim, mid),
            nn.GELU(),
            nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity(),
            nn.Linear(mid, 1),
        )

    # -- helpers ----------------------------------------------------

    @staticmethod
    def _board_pool(board: torch.Tensor) -> torch.Tensor:
        return board.mean(dim=(2, 3))

    @staticmethod
    def _token_pool(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return _masked_mean(tokens, mask)

    def _ablate_token_features(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.ablation == "random_token_coords":
            features = features.clone()
            shuffled = features[:, :, self.COORD_SLICE].roll(shifts=1, dims=1)
            features[:, :, self.COORD_SLICE] = shuffled
        return features * mask.to(dtype=features.dtype).unsqueeze(-1)

    # -- forward ----------------------------------------------------

    def forward(self, board: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(board, self.spec)
        token_batch = self.extractor(x)
        token_features = self._ablate_token_features(token_batch.features, token_batch.mask)

        h_board = self.stem(x)  # (B, C, 8, 8)
        token_mask_f = token_batch.mask.to(dtype=token_features.dtype).unsqueeze(-1)
        h_tokens = self.token_embed(token_features) * token_mask_f  # (B, P, C)

        # Per-stage diagnostic accumulators.
        offdiag_energies: list[torch.Tensor] = []
        board_norms_pre: list[torch.Tensor] = []
        token_norms_pre: list[torch.Tensor] = []
        board_norms_post: list[torch.Tensor] = []
        token_norms_post: list[torch.Tensor] = []
        board_to_token_transfer: list[torch.Tensor] = []
        token_to_board_transfer: list[torch.Tensor] = []
        cross_stitch_matrices: list[torch.Tensor] = []

        for stage in range(self.num_stages):
            h_board = self.board_blocks[stage](h_board)
            h_tokens = self.token_blocks[stage](h_tokens, token_batch.mask)

            b_pool = self._board_pool(h_board)
            p_pool = self._token_pool(h_tokens, token_batch.mask)

            board_norms_pre.append(b_pool.detach().norm(dim=-1))
            token_norms_pre.append(p_pool.detach().norm(dim=-1))

            unit = self.cross_stitch[stage]
            matrix = unit.effective_matrix()  # (G, 2, 2)
            cross_stitch_matrices.append(matrix.detach())
            # off-diagonal energy across groups: |a01|^2 + |a10|^2 averaged.
            off = (matrix[:, 0, 1].pow(2) + matrix[:, 1, 0].pow(2)).mean()
            offdiag_energies.append(off)

            if self.use_intermediate_cross_stitch:
                b_mixed, p_mixed = unit(b_pool, p_pool)
            else:
                b_mixed, p_mixed = b_pool, p_pool

            # Board-to-token and token-to-board transfer magnitudes
            # (the contribution of the *other* branch in the mixed
            # summary, before the adapter). For the no-op identity
            # initial matrix these start near zero.
            board_to_token_transfer.append((p_mixed - p_pool).detach().norm(dim=-1))
            token_to_board_transfer.append((b_mixed - b_pool).detach().norm(dim=-1))

            # Inject mixed summaries back through adapters.
            if self.use_intermediate_cross_stitch:
                b_inject = self.board_adapters[stage](b_mixed).view(-1, self.channels, 1, 1)
                p_inject = self.token_adapters[stage](p_mixed).unsqueeze(1)
                if self.ablation == "token_only":
                    b_inject = torch.zeros_like(b_inject)
                if self.ablation == "board_only":
                    p_inject = torch.zeros_like(p_inject)
                h_board = h_board + b_inject
                h_tokens = (h_tokens + p_inject) * token_mask_f

            board_norms_post.append(self._board_pool(h_board).detach().norm(dim=-1))
            token_norms_post.append(self._token_pool(h_tokens, token_batch.mask).detach().norm(dim=-1))

        b_final = self._board_pool(h_board)
        p_final = self._token_pool(h_tokens, token_batch.mask)
        material_summary = token_batch.material_summary

        if self.ablation == "token_only":
            b_final = torch.zeros_like(b_final)
        elif self.ablation == "board_only":
            p_final = torch.zeros_like(p_final)
            material_summary = torch.zeros_like(material_summary)

        offdiag_stack = torch.stack(offdiag_energies, dim=0).unsqueeze(0).expand(b_final.shape[0], -1)
        fused = torch.cat([b_final, p_final, material_summary, offdiag_stack], dim=1)
        logits = self.head(fused).view(-1)

        cross_stitch_matrix_stack = torch.stack(cross_stitch_matrices, dim=0)  # (S, G, 2, 2)
        offdiag_per_stage = torch.stack(offdiag_energies, dim=0).detach()  # (S,)
        b2t = torch.stack(board_to_token_transfer, dim=1)  # (B, S)
        t2b = torch.stack(token_to_board_transfer, dim=1)  # (B, S)
        return {
            "logits": logits,
            "logit": logits,
            "prob": torch.sigmoid(logits),
            "board_latent": h_board,
            "token_latent": h_tokens,
            "board_pool_final": b_final,
            "token_pool_final": p_final,
            "material_summary": material_summary,
            "token_count": token_batch.token_count,
            "cross_stitch_matrices": cross_stitch_matrix_stack,
            "offdiag_energy_per_stage": offdiag_per_stage,
            "board_to_token_transfer": b2t,
            "token_to_board_transfer": t2b,
            "board_norms_pre": torch.stack(board_norms_pre, dim=1),
            "board_norms_post": torch.stack(board_norms_post, dim=1),
            "token_norms_pre": torch.stack(token_norms_pre, dim=1),
            "token_norms_post": torch.stack(token_norms_post, dim=1),
        }


def build_cross_stitch_cnn_token_fusion_net_from_config(
    config: dict[str, Any],
) -> CrossStitchCNNTokenFusionNet:
    cfg = dict(config)
    cfg.setdefault("input_channels", 18)
    cfg.setdefault("num_classes", 1)
    # Map the idea config aliases to model kwargs. ``channels`` is the
    # shared branch width (board_width == token_width) used by the
    # cross-stitch units; ``depth`` becomes ``num_stages``.
    channels = int(cfg.get("channels", cfg.get("board_width", 64)))
    board_width = int(cfg.get("board_width", channels))
    token_width = int(cfg.get("token_width", channels))
    num_stages = int(cfg.get("num_stages", cfg.get("stages", cfg.get("depth", 3))))
    cross_stitch_groups = int(cfg.get("cross_stitch_groups", cfg.get("num_groups", 8)))
    return CrossStitchCNNTokenFusionNet(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        board_width=board_width,
        token_width=token_width,
        num_stages=num_stages,
        cross_stitch_groups=cross_stitch_groups,
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        max_piece_tokens=int(cfg.get("max_piece_tokens", 32)),
        ablation=str(cfg.get("ablation", "none")),
        encoding=str(cfg.get("encoding", SIMPLE_18)),
    )

"""Counterfactual move-delta spectrum network for idea i026.

The model implements the rule-only one-ply counterfactual move-delta spectrum
described in `ideas/i026_counterfactual_move_delta_spectrum_network`.

Pipeline (per board ``x``):

1. ``Simple18BoardAdapter`` parses ``simple_18`` planes into a current-board
   state ``B(x)`` (piece type/color/plane, side to move, castling, en-passant).
2. ``PseudoLegalDeltaEnumerator`` enumerates the side-to-move pseudo-legal
   one-ply move-delta set ``A(x)`` without engine evaluation, self-check
   filtering, or checkmate/stalemate oracles.
3. ``BoardStem`` produces a learned 8x8xd square map ``H_sq`` and a global
   feature ``g``.
4. ``MoveTokenEncoder`` gathers ``H_from``, ``H_to`` and the finite-difference
   ``H_to - H_from`` together with deterministic move descriptors and the
   broadcast global feature, then produces per-move response vectors
   ``r in R^k``.
5. ``CounterfactualSpectrumPool`` computes masked mean/max responses, the
   uniform-weighted covariance ``K`` and its eigenvalues via
   ``torch.linalg.eigvalsh``, and derives spectral statistics: trace,
   leading-eigenvalue fraction, participation ratio, normalised spectral
   entropy and Frobenius norm.
6. ``MoveDeltaSpectrumHead`` concatenates ``g``, ``r_mean``, ``r_max``,
   ``r_var`` and the spectral stats and returns the puzzle logit(s).

Returns a dict including ``logits`` shaped ``(B,)`` for the
``num_classes == 1`` puzzle-binary contract, plus diagnostics named in
``ideas/i026_counterfactual_move_delta_spectrum_network/architecture.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor
from chess_nn_playground.models.move_landscape_net import (
    PIECE_PLANES,
    PROMO_TYPES,
    RELATIVE_BUCKETS,
    SPECIAL_TYPES,
    MoveRecords,
    PseudoLegalDeltaEnumerator,
    Simple18BoardAdapter,
)


@dataclass(frozen=True)
class SpectrumPoolOutputs:
    r_mean: torch.Tensor
    r_max: torch.Tensor
    r_var: torch.Tensor
    eigvals: torch.Tensor
    trace: torch.Tensor
    leading_fraction: torch.Tensor
    participation_ratio: torch.Tensor
    spectral_entropy: torch.Tensor
    frobenius_norm: torch.Tensor


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


class _ConvBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class BoardStem(nn.Module):
    def __init__(
        self,
        input_channels: int,
        square_dim: int,
        global_dim: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(input_channels, square_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, square_dim),
            nn.GELU(),
        ]
        for _ in range(max(1, int(depth))):
            layers.append(_ConvBlock(square_dim, dropout))
        self.grid = nn.Sequential(*layers)
        coord = torch.zeros(2, 8, 8)
        for r in range(8):
            for f in range(8):
                coord[0, r, f] = r / 7.0
                coord[1, r, f] = f / 7.0
        self.register_buffer("coord_planes", coord.unsqueeze(0))
        self.coord_proj = nn.Conv2d(square_dim + 2, square_dim, kernel_size=1)
        self.global_proj = nn.Sequential(
            nn.LayerNorm(2 * square_dim),
            nn.Linear(2 * square_dim, global_dim),
            nn.GELU(),
            nn.Linear(global_dim, global_dim),
            nn.LayerNorm(global_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.grid(x)
        coord = self.coord_planes.to(dtype=h.dtype, device=h.device).expand(h.shape[0], -1, -1, -1)
        h = self.coord_proj(torch.cat([h, coord], dim=1))
        h_sq = h.flatten(2).transpose(1, 2)
        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        g = self.global_proj(pooled)
        return h, h_sq, g


class MoveTokenEncoder(nn.Module):
    def __init__(
        self,
        square_dim: int,
        global_dim: int,
        token_hidden: int,
        move_response_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.piece_embedding = nn.Embedding(PIECE_PLANES + 1, 8)
        self.capture_embedding = nn.Embedding(PIECE_PLANES + 1, 8)
        self.promo_embedding = nn.Embedding(PROMO_TYPES, 4)
        self.special_embedding = nn.Embedding(SPECIAL_TYPES, 4)
        self.rel_embedding = nn.Embedding(RELATIVE_BUCKETS, 12)
        det_dim = 8 + 8 + 4 + 4 + 12 + 2 + 2
        in_dim = 3 * square_dim + global_dim + det_dim
        self.det_dim = det_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, token_hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(token_hidden, move_response_dim),
        )

    @staticmethod
    def _gather_square(h_sq: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return h_sq.gather(1, index.unsqueeze(-1).expand(-1, -1, h_sq.shape[-1]))

    def forward(
        self,
        h_sq: torch.Tensor,
        g: torch.Tensor,
        moves: MoveRecords,
    ) -> torch.Tensor:
        dtype = h_sq.dtype
        h_from = self._gather_square(h_sq, moves.from_sq)
        h_to = self._gather_square(h_sq, moves.to_sq)
        h_diff = h_to - h_from
        g_broadcast = g.unsqueeze(1).expand(-1, h_from.shape[1], -1)
        pieces = self.piece_embedding(moves.piece_id.clamp(0, PIECE_PLANES)).to(dtype=dtype)
        captures = self.capture_embedding(moves.capture_id.clamp(0, PIECE_PLANES)).to(dtype=dtype)
        promos = self.promo_embedding(moves.promo_id.clamp(0, PROMO_TYPES - 1)).to(dtype=dtype)
        specials = self.special_embedding(moves.special_id.clamp(0, SPECIAL_TYPES - 1)).to(dtype=dtype)
        rels = self.rel_embedding(moves.rel_id.clamp(0, RELATIVE_BUCKETS - 1)).to(dtype=dtype)
        delta = torch.stack([moves.delta_rank / 7.0, moves.delta_file / 7.0], dim=-1).to(dtype=dtype)
        is_capture = (moves.capture_id > 0).to(dtype=dtype).unsqueeze(-1)
        is_promotion = (moves.promo_id > 0).to(dtype=dtype).unsqueeze(-1)
        token_in = torch.cat(
            [h_from, h_to, h_diff, g_broadcast, pieces, captures, promos, specials, rels, delta, is_capture, is_promotion],
            dim=-1,
        )
        encoded = self.net(token_in)
        mask = moves.valid_mask.to(dtype=dtype).unsqueeze(-1)
        return encoded * mask


class CounterfactualSpectrumPool(nn.Module):
    def __init__(self, move_response_dim: int, eps: float = 1.0e-4) -> None:
        super().__init__()
        self.move_response_dim = int(move_response_dim)
        self.eps = float(eps)
        self.scalar_dim = 5
        self.output_dim = 3 * self.move_response_dim + self.move_response_dim + self.scalar_dim

    def forward(self, responses: torch.Tensor, moves: MoveRecords) -> SpectrumPoolOutputs:
        dtype = responses.dtype
        mask = moves.valid_mask.to(dtype=dtype).unsqueeze(-1)
        count = mask.sum(dim=1).clamp_min(1.0)
        weights = mask / count.unsqueeze(-1)
        r_mean = (weights * responses).sum(dim=1)
        centered = (responses - r_mean.unsqueeze(1)) * mask
        r_var = (weights * centered.square()).sum(dim=1)
        neg_large = torch.finfo(responses.dtype).min / 4.0
        masked_for_max = torch.where(moves.valid_mask.unsqueeze(-1), responses, responses.new_full((), neg_large))
        r_max = masked_for_max.amax(dim=1)
        no_valid = ~moves.valid_mask.any(dim=1, keepdim=True)
        r_max = torch.where(no_valid, torch.zeros_like(r_max), r_max)

        weighted_centered = centered * weights
        cov = torch.einsum("bmi,bmj->bij", weighted_centered, centered)
        eye = torch.eye(self.move_response_dim, dtype=dtype, device=responses.device).unsqueeze(0)
        cov = cov + self.eps * eye

        cov32 = cov.to(dtype=torch.float32)
        eigvals32 = torch.linalg.eigvalsh(cov32)
        eigvals32 = eigvals32.clamp_min(0.0).flip(dims=(-1,))
        eigvals = eigvals32.to(dtype=dtype)

        trace = eigvals.sum(dim=-1)
        trace_sq = (eigvals * eigvals).sum(dim=-1)
        leading = eigvals[:, 0]
        leading_fraction = leading / trace.clamp_min(self.eps)
        participation_ratio = (trace * trace) / trace_sq.clamp_min(self.eps * self.eps)
        normalized = eigvals / trace.clamp_min(self.eps).unsqueeze(-1)
        spectral_entropy = -(normalized * (normalized + self.eps).log()).sum(dim=-1)
        frobenius_norm = torch.sqrt(trace_sq.clamp_min(0.0) + self.eps)
        return SpectrumPoolOutputs(
            r_mean=r_mean,
            r_max=r_max,
            r_var=r_var,
            eigvals=eigvals,
            trace=trace,
            leading_fraction=leading_fraction,
            participation_ratio=participation_ratio,
            spectral_entropy=spectral_entropy,
            frobenius_norm=frobenius_norm,
        )


class MoveDeltaSpectrumHead(nn.Module):
    def __init__(
        self,
        global_dim: int,
        spectrum_feature_dim: int,
        eigenvalue_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        in_dim = global_dim + spectrum_feature_dim + eigenvalue_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class MoveDeltaSpectrumNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        encoding: str = "simple_18",
        square_dim: int = 64,
        global_dim: int = 128,
        move_response_dim: int = 16,
        token_hidden_dim: int = 128,
        head_hidden_dim: int = 128,
        stem_depth: int = 2,
        max_moves: int = 256,
        dropout: float = 0.1,
        trace_penalty_beta: float = 1.0e-4,
        enable_castling_deltas: bool = False,
        adapter_strict: bool = True,
        spectrum_eps: float = 1.0e-4,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        if move_response_dim < 2:
            raise ValueError("move_response_dim must be >= 2 to define a covariance spectrum")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.move_response_dim = int(move_response_dim)
        self.trace_penalty_beta = float(trace_penalty_beta)
        self.adapter = Simple18BoardAdapter(
            input_channels=input_channels,
            encoding=encoding,
            adapter_strict=adapter_strict,
        )
        self.enumerator = PseudoLegalDeltaEnumerator(
            max_moves=max_moves,
            include_castling_candidates=enable_castling_deltas,
        )
        self.stem = BoardStem(
            input_channels=input_channels,
            square_dim=square_dim,
            global_dim=global_dim,
            depth=stem_depth,
            dropout=dropout,
        )
        self.token_encoder = MoveTokenEncoder(
            square_dim=square_dim,
            global_dim=global_dim,
            token_hidden=token_hidden_dim,
            move_response_dim=move_response_dim,
            dropout=dropout,
        )
        self.spectrum_pool = CounterfactualSpectrumPool(
            move_response_dim=move_response_dim,
            eps=spectrum_eps,
        )
        spectrum_feature_dim = (
            3 * move_response_dim
            + self.spectrum_pool.scalar_dim
        )
        self.head = MoveDeltaSpectrumHead(
            global_dim=global_dim,
            spectrum_feature_dim=spectrum_feature_dim,
            eigenvalue_dim=move_response_dim,
            hidden_dim=head_hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        board = self.adapter(x)
        _h_grid, h_sq, g = self.stem(x)
        moves = self.enumerator(board)
        responses = self.token_encoder(h_sq, g, moves)
        pool = self.spectrum_pool(responses, moves)

        scalars = torch.stack(
            [
                pool.trace,
                pool.leading_fraction,
                pool.participation_ratio,
                pool.spectral_entropy,
                pool.frobenius_norm,
            ],
            dim=1,
        )
        features = torch.cat(
            [
                g,
                pool.r_mean,
                pool.r_max,
                pool.r_var,
                pool.eigvals,
                scalars,
            ],
            dim=1,
        )
        logits = _format_logits(self.head(features), self.num_classes)
        capture_fraction = (
            ((moves.capture_id > 0) & moves.valid_mask).to(dtype=responses.dtype).sum(dim=1)
            / moves.move_count.clamp_min(1.0)
        )
        promotion_fraction = (
            ((moves.promo_id > 0) & moves.valid_mask).to(dtype=responses.dtype).sum(dim=1)
            / moves.move_count.clamp_min(1.0)
        )
        return {
            "logits": logits,
            "spectrum_trace": pool.trace,
            "spectrum_leading_fraction": pool.leading_fraction,
            "spectrum_participation_ratio": pool.participation_ratio,
            "spectrum_entropy": pool.spectral_entropy,
            "spectrum_frobenius_norm": pool.frobenius_norm,
            "spectrum_top_eigenvalue": pool.eigvals[:, 0],
            "spectrum_response_mean_norm": pool.r_mean.norm(dim=1),
            "spectrum_response_max_norm": pool.r_max.norm(dim=1),
            "spectrum_response_var_sum": pool.r_var.sum(dim=1),
            "pseudo_legal_move_count": moves.move_count,
            "capture_move_fraction": capture_fraction,
            "promotion_move_fraction": promotion_fraction,
            "trace_penalty_beta": logits.new_full((x.shape[0],), float(self.trace_penalty_beta)),
            "mechanism_energy": pool.leading_fraction,
            "proposal_profile_strength": pool.participation_ratio,
            "proposal_keyword_count": moves.move_count,
        }


def build_counterfactual_move_delta_spectrum_network_from_config(config: dict[str, Any]) -> MoveDeltaSpectrumNet:
    square_dim = int(config.get("square_dim", config.get("channels", 64)))
    global_dim = int(config.get("global_dim", max(96, 2 * square_dim)))
    return MoveDeltaSpectrumNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        encoding=str(config.get("encoding", "simple_18")),
        square_dim=square_dim,
        global_dim=global_dim,
        move_response_dim=int(config.get("move_response_dim", 16)),
        token_hidden_dim=int(config.get("token_hidden_dim", config.get("hidden_dim", 128))),
        head_hidden_dim=int(config.get("head_hidden_dim", config.get("hidden_dim", 128))),
        stem_depth=int(config.get("stem_depth", config.get("depth", 2))),
        max_moves=int(config.get("max_moves", 256)),
        dropout=float(config.get("dropout", 0.1)),
        trace_penalty_beta=float(config.get("trace_penalty_beta", 1.0e-4)),
        enable_castling_deltas=bool(config.get("enable_castling_deltas", False)),
        adapter_strict=bool(config.get("adapter_strict", True)),
        spectrum_eps=float(config.get("spectrum_eps", 1.0e-4)),
    )

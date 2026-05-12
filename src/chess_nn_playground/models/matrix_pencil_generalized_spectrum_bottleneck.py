"""Matrix-Pencil Generalized Spectrum Bottleneck model for idea i062.

Implements the markdown thesis under
``ideas/all_ideas/registry/i062_matrix_pencil_generalized_spectrum_bottleneck/``: puzzle-likeness
is tested by the **generalized eigenvalue spectrum of a learned PSD matrix
pair** ``(A(x), B(x))`` built from current-board occupied tokens.

For each board ``x`` the model constructs

    U_A(x), U_B(x) in R^{K x M}    (low-rank token summaries)
    A(x) = U_A^T U_A / K + eps * I_M
    B(x) = U_B^T U_B / K + eps * I_M

and computes the generalized eigenvalues solving ``A v = lambda B v`` via
the whitened symmetric form

    L = cholesky(B)
    Y = solve_triangular(L, A, upper=False)
    C = solve_triangular(L, Y^T, upper=False)^T
    C_sym = 0.5 * (C + C^T)
    lambda(x) = eigvalsh(C_sym)

The generalized spectrum, generalized Rayleigh quotients along learned probe
directions, separate spectra of ``A`` and ``B``, and matrix-norm summaries
form the feature vector consumed by the puzzle head.

The central operator is therefore Cholesky + symmetric eigendecomposition of
the whitened pencil ``L^{-1} A L^{-T}``, not convolution, residual stacking,
square attention, sheaf propagation, transport, or move enumeration.

Forward pipeline:

    Simple18OccupiedTokenExtractor  ->  (B, N_max, F) tokens, mask
    PieceSquareTokenEncoder         ->  (B, N_max, D) token embeddings h
    LowRankBoardMatrixPair          ->  (B, M, M) A, (B, M, M) B,
                                        (B, K, M) U_A, U_B
    GeneralizedSpectrumLayer        ->  (B, M) generalized eigenvalues,
                                        (B, P) Rayleigh probe quotients,
                                        separate spectra, norms, traces
    MatrixPencilHead                ->  (B,) puzzle logit + diagnostics

Section 9 falsifier ablations exposed via ``ablation``:

    * ``"none"``                       -- main model.
    * ``"separate_spectra_only"``      -- markdown's central falsifier:
      keep eigenvalues / norms / traces of ``A`` and ``B`` separately,
      remove the generalized eigenvalues and Rayleigh probes from the head.
    * ``"trace_ratio_only"``           -- collapse the head input to scalar
      ``tr(A)``, ``tr(B)``, ``tr(A)/tr(B)``, ``||A||_F``, ``||B||_F`` and
      a global broadcast (other features zeroed).
    * ``"batch_shuffled_b"``           -- pair each ``A(x)`` with a
      ``B(x')`` from a deterministic batch permutation so the matrix pair
      no longer comes from the same sample.
    * ``"random_factors"``             -- freeze the factor builders at
      initialization (no gradient through them) so only the head trains.
    * ``"single_matrix_spectrum"``     -- use only the eigenvalues / norms
      / trace of ``A`` (no ``B``, no pencil).
    * ``"mean_pool_head"``             -- bypass the pencil and feed
      mean / max / std token-pooled embeddings through a learned linear
      projection of the same dimensionality as the pencil features.
    * ``"material_only_tokens"``       -- zero the coordinate / castling /
      en-passant / own-flag features so only piece identity remains.

Engine, source, verification, and CRTK metadata are never used as input.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models._packet_bespoke_base import (
    BoardTensorSpec,
    require_board_tensor,
)


_PIECE_PLANES = 12
_MAX_PIECES = 32
_DEFAULT_TOKEN_DIM = 64
_DEFAULT_FACTOR_RANK = 16
_DEFAULT_MATRIX_DIM = 16
_DEFAULT_HEAD_HIDDEN = 128
_DEFAULT_PROBE_COUNT = 8
_DEFAULT_MATRIX_EPS = 1.0e-3
_DEFAULT_LOG_FLOOR = 1.0e-6
_TOKEN_FEATURE_DIM = 22
_VALID_ABLATIONS = {
    "none",
    "separate_spectra_only",
    "trace_ratio_only",
    "batch_shuffled_b",
    "random_factors",
    "single_matrix_spectrum",
    "mean_pool_head",
    "material_only_tokens",
}


@dataclass(frozen=True)
class ExtractedTokens:
    features: torch.Tensor    # (B, N_max, F)
    mask: torch.Tensor        # (B, N_max)
    occupancy: torch.Tensor   # (B, 64)
    side_to_move_white: torch.Tensor  # (B,)
    castling: torch.Tensor    # (B, 4)
    en_passant_file: torch.Tensor  # (B, 8)


class Simple18OccupiedTokenExtractor(nn.Module):
    """Decode simple_18 piece planes into up to ``max_tokens`` occupied tokens.

    Token features (deterministic, board-only):
        - 12 piece-color one-hot
        - 1 own/enemy flag (1 if same side as side-to-move)
        - 2 absolute coordinates (row/7, col/7)
        - 2 side-relative coordinates (mirrored row when black to move)
        - 4 castling broadcast flags
        - 1 en-passant flag (1 iff this square is the EP target)
    => 22 features per token.
    """

    feature_dim: int = _TOKEN_FEATURE_DIM

    def __init__(self, input_channels: int = 18, max_tokens: int = _MAX_PIECES) -> None:
        super().__init__()
        if input_channels < 18:
            raise ValueError(
                f"Simple18OccupiedTokenExtractor requires 18-plane simple_18 input, got {input_channels}"
            )
        self.input_channels = int(input_channels)
        self.max_tokens = int(max_tokens)
        self.spec = BoardTensorSpec(input_channels=self.input_channels)
        rows = torch.arange(8, dtype=torch.float32).view(8, 1).expand(8, 8) / 7.0
        cols = torch.arange(8, dtype=torch.float32).view(1, 8).expand(8, 8) / 7.0
        self.register_buffer("_rows", rows.reshape(64), persistent=False)
        self.register_buffer("_cols", cols.reshape(64), persistent=False)

    def forward(self, x: torch.Tensor) -> ExtractedTokens:
        require_board_tensor(x, self.spec)
        dtype = torch.float32
        x = x.to(dtype)
        batch = x.shape[0]
        piece_planes = x[:, :_PIECE_PLANES].clamp(0.0, 1.0)
        side_plane = x[:, 12].clamp(0.0, 1.0)
        side_white = (side_plane.mean(dim=(-1, -2)) > 0.5).to(dtype)
        castling = torch.stack(
            [x[:, 13].mean(dim=(-1, -2)),
             x[:, 14].mean(dim=(-1, -2)),
             x[:, 15].mean(dim=(-1, -2)),
             x[:, 16].mean(dim=(-1, -2))],
            dim=-1,
        ).clamp(0.0, 1.0)
        ep_plane = x[:, 17].clamp(0.0, 1.0)
        ep_files = ep_plane.amax(dim=-2)

        flat_planes = piece_planes.reshape(batch, _PIECE_PLANES, 64).transpose(1, 2)  # (B, 64, 12)
        occupancy = flat_planes.sum(dim=-1).clamp(0.0, 1.0)  # (B, 64)

        is_white_piece = flat_planes[..., :6].sum(dim=-1)
        is_black_piece = flat_planes[..., 6:12].sum(dim=-1)
        side = side_white.view(batch, 1)
        own_flag = side * is_white_piece + (1.0 - side) * is_black_piece

        rows = self._rows.view(1, 64).expand(batch, 64)
        cols = self._cols.view(1, 64).expand(batch, 64)
        rel_rows = side * rows + (1.0 - side) * (1.0 - rows)
        rel_cols = cols
        ep_per_square = ep_plane.reshape(batch, 64)
        castling_bcast = castling.unsqueeze(1).expand(batch, 64, 4)

        per_square = torch.cat(
            [
                flat_planes,
                own_flag.unsqueeze(-1),
                rows.unsqueeze(-1),
                cols.unsqueeze(-1),
                rel_rows.unsqueeze(-1),
                rel_cols.unsqueeze(-1),
                castling_bcast,
                ep_per_square.unsqueeze(-1),
            ],
            dim=-1,
        )
        assert per_square.shape[-1] == self.feature_dim

        topk = occupancy.topk(self.max_tokens, dim=-1)
        idx = topk.indices
        mask = (occupancy.gather(1, idx) > 0.5).to(dtype)
        gather_idx = idx.unsqueeze(-1).expand(batch, self.max_tokens, self.feature_dim)
        token_features = per_square.gather(1, gather_idx) * mask.unsqueeze(-1)

        return ExtractedTokens(
            features=token_features,
            mask=mask,
            occupancy=occupancy,
            side_to_move_white=side_white,
            castling=castling,
            en_passant_file=ep_files,
        )


class PieceSquareTokenEncoder(nn.Module):
    """Small MLP that maps token features to a ``token_dim``-d embedding."""

    def __init__(self, input_dim: int, token_dim: int = _DEFAULT_TOKEN_DIM, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = max(input_dim, token_dim)
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, token_dim))
        self.mlp = nn.Sequential(*layers)
        self.token_dim = int(token_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embedded = self.mlp(tokens)
        return embedded * mask.unsqueeze(-1)


@dataclass(frozen=True)
class MatrixPairState:
    matrix_a: torch.Tensor      # (B, M, M)
    matrix_b: torch.Tensor      # (B, M, M)
    factor_u_a: torch.Tensor    # (B, K, M)
    factor_u_b: torch.Tensor    # (B, K, M)


class LowRankBoardMatrixPair(nn.Module):
    """Build PSD matrix pair ``(A, B)`` from low-rank token summaries.

    For each branch ``r in {A, B}`` we learn

        weight_logits_r:   linear(h)              -> (B, N, K)
        value_r:           linear(h)              -> (B, N, M)
        weight_r           = softmax over occupied tokens (masked) of the logits
        U_r[k, :]          = sum_i weight_r[i, k] * value_r[i, :]
        Mat_r              = U_r^T U_r / K + eps * I_M

    Softmax over occupied tokens is permutation-invariant in token order
    (the sum over ``i`` does not depend on order) and ignores padded
    slots through the mask.
    """

    def __init__(
        self,
        token_dim: int,
        factor_rank: int = _DEFAULT_FACTOR_RANK,
        matrix_dim: int = _DEFAULT_MATRIX_DIM,
        matrix_eps: float = _DEFAULT_MATRIX_EPS,
    ) -> None:
        super().__init__()
        if factor_rank < 1:
            raise ValueError("factor_rank must be >= 1")
        if matrix_dim < 2:
            raise ValueError("matrix_dim must be >= 2 to define a non-trivial pencil")
        if matrix_eps <= 0:
            raise ValueError("matrix_eps must be > 0 to keep B positive definite")
        self.token_dim = int(token_dim)
        self.factor_rank = int(factor_rank)
        self.matrix_dim = int(matrix_dim)
        self.matrix_eps = float(matrix_eps)
        self.weight_a = nn.Linear(self.token_dim, self.factor_rank)
        self.weight_b = nn.Linear(self.token_dim, self.factor_rank)
        self.value_a = nn.Linear(self.token_dim, self.matrix_dim)
        self.value_b = nn.Linear(self.token_dim, self.matrix_dim)

    def _factor(
        self,
        weight_logits: torch.Tensor,   # (B, N, K)
        values: torch.Tensor,          # (B, N, M)
        mask: torch.Tensor,            # (B, N)
    ) -> torch.Tensor:
        masked = weight_logits.masked_fill(~mask.bool().unsqueeze(-1), float("-inf"))
        # If a sample has zero occupied tokens (impossible for legal boards,
        # but covered for safety), softmax of -inf would be NaN. Replace with
        # uniform weights over the (still-zero) values, which yields zeros.
        no_active = (~mask.bool().any(dim=-1)).view(-1, 1, 1)
        masked = torch.where(no_active.expand_as(masked), torch.zeros_like(masked), masked)
        weights = F.softmax(masked, dim=1)
        # Re-zero the all-padded rows so they contribute nothing.
        weights = torch.where(no_active.expand_as(weights), torch.zeros_like(weights), weights)
        # einsum over occupied tokens -> (B, K, M)
        u = torch.einsum("bnk,bnm->bkm", weights, values * mask.unsqueeze(-1))
        return u

    def forward(self, token_embed: torch.Tensor, mask: torch.Tensor) -> MatrixPairState:
        if token_embed.shape[-1] != self.token_dim:
            raise ValueError(
                f"Expected token_dim={self.token_dim}, got {token_embed.shape[-1]}"
            )
        weight_logits_a = self.weight_a(token_embed)
        weight_logits_b = self.weight_b(token_embed)
        values_a = self.value_a(token_embed)
        values_b = self.value_b(token_embed)
        u_a = self._factor(weight_logits_a, values_a, mask)
        u_b = self._factor(weight_logits_b, values_b, mask)

        eye = torch.eye(self.matrix_dim, device=token_embed.device, dtype=token_embed.dtype)
        scale = 1.0 / float(self.factor_rank)
        a = torch.matmul(u_a.transpose(-1, -2), u_a) * scale
        b = torch.matmul(u_b.transpose(-1, -2), u_b) * scale
        a = 0.5 * (a + a.transpose(-1, -2)) + self.matrix_eps * eye
        b = 0.5 * (b + b.transpose(-1, -2)) + self.matrix_eps * eye
        return MatrixPairState(matrix_a=a, matrix_b=b, factor_u_a=u_a, factor_u_b=u_b)


@dataclass(frozen=True)
class GeneralizedSpectrumPack:
    generalized_eigvals: torch.Tensor      # (B, M) descending
    log_generalized_eigvals: torch.Tensor  # (B, M)
    eigvals_a: torch.Tensor                # (B, M) descending
    eigvals_b: torch.Tensor                # (B, M) descending
    rayleigh_probes: torch.Tensor          # (B, P)
    matrix_a: torch.Tensor                 # (B, M, M)
    matrix_b: torch.Tensor                 # (B, M, M)


class GeneralizedSpectrumLayer(nn.Module):
    """Compute the generalized eigenvalue spectrum of ``(A, B)``.

    Solves the symmetric-definite generalized eigenproblem ``A v = lambda B v``
    by whitening with the Cholesky factor of ``B``:

        L = cholesky(B)
        Y = solve_triangular(L, A, upper=False)
        C = solve_triangular(L, Y.transpose(-1, -2), upper=False).transpose(-1, -2)
        C_sym = 0.5 * (C + C.transpose(-1, -2))
        eigvals = eigvalsh(C_sym)

    Optionally also returns generalized Rayleigh quotients
    ``R(z) = z^T A z / z^T B z`` along learned probe directions ``z``.
    """

    def __init__(
        self,
        matrix_dim: int,
        probe_count: int = _DEFAULT_PROBE_COUNT,
        include_rayleigh_probes: bool = True,
    ) -> None:
        super().__init__()
        if matrix_dim < 2:
            raise ValueError("matrix_dim must be >= 2")
        self.matrix_dim = int(matrix_dim)
        self.include_rayleigh_probes = bool(include_rayleigh_probes)
        self.probe_count = int(probe_count) if include_rayleigh_probes else 0
        if self.probe_count > 0:
            probes = torch.randn(self.probe_count, self.matrix_dim) * 0.1
            self.probes = nn.Parameter(probes)
        else:
            self.register_parameter("probes", None)

    def forward(self, pair: MatrixPairState) -> GeneralizedSpectrumPack:
        a = pair.matrix_a
        b = pair.matrix_b
        # Cholesky of B (PD by construction).
        l = torch.linalg.cholesky(b)
        # Solve L @ Y = A   =>   Y = L^{-1} A
        y = torch.linalg.solve_triangular(l, a, upper=False)
        # Solve L @ Z = Y^T  =>  Z = L^{-1} A L^{-T}, returned via transpose
        c = torch.linalg.solve_triangular(l, y.transpose(-1, -2), upper=False).transpose(-1, -2)
        c_sym = 0.5 * (c + c.transpose(-1, -2))
        # eigvalsh returns ascending; flip to descending so spectrum[0] is dominant.
        gen_eigvals = torch.linalg.eigvalsh(c_sym).flip(-1)
        log_gen = torch.log(gen_eigvals.clamp_min(_DEFAULT_LOG_FLOOR))

        eigvals_a = torch.linalg.eigvalsh(a).flip(-1)
        eigvals_b = torch.linalg.eigvalsh(b).flip(-1)

        if self.probe_count > 0 and self.probes is not None:
            # Normalize learned probes so each is unit-norm; (P, M).
            z = F.normalize(self.probes, dim=-1, eps=1.0e-8)
            # quad_a[b, p] = z[p]^T A[b] z[p]
            za = torch.einsum("pm,bmn,pn->bp", z, a, z)
            zb = torch.einsum("pm,bmn,pn->bp", z, b, z)
            rayleigh = za / zb.clamp_min(_DEFAULT_LOG_FLOOR)
        else:
            rayleigh = a.new_zeros(a.shape[0], 0)

        return GeneralizedSpectrumPack(
            generalized_eigvals=gen_eigvals,
            log_generalized_eigvals=log_gen,
            eigvals_a=eigvals_a,
            eigvals_b=eigvals_b,
            rayleigh_probes=rayleigh,
            matrix_a=a,
            matrix_b=b,
        )


class MatrixPencilHead(nn.Module):
    """LayerNorm + 2-layer MLP over pencil / spectrum / global features."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = _DEFAULT_HEAD_HIDDEN,
        num_classes: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, max(1, int(num_classes))))
        self.mlp = nn.Sequential(*layers)
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features)


class MatrixPencilGeneralizedSpectrumNet(nn.Module):
    """Complete bespoke architecture for idea i062.

    The central operator is **batched Cholesky + symmetric eigendecomposition
    of the whitened pencil matrix** ``L^{-1} A L^{-T}``, with optional
    generalized Rayleigh quotients ``z^T A z / z^T B z`` along learned probe
    directions, paired with separate spectra of ``A`` and ``B`` for the
    section-9 ablations. The model is permutation-invariant over occupied
    tokens (factor builders use a sum after softmax).
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        token_dim: int = _DEFAULT_TOKEN_DIM,
        factor_rank: int = _DEFAULT_FACTOR_RANK,
        matrix_dim: int = _DEFAULT_MATRIX_DIM,
        head_hidden: int = _DEFAULT_HEAD_HIDDEN,
        matrix_eps: float = _DEFAULT_MATRIX_EPS,
        probe_count: int = _DEFAULT_PROBE_COUNT,
        include_separate_spectra: bool = True,
        include_rayleigh_probes: bool = True,
        max_tokens: int = _MAX_PIECES,
        ablation: str = "none",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        ablation = (ablation or "none").lower()
        if ablation not in _VALID_ABLATIONS:
            raise ValueError(
                f"Unsupported matrix_pencil ablation {ablation!r}; "
                f"expected one of {sorted(_VALID_ABLATIONS)}"
            )
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.token_dim = int(token_dim)
        self.factor_rank = int(factor_rank)
        self.matrix_dim = int(matrix_dim)
        self.max_tokens = int(max_tokens)
        self.include_separate_spectra = bool(include_separate_spectra)
        self.include_rayleigh_probes = bool(include_rayleigh_probes)
        self.probe_count = int(probe_count) if include_rayleigh_probes else 0
        self.ablation = ablation

        self.token_extractor = Simple18OccupiedTokenExtractor(
            input_channels=input_channels,
            max_tokens=max_tokens,
        )
        self.token_encoder = PieceSquareTokenEncoder(
            input_dim=self.token_extractor.feature_dim,
            token_dim=self.token_dim,
            dropout=dropout,
        )
        self.matrix_pair = LowRankBoardMatrixPair(
            token_dim=self.token_dim,
            factor_rank=self.factor_rank,
            matrix_dim=self.matrix_dim,
            matrix_eps=matrix_eps,
        )
        self.spectrum = GeneralizedSpectrumLayer(
            matrix_dim=self.matrix_dim,
            probe_count=self.probe_count,
            include_rayleigh_probes=self.include_rayleigh_probes,
        )

        # Feature layout fed to the head:
        #   generalized eigenvalues (M) + log generalized eigenvalues (M)   -> 2 * M
        #   spread / cond ratio / trace ratio / Frobenius norms             -> 5
        #   separate eigenvalues of A (M) + B (M) (descending)              -> 2 * M
        #   diag(A) + diag(B)                                                -> 2 * M
        #   Rayleigh probes (P)                                              -> P
        #   global broadcast: side-to-move + 4 castling + 8 EP +              -> 14
        #     active_count_norm
        self.pencil_feature_dim = 2 * self.matrix_dim + 5
        self.separate_feature_dim = 4 * self.matrix_dim if self.include_separate_spectra else 0
        self.probe_feature_dim = self.probe_count
        self.global_feature_dim = 1 + 4 + 8 + 1
        self.feature_dim = (
            self.pencil_feature_dim
            + self.separate_feature_dim
            + self.probe_feature_dim
            + self.global_feature_dim
        )
        self.head = MatrixPencilHead(
            feature_dim=self.feature_dim,
            hidden_dim=int(head_hidden),
            num_classes=self.num_classes,
            dropout=float(dropout),
        )

        # mean_pool_head ablation: deterministic projection from
        # mean / max / std token-pooled embeddings to the same head input
        # dimensionality minus the global broadcast vector.
        non_global_dim = self.feature_dim - self.global_feature_dim
        self.pooled_token_proj = nn.Linear(3 * self.token_dim, non_global_dim)

        if self.ablation == "random_factors":
            for module in (
                self.token_encoder,
                self.matrix_pair,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _global_features(self, tokens: ExtractedTokens) -> torch.Tensor:
        active_norm = tokens.mask.sum(dim=-1, keepdim=True) / float(self.max_tokens)
        return torch.cat(
            [
                tokens.side_to_move_white.unsqueeze(-1),
                tokens.castling,
                tokens.en_passant_file,
                active_norm,
            ],
            dim=-1,
        )

    def _pencil_summary(
        self,
        gen_eig: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        # spread, condition-like ratio, trace ratio, ||A||_F, ||B||_F
        spread = gen_eig.amax(dim=-1) - gen_eig.amin(dim=-1)
        cond = gen_eig.amax(dim=-1) / gen_eig.amin(dim=-1).clamp_min(_DEFAULT_LOG_FLOOR)
        trace_a = torch.diagonal(a, dim1=-2, dim2=-1).sum(dim=-1)
        trace_b = torch.diagonal(b, dim1=-2, dim2=-1).sum(dim=-1)
        trace_ratio = trace_a / trace_b.clamp_min(_DEFAULT_LOG_FLOOR)
        fro_a = torch.linalg.matrix_norm(a, ord="fro")
        fro_b = torch.linalg.matrix_norm(b, ord="fro")
        return torch.stack([spread, cond, trace_ratio, fro_a, fro_b], dim=-1)

    def _build_features(
        self,
        pack: GeneralizedSpectrumPack,
    ) -> torch.Tensor:
        gen = pack.generalized_eigvals
        log_gen = pack.log_generalized_eigvals
        summary = self._pencil_summary(gen, pack.matrix_a, pack.matrix_b)
        pencil = torch.cat([gen, log_gen, summary], dim=-1)

        if self.include_separate_spectra:
            diag_a = torch.diagonal(pack.matrix_a, dim1=-2, dim2=-1)
            diag_b = torch.diagonal(pack.matrix_b, dim1=-2, dim2=-1)
            separate = torch.cat([pack.eigvals_a, pack.eigvals_b, diag_a, diag_b], dim=-1)
        else:
            separate = pencil.new_zeros(pencil.shape[0], 0)

        if self.probe_feature_dim > 0:
            probes = pack.rayleigh_probes
        else:
            probes = pencil.new_zeros(pencil.shape[0], 0)

        return torch.cat([pencil, separate, probes], dim=-1)

    def _apply_ablation(
        self,
        features: torch.Tensor,
        token_embed: torch.Tensor,
        mask: torch.Tensor,
        pack: GeneralizedSpectrumPack,
    ) -> torch.Tensor:
        if self.ablation == "none" or self.ablation == "random_factors":
            return features
        batch = features.shape[0]
        device = features.device
        dtype = features.dtype

        if self.ablation == "separate_spectra_only":
            # zero pencil block and Rayleigh probes; keep separate spectra.
            new_features = features.clone()
            new_features[:, : self.pencil_feature_dim] = 0.0
            if self.probe_feature_dim > 0:
                end_separate = self.pencil_feature_dim + self.separate_feature_dim
                new_features[:, end_separate : end_separate + self.probe_feature_dim] = 0.0
            return new_features

        if self.ablation == "trace_ratio_only":
            # zero everything except scalar norms / traces in the pencil
            # summary block. The summary occupies positions
            # [2M : 2M + 5] within the pencil block.
            new_features = features.new_zeros(features.shape)
            summary_start = 2 * self.matrix_dim
            new_features[:, summary_start : summary_start + 5] = features[
                :, summary_start : summary_start + 5
            ]
            return new_features

        if self.ablation == "single_matrix_spectrum":
            new_features = features.new_zeros(features.shape)
            if self.include_separate_spectra:
                # Keep eigvals_a and diag_a from the separate block.
                start = self.pencil_feature_dim
                new_features[:, start : start + self.matrix_dim] = features[
                    :, start : start + self.matrix_dim
                ]
                diag_a_start = self.pencil_feature_dim + 2 * self.matrix_dim
                new_features[:, diag_a_start : diag_a_start + self.matrix_dim] = features[
                    :, diag_a_start : diag_a_start + self.matrix_dim
                ]
            # Keep ||A||_F (position 2M + 3 in pencil block).
            new_features[:, 2 * self.matrix_dim + 3] = features[:, 2 * self.matrix_dim + 3]
            return new_features

        if self.ablation == "batch_shuffled_b":
            # Permute the contributions from B across the batch so the
            # generalized spectrum no longer matches the sample's own A.
            # We rebuild the spectrum with shuffled B then rebuild features.
            if batch <= 1:
                return features
            gen = torch.Generator(device="cpu").manual_seed(0xB0BB1E)
            permutation = torch.randperm(batch, generator=gen).to(device)
            if torch.equal(permutation, torch.arange(batch, device=device)):
                permutation = torch.roll(permutation, shifts=1, dims=0)
            shuffled_b = pack.matrix_b.index_select(0, permutation)
            shuffled_pack = self._recompute_pack(pack.matrix_a, shuffled_b)
            return self._build_features(shuffled_pack)

        if self.ablation == "mean_pool_head":
            # Replace pencil + separate + probes with a learned projection of
            # mean / max / std token pools.
            mask_f = mask.unsqueeze(-1)
            denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            mean_pool = (token_embed * mask_f).sum(dim=1) / denom
            very_neg = torch.full_like(token_embed, -1.0e9)
            masked_for_max = torch.where(mask_f.bool(), token_embed, very_neg)
            max_pool = masked_for_max.amax(dim=1)
            max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
            sq_mean = ((token_embed * mask_f) ** 2).sum(dim=1) / denom
            std_pool = (sq_mean - mean_pool ** 2).clamp_min(0.0).sqrt()
            pooled = torch.cat([mean_pool, max_pool, std_pool], dim=-1)
            non_global_dim = self.feature_dim - self.global_feature_dim
            new_features = features.new_zeros(features.shape)
            new_features[:, :non_global_dim] = self.pooled_token_proj(pooled)
            return new_features

        # material_only_tokens is handled upstream by zeroing non-piece
        # token features before building matrices, so by the time we get
        # here the spectrum already reflects the ablation.
        return features

    def _recompute_pack(self, a: torch.Tensor, b: torch.Tensor) -> GeneralizedSpectrumPack:
        """Recompute the spectrum from explicit (A, B) without re-pooling."""
        l = torch.linalg.cholesky(b)
        y = torch.linalg.solve_triangular(l, a, upper=False)
        c = torch.linalg.solve_triangular(l, y.transpose(-1, -2), upper=False).transpose(-1, -2)
        c_sym = 0.5 * (c + c.transpose(-1, -2))
        gen = torch.linalg.eigvalsh(c_sym).flip(-1)
        log_gen = torch.log(gen.clamp_min(_DEFAULT_LOG_FLOOR))
        eigvals_a = torch.linalg.eigvalsh(a).flip(-1)
        eigvals_b = torch.linalg.eigvalsh(b).flip(-1)
        if self.probe_count > 0 and self.spectrum.probes is not None:
            z = F.normalize(self.spectrum.probes, dim=-1, eps=1.0e-8)
            za = torch.einsum("pm,bmn,pn->bp", z, a, z)
            zb = torch.einsum("pm,bmn,pn->bp", z, b, z)
            rayleigh = za / zb.clamp_min(_DEFAULT_LOG_FLOOR)
        else:
            rayleigh = a.new_zeros(a.shape[0], 0)
        return GeneralizedSpectrumPack(
            generalized_eigvals=gen,
            log_generalized_eigvals=log_gen,
            eigvals_a=eigvals_a,
            eigvals_b=eigvals_b,
            rayleigh_probes=rayleigh,
            matrix_a=a,
            matrix_b=b,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.token_extractor(x)
        token_features = tokens.features
        if self.ablation == "material_only_tokens":
            # Keep only the 12 piece-color one-hot dimensions; zero own_flag,
            # absolute / side-relative coordinates, castling broadcasts and
            # the en-passant flag.
            material_only = token_features.clone()
            material_only[:, :, _PIECE_PLANES:] = 0.0
            token_features = material_only
        token_embed = self.token_encoder(token_features, tokens.mask)
        pair = self.matrix_pair(token_embed, tokens.mask)
        pack = self.spectrum(pair)

        features = self._build_features(pack)
        non_global = self._apply_ablation(features, token_embed, tokens.mask, pack)
        global_features = self._global_features(tokens)
        head_input = torch.cat([non_global, global_features], dim=-1)
        raw_logits = self.head(head_input)

        if self.num_classes == 1:
            logits = raw_logits.view(-1)
            two_class = torch.stack([-0.5 * logits, 0.5 * logits], dim=-1)
        else:
            logits = raw_logits
            two_class = raw_logits if raw_logits.shape[-1] >= 2 else logits

        # Diagnostic proxies.
        gen = pack.generalized_eigvals
        condition_b = pack.eigvals_b.amax(dim=-1) / pack.eigvals_b.amin(dim=-1).clamp_min(_DEFAULT_LOG_FLOOR)
        trace_a = torch.diagonal(pack.matrix_a, dim1=-2, dim2=-1).sum(dim=-1)
        trace_b = torch.diagonal(pack.matrix_b, dim1=-2, dim2=-1).sum(dim=-1)
        trace_ratio = trace_a / trace_b.clamp_min(_DEFAULT_LOG_FLOOR)
        a_norm = pack.matrix_a / trace_a.clamp_min(_DEFAULT_LOG_FLOOR).view(-1, 1, 1)
        b_norm = pack.matrix_b / trace_b.clamp_min(_DEFAULT_LOG_FLOOR).view(-1, 1, 1)
        proportionality = torch.linalg.matrix_norm(a_norm - b_norm, ord="fro")
        mechanism_energy = gen.std(dim=-1, unbiased=False)

        diagnostics: dict[str, torch.Tensor] = {
            "logits": logits,
            "two_class_logits": two_class,
            "generalized_eigenvalues": pack.generalized_eigvals,
            "log_generalized_eigenvalues": pack.log_generalized_eigvals,
            "rayleigh_probes": pack.rayleigh_probes,
            "matrix_a": pack.matrix_a,
            "matrix_b": pack.matrix_b,
            "eigenvalues_a": pack.eigvals_a,
            "eigenvalues_b": pack.eigvals_b,
            "trace_a": trace_a,
            "trace_b": trace_b,
            "trace_ratio": trace_ratio,
            "condition_b": condition_b,
            "proportionality_diagnostic": proportionality,
            "mechanism_energy": mechanism_energy,
            "active_token_count": tokens.mask.sum(dim=-1),
            "ablation_active": torch.full(
                (logits.shape[0],),
                1.0 if self.ablation != "none" else 0.0,
                device=logits.device,
                dtype=logits.dtype,
            ),
        }
        return diagnostics


def build_matrix_pencil_generalized_spectrum_bottleneck_from_config(
    config: dict[str, Any],
) -> MatrixPencilGeneralizedSpectrumNet:
    cfg = dict(config)
    head_hidden = cfg.get("head_hidden", cfg.get("hidden_dim", _DEFAULT_HEAD_HIDDEN))
    return MatrixPencilGeneralizedSpectrumNet(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        token_dim=int(cfg.get("token_dim", _DEFAULT_TOKEN_DIM)),
        factor_rank=int(cfg.get("factor_rank", _DEFAULT_FACTOR_RANK)),
        matrix_dim=int(cfg.get("matrix_dim", _DEFAULT_MATRIX_DIM)),
        head_hidden=int(head_hidden),
        matrix_eps=float(cfg.get("matrix_eps", _DEFAULT_MATRIX_EPS)),
        probe_count=int(cfg.get("probe_count", _DEFAULT_PROBE_COUNT)),
        include_separate_spectra=bool(cfg.get("include_separate_spectra", True)),
        include_rayleigh_probes=bool(cfg.get("include_rayleigh_probes", True)),
        max_tokens=int(cfg.get("max_tokens", _MAX_PIECES)),
        ablation=str(cfg.get("ablation", "none")),
        dropout=float(cfg.get("dropout", 0.0)),
    )

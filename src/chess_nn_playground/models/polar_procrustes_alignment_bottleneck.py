"""Polar-Procrustes Alignment Bottleneck model for idea i063.

Implements the markdown thesis under
``ideas/i063_polar_procrustes_alignment_bottleneck/``: puzzle-likeness
is tested by **the orthogonal Procrustes alignment between learned
own/opponent role matrices and the polar-decomposition strain spectrum
of their cross-covariance**.

For each board ``x`` the model splits occupied tokens into
side-to-move-relative own / opponent sets, encodes them with a shared
MLP, pools each side into a role matrix

    X(x) in R^{R x D}    (own role matrix)
    Y(x) in R^{R x D}    (opponent role matrix)

via learned masked-softmax role queries, optionally row-normalises
``X``, ``Y`` and computes the cross-covariance

    C(x) = X^T Y / R     in R^{D x D}    (matrix_space = "embedding")
    C(x) = X Y^T / D     in R^{R x R}    (matrix_space = "role")

The orthogonal Procrustes problem ``min_{Q^T Q = I} ||X Q - Y||_F`` is
solved through the singular value decomposition

    C = U Sigma V^T
    Q_star = U V^T
    H      = V Sigma V^T            (symmetric polar factor / strain)

so the polar decomposition is ``C = Q_star H``. The Procrustes residual,
the alignment improvement over identity, the singular spectrum, the
diagonal of ``H`` and the separate row-norms / singular values of ``X``
and ``Y`` form the feature vector consumed by the puzzle head.

The central operator is therefore SVD of the cross-covariance
``C(x) = X(x)^T Y(x)``, not convolution, residual stacking, attention,
sheaf propagation, transport, generalized eigenvalues, principal
angles, or move enumeration.

Forward pipeline:

    Simple18OwnOpponentTokenExtractor -> (B, N_max, F) tokens, mask,
                                          own_mask, opp_mask
    PieceSquareTokenEncoder           -> (B, N_max, D) embeddings h
    RoleMatrixPooler                  -> (B, R, D) X (own), (B, R, D) Y
    PolarProcrustesLayer              -> (B, R or D, R or D) Q_star,
                                          (B, K) singular values,
                                          residual stats
    PolarProcrustesHead               -> (B,) puzzle logit + diagnostics

Section 9 falsifier ablations exposed via ``ablation``:

    * ``"none"``                            -- main model.
    * ``"separate_matrix_stats_only"``      -- markdown's central
      falsifier: keep separate row norms / singular values / role mass
      of ``X`` and ``Y`` but zero the cross-covariance / Procrustes
      block fed to the head.
    * ``"identity_alignment_only"``         -- replace ``Q_star`` with
      the identity, so only ``||X - Y||_F`` and identity-aligned per-
      role residuals enter the head.
    * ``"random_orthogonal_alignment"``     -- replace the learned
      sample-specific ``Q_star`` with a deterministic batch-shared
      random orthogonal matrix.
    * ``"batch_shuffled_opponent"``         -- pair ``X(x)`` with
      ``Y(x')`` from a deterministic batch permutation before
      computing ``C`` and the residuals.
    * ``"material_only_matrices"``          -- zero the coordinate /
      castling / en-passant / own-flag features in each token before
      encoding so only piece-color identity remains.
    * ``"role_pool_mean_only"``             -- replace learned role
      queries with a deterministic mean / max / std side-wise pool
      tiled across the same role matrix shape.
    * ``"singular_values_only"``            -- keep only the singular
      values of ``C`` (and the polar diagonal); zero residual /
      improvement / Q_star-derived features.

Engine, source, verification, and CRTK metadata are never used as
input.
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
_DEFAULT_TOKEN_DIM = 48
_DEFAULT_ROLE_COUNT = 8
_DEFAULT_HEAD_HIDDEN = 128
_DEFAULT_MATRIX_EPS = 1.0e-3
_DEFAULT_LOG_FLOOR = 1.0e-6
_TOKEN_FEATURE_DIM = 22
_VALID_MATRIX_SPACES = {"embedding", "role"}
_VALID_ABLATIONS = {
    "none",
    "separate_matrix_stats_only",
    "identity_alignment_only",
    "random_orthogonal_alignment",
    "batch_shuffled_opponent",
    "material_only_matrices",
    "role_pool_mean_only",
    "singular_values_only",
}


@dataclass(frozen=True)
class ExtractedTokens:
    features: torch.Tensor          # (B, N_max, F)
    mask: torch.Tensor              # (B, N_max)
    own_mask: torch.Tensor          # (B, N_max)
    opp_mask: torch.Tensor          # (B, N_max)
    occupancy: torch.Tensor         # (B, 64)
    side_to_move_white: torch.Tensor  # (B,)
    castling: torch.Tensor          # (B, 4)
    en_passant_file: torch.Tensor   # (B, 8)


class Simple18OwnOpponentTokenExtractor(nn.Module):
    """Decode simple_18 piece planes into up to ``max_tokens`` occupied tokens
    with side-to-move-relative own / opponent flags.

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
                f"Simple18OwnOpponentTokenExtractor requires 18-plane simple_18 input, "
                f"got {input_channels}"
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

        flat_planes = piece_planes.reshape(batch, _PIECE_PLANES, 64).transpose(1, 2)
        occupancy = flat_planes.sum(dim=-1).clamp(0.0, 1.0)

        is_white_piece = flat_planes[..., :6].sum(dim=-1)
        is_black_piece = flat_planes[..., 6:12].sum(dim=-1)
        side = side_white.view(batch, 1)
        own_per_square = side * is_white_piece + (1.0 - side) * is_black_piece
        opp_per_square = side * is_black_piece + (1.0 - side) * is_white_piece

        rows = self._rows.view(1, 64).expand(batch, 64)
        cols = self._cols.view(1, 64).expand(batch, 64)
        rel_rows = side * rows + (1.0 - side) * (1.0 - rows)
        rel_cols = cols
        ep_per_square = ep_plane.reshape(batch, 64)
        castling_bcast = castling.unsqueeze(1).expand(batch, 64, 4)

        per_square = torch.cat(
            [
                flat_planes,
                own_per_square.unsqueeze(-1),
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
        own_mask = own_per_square.gather(1, idx) * mask
        opp_mask = opp_per_square.gather(1, idx) * mask

        return ExtractedTokens(
            features=token_features,
            mask=mask,
            own_mask=own_mask,
            opp_mask=opp_mask,
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
class RoleMatrixState:
    own_matrix: torch.Tensor    # (B, R, D)
    opp_matrix: torch.Tensor    # (B, R, D)
    own_mass: torch.Tensor      # (B, R)
    opp_mass: torch.Tensor      # (B, R)


class RoleMatrixPooler(nn.Module):
    """Pool side-tagged token embeddings into role matrices.

    For each side we build per-role weights via masked softmax over its
    occupied tokens and a learned per-side query MLP, then summarise
    each role as a weighted sum of token embeddings:

        weight_logits_r(h)  = mlp_r(h)               (B, N, R)
        weight_r            = softmax_n(masked logits)
        side_matrix[r, :]   = sum_n weight_r[n] * h_n

    Softmax sums to one over occupied tokens, so role rows mix the same
    embedding components but with side-specific weights. Padded /
    wrong-side tokens are masked out before the softmax so they cannot
    influence the role row.

    The pooler is permutation-invariant in token order and shares
    nothing across own / opponent except the embedding it consumes.
    """

    def __init__(
        self,
        token_dim: int,
        role_count: int = _DEFAULT_ROLE_COUNT,
    ) -> None:
        super().__init__()
        if role_count < 2:
            raise ValueError("role_count must be >= 2 to define a non-trivial Procrustes problem")
        self.token_dim = int(token_dim)
        self.role_count = int(role_count)
        self.own_query = nn.Linear(self.token_dim, self.role_count)
        self.opp_query = nn.Linear(self.token_dim, self.role_count)

    @staticmethod
    def _masked_softmax_pool(
        token_embed: torch.Tensor,    # (B, N, D)
        weight_logits: torch.Tensor,  # (B, N, R)
        side_mask: torch.Tensor,      # (B, N)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        active = (side_mask > 0.5).bool()                # (B, N)
        any_active = active.any(dim=-1)                  # (B,)
        # Mask out tokens not on this side via a large negative bias.
        bias = (1.0 - active.float()).unsqueeze(-1) * -1.0e9
        masked_logits = weight_logits + bias
        # Samples with no active tokens on this side: replace -inf with zeros so
        # softmax is well-defined; the resulting row is uniform across roles
        # but the side mass below is zero, so the role row stays at zero.
        no_active = (~any_active).view(-1, 1, 1).expand_as(masked_logits)
        masked_logits = torch.where(no_active, torch.zeros_like(masked_logits), masked_logits)
        weights = F.softmax(masked_logits, dim=1)        # (B, N, R)
        weights = weights * active.unsqueeze(-1).to(weights.dtype)
        # Re-zero rows where no active tokens existed.
        weights = torch.where(no_active, torch.zeros_like(weights), weights)
        # role row: einsum over occupied tokens
        side_matrix = torch.einsum("bnr,bnd->brd", weights, token_embed)
        side_mass = weights.sum(dim=1)                   # (B, R)
        return side_matrix, side_mass

    def forward(
        self,
        token_embed: torch.Tensor,     # (B, N, D)
        own_mask: torch.Tensor,        # (B, N)
        opp_mask: torch.Tensor,        # (B, N)
    ) -> RoleMatrixState:
        if token_embed.shape[-1] != self.token_dim:
            raise ValueError(
                f"Expected token_dim={self.token_dim}, got {token_embed.shape[-1]}"
            )
        own_logits = self.own_query(token_embed)
        opp_logits = self.opp_query(token_embed)
        own_matrix, own_mass = self._masked_softmax_pool(token_embed, own_logits, own_mask)
        opp_matrix, opp_mass = self._masked_softmax_pool(token_embed, opp_logits, opp_mask)
        return RoleMatrixState(
            own_matrix=own_matrix,
            opp_matrix=opp_matrix,
            own_mass=own_mass,
            opp_mass=opp_mass,
        )


def _row_layer_norm(matrix: torch.Tensor, eps: float = 1.0e-5) -> torch.Tensor:
    """LayerNorm-style per-row centering and unit-variance scaling.

    Shape preserved: (B, R, D) -> (B, R, D).
    """
    mean = matrix.mean(dim=-1, keepdim=True)
    var = matrix.var(dim=-1, keepdim=True, unbiased=False)
    return (matrix - mean) / torch.sqrt(var + eps)


@dataclass(frozen=True)
class PolarProcrustesPack:
    cross_cov: torch.Tensor             # (B, M, M) where M is D or R
    q_star: torch.Tensor                # (B, M, M) orthogonal
    singular_values: torch.Tensor       # (B, K) descending, K = min(M, M) = M
    polar_diag: torch.Tensor            # (B, M) diagonal of H = V Sigma V^T
    procrustes_residual: torch.Tensor   # (B,) ||X Q* - Y||_F
    identity_residual: torch.Tensor     # (B,) ||X - Y||_F
    alignment_improvement: torch.Tensor # (B,) identity - procrustes residual
    per_role_residual: torch.Tensor     # (B, R)
    nuclear_norm: torch.Tensor          # (B,)
    spectral_norm: torch.Tensor         # (B,)
    stable_rank: torch.Tensor           # (B,)
    x_norm: torch.Tensor                # (B,)
    y_norm: torch.Tensor                # (B,)
    x_singular_values: torch.Tensor     # (B, R)
    y_singular_values: torch.Tensor     # (B, R)


class PolarProcrustesLayer(nn.Module):
    """Compute the optimal orthogonal alignment of own / opp role matrices.

    Operates on row-normalised role matrices ``X`` and ``Y`` of shape
    ``(B, R, D)``. Builds the cross-covariance

        C = X^T Y / R     (matrix_space = "embedding", shape (B, D, D))
        C = X Y^T / D     (matrix_space = "role",      shape (B, R, R))

    factors ``C = U Sigma V^T`` and recovers the polar pieces

        Q* = U V^T                    (B, M, M) orthogonal
        H  = V Sigma V^T              (B, M, M) symmetric PSD strain

    The Procrustes residual ``||X Q* - Y||_F`` is computed in
    embedding-space; for ``matrix_space = "role"`` we apply ``Q*`` from
    the left of ``X`` (``Q* @ X``) so the alignment acts on roles.

    Numerical stability:
        We add a tiny diagonal tilt ``cross_cov_eps * diag(1, 2, ..., M) / M``
        to ``C`` before the SVD. This breaks any otherwise-coincident
        singular values so ``torch.linalg.svd`` backward stays finite,
        without changing qualitative behaviour. A ``cross_cov_eps``
        of ``1e-3`` is well below the typical signal scale.
    """

    def __init__(
        self,
        token_dim: int,
        role_count: int,
        matrix_space: str = "embedding",
        cross_cov_eps: float = _DEFAULT_MATRIX_EPS,
    ) -> None:
        super().__init__()
        matrix_space = (matrix_space or "embedding").lower()
        if matrix_space not in _VALID_MATRIX_SPACES:
            raise ValueError(
                f"matrix_space must be one of {sorted(_VALID_MATRIX_SPACES)}, got {matrix_space!r}"
            )
        if cross_cov_eps <= 0:
            raise ValueError("cross_cov_eps must be > 0")
        self.token_dim = int(token_dim)
        self.role_count = int(role_count)
        self.matrix_space = matrix_space
        self.cross_cov_eps = float(cross_cov_eps)
        self.matrix_dim = self.token_dim if matrix_space == "embedding" else self.role_count

    def _build_cross_covariance(self, x_mat: torch.Tensor, y_mat: torch.Tensor) -> torch.Tensor:
        if self.matrix_space == "embedding":
            return torch.matmul(x_mat.transpose(-1, -2), y_mat) / float(self.role_count)
        return torch.matmul(x_mat, y_mat.transpose(-1, -2)) / float(self.token_dim)

    def _stable_svd(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m = c.shape[-1]
        device = c.device
        dtype = c.dtype
        tilt = torch.arange(m, device=device, dtype=dtype) + 1.0
        tilt = tilt / float(m) * self.cross_cov_eps
        c_reg = c + torch.diag_embed(tilt.unsqueeze(0).expand(c.shape[0], -1))
        return torch.linalg.svd(c_reg, full_matrices=False)

    def forward(
        self,
        x_mat: torch.Tensor,    # (B, R, D)
        y_mat: torch.Tensor,    # (B, R, D)
        *,
        force_q_star: torch.Tensor | None = None,
    ) -> PolarProcrustesPack:
        if x_mat.shape != y_mat.shape:
            raise ValueError(f"X and Y must have the same shape, got {x_mat.shape} vs {y_mat.shape}")
        cross_cov = self._build_cross_covariance(x_mat, y_mat)
        u, sigma, vh = self._stable_svd(cross_cov)

        if force_q_star is not None:
            q_star = force_q_star
            if q_star.shape != cross_cov.shape:
                raise ValueError(
                    f"force_q_star shape {q_star.shape} must match cross-cov {cross_cov.shape}"
                )
        else:
            q_star = torch.matmul(u, vh)

        # Polar factor H = V Sigma V^T, diagonal sufficient for the head.
        v = vh.transpose(-1, -2)
        polar_h = torch.matmul(v * sigma.unsqueeze(-2), vh)
        polar_diag = torch.diagonal(polar_h, dim1=-2, dim2=-1)

        # Procrustes residual: in embedding-space we apply Q* on the right of X.
        # In role-space we apply Q* on the left of X.
        if self.matrix_space == "embedding":
            aligned = torch.matmul(x_mat, q_star)            # (B, R, D)
        else:
            aligned = torch.matmul(q_star, x_mat)            # (B, R, D)
        diff = aligned - y_mat
        per_role_residual = torch.linalg.vector_norm(diff, dim=-1)             # (B, R)
        procrustes_residual = torch.linalg.vector_norm(per_role_residual, dim=-1)
        identity_residual = torch.linalg.matrix_norm(x_mat - y_mat, ord="fro")
        alignment_improvement = identity_residual - procrustes_residual

        nuclear_norm = sigma.sum(dim=-1)
        spectral_norm = sigma.amax(dim=-1)
        sigma_sq_sum = (sigma * sigma).sum(dim=-1).clamp_min(_DEFAULT_LOG_FLOOR)
        stable_rank = sigma_sq_sum / spectral_norm.clamp_min(_DEFAULT_LOG_FLOOR).pow(2)

        x_norm = torch.linalg.matrix_norm(x_mat, ord="fro")
        y_norm = torch.linalg.matrix_norm(y_mat, ord="fro")
        # svdvals returns descending; pad if R < min(R, D) is the only dimension.
        x_singular_values = torch.linalg.svdvals(x_mat)         # (B, min(R, D))
        y_singular_values = torch.linalg.svdvals(y_mat)
        # Truncate / pad to length R for a stable head feature.
        x_singular_values = self._fit_to_role(x_singular_values, self.role_count)
        y_singular_values = self._fit_to_role(y_singular_values, self.role_count)

        return PolarProcrustesPack(
            cross_cov=cross_cov,
            q_star=q_star,
            singular_values=sigma,
            polar_diag=polar_diag,
            procrustes_residual=procrustes_residual,
            identity_residual=identity_residual,
            alignment_improvement=alignment_improvement,
            per_role_residual=per_role_residual,
            nuclear_norm=nuclear_norm,
            spectral_norm=spectral_norm,
            stable_rank=stable_rank,
            x_norm=x_norm,
            y_norm=y_norm,
            x_singular_values=x_singular_values,
            y_singular_values=y_singular_values,
        )

    @staticmethod
    def _fit_to_role(values: torch.Tensor, role_count: int) -> torch.Tensor:
        if values.shape[-1] == role_count:
            return values
        if values.shape[-1] > role_count:
            return values[..., :role_count]
        pad = role_count - values.shape[-1]
        zeros = values.new_zeros(values.shape[:-1] + (pad,))
        return torch.cat([values, zeros], dim=-1)


class PolarProcrustesHead(nn.Module):
    """LayerNorm + 2-layer MLP over the Procrustes / polar feature vector."""

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


class PolarProcrustesAlignmentNet(nn.Module):
    """Complete bespoke architecture for idea i063.

    Central operator: SVD of the cross-covariance ``C(x) = X(x)^T Y(x)``
    of learned own / opponent role matrices, yielding the orthogonal
    Procrustes alignment ``Q* = U V^T`` and polar strain ``H = V Sigma V^T``.
    Not a CNN, residual stack, attention block, sheaf, transport,
    matrix-pencil, generalized-spectrum, principal-angle, or move-delta
    variant.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        token_dim: int = _DEFAULT_TOKEN_DIM,
        role_count: int = _DEFAULT_ROLE_COUNT,
        head_hidden: int = _DEFAULT_HEAD_HIDDEN,
        cross_cov_eps: float = _DEFAULT_MATRIX_EPS,
        matrix_space: str = "embedding",
        normalize_rows: bool = True,
        include_separate_spectra: bool = True,
        max_tokens: int = _MAX_PIECES,
        ablation: str = "none",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        ablation = (ablation or "none").lower()
        if ablation not in _VALID_ABLATIONS:
            raise ValueError(
                f"Unsupported polar_procrustes ablation {ablation!r}; "
                f"expected one of {sorted(_VALID_ABLATIONS)}"
            )
        matrix_space = (matrix_space or "embedding").lower()
        if matrix_space not in _VALID_MATRIX_SPACES:
            raise ValueError(
                f"matrix_space must be one of {sorted(_VALID_MATRIX_SPACES)}, got {matrix_space!r}"
            )
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.token_dim = int(token_dim)
        self.role_count = int(role_count)
        self.matrix_space = matrix_space
        self.normalize_rows = bool(normalize_rows)
        self.include_separate_spectra = bool(include_separate_spectra)
        self.max_tokens = int(max_tokens)
        self.ablation = ablation

        self.token_extractor = Simple18OwnOpponentTokenExtractor(
            input_channels=input_channels,
            max_tokens=max_tokens,
        )
        self.token_encoder = PieceSquareTokenEncoder(
            input_dim=self.token_extractor.feature_dim,
            token_dim=self.token_dim,
            dropout=dropout,
        )
        self.role_pooler = RoleMatrixPooler(
            token_dim=self.token_dim,
            role_count=self.role_count,
        )
        self.procrustes = PolarProcrustesLayer(
            token_dim=self.token_dim,
            role_count=self.role_count,
            matrix_space=self.matrix_space,
            cross_cov_eps=cross_cov_eps,
        )
        self.matrix_dim = self.procrustes.matrix_dim
        # Singular spectrum length always equals matrix_dim.
        sigma_dim = self.matrix_dim

        # Feature layout fed to the head:
        #   procrustes block: per-role residual (R) + procrustes / identity /
        #   improvement / x_norm / y_norm / nuclear / spectral / stable_rank (8)
        #     -> R + 8 = procrustes_block
        #   spectrum block: singular values (sigma_dim) + polar diag (sigma_dim)
        #     -> 2 * sigma_dim
        #   separate spectra: x_singular (R) + y_singular (R) + own_mass (R) + opp_mass (R)
        #     -> 4 * R when include_separate_spectra else 0
        #   global broadcast: side-to-move + 4 castling + 8 EP +
        #     active-token-norm + own-fraction + opp-fraction = 15
        self.procrustes_block_dim = self.role_count + 8
        self.spectrum_block_dim = 2 * sigma_dim
        self.separate_block_dim = 4 * self.role_count if self.include_separate_spectra else 0
        self.global_feature_dim = 1 + 4 + 8 + 1 + 1 + 1
        self.feature_dim = (
            self.procrustes_block_dim
            + self.spectrum_block_dim
            + self.separate_block_dim
            + self.global_feature_dim
        )
        self.head = PolarProcrustesHead(
            feature_dim=self.feature_dim,
            hidden_dim=int(head_hidden),
            num_classes=self.num_classes,
            dropout=float(dropout),
        )

        # role_pool_mean_only ablation: a deterministic projection from
        # mean / max / std side-wise pooled token embeddings to the role-matrix
        # shape, replacing the learned role queries.
        self.mean_pool_proj_own = nn.Linear(3 * self.token_dim, self.role_count * self.token_dim)
        self.mean_pool_proj_opp = nn.Linear(3 * self.token_dim, self.role_count * self.token_dim)

        # Deterministic batch-shared random orthogonal matrix for the
        # random_orthogonal_alignment ablation. Built via QR of a fixed
        # Gaussian matrix so it is the same across batches but not learned.
        gen = torch.Generator(device="cpu").manual_seed(0xA1107)
        random_mat = torch.randn(self.matrix_dim, self.matrix_dim, generator=gen)
        q_random, _ = torch.linalg.qr(random_mat)
        self.register_buffer("_random_orthogonal", q_random, persistent=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _global_features(
        self,
        tokens: ExtractedTokens,
        own_mass_total: torch.Tensor,
        opp_mass_total: torch.Tensor,
    ) -> torch.Tensor:
        active_norm = tokens.mask.sum(dim=-1, keepdim=True) / float(self.max_tokens)
        own_frac = own_mass_total.unsqueeze(-1) / float(self.max_tokens)
        opp_frac = opp_mass_total.unsqueeze(-1) / float(self.max_tokens)
        return torch.cat(
            [
                tokens.side_to_move_white.unsqueeze(-1),
                tokens.castling,
                tokens.en_passant_file,
                active_norm,
                own_frac,
                opp_frac,
            ],
            dim=-1,
        )

    def _maybe_normalise_rows(self, x_mat: torch.Tensor, y_mat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.normalize_rows:
            return x_mat, y_mat
        return _row_layer_norm(x_mat), _row_layer_norm(y_mat)

    def _side_pool_replacement(
        self,
        token_embed: torch.Tensor,    # (B, N, D)
        side_mask: torch.Tensor,      # (B, N)
        proj: nn.Linear,
    ) -> torch.Tensor:
        mask_f = side_mask.unsqueeze(-1)
        denom = side_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean_pool = (token_embed * mask_f).sum(dim=1) / denom
        very_neg = torch.full_like(token_embed, -1.0e9)
        masked_for_max = torch.where(mask_f.bool(), token_embed, very_neg)
        max_pool = masked_for_max.amax(dim=1)
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        sq_mean = ((token_embed * mask_f) ** 2).sum(dim=1) / denom
        std_pool = (sq_mean - mean_pool ** 2).clamp_min(0.0).sqrt()
        pooled = torch.cat([mean_pool, max_pool, std_pool], dim=-1)
        flat = proj(pooled)
        return flat.view(flat.shape[0], self.role_count, self.token_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.token_extractor(x)
        token_features = tokens.features
        if self.ablation == "material_only_matrices":
            material_only = token_features.clone()
            # Keep only the 12 piece-color one-hot dimensions; zero own_flag,
            # absolute / side-relative coordinates, castling broadcasts and
            # the en-passant flag.
            material_only[:, :, _PIECE_PLANES:] = 0.0
            token_features = material_only
        token_embed = self.token_encoder(token_features, tokens.mask)

        if self.ablation == "role_pool_mean_only":
            x_raw = self._side_pool_replacement(token_embed, tokens.own_mask, self.mean_pool_proj_own)
            y_raw = self._side_pool_replacement(token_embed, tokens.opp_mask, self.mean_pool_proj_opp)
            own_mass = tokens.own_mask.sum(dim=-1, keepdim=True).expand(-1, self.role_count) / float(
                self.role_count
            )
            opp_mass = tokens.opp_mask.sum(dim=-1, keepdim=True).expand(-1, self.role_count) / float(
                self.role_count
            )
            roles = RoleMatrixState(
                own_matrix=x_raw,
                opp_matrix=y_raw,
                own_mass=own_mass,
                opp_mass=opp_mass,
            )
        else:
            roles = self.role_pooler(token_embed, tokens.own_mask, tokens.opp_mask)

        x_mat, y_mat = self._maybe_normalise_rows(roles.own_matrix, roles.opp_matrix)
        if self.ablation == "batch_shuffled_opponent" and y_mat.shape[0] > 1:
            gen = torch.Generator(device="cpu").manual_seed(0xC0FFEE)
            permutation = torch.randperm(y_mat.shape[0], generator=gen).to(y_mat.device)
            if torch.equal(permutation, torch.arange(y_mat.shape[0], device=y_mat.device)):
                permutation = torch.roll(permutation, shifts=1, dims=0)
            y_mat = y_mat.index_select(0, permutation)

        force_q_star: torch.Tensor | None = None
        batch = x_mat.shape[0]
        if self.ablation == "identity_alignment_only":
            eye = torch.eye(self.matrix_dim, device=x_mat.device, dtype=x_mat.dtype)
            force_q_star = eye.unsqueeze(0).expand(batch, -1, -1).contiguous()
        elif self.ablation == "random_orthogonal_alignment":
            force_q_star = self._random_orthogonal.to(device=x_mat.device, dtype=x_mat.dtype)
            force_q_star = force_q_star.unsqueeze(0).expand(batch, -1, -1).contiguous()

        pack = self.procrustes(x_mat, y_mat, force_q_star=force_q_star)

        # Build the head feature vector.
        procrustes_summary = torch.stack(
            [
                pack.procrustes_residual,
                pack.identity_residual,
                pack.alignment_improvement,
                pack.x_norm,
                pack.y_norm,
                pack.nuclear_norm,
                pack.spectral_norm,
                pack.stable_rank,
            ],
            dim=-1,
        )                                                                # (B, 8)
        procrustes_block = torch.cat([pack.per_role_residual, procrustes_summary], dim=-1)
        spectrum_block = torch.cat([pack.singular_values, pack.polar_diag], dim=-1)

        if self.include_separate_spectra:
            separate_block = torch.cat(
                [
                    pack.x_singular_values,
                    pack.y_singular_values,
                    roles.own_mass,
                    roles.opp_mass,
                ],
                dim=-1,
            )
        else:
            separate_block = procrustes_block.new_zeros(batch, 0)

        if self.ablation == "separate_matrix_stats_only":
            procrustes_block = torch.zeros_like(procrustes_block)
            spectrum_block = torch.zeros_like(spectrum_block)
        elif self.ablation == "singular_values_only":
            procrustes_block = torch.zeros_like(procrustes_block)
            # Keep singular values + polar diag but zero polar_diag's contribution
            # by zeroing only the residual / Q*-derived block. Spectrum_block
            # stays intact.
            separate_block = torch.zeros_like(separate_block)

        own_mass_total = roles.own_mass.sum(dim=-1)
        opp_mass_total = roles.opp_mass.sum(dim=-1)
        global_features = self._global_features(tokens, own_mass_total, opp_mass_total)
        head_input = torch.cat(
            [procrustes_block, spectrum_block, separate_block, global_features],
            dim=-1,
        )
        raw_logits = self.head(head_input)

        if self.num_classes == 1:
            logits = raw_logits.view(-1)
            two_class = torch.stack([-0.5 * logits, 0.5 * logits], dim=-1)
        else:
            logits = raw_logits
            two_class = raw_logits if raw_logits.shape[-1] >= 2 else logits

        # Diagnostic proxies.
        mechanism_energy = pack.singular_values.std(dim=-1, unbiased=False)

        diagnostics: dict[str, torch.Tensor] = {
            "logits": logits,
            "two_class_logits": two_class,
            "cross_covariance": pack.cross_cov,
            "orthogonal_alignment": pack.q_star,
            "singular_values": pack.singular_values,
            "polar_strain_diagonal": pack.polar_diag,
            "procrustes_residual": pack.procrustes_residual,
            "identity_residual": pack.identity_residual,
            "alignment_improvement": pack.alignment_improvement,
            "per_role_residual": pack.per_role_residual,
            "nuclear_norm": pack.nuclear_norm,
            "spectral_norm": pack.spectral_norm,
            "stable_rank": pack.stable_rank,
            "x_norm": pack.x_norm,
            "y_norm": pack.y_norm,
            "x_singular_values": pack.x_singular_values,
            "y_singular_values": pack.y_singular_values,
            "own_role_mass": roles.own_mass,
            "opp_role_mass": roles.opp_mass,
            "active_token_count": tokens.mask.sum(dim=-1),
            "mechanism_energy": mechanism_energy,
            "ablation_active": torch.full(
                (logits.shape[0],),
                1.0 if self.ablation != "none" else 0.0,
                device=logits.device,
                dtype=logits.dtype,
            ),
        }
        return diagnostics


def build_polar_procrustes_alignment_bottleneck_from_config(
    config: dict[str, Any],
) -> PolarProcrustesAlignmentNet:
    cfg = dict(config)
    head_hidden = cfg.get("head_hidden", cfg.get("hidden_dim", _DEFAULT_HEAD_HIDDEN))
    return PolarProcrustesAlignmentNet(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        token_dim=int(cfg.get("token_dim", _DEFAULT_TOKEN_DIM)),
        role_count=int(cfg.get("role_count", _DEFAULT_ROLE_COUNT)),
        head_hidden=int(head_hidden),
        cross_cov_eps=float(cfg.get("cross_cov_eps", _DEFAULT_MATRIX_EPS)),
        matrix_space=str(cfg.get("matrix_space", "embedding")),
        normalize_rows=bool(cfg.get("normalize_rows", True)),
        include_separate_spectra=bool(cfg.get("include_separate_spectra", True)),
        max_tokens=int(cfg.get("max_tokens", _MAX_PIECES)),
        ablation=str(cfg.get("ablation", "none")),
        dropout=float(cfg.get("dropout", 0.0)),
    )

"""Grassmannian Principal-Angle Bottleneck model for idea i061.

Implements the markdown thesis (`ideas/registry/i061_grassmannian_principal_angle_bottleneck/`):
puzzle-likeness is tested by measuring the **principal angles** between
learned role-gated occupied-token subspaces. For each role ``r``, the
weighted token covariance

    C_r(x) = sum_i g_{r,i} (h_i - mu_r)(h_i - mu_r)^T + eps * I

defines a top-``K`` orthonormal basis ``Q_r in R^{D x K}`` via
``torch.linalg.eigh``. The point ``span(Q_r)`` is a member of the
Grassmannian ``Gr(K, D)``; for any ordered role pair ``(a, b)`` the
singular values of ``Q_a^T Q_b`` are the cosines of the principal angles
between the two subspaces:

    sigma_{a,b,j} = svdvals(Q_a^T Q_b)_j      in [0, 1]
    theta_{a,b,j} = arccos(clamp(sigma_{a,b,j}, 0, 1))

These angle spectra and the within-role eigenvalue spectra drive a small
MLP head that emits a single puzzle logit. The architecture is
deliberately not a CNN/Transformer/sheaf/move-delta variant: the central
operator is eigh + svd over role-gated covariance matrices, not
convolution, residual stacking, attention, or move enumeration.

Forward pipeline:

    Simple18OccupiedTokenExtractor  ->  (B, N_max, F) tokens, mask
    PieceSquareTokenEncoder         ->  (B, N_max, D) token embeddings Phi
    RoleGatedCovarianceSubspaces    ->  (B, R, D, K) bases, (B, R, K) eigenvalues,
                                        (B, R) gate masses
    PrincipalAngleSpectrum          ->  (B, P, K) cosines, (B, P, K) angles
    GrassmannianAngleHead           ->  (B,) puzzle logit + diagnostics

Section 9 falsifier ablations exposed via ``ablation``:

    * ``"none"``                       -- main model.
    * ``"no_cross_angles"``            -- replace pair principal-angle spectra
      with constants while keeping role eigenvalues / gate masses / token
      stats. This is the markdown's central falsifier.
    * ``"batch_shuffled_angles"``      -- shuffle the per-sample principal-angle
      spectra across the batch so angle structure is no longer tied to the
      sample's roles.
    * ``"eigenvalues_only"``           -- alias for ``no_cross_angles`` that also
      zeroes the angle-derived summary statistics fed to the head.
    * ``"pooled_token_head"``          -- bypass the subspace bottleneck and use
      mean / max token pooling projected to the same head input dimensionality.
    * ``"no_orthonormalization"``      -- replace ``Q_r^T Q_s`` with raw role
      mean dot products (no eigendecomposition / orthonormalization), so
      pair "spectra" become rank-1 cosines without basis-rotation invariance.

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
_DEFAULT_TOKEN_DIM = 48
_DEFAULT_ROLE_COUNT = 8
_DEFAULT_SUBSPACE_DIM = 6
_DEFAULT_HEAD_HIDDEN = 96
_DEFAULT_COVARIANCE_EPS = 1.0e-3
_DEFAULT_ANGLE_EPS = 1.0e-6
_VALID_ABLATIONS = {
    "none",
    "no_cross_angles",
    "batch_shuffled_angles",
    "eigenvalues_only",
    "pooled_token_head",
    "no_orthonormalization",
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

    feature_dim: int = 22

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
class RoleSubspaceState:
    bases: torch.Tensor          # (B, R, D, K) orthonormal columns
    eigenvalues: torch.Tensor    # (B, R, K)
    means: torch.Tensor          # (B, R, D)
    gate_mass: torch.Tensor      # (B, R)
    active_count: torch.Tensor   # (B,)


class RoleGatedCovarianceSubspaces(nn.Module):
    """Build top-K eigenspaces of role-gated weighted token covariances.

    For each role ``r`` we compute

        mu_r       = sum_i g_{r,i} h_i / (sum_i g_{r,i} + eps)
        C_r        = sum_i g_{r,i} (h_i - mu_r)(h_i - mu_r)^T + eps * I_D
        Q_r,lam_r  = top-K eigenvectors / eigenvalues of C_r

    Gates are produced by a small MLP and masked by the occupied-token
    mask so padded tokens never enter the covariance. The eigenvalues
    returned are the top-K (descending) values of ``C_r``.
    """

    def __init__(
        self,
        token_dim: int,
        role_count: int = _DEFAULT_ROLE_COUNT,
        subspace_dim: int = _DEFAULT_SUBSPACE_DIM,
        covariance_eps: float = _DEFAULT_COVARIANCE_EPS,
    ) -> None:
        super().__init__()
        if role_count < 2:
            raise ValueError("role_count must be >= 2 to define principal angles")
        if subspace_dim < 1:
            raise ValueError("subspace_dim must be >= 1")
        if subspace_dim > token_dim:
            raise ValueError("subspace_dim must be <= token_dim")
        if covariance_eps <= 0:
            raise ValueError("covariance_eps must be > 0")
        self.token_dim = int(token_dim)
        self.role_count = int(role_count)
        self.subspace_dim = int(subspace_dim)
        self.covariance_eps = float(covariance_eps)
        gate_hidden = max(self.token_dim, self.role_count * 2)
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.token_dim, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, self.role_count),
        )

    def forward(
        self,
        token_embed: torch.Tensor,    # (B, N, D)
        mask: torch.Tensor,           # (B, N)
    ) -> RoleSubspaceState:
        batch, n_tokens, dim = token_embed.shape
        if dim != self.token_dim:
            raise ValueError(f"Expected token_dim={self.token_dim}, got {dim}")
        gates = torch.sigmoid(self.gate_mlp(token_embed)) * mask.unsqueeze(-1)  # (B, N, R)
        gate_mass = gates.sum(dim=1)                                            # (B, R)
        active_count = mask.sum(dim=-1).clamp_min(1.0)

        # weights w_{b,r,i} = g_{r,i}; transpose to (B, R, N) for einsum
        w = gates.transpose(1, 2)  # (B, R, N)
        w_sum = w.sum(dim=-1).clamp_min(self.covariance_eps)  # (B, R)

        # weighted means: mu_{b,r,d} = sum_i w_{b,r,i} h_{b,i,d} / w_sum
        weighted_h = torch.einsum("brn,bnd->brd", w, token_embed)
        means = weighted_h / w_sum.unsqueeze(-1)                                # (B, R, D)

        # centered tokens per role: (B, R, N, D)
        centered = token_embed.unsqueeze(1) - means.unsqueeze(2)
        # weighted covariance: C_{b,r,d,e} = sum_i w_{b,r,i} centered * centered
        scaled = centered * w.unsqueeze(-1)
        cov = torch.einsum("brnd,brne->brde", scaled, centered)                 # (B, R, D, D)
        cov = 0.5 * (cov + cov.transpose(-1, -2))

        eye_d = torch.eye(self.token_dim, device=cov.device, dtype=cov.dtype)
        # Symmetric isotropic regulariser. When fewer than K tokens contribute
        # to a role, the rank-deficient directions of ``cov`` would all map to
        # eigenvalue ``covariance_eps`` and ``eigh`` backward would return NaN
        # because of repeated eigenvalues. We add a tiny linearly-tilted
        # diagonal in the canonical basis on top of ``eps * I`` to break that
        # degeneracy deterministically while leaving the principal eigenvectors
        # essentially unchanged.
        tilt = torch.arange(self.token_dim, device=cov.device, dtype=cov.dtype)
        tilt = (tilt / max(1.0, float(self.token_dim - 1))) * self.covariance_eps
        diag_pert = torch.diag(tilt).view(1, 1, self.token_dim, self.token_dim)
        cov = cov + self.covariance_eps * eye_d.view(1, 1, self.token_dim, self.token_dim) + diag_pert

        # eigh: returns ascending eigenvalues. Take the last K columns/values.
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)                      # (B, R, D), (B, R, D, D)
        top_eigenvalues = eigenvalues[..., -self.subspace_dim:].flip(-1)        # descending
        top_eigenvectors = eigenvectors[..., -self.subspace_dim:].flip(-1)      # (B, R, D, K), descending

        return RoleSubspaceState(
            bases=top_eigenvectors,
            eigenvalues=top_eigenvalues,
            means=means,
            gate_mass=gate_mass,
            active_count=active_count,
        )


@dataclass(frozen=True)
class PrincipalAnglePack:
    cosines: torch.Tensor              # (B, P, K) sorted descending
    angles: torch.Tensor               # (B, P, K) sorted ascending
    pair_min_angle: torch.Tensor       # (B, P)
    pair_max_angle: torch.Tensor       # (B, P)
    pair_mean_angle: torch.Tensor      # (B, P)
    pair_entropy: torch.Tensor         # (B, P)
    role_a_index: torch.Tensor         # (P,)
    role_b_index: torch.Tensor         # (P,)


class PrincipalAngleSpectrum(nn.Module):
    """Compute principal-angle spectra between every unordered role-pair.

    For roles ``(a, b)`` with ``a < b`` we compute

        M_{a,b} = Q_a^T Q_b            in R^{K x K}
        sigma   = svdvals(M_{a,b})     clamped to [0, 1]
        theta   = arccos(sigma)        in [0, pi/2]

    The spectra are sorted so ``cosines`` are descending (largest cos =
    smallest angle = most aligned) and ``angles`` are ascending.
    """

    def __init__(self, role_count: int, subspace_dim: int) -> None:
        super().__init__()
        self.role_count = int(role_count)
        self.subspace_dim = int(subspace_dim)
        idx_a, idx_b = torch.triu_indices(self.role_count, self.role_count, offset=1).unbind(0)
        self.register_buffer("role_a_index", idx_a, persistent=False)
        self.register_buffer("role_b_index", idx_b, persistent=False)
        self.num_pairs = int(idx_a.shape[0])

    def forward(
        self,
        bases: torch.Tensor,                  # (B, R, D, K)
        *,
        no_orthonormalization: bool = False,
        means: torch.Tensor | None = None,    # (B, R, D), required if no_orthonormalization
    ) -> PrincipalAnglePack:
        batch = bases.shape[0]
        if no_orthonormalization:
            if means is None:
                raise ValueError("means must be provided when no_orthonormalization=True")
            # Replace each role basis with the role mean direction broadcast K times,
            # so the principal-angle SVD reduces to rank-1 cosines and the
            # basis-rotation invariance is destroyed.
            mu_norm = F.normalize(means, dim=-1, eps=1.0e-8)               # (B, R, D)
            broadcast = mu_norm.unsqueeze(-1).expand(-1, -1, -1, self.subspace_dim)
            q = broadcast
        else:
            q = bases

        q_a = q[:, self.role_a_index]   # (B, P, D, K)
        q_b = q[:, self.role_b_index]   # (B, P, D, K)
        cross = torch.matmul(q_a.transpose(-1, -2), q_b)  # (B, P, K, K)
        # svdvals returns singular values in descending order, so cosines are
        # already sorted with the most-aligned direction first and angles are
        # ascending.
        cosines = torch.linalg.svdvals(cross).clamp(0.0, 1.0)
        angles = torch.arccos(cosines.clamp(-1.0 + _DEFAULT_ANGLE_EPS, 1.0 - _DEFAULT_ANGLE_EPS))

        pair_min_angle = angles.amin(dim=-1)
        pair_max_angle = angles.amax(dim=-1)
        pair_mean_angle = angles.mean(dim=-1)
        # softmax entropy over cosines: more peaked alignment -> lower entropy.
        log_probs = F.log_softmax(cosines, dim=-1)
        probs = log_probs.exp()
        pair_entropy = -(probs * log_probs).sum(dim=-1)

        return PrincipalAnglePack(
            cosines=cosines,
            angles=angles,
            pair_min_angle=pair_min_angle,
            pair_max_angle=pair_max_angle,
            pair_mean_angle=pair_mean_angle,
            pair_entropy=pair_entropy,
            role_a_index=self.role_a_index,
            role_b_index=self.role_b_index,
        )


class GrassmannianAngleHead(nn.Module):
    """LayerNorm + 2-layer MLP over angle/eigenvalue/global features."""

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


class GrassmannianPrincipalAngleNet(nn.Module):
    """Complete bespoke architecture for idea i061.

    The central operator is eigendecomposition of role-gated token
    covariance plus SVD of role-pair Gram matrices, yielding principal-
    angle spectra between learned subspaces of the Grassmannian
    ``Gr(K, D)``. The model is permutation-invariant over occupied tokens
    (covariance is a sum) and basis-rotation invariant inside each role
    subspace (singular values of ``Q_a^T Q_b`` are unchanged by
    ``Q_a -> Q_a U`` for any orthogonal ``U``).
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        token_dim: int = _DEFAULT_TOKEN_DIM,
        role_count: int = _DEFAULT_ROLE_COUNT,
        subspace_dim: int = _DEFAULT_SUBSPACE_DIM,
        head_hidden: int = _DEFAULT_HEAD_HIDDEN,
        covariance_eps: float = _DEFAULT_COVARIANCE_EPS,
        max_tokens: int = _MAX_PIECES,
        ablation: str = "none",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        ablation = (ablation or "none").lower()
        if ablation not in _VALID_ABLATIONS:
            raise ValueError(
                f"Unsupported grassmannian_principal_angle ablation {ablation!r}; "
                f"expected one of {sorted(_VALID_ABLATIONS)}"
            )
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.token_dim = int(token_dim)
        self.role_count = int(role_count)
        self.subspace_dim = int(subspace_dim)
        self.max_tokens = int(max_tokens)
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
        self.role_subspaces = RoleGatedCovarianceSubspaces(
            token_dim=self.token_dim,
            role_count=self.role_count,
            subspace_dim=self.subspace_dim,
            covariance_eps=covariance_eps,
        )
        self.angle_module = PrincipalAngleSpectrum(
            role_count=self.role_count,
            subspace_dim=self.subspace_dim,
        )
        self.num_pairs = self.angle_module.num_pairs

        # Feature layout fed to the head:
        #   pair cosines (P, K) + pair angles (P, K)              -> 2 * P * K
        #   pair summary (min, max, mean, entropy) (P, 4)         -> 4 * P
        #   eigenvalues per role (R, K)                           -> R * K
        #   log eigenvalues per role (R, K)                       -> R * K
        #   gate mass per role (R)                                -> R
        #   global broadcast: side_to_move + 4 castling + 8 EP +  -> 14
        #     active_count_norm
        self.pair_feature_dim = 2 * self.num_pairs * self.subspace_dim + 4 * self.num_pairs
        self.eig_feature_dim = 2 * self.role_count * self.subspace_dim + self.role_count
        self.global_feature_dim = 1 + 4 + 8 + 1
        self.feature_dim = self.pair_feature_dim + self.eig_feature_dim + self.global_feature_dim
        self.head = GrassmannianAngleHead(
            feature_dim=self.feature_dim,
            hidden_dim=int(head_hidden),
            num_classes=self.num_classes,
            dropout=float(dropout),
        )

        # Pooled-token-head ablation: a deterministic projection from
        # mean / max / std token-pooled embeddings to the same input
        # dimensionality. We keep the projection trainable so the
        # ablation is not artificially handicapped on capacity.
        self.pooled_token_proj = nn.Linear(3 * self.token_dim, self.feature_dim - self.global_feature_dim)

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

    def _zeroed_pair_features(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch, self.pair_feature_dim, device=device, dtype=dtype)

    def _flatten_pair_features(self, pack: PrincipalAnglePack) -> torch.Tensor:
        cosines = pack.cosines.reshape(pack.cosines.shape[0], -1)
        angles = pack.angles.reshape(pack.angles.shape[0], -1)
        summary = torch.stack(
            [pack.pair_min_angle, pack.pair_max_angle, pack.pair_mean_angle, pack.pair_entropy],
            dim=-1,
        ).reshape(pack.cosines.shape[0], -1)
        return torch.cat([cosines, angles, summary], dim=-1)

    def _flatten_eigen_features(self, eigenvalues: torch.Tensor, gate_mass: torch.Tensor) -> torch.Tensor:
        eig = eigenvalues.reshape(eigenvalues.shape[0], -1)
        log_eig = torch.log(eigenvalues.clamp_min(1.0e-8)).reshape(eigenvalues.shape[0], -1)
        gate = gate_mass.reshape(gate_mass.shape[0], -1)
        return torch.cat([eig, log_eig, gate], dim=-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.token_extractor(x)
        token_embed = self.token_encoder(tokens.features, tokens.mask)
        subspaces = self.role_subspaces(token_embed, tokens.mask)
        no_orthonormalization = self.ablation == "no_orthonormalization"
        pack = self.angle_module(
            subspaces.bases,
            no_orthonormalization=no_orthonormalization,
            means=subspaces.means if no_orthonormalization else None,
        )

        pair_features = self._flatten_pair_features(pack)
        eigen_features = self._flatten_eigen_features(subspaces.eigenvalues, subspaces.gate_mass)
        global_features = self._global_features(tokens)
        batch = pair_features.shape[0]

        if self.ablation in {"no_cross_angles", "eigenvalues_only"}:
            # Replace pair-derived features with zeros, preserving head input
            # dimensionality so the parameter count is matched.
            pair_features = self._zeroed_pair_features(batch, pair_features.device, pair_features.dtype)
        elif self.ablation == "batch_shuffled_angles":
            if batch > 1:
                gen = torch.Generator(device="cpu").manual_seed(0xA1A1)
                permutation = torch.randperm(batch, generator=gen).to(pair_features.device)
                pair_features = pair_features.index_select(0, permutation)

        if self.ablation == "pooled_token_head":
            mask = tokens.mask.unsqueeze(-1)
            denom = tokens.mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            mean_pool = (token_embed * mask).sum(dim=1) / denom
            very_neg = torch.full_like(token_embed, -1.0e9)
            masked_for_max = torch.where(mask.bool(), token_embed, very_neg)
            max_pool = masked_for_max.amax(dim=1)
            max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
            sq_mean = ((token_embed * mask) ** 2).sum(dim=1) / denom
            std_pool = (sq_mean - mean_pool ** 2).clamp_min(0.0).sqrt()
            pooled = torch.cat([mean_pool, max_pool, std_pool], dim=-1)
            non_global = self.pooled_token_proj(pooled)
        else:
            non_global = torch.cat([pair_features, eigen_features], dim=-1)

        features = torch.cat([non_global, global_features], dim=-1)
        raw_logits = self.head(features)

        if self.num_classes == 1:
            logits = raw_logits.view(-1)
            two_class = torch.stack([-0.5 * logits, 0.5 * logits], dim=-1)
        else:
            logits = raw_logits
            two_class = raw_logits if raw_logits.shape[-1] >= 2 else logits

        # Mean pairwise alignment: 1 - mean cos angle ~ how mutually orthogonal
        # the role subspaces are on this sample. Larger -> more separated.
        mean_cos = pack.cosines.mean(dim=(-1, -2))
        mean_angle = pack.angles.mean(dim=(-1, -2))
        # Subspace collapse proxy: standard deviation of pair_mean_angle across pairs;
        # if all pairs are nearly identical, role gates have collapsed.
        pair_mean_std = pack.pair_mean_angle.std(dim=-1, unbiased=False)
        mechanism_energy = pack.pair_entropy.mean(dim=-1)
        eigen_mass = subspaces.eigenvalues.sum(dim=-1).mean(dim=-1)

        diagnostics: dict[str, torch.Tensor] = {
            "logits": logits,
            "two_class_logits": two_class,
            "principal_angle_cosines": pack.cosines,
            "principal_angle_radians": pack.angles,
            "pair_min_angle": pack.pair_min_angle,
            "pair_max_angle": pack.pair_max_angle,
            "pair_mean_angle": pack.pair_mean_angle,
            "pair_entropy": pack.pair_entropy,
            "role_eigenvalues": subspaces.eigenvalues,
            "role_gate_mass": subspaces.gate_mass,
            "active_token_count": subspaces.active_count,
            "mean_pair_cosine": mean_cos,
            "mean_pair_angle": mean_angle,
            "pair_mean_angle_std": pair_mean_std,
            "mechanism_energy": mechanism_energy,
            "eigen_mass": eigen_mass,
            "ablation_active": torch.full(
                (logits.shape[0],),
                1.0 if self.ablation != "none" else 0.0,
                device=logits.device,
                dtype=logits.dtype,
            ),
        }
        return diagnostics


def build_grassmannian_principal_angle_bottleneck_from_config(
    config: dict[str, Any],
) -> GrassmannianPrincipalAngleNet:
    cfg = dict(config)
    head_hidden = cfg.get("head_hidden", cfg.get("hidden_dim", _DEFAULT_HEAD_HIDDEN))
    return GrassmannianPrincipalAngleNet(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        token_dim=int(cfg.get("token_dim", _DEFAULT_TOKEN_DIM)),
        role_count=int(cfg.get("role_count", _DEFAULT_ROLE_COUNT)),
        subspace_dim=int(cfg.get("subspace_dim", _DEFAULT_SUBSPACE_DIM)),
        head_hidden=int(head_hidden),
        covariance_eps=float(cfg.get("covariance_eps", _DEFAULT_COVARIANCE_EPS)),
        max_tokens=int(cfg.get("max_tokens", _MAX_PIECES)),
        ablation=str(cfg.get("ablation", "none")),
        dropout=float(cfg.get("dropout", 0.0)),
    )

"""Determinantal Tactical Volume Bottleneck model for idea i058.

Implements the markdown thesis (`ideas/registry/i058_determinantal_tactical_volume_bottleneck/`):
puzzle-likeness is tested by measuring the **log-volume** of role-gated PSD
kernels over occupied piece tokens. The central feature is

    K_r(x) = D_r Phi A_r A_r^T Phi^T D_r + eps * I_N
    V_r(x) = log det K_r(x)

evaluated jointly over all occupied tokens.

Forward pipeline:

    Simple18OccupiedTokenExtractor  ->  (B, N_max, F) tokens, mask
    PieceSquareTokenEncoder         ->  (B, N_max, d) token embeddings Phi
    RoleGatedPSDVolume              ->  (B, R, stats) role-volume features
    DeterminantalVolumeHead         ->  (B,) puzzle logit + diagnostics

The bottleneck is permutation-invariant over occupied tokens, since
`det(P K_r P^T) = det(K_r)` for any permutation P. The Sylvester/Weinstein
identity reduces the determinant of the (N x N) PSD kernel to a (q x q)
log-determinant, where q is the role rank (default 16):

    log det(Z Z^T + eps I_N) = N * log(eps) + log det(I_q + Z^T Z / eps).

Subtracting the constant ``N * log(eps)`` yields the active log-volume,
which is invariant to padded (mask=0) tokens because they contribute zero
rows to ``Z = D Phi A`` and therefore vanish from ``Z^T Z``.

The architecture is deliberately not a CNN/Transformer/sheaf/move-delta
variant: the central computation is a determinant over a role-gated Gram
matrix of occupied tokens, so the diagonal-trace ablation is the central
falsifier exposed by ``ablation='diagonal_trace_only'``.
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


_EPS = 1.0e-6
_PIECE_PLANES = 12
_MAX_PIECES = 32


@dataclass(frozen=True)
class ExtractedTokens:
    features: torch.Tensor    # (B, N_max, F)
    mask: torch.Tensor        # (B, N_max)
    occupancy: torch.Tensor   # (B, 64) flat occupancy
    side_to_move_white: torch.Tensor  # (B,)
    castling: torch.Tensor    # (B, 4)
    en_passant_file: torch.Tensor  # (B, 8)


class Simple18OccupiedTokenExtractor(nn.Module):
    """Decode simple_18 piece planes into up to ``max_tokens`` occupied tokens.

    Token features (deterministic, board-only):
        - 12 piece-color one-hot (P,N,B,R,Q,K white + black)
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
        device = x.device
        dtype = torch.float32
        x = x.to(dtype)
        batch = x.shape[0]
        piece_planes = x[:, :_PIECE_PLANES].clamp(0.0, 1.0)  # (B, 12, 8, 8)
        side_plane = x[:, 12].clamp(0.0, 1.0)
        side_white = (side_plane.mean(dim=(-1, -2)) > 0.5).to(dtype)  # (B,)
        castling = torch.stack(
            [x[:, 13].mean(dim=(-1, -2)),
             x[:, 14].mean(dim=(-1, -2)),
             x[:, 15].mean(dim=(-1, -2)),
             x[:, 16].mean(dim=(-1, -2))],
            dim=-1,
        ).clamp(0.0, 1.0)  # (B, 4)

        ep_plane = x[:, 17].clamp(0.0, 1.0)  # (B, 8, 8)
        ep_files = ep_plane.amax(dim=-2)  # (B, 8) any rank with EP marker

        # Per-square presence per piece-color
        flat_planes = piece_planes.reshape(batch, _PIECE_PLANES, 64).transpose(1, 2)  # (B, 64, 12)
        occupancy = flat_planes.sum(dim=-1).clamp(0.0, 1.0)  # (B, 64)

        # own/enemy flag: white pieces (planes 0..5) vs black (6..11) gated by side
        is_white_piece = flat_planes[..., :6].sum(dim=-1)
        is_black_piece = flat_planes[..., 6:12].sum(dim=-1)
        side = side_white.view(batch, 1)
        own_flag = side * is_white_piece + (1.0 - side) * is_black_piece  # (B, 64)

        rows = self._rows.view(1, 64).expand(batch, 64)  # (B, 64)
        cols = self._cols.view(1, 64).expand(batch, 64)
        # side-relative coords (mirror row when black to move)
        rel_rows = side * rows + (1.0 - side) * (1.0 - rows)
        rel_cols = cols  # files are not mirrored

        ep_per_square = ep_plane.reshape(batch, 64)

        castling_bcast = castling.unsqueeze(1).expand(batch, 64, 4)  # (B, 64, 4)

        per_square = torch.cat(
            [
                flat_planes,                     # (B, 64, 12)
                own_flag.unsqueeze(-1),          # (B, 64, 1)
                rows.unsqueeze(-1),              # (B, 64, 1)
                cols.unsqueeze(-1),              # (B, 64, 1)
                rel_rows.unsqueeze(-1),          # (B, 64, 1)
                rel_cols.unsqueeze(-1),          # (B, 64, 1)
                castling_bcast,                  # (B, 64, 4)
                ep_per_square.unsqueeze(-1),     # (B, 64, 1)
            ],
            dim=-1,
        )
        assert per_square.shape[-1] == self.feature_dim

        # Select top-K occupied squares by occupancy score (deterministic order).
        # Tokens beyond actual occupancy will have mask=0 and contribute nothing.
        scores = occupancy + 1e-6 * torch.arange(64, device=device, dtype=dtype).view(1, 64) * 0.0
        # ``topk`` is stable enough; tie-break by square index ascending so the
        # token order is deterministic across calls and permutation-invariant
        # at the kernel level (any permutation conjugates the Gram matrix).
        topk = occupancy.topk(self.max_tokens, dim=-1)
        idx = topk.indices  # (B, N_max)
        mask = (occupancy.gather(1, idx) > 0.5).to(dtype)  # (B, N_max)
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

    def __init__(self, input_dim: int, token_dim: int = 48, dropout: float = 0.0) -> None:
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
class RoleVolumeStats:
    """Per-role statistics stacked into ``(B, R, stats_per_role)``."""
    stats: torch.Tensor          # (B, R, S)
    log_volume: torch.Tensor     # (B, R)
    trace: torch.Tensor          # (B, R)
    gate_mass: torch.Tensor      # (B, R)
    top_eig_ratio: torch.Tensor  # (B, R)
    active_count: torch.Tensor   # (B,)


class RoleGatedPSDVolume(nn.Module):
    """Build role-gated Gram matrices and compute volume statistics.

    For each role ``r``:
        - sigmoid gate ``g_{ri}`` per token (mask-aware)
        - low-rank projection ``A_r`` of dimension ``token_dim x q``
        - Gram matrix ``K_r = D_r Phi A_r A_r^T Phi^T D_r + eps I_N`` with
          ``D_r = diag(sqrt(g_{ri} * mask_i))``
        - log-volume ``V_r = log det K_r - N * log(eps)`` via Sylvester:
          ``V_r = log det(I_q + (A_r^T Phi^T D_r^2 Phi A_r) / eps)``
        - trace ``Tr K_r - eps * N = sum_i g_{ri} * mask_i * ||A_r^T phi_i||^2``
        - top-eigenvalue ratio ``lambda_max / Tr_active`` of the active part

    Ablation ``"diagonal_trace_only"`` replaces ``V_r`` with the diagonal
    trace, preserving gates / norms / role marginals while removing all
    off-diagonal determinant interaction; this is the central falsifier
    described in section 9 of the markdown packet.
    """

    def __init__(
        self,
        token_dim: int,
        role_count: int = 8,
        role_rank: int = 16,
        determinant_eps: float = 1.0e-3,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if role_count < 1:
            raise ValueError("role_count must be >= 1")
        if role_rank < 1:
            raise ValueError("role_rank must be >= 1")
        ablation = (ablation or "none").lower()
        if ablation not in {"none", "diagonal_trace_only"}:
            raise ValueError(
                "Unsupported determinantal_volume ablation; expected 'none' or 'diagonal_trace_only'"
            )
        self.token_dim = int(token_dim)
        self.role_count = int(role_count)
        self.role_rank = int(role_rank)
        self.determinant_eps = float(determinant_eps)
        self.ablation = ablation

        # gate_mlp: (B, N, d) -> (B, N, R)
        gate_hidden = max(self.token_dim, self.role_count * 2)
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.token_dim, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, self.role_count),
        )
        # role projectors A_r as a single tensor of shape (R, d, q)
        self.role_projectors = nn.Parameter(
            torch.empty(self.role_count, self.token_dim, self.role_rank)
        )
        nn.init.xavier_uniform_(self.role_projectors)

        # Stats per role: [log_volume, normalized_log_volume, trace, top_eig_ratio, gate_mass, active_count]
        self.stats_per_role = 6

    @property
    def feature_dim(self) -> int:
        return self.role_count * self.stats_per_role

    def forward(
        self,
        token_embed: torch.Tensor,    # (B, N, d)
        mask: torch.Tensor,           # (B, N)
    ) -> RoleVolumeStats:
        batch, n_tokens, _ = token_embed.shape
        # gates per token per role: (B, N, R)
        gates = torch.sigmoid(self.gate_mlp(token_embed)) * mask.unsqueeze(-1)
        # active counts (per batch)
        active_count = mask.sum(dim=-1).clamp_min(1.0)
        # gate_mass per role: sum_i g_{ri} (B, R)
        gate_mass = gates.sum(dim=1)

        # Project tokens for all roles: (B, R, N, q) = (B, N, d) @ (R, d, q)
        # einsum: bnd, rdq -> brnq
        proj = torch.einsum("bnd,rdq->brnq", token_embed, self.role_projectors)
        # D_r entries: sqrt(gate * mask)
        gate_sqrt = (gates * mask.unsqueeze(-1)).clamp_min(0.0).sqrt()  # (B, N, R)
        gate_sqrt = gate_sqrt.transpose(1, 2)  # (B, R, N)
        z = proj * gate_sqrt.unsqueeze(-1)  # (B, R, N, q)

        # token-wise squared norm ||z_{r,i}||^2: (B, R, N)
        z_sq_norm = (z * z).sum(dim=-1)
        trace_active = z_sq_norm.sum(dim=-1)  # (B, R)

        # C_r = Z_r^T Z_r: (B, R, q, q)
        c_mat = torch.einsum("brnp,brnq->brpq", z, z)
        eps = self.determinant_eps
        q = self.role_rank
        eye_q = torch.eye(q, device=z.device, dtype=z.dtype)
        # log det(I_q + C_r / eps); use Cholesky via slogdet for stability
        scaled = eye_q.view(1, 1, q, q) + c_mat / max(eps, 1.0e-12)
        sign, logabsdet = torch.linalg.slogdet(scaled)
        # If for any reason sign != 1 (numerical), fall back to clamping
        log_volume = torch.where(sign > 0, logabsdet, torch.zeros_like(logabsdet))

        # Top eigenvalue ratio: largest eigenvalue of C_r over its trace.
        # Use eigvalsh for symmetric PSD matrix.
        # Symmetrize for numerical safety.
        c_sym = 0.5 * (c_mat + c_mat.transpose(-1, -2))
        eig = torch.linalg.eigvalsh(c_sym)  # (B, R, q)
        top_eig = eig.amax(dim=-1)
        trace_c = c_sym.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(eps)
        top_eig_ratio = top_eig / trace_c

        normalized_log_volume = log_volume / active_count.view(batch, 1)

        if self.ablation == "diagonal_trace_only":
            # Replace V_r with the gated trace (sum of diagonal entries of K_r minus eps*N)
            # so that no off-diagonal determinant interaction reaches the head.
            log_volume_signal = trace_active
            normalized_signal = trace_active / active_count.view(batch, 1)
        else:
            log_volume_signal = log_volume
            normalized_signal = normalized_log_volume

        active_count_feat = active_count.view(batch, 1).expand(batch, self.role_count) / float(n_tokens)
        stats = torch.stack(
            [
                log_volume_signal,
                normalized_signal,
                trace_active,
                top_eig_ratio,
                gate_mass,
                active_count_feat,
            ],
            dim=-1,
        )  # (B, R, 6)

        return RoleVolumeStats(
            stats=stats,
            log_volume=log_volume,
            trace=trace_active,
            gate_mass=gate_mass,
            top_eig_ratio=top_eig_ratio,
            active_count=active_count,
        )


class DeterminantalVolumeHead(nn.Module):
    """MLP head that consumes role-volume stats and global broadcast features."""

    def __init__(
        self,
        role_feature_dim: int,
        global_feature_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        in_dim = role_feature_dim + global_feature_dim
        layers: list[nn.Module] = [
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, max(1, int(num_classes))))
        self.mlp = nn.Sequential(*layers)
        self.num_classes = int(num_classes)

    def forward(self, role_features: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([role_features, global_features], dim=-1))


class DeterminantalTacticalVolumeNet(nn.Module):
    """Complete bespoke architecture for idea i058."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        token_dim: int = 48,
        role_count: int = 8,
        role_rank: int = 16,
        head_hidden: int = 128,
        determinant_eps: float = 1.0e-3,
        ablation: str = "none",
        max_tokens: int = _MAX_PIECES,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.max_tokens = int(max_tokens)
        self.token_extractor = Simple18OccupiedTokenExtractor(
            input_channels=input_channels,
            max_tokens=max_tokens,
        )
        self.token_encoder = PieceSquareTokenEncoder(
            input_dim=self.token_extractor.feature_dim,
            token_dim=token_dim,
            dropout=dropout,
        )
        self.volume = RoleGatedPSDVolume(
            token_dim=token_dim,
            role_count=role_count,
            role_rank=role_rank,
            determinant_eps=determinant_eps,
            ablation=ablation,
        )
        # Globals: side_to_move + 4 castling + 8 EP file + token_count_norm = 14
        self.global_feature_dim = 1 + 4 + 8 + 1
        self.head = DeterminantalVolumeHead(
            role_feature_dim=self.volume.feature_dim,
            global_feature_dim=self.global_feature_dim,
            hidden_dim=head_hidden,
            num_classes=self.num_classes,
            dropout=dropout,
        )

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

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.token_extractor(x)
        token_embed = self.token_encoder(tokens.features, tokens.mask)
        volume = self.volume(token_embed, tokens.mask)

        role_features = volume.stats.reshape(volume.stats.shape[0], -1)
        global_features = self._global_features(tokens)
        raw_logits = self.head(role_features, global_features)

        if self.num_classes == 1:
            logits = raw_logits.view(-1)
            two_class = torch.stack([-0.5 * logits, 0.5 * logits], dim=-1)
        else:
            logits = raw_logits
            two_class = raw_logits if raw_logits.shape[-1] >= 2 else None

        diagnostics = {
            "logits": logits,
            "two_class_logits": two_class if two_class is not None else logits,
            "log_volume": volume.log_volume,
            "log_volume_mean": volume.log_volume.mean(dim=-1),
            "log_volume_max": volume.log_volume.amax(dim=-1),
            "log_volume_min": volume.log_volume.amin(dim=-1),
            "trace": volume.trace,
            "trace_mean": volume.trace.mean(dim=-1),
            "gate_mass": volume.gate_mass,
            "gate_mass_mean": volume.gate_mass.mean(dim=-1),
            "top_eig_ratio": volume.top_eig_ratio,
            "top_eig_ratio_mean": volume.top_eig_ratio.mean(dim=-1),
            "active_count": volume.active_count,
            "mechanism_energy": volume.log_volume.abs().mean(dim=-1),
            "ablation_active": torch.full(
                (logits.shape[0],),
                1.0 if self.volume.ablation != "none" else 0.0,
                device=logits.device,
                dtype=logits.dtype,
            ),
        }
        return diagnostics


def build_determinantal_tactical_volume_bottleneck_from_config(
    config: dict[str, Any],
) -> DeterminantalTacticalVolumeNet:
    cfg = dict(config)
    return DeterminantalTacticalVolumeNet(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        token_dim=int(cfg.get("token_dim", 48)),
        role_count=int(cfg.get("role_count", 8)),
        role_rank=int(cfg.get("role_rank", 16)),
        head_hidden=int(cfg.get("head_hidden", cfg.get("hidden_dim", 128))),
        determinant_eps=float(cfg.get("determinant_eps", 1.0e-3)),
        ablation=str(cfg.get("ablation", "none")),
        max_tokens=int(cfg.get("max_tokens", _MAX_PIECES)),
        dropout=float(cfg.get("dropout", 0.0)),
    )

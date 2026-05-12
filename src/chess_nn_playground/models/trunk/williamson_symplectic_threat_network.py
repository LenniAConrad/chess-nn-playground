"""Williamson Symplectic-Eigenvalue Threat Network for idea i229.

Builds a per-board SPD operator ``M`` of size ``2n x 2n`` over a chess
position-momentum phase space and reads off its symplectic eigenvalues
``{d_i}`` from Williamson's normal form ``M = S^T D S``, ``S in Sp(2n, R)``,
``D = diag(d_1, ..., d_n, d_1, ..., d_n)``. The puzzle logit is produced from
the symplectic spectrum (top-k, gaps, entropy, Heisenberg-slack) plus the
ordinary spectrum of ``M`` for contrast and a pooled board summary.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


class _Trunk(nn.Module):
    def __init__(self, input_channels: int, channels: int, depth: int, dropout: float, use_batchnorm: bool) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = input_channels
        for _ in range(max(1, depth)):
            layers.append(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            in_channels = channels
        self.trunk = nn.Sequential(*layers)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


class WilliamsonSymplecticThreatNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        phase_n: int = 32,
        num_primitives: int = 16,
        primitive_rank: int = 4,
        lambda_floor: float = 1.0e-3,
        spd_floor: float = 1.0e-5,
        topk_d: int = 12,
        topk_eig: int = 8,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("WilliamsonSymplecticThreatNetwork supports the puzzle_binary one-logit contract")
        if phase_n < 2:
            raise ValueError("phase_n must be >= 2")
        if num_primitives < 1 or primitive_rank < 1:
            raise ValueError("num_primitives and primitive_rank must be >= 1")

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.phase_n = int(phase_n)
        self.two_n = 2 * self.phase_n
        self.num_primitives = int(num_primitives)
        self.primitive_rank = int(primitive_rank)
        self.lambda_floor = float(lambda_floor)
        self.spd_floor = float(spd_floor)
        self.topk_d = max(1, min(int(topk_d), self.phase_n))
        self.topk_eig = max(1, min(int(topk_eig), self.two_n))

        self.trunk = _Trunk(input_channels, channels, depth, dropout, use_batchnorm)
        pooled_dim = channels * 2
        self.weight_head = nn.Linear(pooled_dim, self.num_primitives)

        primitive_seed = torch.randn(self.num_primitives, self.two_n, self.primitive_rank) * 0.1
        # Bias the first primitives toward chess-natural position-momentum couplings:
        # block-diagonal (position-position, momentum-momentum) and
        # off-diagonal symmetric (position-momentum) PSD primitives.
        n = self.phase_n
        if self.num_primitives >= 1:
            base = torch.zeros(self.two_n, self.primitive_rank)
            base[:n, : min(self.primitive_rank, n)] = torch.eye(n, min(self.primitive_rank, n))
            primitive_seed[0] = primitive_seed[0] + 0.5 * base
        if self.num_primitives >= 2:
            base = torch.zeros(self.two_n, self.primitive_rank)
            base[n:, : min(self.primitive_rank, n)] = torch.eye(n, min(self.primitive_rank, n))
            primitive_seed[1] = primitive_seed[1] + 0.5 * base
        if self.num_primitives >= 3:
            base = torch.zeros(self.two_n, self.primitive_rank)
            r = min(self.primitive_rank, n)
            base[:n, :r] = torch.eye(n, r) * 0.7071
            base[n:, :r] = torch.eye(n, r) * 0.7071
            primitive_seed[2] = primitive_seed[2] + 0.5 * base
        self.primitive_factors = nn.Parameter(primitive_seed)

        # Fixed symplectic form J = [[0, I_n], [-I_n, 0]] on R^{2n}.
        eye_n = torch.eye(self.phase_n)
        zeros_n = torch.zeros(self.phase_n, self.phase_n)
        j = torch.cat(
            [
                torch.cat([zeros_n, eye_n], dim=1),
                torch.cat([-eye_n, zeros_n], dim=1),
            ],
            dim=0,
        )
        self.register_buffer("J", j, persistent=False)
        self.register_buffer("eye_2n", torch.eye(self.two_n), persistent=False)

        feat_dim = (
            self.topk_d  # top-k symplectic eigenvalues
            + self.topk_eig  # top-k ordinary eigenvalues of M
            + (self.topk_d - 1)  # adjacent symplectic gaps within the top-k
            + self.topk_d  # per-mode Heisenberg slack (d_i - 1/2) on top-k
            + 5  # symplectic_entropy, log_det_M, d_min, d_max, heisenberg_violation
        )
        self.head = nn.Sequential(
            nn.LayerNorm(pooled_dim + feat_dim),
            nn.Linear(pooled_dim + feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def _build_M(self, pooled: torch.Tensor) -> torch.Tensor:
        b = pooled.shape[0]
        weights = nn.functional.softplus(self.weight_head(pooled))  # (B, P)
        sqrt_w = weights.clamp_min(self.spd_floor).sqrt().unsqueeze(-1).unsqueeze(-1)  # (B, P, 1, 1)
        scaled = sqrt_w * self.primitive_factors.unsqueeze(0)  # (B, P, 2n, r)
        f_cat = scaled.transpose(1, 2).reshape(b, self.two_n, self.num_primitives * self.primitive_rank)
        m = f_cat @ f_cat.transpose(-1, -2)
        m = m + self.lambda_floor * self.eye_2n
        return 0.5 * (m + m.transpose(-1, -2))

    def _symplectic_spectrum(self, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eigvals_m, eigvecs_m = torch.linalg.eigh(m)
        eigvals_m = eigvals_m.clamp_min(self.spd_floor)
        sqrt_diag = torch.diag_embed(eigvals_m.sqrt())
        m_half = eigvecs_m @ sqrt_diag @ eigvecs_m.transpose(-1, -2)
        k = m_half @ self.J @ m_half  # skew: K^T = -K
        ktk = k.transpose(-1, -2) @ k
        ktk = 0.5 * (ktk + ktk.transpose(-1, -2))
        ktk_eigs = torch.linalg.eigvalsh(ktk).clamp_min(0.0)  # eigenvalues d_i^2 with multiplicity 2
        ktk_sorted, _ = ktk_eigs.sort(dim=-1, descending=True)
        # Average paired entries to mitigate numerical multiplicity-2 splitting.
        d_squared = 0.5 * (ktk_sorted[..., 0::2] + ktk_sorted[..., 1::2])  # (B, n)
        d = d_squared.clamp_min(self.spd_floor).sqrt()
        return d, eigvals_m, ktk_sorted

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feat = self.trunk(x)
        pooled = torch.cat([feat.mean(dim=(2, 3)), feat.amax(dim=(2, 3))], dim=1)

        m = self._build_M(pooled)
        d, m_eigvals, _ = self._symplectic_spectrum(m)

        d_sorted, _ = d.sort(dim=-1, descending=True)
        m_eigvals_sorted, _ = m_eigvals.sort(dim=-1, descending=True)

        d_topk = d_sorted[:, : self.topk_d]
        eig_topk = m_eigvals_sorted[:, : self.topk_eig]

        log_d = d.clamp_min(self.spd_floor).log()
        symplectic_entropy = -log_d.sum(dim=-1)
        log_det_m = 2.0 * log_d.sum(dim=-1)

        heisenberg_slack = d_topk - 0.5
        heisenberg_violation = (0.5 - d).clamp_min(0.0).sum(dim=-1)

        gaps = d_topk[:, :-1] - d_topk[:, 1:]
        d_min = d.amin(dim=-1)
        d_max = d.amax(dim=-1)

        feat_vec = torch.cat(
            [
                d_topk,
                eig_topk,
                gaps,
                heisenberg_slack,
                symplectic_entropy.unsqueeze(-1),
                log_det_m.unsqueeze(-1),
                d_min.unsqueeze(-1),
                d_max.unsqueeze(-1),
                heisenberg_violation.unsqueeze(-1),
            ],
            dim=-1,
        )
        logits = self.head(torch.cat([pooled, feat_vec], dim=-1)).view(-1)
        return {
            "logits": logits,
            "symplectic_spectrum": d_sorted,
            "symplectic_top_d": d_topk,
            "symplectic_entropy": symplectic_entropy,
            "symplectic_log_det_M": log_det_m,
            "symplectic_d_min": d_min,
            "symplectic_d_max": d_max,
            "symplectic_spectral_gaps": gaps,
            "ordinary_eigvals_topk": eig_topk,
            "heisenberg_slack": heisenberg_slack,
            "heisenberg_violation": heisenberg_violation,
        }


def build_williamson_symplectic_threat_network_from_config(
    config: dict[str, Any],
) -> WilliamsonSymplecticThreatNetwork:
    cfg = dict(config)
    return WilliamsonSymplecticThreatNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        phase_n=int(cfg.get("phase_n", 32)),
        num_primitives=int(cfg.get("num_primitives", 16)),
        primitive_rank=int(cfg.get("primitive_rank", 4)),
        lambda_floor=float(cfg.get("lambda_floor", cfg.get("M_floor_lambda", 1.0e-3))),
        spd_floor=float(cfg.get("spd_floor", 1.0e-5)),
        topk_d=int(cfg.get("topk_d", 12)),
        topk_eig=int(cfg.get("topk_eig", 8)),
        num_classes=int(cfg.get("num_classes", 1)),
    )

"""Toda Isospectral Flow Network (idea i235).

Constructs a learned symmetric tridiagonal chess operator L_0 from board features
and integrates the Toda Lax flow ``\\dot L = [L, B(L)]`` with ``B(L) = L_- - L_+``
(strictly lower minus strictly upper triangular parts). Because B is skew-
symmetric, the flow is isospectral (Symes' theorem): the eigenvalues of L are
conserved while the diagonal sorts in descending order and the off-diagonal
entries decay at rates set by adjacent eigenvalue gaps. The classifier reads
off:

  * the sorting score of the diagonal at flow time T,
  * Manakov drift ``Tr(L_T^k) - Tr(L_0^k)`` (must be ~0 for an isospectral
    flow; the drift quantifies numerical fidelity / instability),
  * off-diagonal log-decay summaries and the slowest-decaying off-diagonal
    residue,
  * a smallest-spectral-gap proxy ``-max_i log(b_i(T)/b_i(0)) / T`` (the
    slowest decay rate; clamped to zero).

The flow is integrated by an explicit Euler step on the matrix Lax form so the
gradient signal flows through the iterates and the head can correlate decay
diagnostics with puzzle-likeness.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardConvStem, BoardTensorSpec, require_board_tensor


def _toda_lax_step(L: torch.Tensor, dt: float) -> torch.Tensor:
    """One explicit Euler step of the Toda Lax flow ``dot L = [L, B]``.

    For a symmetric L, B = L_- - L_+ is the skew-symmetric projection that
    drives the standard (non-periodic) Toda lattice.
    """
    lower = torch.tril(L, diagonal=-1)
    upper = torch.triu(L, diagonal=1)
    B = lower - upper
    bracket = L @ B - B @ L
    return L + dt * bracket


def _symmetrize(L: torch.Tensor) -> torch.Tensor:
    """Re-symmetrize after an Euler step to absorb roundoff drift."""
    return 0.5 * (L + L.transpose(-1, -2))


def _build_tridiagonal(diag: torch.Tensor, off: torch.Tensor) -> torch.Tensor:
    """Materialise a (B, n, n) symmetric tridiagonal matrix from diagonals."""
    n = diag.shape[-1]
    L = torch.diag_embed(diag)
    idx = torch.arange(n - 1, device=diag.device)
    L[:, idx, idx + 1] = off
    L[:, idx + 1, idx] = off
    return L


def _sortedness_score(values: torch.Tensor) -> torch.Tensor:
    """Return a normalised score in [-1, 1] capturing how monotone-decreasing the
    diagonal is. +1 for a perfectly sorted diagonal, -1 for fully reversed."""
    sorted_desc, _ = torch.sort(values, dim=-1, descending=True)
    sorted_asc, _ = torch.sort(values, dim=-1, descending=False)
    norm = (sorted_desc - sorted_asc).square().sum(dim=-1).clamp_min(1e-8)
    err_desc = (values - sorted_desc).square().sum(dim=-1)
    return 1.0 - 2.0 * err_desc / norm


class TodaIsospectralFlowNetwork(nn.Module):
    """Bespoke Toda Lax flow operator network for puzzle_binary classification."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        depth: int = 2,
        operator_dim: int = 12,
        flow_steps: int = 8,
        flow_dt: float = 0.08,
        manakov_order: int = 4,
        hidden_dim: int = 96,
        head_hidden: int = 96,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        off_diagonal_floor: float = 1.0e-3,
    ) -> None:
        super().__init__()
        if operator_dim < 3:
            raise ValueError("operator_dim must be >= 3 to form a tridiagonal Toda lattice")
        if flow_steps < 1:
            raise ValueError("flow_steps must be >= 1")
        if manakov_order < 2:
            raise ValueError("manakov_order must be >= 2")

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.operator_dim = int(operator_dim)
        self.flow_steps = int(flow_steps)
        self.flow_dt = float(flow_dt)
        self.manakov_order = int(manakov_order)
        self.off_diagonal_floor = float(off_diagonal_floor)

        self.stem = BoardConvStem(
            input_channels=input_channels,
            channels=channels,
            depth=depth,
            use_batchnorm=use_batchnorm,
        )

        n = self.operator_dim
        self.param_dim = 2 * n - 1  # n diagonals + (n-1) off-diagonals
        self.operator_proj = nn.Linear(channels * 2, self.param_dim)

        diag_feature_dim = (
            2 * n  # initial + final diagonals
            + 2 * (n - 1)  # initial + final off-diagonals
            + 5  # sorting score, max/mean decay, slowest off-diag, gap estimate
            + (self.manakov_order - 1)  # per-order Manakov drift
            + 1  # |max| Manakov drift
        )
        feature_dim = channels * 2 + diag_feature_dim
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(head_hidden, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_classes),
        )

    def _initial_operator(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        params = self.operator_proj(pooled)
        n = self.operator_dim
        diag = params[:, :n]
        off = F.softplus(params[:, n:]) + self.off_diagonal_floor
        L0 = _build_tridiagonal(diag, off)
        return L0, diag, off

    def _manakov_traces(self, L: torch.Tensor) -> torch.Tensor:
        """Compute trace(L^k) for k=2..manakov_order, stacked along the last axis."""
        traces: list[torch.Tensor] = []
        cur = L
        for _ in range(2, self.manakov_order + 1):
            cur = cur @ L
            traces.append(torch.diagonal(cur, dim1=-2, dim2=-1).sum(-1))
        return torch.stack(traces, dim=-1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feat = self.stem(x)
        pooled = torch.cat([feat.mean(dim=(-2, -1)), feat.amax(dim=(-2, -1))], dim=-1)
        L0, diag_initial, off_initial = self._initial_operator(pooled)
        manakov_initial = self._manakov_traces(L0)

        L = L0
        for _ in range(self.flow_steps):
            L = _symmetrize(_toda_lax_step(L, self.flow_dt))
        L_T = L

        n = self.operator_dim
        idx = torch.arange(n - 1, device=L.device)
        diag_final = torch.diagonal(L_T, dim1=-2, dim2=-1)
        off_final = L_T[:, idx, idx + 1]

        manakov_final = self._manakov_traces(L_T)
        manakov_drift = manakov_final - manakov_initial
        manakov_drift_max = manakov_drift.abs().amax(dim=-1)

        sortedness = _sortedness_score(diag_final)

        off_initial_abs = off_initial.abs().clamp_min(1.0e-6)
        off_final_abs = off_final.abs().clamp_min(1.0e-6)
        # ``decay`` is positive when the off-diagonal shrinks under the flow
        # and negative when it grows.
        decay = -torch.log(off_final_abs / off_initial_abs)
        max_decay = decay.amax(dim=-1)  # largest decay (fastest-decaying band)
        mean_decay = decay.mean(dim=-1)
        slowest_decay = decay.amin(dim=-1)  # smallest decay; negative if growing
        slowest_off = off_final_abs.amax(dim=-1)  # largest residue magnitude

        total_t = max(self.flow_steps * self.flow_dt, 1.0e-3)
        # Smallest-spectral-gap proxy: under Toda flow b_i decays at rate
        # |lambda_i - lambda_{i+1}|, so the slowest-decaying off-diagonal
        # tracks the smallest adjacent eigenvalue gap. Clamp at zero so
        # short-time numerics that grow an off-diagonal do not produce a
        # negative "gap".
        gap_estimate = (slowest_decay / total_t).clamp_min(0.0)

        feature = torch.cat(
            [
                pooled,
                diag_initial,
                diag_final,
                off_initial,
                off_final,
                sortedness.unsqueeze(-1),
                max_decay.unsqueeze(-1),
                mean_decay.unsqueeze(-1),
                slowest_off.unsqueeze(-1),
                gap_estimate.unsqueeze(-1),
                manakov_drift,
                manakov_drift_max.unsqueeze(-1),
            ],
            dim=-1,
        )
        logits = self.head(feature)
        if self.num_classes == 1:
            logits = logits.squeeze(-1)

        return {
            "logits": logits,
            "diag_initial": diag_initial,
            "diag_final": diag_final,
            "off_initial": off_initial,
            "off_final": off_final,
            "sorting_score": sortedness,
            "max_off_diag_decay": max_decay,
            "mean_off_diag_decay": mean_decay,
            "slowest_off_diag": slowest_off,
            "spectral_gap_estimate": gap_estimate,
            "manakov_drift": manakov_drift,
            "manakov_drift_max": manakov_drift_max,
            "operator_frobenius_norm": L_T.flatten(1).norm(dim=1),
        }


def build_toda_isospectral_flow_network_from_config(config: dict[str, Any]) -> TodaIsospectralFlowNetwork:
    return TodaIsospectralFlowNetwork(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        depth=int(config.get("depth", 2)),
        operator_dim=int(config.get("operator_dim", 12)),
        flow_steps=int(config.get("flow_steps", 8)),
        flow_dt=float(config.get("flow_dt", 0.08)),
        manakov_order=int(config.get("manakov_order", 4)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        head_hidden=int(config.get("head_hidden", config.get("hidden_dim", 96))),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        off_diagonal_floor=float(config.get("off_diagonal_floor", 1.0e-3)),
    )

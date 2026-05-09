"""Riccati Optimal-Defense Network for idea i231.

Treats each board as an LQR control problem with attacker dynamics ``A``,
defender input ``B``, target weighting ``Q`` and effort weighting ``R``.
Solves the continuous algebraic Riccati equation

    A^T P + P A - P B R^{-1} B^T P + Q = 0

via the Hamiltonian H = [[A, -B R^{-1} B^T], [-Q, -A^T]]: the stabilizing
solution P is read off from the stable invariant subspace of H as
P = V_2 V_1^{-1}. Puzzle-likeness is then driven by the optimal-defense
cost J* = trace(P), the closed-loop spectral margin of A_cl = A - B K with
K = R^{-1} B^T P, the spectrum of P, and a CARE residual diagnostic.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


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


def _solve_care_via_hamiltonian(
    a: torch.Tensor,
    b_rinv_bt: torch.Tensor,
    q: torch.Tensor,
    *,
    floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve A^T P + P A - P (B R^{-1} B^T) P + Q = 0 batched via Hamiltonian.

    Returns (P, hamiltonian_eigvals, stable_count) where ``P`` is the
    stabilizing solution recovered from the stable invariant subspace of
    H = [[A, -B R^{-1} B^T], [-Q, -A^T]].
    """
    bsz, r, _ = a.shape
    two_r = 2 * r
    a_t = a.transpose(-1, -2)
    top = torch.cat([a, -b_rinv_bt], dim=-1)
    bot = torch.cat([-q, -a_t], dim=-1)
    h = torch.cat([top, bot], dim=-2)

    eigvals, eigvecs = torch.linalg.eig(h)
    real_parts = eigvals.real
    # Stable eigenvalues have Re < 0; rank by ascending Re so the most stable
    # come first. Symplectic structure guarantees exactly r stable / r unstable
    # eigenvalues (modulo numerical noise on the imaginary axis).
    order = torch.argsort(real_parts, dim=-1)
    stable_idx = order[..., :r]
    stable_count = (real_parts < -floor).sum(dim=-1).to(real_parts.dtype)

    gather_idx = stable_idx.unsqueeze(-2).expand(bsz, two_r, r)
    v_stable = torch.gather(eigvecs, dim=-1, index=gather_idx)
    v1 = v_stable[:, :r, :]
    v2 = v_stable[:, r:, :]

    eye_r = torch.eye(r, dtype=v1.dtype, device=v1.device).expand(bsz, r, r)
    reg = eye_r * floor
    # Solve V1^T X^T = V2^T  =>  X = P = V2 V1^{-1}
    p_complex = torch.linalg.solve(
        v1.transpose(-1, -2) + reg, v2.transpose(-1, -2)
    ).transpose(-1, -2)
    p = p_complex.real
    p = 0.5 * (p + p.transpose(-1, -2))
    return p, eigvals, stable_count


class RiccatiOptimalDefenseNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        riccati_rank_r: int = 12,
        defender_dim_m: int = 4,
        num_a_primitives: int = 8,
        num_b_primitives: int = 8,
        num_q_primitives: int = 4,
        num_r_primitives: int = 4,
        alpha_init: float = 1.0,
        hurwitz_safety: float = 0.1,
        q_floor_beta: float = 1.0e-3,
        r_floor_gamma: float = 1.0e-3,
        care_floor: float = 1.0e-4,
        topk_p: int = 6,
        topk_acl: int = 4,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("RiccatiOptimalDefenseNetwork supports the puzzle_binary one-logit contract")
        if riccati_rank_r < 2:
            raise ValueError("riccati_rank_r must be >= 2")
        if defender_dim_m < 1:
            raise ValueError("defender_dim_m must be >= 1")

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.r = int(riccati_rank_r)
        self.m = int(defender_dim_m)
        self.num_a_primitives = int(num_a_primitives)
        self.num_b_primitives = int(num_b_primitives)
        self.num_q_primitives = int(num_q_primitives)
        self.num_r_primitives = int(num_r_primitives)
        self.hurwitz_safety = float(hurwitz_safety)
        self.q_floor_beta = float(q_floor_beta)
        self.r_floor_gamma = float(r_floor_gamma)
        self.care_floor = float(care_floor)
        self.topk_p = max(1, min(int(topk_p), self.r))
        self.topk_acl = max(1, min(int(topk_acl), self.r))

        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        self.trunk = _Trunk(input_channels, channels, depth, dropout, use_batchnorm)
        pooled_dim = channels * 2

        self.gate_a = nn.Linear(pooled_dim, self.num_a_primitives)
        self.gate_b = nn.Linear(pooled_dim, self.num_b_primitives)
        self.gate_q = nn.Linear(pooled_dim, self.num_q_primitives)
        self.gate_r = nn.Linear(pooled_dim, self.num_r_primitives)

        scale = 1.0 / math.sqrt(self.r)
        self.primitives_a = nn.Parameter(torch.randn(self.num_a_primitives, self.r, self.r) * scale)
        self.primitives_b = nn.Parameter(torch.randn(self.num_b_primitives, self.r, self.m) * scale)
        # Q and R primitives are stored as factors so we can build PSD
        # contributions as N N^T + N^T N (kept symmetric, non-negative).
        self.primitives_q = nn.Parameter(torch.randn(self.num_q_primitives, self.r, self.r) * scale)
        self.primitives_r = nn.Parameter(torch.randn(self.num_r_primitives, self.m, self.m) * scale)

        eye_r = torch.eye(self.r)
        eye_m = torch.eye(self.m)
        self.register_buffer("eye_r", eye_r, persistent=False)
        self.register_buffer("eye_m", eye_m, persistent=False)

        feat_dim = (
            self.topk_p  # top-k eigenvalues of P
            + 5  # trace(P), log|det P|, J*, ||K||_F, CARE residual
            + self.topk_acl  # top-k Re(spec(A_cl))
            + 4  # min Re A_cl, max Re A_cl, hamiltonian imag count, hurwitz indicator
        )
        self.head = nn.Sequential(
            nn.LayerNorm(pooled_dim + feat_dim),
            nn.Linear(pooled_dim + feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def _build_a(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = pooled.shape[0]
        gates = torch.tanh(self.gate_a(pooled))  # bounded contribution
        flow = torch.einsum("bp,prc->brc", gates, self.primitives_a)
        damping = F.softplus(self.alpha) + 1.0e-3
        a = -damping * self.eye_r.unsqueeze(0).expand(b, -1, -1) + flow
        eig_a = torch.linalg.eigvals(a).real
        max_real = eig_a.amax(dim=-1)
        clip = (max_real + self.hurwitz_safety).clamp_min(0.0)
        a_clipped = a - clip.view(b, 1, 1) * self.eye_r.unsqueeze(0)
        return a_clipped, max_real

    def _build_b(self, pooled: torch.Tensor) -> torch.Tensor:
        gates = self.gate_b(pooled)
        return torch.einsum("bp,prm->brm", gates, self.primitives_b)

    def _build_q(self, pooled: torch.Tensor) -> torch.Tensor:
        b = pooled.shape[0]
        gates = F.softplus(self.gate_q(pooled))
        sqrt_w = gates.clamp_min(self.care_floor).sqrt().unsqueeze(-1).unsqueeze(-1)
        scaled = sqrt_w * self.primitives_q.unsqueeze(0)  # (B, P, r, r)
        f_cat = scaled.transpose(1, 2).reshape(b, self.r, self.num_q_primitives * self.r)
        q = f_cat @ f_cat.transpose(-1, -2)
        q = q + self.q_floor_beta * self.eye_r.unsqueeze(0)
        return 0.5 * (q + q.transpose(-1, -2))

    def _build_r(self, pooled: torch.Tensor) -> torch.Tensor:
        b = pooled.shape[0]
        gates = F.softplus(self.gate_r(pooled))
        sqrt_w = gates.clamp_min(self.care_floor).sqrt().unsqueeze(-1).unsqueeze(-1)
        scaled = sqrt_w * self.primitives_r.unsqueeze(0)
        f_cat = scaled.transpose(1, 2).reshape(b, self.m, self.num_r_primitives * self.m)
        r = f_cat @ f_cat.transpose(-1, -2)
        r = r + self.r_floor_gamma * self.eye_m.unsqueeze(0)
        return 0.5 * (r + r.transpose(-1, -2))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feat = self.trunk(x)
        pooled = torch.cat([feat.mean(dim=(2, 3)), feat.amax(dim=(2, 3))], dim=1)
        bsz = pooled.shape[0]

        a, max_real_a = self._build_a(pooled)
        b_op = self._build_b(pooled)
        q = self._build_q(pooled)
        r = self._build_r(pooled)

        # B R^{-1} B^T
        rinv_bt = torch.linalg.solve(r, b_op.transpose(-1, -2))
        b_rinv_bt = b_op @ rinv_bt
        b_rinv_bt = 0.5 * (b_rinv_bt + b_rinv_bt.transpose(-1, -2))

        p, hamiltonian_eigvals, _ = _solve_care_via_hamiltonian(
            a, b_rinv_bt, q, floor=self.care_floor
        )

        # Diagnostics on P (force PSD-ish by clamping eigenvalues for log det only).
        eigvals_p = torch.linalg.eigvalsh(p)
        eigvals_p_sorted, _ = eigvals_p.sort(dim=-1, descending=True)
        eig_topk_p = eigvals_p_sorted[:, : self.topk_p]
        trace_p = p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        log_det_p = eigvals_p.abs().clamp_min(self.care_floor).log().sum(dim=-1)
        j_star = trace_p  # x_0 ~ unit Gaussian => E[x_0^T P x_0] = trace(P)

        # Optimal feedback gain K = R^{-1} B^T P and closed-loop A_cl = A - B K.
        k_gain = torch.linalg.solve(r, b_op.transpose(-1, -2) @ p)
        k_norm = torch.linalg.matrix_norm(k_gain, ord="fro")
        a_cl = a - b_op @ k_gain
        eigvals_acl = torch.linalg.eigvals(a_cl).real
        eigvals_acl_sorted, _ = eigvals_acl.sort(dim=-1, descending=False)
        # Worst-case (largest Re) modes of A_cl - puzzles approach 0.
        acl_topk = eigvals_acl_sorted[:, -self.topk_acl :]
        acl_min = eigvals_acl.amin(dim=-1)
        acl_max = eigvals_acl.amax(dim=-1)
        hurwitz_indicator = torch.sigmoid(-8.0 * acl_max)

        # Hamiltonian imaginary-axis count: marginal modes near Re ~ 0.
        ham_imag_count = (hamiltonian_eigvals.real.abs() < self.hurwitz_safety).sum(dim=-1).to(p.dtype)

        # CARE residual ||A^T P + P A - P (B R^{-1} B^T) P + Q||_F
        residual = a.transpose(-1, -2) @ p + p @ a - p @ b_rinv_bt @ p + q
        care_residual = torch.linalg.matrix_norm(residual, ord="fro")

        scalars = torch.stack(
            [trace_p, log_det_p, j_star, k_norm, care_residual],
            dim=-1,
        )
        acl_scalars = torch.stack(
            [acl_min, acl_max, ham_imag_count, hurwitz_indicator],
            dim=-1,
        )
        feat_vec = torch.cat([eig_topk_p, scalars, acl_topk, acl_scalars], dim=-1)
        logits = self.head(torch.cat([pooled, feat_vec], dim=-1)).view(-1)
        return {
            "logits": logits,
            "riccati_eigvals_P": eigvals_p_sorted,
            "riccati_top_eig_P": eig_topk_p,
            "riccati_trace_P": trace_p,
            "riccati_log_det_P": log_det_p,
            "riccati_optimal_cost_J_star": j_star,
            "riccati_gain_norm_K": k_norm,
            "riccati_closed_loop_top_real": acl_topk,
            "riccati_closed_loop_min_real": acl_min,
            "riccati_closed_loop_max_real": acl_max,
            "riccati_hurwitz_indicator": hurwitz_indicator,
            "riccati_hamiltonian_imag_count": ham_imag_count,
            "riccati_care_residual_F": care_residual,
            "riccati_open_loop_max_real_A": max_real_a,
        }


def build_riccati_optimal_defense_network_from_config(
    config: dict[str, Any],
) -> RiccatiOptimalDefenseNetwork:
    cfg = dict(config)
    return RiccatiOptimalDefenseNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        riccati_rank_r=int(cfg.get("riccati_rank_r", 12)),
        defender_dim_m=int(cfg.get("defender_dim_m", 4)),
        num_a_primitives=int(cfg.get("num_a_primitives", 8)),
        num_b_primitives=int(cfg.get("num_b_primitives", 8)),
        num_q_primitives=int(cfg.get("num_q_primitives", 4)),
        num_r_primitives=int(cfg.get("num_r_primitives", 4)),
        alpha_init=float(cfg.get("alpha_init", 1.0)),
        hurwitz_safety=float(cfg.get("hurwitz_safety", 0.1)),
        q_floor_beta=float(cfg.get("q_floor_beta", 1.0e-3)),
        r_floor_gamma=float(cfg.get("r_floor_gamma", 1.0e-3)),
        care_floor=float(cfg.get("care_floor", 1.0e-4)),
        topk_p=int(cfg.get("topk_p", 6)),
        topk_acl=int(cfg.get("topk_acl", 4)),
        num_classes=int(cfg.get("num_classes", 1)),
    )

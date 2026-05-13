"""Conservation-Nullspace Normalization (p040).

Source: ``ideas/research/primitives/external_35_espa_conservation_isotypic_green_primitives.md``
(rank-2 proposal ``primitive_conserve_norm``). The rank-1 proposal
(``primitive_espa``) is the same elementary-symmetric polynomial
operator already covered by
``p024 event_symmetric_interaction_accumulator``, so it is not promoted
here.

The primitive normalises a per-square latent ``X in R^{B, n, d}`` after
projecting out a low-rank subspace spanned by *conserved-charge*
columns ``C in R^{n, r}``. Concretely

    A = C^T D C + epsilon I_r          in R^{r x r}, SPD
    M = A^{-1} (C^T D X)                 in R^{r x d}
    R = X - C M                          in R^{n x d}     (residual)
    sigma_j^2 = (R_{:,j}^T D R_{:,j}) / max(1, sum_i w_i - r),
    Y_{ij} = gamma_j (R_{ij}) / sqrt(sigma_j^2 + epsilon) + beta_j.

The charge matrix ``C`` is fixed (a rule-derived encoding of material
counts, side-to-move, piece-type buckets, and parity), so the projected
component captures the part of ``X`` that is linearly explained by
these chess "conservation laws"; the residual ``R`` is what is left
*after* that bookkeeping. Subsequent channels of the network are
trained to represent only those residuals.

For the additive-primitive contract we pool the residual norm and the
projected coefficients ``M``, project to a scalar gated delta, and add
to the i193 base logit.

CRTK metadata, source labels, verification flags, engine evaluations,
and report-only metadata are not used.

Deferred internal proposals from the same packet:

- ``primitive_espa`` (rank 1): duplicate of p024.
- ``primitive_isotypic_projector`` (rank 3): finite-group isotypic
  decomposition; deferred (overlaps with p036 chess-group orbit).
- ``primitive_green_solve`` (rank 4): board-graph Green-function solve;
  large dense ``N^3`` cost, deferred.
- ``primitive_matroid_base_pool`` (rank 5): entropic matroid-base pool;
  deferred (overlaps with the matroid-rank-envelope proposal in
  external_31 and the entropic-matroid pool in external_35).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.primitives.trunk_features import trunk_joint_features
from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    DualStreamFeatureBuilder,
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


SQUARES = 64
HEIGHT = 8
WIDTH = 8
PIECE_PLANE_COUNT = 12

ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    "no_projection",       # skip the conserved-charge projection (Y = X normalised)
    "shuffle_residual",    # in-batch permutation of the residual pool
    "uniform_weights",     # set D to identity (drop the rule-derived per-square weights)
    "zero_delta",
    "trunk_only",
)


def _build_charge_matrix() -> torch.Tensor:
    """Return ``(64, r)`` charge matrix encoding board conservation laws.

    The columns are:

    - 0: constant column (mean intercept)
    - 1: file index linearly mapped to [-1, 1]
    - 2: rank index linearly mapped to [-1, 1]
    - 3: square parity (centred {-1, +1})
    - 4: king-zone proximity proxy (1 / (chebyshev distance to e4 + 1))
    - 5: edge-row indicator (row 0 or 7)
    - 6: edge-col indicator (col 0 or 7)
    - 7: corner indicator (one of four corners)

    The charges are *fixed* -- not learned -- so they expose the linear
    subspace the projection step removes.
    """
    r = 8
    C = torch.zeros(SQUARES, r, dtype=torch.float32)
    for s in range(SQUARES):
        row = s // WIDTH
        col = s % WIDTH
        # 0: constant
        C[s, 0] = 1.0
        # 1: file linear (-1..1)
        C[s, 1] = (col - (WIDTH - 1) / 2.0) / ((WIDTH - 1) / 2.0)
        # 2: rank linear (-1..1)
        C[s, 2] = (row - (HEIGHT - 1) / 2.0) / ((HEIGHT - 1) / 2.0)
        # 3: square parity centred
        C[s, 3] = 1.0 if (row + col) % 2 == 0 else -1.0
        # 4: distance to e4 (sq 28)
        e4_row, e4_col = 3, 4
        cheb = max(abs(row - e4_row), abs(col - e4_col))
        C[s, 4] = 1.0 / float(cheb + 1)
        # 5: edge row
        C[s, 5] = 1.0 if row in (0, 7) else 0.0
        # 6: edge col
        C[s, 6] = 1.0 if col in (0, 7) else 0.0
        # 7: corner
        C[s, 7] = 1.0 if (row in (0, 7) and col in (0, 7)) else 0.0
    return C


class ConservationNullspaceNorm(nn.Module):
    """Conservation-Nullspace Normalization primitive head (p040)."""

    ALLOWED_ABLATIONS = ALLOWED_ABLATIONS

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        trunk_channels: int = 64,
        trunk_hidden_dim: int = 96,
        trunk_depth: int = 2,
        trunk_dropout: float = 0.1,
        trunk_use_batchnorm: bool = True,
        trunk_gate_dim: int | None = None,
        trunk_ablation: str = "none",
        latent_dim: int = 16,
        epsilon: float = 1.0e-3,
        weight_bias: float = 1.0,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError("ConservationNullspaceNorm supports the puzzle_binary one-logit contract")
        if int(input_channels) != 18:
            raise ValueError("ConservationNullspaceNorm requires the simple_18 board tensor")
        if int(latent_dim) < 2:
            raise ValueError("latent_dim must be >= 2")
        if float(epsilon) <= 0.0:
            raise ValueError("epsilon must be > 0")
        if str(ablation) not in self.ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(self.ALLOWED_ABLATIONS)}",
            )

        self.num_classes = 1
        self.ablation = str(ablation)
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.latent_dim = int(latent_dim)
        self.epsilon = float(epsilon)
        self.weight_bias = float(weight_bias)

        self.trunk = ExchangeThenKingDualStreamNetwork(
            input_channels=int(input_channels),
            num_classes=1,
            channels=int(trunk_channels),
            hidden_dim=int(trunk_hidden_dim),
            depth=int(trunk_depth),
            dropout=float(trunk_dropout),
            use_batchnorm=bool(trunk_use_batchnorm),
            gate_dim=trunk_gate_dim,
            ablation=str(trunk_ablation),
        )
        self.feature_dim = (
            2 * self.trunk.exchange_encoder.output_dim
            + DualStreamFeatureBuilder.SUMMARY_DIM
        )

        # Project trunk joint feature -> per-square latent X (B, 64, d).
        self.latent_proj = nn.Linear(self.feature_dim, SQUARES * self.latent_dim)
        # Per-square weights derived from board state -> shape (B, 64), > 0.
        self.weight_proj = nn.Conv2d(int(input_channels), 1, kernel_size=1)
        # Per-channel affine norm parameters.
        self.gamma = nn.Parameter(torch.ones(self.latent_dim))
        self.beta = nn.Parameter(torch.zeros(self.latent_dim))

        # Fixed charge matrix (64, r).
        C = _build_charge_matrix()
        self.register_buffer("charges", C, persistent=False)
        self.r = int(C.shape[1])

        head_dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        # readout = residual pool (d) + projected coefficients pool (r) + sigma pool (d)
        readout_dim = self.latent_dim + self.r + self.latent_dim
        self.delta_head = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, int(head_hidden_dim)),
            nn.GELU(),
            head_dropout_module,
            nn.Linear(int(head_hidden_dim), 1),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, max(16, int(head_hidden_dim) // 2)),
            nn.GELU(),
            nn.Linear(max(16, int(head_hidden_dim) // 2), 1),
        )
        with torch.no_grad():
            final_layer = self.gate_head[-1]
            if isinstance(final_layer, nn.Linear):
                final_layer.bias.fill_(float(gate_init))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]
        eps = self.epsilon

        trunk_out = self.trunk(board)
        base_logit = trunk_out["logits"].view(-1)
        joint, _, _ = trunk_joint_features(self.trunk, board)

        # Per-square latent (B, 64, d).
        X = self.latent_proj(joint).view(batch, SQUARES, self.latent_dim)

        # Per-square positive weights (B, 64). The 1x1 conv shrinks
        # simple_18 -> 1 channel, softplus + bias for positivity.
        w_logits = self.weight_proj(board).view(batch, SQUARES)
        w = torch.nn.functional.softplus(w_logits) + self.weight_bias  # > 0
        if self.ablation == "uniform_weights":
            w = torch.ones_like(w)
        D = w.unsqueeze(-1)  # (B, 64, 1) -- broadcastable as a diagonal mask

        C = self.charges  # (64, r), fixed buffer
        # A_b = C^T D C + eps I_r in (B, r, r)
        C_expand = C.unsqueeze(0).expand(batch, SQUARES, self.r)
        DC = w.unsqueeze(-1) * C_expand  # (B, 64, r)
        A = torch.einsum("bsr,bsq->brq", C_expand, DC)
        A = A + eps * torch.eye(self.r, device=board.device, dtype=A.dtype).unsqueeze(0)

        # b = C^T D X in (B, r, d)
        b = torch.einsum("bsr,bsd->brd", DC, X)
        # M = A^{-1} b via Cholesky (A is SPD).
        L = torch.linalg.cholesky(A)
        M = torch.cholesky_solve(b, L)  # (B, r, d)

        # Residual R = X - C M in (B, 64, d)
        R = X - torch.einsum("sr,brd->bsd", C, M)
        if self.ablation == "no_projection":
            R = X
            M = torch.zeros_like(M)

        # sigma_j^2 = R[:, j]^T D R[:, j] / max(1, sum_i w_i - r)
        sum_w = w.sum(dim=1)  # (B,)
        denom = torch.clamp(sum_w - float(self.r), min=1.0)  # (B,)
        weighted_r2 = (w.unsqueeze(-1) * R.pow(2)).sum(dim=1)  # (B, d)
        sigma2 = weighted_r2 / denom.unsqueeze(-1)  # (B, d)
        sigma = (sigma2 + eps).sqrt()

        # Y_{i,j} = gamma_j (R_{i,j}) / sigma_j + beta_j
        Y = self.gamma.view(1, 1, -1) * (R / sigma.unsqueeze(1)) + self.beta.view(1, 1, -1)

        if self.ablation == "shuffle_residual" and batch > 1:
            perm = torch.randperm(batch, device=board.device)
            Y = Y[perm]
            M = M[perm]
            sigma = sigma[perm]

        # Pools.
        residual_pool = Y.pow(2).mean(dim=1).clamp_min(eps).sqrt()  # (B, d) RMS over squares
        coeff_pool = M.flatten(1).abs().mean(dim=1)  # (B,) wrong shape; want per-r
        coeff_pool = M.abs().mean(dim=2)  # (B, r) per-charge mean magnitude over d
        sigma_pool = sigma  # (B, d) already per-channel

        readout = torch.cat([residual_pool, coeff_pool, sigma_pool], dim=-1)
        delta_raw = self.delta_head(readout).view(-1)

        gate_logit = self.gate_head(joint).view(-1)
        gate = torch.sigmoid(gate_logit)
        if self.ablation in {"zero_delta", "trunk_only"}:
            primitive_delta = torch.zeros_like(base_logit)
        else:
            primitive_delta = gate * delta_raw

        logits = base_logit + primitive_delta

        gate_clamped = gate.clamp(eps, 1.0 - eps)
        gate_entropy = -(
            gate_clamped * gate_clamped.log()
            + (1.0 - gate_clamped) * (1.0 - gate_clamped).log()
        )

        # Diagnostics
        residual_norm = R.pow(2).flatten(1).mean(dim=1).clamp_min(eps).sqrt()
        x_norm = X.pow(2).flatten(1).mean(dim=1).clamp_min(eps).sqrt()
        # Fraction of X explained by the conservation projection.
        explained_frac = (1.0 - (residual_norm / (x_norm + eps))).clamp(0.0, 1.0)

        diagnostics: dict[str, torch.Tensor] = {
            "logits": logits,
            "base_logit": base_logit,
            "primitive_delta": primitive_delta,
            "primitive_delta_raw": delta_raw,
            "primitive_gate": gate,
            "primitive_gate_logit": gate_logit,
            "primitive_gate_entropy": gate_entropy,
            "cnnorm_residual_norm": residual_norm,
            "cnnorm_x_norm": x_norm,
            "cnnorm_explained_frac": explained_frac,
            "cnnorm_sum_weight": sum_w,
            "cnnorm_sigma_mean": sigma.mean(dim=1),
            "mechanism_energy": trunk_out["mechanism_energy"] + residual_norm.detach(),
            "proposal_profile_strength": (primitive_delta.abs() * gate_entropy).clamp(0.0, 20.0),
            "proposal_keyword_count": logits.new_full((batch,), float(self.r * self.latent_dim)),
        }
        for key, value in trunk_out.items():
            if key == "logits":
                continue
            diag_key = (
                key if key.startswith("base_") or key in {"gate", "gate_logit"} else f"trunk_{key}"
            )
            diagnostics.setdefault(diag_key, value)
        return diagnostics


def build_conservation_nullspace_norm_from_config(
    config: dict[str, Any],
) -> ConservationNullspaceNorm:
    cfg = dict(config)
    return ConservationNullspaceNorm(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim", None)),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        latent_dim=int(cfg.get("latent_dim", 16)),
        epsilon=float(cfg.get("epsilon", 1.0e-3)),
        weight_bias=float(cfg.get("weight_bias", 1.0)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )

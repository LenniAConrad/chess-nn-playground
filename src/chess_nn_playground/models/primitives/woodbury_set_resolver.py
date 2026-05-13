"""Woodbury Set Resolver (p038).

Source: ``ideas/research/primitives/external_33_esp_permanent_woodbury_orbit_primitives.md``
(rank-3 proposal ``primitive_woodbury_resolver``). The rank-1 proposal in
the same packet (``primitive_esp_set``) is the same elementary-symmetric
polynomial operator already covered by
``p024 event_symmetric_interaction_accumulator``, so it is not promoted
here.

The primitive maintains an active-set inverse-precision memory

    A_b = lambda * I + sum_i  m_{b,i} U_{b,i} U_{b,i}^T   in R^{r x r}
    S_b = sum_i  m_{b,i} U_{b,i} V_{b,i}^T                 in R^{r x d_v}
    Y_b = Q_b @ A_b^{-1} @ S_b                              in R^{m x d_v}

and exposes per-token leverage scores ``l_{b,i} = U_{b,i}^T A_b^{-1}
U_{b,i}`` and the log-determinant ``log det A_b``. Adds and deletes are
exact rank-one Sherman-Morrison updates; we compute the static result
here (training time is one position per sample). The same operator also
covers the ``primitive_rank1_resolvent_pool`` proposal from
``external_36_exterior_product_rank1_resolvent_primitives.md``, so that
one is deferred to keep the worktree non-redundant.

CRTK metadata, source labels, verification flags, engine evaluations,
and report-only metadata are not consumed.

Deferred internal proposals from the same packet:

- ``primitive_esp_set`` (rank 1): duplicate of p024.
- ``primitive_permanent_roles`` (rank 2): exact permanent role
  assignment; deferred (``permanent_ryser_network`` already implements a
  related Ryser-permanent operator on a different scale).
- ``primitive_orbit_canonicalizer`` (rank 4): duplicate of p036.
- ``primitive_component_pool`` (rank 5): transitive-component pool;
  deferred (graph connected-component infrastructure).
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
PIECE_PLANE_COUNT = 12

ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    "diagonal_only",          # zero off-diagonal of A_b before solve
    "shuffle_active_tokens",  # in-batch permutation of the active piece tokens
    "uniform_queries",        # replace Q with all-ones (matched dim)
    "zero_delta",
    "trunk_only",
)


def _piece_tokens(board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-square token input + occupancy mask drawn from simple_18 planes."""
    piece_planes = board[:, :PIECE_PLANE_COUNT].clamp(0.0, 1.0)
    stm = board[:, 12:13].clamp(0.0, 1.0)
    castling = board[:, 13:17].clamp(0.0, 1.0)
    token_input = torch.cat([piece_planes, stm, castling], dim=1).flatten(2).transpose(1, 2).contiguous()
    occupancy = piece_planes.flatten(2).sum(dim=1).clamp(0.0, 1.0)
    return token_input, occupancy


class WoodburySetResolver(nn.Module):
    """Woodbury Set Resolver primitive head (p038)."""

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
        token_input_dim: int | None = None,
        u_dim: int = 12,
        v_dim: int = 16,
        num_queries: int = 4,
        lambda_reg: float = 1.0e-2,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError("WoodburySetResolver supports the puzzle_binary one-logit contract")
        if int(input_channels) != 18:
            raise ValueError("WoodburySetResolver requires the simple_18 board tensor")
        if int(u_dim) < 2:
            raise ValueError("u_dim must be >= 2")
        if int(v_dim) < 1:
            raise ValueError("v_dim must be >= 1")
        if int(num_queries) < 1:
            raise ValueError("num_queries must be >= 1")
        if float(lambda_reg) <= 0.0:
            raise ValueError("lambda_reg must be > 0 to keep A SPD")
        if str(ablation) not in self.ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(self.ALLOWED_ABLATIONS)}",
            )

        self.num_classes = 1
        self.ablation = str(ablation)
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.u_dim = int(u_dim)
        self.v_dim = int(v_dim)
        self.num_queries = int(num_queries)
        self.lambda_reg = float(lambda_reg)

        if token_input_dim is None:
            token_input_dim = PIECE_PLANE_COUNT + 1 + 4  # piece planes + stm + 4 castling
        self.token_input_dim = int(token_input_dim)

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

        self.u_proj = nn.Linear(self.token_input_dim, self.u_dim)
        self.v_proj = nn.Linear(self.token_input_dim, self.v_dim)
        # Queries are produced from the trunk joint feature.
        self.query_proj = nn.Linear(self.feature_dim, self.num_queries * self.u_dim)

        head_dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        readout_dim = self.num_queries * self.v_dim + 1 + 1  # queries + logdet + leverage mean
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
        eps = 1.0e-6

        trunk_out = self.trunk(board)
        base_logit = trunk_out["logits"].view(-1)
        joint, _, _ = trunk_joint_features(self.trunk, board)

        token_input, occupancy = _piece_tokens(board)  # (B, 64, token_input_dim), (B, 64)
        if self.ablation == "shuffle_active_tokens" and batch > 1:
            perm = torch.randperm(batch, device=board.device)
            token_input = token_input[perm]
            occupancy = occupancy[perm]

        u_tokens = self.u_proj(token_input)  # (B, 64, r)
        v_tokens = self.v_proj(token_input)  # (B, 64, d_v)

        # Mask out empty squares so they contribute zero to A_b and S_b.
        mask = occupancy.unsqueeze(-1)  # (B, 64, 1)
        u_active = u_tokens * mask
        v_active = v_tokens * mask

        # A_b = lambda * I + sum_i U_i U_i^T   (B, r, r)
        A = torch.einsum("bnr,bns->brs", u_active, u_active)
        A = A + self.lambda_reg * torch.eye(self.u_dim, device=board.device, dtype=A.dtype).unsqueeze(0)
        if self.ablation == "diagonal_only":
            diag_A = torch.diagonal(A, dim1=-2, dim2=-1)  # (B, r)
            A = torch.diag_embed(diag_A)

        # S_b = sum_i U_i V_i^T                (B, r, d_v)
        S = torch.einsum("bnr,bnv->brv", u_active, v_active)

        # Queries (B, m, r) from trunk joint feature.
        if self.ablation == "uniform_queries":
            Q = u_active.new_ones(batch, self.num_queries, self.u_dim) / float(self.u_dim)
        else:
            Q = self.query_proj(joint).view(batch, self.num_queries, self.u_dim)

        # Solve A_b @ P = I and apply Q @ P @ S.
        # P_b = A_b^{-1}  -- use Cholesky for SPD.
        L = torch.linalg.cholesky(A)
        # Solve A_b Y_in = S for Y_in = A_b^{-1} S  in (B, r, d_v).
        SinvA = torch.cholesky_solve(S, L)
        # Y_b = Q_b @ SinvA  in (B, m, d_v).
        Y = torch.einsum("bmr,brv->bmv", Q, SinvA)

        # Leverage scores per active token: l_i = u_i^T A^{-1} u_i.
        # Compute via A^{-1} u_i -> ainv_U  (B, 64, r).
        ainv_U = torch.cholesky_solve(u_active.transpose(1, 2), L).transpose(1, 2)
        leverage_per_token = (u_active * ainv_U).sum(dim=-1)  # (B, 64)
        leverage_per_token = leverage_per_token * occupancy  # zero on empty squares
        active_count = occupancy.sum(dim=1).clamp_min(1.0)
        leverage_mean = leverage_per_token.sum(dim=1) / active_count  # (B,)

        # log det A_b = 2 * sum(log(diag(L))).
        log_det = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1).clamp_min(eps)).sum(dim=-1)

        readout = torch.cat(
            [Y.flatten(1), log_det.unsqueeze(-1), leverage_mean.unsqueeze(-1)],
            dim=-1,
        )
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

        leverage_max = leverage_per_token.amax(dim=1)
        a_norm = A.flatten(1).pow(2).mean(dim=1).clamp_min(eps).sqrt()

        diagnostics: dict[str, torch.Tensor] = {
            "logits": logits,
            "base_logit": base_logit,
            "primitive_delta": primitive_delta,
            "primitive_delta_raw": delta_raw,
            "primitive_gate": gate,
            "primitive_gate_logit": gate_logit,
            "primitive_gate_entropy": gate_entropy,
            "wsr_active_count": occupancy.sum(dim=1),
            "wsr_logdet_A": log_det,
            "wsr_leverage_mean": leverage_mean,
            "wsr_leverage_max": leverage_max,
            "wsr_A_norm": a_norm,
            "mechanism_energy": trunk_out["mechanism_energy"] + a_norm.detach(),
            "proposal_profile_strength": (primitive_delta.abs() * gate_entropy).clamp(0.0, 20.0),
            "proposal_keyword_count": logits.new_full((batch,), float(self.u_dim * self.v_dim)),
        }
        for key, value in trunk_out.items():
            if key == "logits":
                continue
            diag_key = (
                key if key.startswith("base_") or key in {"gate", "gate_logit"} else f"trunk_{key}"
            )
            diagnostics.setdefault(diag_key, value)
        return diagnostics


def build_woodbury_set_resolver_from_config(
    config: dict[str, Any],
) -> WoodburySetResolver:
    cfg = dict(config)
    return WoodburySetResolver(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim", None)),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        token_input_dim=cfg.get("token_input_dim", None),
        u_dim=int(cfg.get("u_dim", 12)),
        v_dim=int(cfg.get("v_dim", 16)),
        num_queries=int(cfg.get("num_queries", 4)),
        lambda_reg=float(cfg.get("lambda_reg", 1.0e-2)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )

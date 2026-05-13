"""Regret Saddlepoint Network (p002).

Bespoke additive head over the i193 dual-stream trunk that compiles
candidate and reply tokens from the trunk's spatial features, builds a
bilinear payoff table over them, and reduces the table with the
differentiable entropy-regularized saddlepoint solver described in
``ideas/research/primitives/codex_02_regret_saddlepoint.md``. The
operator returns the saddle value, both equilibrium strategies, regrets,
and an exploitability scalar which are concatenated with the trunk
joint feature and fused through a small gated MLP head.

The model is additive and gated: ``trunk_only`` / ``zero_delta``
recover the i193 baseline exactly, so the primitive can be removed
without disturbing the trunk. No CRTK metadata, source labels, engine
evaluations, or report-only metadata are consumed.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.primitives.codex_reply_primitives import (
    BoardTokenAttention,
    regret_saddlepoint,
    run_i193_trunk_with_spatial,
)
from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    "row_shuffle_payoff",      # permute payoff rows: destroys candidate-side game structure
    "col_shuffle_payoff",      # permute payoff columns: destroys reply-side game structure
    "uniform_payoff",          # collapse payoff to per-batch mean: no game structure
    "pure_max_min",            # disable solver: use raw max_i min_j A_ij
    "zero_delta",              # primitive_delta == 0
    "disable_gate",            # gate clamped to 1.0
    "trunk_only",              # disable both features and delta
)


class RegretSaddlepointNetwork(nn.Module):
    """p002 — Regret Saddlepoint head over the i193 dual-stream trunk."""

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
        num_candidates: int = 16,
        num_replies: int = 12,
        token_dim: int = 48,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        solver_iters: int = 24,
        tau_p: float = 0.45,
        tau_q: float = 0.45,
        solver_damp: float = 0.35,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "RegretSaddlepointNetwork supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError(
                "RegretSaddlepointNetwork requires the simple_18 board tensor"
            )
        if int(num_candidates) < 2 or int(num_replies) < 2:
            raise ValueError("num_candidates and num_replies must both be >= 2")
        if str(ablation) not in ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(ALLOWED_ABLATIONS)}"
            )

        self.num_classes = 1
        self.ablation = str(ablation)
        self.num_candidates = int(num_candidates)
        self.num_replies = int(num_replies)
        self.token_dim = int(token_dim)
        self.solver_iters = int(solver_iters)
        self.tau_p = float(tau_p)
        self.tau_q = float(tau_q)
        self.solver_damp = float(solver_damp)

        self.spec = BoardTensorSpec(input_channels=int(input_channels))
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
        spatial_channels = 2 * self.trunk.channels
        self.candidate_pool = BoardTokenAttention(
            in_channels=spatial_channels,
            num_tokens=self.num_candidates,
            token_dim=self.token_dim,
            dropout=float(head_dropout),
        )
        self.reply_pool = BoardTokenAttention(
            in_channels=spatial_channels,
            num_tokens=self.num_replies,
            token_dim=self.token_dim,
            dropout=float(head_dropout),
        )

        joint_dim = (
            2 * self.trunk.exchange_encoder.output_dim
            + self.trunk.feature_builder.SUMMARY_DIM
        )
        self.cand_proj = nn.Linear(self.token_dim, self.token_dim)
        self.reply_proj = nn.Linear(self.token_dim, self.token_dim)
        self.payoff_scale = nn.Parameter(torch.zeros(()))
        self.payoff_bias = nn.Linear(joint_dim, 1)

        fusion_dim = 2 * self.token_dim + joint_dim + 5
        self.delta_mlp = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, int(head_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity(),
            nn.Linear(int(head_hidden_dim), 1),
        )
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(joint_dim + 3),
            nn.Linear(joint_dim + 3, int(head_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(head_hidden_dim), 1),
        )
        with torch.no_grad():
            final_gate = self.gate_mlp[-1]
            if isinstance(final_gate, nn.Linear):
                final_gate.bias.fill_(float(gate_init))

    def _build_payoff(
        self,
        cand_tokens: torch.Tensor,
        reply_tokens: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        cand = self.cand_proj(cand_tokens)
        reply = self.reply_proj(reply_tokens)
        bias = self.payoff_bias(context).view(-1, 1, 1)
        scale = torch.tanh(self.payoff_scale)
        payoff = scale * torch.einsum("bkd,brd->bkr", cand, reply) + bias
        if self.ablation == "row_shuffle_payoff" and payoff.shape[1] > 1:
            perm = torch.randperm(payoff.shape[1], device=payoff.device)
            payoff = payoff[:, perm]
        elif self.ablation == "col_shuffle_payoff" and payoff.shape[2] > 1:
            perm = torch.randperm(payoff.shape[2], device=payoff.device)
            payoff = payoff[:, :, perm]
        elif self.ablation == "uniform_payoff":
            payoff = payoff.mean(dim=(1, 2), keepdim=True).expand_as(payoff)
        return payoff

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        trunk_feats = run_i193_trunk_with_spatial(self.trunk, board)
        base_logit = trunk_feats.base_logit

        cand = self.candidate_pool(trunk_feats.spatial)
        reply = self.reply_pool(trunk_feats.spatial)
        payoff = self._build_payoff(cand.tokens, reply.tokens, trunk_feats.pool)

        if self.ablation == "pure_max_min":
            best_per_row = payoff.amin(dim=-1)
            value = best_per_row.amax(dim=-1)
            B = payoff.shape[0]
            p = payoff.new_zeros(B, self.num_candidates)
            q = payoff.new_zeros(B, self.num_replies)
            p[torch.arange(B), best_per_row.argmax(dim=-1)] = 1.0
            q[torch.arange(B), payoff.amin(dim=-1).argmin(dim=-1)] = 1.0
            attacker_regret = payoff.new_zeros(B)
            defender_regret = payoff.new_zeros(B)
            exploitability = payoff.new_zeros(B)
            attacker_entropy = payoff.new_zeros(B)
            defender_entropy = payoff.new_zeros(B)
        else:
            reduced = regret_saddlepoint(
                payoff,
                tau_p=self.tau_p,
                tau_q=self.tau_q,
                iters=self.solver_iters,
                damp=self.solver_damp,
            )
            value = reduced["value"]
            p = reduced["attacker_strategy"]
            q = reduced["defender_strategy"]
            attacker_regret = reduced["attacker_regret"]
            defender_regret = reduced["defender_regret"]
            exploitability = reduced["exploitability"]
            attacker_entropy = reduced["attacker_entropy"]
            defender_entropy = reduced["defender_entropy"]

        attacker_pool = torch.einsum("bk,bkd->bd", p, cand.tokens)
        defender_pool = torch.einsum("br,brd->bd", q, reply.tokens)
        diagnostics = torch.stack(
            [value, attacker_regret, defender_regret, attacker_entropy, defender_entropy],
            dim=-1,
        )
        fusion = torch.cat([attacker_pool, defender_pool, trunk_feats.pool, diagnostics], dim=-1)
        delta_raw = self.delta_mlp(fusion).view(-1)
        gate_input = torch.cat(
            [
                trunk_feats.pool,
                value.unsqueeze(-1),
                exploitability.unsqueeze(-1),
                attacker_entropy.unsqueeze(-1),
            ],
            dim=-1,
        )
        gate_logit = self.gate_mlp(gate_input).view(-1)
        gate = torch.sigmoid(gate_logit)
        if self.ablation == "disable_gate":
            gate = torch.ones_like(gate)
        if self.ablation in {"zero_delta", "trunk_only"}:
            primitive_delta = torch.zeros_like(base_logit)
        else:
            primitive_delta = gate * delta_raw

        logits = base_logit + primitive_delta

        out: dict[str, torch.Tensor] = dict(trunk_feats.base_output)
        out["logits"] = logits
        out["base_logit"] = base_logit
        out["primitive_delta"] = primitive_delta
        out["primitive_delta_raw"] = delta_raw
        out["primitive_gate"] = gate
        out["primitive_gate_logit"] = gate_logit
        out["rsp_saddle_value"] = value
        out["rsp_attacker_regret"] = attacker_regret
        out["rsp_defender_regret"] = defender_regret
        out["rsp_exploitability"] = exploitability
        out["rsp_attacker_entropy"] = attacker_entropy
        out["rsp_defender_entropy"] = defender_entropy
        out["rsp_best_witness_index"] = p.argmax(dim=-1).to(dtype=base_logit.dtype)
        out["rsp_best_reply_index"] = q.argmax(dim=-1).to(dtype=base_logit.dtype)
        return out


def build_regret_saddlepoint_network_from_config(
    config: dict[str, Any],
) -> RegretSaddlepointNetwork:
    cfg = dict(config)
    return RegretSaddlepointNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim", None)),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        num_candidates=int(cfg.get("num_candidates", 16)),
        num_replies=int(cfg.get("num_replies", 12)),
        token_dim=int(cfg.get("token_dim", 48)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        solver_iters=int(cfg.get("solver_iters", 24)),
        tau_p=float(cfg.get("tau_p", 0.45)),
        tau_q=float(cfg.get("tau_q", 0.45)),
        solver_damp=float(cfg.get("solver_damp", 0.35)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )

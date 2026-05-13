"""Reply Channel Capacity Network (p003).

Bespoke additive head over the i193 dual-stream trunk that compiles
candidate and reply tokens from the trunk's spatial features, builds a
candidate-conditional reply-logit table, and reduces the table with the
differentiable Blahut-Arimoto channel-capacity reducer described in
``ideas/research/primitives/codex_03_reply_channel_capacity.md``. The
operator returns capacity (nats and bits), the capacity-achieving
candidate prior, the reply marginal, the conditional and output
entropies, and the capacity gap. These are concatenated with the trunk
joint feature and fused via a small gated MLP head.

The model is additive and gated: ``trunk_only`` / ``zero_delta``
recover the i193 baseline exactly. No CRTK metadata, source labels,
engine evaluations, or report-only metadata are consumed.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.primitives.codex_reply_primitives import (
    BoardTokenAttention,
    reply_channel_capacity,
    run_i193_trunk_with_spatial,
)
from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    "row_shuffle_channel",     # permute per-row reply distributions (kills capacity)
    "duplicate_rows",          # all rows = first row's distribution (capacity collapses)
    "uniform_replies",         # uniform reply distribution per row
    "entropy_only",            # bypass operator: feed conditional entropy as the only summary
    "zero_delta",
    "disable_gate",
    "trunk_only",
)


class ReplyChannelCapacityNetwork(nn.Module):
    """p003 — Reply Channel Capacity head over the i193 dual-stream trunk."""

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
        capacity_tau: float = 1.0,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "ReplyChannelCapacityNetwork supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError(
                "ReplyChannelCapacityNetwork requires the simple_18 board tensor"
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
        self.capacity_tau = float(capacity_tau)

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
        self.logit_scale = nn.Parameter(torch.zeros(()))

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

    def _build_logits(
        self,
        cand_tokens: torch.Tensor,
        reply_tokens: torch.Tensor,
    ) -> torch.Tensor:
        cand = self.cand_proj(cand_tokens)
        reply = self.reply_proj(reply_tokens)
        scale = torch.tanh(self.logit_scale) + 1.0
        logits = scale * torch.einsum("bkd,brd->bkr", cand, reply)
        if self.ablation == "row_shuffle_channel" and logits.shape[1] > 1:
            perm = torch.randperm(logits.shape[1], device=logits.device)
            logits = logits[:, perm]
        elif self.ablation == "duplicate_rows":
            logits = logits[:, :1, :].expand(-1, logits.shape[1], -1).contiguous()
        elif self.ablation == "uniform_replies":
            logits = logits.mean(dim=-1, keepdim=True).expand_as(logits)
        return logits

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        trunk_feats = run_i193_trunk_with_spatial(self.trunk, board)
        base_logit = trunk_feats.base_logit

        cand = self.candidate_pool(trunk_feats.spatial)
        reply = self.reply_pool(trunk_feats.spatial)
        reply_logits = self._build_logits(cand.tokens, reply.tokens)

        reduced = reply_channel_capacity(
            reply_logits,
            iters=self.solver_iters,
            tau=self.capacity_tau,
        )
        capacity = reduced["capacity_nats"]
        cond_entropy = reduced["conditional_entropy"]
        out_entropy = reduced["output_entropy"]
        capacity_gap = reduced["capacity_gap"]
        capacity_bits = reduced["capacity_bits"]
        prior = reduced["capacity_achieving_prior"]
        marginal = reduced["reply_marginal"]

        attacker_pool = torch.einsum("bk,bkd->bd", prior, cand.tokens)
        defender_pool = torch.einsum("br,brd->bd", marginal, reply.tokens)
        diagnostics = torch.stack(
            [capacity, capacity_bits, cond_entropy, out_entropy, capacity_gap],
            dim=-1,
        )
        if self.ablation == "entropy_only":
            zeros = torch.zeros_like(diagnostics)
            zeros[..., 2] = diagnostics[..., 2]
            diagnostics = zeros
            attacker_pool = torch.zeros_like(attacker_pool)
            defender_pool = torch.zeros_like(defender_pool)

        fusion = torch.cat([attacker_pool, defender_pool, trunk_feats.pool, diagnostics], dim=-1)
        delta_raw = self.delta_mlp(fusion).view(-1)
        gate_input = torch.cat(
            [
                trunk_feats.pool,
                capacity.unsqueeze(-1),
                capacity_gap.unsqueeze(-1),
                cond_entropy.unsqueeze(-1),
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
        out["rcc_capacity_nats"] = capacity
        out["rcc_capacity_bits"] = capacity_bits
        out["rcc_conditional_entropy"] = cond_entropy
        out["rcc_output_entropy"] = out_entropy
        out["rcc_capacity_gap"] = capacity_gap
        out["rcc_prior_entropy"] = -(
            prior.clamp_min(1.0e-8) * prior.clamp_min(1.0e-8).log()
        ).sum(dim=-1)
        out["rcc_marginal_entropy"] = out_entropy
        return out


def build_reply_channel_capacity_network_from_config(
    config: dict[str, Any],
) -> ReplyChannelCapacityNetwork:
    cfg = dict(config)
    return ReplyChannelCapacityNetwork(
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
        capacity_tau=float(cfg.get("capacity_tau", 1.0)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )

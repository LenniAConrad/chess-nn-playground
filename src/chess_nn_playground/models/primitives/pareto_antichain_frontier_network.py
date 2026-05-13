"""Pareto Antichain Frontier Network (p001).

Bespoke additive head over the i193 ``ExchangeThenKingDualStreamNetwork``
trunk that consumes a learned candidate-utility table and reduces it
with the differentiable Pareto-antichain operator from
``ideas/research/primitives/codex_01_pareto_antichain_frontier.md``.

Candidate tokens are compiled from the i193 trunk's spatial features via
a small set-query attention pool. CRTK metadata, source labels,
verification flags, and engine evaluations are *not* consumed. The
utility channels are bilinear projections of the candidate tokens
through the trunk-pool context, so the partial-order frontier is a
function of board state only.

The architecture is additive and gated so the i193 baseline is
recovered exactly under the ``zero_delta`` / ``trunk_only`` ablations,
and the primitive can be removed in one place without disturbing the
trunk:

    final_logit = base_logit + primitive_gate * primitive_delta_raw
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.primitives.codex_reply_primitives import (
    BoardTokenAttention,
    pareto_antichain_frontier,
    run_i193_trunk_with_spatial,
)
from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    "scalar_max",          # collapse utilities to a scalar max before reducing
    "single_channel",      # use only the first utility channel (Pareto degenerates)
    "shuffle_channels",    # permute utility channels across candidates
    "uniform_frontier",    # disable frontier softmax (uniform weights over valid)
    "zero_delta",          # primitive_delta == 0
    "disable_gate",        # gate clamped to 1.0
    "trunk_only",          # disable both features and delta
)


class ParetoAntichainFrontierNetwork(nn.Module):
    """p001 — Pareto Antichain Frontier head over the i193 dual-stream trunk."""

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
        token_dim: int = 48,
        utility_channels: int = 6,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        tau_dim: float = 0.08,
        tau_set: float = 0.25,
        eps_margin: float = 0.03,
        beta: float = 0.35,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "ParetoAntichainFrontierNetwork supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError(
                "ParetoAntichainFrontierNetwork requires the simple_18 board tensor"
            )
        if int(num_candidates) < 2:
            raise ValueError("num_candidates must be >= 2 for a non-trivial frontier")
        if int(utility_channels) < 1:
            raise ValueError("utility_channels must be >= 1")
        if str(ablation) not in ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(ALLOWED_ABLATIONS)}"
            )

        self.num_classes = 1
        self.ablation = str(ablation)
        self.num_candidates = int(num_candidates)
        self.token_dim = int(token_dim)
        self.utility_channels = int(utility_channels)
        self.tau_dim = float(tau_dim)
        self.tau_set = float(tau_set)
        self.eps_margin = float(eps_margin)
        self.beta = float(beta)

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

        joint_dim = (
            2 * self.trunk.exchange_encoder.output_dim
            + self.trunk.feature_builder.SUMMARY_DIM
        )
        self.context_proj = nn.Linear(joint_dim, self.token_dim)
        self.utility_head = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, int(head_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity(),
            nn.Linear(int(head_hidden_dim), self.utility_channels),
        )
        self.mask_head = nn.Linear(self.token_dim, 1)
        fusion_dim = self.token_dim + 2 + joint_dim
        self.delta_mlp = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, int(head_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity(),
            nn.Linear(int(head_hidden_dim), 1),
        )
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(joint_dim + 2),
            nn.Linear(joint_dim + 2, int(head_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(head_hidden_dim), 1),
        )
        with torch.no_grad():
            final_gate = self.gate_mlp[-1]
            if isinstance(final_gate, nn.Linear):
                final_gate.bias.fill_(float(gate_init))

    def _build_utilities(
        self,
        cand_tokens: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = self.context_proj(context).unsqueeze(1)
        gated = cand_tokens * torch.tanh(ctx)
        utilities = self.utility_head(gated)
        if self.ablation == "single_channel":
            zeros = torch.zeros_like(utilities)
            zeros[..., 0] = utilities[..., 0]
            utilities = zeros
        elif self.ablation == "scalar_max":
            channelwise_max = utilities.amax(dim=-1, keepdim=True)
            utilities = channelwise_max.expand_as(utilities)
        elif self.ablation == "shuffle_channels":
            if cand_tokens.shape[1] > 1:
                perm = torch.randperm(cand_tokens.shape[1], device=cand_tokens.device)
                utilities = utilities[:, perm]
        mask_logits = self.mask_head(cand_tokens).squeeze(-1)
        soft_mask = torch.sigmoid(mask_logits)
        bool_mask = torch.ones_like(soft_mask, dtype=torch.bool)
        return utilities, bool_mask

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        trunk_feats = run_i193_trunk_with_spatial(self.trunk, board)
        base_logit = trunk_feats.base_logit

        compiler = self.candidate_pool(trunk_feats.spatial)
        utilities, mask = self._build_utilities(compiler.tokens, trunk_feats.pool)

        if self.ablation == "uniform_frontier":
            uniform = torch.zeros_like(utilities)
            reduced = pareto_antichain_frontier(
                uniform,
                mask=mask,
                values=compiler.tokens,
                tau_dim=self.tau_dim,
                tau_set=self.tau_set,
                eps=self.eps_margin,
                beta=0.0,
            )
        else:
            reduced = pareto_antichain_frontier(
                utilities,
                mask=mask,
                values=compiler.tokens,
                tau_dim=self.tau_dim,
                tau_set=self.tau_set,
                eps=self.eps_margin,
                beta=self.beta,
            )
        summary = reduced["summary"]
        width = reduced["width"]
        entropy = reduced["entropy"]

        fusion = torch.cat(
            [summary, width.unsqueeze(-1), entropy.unsqueeze(-1), trunk_feats.pool],
            dim=-1,
        )
        delta_raw = self.delta_mlp(fusion).view(-1)
        gate_input = torch.cat(
            [trunk_feats.pool, width.unsqueeze(-1), entropy.unsqueeze(-1)],
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
        out["pafr_frontier_width"] = width
        out["pafr_frontier_entropy"] = entropy
        out["pafr_max_nondominated_prob"] = reduced["nondominated_prob"].amax(dim=-1)
        out["pafr_summary_norm"] = summary.pow(2).mean(dim=-1)
        out["pafr_utility_mean"] = utilities.mean(dim=(1, 2))
        out["pafr_utility_max"] = utilities.amax(dim=(1, 2))
        return out


def build_pareto_antichain_frontier_network_from_config(
    config: dict[str, Any],
) -> ParetoAntichainFrontierNetwork:
    cfg = dict(config)
    return ParetoAntichainFrontierNetwork(
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
        token_dim=int(cfg.get("token_dim", 48)),
        utility_channels=int(cfg.get("utility_channels", 6)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        tau_dim=float(cfg.get("tau_dim", 0.08)),
        tau_set=float(cfg.get("tau_set", 0.25)),
        eps_margin=float(cfg.get("eps_margin", 0.03)),
        beta=float(cfg.get("beta", 0.35)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )

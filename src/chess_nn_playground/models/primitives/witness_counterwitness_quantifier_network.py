"""Witness-Counterwitness Quantifier Network (p005).

Bespoke additive head over the i193 dual-stream trunk that compiles
witness (candidate) and counterwitness (reply) tokens from the trunk's
spatial features, scores per-candidate claims and per-pair
counterwitnesses, and reduces them with the differentiable nested
adversarial quantifier described in
``ideas/research/primitives/codex_05_witness_counterwitness_quantifier.md``.
The operator produces a board-level "exists witness / forall
counterwitness" value plus margin, counter-envelope, witness and
counter-witness soft assignments. The fusion head consumes those
diagnostics together with the trunk pool feature.

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
    run_i193_trunk_with_spatial,
    witness_counterwitness_quantifier,
)
from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    "max_claim_only",          # disable counter branch: value = max_i claim_i
    "mean_counter_penalty",    # replace forall-soft with mean per-row penalty
    "random_counter_assign",   # permute counter rows across candidates
    "no_counter_branch",       # zero counter scores -> equivalent to claim only
    "zero_delta",
    "disable_gate",
    "trunk_only",
)


class WitnessCounterwitnessQuantifierNetwork(nn.Module):
    """p005 — Witness-Counterwitness Quantifier head over the i193 dual-stream trunk."""

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
        tau_forall: float = 0.20,
        tau_exists: float = 0.20,
        compat_dim: int = 16,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "WitnessCounterwitnessQuantifierNetwork supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError(
                "WitnessCounterwitnessQuantifierNetwork requires the simple_18 board tensor"
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
        self.tau_forall = float(tau_forall)
        self.tau_exists = float(tau_exists)

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
        self.context_proj = nn.Linear(joint_dim, self.token_dim)
        self.claim_head = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, int(head_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(head_hidden_dim), 1),
        )
        self.cand_pair_proj = nn.Linear(self.token_dim, int(compat_dim))
        self.reply_pair_proj = nn.Linear(self.token_dim, int(compat_dim))
        self.counter_head = nn.Sequential(
            nn.LayerNorm(int(compat_dim)),
            nn.Linear(int(compat_dim), int(head_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(head_hidden_dim), 1),
        )

        fusion_dim = 2 * self.token_dim + joint_dim + 4
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

    def _build_scores(
        self,
        cand_tokens: torch.Tensor,
        reply_tokens: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = self.context_proj(context).unsqueeze(1)
        cand_with_ctx = cand_tokens + torch.tanh(ctx)
        claim = self.claim_head(cand_with_ctx).squeeze(-1)

        cand_pair = self.cand_pair_proj(cand_tokens).unsqueeze(2)
        reply_pair = self.reply_pair_proj(reply_tokens).unsqueeze(1)
        pair_feat = cand_pair * reply_pair
        counter = self.counter_head(pair_feat).squeeze(-1)

        if self.ablation == "random_counter_assign" and counter.shape[1] > 1:
            perm = torch.randperm(counter.shape[1], device=counter.device)
            counter = counter[:, perm]
        elif self.ablation == "no_counter_branch":
            counter = torch.zeros_like(counter)
        return claim, counter

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        trunk_feats = run_i193_trunk_with_spatial(self.trunk, board)
        base_logit = trunk_feats.base_logit

        cand = self.candidate_pool(trunk_feats.spatial)
        reply = self.reply_pool(trunk_feats.spatial)
        claim, counter = self._build_scores(cand.tokens, reply.tokens, trunk_feats.pool)

        if self.ablation == "max_claim_only":
            value = claim.amax(dim=-1)
            margin = claim
            counter_envelope = torch.zeros_like(claim)
            witness_weights = torch.softmax(claim, dim=-1)
            counter_weights = torch.zeros_like(counter)
            best_witness = claim.argmax(dim=-1)
            best_counter = torch.zeros_like(best_witness)
            witness_entropy = -(
                witness_weights.clamp_min(1.0e-8) * witness_weights.clamp_min(1.0e-8).log()
            ).sum(dim=-1)
        elif self.ablation == "mean_counter_penalty":
            counter_envelope = counter.mean(dim=-1)
            margin = claim - counter_envelope
            value = margin.amax(dim=-1)
            witness_weights = torch.softmax(margin / max(self.tau_exists, 1.0e-6), dim=-1)
            counter_weights = torch.softmax(counter, dim=-1)
            best_witness = margin.argmax(dim=-1)
            best_counter = counter.argmax(dim=-1).gather(
                1, best_witness.unsqueeze(-1)
            ).squeeze(-1)
            witness_entropy = -(
                witness_weights.clamp_min(1.0e-8) * witness_weights.clamp_min(1.0e-8).log()
            ).sum(dim=-1)
        else:
            reduced = witness_counterwitness_quantifier(
                claim,
                counter,
                tau_forall=self.tau_forall,
                tau_exists=self.tau_exists,
            )
            value = reduced["value"]
            margin = reduced["margin"]
            counter_envelope = reduced["counter_envelope"]
            witness_weights = reduced["witness_weights"]
            counter_weights = reduced["counter_weights"]
            best_witness = reduced["best_witness_index"]
            best_counter = reduced["best_counter_index"]
            witness_entropy = reduced["witness_entropy"]

        attacker_pool = torch.einsum("bk,bkd->bd", witness_weights, cand.tokens)
        counter_pool = torch.einsum(
            "bkr,brd->bd",
            counter_weights,
            reply.tokens,
        )
        diagnostics = torch.stack(
            [value, margin.amax(dim=-1), counter_envelope.amax(dim=-1), witness_entropy],
            dim=-1,
        )
        fusion = torch.cat([attacker_pool, counter_pool, trunk_feats.pool, diagnostics], dim=-1)
        delta_raw = self.delta_mlp(fusion).view(-1)
        gate_input = torch.cat(
            [
                trunk_feats.pool,
                value.unsqueeze(-1),
                margin.amax(dim=-1).unsqueeze(-1),
                witness_entropy.unsqueeze(-1),
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
        out["wcq_value"] = value
        out["wcq_max_margin"] = margin.amax(dim=-1)
        out["wcq_min_margin"] = margin.amin(dim=-1)
        out["wcq_counter_envelope_max"] = counter_envelope.amax(dim=-1)
        out["wcq_witness_entropy"] = witness_entropy
        out["wcq_best_witness_index"] = best_witness.to(dtype=base_logit.dtype)
        out["wcq_best_counter_index"] = best_counter.to(dtype=base_logit.dtype)
        out["wcq_claim_max"] = claim.amax(dim=-1)
        return out


def build_witness_counterwitness_quantifier_network_from_config(
    config: dict[str, Any],
) -> WitnessCounterwitnessQuantifierNetwork:
    cfg = dict(config)
    return WitnessCounterwitnessQuantifierNetwork(
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
        tau_forall=float(cfg.get("tau_forall", 0.20)),
        tau_exists=float(cfg.get("tau_exists", 0.20)),
        compat_dim=int(cfg.get("compat_dim", 16)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )

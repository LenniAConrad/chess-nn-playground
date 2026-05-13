"""Tail Copula Concordance Network (p004).

Bespoke additive head over the i193 dual-stream trunk that consumes
per-square evidence channels (projected from the trunk's spatial
features) and reduces them with the differentiable rank-copula
upper-tail concordance operator described in
``ideas/research/primitives/codex_04_tail_copula_concordance.md``. The
operator measures whether independent evidence channels become extreme
on the same squares (the chess motivation: a tactical site lighting up
across several orthogonal signals).

The model is additive and gated: ``trunk_only`` / ``zero_delta``
recover the i193 baseline exactly. No CRTK metadata, source labels,
engine evaluations, or report-only metadata are consumed.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.primitives.codex_reply_primitives import (
    run_i193_trunk_with_spatial,
    tail_copula_concordance,
)
from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    "square_shuffle",         # shuffle squares per channel (kills site alignment)
    "channel_shuffle",        # permute channels (kills cross-channel structure)
    "rank_quantile_only",     # bypass tail mask: use channel ranks only (i095-style control)
    "single_channel",         # use only channel 0
    "zero_delta",
    "disable_gate",
    "trunk_only",
)


class TailCopulaConcordanceNetwork(nn.Module):
    """p004 — Tail Copula Concordance head over the i193 dual-stream trunk."""

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
        evidence_channels: int = 6,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        quantile: float = 0.80,
        tau_rank: float = 0.35,
        tau_tail: float = 0.06,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "TailCopulaConcordanceNetwork supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError(
                "TailCopulaConcordanceNetwork requires the simple_18 board tensor"
            )
        if int(evidence_channels) < 2:
            raise ValueError(
                "evidence_channels must be >= 2; tail-copula concordance is between channels"
            )
        if str(ablation) not in ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(ALLOWED_ABLATIONS)}"
            )
        if not (0.0 < float(quantile) < 1.0):
            raise ValueError("quantile must lie strictly in (0, 1)")

        self.num_classes = 1
        self.ablation = str(ablation)
        self.evidence_channels = int(evidence_channels)
        self.quantile = float(quantile)
        self.tau_rank = float(tau_rank)
        self.tau_tail = float(tau_tail)

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
        self.evidence_proj = nn.Conv2d(spatial_channels, self.evidence_channels, kernel_size=1)

        joint_dim = (
            2 * self.trunk.exchange_encoder.output_dim
            + self.trunk.feature_builder.SUMMARY_DIM
        )
        flat_matrix_dim = self.evidence_channels * self.evidence_channels
        fusion_dim = (
            flat_matrix_dim
            + self.evidence_channels
            + joint_dim
            + 2
        )
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

    def _evidence_from_spatial(self, spatial: torch.Tensor) -> torch.Tensor:
        per_square = self.evidence_proj(spatial)
        batch, channels, height, width = per_square.shape
        evidence = per_square.view(batch, channels, height * width).transpose(1, 2)
        if self.ablation == "square_shuffle" and evidence.shape[1] > 1:
            perm = torch.randperm(evidence.shape[1], device=evidence.device)
            evidence = evidence[:, perm]
        elif self.ablation == "channel_shuffle" and evidence.shape[2] > 1:
            perm = torch.randperm(evidence.shape[2], device=evidence.device)
            evidence = evidence[:, :, perm]
        elif self.ablation == "single_channel":
            zeros = torch.zeros_like(evidence)
            zeros[..., 0] = evidence[..., 0]
            evidence = zeros
        return evidence

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        trunk_feats = run_i193_trunk_with_spatial(self.trunk, board)
        base_logit = trunk_feats.base_logit

        evidence = self._evidence_from_spatial(trunk_feats.spatial)
        reduced = tail_copula_concordance(
            evidence,
            quantile=self.quantile,
            tau_rank=self.tau_rank,
            tau_tail=self.tau_tail,
        )
        concordance = reduced["concordance"]
        if self.ablation == "rank_quantile_only":
            concordance = torch.eye(
                concordance.shape[-1],
                device=concordance.device,
                dtype=concordance.dtype,
            ).unsqueeze(0).expand_as(concordance)
        tail_mean = reduced["tail_mean"]
        tail_max = reduced["tail_max"]
        channel_mass = reduced["channel_tail_mass"]

        batch = concordance.shape[0]
        matrix_flat = concordance.view(batch, -1)
        fusion = torch.cat(
            [
                matrix_flat,
                channel_mass,
                trunk_feats.pool,
                tail_mean.unsqueeze(-1),
                tail_max.unsqueeze(-1),
            ],
            dim=-1,
        )
        delta_raw = self.delta_mlp(fusion).view(-1)
        gate_input = torch.cat(
            [trunk_feats.pool, tail_mean.unsqueeze(-1), tail_max.unsqueeze(-1)],
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
        out["tcc_tail_mean"] = tail_mean
        out["tcc_tail_max"] = tail_max
        out["tcc_channel_mass_mean"] = channel_mass.mean(dim=-1)
        out["tcc_channel_mass_max"] = channel_mass.amax(dim=-1)
        out["tcc_concordance_trace"] = torch.diagonal(concordance, dim1=1, dim2=2).mean(dim=-1)
        out["tcc_site_mass_max"] = reduced["tail_site_mass"].amax(dim=-1)
        return out


def build_tail_copula_concordance_network_from_config(
    config: dict[str, Any],
) -> TailCopulaConcordanceNetwork:
    cfg = dict(config)
    return TailCopulaConcordanceNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim", None)),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        evidence_channels=int(cfg.get("evidence_channels", 6)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        quantile=float(cfg.get("quantile", 0.80)),
        tau_rank=float(cfg.get("tau_rank", 0.35)),
        tau_tail=float(cfg.get("tau_tail", 0.06)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )

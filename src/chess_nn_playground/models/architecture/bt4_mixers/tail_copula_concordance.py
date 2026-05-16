"""Tail-copula-concordance spatial mixer (TCC).

Embodies the p004 primitive's core operator: rank-copula upper-tail
concordance across per-square evidence channels.

This primitive is the natural fit for a spatial mixer -- it already
operates over the 64 squares as its site index ``N``. The channels are
projected to ``C`` learned evidence channels; each is converted to soft
uniform ranks over the 64 squares (the Sklar copula representation),
then to a soft upper-tail membership ``m_{n,c} = sigmoid((u-q)/tau)``.
The directional tail-dependence ``lambda_{c->d}`` and its symmetric
concordance ``Lambda_{c,d} = sqrt(lambda_{c->d} lambda_{d->c})`` measure
how often channels spike on the *same* square. The mixer uses ``Lambda``
to remix evidence channels (concordant channels reinforce) and the
per-site tail mass to spatially gate the result -- so the spatial mix is
driven by cross-site / cross-channel tail alignment, not marginal pooling.

Shape contract: ``(B, C, 8, 8) -> (B, C, 8, 8)``, channel-agnostic.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer

_LOG_EPS = 1.0e-8


class TailCopulaConcordanceMixer(nn.Module):
    def __init__(
        self,
        channels: int,
        evidence_channels: int = 8,
        quantile: float = 0.75,
        tau_rank: float = 0.35,
        tau_tail: float = 0.06,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.evidence_channels = int(evidence_channels)
        self.quantile = float(quantile)
        self.tau_rank = float(tau_rank)
        self.tau_tail = float(tau_tail)

        self.norm = nn.LayerNorm(channels)
        # Project board channels -> C evidence channels (per square).
        self.evidence_proj = nn.Linear(channels, self.evidence_channels)
        # Per-square value field that the concordance-remixed tail field gates.
        self.value_proj = nn.Linear(channels, channels)
        # Map the concordance-remixed evidence field back to channels.
        self.out_proj = nn.Linear(self.evidence_channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        seq = self.norm(x.flatten(2).transpose(1, 2))  # (B, 64, C)

        evidence = self.evidence_proj(seq)  # (B, 64, C_e)

        # --- Soft uniform ranks over the 64 squares (Sklar copula repr).
        diff = evidence.unsqueeze(2) - evidence.unsqueeze(1)  # (B,64,64,C_e)
        soft_le = torch.sigmoid(diff / max(self.tau_rank, 1.0e-6))
        ranks = soft_le.sum(dim=2) / n  # (B, 64, C_e) in (0, 1]

        # --- Soft upper-tail membership.
        tail = torch.sigmoid((ranks - self.quantile) / max(self.tau_tail, 1.0e-6))  # (B,64,C_e)

        # --- Directional tail dependence + symmetric concordance.
        numerator = torch.einsum("bnc,bnd->bcd", tail, tail)  # (B,C_e,C_e)
        denom = tail.sum(dim=1).clamp_min(_LOG_EPS)            # (B,C_e)
        directional = numerator / denom.unsqueeze(-1)          # lambda_{c->d}
        concordance = torch.sqrt(
            (directional * directional.transpose(1, 2)).clamp_min(0.0)
        )  # (B, C_e, C_e) symmetric

        # --- Remix the tail field through the concordance matrix: concordant
        # channels reinforce each other. Row-normalize for stable scale.
        conc_norm = concordance / concordance.sum(dim=-1, keepdim=True).clamp_min(_LOG_EPS)
        remixed = torch.einsum("bnc,bcd->bnd", tail, conc_norm)  # (B, 64, C_e)

        # Per-site tail mass -> soft spatial hotspot gate.
        site_mass = tail.mean(dim=-1, keepdim=True)  # (B, 64, 1)
        evidence_field = self.out_proj(remixed) * site_mass  # (B, 64, C)

        out = self.value_proj(seq) * torch.sigmoid(evidence_field)
        return out.transpose(1, 2).reshape(b, c, h, w)


@register_mixer("tail_copula_concordance")
def build(channels: int, **_: object) -> nn.Module:
    return TailCopulaConcordanceMixer(channels=channels)

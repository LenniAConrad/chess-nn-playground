"""Rule-aware tactical (terminal-state) spatial mixer (i248 / TSDP primitive).

The TSDP primitive's core idea: chess *rule-exact* terminal-state indicators
(is_checkmate, is_stalemate, is_check, is_promotion, is_capture,
is_castling) computed over the legal-move set, aggregated into an 11-d
forcing-feature vector, and fused into the logit through a learned gate:

    final_logit = trunk(x) + gate(x) * delta(rule_features)

HONEST COMPROMISE
-----------------
This primitive is the *worst fit* of the batch for a spatial-mixer mould.
Its essential machinery -- reconstruct a ``chess.Board``, enumerate legal
moves, classify each resulting position with exact rules -- is (a)
non-differentiable, (b) needs python-chess (a disallowed import), and (c)
needs the simple_18 channel semantics (castling planes, side-to-move) that a
swappable mixer operating on an arbitrary (B, C, 8, 8) activation tensor
does not have. A faithful transcription is therefore impossible inside the
mixer contract.

The adaptation keeps TSDP's *structural skeleton* -- a differentiable
"forcing-geometry" probe whose output is fused through a learned per-square
gate -- while being honest that the rule-exactness is replaced by a learned
surrogate:

  1. Ray scans. Eight directional depthwise convolutions (the 4 rook +
     4 bishop directions) act as one-step "move geometry" probes: they
     measure, per square and per channel, how features propagate along the
     lines that checks / captures / promotions travel along. This is the
     learned stand-in for "enumerate the moves that leave this square".
  2. Forcing features. A small pointwise stack turns the stacked
     directional responses into a per-square ``forcing`` field -- the
     differentiable analogue of (check_count, capture_count,
     forcing_density, ...).
  3. Gated fusion. ``delta = MLP(forcing)`` and ``gate = sigmoid(MLP(...))``;
     the mixer returns ``base_mix + gate * delta`` exactly mirroring the
     primitive's additive-gated fusion ``base + gate * delta``.

So: the additive-gated forcing-feature *architecture* of TSDP is faithfully
reproduced; the rule-exact terminal-state oracle is replaced by a learned
ray-geometry surrogate because exact chess rules cannot be evaluated
differentiably from an unlabelled (B, C, 8, 8) tensor.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer

# 8 move directions: 4 rook (N, S, E, W) + 4 bishop (NE, NW, SE, SW).
_DIRECTIONS = ((-1, 0), (1, 0), (0, 1), (0, -1), (-1, 1), (-1, -1), (1, 1), (1, -1))


class RuleAwareTacticalMixer(nn.Module):
    def __init__(self, channels: int, hidden: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.channels = channels
        self.num_dirs = len(_DIRECTIONS)

        # One depthwise 3x3 conv per move direction; init each kernel so it
        # primarily reads the single neighbour cell in its direction.
        self.dir_convs = nn.ModuleList(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
            for _ in range(self.num_dirs)
        )
        with torch.no_grad():
            for conv, (dr, dc) in zip(self.dir_convs, _DIRECTIONS):
                conv.weight.zero_()
                # kernel index (1+dr, 1+dc) is the neighbour in that direction.
                conv.weight[:, :, 1 + dr, 1 + dc] = 1.0
                conv.weight += 0.01 * torch.randn_like(conv.weight)

        # Base spatial mix (the "trunk" analogue inside the mixer).
        self.base_mix = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        # Forcing-feature stack over the 8 stacked directional responses.
        self.forcing = nn.Sequential(
            nn.Conv2d(self.num_dirs * channels, hidden, kernel_size=1),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.GELU(),
        )
        # delta and gate heads -- the additive-gated TSDP fusion.
        self.delta_head = nn.Conv2d(hidden, channels, kernel_size=1)
        self.gate_head = nn.Conv2d(hidden, channels, kernel_size=1)
        with torch.no_grad():
            self.gate_head.bias.fill_(-2.0)  # start as a small additive correction
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_mix(x)

        dir_responses = torch.cat([conv(x) for conv in self.dir_convs], dim=1)
        forcing = self.forcing(dir_responses)

        delta = self.delta_head(forcing)
        gate = torch.sigmoid(self.gate_head(forcing))

        out = base + gate * delta
        return self.dropout(out)


@register_mixer("rule_aware_tactical_head")
def build(channels: int, hidden: int = 64, dropout: float = 0.1, **_: object) -> nn.Module:
    return RuleAwareTacticalMixer(channels=channels, hidden=hidden, dropout=dropout)

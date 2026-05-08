"""Tempo-Alignment Gate Network for idea i183.

Faithful implementation of the markdown thesis under
``ideas/i183_tempo_alignment_gate_network/``. The packet thesis is
that many near-puzzles look tactical for the wrong side or require a
tempo that the side to move does not have. The model should
explicitly *gate* static tactical danger by side-to-move tempo
alignment, instead of letting an undirected CNN absorb the danger
signal regardless of who is to move.

Forward pass:

* A compact convolutional trunk produces per-square features
  ``H ∈ R^{B×C×8×8}`` from the 18-plane ``simple_18`` board.
* A 1x1 ``danger_head`` produces an undirected per-square static
  tactical danger field ``d(s) ∈ R``. This is the "tactical danger
  for somebody" signal that the markdown calls "tactical-looking".
* A 1x1 ``side_head`` produces a signed per-square attacker-side
  field ``a(s) ∈ R``: positive means the local tactic is white's,
  negative means black's. The piece occupancy planes 0-5 (white) and
  6-11 (black) are explicitly summed and concatenated with the
  trunk features so the side head can ground itself on which color
  is doing the attacking at each square.
* A scalar tempo signal ``stm ∈ {-1, +1}`` is read from plane 12
  (``white_to_move``); together with a global pool of trunk features
  and the white/black material totals it feeds a small MLP that
  returns a ``tempo_gate ∈ (0, 1)`` -- how much the side-to-move
  alignment should gate the static danger.
* Per-square *alignment* is the sigmoid of ``γ · stm · a(s) + β``:
  ``alignment(s) ≈ 1`` when the local attacker side matches the side
  to move (we have the tempo) and ``≈ 0`` when it does not (the
  position is tactical-looking for the wrong side).
* The gated danger at each square is
  ``g(s) = tempo_gate * alignment(s) * relu(d(s))`` -- a *multi-
  plicative* gate so the score collapses if either the alignment or
  the tempo gate is small. Pooled diagnostics:
  - ``own_pressure``  = mean over squares of ``alignment * relu(d)``.
  - ``opp_pressure``  = mean over squares of ``(1-alignment) * relu(d)``.
  - ``alignment_gap`` = ``own_pressure - opp_pressure``.
  - ``gated_pressure`` = mean over squares of ``g(s)``.
* A counterfactual stream re-runs the head with the side-to-move
  plane flipped (and the trunk re-evaluated, so the gate-and-align
  interaction is faithful, not a no-op). The contrast between the
  real and flipped streams is fed into the puzzle head -- a real
  tempo-aligned puzzle should look much weaker after the flip.
* The puzzle logit reads from a ``LayerNorm → Linear → GELU →
  Dropout → Linear`` head over the concatenated pooled and contrast
  features.

The packet's required ablations are wired through an ``ablation``
flag:

* ``"none"`` -- main multiplicative tempo-alignment gate.
* ``"no_tempo_gate"`` -- ``tempo_gate`` is forced to 1, so the
  alignment can no longer be tuned by the side-to-move signal.
* ``"no_alignment"`` -- per-square alignment is forced to 0.5, so
  the gate cannot tell which side the local danger is for. This is
  the "undirected tactical CNN" baseline implied by the markdown.
* ``"additive_gate"`` -- replace
  ``tempo_gate * alignment * relu(d)`` with
  ``tempo_gate + alignment + relu(d)``, killing the multiplicative
  conjunction the markdown calls out as the gate's defining
  property.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import (
    BoardConvStem,
    BoardTensorSpec,
    require_board_tensor,
)


_VALID_ABLATIONS = {
    "none",
    "no_tempo_gate",
    "no_alignment",
    "additive_gate",
}


class TempoAlignmentGateNetwork(nn.Module):
    """Tempo-alignment gated tactical-danger network.

    Forward returns a dict with at least:

      - ``logits``: ``(B,)`` puzzle logit fed to BCE-with-logits
        (``(B, num_classes)`` if ``num_classes > 1``, with the puzzle
        scalar written into the last column of a zero-padded tensor).
      - ``prob``: ``sigmoid(logits)`` when ``num_classes == 1``.
      - ``danger_field``: ``(B, 8, 8)`` undirected static tactical
        danger ``relu(d(s))``.
      - ``side_field``: ``(B, 8, 8)`` signed attacker-side logit
        ``a(s)`` -- positive = white tactic, negative = black tactic.
      - ``alignment_field``: ``(B, 8, 8)`` per-square
        ``sigmoid(γ · stm · a(s) + β)``.
      - ``own_pressure``, ``opp_pressure``, ``alignment_gap``,
        ``gated_pressure``: ``(B,)`` scalar diagnostics.
      - ``tempo_gate``: ``(B,)`` value in ``(0, 1)``.
      - ``own_pressure_flipped``, ``gated_pressure_flipped``,
        ``tempo_gate_flipped``, ``flip_contrast``: ``(B,)`` from the
        side-to-move-flipped counterfactual stream.
      - ``trunk_features``: ``(B, channels, 8, 8)``.
      - ``ablation_active``, ``uses_tempo_gate``,
        ``uses_alignment``, ``uses_multiplicative_gate``:
        ``(B,)`` flags exposing the running ablation.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        depth: int = 2,
        hidden_dim: int = 96,
        tempo_dim: int = 32,
        align_scale_init: float = 4.0,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        ablation: str = "none",
        side_to_move_plane_index: int = 12,
        white_piece_plane_start: int = 0,
        white_piece_plane_end: int = 6,
        black_piece_plane_start: int = 6,
        black_piece_plane_end: int = 12,
    ) -> None:
        super().__init__()
        if depth < 1 or channels < 1 or num_classes < 1:
            raise ValueError("depth, channels, num_classes must be >= 1")
        if hidden_dim < 1 or tempo_dim < 1:
            raise ValueError("hidden_dim, tempo_dim must be >= 1")
        if not 0 <= side_to_move_plane_index < input_channels:
            raise ValueError("side_to_move_plane_index must be in [0, input_channels)")
        if not 0 <= white_piece_plane_start < white_piece_plane_end <= input_channels:
            raise ValueError("invalid white piece plane range")
        if not 0 <= black_piece_plane_start < black_piece_plane_end <= input_channels:
            raise ValueError("invalid black piece plane range")
        if ablation not in _VALID_ABLATIONS:
            raise ValueError(
                f"ablation must be one of {sorted(_VALID_ABLATIONS)}, got {ablation!r}"
            )

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.depth = int(depth)
        self.hidden_dim = int(hidden_dim)
        self.tempo_dim = int(tempo_dim)
        self.dropout = float(dropout)
        self.use_batchnorm = bool(use_batchnorm)
        self.ablation = str(ablation)
        self.side_to_move_plane_index = int(side_to_move_plane_index)
        self.white_piece_plane_start = int(white_piece_plane_start)
        self.white_piece_plane_end = int(white_piece_plane_end)
        self.black_piece_plane_start = int(black_piece_plane_start)
        self.black_piece_plane_end = int(black_piece_plane_end)

        self.uses_tempo_gate = self.ablation != "no_tempo_gate"
        self.uses_alignment = self.ablation != "no_alignment"
        self.uses_multiplicative_gate = self.ablation != "additive_gate"

        self.trunk = BoardConvStem(
            input_channels=self.input_channels,
            channels=self.channels,
            depth=self.depth,
            use_batchnorm=self.use_batchnorm,
        )

        # Per-square static tactical danger -- undirected ("for
        # somebody"). Output is one channel; ReLU is applied later.
        self.danger_head = nn.Conv2d(self.channels, 1, kernel_size=1)

        # Per-square signed attacker-side logit. The 1x1 conv reads
        # the trunk plus a 2-channel white/black occupancy summary so
        # the side head can ground itself on which color is local.
        self.side_head = nn.Conv2d(self.channels + 2, 1, kernel_size=1)

        # Learnable alignment scale and bias: alignment(s) =
        # sigmoid(γ * stm * a(s) + β). γ initialized > 0 so the
        # alignment can saturate when stm and a(s) agree.
        self.align_scale = nn.Parameter(torch.tensor(float(align_scale_init)))
        self.align_bias = nn.Parameter(torch.tensor(0.0))

        # Tempo gate from a global summary: pooled trunk features +
        # white/black material totals + scalar stm.
        tempo_input_dim = self.channels + 2 + 1
        self.tempo_mlp = nn.Sequential(
            nn.Linear(tempo_input_dim, self.tempo_dim),
            nn.GELU(),
            nn.Linear(self.tempo_dim, 1),
        )

        # Final head consumes:
        #   own_pressure, opp_pressure, alignment_gap,
        #   gated_pressure, tempo_gate, mean_danger, max_danger,
        #   own_pressure_flipped, gated_pressure_flipped,
        #   tempo_gate_flipped, flip_contrast (= gated -
        #     gated_flipped)
        head_input_dim = 11
        self.head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(self.hidden_dim, 1),
        )

    # ----- helpers -------------------------------------------------

    def _occupancy_summary(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(B, 2, 8, 8)`` white/black occupancy summary."""
        white = x[:, self.white_piece_plane_start : self.white_piece_plane_end].sum(
            dim=1, keepdim=True
        )
        black = x[:, self.black_piece_plane_start : self.black_piece_plane_end].sum(
            dim=1, keepdim=True
        )
        return torch.cat([white, black], dim=1)

    def _stm_signed(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(B,)`` ±1 scalar stm from plane 12.

        The plane is constant across the 8x8 board, so we read the
        top-left value. white_to_move=1 ⇒ +1, black_to_move=0 ⇒ -1.
        """
        plane = x[:, self.side_to_move_plane_index, 0, 0]
        return 2.0 * plane - 1.0

    def _compute_stream(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        feats = self.trunk(x)  # (B, C, 8, 8)
        occupancy = self._occupancy_summary(x)  # (B, 2, 8, 8)
        side_input = torch.cat([feats, occupancy], dim=1)
        danger_logit = self.danger_head(feats).squeeze(1)  # (B, 8, 8)
        danger = F.relu(danger_logit)  # undirected static danger field
        side_logit = self.side_head(side_input).squeeze(1)  # (B, 8, 8)

        stm = self._stm_signed(x)  # (B,)
        if self.uses_alignment:
            alignment = torch.sigmoid(
                self.align_scale * stm.view(-1, 1, 1) * side_logit + self.align_bias
            )
        else:
            alignment = torch.full_like(side_logit, 0.5)

        # Tempo gate from global pooled features + material totals.
        pooled = feats.mean(dim=(2, 3))  # (B, C)
        material = occupancy.sum(dim=(2, 3))  # (B, 2)
        tempo_input = torch.cat([pooled, material, stm.unsqueeze(-1)], dim=-1)
        tempo_gate_raw = torch.sigmoid(self.tempo_mlp(tempo_input).squeeze(-1))
        if self.uses_tempo_gate:
            tempo_gate = tempo_gate_raw
        else:
            tempo_gate = torch.ones_like(tempo_gate_raw)

        if self.uses_multiplicative_gate:
            gated = tempo_gate.view(-1, 1, 1) * alignment * danger
        else:
            gated = tempo_gate.view(-1, 1, 1) + alignment + danger

        own_pressure = (alignment * danger).flatten(1).mean(dim=-1)
        opp_pressure = ((1.0 - alignment) * danger).flatten(1).mean(dim=-1)
        alignment_gap = own_pressure - opp_pressure
        gated_pressure = gated.flatten(1).mean(dim=-1)
        mean_danger = danger.flatten(1).mean(dim=-1)
        max_danger = danger.flatten(1).amax(dim=-1)

        return {
            "trunk_features": feats,
            "danger_field": danger,
            "side_field": side_logit,
            "alignment_field": alignment,
            "own_pressure": own_pressure,
            "opp_pressure": opp_pressure,
            "alignment_gap": alignment_gap,
            "gated_pressure": gated_pressure,
            "tempo_gate": tempo_gate,
            "tempo_gate_raw": tempo_gate_raw,
            "mean_danger": mean_danger,
            "max_danger": max_danger,
            "stm": stm,
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        batch_size = x.shape[0]

        own = self._compute_stream(x)

        # Counterfactual: flip the side-to-move plane and re-run.
        # The gate-and-align interaction depends on stm via both the
        # tempo MLP and the alignment scale, so the contrast is a
        # faithful intervention rather than a no-op.
        x_flipped = x.clone()
        x_flipped[:, self.side_to_move_plane_index] = (
            1.0 - x_flipped[:, self.side_to_move_plane_index]
        )
        flipped = self._compute_stream(x_flipped)

        flip_contrast = own["gated_pressure"] - flipped["gated_pressure"]

        head_input = torch.stack(
            [
                own["own_pressure"],
                own["opp_pressure"],
                own["alignment_gap"],
                own["gated_pressure"],
                own["tempo_gate"],
                own["mean_danger"],
                own["max_danger"],
                flipped["own_pressure"],
                flipped["gated_pressure"],
                flipped["tempo_gate"],
                flip_contrast,
            ],
            dim=-1,
        )
        scalar_logit = self.head(head_input).squeeze(-1)  # (B,)

        if self.num_classes == 1:
            logits = scalar_logit
        else:
            logits = torch.zeros(
                batch_size,
                self.num_classes,
                device=scalar_logit.device,
                dtype=scalar_logit.dtype,
            )
            logits[:, -1] = scalar_logit

        with torch.no_grad():
            ones = torch.ones(
                batch_size, device=scalar_logit.device, dtype=scalar_logit.dtype
            )
            ablation_active = ones * (0.0 if self.ablation == "none" else 1.0)
            uses_tempo = ones * (1.0 if self.uses_tempo_gate else 0.0)
            uses_align = ones * (1.0 if self.uses_alignment else 0.0)
            uses_mult = ones * (1.0 if self.uses_multiplicative_gate else 0.0)

        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "danger_field": own["danger_field"],
            "side_field": own["side_field"],
            "alignment_field": own["alignment_field"],
            "own_pressure": own["own_pressure"],
            "opp_pressure": own["opp_pressure"],
            "alignment_gap": own["alignment_gap"],
            "gated_pressure": own["gated_pressure"],
            "tempo_gate": own["tempo_gate"],
            "mean_danger": own["mean_danger"],
            "max_danger": own["max_danger"],
            "own_pressure_flipped": flipped["own_pressure"],
            "gated_pressure_flipped": flipped["gated_pressure"],
            "tempo_gate_flipped": flipped["tempo_gate"],
            "flip_contrast": flip_contrast,
            "trunk_features": own["trunk_features"],
            "ablation_active": ablation_active,
            "uses_tempo_gate": uses_tempo,
            "uses_alignment": uses_align,
            "uses_multiplicative_gate": uses_mult,
        }
        if self.num_classes == 1:
            output["prob"] = torch.sigmoid(logits)
        return output


def build_tempo_alignment_gate_network_from_config(
    config: dict[str, Any],
) -> TempoAlignmentGateNetwork:
    cfg = dict(config)
    cfg.pop("name", None)
    cfg.pop("mechanism_family", None)
    cfg.pop("packet_profile", None)
    return TempoAlignmentGateNetwork(
        input_channels=int(cfg.pop("input_channels", 18)),
        num_classes=int(cfg.pop("num_classes", 1)),
        channels=int(cfg.pop("channels", 64)),
        depth=int(cfg.pop("depth", 2)),
        hidden_dim=int(cfg.pop("hidden_dim", 96)),
        tempo_dim=int(cfg.pop("tempo_dim", 32)),
        align_scale_init=float(cfg.pop("align_scale_init", 4.0)),
        dropout=float(cfg.pop("dropout", 0.1)),
        use_batchnorm=bool(cfg.pop("use_batchnorm", True)),
        ablation=str(cfg.pop("ablation", "none")),
        side_to_move_plane_index=int(cfg.pop("side_to_move_plane_index", 12)),
        white_piece_plane_start=int(cfg.pop("white_piece_plane_start", 0)),
        white_piece_plane_end=int(cfg.pop("white_piece_plane_end", 6)),
        black_piece_plane_start=int(cfg.pop("black_piece_plane_start", 6)),
        black_piece_plane_end=int(cfg.pop("black_piece_plane_end", 12)),
    )

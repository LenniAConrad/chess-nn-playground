"""Phase-Transition Pressure Network for idea i180.

Faithful implementation of the markdown thesis under
``ideas/i180_phase_transition_pressure_network/``. The model classifies
a position by asking whether the board sits near a *threshold* where
small increases in pressure, line opening, or defender loss would
cause a tactical collapse. The signal is the *transition curve*
across thresholds rather than the magnitude of pressure itself.

Concretely the forward pass is:

    feats = trunk(board)
    pressures[B, F, 8, 8] = pressure_head(feats)          # F learned fields
    field_tau[B, F, T, 8, 8] = sigmoid((pressures - tau_t) / temperature)
    summaries_s[B, F, T] = differentiable_summary_s(field_tau)
    critical_curves[B, F, T-1, S] = summaries_s[..., 1:] - summaries_s[..., :-1]
    puzzle_logit = mlp([critical_curves, anchor_summary_at_central_threshold])

The packet calls out five pressure fields:

    attack_pressure
    defense_pressure
    escape_pressure
    line_block_pressure
    target_value_pressure

and five differentiable summaries per (field, threshold):

    mass
    connected king-zone mass
    largest soft component approximation
    boundary length
    pressure surplus around king/queen/rook

Required ablations:

* ``"none"`` -- main model.
* ``"single_threshold"`` -- collapse the threshold sweep to one
  threshold so the readout only sees magnitude, not transition.
* ``"pressure_mean_only"`` -- replace the threshold sweep with a plain
  mean pressure per field so criticality is removed entirely.
* ``"no_king_zone_features"`` -- drop the king-zone-mass and
  king/queen/rook-surplus features so the readout cannot rely on
  king-local transitions.
"""

from __future__ import annotations

import math
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
    "single_threshold",
    "pressure_mean_only",
    "no_king_zone_features",
}


_FIELD_NAMES = (
    "attack_pressure",
    "defense_pressure",
    "escape_pressure",
    "line_block_pressure",
    "target_value_pressure",
)
_NUM_FIELDS = len(_FIELD_NAMES)

_SUMMARY_NAMES = (
    "mass",
    "king_zone_mass",
    "largest_component",
    "boundary_length",
    "king_surplus",
    "queen_surplus",
    "rook_surplus",
)
_NUM_SUMMARIES = len(_SUMMARY_NAMES)
_KING_ZONE_SUMMARY_INDICES = (1, 4, 5, 6)  # king_zone_mass, king/queen/rook surplus

# simple_18 piece-plane offsets: (white, black) per piece type.
_PIECE_PLANE_PAIRS = {
    "king": (5, 11),
    "queen": (4, 10),
    "rook": (3, 9),
}
_SIDE_TO_MOVE_PLANE = 12


def _neighbor_kernel(kernel_size: int = 3) -> torch.Tensor:
    return torch.ones(1, 1, kernel_size, kernel_size, dtype=torch.float32)


class PhaseTransitionPressureNetwork(nn.Module):
    """Threshold-sweep pressure network for the puzzle_binary head.

    Forward returns a dict with at least:

      - ``logits``: ``(B,)`` puzzle logit fed to the BCE-with-logits trainer
        (``(B, num_classes)`` if ``num_classes > 1``).
      - ``prob``: ``sigmoid(logits)`` when ``num_classes == 1``.
      - ``pressure_fields``: ``(B, F, 8, 8)`` raw learned pressure fields.
      - ``thresholds``: ``(T_eff,)`` effective threshold values.
      - ``temperature``: scalar effective temperature.
      - ``field_tau``: ``(B, F, T_eff, 8, 8)`` sigmoid-thresholded fields.
      - ``summaries``: ``(B, F, T_eff, S_eff)`` differentiable summaries.
      - ``critical_curves``: ``(B, F, max(T_eff-1, 1), S_eff)`` first
        differences across thresholds (zeros when ``T_eff == 1``).
      - ``mass_curve``, ``king_zone_mass_curve``,
        ``largest_component_curve``, ``boundary_length_curve``: each
        ``(B, F, T_eff)`` per-summary mass profile.
      - ``critical_pressure_score``: ``(B,)`` total |first difference|
        energy across all (field, summary, threshold) entries.
      - ``readout_features``: ``(B, R)`` flattened readout input.
      - ``trunk_features``: ``(B, channels, 8, 8)``.
      - ``pressure_mean``: ``(B, F)`` mean pressure per field.
      - ``ablation_active``, ``uses_threshold_sweep``,
        ``uses_pressure_curve``, ``uses_king_zone_features``,
        ``num_thresholds_effective``, ``num_summaries_effective``:
        ``(B,)`` flags exposing the running ablation.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        depth: int = 2,
        thresholds: int = 8,
        temperature: float = 0.2,
        hidden_dim: int = 96,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        learn_thresholds: bool = True,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if depth < 1 or channels < 1 or num_classes < 1:
            raise ValueError("depth, channels, num_classes must be >= 1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if thresholds < 1:
            raise ValueError("thresholds must be >= 1")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        if ablation not in _VALID_ABLATIONS:
            raise ValueError(
                f"ablation must be one of {sorted(_VALID_ABLATIONS)}, got {ablation!r}"
            )
        if input_channels <= max(_PIECE_PLANE_PAIRS["king"]):
            raise ValueError(
                f"input_channels={input_channels} must include all simple_18 piece planes"
            )

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.depth = int(depth)
        self.num_thresholds = int(thresholds)
        self.base_temperature = float(temperature)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.use_batchnorm = bool(use_batchnorm)
        self.learn_thresholds = bool(learn_thresholds)
        self.ablation = str(ablation)

        # Effective sweep parameters under the running ablation.
        if self.ablation == "single_threshold":
            self.num_thresholds_effective = 1
        elif self.ablation == "pressure_mean_only":
            self.num_thresholds_effective = 0
        else:
            self.num_thresholds_effective = self.num_thresholds
        self.uses_threshold_sweep = self.ablation != "pressure_mean_only"
        self.uses_pressure_curve = self.uses_threshold_sweep and self.num_thresholds_effective >= 2
        self.uses_king_zone_features = self.ablation != "no_king_zone_features"
        if self.uses_king_zone_features:
            self.summary_indices = tuple(range(_NUM_SUMMARIES))
        else:
            self.summary_indices = tuple(
                i for i in range(_NUM_SUMMARIES) if i not in _KING_ZONE_SUMMARY_INDICES
            )
        self.num_summaries_effective = len(self.summary_indices)

        self.trunk = BoardConvStem(
            input_channels=self.input_channels,
            channels=self.channels,
            depth=self.depth,
            use_batchnorm=self.use_batchnorm,
        )
        # 5 learned pressure fields (one per packet field) over the trunk.
        self.pressure_head = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.channels, _NUM_FIELDS, kernel_size=1),
        )
        # Threshold grid: linspace covering the typical activation range
        # of a Conv head before sigmoid. Made learnable by default so the
        # model can pick the operating regime where transitions matter.
        init_thresholds = torch.linspace(-2.0, 2.0, max(self.num_thresholds, 1))
        if self.learn_thresholds:
            self.threshold_param = nn.Parameter(init_thresholds)
        else:
            self.register_buffer("threshold_param", init_thresholds, persistent=False)
        # Learnable temperature in log-space to stay positive.
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(self.base_temperature), dtype=torch.float32)
        )
        # Soft-max pooling temperature for the largest-component proxy.
        self.register_buffer(
            "component_softmax_temperature",
            torch.tensor(0.5, dtype=torch.float32),
            persistent=False,
        )
        # 3x3 neighbour kernel used for local-mass / king-zone /
        # piece-zone dilations. Re-used across all pressure fields and
        # piece types.
        self.register_buffer("neighbor_kernel", _neighbor_kernel(3), persistent=False)

        readout_dim = self._readout_dim()
        self.readout = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(self.hidden_dim, 1),
        )

    def _readout_dim(self) -> int:
        if not self.uses_threshold_sweep:
            return _NUM_FIELDS  # pressure_mean per field
        T = self.num_thresholds_effective
        S = self.num_summaries_effective
        # Anchor summary at the central threshold + critical-curve
        # differences across thresholds (or zeros when T == 1).
        curve_T = max(T - 1, 1)
        return _NUM_FIELDS * S * (1 + curve_T)

    def _piece_zone(self, board: torch.Tensor, white_idx: int, black_idx: int) -> torch.Tensor:
        plane = (board[:, white_idx] + board[:, black_idx]).clamp(0.0, 1.0).unsqueeze(1)
        zone = F.conv2d(plane, self.neighbor_kernel, padding=1)
        return zone.clamp(0.0, 1.0)

    def _summaries_for_field_tau(
        self,
        field_tau: torch.Tensor,
        zones: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return ``(B, F, T, S_full)`` summaries for the field-tau grid.

        ``field_tau`` has shape ``(B, F, T, 8, 8)``. The seven full
        summaries (mass, king-zone mass, largest component, boundary
        length, king/queen/rook surplus) are computed before the
        ablation slice in ``self.summary_indices``.
        """
        B, F_dim, T, H, W = field_tau.shape
        flat = field_tau.reshape(B * F_dim * T, 1, H, W)
        # mass
        mass = flat.sum(dim=(2, 3)).view(B, F_dim, T)
        # king-zone mass
        king_zone = zones["king"].unsqueeze(1).unsqueeze(1)  # (B, 1, 1, 8, 8)
        king_zone_mass = (field_tau * king_zone).sum(dim=(-2, -1))
        # largest soft component proxy: 3x3 local-sum, then soft-max
        # pool across the 64 squares with a learnable temperature.
        local_sum = F.conv2d(flat, self.neighbor_kernel, padding=1).view(B, F_dim, T, H * W)
        component_temperature = self.component_softmax_temperature.clamp_min(1.0e-3)
        soft_weights = torch.softmax(local_sum / component_temperature, dim=-1)
        largest_component = (soft_weights * local_sum).sum(dim=-1)
        # boundary length: vertical + horizontal first differences.
        diff_h = (field_tau[..., 1:, :] - field_tau[..., :-1, :]).abs().sum(dim=(-2, -1))
        diff_w = (field_tau[..., :, 1:] - field_tau[..., :, :-1]).abs().sum(dim=(-2, -1))
        boundary_length = diff_h + diff_w
        # piece-surplus around K / Q / R (own + opp, both sides combined).
        surpluses: list[torch.Tensor] = []
        for piece in ("king", "queen", "rook"):
            zone = zones[piece].unsqueeze(1).unsqueeze(1)
            surpluses.append((field_tau * zone).sum(dim=(-2, -1)))

        full = torch.stack(
            [
                mass,
                king_zone_mass,
                largest_component,
                boundary_length,
                surpluses[0],
                surpluses[1],
                surpluses[2],
            ],
            dim=-1,
        )  # (B, F, T, 7)
        return full

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feats = self.trunk(x)  # (B, C, 8, 8)
        batch_size = feats.shape[0]
        pressures = self.pressure_head(feats)  # (B, F, 8, 8)
        pressure_mean = pressures.mean(dim=(-2, -1))  # (B, F)

        # Piece zones used by the king-zone / piece-surplus summaries.
        zones = {
            piece: self._piece_zone(x, white_idx, black_idx).squeeze(1)
            for piece, (white_idx, black_idx) in _PIECE_PLANE_PAIRS.items()
        }

        if not self.uses_threshold_sweep:
            # ``pressure_mean_only`` ablation: skip the threshold sweep
            # entirely. Readout is just the mean pressure per field.
            T_eff = 0
            S_eff = self.num_summaries_effective
            field_tau = pressures.new_zeros(batch_size, _NUM_FIELDS, 0, 8, 8)
            summaries_full = pressures.new_zeros(batch_size, _NUM_FIELDS, 0, _NUM_SUMMARIES)
            summaries = pressures.new_zeros(batch_size, _NUM_FIELDS, 0, S_eff)
            critical_curves = pressures.new_zeros(batch_size, _NUM_FIELDS, 1, S_eff)
            mass_curve = pressures.new_zeros(batch_size, _NUM_FIELDS, 0)
            kzm_curve = pressures.new_zeros(batch_size, _NUM_FIELDS, 0)
            comp_curve = pressures.new_zeros(batch_size, _NUM_FIELDS, 0)
            boundary_curve = pressures.new_zeros(batch_size, _NUM_FIELDS, 0)
            critical_pressure_score = pressures.new_zeros(batch_size)
            readout_features = pressure_mean
            thresholds_eff = pressures.new_zeros(0)
            temperature_eff = self.log_temperature.exp()
        else:
            if self.ablation == "single_threshold":
                # Use the median threshold of the learned grid so the
                # ablation sees a single representative operating point.
                mid = self.num_thresholds // 2
                thresholds_eff = self.threshold_param[mid : mid + 1]
            else:
                thresholds_eff = self.threshold_param
            T_eff = int(thresholds_eff.shape[0])
            temperature_eff = self.log_temperature.exp().clamp_min(1.0e-3)
            tau = thresholds_eff.view(1, 1, T_eff, 1, 1)
            pressure_expand = pressures.unsqueeze(2)  # (B, F, 1, 8, 8)
            field_tau = torch.sigmoid((pressure_expand - tau) / temperature_eff)
            summaries_full = self._summaries_for_field_tau(field_tau, zones)
            summaries = summaries_full[..., list(self.summary_indices)]
            mass_curve = summaries_full[..., 0]
            kzm_curve = summaries_full[..., 1]
            comp_curve = summaries_full[..., 2]
            boundary_curve = summaries_full[..., 3]
            if T_eff >= 2:
                critical_curves = summaries[..., 1:, :] - summaries[..., :-1, :]
            else:
                critical_curves = pressures.new_zeros(
                    batch_size, _NUM_FIELDS, 1, self.num_summaries_effective
                )
            critical_pressure_score = critical_curves.abs().sum(dim=(1, 2, 3))
            anchor_idx = T_eff // 2
            anchor = summaries[:, :, anchor_idx, :]  # (B, F, S_eff)
            curve_flat = critical_curves.reshape(batch_size, -1)
            anchor_flat = anchor.reshape(batch_size, -1)
            readout_features = torch.cat([anchor_flat, curve_flat], dim=-1)

        scalar_logit = self.readout(readout_features).squeeze(-1)
        if self.num_classes == 1:
            logits = scalar_logit
        else:
            logits = torch.zeros(
                batch_size, self.num_classes, device=feats.device, dtype=feats.dtype
            )
            logits[:, -1] = scalar_logit

        with torch.no_grad():
            ones = torch.ones(batch_size, device=feats.device, dtype=feats.dtype)
            ablation_active = ones * (0.0 if self.ablation == "none" else 1.0)
            uses_threshold_sweep = ones * (1.0 if self.uses_threshold_sweep else 0.0)
            uses_pressure_curve = ones * (1.0 if self.uses_pressure_curve else 0.0)
            uses_king_zone_features = ones * (1.0 if self.uses_king_zone_features else 0.0)
            num_thresholds_effective = ones * float(self.num_thresholds_effective)
            num_summaries_effective = ones * float(self.num_summaries_effective)

        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "pressure_fields": pressures,
            "pressure_mean": pressure_mean,
            "thresholds": thresholds_eff,
            "temperature": temperature_eff,
            "field_tau": field_tau,
            "summaries": summaries,
            "critical_curves": critical_curves,
            "mass_curve": mass_curve,
            "king_zone_mass_curve": kzm_curve,
            "largest_component_curve": comp_curve,
            "boundary_length_curve": boundary_curve,
            "critical_pressure_score": critical_pressure_score,
            "readout_features": readout_features,
            "trunk_features": feats,
            "ablation_active": ablation_active,
            "uses_threshold_sweep": uses_threshold_sweep,
            "uses_pressure_curve": uses_pressure_curve,
            "uses_king_zone_features": uses_king_zone_features,
            "num_thresholds_effective": num_thresholds_effective,
            "num_summaries_effective": num_summaries_effective,
        }
        if self.num_classes == 1:
            output["prob"] = torch.sigmoid(logits)
        return output


def build_phase_transition_pressure_network_from_config(
    config: dict[str, Any],
) -> PhaseTransitionPressureNetwork:
    cfg = dict(config)
    cfg.pop("name", None)
    cfg.pop("mechanism_family", None)
    cfg.pop("packet_profile", None)
    return PhaseTransitionPressureNetwork(
        input_channels=int(cfg.pop("input_channels", 18)),
        num_classes=int(cfg.pop("num_classes", 1)),
        channels=int(cfg.pop("channels", 64)),
        depth=int(cfg.pop("depth", 2)),
        thresholds=int(cfg.pop("thresholds", 8)),
        temperature=float(cfg.pop("temperature", 0.2)),
        hidden_dim=int(cfg.pop("hidden_dim", 96)),
        dropout=float(cfg.pop("dropout", 0.1)),
        use_batchnorm=bool(cfg.pop("use_batchnorm", True)),
        learn_thresholds=bool(cfg.pop("learn_thresholds", True)),
        ablation=str(cfg.pop("ablation", "none")),
    )


__all__ = [
    "PhaseTransitionPressureNetwork",
    "build_phase_transition_pressure_network_from_config",
]

"""Pin / X-ray / Skewer primitive (p049, PXS).

Source: ``ideas/research/primitives/external_44_pin_xray_skewer_primitive.md``
(the source markdown uses the working name ``p043``; the registry slot
was reassigned to ``p049`` because ``p043`` was previously consumed by
``grassmann_rook_pool``). The spec's recommended phase-1 ("standalone
head + i018 hybrid") is implemented here as a gated additive head on
top of the i193 ``ExchangeThenKingDualStreamNetwork`` trunk -- the
repo's current strong baseline -- not on i018 itself. Phase-3 native
i018 relation integration is left to a follow-up idea.

Geometry. The spec calls for *exact ordered blocker facts*: along each
of the 8 queen-style ray directions from every source square, identify
the first and second occupied squares using a cumsum over the existing
``RayGeometry`` table. From those, six typed event masses can be read
off as pure tensor expressions (no Python-side per-ray loops at
training time):

  * `xray1`            -- one-blocker x-ray pressure to a target
  * `abs_pin`          -- our slider, enemy first blocker, enemy king
                          second on the same ray
  * `rel_pin`          -- our slider, enemy first blocker, enemy queen
                          / rook second on the same ray
  * `discovered`       -- our slider, *our* first blocker, enemy target
                          second on the same ray (i.e. the blocker can
                          move to unveil the slider)
  * `skewer`           -- our slider, enemy first blocker, second enemy
                          piece on the same ray, front-target value
                          greater than back-target value
  * `pinned_defender`  -- pinned enemy blocker weighted by the value
                          of the friendly assets it currently defends

Each event mass is computed per source square per ray direction, then
direction-summed and globally pooled to a fixed feature vector. The
feature vector is concatenated with the trunk joint pool feature and
projected to a scalar additive logit delta. The delta is gated with a
sigmoid head over the same joint feature, initialised near-closed
(`gate_init = -2.0`) so the i193 baseline is exactly recovered at the
start of training. The ``zero_delta`` / ``trunk_only`` ablations also
recover the baseline numerically.

Inputs are exactly the ``simple_18`` current-board tensor. Piece colour
is resolved against the side-to-move plane so the operator works
symmetrically for white-to-move and black-to-move positions. CRTK
metadata, source labels, verification flags, engine evaluations, and
principal variations are *not* consumed.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.primitives.ray_geometry import (
    DIRECTIONS,
    NUM_DIRECTIONS,
    RAY_MAX_LEN,
    RayGeometry,
    SQUARES,
)
from chess_nn_playground.models.primitives.trunk_features import trunk_joint_features
from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    DualStreamFeatureBuilder,
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


NUM_PIECE_CHANNELS = 12
NUM_EVENTS = 6
EVENT_NAMES: tuple[str, ...] = (
    "xray1",
    "abs_pin",
    "rel_pin",
    "discovered",
    "skewer",
    "pinned_defender",
)

# Indices into per-side piece planes (P, N, B, R, Q, K).
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 0, 1, 2, 3, 4, 5

# Default piece values, in pawn-equivalents, normalised to [0, 1].
DEFAULT_PIECE_VALUES: tuple[float, ...] = (1.0, 3.0, 3.0, 5.0, 9.0, 12.0)

ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    # Primary falsifier: drop the one-blocker x-ray channel by zeroing
    # the events that depend on `second_occ` (abs_pin, rel_pin, skewer,
    # discovered, pinned_defender, and the xray1 weighting).
    "no_xray1",
    # Uniform piece-value field: all targets weighted equally.
    "uniform_values",
    # Disable pinned-defender event mass (sets channel 5 to 0).
    "no_pin_def",
    # Shuffle the rule-derived ray index table in-batch so the
    # geometry mask is decoupled from the position. This is the spec's
    # "order-scramble ablation".
    "shuffle_rays",
    # Bypass the primitive entirely.
    "zero_delta",
    "trunk_only",
    "disable_gate",
)


def _direction_family_masks() -> tuple[torch.Tensor, torch.Tensor]:
    """Return (dir_is_orth, dir_is_diag) of shape (8,) float32.

    Mirrors the direction order in :data:`DIRECTIONS`. Orthogonal
    directions are rook/queen rays; diagonal directions are
    bishop/queen rays. Queen sources fire on both families.
    """
    orth = torch.zeros(NUM_DIRECTIONS, dtype=torch.float32)
    diag = torch.zeros(NUM_DIRECTIONS, dtype=torch.float32)
    for d, (dr, df) in enumerate(DIRECTIONS):
        if dr == 0 or df == 0:
            orth[d] = 1.0
        else:
            diag[d] = 1.0
    return orth, diag


class PinXraySkewerBuilder(nn.Module):
    """Tensorised ordered-blocker event builder.

    Forward signature::

        forward(piece_state, occupancy, value_field=None)
            -> (events_per_square, summary)

    where:

    * ``piece_state`` is ``(B, 12, 64)`` containing the side-canonical
      piece planes (channels 0-5 = us, 6-11 = them).
    * ``occupancy`` is ``(B, 64)`` boolean-coded as 0.0 / 1.0 floats.
    * ``value_field`` is an optional ``(6,)`` tensor of per-piece-type
      target values; if omitted, the module's :attr:`piece_values`
      learnable parameter (initialised to :data:`DEFAULT_PIECE_VALUES`)
      is used.

    The module owns the ray-geometry buffers and the direction-family
    masks. It exposes a single learnable scalar gate per event channel
    (``event_scale``) so the head can attenuate noisy channels during
    training without changing the primitive's mathematical signature.
    """

    def __init__(self) -> None:
        super().__init__()
        geom = RayGeometry.build()
        # (8, 64, 7) long index table and float mask.
        self.register_buffer("ray_step_index", geom.step_index, persistent=False)
        self.register_buffer("ray_step_mask", geom.step_mask, persistent=False)
        orth, diag = _direction_family_masks()
        self.register_buffer("dir_is_orth", orth, persistent=False)
        self.register_buffer("dir_is_diag", diag, persistent=False)
        # Initialised piece-value field, bounded by softmax in forward().
        self.piece_values = nn.Parameter(torch.tensor(DEFAULT_PIECE_VALUES))
        self.event_scale = nn.Parameter(torch.zeros(NUM_EVENTS))

    def _gather_scalar(self, scalar: torch.Tensor, scramble: torch.Tensor | None) -> torch.Tensor:
        """Gather a per-square scalar along all rays.

        Args:
            scalar: ``(B, 64)`` per-square scalar.
            scramble: Optional ``(8, 64, 7)`` long override for the ray
                index table (used by the ``shuffle_rays`` ablation).

        Returns:
            ``(B, 8, 64, 7)`` with off-board steps masked to 0.0.
        """
        if scramble is None:
            idx = self.ray_step_index
        else:
            idx = scramble
        flat = idx.reshape(-1)
        gathered = scalar[:, flat].reshape(
            scalar.shape[0], NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN
        )
        mask = self.ray_step_mask.to(device=scalar.device, dtype=scalar.dtype).view(
            1, NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN
        )
        return gathered * mask

    def _slider_source(
        self,
        us_planes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``(B, 8, 64)`` per-direction slider activations.

        A direction `d` is active at source square `s` iff:
          * an own queen sits on `s` (queens fire in all 8 directions),
          * an own rook sits on `s` and `d` is orthogonal,
          * an own bishop sits on `s` and `d` is diagonal.

        Args:
            us_planes: ``(B, 6, 64)`` per-square mover-side piece planes.
        """
        # us_planes channels: P=0, N=1, B=2, R=3, Q=4, K=5.
        rook = us_planes[:, ROOK]                                 # (B, 64)
        bishop = us_planes[:, BISHOP]                             # (B, 64)
        queen = us_planes[:, QUEEN]                               # (B, 64)
        orth = self.dir_is_orth.view(1, NUM_DIRECTIONS, 1)
        diag = self.dir_is_diag.view(1, NUM_DIRECTIONS, 1)
        slider = (
            queen.unsqueeze(1)
            + rook.unsqueeze(1) * orth
            + bishop.unsqueeze(1) * diag
        )
        return slider.clamp(0.0, 1.0)

    def forward(
        self,
        piece_state: torch.Tensor,
        occupancy: torch.Tensor,
        ablation: str = "none",
    ) -> dict[str, torch.Tensor]:
        if piece_state.dim() != 3 or piece_state.shape[1] != NUM_PIECE_CHANNELS:
            raise ValueError(
                "piece_state must be (B, 12, 64) in mover-oriented order; "
                f"got {tuple(piece_state.shape)}"
            )
        if occupancy.dim() != 2 or occupancy.shape[1] != SQUARES:
            raise ValueError(
                f"occupancy must be (B, 64); got {tuple(occupancy.shape)}"
            )
        batch = piece_state.shape[0]
        device = piece_state.device
        dtype = piece_state.dtype

        scramble: torch.Tensor | None = None
        if ablation == "shuffle_rays":
            perm = torch.randperm(NUM_DIRECTIONS * SQUARES * RAY_MAX_LEN, device=device)
            scramble = self.ray_step_index.reshape(-1)[perm].view(
                NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN
            )

        us = piece_state[:, :6]              # (B, 6, 64)
        them = piece_state[:, 6:12]          # (B, 6, 64)
        us_any = us.sum(dim=1).clamp(0.0, 1.0)
        them_any = them.sum(dim=1).clamp(0.0, 1.0)
        them_king = them[:, KING]
        them_queen = them[:, QUEEN]
        them_rook = them[:, ROOK]

        # Per-piece-type value field. The softmax keeps values bounded
        # in (0, 1) and sums to 1.0 over piece types; that prevents
        # value explosion under joint scaling.
        if ablation == "uniform_values":
            values = torch.full(
                (6,), 1.0 / 6.0, device=device, dtype=dtype
            )
        else:
            values = torch.softmax(self.piece_values.to(dtype=dtype), dim=0)
        them_value = (them * values.view(1, 6, 1)).sum(dim=1)        # (B, 64)
        us_value = (us * values.view(1, 6, 1)).sum(dim=1)            # (B, 64)

        # Per-direction slider activation (B, 8, 64).
        slider = self._slider_source(us)

        # Gather per-ray sequences: (B, 8, 64, 7).
        occ_seq = self._gather_scalar(occupancy, scramble)
        us_any_seq = self._gather_scalar(us_any, scramble)
        them_any_seq = self._gather_scalar(them_any, scramble)
        them_king_seq = self._gather_scalar(them_king, scramble)
        them_queen_seq = self._gather_scalar(them_queen, scramble)
        them_rook_seq = self._gather_scalar(them_rook, scramble)
        them_value_seq = self._gather_scalar(them_value, scramble)
        us_value_seq = self._gather_scalar(us_value, scramble)

        occ_bool = (occ_seq > 0.5).to(dtype=dtype)
        # Cumulative occupancy along the ray: c_l = sum_{j <= l} o_j.
        cum_occ = occ_bool.cumsum(dim=-1)
        first_occ = occ_bool * (cum_occ <= 1.0).to(dtype=dtype)
        second_occ = occ_bool * ((cum_occ >= 1.5) & (cum_occ <= 2.5)).to(dtype=dtype)

        first_us = first_occ * us_any_seq
        first_them = first_occ * them_any_seq
        second_them_king = second_occ * them_king_seq
        second_them_queen = second_occ * them_queen_seq
        second_them_rook = second_occ * them_rook_seq
        second_them_any = second_occ * them_any_seq
        second_them_value = second_occ * them_value_seq
        first_them_value = first_occ * them_value_seq

        # Reduce each ray to scalar event masses by summing over steps.
        # All step tensors are non-negative; the sum is finite.
        first_us_per_ray = first_us.sum(dim=-1)                       # (B, 8, 64)
        first_them_per_ray = first_them.sum(dim=-1)
        second_king_per_ray = second_them_king.sum(dim=-1)
        second_queen_per_ray = second_them_queen.sum(dim=-1)
        second_rook_per_ray = second_them_rook.sum(dim=-1)
        second_any_per_ray = second_them_any.sum(dim=-1)
        second_value_per_ray = second_them_value.sum(dim=-1)
        first_value_per_ray = first_them_value.sum(dim=-1)

        # Source slider weight (B, 8, 64).
        src = slider

        # Event masses are *per ray*: each is the joint product of the
        # ray's slider activation and the first/second pattern. When
        # the ``no_xray1`` ablation is on, the second-occupancy terms
        # are zeroed so the operator collapses to clear-ray geometry.
        if ablation == "no_xray1":
            zero_second = torch.zeros_like(second_any_per_ray)
            xray1 = src * zero_second
            abs_pin = src * first_them_per_ray * zero_second
            rel_pin = src * first_them_per_ray * zero_second
            skewer = src * first_them_per_ray * zero_second
            disc = src * first_us_per_ray * zero_second
            pinned_def = src * first_them_per_ray * zero_second
        else:
            # X-ray pressure: source ray that lands on a one-blocker
            # target on the same ray (own or enemy blocker, weighted by
            # the value of the latent target behind it).
            xray1 = (
                src
                * (first_them_per_ray + first_us_per_ray)
                * second_value_per_ray
            )
            # Absolute pin: enemy blocker in front of enemy king.
            abs_pin = src * first_them_per_ray * second_king_per_ray
            # Relative pin: enemy blocker in front of enemy queen / rook.
            rel_pin = src * first_them_per_ray * (
                second_queen_per_ray + 0.6 * second_rook_per_ray
            )
            # Discovered attack: our own blocker in front of an enemy
            # target (the blocker can move to unveil the slider).
            disc = src * first_us_per_ray * second_value_per_ray
            # Skewer: enemy first blocker, enemy second target, front
            # more valuable than back. ReLU keeps the channel non-neg.
            margin = torch.relu(first_value_per_ray - second_value_per_ray)
            skewer = src * second_any_per_ray * margin
            # Pinned-defender proxy: absolute-pin event weighted by the
            # value of the pinned piece itself (high-value pinned
            # blockers carry more defensive obligation). This is the
            # simplest defender-load surrogate that avoids building a
            # full own-side defence graph inside the primitive.
            pinned_def = src * first_value_per_ray * second_king_per_ray

        # Direction-sum to (B, 64) per-source-square event masses.
        xray1_sq = xray1.sum(dim=1)
        abs_pin_sq = abs_pin.sum(dim=1)
        rel_pin_sq = rel_pin.sum(dim=1)
        disc_sq = disc.sum(dim=1)
        skewer_sq = skewer.sum(dim=1)
        pinned_def_sq = pinned_def.sum(dim=1)
        if ablation == "no_pin_def":
            pinned_def_sq = torch.zeros_like(pinned_def_sq)

        events_per_square = torch.stack(
            [xray1_sq, abs_pin_sq, rel_pin_sq, disc_sq, skewer_sq, pinned_def_sq],
            dim=1,
        )                                                              # (B, 6, 64)
        # Apply per-event learnable scales (sigmoid-bounded into (0, 1)).
        scales = torch.sigmoid(self.event_scale.to(dtype=dtype)).view(1, NUM_EVENTS, 1)
        events_scaled = events_per_square * scales

        summary = torch.cat(
            [events_scaled.mean(dim=-1), events_scaled.amax(dim=-1)],
            dim=-1,
        )                                                              # (B, 12)

        return {
            "events_per_square": events_scaled,
            "summary": summary,
            "event_scales": scales.view(NUM_EVENTS),
            "values": values,
        }


class PinXraySkewer(nn.Module):
    """p049 -- Pin / X-ray / Skewer head over the i193 trunk."""

    ALLOWED_ABLATIONS = ALLOWED_ABLATIONS

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        # Trunk hyper-parameters mirror the i193 builder.
        trunk_channels: int = 64,
        trunk_hidden_dim: int = 96,
        trunk_depth: int = 2,
        trunk_dropout: float = 0.1,
        trunk_use_batchnorm: bool = True,
        trunk_gate_dim: int | None = None,
        trunk_ablation: str = "none",
        # PXS head hyper-parameters.
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "PinXraySkewer supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError("PinXraySkewer requires the simple_18 board tensor")
        if str(ablation) not in self.ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(self.ALLOWED_ABLATIONS)}"
            )

        self.num_classes = 1
        self.ablation = str(ablation)
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

        self.builder = PinXraySkewerBuilder()

        self.feature_dim = (
            2 * self.trunk.exchange_encoder.output_dim
            + DualStreamFeatureBuilder.SUMMARY_DIM
        )
        event_summary_dim = 2 * NUM_EVENTS
        head_dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim + event_summary_dim),
            nn.Linear(self.feature_dim + event_summary_dim, int(head_hidden_dim)),
            nn.GELU(),
            head_dropout_module,
            nn.Linear(int(head_hidden_dim), 1),
        )
        gate_in = self.feature_dim + 1  # joint + event_total_mean
        self.gate_head = nn.Sequential(
            nn.LayerNorm(gate_in),
            nn.Linear(gate_in, max(16, int(head_hidden_dim) // 2)),
            nn.GELU(),
            nn.Linear(max(16, int(head_hidden_dim) // 2), 1),
        )
        with torch.no_grad():
            final = self.gate_head[-1]
            if isinstance(final, nn.Linear):
                final.bias.fill_(float(gate_init))

    def _mover_piece_state(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mover-oriented (piece_state, occupancy) tensors.

        Channels 0-5 of ``piece_state`` are the side-to-move's pieces;
        channels 6-11 are the opponent's. The simple_18 piece planes
        are *absolute* (white = 0..5, black = 6..11); we swap them
        using the side-to-move scalar plane 12.
        """
        batch = board.shape[0]
        piece_planes = board[:, :NUM_PIECE_CHANNELS].clamp(0.0, 1.0).flatten(2)
        stm = board[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0).view(batch, 1, 1)
        white = piece_planes[:, :6]
        black = piece_planes[:, 6:12]
        us = stm * white + (1.0 - stm) * black
        them = stm * black + (1.0 - stm) * white
        mover = torch.cat([us, them], dim=1)
        occupancy = piece_planes.sum(dim=1).clamp(0.0, 1.0)
        return mover, occupancy

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]

        trunk_out = self.trunk(board)
        base_logit = trunk_out["logits"].view(-1)
        joint, _, _ = trunk_joint_features(self.trunk, board)

        mover, occupancy = self._mover_piece_state(board)
        builder_out = self.builder(mover, occupancy, ablation=self.ablation)
        events_per_square = builder_out["events_per_square"]
        summary = builder_out["summary"]                              # (B, 12)
        event_total_mean = events_per_square.mean(dim=(1, 2))         # (B,)

        delta_input = torch.cat([joint, summary], dim=1)
        delta_raw = self.delta_head(delta_input).view(-1)

        gate_input = torch.cat([joint, event_total_mean.unsqueeze(-1)], dim=1)
        gate_logit = self.gate_head(gate_input).view(-1)
        gate = torch.sigmoid(gate_logit)
        if self.ablation == "disable_gate":
            gate = torch.ones_like(gate)

        if self.ablation in {"zero_delta", "trunk_only"}:
            primitive_delta = torch.zeros_like(base_logit)
            gate_applied = torch.zeros_like(gate)
        else:
            primitive_delta = delta_raw
            gate_applied = gate
        contribution = gate_applied * primitive_delta
        logits = base_logit + contribution

        eps = 1.0e-6
        gate_clamped = gate.clamp(eps, 1.0 - eps)
        gate_entropy = -(
            gate_clamped * gate_clamped.log()
            + (1.0 - gate_clamped) * (1.0 - gate_clamped).log()
        )

        out: dict[str, torch.Tensor] = {}
        for key, value in trunk_out.items():
            if key in {"logits", "proposal_profile_strength", "proposal_keyword_count"}:
                continue
            out[f"trunk_{key}"] = value
        out["logits"] = logits
        out["base_logit"] = base_logit
        out["primitive_delta"] = primitive_delta
        out["primitive_delta_raw"] = delta_raw
        out["primitive_gate"] = gate
        out["primitive_gate_applied"] = gate_applied
        out["primitive_gate_logit"] = gate_logit
        out["primitive_gate_entropy"] = gate_entropy
        out["primitive_contribution"] = contribution
        out["pxs_event_total_mean"] = event_total_mean
        for idx, name in enumerate(EVENT_NAMES):
            out[f"pxs_{name}_mean"] = events_per_square[:, idx].mean(dim=-1)
            out[f"pxs_{name}_max"] = events_per_square[:, idx].amax(dim=-1)
        out["mechanism_energy"] = trunk_out["mechanism_energy"] + event_total_mean.detach()
        out["proposal_profile_strength"] = primitive_delta.detach().abs() * gate_entropy
        out["proposal_keyword_count"] = logits.new_full((batch,), float(NUM_EVENTS))
        return out


def build_pin_xray_skewer_from_config(config: dict[str, Any]) -> PinXraySkewer:
    cfg = dict(config)
    return PinXraySkewer(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim")),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )


__all__ = (
    "ALLOWED_ABLATIONS",
    "EVENT_NAMES",
    "NUM_EVENTS",
    "PinXraySkewer",
    "PinXraySkewerBuilder",
    "build_pin_xray_skewer_from_config",
)

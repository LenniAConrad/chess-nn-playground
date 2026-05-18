"""Efficient Ray Occlusion Scan (p054).

Source: ``ideas/research/primitives/external_49_efficient_ray_occlusion_scan_primitive.md``
(single proposal; the research markdown promotes a compact, tensorized
ray-occlusion scan that operates over the legal queen-direction ray
representation rather than the dense ``(B, 64, 64, 64)`` source-target-
between cube used by i018's ``TacticalIncidenceBuilder``).

Mathematical signature (per batch ``b``, source square ``s``, direction
``d in {N, NE, E, SE, S, SW, W, NW}``, step ``l in {1..7}``):

    o_{b,s,d,l} = Occ(c_{s,d,l})                            # gathered occupancy
    z_{b,s,d,l} = Feat(c_{s,d,l})                            # gathered side/value feats
    k_{b,s,d,l} = sum_{q<=l} o_{b,s,d,q}                     # inclusive blocker count
    visible_{b,s,d,l} = 1[k_{b,s,d,l} - o_{b,s,d,l} == 0]
    first_{b,s,d,l}   = o_{b,s,d,l} * 1[k_{b,s,d,l} == 1]
    second_{b,s,d,l}  = o_{b,s,d,l} * 1[k_{b,s,d,l} == 2]
    xraylane_{b,s,d,l} = 1[k_{b,s,d,l} - o_{b,s,d,l} == 1]

Everything needed for line-of-sight, first/second blocker identity,
mobility length, x-ray pressure, discovered-attack candidates, and pin
candidates falls out of these four tensors. The forward path contains
**no Python loop over ray steps or directions** -- only batched
``gather`` / ``cumsum`` / equality tests / reductions, identical to the
plan in the research markdown.

The primitive is wired as an additive gated logit delta over the i193
ExchangeThenKingDualStreamNetwork trunk so it can be compared to the
existing ``oriented_sheaf_plus_primitive`` hybrid scaffolding (the same
shape as p020, p021, p042 - p046):

    final_logit = base_logit + gate * primitive_delta_raw

CRTK metadata, source labels, verification flags, engine evaluations,
and principal variations are *not* consumed.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.primitives.ray_geometry import (
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


PIECE_PLANE_COUNT = 12
US_PIECE_OFFSET = 0       # planes 0..5  -- our pieces (P, N, B, R, Q, K)
THEM_PIECE_OFFSET = 6     # planes 6..11 -- their pieces

# Standard piece values in pawns (king is given a large weight so xray
# pressure against the enemy king dominates). The order matches the
# simple_18 piece-plane layout: [P, N, B, R, Q, K] per side.
US_PIECE_VALUES: tuple[float, ...] = (1.0, 3.0, 3.0, 5.0, 9.0, 200.0)
THEM_PIECE_VALUES: tuple[float, ...] = (1.0, 3.0, 3.0, 5.0, 9.0, 200.0)

# Ordering of the 8 directions in ``ray_geometry.DIRECTIONS``:
#   (N, NE, E, SE, S, SW, W, NW)
# Orthogonal directions are even-indexed; diagonal directions are odd-indexed.
ORTHO_MASK: tuple[float, ...] = (1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0)
DIAG_MASK: tuple[float, ...] = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)

ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    "first_only",       # primary falsifier: drop second-blocker / xray / discovered / pin
    "no_blocker_id",    # use only visible-step counts; drop side/value identity channels
    "uniform_occupancy",  # treat board as fully occupied -- only step 1 visible
    "empty_occupancy",  # treat board as empty -- no blockers, just mobility geometry
    "shuffle_occupancy",  # in-batch permutation of occupancy -- decouples mask from position
    "zero_delta",
    "trunk_only",
    "disable_gate",
)


def _build_ray_feature_tensor(piece_state: torch.Tensor) -> torch.Tensor:
    """Compose the per-square 14-channel feature tensor used by the scan.

    Channels (per square):
        0       : our occupancy (sum of our piece planes)
        1       : their occupancy (sum of their piece planes)
        2       : our piece value (sum_p us_piece_value_p * us_plane_p)
        3       : their piece value (sum_p them_piece_value_p * them_plane_p)
        4..9    : our piece-type one-hots (P, N, B, R, Q, K)
        10..13  ... wait, we use 6 per side -> 4..15 -> 12 channels

    Returns a ``(B, 64, 14)`` tensor: 2 side flags + 2 value scalars +
    6 us one-hots + 6 them one-hots = 16 channels actually. We split it
    into the smaller pieces in the call site so the scan kernel does not
    need to know the channel layout.

    Concretely the returned tensor has 16 channels:
        [us_occ, them_occ, us_val, them_val, us_P..us_K, them_P..them_K].
    """
    batch = piece_state.shape[0]
    # piece_state: (B, 12, 8, 8) (already simple_18 piece planes, clamped 0/1).
    us = piece_state[:, US_PIECE_OFFSET : US_PIECE_OFFSET + 6]   # (B, 6, 8, 8)
    them = piece_state[:, THEM_PIECE_OFFSET : THEM_PIECE_OFFSET + 6]
    us_flat = us.flatten(2).transpose(1, 2).contiguous()           # (B, 64, 6)
    them_flat = them.flatten(2).transpose(1, 2).contiguous()       # (B, 64, 6)
    us_value_vec = piece_state.new_tensor(US_PIECE_VALUES)         # (6,)
    them_value_vec = piece_state.new_tensor(THEM_PIECE_VALUES)
    us_occ = us_flat.sum(dim=-1, keepdim=True).clamp(0.0, 1.0)     # (B, 64, 1)
    them_occ = them_flat.sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
    us_value = (us_flat * us_value_vec.view(1, 1, 6)).sum(dim=-1, keepdim=True)
    them_value = (them_flat * them_value_vec.view(1, 1, 6)).sum(dim=-1, keepdim=True)
    feat = torch.cat([us_occ, them_occ, us_value, them_value, us_flat, them_flat], dim=-1)
    assert feat.shape == (batch, SQUARES, 16)
    return feat


def ray_occlusion_scan(
    feat: torch.Tensor,
    occupancy: torch.Tensor,
    step_index: torch.Tensor,
    step_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute the compact ray-occlusion scan output.

    Args:
        feat: ``(B, 64, F)`` per-square feature tensor. The expected
            layout is the 16-channel composite returned by
            ``_build_ray_feature_tensor`` (us_occ, them_occ, us_val,
            them_val, 6 us one-hots, 6 them one-hots).
        occupancy: ``(B, 64)`` occupancy indicator (us_occ + them_occ,
            clamped to ``[0, 1]``).
        step_index: ``(8, 64, 7)`` long ray indices from ``RayGeometry``.
        step_mask: ``(8, 64, 7)`` float on-board mask.

    Returns:
        Dict of per-source-per-direction summaries, all batched tensors.
        See ``EfficientRayOcclusionScan.forward`` for the full key list.
    """
    if feat.ndim != 3 or feat.shape[1] != SQUARES:
        raise ValueError(f"Expected feat shape (B, 64, F), got {tuple(feat.shape)}")
    if occupancy.ndim != 2 or occupancy.shape[1] != SQUARES:
        raise ValueError(f"Expected occupancy shape (B, 64), got {tuple(occupancy.shape)}")
    if step_index.shape != (NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN):
        raise ValueError(f"Unexpected step_index shape {tuple(step_index.shape)}")
    if step_mask.shape != (NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN):
        raise ValueError(f"Unexpected step_mask shape {tuple(step_mask.shape)}")

    batch, _, channels = feat.shape
    device = feat.device
    dtype = feat.dtype

    flat_idx = step_index.reshape(-1)  # (D*S*L,)
    ray_mask = step_mask.to(device=device, dtype=dtype).view(1, NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN)

    # Gather occupancy and 16-channel feature in one batched pass.
    occ_ray = occupancy[:, flat_idx].reshape(batch, NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN) * ray_mask
    feat_ray = feat[:, flat_idx, :].reshape(
        batch, NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN, channels
    ) * ray_mask.unsqueeze(-1)

    # Inclusive blocker count along the ray (cumsum of occupancy).
    k = occ_ray.cumsum(dim=-1)
    k_prev = k - occ_ray  # blockers strictly before step l

    visible = (k_prev == 0).to(dtype) * ray_mask         # cells with no prior blocker
    first = occ_ray * (k == 1).to(dtype) * ray_mask      # one-hot selector for first blocker
    second = occ_ray * (k == 2).to(dtype) * ray_mask     # one-hot selector for second blocker
    xray_lane = (k_prev == 1).to(dtype) * ray_mask       # cells behind exactly one blocker

    # First / second blocker feature summaries: (B, D, S, 16).
    first_feat = (first.unsqueeze(-1) * feat_ray).sum(dim=-2)
    second_feat = (second.unsqueeze(-1) * feat_ray).sum(dim=-2)

    # Convenience side / value summaries derived from feature layout.
    first_us_occ = first_feat[..., 0]
    first_them_occ = first_feat[..., 1]
    first_value = first_feat[..., 2] + first_feat[..., 3]   # sum across both sides
    second_us_occ = second_feat[..., 0]
    second_them_occ = second_feat[..., 1]
    second_value = second_feat[..., 2] + second_feat[..., 3]

    first_exists = first.sum(dim=-1).clamp(0.0, 1.0)
    second_exists = second.sum(dim=-1).clamp(0.0, 1.0)

    # Mobility = number of cells reachable from source without crossing a blocker
    # (i.e. visible cells that are themselves empty).
    mobility_len = (visible * (1.0 - occ_ray)).sum(dim=-1)
    visible_count = visible.sum(dim=-1)
    xray_pressure = second_exists * second_value
    # Number of cells behind exactly one blocker (latent x-ray surface).
    xray_lane_len = (xray_lane * (1.0 - occ_ray)).sum(dim=-1)

    # Enemy king is feature channel 9 (us one-hots are 4..9, K is the 6th piece)
    # and us king is channel 15 (them one-hots are 10..15, K is the 6th piece).
    # So:
    #   first_feat[..., 9]  -> us-king at first blocker (== own king walking the ray)
    #   second_feat[..., 9] -> us-king at second blocker (rare)
    #   first_feat[..., 15] -> them-king at first blocker
    #   second_feat[..., 15]-> them-king at second blocker (pinned-enemy-king target)
    us_king_first = first_feat[..., 9]
    us_king_second = second_feat[..., 9]
    them_king_first = first_feat[..., 15]
    them_king_second = second_feat[..., 15]

    return {
        "occ_ray": occ_ray,
        "feat_ray": feat_ray,
        "visible_steps": visible,
        "first_steps": first,
        "second_steps": second,
        "xray_lane_steps": xray_lane,
        "first_feat": first_feat,
        "second_feat": second_feat,
        "first_us_occ": first_us_occ,
        "first_them_occ": first_them_occ,
        "first_value": first_value,
        "first_exists": first_exists,
        "second_us_occ": second_us_occ,
        "second_them_occ": second_them_occ,
        "second_value": second_value,
        "second_exists": second_exists,
        "us_king_first": us_king_first,
        "us_king_second": us_king_second,
        "them_king_first": them_king_first,
        "them_king_second": them_king_second,
        "mobility_len": mobility_len,
        "visible_count": visible_count,
        "xray_pressure": xray_pressure,
        "xray_lane_len": xray_lane_len,
    }


class EfficientRayOcclusionScan(nn.Module):
    """p054 Efficient Ray Occlusion Scan -- additive head on the i193 trunk."""

    ALLOWED_ABLATIONS = ALLOWED_ABLATIONS

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
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "EfficientRayOcclusionScan supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError("EfficientRayOcclusionScan requires the simple_18 board tensor")
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
        self.feature_dim = (
            2 * self.trunk.exchange_encoder.output_dim
            + DualStreamFeatureBuilder.SUMMARY_DIM
        )

        # Per-direction (us-rook-like, us-bishop-like, them-rook-like,
        # them-bishop-like) compatibility weights are simple rule-derived
        # constants, not learnable.
        geom = RayGeometry.build()
        self.register_buffer("ray_step_index", geom.step_index, persistent=False)
        self.register_buffer("ray_step_mask", geom.step_mask, persistent=False)
        self.register_buffer(
            "ortho_mask",
            torch.tensor(ORTHO_MASK, dtype=torch.float32).view(1, NUM_DIRECTIONS, 1),
            persistent=False,
        )
        self.register_buffer(
            "diag_mask",
            torch.tensor(DIAG_MASK, dtype=torch.float32).view(1, NUM_DIRECTIONS, 1),
            persistent=False,
        )

        # The scan emits 13 per-source-per-direction scalar summaries that are
        # then summed over the 8 directions and projected to a compact per-
        # square channel set. Layout (must match the building below):
        #   [visible_count, mobility_len, xray_lane_len,
        #    first_exists, first_value, first_us_occ, first_them_occ,
        #    second_exists, second_value, second_us_occ, second_them_occ,
        #    xray_pressure,
        #    direction_pinned_target]
        self.summary_channels = 13
        # Three line-class channels (rook-line, bishop-line, queen-line)
        # broadcast each summary -> per-square 3 * summary_channels.
        per_square_channels = 3 * self.summary_channels
        readout_dim = per_square_channels * 2  # mean + max pool over 64 squares

        head_dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(readout_dim + self.feature_dim),
            nn.Linear(readout_dim + self.feature_dim, int(head_hidden_dim)),
            nn.GELU(),
            head_dropout_module,
            nn.Linear(int(head_hidden_dim), 1),
        )
        gate_in = self.feature_dim + 3  # joint + (occ_density, mobility_mean, xray_mean)
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

    def _occupancy(self, board: torch.Tensor) -> torch.Tensor:
        planes = board[:, :PIECE_PLANE_COUNT].clamp(0.0, 1.0).flatten(2)
        return planes.sum(dim=1).clamp(0.0, 1.0)

    def _piece_state(self, board: torch.Tensor) -> torch.Tensor:
        return board[:, :PIECE_PLANE_COUNT].clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]
        eps = 1.0e-6

        trunk_out = self.trunk(board)
        base_logit = trunk_out["logits"].view(-1)
        joint, _, _ = trunk_joint_features(self.trunk, board)

        piece_state = self._piece_state(board)
        feat = _build_ray_feature_tensor(piece_state)
        occupancy = self._occupancy(board)

        # Ablation hooks on the occupancy input to the scan.
        if self.ablation == "empty_occupancy":
            occupancy = torch.zeros_like(occupancy)
        elif self.ablation == "uniform_occupancy":
            occupancy = torch.ones_like(occupancy)
        elif self.ablation == "shuffle_occupancy" and batch > 1:
            perm = torch.randperm(batch, device=occupancy.device)
            occupancy = occupancy[perm]

        scan = ray_occlusion_scan(
            feat,
            occupancy,
            self.ray_step_index,
            self.ray_step_mask,
        )

        # Per-source-per-direction summaries -> stack to (B, D, S, summary_channels).
        if self.ablation in {"first_only", "no_blocker_id"}:
            zero = torch.zeros_like(scan["mobility_len"])
        # Falsifier: ``first_only`` zeroes anything beyond first blocker, including
        # xray / discovered / pin features. ``no_blocker_id`` zeroes only the side
        # / value identity channels.
        second_value = scan["second_value"]
        second_us = scan["second_us_occ"]
        second_them = scan["second_them_occ"]
        xray_pressure = scan["xray_pressure"]
        xray_lane_len = scan["xray_lane_len"]
        first_value = scan["first_value"]
        first_us = scan["first_us_occ"]
        first_them = scan["first_them_occ"]
        if self.ablation == "first_only":
            second_value = zero
            second_us = zero
            second_them = zero
            xray_pressure = zero
            xray_lane_len = zero
        if self.ablation == "no_blocker_id":
            first_value = zero
            first_us = zero
            first_them = zero
            second_value = zero
            second_us = zero
            second_them = zero

        # Discovered-attack candidate: our slider line where first blocker is ours
        # and second blocker is theirs (we can move ours away to discover an
        # attack on second). Pinned-target candidate: our slider line whose
        # first blocker is theirs and second blocker is their king.
        # These are direction-wise scalars; we encode them as the
        # ``direction_pinned_target`` channel.
        discovered_pressure = first_us * second_them * second_value
        pinned_to_king = first_them * scan["them_king_second"]

        if self.ablation == "first_only":
            discovered_pressure = zero
            pinned_to_king = zero

        summary = torch.stack(
            [
                scan["visible_count"],
                scan["mobility_len"],
                xray_lane_len,
                scan["first_exists"],
                first_value,
                first_us,
                first_them,
                scan["second_exists"],
                second_value,
                second_us,
                second_them,
                xray_pressure,
                discovered_pressure + pinned_to_king,
            ],
            dim=-1,
        )  # (B, D, S, summary_channels)

        # Direction-class projection: rook-line == orthogonal directions,
        # bishop-line == diagonal directions, queen-line == both.
        ortho = self.ortho_mask.to(dtype=summary.dtype, device=summary.device).unsqueeze(-1)
        diag = self.diag_mask.to(dtype=summary.dtype, device=summary.device).unsqueeze(-1)
        rook_summary = (summary * ortho).sum(dim=1)        # (B, S, summary_channels)
        bishop_summary = (summary * diag).sum(dim=1)
        queen_summary = rook_summary + bishop_summary
        per_square = torch.cat([rook_summary, bishop_summary, queen_summary], dim=-1)
        # (B, S, 3 * summary_channels)

        pooled_mean = per_square.mean(dim=1)
        pooled_max, _ = per_square.max(dim=1)
        readout = torch.cat([pooled_mean, pooled_max], dim=-1)

        delta_input = torch.cat([readout, joint], dim=1)
        delta_raw = self.delta_head(delta_input).view(-1)

        # Diagnostics for the gate: per-sample occupancy density, mean mobility,
        # mean x-ray pressure.
        occ_density = occupancy.mean(dim=1)
        mobility_mean = scan["mobility_len"].mean(dim=(1, 2))
        xray_mean = scan["xray_pressure"].mean(dim=(1, 2))
        gate_input = torch.cat(
            [
                joint,
                occ_density.unsqueeze(-1),
                mobility_mean.unsqueeze(-1),
                xray_mean.unsqueeze(-1),
            ],
            dim=1,
        )
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

        gate_clamped = gate.clamp(eps, 1.0 - eps)
        gate_entropy = -(
            gate_clamped * gate_clamped.log()
            + (1.0 - gate_clamped) * (1.0 - gate_clamped).log()
        )

        # Aggregate visible-edge magnitude to bound the ``mechanism_energy`` diag.
        scan_norm = (
            scan["visible_steps"].sum(dim=(1, 2, 3))
            + scan["xray_lane_steps"].sum(dim=(1, 2, 3))
        ) / float(NUM_DIRECTIONS * SQUARES * RAY_MAX_LEN)

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
        out["eros_occupancy_density"] = occ_density
        out["eros_mobility_mean"] = mobility_mean
        out["eros_xray_pressure_mean"] = xray_mean
        out["eros_visible_density"] = scan_norm
        out["eros_first_blocker_rate"] = scan["first_exists"].mean(dim=(1, 2))
        out["eros_second_blocker_rate"] = scan["second_exists"].mean(dim=(1, 2))
        out["mechanism_energy"] = trunk_out["mechanism_energy"] + scan_norm.detach()
        out["proposal_profile_strength"] = (
            primitive_delta.detach().abs() * gate_entropy
        ).clamp(0.0, 20.0)
        out["proposal_keyword_count"] = logits.new_full(
            (batch,), float(3 * self.summary_channels)
        )
        return out


def build_efficient_ray_occlusion_scan_from_config(
    config: dict[str, Any],
) -> EfficientRayOcclusionScan:
    cfg = dict(config)
    return EfficientRayOcclusionScan(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim", None)),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )


__all__ = (
    "ALLOWED_ABLATIONS",
    "EfficientRayOcclusionScan",
    "build_efficient_ray_occlusion_scan_from_config",
    "ray_occlusion_scan",
)

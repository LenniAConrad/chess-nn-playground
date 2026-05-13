"""Octilinear Selective Scan (p034, OSS) primitive.

Source: ``ideas/research/primitives/external_29_incremental_move_update_octilinear_scan.md``
(Section "primitive_oss"; explicitly ranked first in the file's "Final
Ranking" section, "Highest potential for Mamba-level impact on engine
speed"). The proposal is a Mamba-style selective state-space scan run
along each of the eight chess ray directions:

    h_t = A_k(x_t) * h_{t-1} + B_k(x_t) * x_t

with the scan order tied to the chess board geometry (N/S/E/W cardinal
directions plus the four diagonal directions). The state ``h`` is
propagated or blocked depending on the per-square input feature -- the
gating matches a piece-occupancy "block" by virtue of the seed feature
carrying piece-existence channels.

We implement a per-direction scalar-state SSM where ``A_k(x_t),
B_k(x_t) in R^d`` are channelwise per-direction gates produced by a
small linear over the per-square feature. The eight per-direction
outputs are concatenated and fused into a hidden vector that drives the
additive gated logit delta over the i193 trunk.

The deferred internal proposals from external_29 are documented in the
idea registry notes. IMUA (Incremental Move-Update Accumulator) overlaps
with i248 TSDP and the existing rule-derived terminal features; EPTP
(Equivariant Piece-Type Permutator) is an orthogonal symmetry primitive;
LMHI (Legal-Masked Hyper-Interaction) overlaps with p032 DAG; DBLF
(Differentiable Bit-Logical Filter) is a generic soft-logic primitive.

CRTK metadata, source labels, verification flags, and engine scores are
*not* consulted. Seed features are derived from the simple_18 piece
planes and side-to-move plane.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.primitives.legal_move_graph import SQUARES
from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


NUM_PIECE_CHANNELS = 12

# Direction step offsets (plane-row delta, file delta). "North" in chess
# corresponds to *decreasing* plane row (plane row 0 = rank 8). The order
# matches the source primitive's "8 chess ray directions"; the SSM is run
# per-direction.
DIRECTION_STEPS: tuple[tuple[str, int, int], ...] = (
    ("E", 0, 1),
    ("W", 0, -1),
    ("N", -1, 0),
    ("S", 1, 0),
    ("NE", -1, 1),
    ("NW", -1, -1),
    ("SE", 1, 1),
    ("SW", 1, -1),
)
NUM_DIRECTIONS = len(DIRECTION_STEPS)

ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    "fixed_transition",       # A_k(x_t) := A_k (data-independent transition)
    "single_direction",       # use only the E direction (collapse 8 -> 1)
    "shuffle_features",       # in-batch shuffle of seed features
    "zero_delta",
    "trunk_only",
    "disable_gate",
)


def _build_direction_tracks() -> dict[str, torch.Tensor]:
    """Per-direction precomputed scan paths.

    For each direction returns a ``(num_tracks, 8)`` long tensor of square
    indices, padded with -1 where a track is shorter than 8 steps.
    ``num_tracks`` varies per direction: 8 for cardinals, 15 for diagonals.
    """
    tables: dict[str, torch.Tensor] = {}
    for name, dr, df in DIRECTION_STEPS:
        tracks: list[list[int]] = []
        seen_starts: set[tuple[int, int]] = set()
        for r in range(8):
            for f in range(8):
                pr, pf = r - dr, f - df
                if 0 <= pr < 8 and 0 <= pf < 8:
                    continue  # has a predecessor in this direction
                if (r, f) in seen_starts:
                    continue
                seen_starts.add((r, f))
                track: list[int] = []
                rr, ff = r, f
                while 0 <= rr < 8 and 0 <= ff < 8:
                    track.append(rr * 8 + ff)
                    rr += dr
                    ff += df
                tracks.append(track)
        max_len = 8
        padded = torch.full((len(tracks), max_len), -1, dtype=torch.long)
        for index, track in enumerate(tracks):
            padded[index, : len(track)] = torch.tensor(track, dtype=torch.long)
        tables[name] = padded
    return tables


class OctilinearSelectiveScan(nn.Module):
    """p034 -- Octilinear Selective Scan head over the i193 trunk.

    For each of eight chess ray directions, a channelwise selective SSM

        h_t = sigmoid(A(x_t)) * h_{t-1} + B(x_t) * x_t

    is run along the chess-rule ordering of the squares (``E``, ``W``,
    ``N``, ``S`` plus the four diagonals). The per-direction final
    feature is gathered back to a per-square representation, the eight
    direction-conditioned features are concatenated, and a small MLP
    pools them to a scalar logit delta added to the i193 baseline.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        # Trunk hyper-parameters.
        trunk_channels: int = 64,
        trunk_hidden_dim: int = 96,
        trunk_depth: int = 2,
        trunk_dropout: float = 0.1,
        trunk_use_batchnorm: bool = True,
        trunk_gate_dim: int | None = None,
        trunk_ablation: str = "none",
        # OSS head hyper-parameters.
        feature_dim: int = 16,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError("OctilinearSelectiveScan supports puzzle_binary one-logit contract")
        if int(input_channels) != 18:
            raise ValueError("OctilinearSelectiveScan requires the simple_18 board tensor")
        if str(ablation) not in ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(ALLOWED_ABLATIONS)}"
            )
        self.num_classes = 1
        self.ablation = str(ablation)
        self.feature_dim = int(feature_dim)
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

        self.square_feature_proj = nn.Linear(NUM_PIECE_CHANNELS + 1, self.feature_dim)

        # Per-direction data-dependent transition / input projections. ``A`` and
        # ``B`` are channelwise gates produced from the per-square input.
        self.a_logit_projections = nn.ModuleList(
            [nn.Linear(self.feature_dim, self.feature_dim) for _ in DIRECTION_STEPS]
        )
        self.b_projections = nn.ModuleList(
            [nn.Linear(self.feature_dim, self.feature_dim) for _ in DIRECTION_STEPS]
        )
        # Data-independent fallback ``A`` used by the ``fixed_transition`` ablation.
        self.fixed_a_logits = nn.Parameter(
            torch.full((NUM_DIRECTIONS, self.feature_dim), 0.5)
        )

        # Aggregate the 8 per-direction features into a hidden vector.
        self.direction_fuser = nn.Sequential(
            nn.LayerNorm(self.feature_dim * NUM_DIRECTIONS),
            nn.Linear(self.feature_dim * NUM_DIRECTIONS, int(head_hidden_dim)),
            nn.GELU(),
        )

        dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(int(head_hidden_dim) * 2),
            nn.Linear(int(head_hidden_dim) * 2, max(16, int(head_hidden_dim) // 2)),
            nn.GELU(),
            dropout_module,
            nn.Linear(max(16, int(head_hidden_dim) // 2), 1),
        )

        gate_in = 4 + NUM_DIRECTIONS  # trunk diagnostics + per-direction energy
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

        tracks = _build_direction_tracks()
        for name, _, _ in DIRECTION_STEPS:
            self.register_buffer(f"tracks_{name}", tracks[name], persistent=False)

    def _direction_tracks(self, name: str) -> torch.Tensor:
        return getattr(self, f"tracks_{name}")

    @staticmethod
    def _square_descriptor(board: torch.Tensor) -> torch.Tensor:
        piece_planes = board[:, :NUM_PIECE_CHANNELS].flatten(2).clamp(0.0, 1.0)
        batch = board.shape[0]
        stm_scalar = board[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0)
        stm = stm_scalar.view(batch, 1, 1).expand(batch, 1, SQUARES)
        return torch.cat([piece_planes, stm], dim=1).transpose(1, 2).contiguous()

    def _scan_direction(
        self,
        slot: int,
        name: str,
        x_features: torch.Tensor,
    ) -> torch.Tensor:
        """Run the selective SSM along direction ``name``.

        Returns a (B, 64, feature_dim) tensor where every square holds the
        scan-state value produced just *after* reading its own input. For
        squares whose direction track ends before length 8 (variable-length
        diagonals), the value is the final state at that square.
        """
        tracks = self._direction_tracks(name)  # (num_tracks, 8)
        valid_mask = (tracks >= 0).to(dtype=x_features.dtype)  # (num_tracks, 8)
        safe_indices = tracks.clamp(min=0)  # turn -1 into 0; the mask gates it out
        batch = x_features.shape[0]
        # gather: (B, num_tracks, 8, feature_dim)
        gathered = x_features[:, safe_indices.view(-1), :].view(
            batch, tracks.shape[0], tracks.shape[1], self.feature_dim
        )
        # Zero out the padding positions so they cannot drive the scan state.
        gathered = gathered * valid_mask.unsqueeze(0).unsqueeze(-1)

        if self.ablation == "fixed_transition":
            a = torch.sigmoid(self.fixed_a_logits[slot]).view(1, 1, 1, self.feature_dim)
            a = a.expand_as(gathered)
        else:
            a_logits = self.a_logit_projections[slot](gathered)
            a = torch.sigmoid(a_logits)
        b = self.b_projections[slot](gathered)
        # Scan loop. T = max track length = 8.
        T = gathered.shape[2]
        state = torch.zeros(
            batch, tracks.shape[0], self.feature_dim,
            device=x_features.device, dtype=x_features.dtype,
        )
        outputs = []
        for t in range(T):
            mask_t = valid_mask[:, t].unsqueeze(0).unsqueeze(-1)  # (1, num_tracks, 1)
            state = a[:, :, t, :] * state + b[:, :, t, :] * gathered[:, :, t, :]
            state = state * mask_t  # zero state on padding
            outputs.append(state)
        track_states = torch.stack(outputs, dim=2)  # (B, num_tracks, 8, feature_dim)

        # Scatter back to per-square features.
        per_square = torch.zeros(
            batch, SQUARES, self.feature_dim,
            device=x_features.device, dtype=x_features.dtype,
        )
        flat_indices = safe_indices.view(-1)  # (num_tracks * 8,)
        flat_states = track_states.reshape(batch, -1, self.feature_dim)
        flat_mask = valid_mask.view(-1).unsqueeze(0).unsqueeze(-1)  # (1, NT, 1)
        # Use index_add per square to handle duplicates safely (no duplicates
        # since each square belongs to exactly one track per direction, but the
        # API is symmetric).
        contribution = flat_states * flat_mask
        # Use scatter_add_ on the second dim.
        for b in range(batch):
            per_square[b].index_add_(0, flat_indices, contribution[b])
        return per_square

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]
        dtype = board.dtype

        trunk_out = self.trunk(board)
        base_logit = trunk_out["logits"].view(-1)

        descriptor = self._square_descriptor(board).to(dtype=dtype)
        if self.ablation == "shuffle_features" and batch > 1:
            perm = torch.randperm(batch, device=descriptor.device)
            descriptor = descriptor[perm]
        x_features = self.square_feature_proj(descriptor)  # (B, 64, feature_dim)

        if self.ablation == "single_direction":
            active_slots = [0]  # E only
        else:
            active_slots = list(range(NUM_DIRECTIONS))

        # Per-direction selective scan.
        direction_features = []
        per_direction_energy = []
        for slot, (name, _, _) in enumerate(DIRECTION_STEPS):
            if slot in active_slots:
                scan_out = self._scan_direction(slot, name, x_features)
            else:
                scan_out = torch.zeros_like(x_features)
            direction_features.append(scan_out)
            per_direction_energy.append(scan_out.pow(2).mean(dim=(1, 2)).sqrt())
        per_direction_energy_t = torch.stack(per_direction_energy, dim=1)  # (B, 8)
        directional_stack = torch.cat(direction_features, dim=-1)
        # (B, 64, feature_dim * NUM_DIRECTIONS)
        fused = self.direction_fuser(directional_stack)  # (B, 64, head_hidden_dim)

        piece_planes = board[:, :NUM_PIECE_CHANNELS].flatten(2).clamp(0.0, 1.0)
        white_mask = piece_planes[:, :6].sum(dim=1).clamp(0.0, 1.0)
        black_mask = piece_planes[:, 6:].sum(dim=1).clamp(0.0, 1.0)
        stm = board[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0).view(-1, 1)
        own_mask = stm * white_mask + (1.0 - stm) * black_mask
        own_weight = own_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        own_pooled = (fused * own_mask.unsqueeze(-1)).sum(dim=1) / own_weight
        global_pooled = fused.mean(dim=1)
        delta_raw = self.delta_head(torch.cat([own_pooled, global_pooled], dim=1)).view(-1)

        diag_keys = ("gate", "gate_entropy", "mechanism_energy", "stream_disagreement")
        diag = torch.stack([trunk_out[k].detach() for k in diag_keys], dim=1)
        gate_input = torch.cat([diag, per_direction_energy_t], dim=1)
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
        for slot, (name, _, _) in enumerate(DIRECTION_STEPS):
            out[f"oss_energy_{name}"] = per_direction_energy_t[:, slot]
        out["mechanism_energy"] = trunk_out["mechanism_energy"] + per_direction_energy_t.mean(
            dim=1
        ).detach()
        out["proposal_profile_strength"] = primitive_delta.detach().abs() * gate_entropy
        out["proposal_keyword_count"] = logits.new_full((batch,), float(NUM_DIRECTIONS))
        return out


def build_octilinear_selective_scan_from_config(
    config: dict[str, Any],
) -> OctilinearSelectiveScan:
    cfg = dict(config)
    return OctilinearSelectiveScan(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim")),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        feature_dim=int(cfg.get("feature_dim", 16)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )


__all__ = (
    "ALLOWED_ABLATIONS",
    "DIRECTION_STEPS",
    "NUM_DIRECTIONS",
    "OctilinearSelectiveScan",
    "build_octilinear_selective_scan_from_config",
)

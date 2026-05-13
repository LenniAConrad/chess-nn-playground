"""Pair-Resonance Hessian Network (i245) — DHPE primitive integrated with i193.

This module implements the **D**iscrete **H**essian over **P**iece-**E**xistence
primitive (DHPE) as an *additive, gated* side head on the i193 dual-stream
trunk. The primitive computes the second-order mixed forward-difference of a
small learned scoring function `phi_theta` with respect to pairwise
piece-existence indicators:

    H_ij = phi(P) - phi(P \\ {i}) - phi(P \\ {j}) + phi(P \\ {i, j})

The sign of `H_ij` separates super-additive piece pairs (constructive tactics)
from sub-additive ones (defender / blocker substitution). DHPE aggregates the
signed Hessian over a top-K saliency selection of pieces and exposes a 6-d
fingerprint to a discriminator MLP. The result is a primitive delta logit
added to the i193 base logit through a learned sigmoid gate.

Cost model (default top_k=4):

    1 base + K singles + C(K, 2) pair removals = 1 + 4 + 6 = 11 variants
    per position, all routed through a compact `phi_theta` encoder (separate
    from the i193 trunk for cost reasons). With the encoder roughly 1/8th the
    FLOPs of i193, total wall-clock is ~2.5x i193, well under the 10x worst
    case quoted by the spec when `phi_theta` is the full trunk.

Inputs: simple_18 board tensor (B, 18, 8, 8) only. CRTK metadata, tactic
labels, source labels, verification flags, and Stockfish evaluations are
**never** used as model input. Saliency is deterministic from piece-value
priors that are computed from legal board state (the piece type planes), not
from any reporting-only metadata.

The architecture is documented as an additive head and is **not** a trunk
replacement: removing the DHPE module returns the i193 base logit unchanged.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


PIECE_PLANES = 12
SQUARES = 64
PIECE_PLANE_TOTAL = PIECE_PLANES * SQUARES

# Deterministic piece-value priors for saliency selection. Order matches the
# simple_18 piece planes (white P, N, B, R, Q, K; black P, N, B, R, Q, K). The
# king is intentionally weighted 0 because removing a king is not chess-legal,
# and the saliency stage should never pick a king as a "candidate" to delete.
_PIECE_VALUE_PRIORS = (
    1.0, 3.0, 3.2, 5.0, 9.0, 0.0,  # white
    1.0, 3.0, 3.2, 5.0, 9.0, 0.0,  # black
)


@dataclass(frozen=True)
class DHPEOutputs:
    """Container for DHPE primitive intermediates exposed to the caller."""

    delta_phi: torch.Tensor
    base_phi: torch.Tensor
    singles: torch.Tensor
    pairs: torch.Tensor
    hessian: torch.Tensor
    z_pos: torch.Tensor
    z_neg: torch.Tensor
    z_total: torch.Tensor
    z_ratio: torch.Tensor
    z_top1: torch.Tensor
    valid_count: torch.Tensor


class PhiScorer(nn.Module):
    """Compact scalar scorer used for DHPE forward differences.

    Designed to be much cheaper than the i193 trunk so that running it on
    `1 + top_k + C(top_k, 2)` variants per position stays within the scout
    cost envelope. Uses `GroupNorm` instead of `BatchNorm` so that the same
    encoder can score multiple variant copies of the same board without
    leaking statistics across copies.
    """

    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 32,
        depth: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(depth) < 1:
            raise ValueError("PhiScorer depth must be >= 1")
        layers: list[nn.Module] = []
        in_c = int(input_channels)
        for _ in range(int(depth)):
            layers.append(nn.Conv2d(in_c, int(channels), kernel_size=3, padding=1))
            layers.append(nn.GroupNorm(num_groups=min(8, int(channels)), num_channels=int(channels)))
            layers.append(nn.GELU())
            if float(dropout) > 0:
                layers.append(nn.Dropout2d(float(dropout)))
            in_c = int(channels)
        self.body = nn.Sequential(*layers)
        head_in = int(channels) * 2
        self.score = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, max(16, int(channels))),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(max(16, int(channels)), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        return self.score(pooled).view(-1)


def piece_value_saliency(piece_planes: torch.Tensor) -> torch.Tensor:
    """Compute a deterministic per-position saliency tensor.

    Args:
        piece_planes: (B, 12, 8, 8) tensor of piece occupancy planes.

    Returns:
        Flattened saliency tensor of shape (B, 12*64). Each non-zero entry is
        the deterministic priority used to choose top-K critical pieces. King
        positions get score 0 because removing a king is not chess-meaningful.
    """
    if piece_planes.dim() != 4 or piece_planes.shape[1] != PIECE_PLANES:
        raise ValueError(
            f"piece_value_saliency expects (B, 12, 8, 8), got {tuple(piece_planes.shape)}"
        )
    occupancy = piece_planes.clamp(0.0, 1.0)
    values = piece_planes.new_tensor(_PIECE_VALUE_PRIORS).view(1, PIECE_PLANES, 1, 1)
    return (occupancy * values).reshape(piece_planes.shape[0], -1)


def select_top_k_positions(
    piece_planes: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick the top-K piece-occupied positions per batch element.

    Saliency is the piece-value prior weighted by piece occupancy. Slots that
    correspond to empty squares (saliency == 0) are flagged as invalid via the
    second return value so callers can skip masking those slots.

    Returns:
        top_indices: (B, top_k) flattened (channel*64 + square) indices.
        valid: (B, top_k) float mask; 1.0 if the slot points at an actual
            piece, 0.0 otherwise.
    """
    saliency = piece_value_saliency(piece_planes)
    top_values, top_indices = saliency.topk(int(top_k), dim=1)
    valid = (top_values > 0).to(saliency.dtype)
    return top_indices, valid


def _build_variant_remove_masks(
    top_indices: torch.Tensor,
    valid: torch.Tensor,
    top_k: int,
    pair_count: int,
) -> torch.Tensor:
    """Build per-variant remove masks over the flattened piece planes.

    Args:
        top_indices: (B, top_k) flat indices into the 12*64 piece-plane space.
        valid: (B, top_k) {0,1} validity mask for each slot.
        top_k: number of saliency-selected pieces.
        pair_count: number of pairs = top_k * (top_k - 1) / 2.

    Returns:
        remove: (B, variant_count, 12*64) float tensor with 1.0 at positions to
            zero out and 0.0 elsewhere. variant_count = 1 + top_k + pair_count.
            The first slice (variant 0) is the unperturbed board (all zeros).
    """
    batch = top_indices.shape[0]
    device = top_indices.device
    dtype = valid.dtype
    variant_count = 1 + top_k + pair_count
    remove = torch.zeros(batch, variant_count, PIECE_PLANE_TOTAL, device=device, dtype=dtype)
    rng = torch.arange(PIECE_PLANE_TOTAL, device=device).view(1, -1)

    for k in range(top_k):
        idx_k = top_indices[:, k]
        valid_k = valid[:, k]
        match = (rng == idx_k.clamp(min=0).view(-1, 1)).to(dtype)
        remove[:, 1 + k, :] = match * valid_k.view(-1, 1)

    pair_iter = list(itertools.combinations(range(top_k), 2))
    for p, (i, j) in enumerate(pair_iter):
        idx_i = top_indices[:, i]
        idx_j = top_indices[:, j]
        valid_pair = valid[:, i] * valid[:, j]
        match_i = (rng == idx_i.clamp(min=0).view(-1, 1)).to(dtype)
        match_j = (rng == idx_j.clamp(min=0).view(-1, 1)).to(dtype)
        union = (match_i + match_j).clamp(0.0, 1.0)
        remove[:, 1 + top_k + p, :] = union * valid_pair.view(-1, 1)

    return remove


def assemble_variant_boards(
    board: torch.Tensor,
    top_indices: torch.Tensor,
    valid: torch.Tensor,
    top_k: int,
    pair_count: int,
) -> torch.Tensor:
    """Construct (B, V, 18, 8, 8) variant boards by masking saliency-selected pieces.

    The unperturbed copy lives at index 0. Indices 1..top_k zero out one
    piece each (the top-1, top-2, ... pieces by saliency). Indices
    top_k+1..top_k+pair_count zero out a pair of pieces for the lexicographic
    enumeration of `itertools.combinations(range(top_k), 2)`.
    """
    if board.dim() != 4 or board.shape[1] < PIECE_PLANES + 1:
        raise ValueError(
            f"assemble_variant_boards expects (B, 18, 8, 8), got {tuple(board.shape)}"
        )
    batch = board.shape[0]
    variant_count = 1 + top_k + pair_count
    piece_planes = board[:, :PIECE_PLANES]
    other_planes = board[:, PIECE_PLANES:]
    remove_mask = _build_variant_remove_masks(top_indices, valid, top_k, pair_count)
    keep_mask = 1.0 - remove_mask
    pieces_flat = piece_planes.reshape(batch, 1, -1)
    masked_pieces = (pieces_flat * keep_mask).view(
        batch, variant_count, PIECE_PLANES, 8, 8
    )
    other_expanded = other_planes.unsqueeze(1).expand(
        batch, variant_count, other_planes.shape[1], 8, 8
    )
    variants = torch.cat([masked_pieces, other_expanded], dim=2)
    return variants


class DHPEPrimitiveHead(nn.Module):
    """Discrete-Hessian-over-Piece-Existence primitive head.

    Forward pass:

    1. Identify top-K pieces by deterministic piece-value saliency.
    2. Assemble `1 + K + C(K, 2)` board variants by zeroing the selected
       pieces in the piece planes.
    3. Score all variants with a compact shared `PhiScorer`.
    4. Compute `H_ij = phi(P) - phi(P\\i) - phi(P\\j) + phi(P\\{i,j})` for
       each pair.
    5. Reduce the Hessian to a 6-d fingerprint
       (base, z_pos, z_neg, z_total, z_ratio, z_top1) and run a discriminator
       MLP to produce a primitive delta logit.

    Ablations gate the falsifier modes named in the DHPE spec.
    """

    ALLOWED_ABLATIONS = (
        "none",
        "unsigned",
        "no_dhpe",
        "shuffled_pairs",
        "shuffle_singles",
    )

    def __init__(
        self,
        top_k: int = 4,
        phi_channels: int = 32,
        phi_depth: int = 3,
        phi_dropout: float = 0.0,
        hidden_dim: int = 64,
        head_dropout: float = 0.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(top_k) < 2:
            raise ValueError("DHPE top_k must be >= 2 to form at least one pair")
        if str(ablation) not in self.ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown DHPE ablation={ablation!r}; expected one of {sorted(self.ALLOWED_ABLATIONS)}"
            )
        self.top_k = int(top_k)
        self.pair_count = self.top_k * (self.top_k - 1) // 2
        self.variant_count = 1 + self.top_k + self.pair_count
        self.ablation = str(ablation)
        self.spec = BoardTensorSpec(input_channels=18)

        self.phi = PhiScorer(
            input_channels=18,
            channels=int(phi_channels),
            depth=int(phi_depth),
            dropout=float(phi_dropout),
        )
        self.feature_dim = 6
        dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, int(hidden_dim)),
            nn.GELU(),
            dropout_module,
            nn.Linear(int(hidden_dim), 1),
        )
        pair_idx = list(itertools.combinations(range(self.top_k), 2))
        self.register_buffer(
            "pair_i",
            torch.tensor([i for (i, _j) in pair_idx], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "pair_j",
            torch.tensor([j for (_i, j) in pair_idx], dtype=torch.long),
            persistent=False,
        )

    def _compute_phi_grid(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = board.shape[0]
        top_indices, valid = select_top_k_positions(board[:, :PIECE_PLANES], self.top_k)
        variants = assemble_variant_boards(
            board, top_indices, valid, self.top_k, self.pair_count
        )
        flat = variants.reshape(batch * self.variant_count, *board.shape[1:])
        phi = self.phi(flat).view(batch, self.variant_count)
        return phi, top_indices, valid

    def _aggregate(self, phi: torch.Tensor, valid: torch.Tensor) -> DHPEOutputs:
        base = phi[:, 0]
        singles = phi[:, 1 : 1 + self.top_k]
        pairs = phi[:, 1 + self.top_k :]
        if self.ablation == "shuffle_singles":
            perm = torch.randperm(self.top_k, device=phi.device)
            singles = singles[:, perm]
        idx_i = self.pair_i.to(device=phi.device)
        idx_j = self.pair_j.to(device=phi.device)
        single_i = singles[:, idx_i]
        single_j = singles[:, idx_j]
        H = base.unsqueeze(1) - single_i - single_j + pairs
        valid_pair = valid[:, idx_i] * valid[:, idx_j]
        H = H * valid_pair
        if self.ablation == "shuffled_pairs":
            perm = torch.randperm(self.pair_count, device=phi.device)
            H = H[:, perm]

        if self.ablation == "unsigned":
            absH = H.abs()
            z_pos = absH.sum(dim=1)
            z_neg = torch.zeros_like(z_pos)
        else:
            z_pos = torch.relu(H).sum(dim=1)
            z_neg = torch.relu(-H).sum(dim=1)
        z_total = z_pos + z_neg
        z_ratio = z_pos / (z_total + 1.0e-6)
        z_top1 = H.abs().amax(dim=1)
        valid_count = valid.sum(dim=1)

        if self.ablation == "no_dhpe":
            zero = torch.zeros_like(z_pos)
            z_pos = zero
            z_neg = zero
            z_total = zero
            z_ratio = zero
            z_top1 = zero

        feats = torch.stack([base, z_pos, z_neg, z_total, z_ratio, z_top1], dim=1)
        delta = self.head(feats).view(-1)
        return DHPEOutputs(
            delta_phi=delta,
            base_phi=base,
            singles=singles,
            pairs=pairs,
            hessian=H,
            z_pos=z_pos,
            z_neg=z_neg,
            z_total=z_total,
            z_ratio=z_ratio,
            z_top1=z_top1,
            valid_count=valid_count,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        phi, top_indices, valid = self._compute_phi_grid(board)
        outputs = self._aggregate(phi, valid)
        return {
            "delta_phi": outputs.delta_phi,
            "dhpe_base_phi": outputs.base_phi,
            "dhpe_z_pos": outputs.z_pos,
            "dhpe_z_neg": outputs.z_neg,
            "dhpe_z_total": outputs.z_total,
            "dhpe_z_ratio": outputs.z_ratio,
            "dhpe_z_top1": outputs.z_top1,
            "dhpe_valid_count": outputs.valid_count,
            "dhpe_hessian": outputs.hessian,
            "dhpe_top_indices": top_indices,
        }


class PairResonanceHessianNetwork(nn.Module):
    """i245 — Pair-Resonance Hessian Network = i193 trunk + DHPE primitive head.

    Forward pass:

    1. Run the i193 ExchangeThenKingDualStream trunk for the base logit and
       diagnostics (un-perturbed board only).
    2. Run the DHPE primitive head on the same board to get a primitive delta
       logit and a Hessian fingerprint.
    3. A small gate MLP over the trunk diagnostics + DHPE fingerprint produces
       a sigmoid gate; the final logit is

           final_logit = base_logit + gate * primitive_delta

    Ablations cover the four falsifiers required by the DHPE spec (`unsigned`,
    `no_dhpe`, `shuffled_pairs`) plus a couple of useful additional controls
    (`zero_gate`, `shuffle_singles`).
    """

    ALLOWED_ABLATIONS = (
        "none",
        "unsigned",
        "no_dhpe",
        "shuffled_pairs",
        "shuffle_singles",
        "zero_gate",
        "trunk_only",
    )

    _GATE_DIAG_KEYS: tuple[str, ...] = (
        "gate",
        "gate_entropy",
        "mechanism_energy",
        "stream_disagreement",
    )

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
        top_k: int = 4,
        phi_channels: int = 32,
        phi_depth: int = 3,
        phi_dropout: float = 0.0,
        dhpe_hidden_dim: int = 64,
        dhpe_dropout: float = 0.0,
        gate_hidden_dim: int = 32,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "PairResonanceHessianNetwork supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError(
                "PairResonanceHessianNetwork requires the simple_18 board tensor"
            )
        if str(ablation) not in self.ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(self.ALLOWED_ABLATIONS)}"
            )
        self.spec = BoardTensorSpec(input_channels=18)
        self.ablation = str(ablation)
        self.num_classes = 1

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
        dhpe_ablation = ablation if ablation in DHPEPrimitiveHead.ALLOWED_ABLATIONS else "none"
        self.dhpe = DHPEPrimitiveHead(
            top_k=int(top_k),
            phi_channels=int(phi_channels),
            phi_depth=int(phi_depth),
            phi_dropout=float(phi_dropout),
            hidden_dim=int(dhpe_hidden_dim),
            head_dropout=float(dhpe_dropout),
            ablation=dhpe_ablation,
        )

        gate_in = len(self._GATE_DIAG_KEYS) + self.dhpe.feature_dim
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(gate_in),
            nn.Linear(gate_in, int(gate_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(gate_hidden_dim), 1),
        )
        with torch.no_grad():
            self.gate_mlp[-1].bias.fill_(float(gate_init))

    def _collect_dhpe_features(self, dhpe_out: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack(
            [
                dhpe_out["dhpe_base_phi"],
                dhpe_out["dhpe_z_pos"],
                dhpe_out["dhpe_z_neg"],
                dhpe_out["dhpe_z_total"],
                dhpe_out["dhpe_z_ratio"],
                dhpe_out["dhpe_z_top1"],
            ],
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        trunk_out = self.trunk(board)
        base_logit = trunk_out["logits"]

        dhpe_out = self.dhpe(board)
        primitive_delta_raw = dhpe_out["delta_phi"]

        diag_stack = torch.stack(
            [trunk_out[key].detach() for key in self._GATE_DIAG_KEYS],
            dim=1,
        )
        dhpe_features = self._collect_dhpe_features(dhpe_out)
        gate_input = torch.cat([diag_stack, dhpe_features], dim=1)
        gate_logit = self.gate_mlp(gate_input).view(-1)
        gate = torch.sigmoid(gate_logit)

        if self.ablation == "trunk_only":
            primitive_delta = torch.zeros_like(primitive_delta_raw)
            gate_applied = torch.zeros_like(gate)
        elif self.ablation == "zero_gate":
            primitive_delta = primitive_delta_raw
            gate_applied = torch.zeros_like(gate)
        elif self.ablation == "no_dhpe":
            primitive_delta = torch.zeros_like(primitive_delta_raw)
            gate_applied = torch.zeros_like(gate)
        else:
            primitive_delta = primitive_delta_raw
            gate_applied = gate

        primitive_contribution = gate_applied * primitive_delta
        logits = base_logit + primitive_contribution

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
        out["primitive_delta_raw"] = primitive_delta_raw
        out["primitive_gate"] = gate
        out["primitive_gate_applied"] = gate_applied
        out["primitive_gate_logit"] = gate_logit
        out["primitive_gate_entropy"] = gate_entropy
        out["primitive_contribution"] = primitive_contribution
        out["dhpe_base_phi"] = dhpe_out["dhpe_base_phi"]
        out["dhpe_z_pos"] = dhpe_out["dhpe_z_pos"]
        out["dhpe_z_neg"] = dhpe_out["dhpe_z_neg"]
        out["dhpe_z_total"] = dhpe_out["dhpe_z_total"]
        out["dhpe_z_ratio"] = dhpe_out["dhpe_z_ratio"]
        out["dhpe_z_top1"] = dhpe_out["dhpe_z_top1"]
        out["dhpe_valid_count"] = dhpe_out["dhpe_valid_count"]
        out["mechanism_energy"] = trunk_out["mechanism_energy"]
        out["proposal_profile_strength"] = (
            dhpe_out["dhpe_z_total"] * gate_entropy
        )
        out["proposal_keyword_count"] = logits.new_full(
            (logits.shape[0],), float(self.dhpe.feature_dim)
        )
        return out


def build_pair_resonance_hessian_network_from_config(
    config: dict[str, Any],
) -> PairResonanceHessianNetwork:
    cfg = dict(config)
    return PairResonanceHessianNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim")),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        top_k=int(cfg.get("top_k", 4)),
        phi_channels=int(cfg.get("phi_channels", 32)),
        phi_depth=int(cfg.get("phi_depth", 3)),
        phi_dropout=float(cfg.get("phi_dropout", 0.0)),
        dhpe_hidden_dim=int(cfg.get("dhpe_hidden_dim", 64)),
        dhpe_dropout=float(cfg.get("dhpe_dropout", 0.0)),
        gate_hidden_dim=int(cfg.get("gate_hidden_dim", 32)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )


__all__ = (
    "DHPEPrimitiveHead",
    "PairResonanceHessianNetwork",
    "PhiScorer",
    "build_pair_resonance_hessian_network_from_config",
    "piece_value_saliency",
    "select_top_k_positions",
    "assemble_variant_boards",
)


_ = math  # keep math import for future numerical helpers

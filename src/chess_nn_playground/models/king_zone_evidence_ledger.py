"""King-Zone Evidence Ledger model for idea i174.

Faithful implementation of the markdown thesis under
``ideas/registry/i174_king_zone_evidence_ledger/``: a board-only classifier
that maintains a small bank of learned ``evidence ledger`` slots
around each king and reads the puzzle logit out of the comparison
between own-king and opponent-king ledgers.

The packet's central formulation is implemented verbatim:

    own_king_slots:  K x D
    opp_king_slots:  K x D
    global_slots:    K x D

    slot = slot + gated_pool(board_features,
                             piece_features,
                             king_relative_features)

    puzzle_logit = MLP([
        own_king_ledger,
        opp_king_ledger,
        ledger_difference,
        ledger_product,
        global_board_pool,
    ])

Inputs are the repository ``simple_18`` board tensor only. Engine,
verification, source, and CRTK metadata are never consumed. The two
king positions are read off planes 5 (white king) and 11 (black king)
of the ``simple_18`` encoding and re-keyed to ``own`` / ``opp`` using
the side-to-move plane (12). King-relative coordinates (Chebyshev
distance, signed rank/file offsets, in-king-ring flag) are computed
on the fly so the ledger update is anchored to the actual king
location rather than an absolute square index.

The required ablations from the packet are exposed via ``ablation``:

    * ``"none"`` -- main model.
    * ``"no_king_relative"`` -- drop king-relative coordinate features
      so the gated pool sees only board+piece features.
    * ``"random_king_anchor"`` -- replace the real king anchors with a
      deterministic per-batch random anchor so the ledger anchors are
      meaningless. Tests real king semantics.
    * ``"global_slots_only"`` -- drop the per-king ledgers and feed
      only the global slot ledger to the head. Tests king-specific
      ledger value.
    * ``"slot_count_sweep"`` -- not a structural change, exposed as a
      no-op ablation flag so the run is tagged as a sweep entry; the
      sweep itself is driven by the ``num_slots`` config value.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models._packet_bespoke_base import (
    BoardTensorSpec,
    require_board_tensor,
)


_DEFAULT_INPUT_CHANNELS = 18
_DEFAULT_CHANNELS = 64
_DEFAULT_HIDDEN_DIM = 96
_DEFAULT_DEPTH = 2
_DEFAULT_NUM_SLOTS = 5
_DEFAULT_SLOT_DIM = 32
_DEFAULT_DROPOUT = 0.1
_DEFAULT_KING_RING_RADIUS = 2
_RANDOM_ANCHOR_BASE_SEED = 982451653

WHITE_KING_PLANE = 5
BLACK_KING_PLANE = 11
SIDE_TO_MOVE_PLANE = 12

_VALID_ABLATIONS = {
    "none",
    "no_king_relative",
    "random_king_anchor",
    "global_slots_only",
    "slot_count_sweep",
}


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.view(-1) if num_classes == 1 else logits


def _coord_grid(batch: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    rank = torch.arange(height, device=device, dtype=dtype).view(1, 1, height, 1).expand(batch, 1, height, width)
    file = torch.arange(width, device=device, dtype=dtype).view(1, 1, 1, width).expand(batch, 1, height, width)
    return torch.cat([rank, file], dim=1)


def _argmax_square(plane: torch.Tensor) -> torch.Tensor:
    """Return the (rank, file) of the maximum entry of each (B, H, W) plane.

    If a plane is all-zero (e.g. an illegal FEN missing a king) the
    function returns the centre square so the rest of the network is
    still well-defined.
    """
    batch, height, width = plane.shape
    flat = plane.view(batch, -1)
    has_signal = flat.amax(dim=1) > 0.0
    indices = flat.argmax(dim=1)
    rank = (indices // width).to(torch.long)
    file = (indices % width).to(torch.long)
    fallback = torch.full_like(rank, height // 2)
    rank = torch.where(has_signal, rank, fallback)
    fallback_file = torch.full_like(file, width // 2)
    file = torch.where(has_signal, file, fallback_file)
    return torch.stack([rank, file], dim=1)


def _king_anchors(
    board: torch.Tensor,
    *,
    side_to_move_plane: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (own_anchor, opp_anchor, side) where each anchor is (B, 2)."""
    white_king = board[:, WHITE_KING_PLANE]
    black_king = board[:, BLACK_KING_PLANE]
    white_anchor = _argmax_square(white_king)
    black_anchor = _argmax_square(black_king)
    side_field = board[:, side_to_move_plane]
    side = side_field.amax(dim=(1, 2)).clamp(0.0, 1.0)
    side_long = (side > 0.5).long()
    side_b = side_long.unsqueeze(1)
    own_anchor = torch.where(side_b == 1, white_anchor, black_anchor)
    opp_anchor = torch.where(side_b == 1, black_anchor, white_anchor)
    return own_anchor, opp_anchor, side


def _random_anchor(batch: int, seed_offset: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(_RANDOM_ANCHOR_BASE_SEED + int(seed_offset))
    coords = torch.randint(0, 8, (batch, 2), generator=generator, dtype=torch.long)
    return coords.to(device=device)


def _king_relative_features(
    anchor: torch.Tensor,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    king_ring_radius: int = _DEFAULT_KING_RING_RADIUS,
) -> torch.Tensor:
    """Return (B, 5, H, W) king-relative coordinate features.

    Channels are:
        0: signed rank delta in [-1, 1]
        1: signed file delta in [-1, 1]
        2: Chebyshev distance / max_distance
        3: Manhattan distance / max_distance
        4: in-king-ring indicator (Chebyshev <= king_ring_radius)
    """
    batch = anchor.shape[0]
    rank_grid = torch.arange(height, device=device, dtype=dtype).view(1, height, 1).expand(batch, height, width)
    file_grid = torch.arange(width, device=device, dtype=dtype).view(1, 1, width).expand(batch, height, width)
    king_rank = anchor[:, 0].view(batch, 1, 1).to(dtype)
    king_file = anchor[:, 1].view(batch, 1, 1).to(dtype)
    rank_delta = rank_grid - king_rank
    file_delta = file_grid - king_file
    abs_rank = rank_delta.abs()
    abs_file = file_delta.abs()
    cheb = torch.maximum(abs_rank, abs_file)
    manh = abs_rank + abs_file
    max_cheb = float(max(height, width) - 1)
    max_manh = float((height - 1) + (width - 1))
    in_ring = (cheb <= float(king_ring_radius)).to(dtype)
    return torch.stack(
        [
            rank_delta / max(max_cheb, 1.0),
            file_delta / max(max_cheb, 1.0),
            cheb / max(max_cheb, 1.0),
            manh / max(max_manh, 1.0),
            in_ring,
        ],
        dim=1,
    )


class BoardTrunk(nn.Module):
    """Compact convolutional encoder over the configured board planes."""

    def __init__(
        self,
        input_channels: int,
        channels: int,
        depth: int,
        use_batchnorm: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.input_channels = int(input_channels)
        self.channels = int(channels)
        layers: list[nn.Module] = []
        in_ch = input_channels
        for _ in range(depth):
            layers.append(
                nn.Conv2d(in_ch, channels, kernel_size=3, padding=1, bias=not use_batchnorm)
            )
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            in_ch = channels
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(require_board_tensor(x, self.spec))


class EvidenceLedger(nn.Module):
    """A bank of ``num_slots`` learned evidence-ledger slots updated by gated attention.

    The packet's update rule is ``slot = slot + gated_pool(features)``.
    We implement ``gated_pool`` as a slot-conditioned attention over
    every square: each slot ``k`` produces query :math:`q_k` of width
    ``feature_dim``, scores every square via a dot product with the
    feature vector, applies a softmax over the 64 squares, and pools
    the squares with a sigmoid gate. The packet's "K x D" slot bank
    is the learnable initial state ``slots0``; the residual update is
    ``slot_k <- slot_k + gate_k * Linear(pooled_k)`` so each layer
    accumulates evidence rather than overwriting it.
    """

    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        feature_dim: int,
        layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_slots < 1:
            raise ValueError("num_slots must be >= 1")
        if slot_dim < 1:
            raise ValueError("slot_dim must be >= 1")
        if layers < 1:
            raise ValueError("layers must be >= 1")
        self.num_slots = int(num_slots)
        self.slot_dim = int(slot_dim)
        self.feature_dim = int(feature_dim)
        self.layers = int(layers)
        self.slots0 = nn.Parameter(torch.empty(num_slots, slot_dim))
        nn.init.normal_(self.slots0, mean=0.0, std=0.02)
        self.query = nn.Linear(slot_dim, feature_dim)
        self.value = nn.Linear(feature_dim, slot_dim)
        self.gate = nn.Linear(slot_dim, slot_dim)
        self.update_norm = nn.LayerNorm(slot_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, square_features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Update the slot bank from per-square features.

        Args:
            square_features: ``(B, N, F)`` tensor of per-square features
                where ``N`` is typically ``height * width`` and ``F``
                is ``feature_dim``.

        Returns:
            ``{"slots": (B, K, slot_dim), "attention": (B, K, N)}``.
        """
        batch = square_features.shape[0]
        slots = self.slots0.unsqueeze(0).expand(batch, self.num_slots, self.slot_dim).contiguous()
        attention_history: list[torch.Tensor] = []
        for _ in range(self.layers):
            queries = self.query(slots)  # (B, K, F)
            scores = torch.einsum("bkf,bnf->bkn", queries, square_features)
            scores = scores / (self.feature_dim ** 0.5)
            attention = torch.softmax(scores, dim=-1)
            attention_history.append(attention)
            pooled = torch.einsum("bkn,bnf->bkf", attention, square_features)
            update = self.value(pooled)
            gate = torch.sigmoid(self.gate(slots))
            slots = slots + gate * self.dropout(update)
            slots = self.update_norm(slots)
        return {"slots": slots, "attention": attention_history[-1]}


@dataclass(frozen=True)
class KingZoneEvidenceLedgerConfig:
    input_channels: int = _DEFAULT_INPUT_CHANNELS
    num_classes: int = 1
    channels: int = _DEFAULT_CHANNELS
    hidden_dim: int = _DEFAULT_HIDDEN_DIM
    depth: int = _DEFAULT_DEPTH
    num_slots: int = _DEFAULT_NUM_SLOTS
    slot_dim: int = _DEFAULT_SLOT_DIM
    ledger_layers: int = 2
    dropout: float = _DEFAULT_DROPOUT
    use_batchnorm: bool = True
    king_ring_radius: int = _DEFAULT_KING_RING_RADIUS
    ablation: str = "none"


class KingZoneEvidenceLedger(nn.Module):
    """King-Zone Evidence Ledger classifier for ``puzzle_binary``.

    1. ``BoardTrunk`` turns the 18-plane board into a ``(B, C, 8, 8)``
       feature map.
    2. King anchors are read off planes 5 (white king) and 11 (black
       king) and re-keyed to ``own`` / ``opp`` using the side-to-move
       plane. King-relative coordinate features are computed on the fly.
    3. Three ``EvidenceLedger`` banks (``own_king``, ``opp_king``, and
       ``global``) update their ``num_slots`` learned slots via
       slot-conditioned gated attention over the per-square feature
       map. The own/opp ledgers see king-relative features
       concatenated to the trunk; the global ledger sees only the
       trunk features and a constant zero coordinate field.
    4. Readout: ``MLP([own_king_ledger, opp_king_ledger,
       ledger_difference, ledger_product, global_board_pool])`` emits
       the puzzle logit (``num_classes == 1``).

    Forward returns a dict whose ``logits`` entry has shape ``(B,)``
    for the repository ``puzzle_binary`` BCE-with-logits trainer,
    plus diagnostics that expose ledger energies, attention spread,
    and the king-zone vs global decomposition.
    """

    VALID_ABLATIONS = _VALID_ABLATIONS

    def __init__(
        self,
        input_channels: int = _DEFAULT_INPUT_CHANNELS,
        num_classes: int = 1,
        channels: int = _DEFAULT_CHANNELS,
        hidden_dim: int = _DEFAULT_HIDDEN_DIM,
        depth: int = _DEFAULT_DEPTH,
        num_slots: int = _DEFAULT_NUM_SLOTS,
        slot_dim: int = _DEFAULT_SLOT_DIM,
        ledger_layers: int = 2,
        dropout: float = _DEFAULT_DROPOUT,
        use_batchnorm: bool = True,
        king_ring_radius: int = _DEFAULT_KING_RING_RADIUS,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if ablation not in _VALID_ABLATIONS:
            raise ValueError(f"Unknown ablation: {ablation}")
        if num_slots < 1:
            raise ValueError("num_slots must be >= 1")
        if slot_dim < 1:
            raise ValueError("slot_dim must be >= 1")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if king_ring_radius < 0:
            raise ValueError("king_ring_radius must be >= 0")

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.num_slots = int(num_slots)
        self.slot_dim = int(slot_dim)
        self.ledger_layers = int(ledger_layers)
        self.dropout_rate = float(dropout)
        self.use_batchnorm = bool(use_batchnorm)
        self.king_ring_radius = int(king_ring_radius)
        self.ablation = ablation
        self.config = KingZoneEvidenceLedgerConfig(
            input_channels=int(input_channels),
            num_classes=int(num_classes),
            channels=int(channels),
            hidden_dim=int(hidden_dim),
            depth=int(depth),
            num_slots=int(num_slots),
            slot_dim=int(slot_dim),
            ledger_layers=int(ledger_layers),
            dropout=float(dropout),
            use_batchnorm=bool(use_batchnorm),
            king_ring_radius=int(king_ring_radius),
            ablation=ablation,
        )

        self.trunk = BoardTrunk(
            input_channels=input_channels,
            channels=channels,
            depth=depth,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )

        self.use_king_relative = ablation != "no_king_relative"
        self.use_random_anchor = ablation == "random_king_anchor"
        self.use_global_only = ablation == "global_slots_only"

        # 5 king-relative coord channels: rank/file delta, Chebyshev,
        # Manhattan, in-king-ring flag.
        king_relative_dim = 5
        feature_dim = channels + (king_relative_dim if self.use_king_relative else 0)

        self.own_ledger = EvidenceLedger(
            num_slots=num_slots,
            slot_dim=slot_dim,
            feature_dim=feature_dim,
            layers=ledger_layers,
            dropout=dropout,
        )
        self.opp_ledger = EvidenceLedger(
            num_slots=num_slots,
            slot_dim=slot_dim,
            feature_dim=feature_dim,
            layers=ledger_layers,
            dropout=dropout,
        )
        self.global_ledger = EvidenceLedger(
            num_slots=num_slots,
            slot_dim=slot_dim,
            feature_dim=feature_dim,
            layers=ledger_layers,
            dropout=dropout,
        )

        flat_dim = num_slots * slot_dim
        readout_dim = (
            flat_dim if self.use_global_only else 4 * flat_dim + flat_dim
        )

        self.head = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_classes),
        )

    def _build_square_features(
        self,
        trunk: torch.Tensor,
        anchor: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, _, height, width = trunk.shape
        features = trunk
        if self.use_king_relative and anchor is not None:
            relative = _king_relative_features(
                anchor=anchor,
                height=height,
                width=width,
                device=trunk.device,
                dtype=trunk.dtype,
                king_ring_radius=self.king_ring_radius,
            )
            features = torch.cat([features, relative], dim=1)
        elif self.use_king_relative:
            zeros = trunk.new_zeros(batch, 5, height, width)
            features = torch.cat([features, zeros], dim=1)
        flat = features.flatten(2).transpose(1, 2)  # (B, N, F)
        return flat

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]
        trunk = self.trunk(board)
        height = trunk.shape[2]
        width = trunk.shape[3]

        own_anchor, opp_anchor, side = _king_anchors(
            board, side_to_move_plane=SIDE_TO_MOVE_PLANE
        )
        if self.use_random_anchor:
            own_anchor_used = _random_anchor(batch, seed_offset=0, device=trunk.device)
            opp_anchor_used = _random_anchor(batch, seed_offset=1, device=trunk.device)
        else:
            own_anchor_used = own_anchor
            opp_anchor_used = opp_anchor

        own_features = self._build_square_features(trunk, own_anchor_used)
        opp_features = self._build_square_features(trunk, opp_anchor_used)
        global_features = self._build_square_features(trunk, anchor=None)

        own_packet = self.own_ledger(own_features)
        opp_packet = self.opp_ledger(opp_features)
        global_packet = self.global_ledger(global_features)

        own_slots = own_packet["slots"]
        opp_slots = opp_packet["slots"]
        global_slots = global_packet["slots"]

        own_flat = own_slots.flatten(1)
        opp_flat = opp_slots.flatten(1)
        diff_flat = (own_slots - opp_slots).flatten(1)
        prod_flat = (own_slots * opp_slots).flatten(1)
        global_flat = global_slots.flatten(1)

        if self.use_global_only:
            readout = global_flat
        else:
            readout = torch.cat([own_flat, opp_flat, diff_flat, prod_flat, global_flat], dim=1)

        raw_logits = self.head(readout)
        logits = _format_logits(raw_logits, self.num_classes)

        own_energy = own_slots.square().mean(dim=(1, 2))
        opp_energy = opp_slots.square().mean(dim=(1, 2))
        global_energy = global_slots.square().mean(dim=(1, 2))
        own_minus_opp = own_energy - opp_energy
        own_attention = own_packet["attention"]
        opp_attention = opp_packet["attention"]
        global_attention = global_packet["attention"]

        eps = 1e-8
        own_attn_entropy = -(own_attention.clamp(min=eps).log() * own_attention).sum(dim=-1).mean(dim=-1)
        opp_attn_entropy = -(opp_attention.clamp(min=eps).log() * opp_attention).sum(dim=-1).mean(dim=-1)
        global_attn_entropy = -(global_attention.clamp(min=eps).log() * global_attention).sum(dim=-1).mean(dim=-1)

        own_anchor_f = own_anchor.to(logits.dtype)
        opp_anchor_f = opp_anchor.to(logits.dtype)
        own_anchor_used_f = own_anchor_used.to(logits.dtype)
        opp_anchor_used_f = opp_anchor_used.to(logits.dtype)
        ring_pressure = self._king_ring_attention(own_attention, own_anchor, height, width)
        opp_ring_pressure = self._king_ring_attention(opp_attention, opp_anchor, height, width)

        scalar_shape = logits.shape if self.num_classes == 1 else logits.shape[:1]
        diagnostics: dict[str, torch.Tensor] = {
            "logits": logits,
            "trunk_features": trunk,
            "own_king_ledger": own_slots,
            "opp_king_ledger": opp_slots,
            "global_ledger": global_slots,
            "ledger_difference": own_slots - opp_slots,
            "ledger_product": own_slots * opp_slots,
            "own_king_energy": own_energy,
            "opp_king_energy": opp_energy,
            "global_energy": global_energy,
            "own_minus_opp_energy": own_minus_opp,
            "own_attention": own_attention,
            "opp_attention": opp_attention,
            "global_attention": global_attention,
            "own_attention_entropy": own_attn_entropy,
            "opp_attention_entropy": opp_attn_entropy,
            "global_attention_entropy": global_attn_entropy,
            "own_king_ring_pressure": ring_pressure,
            "opp_king_ring_pressure": opp_ring_pressure,
            "own_anchor_rank": own_anchor_f[:, 0],
            "own_anchor_file": own_anchor_f[:, 1],
            "opp_anchor_rank": opp_anchor_f[:, 0],
            "opp_anchor_file": opp_anchor_f[:, 1],
            "own_anchor_rank_used": own_anchor_used_f[:, 0],
            "own_anchor_file_used": own_anchor_used_f[:, 1],
            "opp_anchor_rank_used": opp_anchor_used_f[:, 0],
            "opp_anchor_file_used": opp_anchor_used_f[:, 1],
            "side_to_move": side.to(logits.dtype),
            "num_slots_levels": logits.new_full(scalar_shape, float(self.num_slots)),
            "slot_dim_levels": logits.new_full(scalar_shape, float(self.slot_dim)),
            "ledger_layers_levels": logits.new_full(scalar_shape, float(self.ledger_layers)),
            "ablation_active": logits.new_full(
                scalar_shape, 0.0 if self.ablation == "none" else 1.0
            ),
            "uses_king_relative": logits.new_full(
                scalar_shape, 1.0 if self.use_king_relative else 0.0
            ),
            "uses_random_anchor": logits.new_full(
                scalar_shape, 1.0 if self.use_random_anchor else 0.0
            ),
            "uses_global_only": logits.new_full(
                scalar_shape, 1.0 if self.use_global_only else 0.0
            ),
        }
        if self.num_classes == 1:
            diagnostics["prob"] = torch.sigmoid(logits)
        return diagnostics

    def _king_ring_attention(
        self,
        attention: torch.Tensor,
        anchor: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Mean attention mass that lands inside the king ring.

        Args:
            attention: ``(B, K, N)`` attention weights.
            anchor: ``(B, 2)`` king square (rank, file).

        Returns ``(B,)`` tensor in ``[0, 1]``.
        """
        batch = attention.shape[0]
        rank_grid = torch.arange(height, device=attention.device).view(1, height, 1).expand(batch, height, width)
        file_grid = torch.arange(width, device=attention.device).view(1, 1, width).expand(batch, height, width)
        anchor_rank = anchor[:, 0].view(batch, 1, 1).to(attention.device)
        anchor_file = anchor[:, 1].view(batch, 1, 1).to(attention.device)
        cheb = torch.maximum((rank_grid - anchor_rank).abs(), (file_grid - anchor_file).abs())
        ring_mask = (cheb <= self.king_ring_radius).to(attention.dtype).flatten(1)  # (B, N)
        ring_mass = (attention * ring_mask.unsqueeze(1)).sum(dim=-1)  # (B, K)
        return ring_mass.mean(dim=-1)


def build_king_zone_evidence_ledger_from_config(
    config: dict[str, Any],
) -> KingZoneEvidenceLedger:
    cfg = dict(config)
    return KingZoneEvidenceLedger(
        input_channels=int(cfg.get("input_channels", _DEFAULT_INPUT_CHANNELS)),
        num_classes=int(cfg.get("num_classes", 1)),
        channels=int(cfg.get("channels", _DEFAULT_CHANNELS)),
        hidden_dim=int(cfg.get("hidden_dim", _DEFAULT_HIDDEN_DIM)),
        depth=int(cfg.get("depth", _DEFAULT_DEPTH)),
        num_slots=int(cfg.get("num_slots", _DEFAULT_NUM_SLOTS)),
        slot_dim=int(cfg.get("slot_dim", _DEFAULT_SLOT_DIM)),
        ledger_layers=int(cfg.get("ledger_layers", 2)),
        dropout=float(cfg.get("dropout", _DEFAULT_DROPOUT)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        king_ring_radius=int(cfg.get("king_ring_radius", _DEFAULT_KING_RING_RADIUS)),
        ablation=str(cfg.get("ablation", "none")),
    )

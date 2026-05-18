"""Controlled-encoding i018 variant for idea i253.

This module promotes the research markdown
`ideas/research/packets/classic/i253_i018_bt4_112_controlled_encoding.md`
into a bespoke i253 architecture. The thesis is that BT4-style 112-channel
encoding should be tested as a controlled additive benefit on top of the
existing i018 sheaf trunk, with exact chess relation masks always kept
load-bearing.

The module wraps the existing i018 building blocks (board adapter, exact
incidence builder, square encoder, sheaf diffusion block, triad pool,
readout) and adds:

* A controlled adapter that emits a 112-channel raw pathway for both
  `simple_18` (padded) and `lc0_bt4_112` (native), so the raw input width
  is identical across encodings.
* An optional `TacticalIncidenceAugmentationBuilder` that returns
  relation-specific geometric template supersets ``T_r`` for use by the
  hybrid relation mode.
* A `RelationConfidenceHead` that consumes the raw 112-channel square
  features and emits per-relation source/target codes plus an optional
  augmentation head. The hybrid mode forms the controlled sheaf weights

  ``W_r = clamp(M_r * sigmoid(C_r) + lambda_aug * T_r * sigmoid(A_r), 0, 1)``.

Three `relation_mode` values are supported:

* ``exact``: ``W_r = M_r`` (hard control - no learned edges, only richer
  raw-input pathway).
* ``confidence``: ``W_r = M_r * sigmoid(C_r)`` (exact support, learned
  edge weights).
* ``hybrid``: exact support plus bounded augmentation on a fixed
  geometric superset.

All three modes share the same trunk shape, same readout, and same
diagnostic contract as i018, so the parameter count delta is small and
the comparison stays interpretable.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor
from chess_nn_playground.models.trunk.oriented_tactical_sheaf import (
    RELATION_NAMES,
    BoardState,
    BoardStateAdapter,
    SheafDiffusionBlock,
    SquareTokenEncoder,
    TacticalIncidence,
    TacticalIncidenceBuilder,
    TriadDefectPool,
    _format_logits,
    _weighted_mean,
)


RELATION_COUNT = len(RELATION_NAMES)


def _augmentation_templates_for(masks: TacticalIncidenceBuilder) -> torch.Tensor:
    """Build relation-specific geometric template supersets ``T_r``.

    Templates are static (per-relation, not per-batch) and live as a single
    ``(R, 64, 64)`` tensor registered as a buffer. The hybrid relation mode
    multiplies them per-batch by the corresponding learned augmentation
    sigmoid, then by ``lambda_aug``.

    The supersets follow the research packet's discipline:

    * slider relations (rook, bishop, queen) get the full geometric ray
      template (before blocker gating);
    * attacker/defender relations get the slider geometric superset only,
      so augmentation is only allowed where occupancy ambiguity is the
      contested variable;
    * knight, king, pawn, and pin-candidate relations use their legal
      geometric shape (no extra augmentation room).
    """

    rook = masks.rook_ray
    bishop = masks.bishop_ray
    knight = masks.knight
    king = masks.king
    our_pawn = masks.our_pawn
    their_pawn = masks.their_pawn
    slider_super = (rook + bishop).clamp(0.0, 1.0)
    pawn_super = (our_pawn + their_pawn).clamp(0.0, 1.0)
    templates = torch.stack(
        [
            slider_super,  # us_attacks_them_piece — slider geometry only
            slider_super,  # them_attacks_us_piece
            slider_super,  # us_defends_us_piece
            slider_super,  # them_defends_them_piece
            slider_super,  # us_attacks_empty_near_king
            slider_super,  # them_attacks_empty_near_king
            bishop,         # bishop_ray_visible
            rook,           # rook_ray_visible
            (rook + bishop).clamp(0.0, 1.0),  # queen_ray_visible
            knight,         # knight_attack — fixed shape
            pawn_super,     # pawn_attack_forward_oriented — fixed shape
            slider_super,  # king_ray_pin_candidate — slider geometry
        ],
        dim=0,
    ).clamp(0.0, 1.0)
    return templates


class BoardStateAdapterControlled(nn.Module):
    """i018 adapter wrapper that emits a 112-channel raw pathway.

    The piece-state pathway stays exact (mover-relative pieces from the
    chosen encoding, never a learned probe). The raw pathway is padded to
    112 channels for `simple_18` and used natively for `lc0_bt4_112`. The
    intent is to make the raw input shape identical across encodings so
    the parameter count and architecture are matched.
    """

    def __init__(self, encoding: str = "simple_18") -> None:
        super().__init__()
        self.encoding = str(encoding)
        if self.encoding not in {"simple_18", "lc0_bt4_112", "lc0_static_112"}:
            raise ValueError(
                "BoardStateAdapterControlled supports encoding in "
                "{'simple_18', 'lc0_bt4_112', 'lc0_static_112'}; "
                f"got {self.encoding!r}"
            )
        native_channels = 18 if self.encoding == "simple_18" else 112
        self._base_adapter = BoardStateAdapter(
            input_channels=native_channels,
            encoding=self.encoding,
            piece_adapter="exact",
        )
        self.native_channels = native_channels
        self.controlled_channels = 112

    def forward(self, x: torch.Tensor) -> BoardState:
        if x.shape[1] != self.native_channels:
            raise ValueError(
                f"BoardStateAdapterControlled expected {self.native_channels} "
                f"channels for encoding {self.encoding!r}, got {x.shape[1]}"
            )
        base = self._base_adapter(x)
        if self.native_channels == self.controlled_channels:
            return base
        batch, squares, _ = base.square_raw.shape
        padded = base.square_raw.new_zeros(batch, squares, self.controlled_channels)
        padded[..., : self.native_channels] = base.square_raw
        return BoardState(
            square_raw=padded,
            piece_state=base.piece_state,
            occupancy=base.occupancy,
            side_info=base.side_info,
        )


class TacticalIncidenceAugmentationBuilder(nn.Module):
    """Pairs exact i018 masks with relation-specific augmentation templates.

    Exact masks are produced by the original i018 ``TacticalIncidenceBuilder``
    so the chess geometry stays load-bearing. Templates are static and
    registered once at module construction.
    """

    def __init__(self) -> None:
        super().__init__()
        self.exact = TacticalIncidenceBuilder()
        templates = _augmentation_templates_for(self.exact)
        self.register_buffer("templates", templates, persistent=False)

    def forward(self, piece_state: torch.Tensor, occupancy: torch.Tensor) -> TacticalIncidence:
        return self.exact(piece_state, occupancy)


class RelationConfidenceHead(nn.Module):
    """Low-rank per-relation confidence and optional augmentation head.

    The head consumes the raw ``(B, 64, 112)`` square tensor. It produces:

    * per-relation source codes ``S_r`` and target codes ``D_r`` of rank
      ``relation_rank`` plus a global relation bias, so that a logit
      ``C_r[i, j] = <S_r[i], D_r[j]> + b_r`` can be applied wherever
      ``M_r[i, j] = 1``;
    * optionally, a symmetric augmentation logit ``A_r[i, j]`` built the
      same way, so that the hybrid mode can decide how strongly to use
      the geometric superset.

    No metadata, labels, or reporting-only fields are consumed; the head
    only sees the raw square tensor.
    """

    def __init__(
        self,
        raw_channels: int = 112,
        relation_count: int = RELATION_COUNT,
        relation_hidden: int = 16,
        relation_rank: int = 8,
        augmentation: bool = False,
    ) -> None:
        super().__init__()
        self.raw_channels = int(raw_channels)
        self.relation_count = int(relation_count)
        self.relation_hidden = int(relation_hidden)
        self.relation_rank = int(relation_rank)
        self.augmentation = bool(augmentation)
        self.pre_proj = nn.Sequential(
            nn.Linear(self.raw_channels, self.relation_hidden),
            nn.GELU(),
            nn.LayerNorm(self.relation_hidden),
        )
        self.confidence_source = nn.Linear(
            self.relation_hidden, self.relation_count * self.relation_rank
        )
        self.confidence_target = nn.Linear(
            self.relation_hidden, self.relation_count * self.relation_rank
        )
        self.confidence_bias = nn.Parameter(torch.zeros(self.relation_count))
        if self.augmentation:
            self.augmentation_source = nn.Linear(
                self.relation_hidden, self.relation_count * self.relation_rank
            )
            self.augmentation_target = nn.Linear(
                self.relation_hidden, self.relation_count * self.relation_rank
            )
            self.augmentation_bias = nn.Parameter(torch.zeros(self.relation_count))

    def _pair_logits(
        self,
        features: torch.Tensor,
        source: nn.Linear,
        target: nn.Linear,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        batch, squares, _ = features.shape
        src = source(features).view(batch, squares, self.relation_count, self.relation_rank)
        dst = target(features).view(batch, squares, self.relation_count, self.relation_rank)
        # logits[b, r, i, j] = <src[b, i, r], dst[b, j, r]> + bias[r]
        logits = torch.einsum("bird,bjrd->brij", src, dst)
        return logits + bias.view(1, self.relation_count, 1, 1)

    def forward(self, square_raw: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.pre_proj(square_raw)
        outputs: dict[str, torch.Tensor] = {
            "confidence": self._pair_logits(
                features, self.confidence_source, self.confidence_target, self.confidence_bias
            )
        }
        if self.augmentation:
            outputs["augmentation"] = self._pair_logits(
                features,
                self.augmentation_source,
                self.augmentation_target,
                self.augmentation_bias,
            )
        return outputs


class OrientedTacticalSheafControlledEncodingNet(nn.Module):
    """i018 trunk with a controlled BT4-vs-simple18 encoding pathway.

    Modes:

    * ``exact``: relations are the exact i018 masks ``M_r``; the raw
      pathway is the only place the richer BT4 encoding can help.
    * ``confidence``: ``W_r = M_r * sigmoid(C_r)``; no new edges, just
      learned per-edge importance on exact chess geometry.
    * ``hybrid``: ``W_r = clamp(M_r * sigmoid(C_r) + lambda * T_r *
      sigmoid(A_r), 0, 1)``; bounded augmentation on a fixed geometric
      superset (slider geometry for sliders, knight/pawn shape for
      knight/pawn relations, etc.).

    Two falsifier switches preserve i018's spirit:

    * ``scramble_exact_relations``: degree-preserving random rewiring of
      ``M_r`` for all three modes (kept identical to i018's existing
      falsifier).
    * ``augmentation_only``: forces the hybrid mode to drop ``M_r``
      entirely so only the bounded template augmentation survives. This
      is the explicit augmentation-only falsifier required by the
      research markdown.
    """

    def __init__(
        self,
        encoding: str = "simple_18",
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        stalk_dim: int = 8,
        dropout: float = 0.1,
        use_triads: bool = True,
        relation_mode: str = "exact",
        relation_hidden: int = 16,
        relation_rank: int = 8,
        augmentation_lambda: float = 0.25,
        scramble_exact_relations: bool = False,
        augmentation_only: bool = False,
    ) -> None:
        super().__init__()
        relation_mode = str(relation_mode).lower()
        if relation_mode not in {"exact", "confidence", "hybrid"}:
            raise ValueError(
                "relation_mode must be one of {'exact', 'confidence', 'hybrid'}; "
                f"got {relation_mode!r}"
            )
        encoding = str(encoding)
        if encoding not in {"simple_18", "lc0_bt4_112", "lc0_static_112"}:
            raise ValueError(
                "encoding must be one of {'simple_18', 'lc0_bt4_112', 'lc0_static_112'}; "
                f"got {encoding!r}"
            )
        native_channels = 18 if encoding == "simple_18" else 112
        controlled_channels = 112
        self.spec = BoardTensorSpec(input_channels=native_channels)
        self.encoding = encoding
        self.input_channels = native_channels
        self.controlled_channels = controlled_channels
        self.num_classes = int(num_classes)
        self.relation_mode = relation_mode
        self.augmentation_lambda = float(augmentation_lambda)
        self.scramble_exact_relations = bool(scramble_exact_relations)
        self.augmentation_only = bool(augmentation_only)
        self.relation_names = RELATION_NAMES

        self.adapter = BoardStateAdapterControlled(encoding=encoding)
        self.incidence = TacticalIncidenceAugmentationBuilder()
        self.encoder = SquareTokenEncoder(
            input_channels=controlled_channels, d_model=channels, dropout=dropout
        )
        self.blocks = nn.ModuleList(
            [
                SheafDiffusionBlock(channels, RELATION_COUNT, stalk_dim, dropout)
                for _ in range(max(1, int(depth)))
            ]
        )
        self.triad_pool = TriadDefectPool(channels, dropout) if use_triads else None
        triad_dim = self.triad_pool.output_dim if self.triad_pool is not None else 0
        board_stats_dim = 8
        readout_dim = channels * 4 + RELATION_COUNT * 4 + triad_dim + board_stats_dim
        self.head = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes),
        )
        if relation_mode == "exact":
            self.relation_head = None
        else:
            self.relation_head = RelationConfidenceHead(
                raw_channels=controlled_channels,
                relation_count=RELATION_COUNT,
                relation_hidden=int(relation_hidden),
                relation_rank=int(relation_rank),
                augmentation=relation_mode == "hybrid",
            )

    def _board_stats(self, board: BoardState, incidence: TacticalIncidence) -> torch.Tensor:
        occupancy = board.occupancy
        rank_counts = torch.matmul(occupancy, self.incidence.exact.rank_one_hot)
        file_counts = torch.matmul(occupancy, self.incidence.exact.file_one_hot)
        return torch.stack(
            [
                occupancy.mean(dim=1),
                incidence.our_piece.sum(dim=1) / 16.0,
                incidence.them_piece.sum(dim=1) / 16.0,
                incidence.our_attack.mean(dim=(1, 2)),
                incidence.them_attack.mean(dim=(1, 2)),
                incidence.pin_mask.mean(dim=(1, 2)),
                rank_counts.std(dim=1),
                file_counts.std(dim=1),
            ],
            dim=1,
        )

    def _scramble_masks(self, masks: torch.Tensor) -> torch.Tensor:
        batch, relations, squares, _ = masks.shape
        perm = torch.argsort(torch.rand(batch, relations, squares, device=masks.device), dim=-1)
        perm_expanded = perm.unsqueeze(-2).expand(-1, -1, squares, -1)
        return torch.gather(masks, dim=-1, index=perm_expanded)

    def _resolve_sheaf_masks(
        self,
        exact_masks: torch.Tensor,
        square_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        diagnostics: dict[str, torch.Tensor] = {}
        if self.scramble_exact_relations:
            exact_masks = self._scramble_masks(exact_masks)

        if self.relation_mode == "exact" or self.relation_head is None:
            return exact_masks, diagnostics

        head_out = self.relation_head(square_raw)
        confidence = torch.sigmoid(head_out["confidence"])
        diagnostics["relation_confidence_mean"] = confidence.mean(dim=(2, 3))
        confidence_masks = exact_masks * confidence
        if self.relation_mode == "confidence":
            return confidence_masks.clamp(0.0, 1.0), diagnostics

        # hybrid: bounded augmentation on the relation-specific template
        templates = self.incidence.templates.to(dtype=exact_masks.dtype)
        templates = templates.unsqueeze(0).expand(exact_masks.shape[0], -1, -1, -1)
        if self.scramble_exact_relations:
            templates = self._scramble_masks(templates)
        augmentation = torch.sigmoid(head_out["augmentation"])
        diagnostics["relation_augmentation_mean"] = augmentation.mean(dim=(2, 3))
        augmentation_masks = templates * augmentation
        if self.augmentation_only:
            return (self.augmentation_lambda * augmentation_masks).clamp(0.0, 1.0), diagnostics
        hybrid = confidence_masks + self.augmentation_lambda * augmentation_masks
        return hybrid.clamp(0.0, 1.0), diagnostics

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        board = self.adapter(x)
        incidence = self.incidence(board.piece_state, board.occupancy)
        sheaf_masks, relation_diagnostics = self._resolve_sheaf_masks(
            incidence.relation_masks, board.square_raw
        )
        h = self.encoder(board.square_raw, board.piece_state)

        block_energies: list[torch.Tensor] = []
        block_gates: list[torch.Tensor] = []
        for block in self.blocks:
            h, energy, gates = block(h, sheaf_masks)
            block_energies.append(energy)
            block_gates.append(gates.unsqueeze(0).expand(x.shape[0], -1))

        energy_stack = torch.stack(block_energies, dim=1)
        gate_stack = torch.stack(block_gates, dim=1)
        energy_mean = energy_stack.mean(dim=1)
        energy_max = energy_stack.amax(dim=1)
        gate_mean = gate_stack.mean(dim=1)
        triad_stats = (
            self.triad_pool(h, incidence)
            if self.triad_pool is not None
            else h.new_zeros(h.shape[0], 0)
        )
        readout = torch.cat(
            [
                h.mean(dim=1),
                h.amax(dim=1),
                _weighted_mean(h, incidence.our_piece),
                _weighted_mean(h, incidence.them_piece),
                energy_mean,
                energy_max,
                incidence.relation_density,
                gate_mean,
                triad_stats,
                self._board_stats(board, incidence),
            ],
            dim=1,
        )
        logits = _format_logits(self.head(readout), self.num_classes)
        sheaf_tension = energy_stack.mean(dim=(1, 2))
        us_pressure = incidence.relation_masks[:, 0].sum(dim=(1, 2))
        them_pressure = incidence.relation_masks[:, 1].sum(dim=(1, 2))
        us_defense = incidence.relation_masks[:, 2].sum(dim=(1, 2))
        them_defense = incidence.relation_masks[:, 3].sum(dim=(1, 2))
        rank_counts = torch.matmul(board.occupancy, self.incidence.exact.rank_one_hot)
        file_counts = torch.matmul(board.occupancy, self.incidence.exact.file_one_hot)
        piece_entropy = -(board.piece_state * board.piece_state.clamp_min(1e-8).log()).sum(dim=-1).mean(dim=1)
        diagnostics = {
            "logits": logits,
            "mechanism_energy": torch.log1p(sheaf_tension),
            "proposal_profile_strength": gate_mean.mean(dim=1),
            "proposal_keyword_count": logits.new_full((x.shape[0],), 4.0),
            "sheaf_tension": sheaf_tension,
            "transport_imbalance": (us_pressure - them_pressure).abs() / (us_pressure + them_pressure).clamp_min(1.0),
            "symmetry_residual": (
                incidence.our_attack.mean(dim=(1, 2)) - incidence.them_attack.mean(dim=(1, 2))
            ).abs(),
            "topology_pressure": incidence.relation_density.mean(dim=1),
            "ray_language_energy": energy_mean[:, 6:9].mean(dim=1),
            "information_surprisal": piece_entropy,
            "sparse_certificate_energy": energy_stack.amax(dim=(1, 2)),
            "rank_file_imbalance": (rank_counts.std(dim=1) - file_counts.std(dim=1)).abs(),
            "king_ring_pressure": incidence.relation_density[:, 4] + incidence.relation_density[:, 5],
            "reply_pressure": 0.5 * (us_pressure + them_pressure) / 64.0,
            "defense_gap": ((us_pressure + them_pressure) - (us_defense + them_defense)) / 64.0,
            "triad_defect_energy": triad_stats[:, 0] if triad_stats.numel() else logits.new_zeros(x.shape[0]),
            "pin_pressure": incidence.relation_density[:, 11],
        }
        if "relation_confidence_mean" in relation_diagnostics:
            diagnostics["controlled_confidence_mean"] = relation_diagnostics["relation_confidence_mean"].mean(dim=1)
        if "relation_augmentation_mean" in relation_diagnostics:
            diagnostics["controlled_augmentation_mean"] = relation_diagnostics["relation_augmentation_mean"].mean(dim=1)
        return diagnostics


def build_i018_bt4_112_controlled_encoding_from_config(
    config: dict[str, Any],
) -> OrientedTacticalSheafControlledEncodingNet:
    encoding = str(config.get("encoding", "simple_18"))
    return OrientedTacticalSheafControlledEncodingNet(
        encoding=encoding,
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        depth=int(config.get("sheaf_layers", config.get("depth", 2))),
        stalk_dim=int(config.get("stalk_dim", 8)),
        dropout=float(config.get("dropout", 0.1)),
        use_triads=bool(config.get("use_triads", True)),
        relation_mode=str(config.get("relation_mode", "exact")),
        relation_hidden=int(config.get("relation_hidden", 16)),
        relation_rank=int(config.get("relation_rank", 8)),
        augmentation_lambda=float(config.get("augmentation_lambda", 0.25)),
        scramble_exact_relations=bool(config.get("scramble_exact_relations", False)),
        augmentation_only=bool(config.get("augmentation_only", False)),
    )

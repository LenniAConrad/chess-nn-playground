"""Counterfactual Defender Dropout Network for idea i189."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


WHITE = 0
BLACK = 1
PAWN = 0
KNIGHT = 1
BISHOP = 2
ROOK = 3
QUEEN = 4
KING = 5
SQUARES = 64
MASK_KINDS = ("defender", "attacker", "king_escape", "ray_blocker")
MASK_KIND_COUNT = len(MASK_KINDS)


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.view(-1) if num_classes == 1 else logits


def _square(row: int, file: int) -> int:
    return row * 8 + file


def _row_file(square: int) -> tuple[int, int]:
    return square // 8, square % 8


def _inside(row: int, file: int) -> bool:
    return 0 <= row < 8 and 0 <= file < 8


def _piece_channel(color: int, piece: int) -> int:
    return piece if color == WHITE else 6 + piece


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _topk_mean(values: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    masked = values.masked_fill(~mask, -torch.finfo(values.dtype).max)
    top = masked.topk(k=min(int(k), values.shape[1]), dim=1).values
    valid_top = torch.isfinite(top) & (top > -torch.finfo(values.dtype).max * 0.5)
    return torch.where(valid_top, top, torch.zeros_like(top)).sum(dim=1) / valid_top.sum(dim=1).clamp_min(1)


def _masked_entropy(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = scores.masked_fill(~mask, -torch.finfo(scores.dtype).max)
    probs = torch.softmax(masked, dim=1) * mask.to(dtype=scores.dtype)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    return -(probs * probs.clamp_min(1.0e-8).log()).sum(dim=1) / math.log(max(scores.shape[1], 2))


@dataclass(frozen=True)
class DropoutMaskBatch:
    masks: torch.Tensor
    mask_types: torch.Tensor
    mask_scores: torch.Tensor
    valid: torch.Tensor
    defender_valid: torch.Tensor
    attacker_valid: torch.Tensor
    king_escape_valid: torch.Tensor
    blocker_valid: torch.Tensor
    summary: torch.Tensor


class DefenderDropoutMaskBuilder(nn.Module):
    """Constructs deterministic intervention masks from current-board facts."""

    summary_dim = 14

    def __init__(self, input_channels: int = 18, max_masks: int = 16) -> None:
        super().__init__()
        if int(input_channels) != 18:
            raise ValueError("DefenderDropoutMaskBuilder supports the simple_18 current-board tensor")
        if int(max_masks) < 4:
            raise ValueError("max_masks must be at least 4")
        self.max_masks = int(max_masks)
        static_relations, geom_attacks, between, king_zone = self._build_geometry()
        self.register_buffer("static_relations", static_relations, persistent=False)
        self.register_buffer("geom_attacks", geom_attacks, persistent=False)
        self.register_buffer("between", between, persistent=False)
        self.register_buffer("king_zone", king_zone, persistent=False)
        self.register_buffer("random_permutation", torch.randperm(SQUARES), persistent=False)

    def forward(self, board: torch.Tensor, *, random_masks: bool = False, defenders_only: bool = False) -> DropoutMaskBatch:
        piece_planes = board[:, :12].flatten(2).clamp(0.0, 1.0)
        stm = board[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0)
        occ = piece_planes.sum(dim=1).clamp(0.0, 1.0)
        attacks, rays = self._attack_relations(piece_planes, occ)

        white_piece = piece_planes[:, :6].sum(dim=1).clamp(0.0, 1.0)
        black_piece = piece_planes[:, 6:12].sum(dim=1).clamp(0.0, 1.0)
        own_piece = self._stm_select(white_piece, black_piece, stm)
        enemy_piece = self._stm_select(black_piece, white_piece, stm)
        own_attacks = self._stm_select(attacks[WHITE], attacks[BLACK], stm)
        enemy_attacks = self._stm_select(attacks[BLACK], attacks[WHITE], stm)
        own_rays = self._stm_select(rays[WHITE], rays[BLACK], stm)
        enemy_rays = self._stm_select(rays[BLACK], rays[WHITE], stm)

        enemy_king = self._stm_select(piece_planes[:, _piece_channel(BLACK, KING)], piece_planes[:, _piece_channel(WHITE, KING)], stm)
        own_slider = self._stm_select(
            piece_planes[:, _piece_channel(WHITE, BISHOP)]
            + piece_planes[:, _piece_channel(WHITE, ROOK)]
            + piece_planes[:, _piece_channel(WHITE, QUEEN)],
            piece_planes[:, _piece_channel(BLACK, BISHOP)]
            + piece_planes[:, _piece_channel(BLACK, ROOK)]
            + piece_planes[:, _piece_channel(BLACK, QUEEN)],
            stm,
        ).clamp(0.0, 1.0)
        enemy_zone = torch.einsum(
            "bs,st->bt",
            enemy_king,
            self.king_zone.to(device=board.device, dtype=board.dtype),
        ).clamp(0.0, 1.0)
        values = board.new_tensor([1.0, 3.0, 3.0, 5.0, 9.0, 10.0]) / 10.0
        white_value = (piece_planes[:, :6] * values.view(1, 6, 1)).sum(dim=1)
        black_value = (piece_planes[:, 6:12] * values.view(1, 6, 1)).sum(dim=1)
        enemy_value = self._stm_select(black_value, white_value, stm) * enemy_piece
        own_value = self._stm_select(white_value, black_value, stm) * own_piece

        own_attacked_enemy_value = own_attacks.sum(dim=1).clamp(0.0, 1.0) * enemy_value
        defender_score = enemy_piece * (enemy_attacks * own_attacked_enemy_value[:, None, :]).sum(dim=2)
        defender_score = defender_score + enemy_piece * (own_rays * enemy_zone[:, None, :]).sum(dim=2) * 0.5

        target_pressure = ((enemy_value > 0.25).to(dtype=board.dtype) + enemy_zone).clamp(0.0, 1.0)
        attacker_score = own_piece * (own_attacks * target_pressure[:, None, :]).sum(dim=2) * (0.5 + own_value)

        attacked_by_own = own_attacks.sum(dim=1).clamp(0.0, 1.0)
        king_escape_score = enemy_zone * (1.0 - enemy_piece) * (1.0 - attacked_by_own)

        blocked_count = torch.einsum("stk,bk->bst", self.between.to(device=board.device, dtype=board.dtype), occ)
        line_relation = self.static_relations[5].to(device=board.device, dtype=board.dtype)
        one_blocker = ((blocked_count > 0.5) & (blocked_count <= 1.5)).to(dtype=board.dtype) * line_relation.unsqueeze(0)
        one_blocker_to_king = one_blocker * own_slider[:, :, None] * enemy_king[:, None, :]
        blocker_score = torch.einsum(
            "stm,bst->bm",
            self.between.to(device=board.device, dtype=board.dtype),
            one_blocker_to_king,
        )
        blocker_score = (blocker_score * occ).clamp(0.0, 1.0)

        if defenders_only:
            attacker_score = attacker_score * 0.0
            king_escape_score = king_escape_score * 0.0
            blocker_score = blocker_score * 0.0
        if random_masks:
            base = torch.linspace(1.0, 0.01, SQUARES, device=board.device, dtype=board.dtype)
            perm = self.random_permutation.to(device=board.device)
            random_scores = base.index_select(0, perm).unsqueeze(0).expand(board.shape[0], -1)
            defender_score = enemy_piece * random_scores
            attacker_score = own_piece * random_scores if not defenders_only else attacker_score
            king_escape_score = enemy_zone * random_scores if not defenders_only else king_escape_score
            blocker_score = occ * random_scores if not defenders_only else blocker_score

        masks, mask_types, mask_scores, valid = self._pack_masks(
            [defender_score, attacker_score, king_escape_score, blocker_score],
            board=board,
        )
        defender_valid = valid & (mask_types == 0)
        attacker_valid = valid & (mask_types == 1)
        king_escape_valid = valid & (mask_types == 2)
        blocker_valid = valid & (mask_types == 3)
        material_values = board.new_tensor([1.0, 3.0, 3.0, 5.0, 9.0, 0.0])
        white_material = (piece_planes[:, :6] * material_values.view(1, 6, 1)).sum(dim=(1, 2))
        black_material = (piece_planes[:, 6:12] * material_values.view(1, 6, 1)).sum(dim=(1, 2))
        own_material = stm * white_material + (1.0 - stm) * black_material
        enemy_material = stm * black_material + (1.0 - stm) * white_material
        summary = torch.cat(
            [
                defender_score.amax(dim=1, keepdim=True),
                attacker_score.amax(dim=1, keepdim=True),
                king_escape_score.amax(dim=1, keepdim=True),
                blocker_score.amax(dim=1, keepdim=True),
                defender_score.sum(dim=1, keepdim=True) / 8.0,
                attacker_score.sum(dim=1, keepdim=True) / 8.0,
                king_escape_score.sum(dim=1, keepdim=True) / 8.0,
                blocker_score.sum(dim=1, keepdim=True) / 8.0,
                own_piece.sum(dim=1, keepdim=True) / 16.0,
                enemy_piece.sum(dim=1, keepdim=True) / 16.0,
                (own_material - enemy_material).unsqueeze(1) / 39.0,
                own_attacks.mean(dim=(1, 2), keepdim=False).unsqueeze(1),
                enemy_attacks.mean(dim=(1, 2), keepdim=False).unsqueeze(1),
                (own_rays.mean(dim=(1, 2)) - enemy_rays.mean(dim=(1, 2))).unsqueeze(1),
            ],
            dim=1,
        )
        return DropoutMaskBatch(
            masks=masks,
            mask_types=mask_types,
            mask_scores=mask_scores,
            valid=valid,
            defender_valid=defender_valid,
            attacker_valid=attacker_valid,
            king_escape_valid=king_escape_valid,
            blocker_valid=blocker_valid,
            summary=summary,
        )

    def _pack_masks(
        self,
        score_groups: list[torch.Tensor],
        *,
        board: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = board.shape[0]
        per_kind = max(1, self.max_masks // MASK_KIND_COUNT)
        extra = self.max_masks - per_kind * MASK_KIND_COUNT
        slots_by_kind = [per_kind + (1 if kind < extra else 0) for kind in range(MASK_KIND_COUNT)]
        masks = board.new_zeros(batch, self.max_masks, SQUARES)
        mask_types = torch.zeros(batch, self.max_masks, dtype=torch.long, device=board.device)
        mask_scores = board.new_zeros(batch, self.max_masks)
        valid = torch.zeros(batch, self.max_masks, dtype=torch.bool, device=board.device)
        slot = 0
        for kind, scores in enumerate(score_groups):
            slots = slots_by_kind[kind]
            values, indices = scores.topk(k=slots, dim=1)
            for local in range(slots):
                masks[:, slot].scatter_(1, indices[:, local : local + 1], 1.0)
                mask_types[:, slot] = kind
                mask_scores[:, slot] = values[:, local]
                valid[:, slot] = values[:, local] > 1.0e-6
                slot += 1
        return masks, mask_types, mask_scores, valid

    def _attack_relations(
        self,
        piece_planes: torch.Tensor,
        occ: torch.Tensor,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        dtype = piece_planes.dtype
        device = piece_planes.device
        blocked_count = torch.einsum("stk,bk->bst", self.between.to(device=device, dtype=dtype), occ)
        clear = (blocked_count <= 0.5).to(dtype=dtype)
        attacks: dict[int, torch.Tensor] = {}
        rays: dict[int, torch.Tensor] = {}
        for color in (WHITE, BLACK):
            attack_sum = piece_planes.new_zeros(piece_planes.shape[0], SQUARES, SQUARES)
            ray_sum = piece_planes.new_zeros(piece_planes.shape[0], SQUARES, SQUARES)
            for piece in (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING):
                source = piece_planes[:, _piece_channel(color, piece)]
                geom = self.geom_attacks[piece, color].to(device=device, dtype=dtype)
                line_clear = clear if piece in {BISHOP, ROOK, QUEEN} else torch.ones_like(clear)
                relation = source[:, :, None] * geom.unsqueeze(0) * line_clear
                attack_sum = attack_sum + relation
                if piece in {BISHOP, ROOK, QUEEN}:
                    ray_sum = ray_sum + relation
            attacks[color] = attack_sum.clamp(0.0, 1.0)
            rays[color] = ray_sum.clamp(0.0, 1.0)
        return attacks, rays

    @staticmethod
    def _stm_select(white_tensor: torch.Tensor, black_tensor: torch.Tensor, stm: torch.Tensor) -> torch.Tensor:
        selector = stm.view(-1, *([1] * (white_tensor.ndim - 1)))
        return selector * white_tensor + (1.0 - selector) * black_tensor

    @staticmethod
    def _build_geometry() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        static_relations = torch.zeros(6, SQUARES, SQUARES, dtype=torch.float32)
        geom_attacks = torch.zeros(6, 2, SQUARES, SQUARES, dtype=torch.float32)
        between = torch.zeros(SQUARES, SQUARES, SQUARES, dtype=torch.float32)
        king_zone = torch.zeros(SQUARES, SQUARES, dtype=torch.float32)
        knight_offsets = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        king_offsets = [(r, f) for r in (-1, 0, 1) for f in (-1, 0, 1) if r != 0 or f != 0]
        bishop_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        rook_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for source in range(SQUARES):
            source_row, source_file = _row_file(source)
            for target in range(SQUARES):
                target_row, target_file = _row_file(target)
                if source == target:
                    king_zone[source, target] = 1.0
                    continue
                row_delta = target_row - source_row
                file_delta = target_file - source_file
                static_relations[0, source, target] = 1.0 if source_row == target_row else 0.0
                static_relations[1, source, target] = 1.0 if source_file == target_file else 0.0
                static_relations[2, source, target] = 1.0 if abs(row_delta) == abs(file_delta) else 0.0
                static_relations[3, source, target] = 1.0 if (abs(row_delta), abs(file_delta)) in {(1, 2), (2, 1)} else 0.0
                static_relations[4, source, target] = 1.0 if max(abs(row_delta), abs(file_delta)) == 1 else 0.0
                aligned = (
                    static_relations[0, source, target]
                    or static_relations[1, source, target]
                    or static_relations[2, source, target]
                )
                static_relations[5, source, target] = 1.0 if aligned else 0.0
                if max(abs(row_delta), abs(file_delta)) <= 1:
                    king_zone[source, target] = 1.0
                if aligned:
                    row_step = _sign(row_delta)
                    file_step = _sign(file_delta)
                    row, file = source_row + row_step, source_file + file_step
                    while (row, file) != (target_row, target_file):
                        between[source, target, _square(row, file)] = 1.0
                        row += row_step
                        file += file_step
            for color in (WHITE, BLACK):
                pawn_forward = -1 if color == WHITE else 1
                for file_delta in (-1, 1):
                    row, file = source_row + pawn_forward, source_file + file_delta
                    if _inside(row, file):
                        geom_attacks[PAWN, color, source, _square(row, file)] = 1.0
                for row_delta, file_delta in knight_offsets:
                    row, file = source_row + row_delta, source_file + file_delta
                    if _inside(row, file):
                        geom_attacks[KNIGHT, color, source, _square(row, file)] = 1.0
                for row_delta, file_delta in king_offsets:
                    row, file = source_row + row_delta, source_file + file_delta
                    if _inside(row, file):
                        geom_attacks[KING, color, source, _square(row, file)] = 1.0
                for piece, directions in ((BISHOP, bishop_dirs), (ROOK, rook_dirs), (QUEEN, bishop_dirs + rook_dirs)):
                    for row_delta, file_delta in directions:
                        row, file = source_row + row_delta, source_file + file_delta
                        while _inside(row, file):
                            geom_attacks[piece, color, source, _square(row, file)] = 1.0
                            row += row_delta
                            file += file_delta
        return static_relations, geom_attacks, between, king_zone


class DefenderDropoutTrunk(nn.Module):
    def __init__(
        self,
        input_channels: int,
        channels: int,
        depth: int,
        dropout: float,
        use_batchnorm: bool,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        layers: list[nn.Module] = []
        in_channels = int(input_channels) + 2
        for _ in range(max(1, int(depth))):
            layers.append(nn.Conv2d(in_channels, int(channels), kernel_size=3, padding=1, bias=not use_batchnorm))
            layers.append(nn.BatchNorm2d(int(channels)) if use_batchnorm else nn.GroupNorm(1, int(channels)))
            layers.append(nn.GELU())
            if float(dropout) > 0:
                layers.append(nn.Dropout2d(float(dropout)))
            in_channels = int(channels)
        self.layers = nn.Sequential(*layers)
        self.register_buffer("coords", self._coords(), persistent=False)
        self.output_dim = int(channels) * 2

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        coords = self.coords.to(device=board.device, dtype=board.dtype).expand(board.shape[0], -1, -1, -1)
        h = self.layers(torch.cat([board, coords], dim=1))
        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        return h, pooled

    @staticmethod
    def _coords() -> torch.Tensor:
        rows = torch.linspace(-1.0, 1.0, 8).view(1, 8, 1).expand(1, 8, 8)
        files = torch.linspace(-1.0, 1.0, 8).view(1, 1, 8).expand(1, 8, 8)
        return torch.stack([rows[0], files[0]], dim=0).unsqueeze(0)


class InterventionHead(nn.Module):
    def __init__(self, channels: int, context_dim: int, intervention_dim: int, dropout: float) -> None:
        super().__init__()
        self.type_embedding = nn.Embedding(MASK_KIND_COUNT, int(intervention_dim))
        input_dim = int(channels) * 4 + int(context_dim) + int(intervention_dim) + 2
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(intervention_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(int(intervention_dim), max(16, int(intervention_dim) // 2)),
            nn.GELU(),
            nn.Linear(max(16, int(intervention_dim) // 2), 1),
        )

    def forward(
        self,
        h: torch.Tensor,
        context: torch.Tensor,
        masks: torch.Tensor,
        mask_types: torch.Tensor,
        mask_scores: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        batch, slots, squares = masks.shape
        h_flat = h.flatten(2).transpose(1, 2)
        weights = masks.to(dtype=h.dtype).unsqueeze(-1)
        local_mean = (weights * h_flat.unsqueeze(1)).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0e-6)
        inverse = (1.0 - masks.to(dtype=h.dtype)).unsqueeze(-1)
        retained_mean = (inverse * h_flat.unsqueeze(1)).sum(dim=2) / inverse.sum(dim=2).clamp_min(1.0e-6)
        abs_gap = (local_mean - retained_mean).abs()
        product = local_mean * retained_mean
        context_expand = context.unsqueeze(1).expand(batch, slots, -1)
        type_emb = self.type_embedding(mask_types.clamp(0, MASK_KIND_COUNT - 1))
        scalar = torch.stack(
            [
                mask_scores,
                valid.to(dtype=h.dtype),
            ],
            dim=-1,
        )
        features = torch.cat([local_mean, retained_mean, abs_gap, product, context_expand, type_emb, scalar], dim=-1)
        delta = self.net(features).squeeze(-1)
        return delta * valid.to(dtype=delta.dtype)


class CounterfactualDefenderDropoutNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        intervention_dim: int = 64,
        max_masks: int = 16,
        topk: int = 3,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError("CounterfactualDefenderDropoutNetwork supports the puzzle_binary one-logit contract")
        self.num_classes = int(num_classes)
        self.max_masks = int(max_masks)
        self.topk = int(topk)
        self.ablation = str(ablation)
        allowed_ablations = {"none", "random_masks", "defenders_only", "no_intervention_head"}
        if self.ablation not in allowed_ablations:
            raise ValueError(f"Unknown ablation={self.ablation!r}; expected one of {sorted(allowed_ablations)}")
        self.trunk = DefenderDropoutTrunk(
            input_channels=int(input_channels),
            channels=int(channels),
            depth=int(depth),
            dropout=float(dropout),
            use_batchnorm=bool(use_batchnorm),
        )
        self.mask_builder = DefenderDropoutMaskBuilder(input_channels=int(input_channels), max_masks=int(max_masks))
        self.context = nn.Sequential(
            nn.LayerNorm(self.trunk.output_dim + DefenderDropoutMaskBuilder.summary_dim),
            nn.Linear(self.trunk.output_dim + DefenderDropoutMaskBuilder.summary_dim, int(hidden_dim)),
            nn.GELU(),
        )
        self.base_head = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), max(16, int(hidden_dim) // 2)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(max(16, int(hidden_dim) // 2), 1),
        )
        self.intervention_head = InterventionHead(
            channels=int(channels),
            context_dim=int(hidden_dim),
            intervention_dim=int(intervention_dim),
            dropout=float(dropout),
        )
        correction_dim = 13
        self.correction = nn.Sequential(
            nn.LayerNorm(correction_dim),
            nn.Linear(correction_dim, max(16, int(hidden_dim) // 2)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(max(16, int(hidden_dim) // 2), 1),
        )

    def forward(self, x: torch.Tensor, *, return_aux: bool = False) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.trunk.spec)
        masks = self.mask_builder(
            board,
            random_masks=self.ablation == "random_masks",
            defenders_only=self.ablation == "defenders_only",
        )
        h, pooled = self.trunk(board)
        context = self.context(torch.cat([pooled, masks.summary], dim=1))
        base_logit = self.base_head(context).view(-1)
        if self.ablation == "no_intervention_head":
            delta = masks.mask_scores.new_zeros(masks.mask_scores.shape)
        else:
            delta = self.intervention_head(h, context, masks.masks, masks.mask_types, masks.mask_scores, masks.valid)
        abs_delta = delta.abs()
        defender_sensitivity = _topk_mean(abs_delta, masks.defender_valid, self.topk)
        attacker_sensitivity = _topk_mean(abs_delta, masks.attacker_valid, self.topk)
        king_escape_sensitivity = _topk_mean(abs_delta, masks.king_escape_valid, self.topk)
        blocker_sensitivity = _topk_mean(abs_delta, masks.blocker_valid, self.topk)
        asymmetry = defender_sensitivity - attacker_sensitivity
        defender_minus_escape = defender_sensitivity - king_escape_sensitivity
        defender_minus_blocker = defender_sensitivity - blocker_sensitivity
        sensitivity_entropy = _masked_entropy(abs_delta, masks.valid)
        valid_count = masks.valid.to(dtype=delta.dtype).sum(dim=1)
        defender_count = masks.defender_valid.to(dtype=delta.dtype).sum(dim=1)
        attacker_count = masks.attacker_valid.to(dtype=delta.dtype).sum(dim=1)
        max_sensitivity = abs_delta.masked_fill(~masks.valid, 0.0).amax(dim=1)
        mean_sensitivity = (abs_delta * masks.valid.to(dtype=delta.dtype)).sum(dim=1) / valid_count.clamp_min(1.0)
        correction_features = torch.stack(
            [
                asymmetry,
                defender_sensitivity,
                attacker_sensitivity,
                king_escape_sensitivity,
                blocker_sensitivity,
                defender_minus_escape,
                defender_minus_blocker,
                sensitivity_entropy,
                max_sensitivity,
                mean_sensitivity,
                valid_count / float(max(1, self.max_masks)),
                defender_count / float(max(1, self.max_masks)),
                attacker_count / float(max(1, self.max_masks)),
            ],
            dim=1,
        )
        correction = self.correction(correction_features).view(-1)
        logits = _format_logits(base_logit + correction, self.num_classes)
        output = {
            "logits": logits,
            "base_logit": base_logit,
            "intervention_correction": correction,
            "intervention_delta": delta,
            "intervention_mask_scores": masks.mask_scores,
            "intervention_valid": masks.valid.to(dtype=delta.dtype),
            "defender_sensitivity": defender_sensitivity,
            "attacker_sensitivity": attacker_sensitivity,
            "king_escape_sensitivity": king_escape_sensitivity,
            "blocker_sensitivity": blocker_sensitivity,
            "sensitivity_asymmetry": asymmetry,
            "defender_minus_escape": defender_minus_escape,
            "defender_minus_blocker": defender_minus_blocker,
            "sensitivity_entropy": sensitivity_entropy,
            "max_sensitivity": max_sensitivity,
            "mean_sensitivity": mean_sensitivity,
            "defender_mask_count": defender_count,
            "attacker_mask_count": attacker_count,
            "king_escape_mask_count": masks.king_escape_valid.to(dtype=delta.dtype).sum(dim=1),
            "blocker_mask_count": masks.blocker_valid.to(dtype=delta.dtype).sum(dim=1),
            "mechanism_energy": context.pow(2).mean(dim=1),
            "proposal_profile_strength": asymmetry,
            "proposal_keyword_count": logits.new_full((board.shape[0],), 4.0),
        }
        if return_aux:
            output["intervention_masks"] = masks.masks
            output["intervention_mask_types"] = masks.mask_types
        return output


def build_counterfactual_defender_dropout_network_from_config(
    config: dict[str, Any],
) -> CounterfactualDefenderDropoutNetwork:
    cfg = dict(config)
    return CounterfactualDefenderDropoutNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", cfg.get("board_depth", 2))),
        intervention_dim=int(cfg.get("intervention_dim", cfg.get("hidden_dim", 64))),
        max_masks=int(cfg.get("max_masks", 16)),
        topk=int(cfg.get("topk", 3)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        ablation=str(cfg.get("ablation", "none")),
    )

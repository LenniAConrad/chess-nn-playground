"""Latent Reply Entropy Network for idea i192."""
from __future__ import annotations

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
REPLY_KINDS = (
    "king_escape",
    "capture_attacker",
    "block_line",
    "defend_target",
    "counter_threat",
    "quiet_resource",
)
REPLY_KIND_COUNT = len(REPLY_KINDS)


@dataclass(frozen=True)
class ReplyCandidates:
    features: torch.Tensor
    kinds: torch.Tensor
    source: torch.Tensor
    target: torch.Tensor
    base_scores: torch.Tensor
    valid: torch.Tensor
    summary: torch.Tensor
    kind_max: torch.Tensor
    kind_count: torch.Tensor


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


class LatentReplyCandidateBuilder(nn.Module):
    """Builds board-local reply/resource candidates for entropy pooling."""

    feature_dim = 31
    summary_dim = 17

    def __init__(self, input_channels: int = 18, max_replies: int = 96) -> None:
        super().__init__()
        if int(input_channels) != 18:
            raise ValueError("LatentReplyCandidateBuilder supports the simple_18 current-board tensor")
        if int(max_replies) < REPLY_KIND_COUNT:
            raise ValueError("max_replies must provide at least one slot per reply family")
        self.max_replies = int(max_replies)
        attack_geom, move_geom, between, line_relation, king_zone, square_features = self._build_geometry()
        self.register_buffer("attack_geom", attack_geom, persistent=False)
        self.register_buffer("move_geom", move_geom, persistent=False)
        self.register_buffer("between", between, persistent=False)
        self.register_buffer("line_relation", line_relation, persistent=False)
        self.register_buffer("king_zone", king_zone, persistent=False)
        self.register_buffer("square_features", square_features, persistent=False)

    def forward(self, board: torch.Tensor, *, count_only: bool = False) -> ReplyCandidates:
        piece_planes = board[:, :12].flatten(2).clamp(0.0, 1.0)
        stm = board[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0)
        occ = piece_planes.sum(dim=1).clamp(0.0, 1.0)
        attacks, moves = self._relations(piece_planes, occ)

        white_piece = piece_planes[:, :6].sum(dim=1).clamp(0.0, 1.0)
        black_piece = piece_planes[:, 6:12].sum(dim=1).clamp(0.0, 1.0)
        attacker_piece = self._stm_select(white_piece, black_piece, stm)
        defender_piece = self._stm_select(black_piece, white_piece, stm)
        attacker_control = self._stm_select(attacks[WHITE], attacks[BLACK], stm)
        defender_control = self._stm_select(attacks[BLACK], attacks[WHITE], stm)
        defender_moves = self._stm_select(moves[BLACK], moves[WHITE], stm)

        attacker_king = self._stm_select(
            piece_planes[:, _piece_channel(WHITE, KING)],
            piece_planes[:, _piece_channel(BLACK, KING)],
            stm,
        )
        defender_king = self._stm_select(
            piece_planes[:, _piece_channel(BLACK, KING)],
            piece_planes[:, _piece_channel(WHITE, KING)],
            stm,
        )
        attacker_slider = self._stm_select(
            piece_planes[:, _piece_channel(WHITE, BISHOP)]
            + piece_planes[:, _piece_channel(WHITE, ROOK)]
            + piece_planes[:, _piece_channel(WHITE, QUEEN)],
            piece_planes[:, _piece_channel(BLACK, BISHOP)]
            + piece_planes[:, _piece_channel(BLACK, ROOK)]
            + piece_planes[:, _piece_channel(BLACK, QUEEN)],
            stm,
        ).clamp(0.0, 1.0)

        tactical_values = board.new_tensor([1.0, 3.0, 3.0, 5.0, 9.0, 10.0]) / 10.0
        material_values = board.new_tensor([1.0, 3.0, 3.0, 5.0, 9.0, 0.0])
        white_tactical_value = (piece_planes[:, :6] * tactical_values.view(1, 6, 1)).sum(dim=1)
        black_tactical_value = (piece_planes[:, 6:12] * tactical_values.view(1, 6, 1)).sum(dim=1)
        attacker_value = self._stm_select(white_tactical_value, black_tactical_value, stm) * attacker_piece
        defender_value = self._stm_select(black_tactical_value, white_tactical_value, stm) * defender_piece
        attacker_high = ((attacker_value >= 0.5).to(dtype=board.dtype) + attacker_king).clamp(0.0, 1.0)
        defender_high = ((defender_value >= 0.5).to(dtype=board.dtype) + defender_king).clamp(0.0, 1.0)

        white_material = (piece_planes[:, :6] * material_values.view(1, 6, 1)).sum(dim=(1, 2))
        black_material = (piece_planes[:, 6:12] * material_values.view(1, 6, 1)).sum(dim=(1, 2))
        attacker_material = stm * white_material + (1.0 - stm) * black_material
        defender_material = stm * black_material + (1.0 - stm) * white_material
        material_balance = ((attacker_material - defender_material) / 39.0).unsqueeze(1)

        attacker_attack_count = attacker_control.sum(dim=1)
        defender_attack_count = defender_control.sum(dim=1)
        attacker_attacked = (attacker_attack_count > 0.5).to(dtype=board.dtype)
        defender_attacked = (defender_attack_count > 0.5).to(dtype=board.dtype)
        defender_king_zone = torch.einsum(
            "bs,st->bt",
            defender_king,
            self.king_zone.to(device=board.device, dtype=board.dtype),
        ).clamp(0.0, 1.0)
        attacker_king_zone = torch.einsum(
            "bs,st->bt",
            attacker_king,
            self.king_zone.to(device=board.device, dtype=board.dtype),
        ).clamp(0.0, 1.0)

        empty = (1.0 - occ).clamp(0.0, 1.0)
        blocked_count = torch.einsum(
            "stk,bk->bst",
            self.between.to(device=board.device, dtype=board.dtype),
            occ,
        )
        line_threat = (
            attacker_slider[:, :, None]
            * defender_high[:, None, :]
            * self.line_relation.to(device=board.device, dtype=board.dtype).unsqueeze(0)
            * (blocked_count <= 1.5).to(dtype=board.dtype)
        )
        block_square_pressure = torch.einsum(
            "stm,bst->bm",
            self.between.to(device=board.device, dtype=board.dtype),
            line_threat,
        ).clamp(0.0, 1.0)

        king_escape = (
            defender_king[:, :, None]
            * defender_moves
            * empty[:, None, :]
            * (1.0 - attacker_attacked[:, None, :])
        ).clamp(0.0, 1.0)
        capture_attacker = (
            defender_moves
            * attacker_piece[:, None, :]
            * (0.35 + attacker_value[:, None, :])
        ).clamp(0.0, 1.0)
        block_line = (
            defender_moves
            * empty[:, None, :]
            * block_square_pressure[:, None, :]
            * (0.5 + defender_value[:, :, None])
        ).clamp(0.0, 1.0)
        defend_target = (
            defender_control
            * defender_high[:, None, :]
            * attacker_attacked[:, None, :]
            * (0.4 + defender_value[:, None, :])
        ).clamp(0.0, 1.0)
        counter_threat = (
            defender_control
            * (attacker_high + attacker_king_zone).clamp(0.0, 1.0)[:, None, :]
            * (0.4 + attacker_value[:, None, :])
        ).clamp(0.0, 1.0)
        quiet_resource = (
            defender_moves
            * empty[:, None, :]
            * (1.0 - attacker_attacked[:, None, :])
            * (0.1 + defender_value[:, :, None])
        ).clamp(0.0, 1.0)

        scores_by_kind = [
            king_escape,
            capture_attacker,
            block_line,
            defend_target,
            counter_threat,
            quiet_resource,
        ]
        candidates = self._pack_candidates(
            scores_by_kind=scores_by_kind,
            board=board,
            defender_piece=defender_piece,
            attacker_piece=attacker_piece,
            defender_value=defender_value,
            attacker_value=attacker_value,
            defender_king=defender_king,
            attacker_attacked=attacker_attacked,
            defender_attacked=defender_attacked,
            attacker_king_zone=attacker_king_zone,
            defender_king_zone=defender_king_zone,
            block_square_pressure=block_square_pressure,
        )

        kind_max = torch.stack([scores.flatten(1).amax(dim=1) for scores in scores_by_kind], dim=1)
        kind_count = torch.stack(
            [(scores > 1.0e-6).to(dtype=board.dtype).sum(dim=(1, 2)) / 64.0 for scores in scores_by_kind],
            dim=1,
        )
        valid_count = candidates.valid.to(dtype=board.dtype).sum(dim=1, keepdim=True) / float(self.max_replies)
        pressure_on_high = (attacker_attacked * defender_high).sum(dim=1, keepdim=True) / defender_high.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        mobility_mass = (defender_moves * (1.0 - defender_piece[:, None, :])).sum(dim=(1, 2), keepdim=False).view(-1, 1)
        mobility_mass = mobility_mass / 64.0
        king_escape_mass = (king_escape > 1.0e-6).to(dtype=board.dtype).sum(dim=(1, 2), keepdim=False).view(-1, 1) / 8.0
        block_mass = (block_line > 1.0e-6).to(dtype=board.dtype).sum(dim=(1, 2), keepdim=False).view(-1, 1) / 64.0
        summary = torch.cat(
            [
                kind_max,
                kind_count,
                valid_count,
                material_balance,
                pressure_on_high,
                mobility_mass,
                king_escape_mass + block_mass,
            ],
            dim=1,
        )

        features = candidates.features
        base_scores = candidates.base_scores
        if count_only:
            count_features = torch.zeros_like(features)
            count_features[:, :, :REPLY_KIND_COUNT] = F.one_hot(
                candidates.kinds.clamp_min(0),
                num_classes=REPLY_KIND_COUNT,
            ).to(dtype=features.dtype)
            count_features[:, :, REPLY_KIND_COUNT] = candidates.valid.to(dtype=features.dtype)
            features = count_features
            base_scores = candidates.valid.to(dtype=features.dtype)

        return ReplyCandidates(
            features=features,
            kinds=candidates.kinds,
            source=candidates.source,
            target=candidates.target,
            base_scores=base_scores,
            valid=candidates.valid,
            summary=summary,
            kind_max=kind_max,
            kind_count=kind_count,
        )

    def _pack_candidates(
        self,
        *,
        scores_by_kind: list[torch.Tensor],
        board: torch.Tensor,
        defender_piece: torch.Tensor,
        attacker_piece: torch.Tensor,
        defender_value: torch.Tensor,
        attacker_value: torch.Tensor,
        defender_king: torch.Tensor,
        attacker_attacked: torch.Tensor,
        defender_attacked: torch.Tensor,
        attacker_king_zone: torch.Tensor,
        defender_king_zone: torch.Tensor,
        block_square_pressure: torch.Tensor,
    ) -> ReplyCandidates:
        batch = board.shape[0]
        per_kind = max(1, self.max_replies // REPLY_KIND_COUNT)
        extra = self.max_replies - per_kind * REPLY_KIND_COUNT
        slots_by_kind = [per_kind + (1 if kind < extra else 0) for kind in range(REPLY_KIND_COUNT)]
        features = board.new_zeros(batch, self.max_replies, self.feature_dim)
        kinds = torch.zeros(batch, self.max_replies, dtype=torch.long, device=board.device)
        source = torch.zeros_like(kinds)
        target = torch.zeros_like(kinds)
        base_scores = board.new_zeros(batch, self.max_replies)
        valid = torch.zeros(batch, self.max_replies, dtype=torch.bool, device=board.device)
        square_features = self.square_features.to(device=board.device, dtype=board.dtype)
        source_context = [
            defender_piece,
            defender_value,
            attacker_attacked,
            defender_attacked,
            defender_king,
        ]
        target_context = [
            attacker_piece,
            defender_piece,
            attacker_value,
            defender_value,
            attacker_attacked,
            defender_attacked,
            attacker_king_zone,
            defender_king_zone,
            block_square_pressure,
        ]
        slot = 0
        for kind, scores in enumerate(scores_by_kind):
            slots = slots_by_kind[kind]
            values, flat_indices = scores.reshape(batch, -1).topk(k=slots, dim=1)
            src = torch.div(flat_indices, SQUARES, rounding_mode="floor")
            dst = flat_indices.remainder(SQUARES)
            kind_onehot = board.new_zeros(batch, slots, REPLY_KIND_COUNT)
            kind_onehot[:, :, kind] = 1.0
            src_geom = square_features.index_select(0, src.reshape(-1)).view(batch, slots, -1)
            dst_geom = square_features.index_select(0, dst.reshape(-1)).view(batch, slots, -1)
            src_values = [tensor.gather(1, src) for tensor in source_context]
            dst_values = [tensor.gather(1, dst) for tensor in target_context]
            context = torch.stack([*src_values, *dst_values], dim=-1)
            local_features = torch.cat([kind_onehot, values.unsqueeze(-1), src_geom, dst_geom, context], dim=-1)
            features[:, slot : slot + slots] = local_features
            kinds[:, slot : slot + slots] = kind
            source[:, slot : slot + slots] = src
            target[:, slot : slot + slots] = dst
            base_scores[:, slot : slot + slots] = values
            valid[:, slot : slot + slots] = values > 1.0e-6
            slot += slots

        no_candidate = ~valid.any(dim=1)
        if no_candidate.any():
            features[no_candidate, 0] = 0.0
            features[no_candidate, 0, REPLY_KIND_COUNT - 1] = 1.0
            features[no_candidate, 0, REPLY_KIND_COUNT] = 1.0e-6
            kinds[no_candidate, 0] = REPLY_KIND_COUNT - 1
            base_scores[no_candidate, 0] = 1.0e-6
            valid[no_candidate, 0] = True

        return ReplyCandidates(
            features=features,
            kinds=kinds,
            source=source,
            target=target,
            base_scores=base_scores,
            valid=valid,
            summary=board.new_zeros(batch, self.summary_dim),
            kind_max=board.new_zeros(batch, REPLY_KIND_COUNT),
            kind_count=board.new_zeros(batch, REPLY_KIND_COUNT),
        )

    def _relations(
        self,
        piece_planes: torch.Tensor,
        occ: torch.Tensor,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        dtype = piece_planes.dtype
        device = piece_planes.device
        blocked_count = torch.einsum("stk,bk->bst", self.between.to(device=device, dtype=dtype), occ)
        clear = (blocked_count <= 0.5).to(dtype=dtype)
        empty = (1.0 - occ).clamp(0.0, 1.0)
        attacks: dict[int, torch.Tensor] = {}
        moves: dict[int, torch.Tensor] = {}
        for color in (WHITE, BLACK):
            attack_sum = piece_planes.new_zeros(piece_planes.shape[0], SQUARES, SQUARES)
            move_sum = piece_planes.new_zeros(piece_planes.shape[0], SQUARES, SQUARES)
            own_piece = piece_planes[:, :6].sum(dim=1) if color == WHITE else piece_planes[:, 6:12].sum(dim=1)
            for piece in (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING):
                source = piece_planes[:, _piece_channel(color, piece)]
                attack_geom = self.attack_geom[piece, color].to(device=device, dtype=dtype)
                move_geom = self.move_geom[piece, color].to(device=device, dtype=dtype)
                line_clear = clear if piece in {BISHOP, ROOK, QUEEN} else torch.ones_like(clear)
                attack_relation = source[:, :, None] * attack_geom.unsqueeze(0) * line_clear
                if piece == PAWN:
                    move_target = empty[:, None, :]
                else:
                    move_target = (1.0 - own_piece[:, None, :]).clamp(0.0, 1.0)
                move_relation = source[:, :, None] * move_geom.unsqueeze(0) * line_clear * move_target
                attack_sum = attack_sum + attack_relation
                move_sum = move_sum + move_relation
            attacks[color] = attack_sum.clamp(0.0, 1.0)
            moves[color] = move_sum.clamp(0.0, 1.0)
        return attacks, moves

    @staticmethod
    def _stm_select(white_tensor: torch.Tensor, black_tensor: torch.Tensor, stm: torch.Tensor) -> torch.Tensor:
        view_shape = [stm.shape[0], *([1] * (white_tensor.ndim - 1))]
        stm_view = stm.view(*view_shape)
        return stm_view * white_tensor + (1.0 - stm_view) * black_tensor

    @staticmethod
    def _build_geometry() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        attack_geom = torch.zeros(6, 2, SQUARES, SQUARES)
        move_geom = torch.zeros_like(attack_geom)
        between = torch.zeros(SQUARES, SQUARES, SQUARES)
        line_relation = torch.zeros(SQUARES, SQUARES)
        king_zone = torch.zeros(SQUARES, SQUARES)
        square_features = torch.zeros(SQUARES, 5)
        knight_steps = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        king_steps = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        bishop_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        rook_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for source in range(SQUARES):
            row, file = _row_file(source)
            row_norm = row / 7.0
            file_norm = file / 7.0
            edge_distance = min(row, file, 7 - row, 7 - file) / 3.5
            center_distance = (abs(row - 3.5) + abs(file - 3.5)) / 7.0
            square_features[source] = torch.tensor(
                [row_norm, file_norm, 1.0 - center_distance, edge_distance, float((row + file) % 2)]
            )
            for dr, df in knight_steps:
                target_row, target_file = row + dr, file + df
                if _inside(target_row, target_file):
                    target = _square(target_row, target_file)
                    attack_geom[KNIGHT, :, source, target] = 1.0
                    move_geom[KNIGHT, :, source, target] = 1.0
            for dr, df in king_steps:
                target_row, target_file = row + dr, file + df
                if _inside(target_row, target_file):
                    target = _square(target_row, target_file)
                    attack_geom[KING, :, source, target] = 1.0
                    move_geom[KING, :, source, target] = 1.0
                    king_zone[source, target] = 1.0
            for color, pawn_dir in ((WHITE, -1), (BLACK, 1)):
                move_row = row + pawn_dir
                if _inside(move_row, file):
                    move_geom[PAWN, color, source, _square(move_row, file)] = 1.0
                for df in (-1, 1):
                    target_row, target_file = row + pawn_dir, file + df
                    if _inside(target_row, target_file):
                        attack_geom[PAWN, color, source, _square(target_row, target_file)] = 1.0
            for dirs, piece in ((bishop_dirs, BISHOP), (rook_dirs, ROOK), (bishop_dirs + rook_dirs, QUEEN)):
                for dr, df in dirs:
                    ray: list[int] = []
                    target_row, target_file = row + dr, file + df
                    while _inside(target_row, target_file):
                        target = _square(target_row, target_file)
                        attack_geom[piece, :, source, target] = 1.0
                        move_geom[piece, :, source, target] = 1.0
                        line_relation[source, target] = 1.0
                        for mid in ray:
                            between[source, target, mid] = 1.0
                        ray.append(target)
                        target_row += dr
                        target_file += df
        return attack_geom, move_geom, between, line_relation, king_zone, square_features


class LatentReplyEntropyNetwork(nn.Module):
    """Classifier that reads puzzle pressure through reply-set entropy."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        reply_dim: int = 64,
        temperature: float = 0.7,
        max_replies: int = 96,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.num_classes = int(num_classes)
        self.temperature = float(temperature)
        self.ablation = str(ablation or "none")
        self.builder = LatentReplyCandidateBuilder(input_channels=int(input_channels), max_replies=int(max_replies))
        trunk_layers: list[nn.Module] = []
        in_channels = int(input_channels) + 2
        for _ in range(max(1, int(depth))):
            trunk_layers.append(nn.Conv2d(in_channels, int(channels), kernel_size=3, padding=1, bias=not use_batchnorm))
            if use_batchnorm:
                trunk_layers.append(nn.BatchNorm2d(int(channels)))
            trunk_layers.append(nn.GELU())
            if dropout > 0:
                trunk_layers.append(nn.Dropout2d(float(dropout)))
            in_channels = int(channels)
        self.trunk = nn.Sequential(*trunk_layers)
        self.square_projection = nn.Linear(int(channels), int(reply_dim))
        board_context_dim = int(channels) * 2 + self.builder.summary_dim
        self.global_projection = nn.Sequential(
            nn.LayerNorm(board_context_dim),
            nn.Linear(board_context_dim, int(reply_dim)),
            nn.GELU(),
        )
        self.reply_encoder = nn.Sequential(
            nn.LayerNorm(self.builder.feature_dim + int(reply_dim) * 3),
            nn.Linear(self.builder.feature_dim + int(reply_dim) * 3, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(int(hidden_dim), int(reply_dim)),
            nn.GELU(),
        )
        self.safe_reply_scorer = nn.Linear(int(reply_dim), 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(board_context_dim + 4),
            nn.Linear(board_context_dim + 4, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(int(hidden_dim), self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        candidates = self.builder(board, count_only=self.ablation == "reply_count_only")
        features = self.trunk(torch.cat([board, self._coordinate_planes(board)], dim=1))
        pooled_mean = features.mean(dim=(2, 3))
        pooled_max = features.amax(dim=(2, 3))
        board_context = torch.cat([pooled_mean, pooled_max, candidates.summary], dim=1)
        global_token = self.global_projection(board_context)
        square_tokens = self.square_projection(features.flatten(2).transpose(1, 2))
        source_tokens = self._gather_square_tokens(square_tokens, candidates.source)
        target_tokens = self._gather_square_tokens(square_tokens, candidates.target)
        expanded_global = global_token.unsqueeze(1).expand(-1, candidates.features.shape[1], -1)
        encoded = self.reply_encoder(
            torch.cat([candidates.features, source_tokens, target_tokens, expanded_global], dim=-1)
        )

        learned_scores = self.safe_reply_scorer(encoded).squeeze(-1) + torch.log1p(candidates.base_scores)
        if self.ablation == "fixed_uniform_scores":
            learned_scores = torch.zeros_like(learned_scores)
        temperature = max(self.temperature, 1.0e-3)
        masked_scores = (learned_scores / temperature).masked_fill(~candidates.valid, -1.0e4)
        probabilities = torch.softmax(masked_scores, dim=1) * candidates.valid.to(dtype=learned_scores.dtype)
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1.0e-8)

        reply_entropy = -(probabilities * probabilities.clamp_min(1.0e-8).log()).sum(dim=1)
        valid_reply_count = candidates.valid.to(dtype=board.dtype).sum(dim=1)
        entropy_denom = valid_reply_count.clamp_min(2.0).log()
        reply_entropy_normalized = reply_entropy / entropy_denom
        top2_values = probabilities.topk(k=2, dim=1).values
        reply_top1 = top2_values[:, 0]
        reply_top2_gap = top2_values[:, 0] - top2_values[:, 1]
        effective_reply_count = reply_entropy.exp()
        safe_reply_mass = (probabilities * torch.sigmoid(learned_scores) * candidates.valid.to(dtype=board.dtype)).sum(dim=1)

        entropy_features = torch.stack([reply_entropy, reply_top1, reply_top2_gap, effective_reply_count], dim=1)
        if self.ablation == "no_entropy_features":
            entropy_features = torch.zeros_like(entropy_features)
        logits = self.classifier(torch.cat([board_context, entropy_features], dim=1))

        kind_probability_mass = []
        kind_score_max = []
        for kind in range(REPLY_KIND_COUNT):
            mask = candidates.valid & (candidates.kinds == kind)
            kind_probability_mass.append(probabilities.masked_fill(~mask, 0.0).sum(dim=1))
            kind_score_max.append(learned_scores.masked_fill(~mask, -1.0e4).amax(dim=1))
        kind_probability_tensor = torch.stack(kind_probability_mass, dim=1)
        kind_score_tensor = torch.stack(kind_score_max, dim=1)
        kind_score_tensor = torch.where(kind_score_tensor > -9999.0, kind_score_tensor, torch.zeros_like(kind_score_tensor))

        return {
            "logits": _format_logits(logits, self.num_classes),
            "reply_entropy": reply_entropy,
            "reply_entropy_normalized": reply_entropy_normalized,
            "reply_top1": reply_top1,
            "reply_top2_gap": reply_top2_gap,
            "effective_reply_count": effective_reply_count,
            "valid_reply_count": valid_reply_count,
            "safe_reply_mass": safe_reply_mass,
            "reply_score": learned_scores,
            "reply_probability": probabilities,
            "reply_valid": candidates.valid,
            "reply_kind": candidates.kinds,
            "reply_source_square": candidates.source,
            "reply_target_square": candidates.target,
            "reply_kind_count": candidates.kind_count,
            "reply_kind_max": candidates.kind_max,
            "reply_kind_probability": kind_probability_tensor,
            "reply_kind_score": kind_score_tensor,
            "king_escape_reply_mass": kind_probability_tensor[:, 0],
            "capture_attacker_reply_mass": kind_probability_tensor[:, 1],
            "block_line_reply_mass": kind_probability_tensor[:, 2],
            "defend_target_reply_mass": kind_probability_tensor[:, 3],
            "counter_threat_reply_mass": kind_probability_tensor[:, 4],
            "quiet_resource_reply_mass": kind_probability_tensor[:, 5],
        }

    @staticmethod
    def _coordinate_planes(board: torch.Tensor) -> torch.Tensor:
        coord = torch.linspace(-1.0, 1.0, 8, device=board.device, dtype=board.dtype)
        rows = coord.view(1, 1, 8, 1).expand(board.shape[0], 1, 8, 8)
        files = coord.view(1, 1, 1, 8).expand(board.shape[0], 1, 8, 8)
        return torch.cat([rows, files], dim=1)

    @staticmethod
    def _gather_square_tokens(square_tokens: torch.Tensor, square_index: torch.Tensor) -> torch.Tensor:
        return square_tokens.gather(1, square_index.unsqueeze(-1).expand(-1, -1, square_tokens.shape[-1]))


def build_latent_reply_entropy_network_from_config(config: dict[str, Any]) -> LatentReplyEntropyNetwork:
    model_cfg = dict(config)
    return LatentReplyEntropyNetwork(
        input_channels=int(model_cfg.get("input_channels", 18)),
        num_classes=int(model_cfg.get("num_classes", 1)),
        channels=int(model_cfg.get("channels", 64)),
        hidden_dim=int(model_cfg.get("hidden_dim", 96)),
        depth=int(model_cfg.get("depth", 2)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        use_batchnorm=bool(model_cfg.get("use_batchnorm", True)),
        reply_dim=int(model_cfg.get("reply_dim", min(96, int(model_cfg.get("hidden_dim", 96))))),
        temperature=float(model_cfg.get("temperature", 0.7)),
        max_replies=int(model_cfg.get("max_replies", 96)),
        ablation=str(model_cfg.get("ablation", "none")),
    )

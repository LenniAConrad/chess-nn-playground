"""Safe-Reply Certificate Verifier for idea i191."""
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
CERTIFICATE_KINDS = (
    "move_away_king",
    "capture_attacker",
    "block_line",
    "defend_target",
    "counter_threat",
    "trade_down",
)
CERTIFICATE_KIND_COUNT = len(CERTIFICATE_KINDS)


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


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


@dataclass(frozen=True)
class SafeReplyCertificates:
    features: torch.Tensor
    kinds: torch.Tensor
    base_scores: torch.Tensor
    valid: torch.Tensor
    summary: torch.Tensor
    kind_max: torch.Tensor
    kind_count: torch.Tensor


class SafeReplyCertificateBuilder(nn.Module):
    """Generates deterministic board-local safe-reply certificate candidates."""

    feature_dim = 24
    summary_dim = 22

    def __init__(self, input_channels: int = 18, max_certificates: int = 128) -> None:
        super().__init__()
        if int(input_channels) != 18:
            raise ValueError("SafeReplyCertificateBuilder supports the simple_18 current-board tensor")
        if int(max_certificates) < CERTIFICATE_KIND_COUNT:
            raise ValueError("max_certificates must provide at least one slot per certificate kind")
        self.max_certificates = int(max_certificates)
        geom_attacks, between, king_zone, line_relation, square_features = self._build_geometry()
        self.register_buffer("geom_attacks", geom_attacks, persistent=False)
        self.register_buffer("between", between, persistent=False)
        self.register_buffer("king_zone", king_zone, persistent=False)
        self.register_buffer("line_relation", line_relation, persistent=False)
        self.register_buffer("square_features", square_features, persistent=False)

    def forward(self, board: torch.Tensor, *, count_only: bool = False) -> SafeReplyCertificates:
        piece_planes = board[:, :12].flatten(2).clamp(0.0, 1.0)
        stm = board[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0)
        occ = piece_planes.sum(dim=1).clamp(0.0, 1.0)
        attacks, rays = self._attack_relations(piece_planes, occ)

        white_piece = piece_planes[:, :6].sum(dim=1).clamp(0.0, 1.0)
        black_piece = piece_planes[:, 6:12].sum(dim=1).clamp(0.0, 1.0)
        attacker_piece = self._stm_select(white_piece, black_piece, stm)
        defender_piece = self._stm_select(black_piece, white_piece, stm)
        attacker_attacks = self._stm_select(attacks[WHITE], attacks[BLACK], stm)
        defender_attacks = self._stm_select(attacks[BLACK], attacks[WHITE], stm)

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

        attacker_attack_count = attacker_attacks.sum(dim=1)
        defender_attack_count = defender_attacks.sum(dim=1)
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
        move_away_king = defender_king_zone * (1.0 - defender_piece) * (1.0 - attacker_attacked)
        capture_attacker = attacker_piece * defender_attacked * (0.35 + attacker_value)

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
        block_line = torch.einsum(
            "stm,bst->bm",
            self.between.to(device=board.device, dtype=board.dtype),
            line_threat,
        ).clamp(0.0, 1.0) * empty

        defended_pressure = (defender_attack_count / 4.0).clamp(0.0, 1.0)
        defend_target = defender_high * attacker_attacked * defended_pressure * (0.5 + defender_value)
        counter_threat = (
            (attacker_high + attacker_king_zone).clamp(0.0, 1.0)
            * defender_attacked
            * (0.4 + attacker_value)
        )
        favorable_margin = (attacker_value[:, None, :] + 0.15 - defender_value[:, :, None]).clamp_min(0.0)
        trade_down = (
            defender_attacks
            * defender_piece[:, :, None]
            * attacker_piece[:, None, :]
            * favorable_margin
        ).sum(dim=1).clamp(0.0, 1.0)

        scores_by_kind = [
            move_away_king,
            capture_attacker,
            block_line,
            defend_target,
            counter_threat,
            trade_down,
        ]
        certificates = self._pack_certificates(
            scores_by_kind=scores_by_kind,
            board=board,
            attacker_piece=attacker_piece,
            defender_piece=defender_piece,
            attacker_value=attacker_value,
            defender_value=defender_value,
            attacker_high=attacker_high,
            defender_high=defender_high,
            attacker_attacked=attacker_attacked,
            defender_attacked=defender_attacked,
            attacker_king_zone=attacker_king_zone,
            defender_king_zone=defender_king_zone,
            block_line=block_line,
        )

        kind_scores = torch.stack(scores_by_kind, dim=1)
        kind_max = kind_scores.amax(dim=2)
        kind_sum = kind_scores.sum(dim=2) / 8.0
        kind_count = (kind_scores > 1.0e-6).to(dtype=board.dtype).sum(dim=2) / 8.0
        summary = torch.cat(
            [
                kind_max,
                kind_sum,
                kind_count,
                material_balance,
                attacker_attack_count.mean(dim=1, keepdim=True) / 8.0,
                defender_attack_count.mean(dim=1, keepdim=True) / 8.0,
                defender_king_zone.mean(dim=1, keepdim=True),
            ],
            dim=1,
        )
        features = certificates.features
        base_scores = certificates.base_scores
        if count_only:
            count_features = torch.zeros_like(features)
            count_features[:, :, :CERTIFICATE_KIND_COUNT] = F.one_hot(
                certificates.kinds.clamp_min(0),
                num_classes=CERTIFICATE_KIND_COUNT,
            ).to(dtype=features.dtype)
            count_features[:, :, CERTIFICATE_KIND_COUNT] = certificates.valid.to(dtype=features.dtype)
            features = count_features
            base_scores = certificates.valid.to(dtype=features.dtype)

        return SafeReplyCertificates(
            features=features,
            kinds=certificates.kinds,
            base_scores=base_scores,
            valid=certificates.valid,
            summary=summary,
            kind_max=kind_max,
            kind_count=kind_count,
        )

    def _pack_certificates(
        self,
        *,
        scores_by_kind: list[torch.Tensor],
        board: torch.Tensor,
        attacker_piece: torch.Tensor,
        defender_piece: torch.Tensor,
        attacker_value: torch.Tensor,
        defender_value: torch.Tensor,
        attacker_high: torch.Tensor,
        defender_high: torch.Tensor,
        attacker_attacked: torch.Tensor,
        defender_attacked: torch.Tensor,
        attacker_king_zone: torch.Tensor,
        defender_king_zone: torch.Tensor,
        block_line: torch.Tensor,
    ) -> SafeReplyCertificates:
        batch = board.shape[0]
        per_kind = max(1, self.max_certificates // CERTIFICATE_KIND_COUNT)
        extra = self.max_certificates - per_kind * CERTIFICATE_KIND_COUNT
        slots_by_kind = [per_kind + (1 if kind < extra else 0) for kind in range(CERTIFICATE_KIND_COUNT)]
        features = board.new_zeros(batch, self.max_certificates, self.feature_dim)
        kinds = torch.zeros(batch, self.max_certificates, dtype=torch.long, device=board.device)
        base_scores = board.new_zeros(batch, self.max_certificates)
        valid = torch.zeros(batch, self.max_certificates, dtype=torch.bool, device=board.device)
        square_features = self.square_features.to(device=board.device, dtype=board.dtype)
        context_tensors = [
            defender_piece,
            attacker_piece,
            (1.0 - (attacker_piece + defender_piece).clamp(0.0, 1.0)).clamp(0.0, 1.0),
            defender_value,
            attacker_value,
            defender_high,
            attacker_high,
            attacker_attacked,
            defender_attacked,
            defender_king_zone,
            attacker_king_zone,
            block_line,
        ]
        slot = 0
        for kind, scores in enumerate(scores_by_kind):
            slots = slots_by_kind[kind]
            values, indices = scores.topk(k=slots, dim=1)
            kind_onehot = board.new_zeros(batch, slots, CERTIFICATE_KIND_COUNT)
            kind_onehot[:, :, kind] = 1.0
            geom = square_features.index_select(0, indices.reshape(-1)).view(batch, slots, -1)
            gathered_context = [tensor.gather(1, indices) for tensor in context_tensors]
            context = torch.stack(gathered_context, dim=-1)
            local_features = torch.cat([kind_onehot, values.unsqueeze(-1), geom, context], dim=-1)
            features[:, slot : slot + slots] = local_features
            kinds[:, slot : slot + slots] = kind
            base_scores[:, slot : slot + slots] = values
            valid[:, slot : slot + slots] = values > 1.0e-6
            slot += slots
        return SafeReplyCertificates(
            features=features,
            kinds=kinds,
            base_scores=base_scores,
            valid=valid,
            summary=board.new_zeros(batch, self.summary_dim),
            kind_max=board.new_zeros(batch, CERTIFICATE_KIND_COUNT),
            kind_count=board.new_zeros(batch, CERTIFICATE_KIND_COUNT),
        )

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
        view_shape = [stm.shape[0], *([1] * (white_tensor.ndim - 1))]
        stm_view = stm.view(*view_shape)
        return stm_view * white_tensor + (1.0 - stm_view) * black_tensor

    @staticmethod
    def _build_geometry() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        geom = torch.zeros(6, 2, SQUARES, SQUARES)
        between = torch.zeros(SQUARES, SQUARES, SQUARES)
        king_zone = torch.zeros(SQUARES, SQUARES)
        line_relation = torch.zeros(SQUARES, SQUARES)
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
                    geom[KNIGHT, :, source, _square(target_row, target_file)] = 1.0
            for dr, df in king_steps:
                target_row, target_file = row + dr, file + df
                if _inside(target_row, target_file):
                    target = _square(target_row, target_file)
                    geom[KING, :, source, target] = 1.0
                    king_zone[source, target] = 1.0
            for color, pawn_dir in ((WHITE, -1), (BLACK, 1)):
                for df in (-1, 1):
                    target_row, target_file = row + pawn_dir, file + df
                    if _inside(target_row, target_file):
                        geom[PAWN, color, source, _square(target_row, target_file)] = 1.0
            for dirs, piece in ((bishop_dirs, BISHOP), (rook_dirs, ROOK), (bishop_dirs + rook_dirs, QUEEN)):
                for dr, df in dirs:
                    ray: list[int] = []
                    target_row, target_file = row + dr, file + df
                    while _inside(target_row, target_file):
                        target = _square(target_row, target_file)
                        geom[piece, :, source, target] = 1.0
                        line_relation[source, target] = 1.0
                        for mid in ray:
                            between[source, target, mid] = 1.0
                        ray.append(target)
                        target_row += dr
                        target_file += df
        return geom, between, king_zone, line_relation, square_features


class SafeReplyCertificateVerifier(nn.Module):
    """Verifier-style classifier that subtracts the strongest safe-reply witness."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        certificate_dim: int = 96,
        max_certificates: int = 128,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.num_classes = int(num_classes)
        self.ablation = str(ablation or "none")
        self.builder = SafeReplyCertificateBuilder(
            input_channels=int(input_channels),
            max_certificates=int(max_certificates),
        )
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
        self.square_projection = nn.Linear(int(channels), int(certificate_dim))
        self.global_projection = nn.Sequential(
            nn.LayerNorm(int(channels) * 2 + self.builder.summary_dim),
            nn.Linear(int(channels) * 2 + self.builder.summary_dim, int(certificate_dim)),
            nn.GELU(),
        )
        self.certificate_encoder = nn.Sequential(
            nn.LayerNorm(int(certificate_dim) * 2 + self.builder.feature_dim + 1),
            nn.Linear(int(certificate_dim) * 2 + self.builder.feature_dim + 1, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(int(hidden_dim), int(certificate_dim)),
            nn.GELU(),
        )
        self.validity_head = nn.Linear(int(certificate_dim), 1)
        self.strength_head = nn.Linear(int(certificate_dim), 1)
        self.positive_head = nn.Sequential(
            nn.LayerNorm(int(channels) * 2 + self.builder.summary_dim),
            nn.Linear(int(channels) * 2 + self.builder.summary_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(int(hidden_dim), self.num_classes),
        )
        self.disproof_scale = nn.Parameter(torch.ones(()))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        certificates = self.builder(board, count_only=self.ablation == "certificate_count_only")
        features = self.trunk(torch.cat([board, self._coordinate_planes(board)], dim=1))
        pooled_mean = features.mean(dim=(2, 3))
        pooled_max = features.amax(dim=(2, 3))
        global_input = torch.cat([pooled_mean, pooled_max, certificates.summary], dim=1)
        global_token = self.global_projection(global_input)
        square_tokens = self.square_projection(features.flatten(2).transpose(1, 2))
        local_tokens = self._local_square_tokens(square_tokens, certificates.features)
        expanded_global = global_token.unsqueeze(1).expand(-1, certificates.features.shape[1], -1)
        encoded = self.certificate_encoder(
            torch.cat(
                [
                    certificates.features,
                    certificates.base_scores.unsqueeze(-1),
                    local_tokens,
                    expanded_global,
                ],
                dim=-1,
            )
        )

        validity = torch.sigmoid(self.validity_head(encoded).squeeze(-1))
        if self.ablation == "no_validity_gate":
            validity = certificates.valid.to(dtype=validity.dtype)
        strength = F.softplus(self.strength_head(encoded).squeeze(-1))
        witness_score = validity * strength * certificates.valid.to(dtype=strength.dtype)
        if self.ablation == "mean_disproof_instead_of_max":
            best_disproof = _masked_mean(witness_score, certificates.valid, dim=1)
        else:
            best_disproof = witness_score.masked_fill(~certificates.valid, 0.0).amax(dim=1)
        positive_logit = self.positive_head(global_input)
        logits = positive_logit - F.softplus(self.disproof_scale) * best_disproof.unsqueeze(1)

        kind_scores = []
        for kind in range(CERTIFICATE_KIND_COUNT):
            mask = certificates.valid & (certificates.kinds == kind)
            kind_scores.append(witness_score.masked_fill(~mask, 0.0).amax(dim=1))
        kind_scores_tensor = torch.stack(kind_scores, dim=1)
        output = {
            "logits": _format_logits(logits, self.num_classes),
            "positive_puzzle_logit": _format_logits(positive_logit, self.num_classes),
            "best_disproof": best_disproof,
            "certificate_validity": validity,
            "certificate_strength": strength,
            "certificate_score": witness_score,
            "certificate_valid": certificates.valid,
            "certificate_kind": certificates.kinds,
            "safe_reply_certificate_count": certificates.valid.to(dtype=board.dtype).sum(dim=1),
            "validity_mean": _masked_mean(validity, certificates.valid, dim=1),
            "strength_mean": _masked_mean(strength, certificates.valid, dim=1),
            "move_away_king_certificate": kind_scores_tensor[:, 0],
            "capture_attacker_certificate": kind_scores_tensor[:, 1],
            "block_line_certificate": kind_scores_tensor[:, 2],
            "defend_target_certificate": kind_scores_tensor[:, 3],
            "counter_threat_certificate": kind_scores_tensor[:, 4],
            "trade_down_certificate": kind_scores_tensor[:, 5],
            "certificate_kind_count": certificates.kind_count,
            "certificate_kind_max": certificates.kind_max,
        }
        return output

    @staticmethod
    def _coordinate_planes(board: torch.Tensor) -> torch.Tensor:
        coord = torch.linspace(-1.0, 1.0, 8, device=board.device, dtype=board.dtype)
        rows = coord.view(1, 1, 8, 1).expand(board.shape[0], 1, 8, 8)
        files = coord.view(1, 1, 1, 8).expand(board.shape[0], 1, 8, 8)
        return torch.cat([rows, files], dim=1)

    @staticmethod
    def _local_square_tokens(square_tokens: torch.Tensor, certificate_features: torch.Tensor) -> torch.Tensor:
        row = (certificate_features[:, :, CERTIFICATE_KIND_COUNT + 1] * 7.0).round().long().clamp(0, 7)
        file = (certificate_features[:, :, CERTIFICATE_KIND_COUNT + 2] * 7.0).round().long().clamp(0, 7)
        square_index = (row * 8 + file).clamp(0, SQUARES - 1)
        return square_tokens.gather(1, square_index.unsqueeze(-1).expand(-1, -1, square_tokens.shape[-1]))


def build_safe_reply_certificate_verifier_from_config(config: dict[str, Any]) -> SafeReplyCertificateVerifier:
    model_cfg = dict(config)
    return SafeReplyCertificateVerifier(
        input_channels=int(model_cfg.get("input_channels", 18)),
        num_classes=int(model_cfg.get("num_classes", 1)),
        channels=int(model_cfg.get("channels", 64)),
        hidden_dim=int(model_cfg.get("hidden_dim", 96)),
        depth=int(model_cfg.get("depth", 2)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        use_batchnorm=bool(model_cfg.get("use_batchnorm", True)),
        certificate_dim=int(model_cfg.get("certificate_dim", model_cfg.get("hidden_dim", 96))),
        max_certificates=int(model_cfg.get("max_certificates", 128)),
        ablation=str(model_cfg.get("ablation", "none")),
    )

"""Exchange-Then-King Dual Stream network for idea i193."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
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


def _build_geometry() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    geom_attacks = torch.zeros(6, 2, SQUARES, SQUARES, dtype=torch.float32)
    between = torch.zeros(SQUARES, SQUARES, SQUARES, dtype=torch.float32)
    king_zone = torch.zeros(SQUARES, SQUARES, dtype=torch.float32)
    knight_offsets = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
    king_offsets = [(r, f) for r in (-1, 0, 1) for f in (-1, 0, 1) if r != 0 or f != 0]
    bishop_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    rook_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for source in range(SQUARES):
        sr, sf = _row_file(source)
        king_zone[source, source] = 1.0
        for target in range(SQUARES):
            tr, tf = _row_file(target)
            if source == target:
                continue
            if max(abs(tr - sr), abs(tf - sf)) <= 1:
                king_zone[source, target] = 1.0
            aligned = (sr == tr) or (sf == tf) or (abs(tr - sr) == abs(tf - sf))
            if aligned:
                row_step = _sign(tr - sr)
                file_step = _sign(tf - sf)
                row, file = sr + row_step, sf + file_step
                while (row, file) != (tr, tf):
                    between[source, target, _square(row, file)] = 1.0
                    row += row_step
                    file += file_step
        for color in (WHITE, BLACK):
            pawn_forward = -1 if color == WHITE else 1
            for fd in (-1, 1):
                r, f = sr + pawn_forward, sf + fd
                if _inside(r, f):
                    geom_attacks[PAWN, color, source, _square(r, f)] = 1.0
            for rd, fd in knight_offsets:
                r, f = sr + rd, sf + fd
                if _inside(r, f):
                    geom_attacks[KNIGHT, color, source, _square(r, f)] = 1.0
            for rd, fd in king_offsets:
                r, f = sr + rd, sf + fd
                if _inside(r, f):
                    geom_attacks[KING, color, source, _square(r, f)] = 1.0
            for piece, dirs in ((BISHOP, bishop_dirs), (ROOK, rook_dirs), (QUEEN, bishop_dirs + rook_dirs)):
                for rd, fd in dirs:
                    r, f = sr + rd, sf + fd
                    while _inside(r, f):
                        geom_attacks[piece, color, source, _square(r, f)] = 1.0
                        r += rd
                        f += fd
    return geom_attacks, between, king_zone


@dataclass(frozen=True)
class StreamFeatures:
    exchange: torch.Tensor
    king: torch.Tensor
    summary: torch.Tensor


class DualStreamFeatureBuilder(nn.Module):
    """Closed-form deterministic feature stacks for the exchange and king streams."""

    EXCHANGE_PLANES = 8
    KING_PLANES = 8
    SUMMARY_DIM = 8

    def __init__(self, input_channels: int = 18) -> None:
        super().__init__()
        if int(input_channels) != 18:
            raise ValueError("DualStreamFeatureBuilder requires the simple_18 current-board tensor")
        geom_attacks, between, king_zone = _build_geometry()
        self.register_buffer("geom_attacks", geom_attacks, persistent=False)
        self.register_buffer("between", between, persistent=False)
        self.register_buffer("king_zone", king_zone, persistent=False)

    def forward(self, board: torch.Tensor) -> StreamFeatures:
        device = board.device
        dtype = board.dtype
        batch = board.shape[0]
        piece_planes = board[:, :12].flatten(2).clamp(0.0, 1.0)
        stm = board[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0)
        occ = piece_planes.sum(dim=1).clamp(0.0, 1.0)

        between = self.between.to(device=device, dtype=dtype)
        king_zone_tbl = self.king_zone.to(device=device, dtype=dtype)
        geom = self.geom_attacks.to(device=device, dtype=dtype)

        blocked_count = torch.einsum("stk,bk->bst", between, occ)
        clear = (blocked_count <= 0.5).to(dtype=dtype)
        ones_clear = torch.ones_like(clear)

        attacks: dict[int, torch.Tensor] = {}
        rays: dict[int, torch.Tensor] = {}
        for color in (WHITE, BLACK):
            attack_sum = piece_planes.new_zeros(batch, SQUARES, SQUARES)
            ray_sum = piece_planes.new_zeros(batch, SQUARES, SQUARES)
            for piece in (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING):
                source = piece_planes[:, _piece_channel(color, piece)]
                line_clear = clear if piece in {BISHOP, ROOK, QUEEN} else ones_clear
                relation = source[:, :, None] * geom[piece, color].unsqueeze(0) * line_clear
                attack_sum = attack_sum + relation
                if piece in {BISHOP, ROOK, QUEEN}:
                    ray_sum = ray_sum + relation
            attacks[color] = attack_sum
            rays[color] = ray_sum

        white_piece = piece_planes[:, :6].sum(dim=1).clamp(0.0, 1.0)
        black_piece = piece_planes[:, 6:12].sum(dim=1).clamp(0.0, 1.0)
        own_piece = self._stm_select(white_piece, black_piece, stm)
        enemy_piece = self._stm_select(black_piece, white_piece, stm)

        own_attacks_per_sq = self._stm_select(attacks[WHITE], attacks[BLACK], stm).sum(dim=1)
        enemy_attacks_per_sq = self._stm_select(attacks[BLACK], attacks[WHITE], stm).sum(dim=1)
        own_rays_per_sq = self._stm_select(rays[WHITE], rays[BLACK], stm).sum(dim=1)
        enemy_rays_per_sq = self._stm_select(rays[BLACK], rays[WHITE], stm).sum(dim=1)

        values = board.new_tensor([1.0, 3.0, 3.0, 5.0, 9.0, 0.0]) / 10.0
        white_value = (piece_planes[:, :6] * values.view(1, 6, 1)).sum(dim=1)
        black_value = (piece_planes[:, 6:12] * values.view(1, 6, 1)).sum(dim=1)
        own_value = self._stm_select(white_value, black_value, stm)
        enemy_value = self._stm_select(black_value, white_value, stm)

        defender_pressure = (own_attacks_per_sq * own_piece).clamp(0.0, 4.0) / 4.0
        attacker_pressure = (enemy_attacks_per_sq * own_piece).clamp(0.0, 4.0) / 4.0

        exchange = torch.stack(
            [
                own_piece,
                enemy_piece,
                own_value,
                enemy_value,
                own_attacks_per_sq.clamp(0.0, 4.0) / 4.0,
                enemy_attacks_per_sq.clamp(0.0, 4.0) / 4.0,
                defender_pressure,
                attacker_pressure,
            ],
            dim=1,
        ).view(batch, self.EXCHANGE_PLANES, 8, 8)

        own_king = self._stm_select(
            piece_planes[:, _piece_channel(WHITE, KING)],
            piece_planes[:, _piece_channel(BLACK, KING)],
            stm,
        )
        enemy_king = self._stm_select(
            piece_planes[:, _piece_channel(BLACK, KING)],
            piece_planes[:, _piece_channel(WHITE, KING)],
            stm,
        )
        own_zone = torch.einsum("bs,st->bt", own_king, king_zone_tbl).clamp(0.0, 1.0)
        enemy_zone = torch.einsum("bs,st->bt", enemy_king, king_zone_tbl).clamp(0.0, 1.0)
        check = (own_attacks_per_sq * enemy_king).clamp(0.0, 1.0)
        attacked_by_own = own_attacks_per_sq.clamp(0.0, 1.0)
        escape = enemy_zone * (1.0 - attacked_by_own) * (1.0 - enemy_piece)
        own_line_to_zone = (own_rays_per_sq * enemy_zone).clamp(0.0, 1.0)
        enemy_line_to_zone = (enemy_rays_per_sq * own_zone).clamp(0.0, 1.0)

        king = torch.stack(
            [
                own_king,
                enemy_king,
                own_zone,
                enemy_zone,
                check,
                escape,
                own_line_to_zone,
                enemy_line_to_zone,
            ],
            dim=1,
        ).view(batch, self.KING_PLANES, 8, 8)

        material_diff = (own_value.sum(dim=1) - enemy_value.sum(dim=1))
        summary = torch.stack(
            [
                own_piece.sum(dim=1) / 16.0,
                enemy_piece.sum(dim=1) / 16.0,
                material_diff,
                own_attacks_per_sq.mean(dim=1),
                enemy_attacks_per_sq.mean(dim=1),
                own_zone.sum(dim=1) / 9.0,
                enemy_zone.sum(dim=1) / 9.0,
                check.sum(dim=1).clamp(0.0, 1.0),
            ],
            dim=1,
        )

        return StreamFeatures(exchange=exchange, king=king, summary=summary)

    @staticmethod
    def _stm_select(white_tensor: torch.Tensor, black_tensor: torch.Tensor, stm: torch.Tensor) -> torch.Tensor:
        selector = stm.view(-1, *([1] * (white_tensor.ndim - 1)))
        return selector * white_tensor + (1.0 - selector) * black_tensor


class StreamEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        channels: int,
        depth: int,
        dropout: float,
        use_batchnorm: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_c = int(input_channels)
        for _ in range(max(1, int(depth))):
            layers.append(nn.Conv2d(in_c, int(channels), kernel_size=3, padding=1, bias=not use_batchnorm))
            layers.append(nn.BatchNorm2d(int(channels)) if use_batchnorm else nn.GroupNorm(1, int(channels)))
            layers.append(nn.GELU())
            if float(dropout) > 0:
                layers.append(nn.Dropout2d(float(dropout)))
            in_c = int(channels)
        self.stack = nn.Sequential(*layers)
        self.output_dim = int(channels) * 2

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.stack(x)
        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        return h, pooled


class ExchangeThenKingDualStreamNetwork(nn.Module):
    """Bespoke dual-stream puzzle_binary network for idea i193.

    Branches a board-only encoder into an exchange stream
    (piece/value/attacker/defender features) and a king stream
    (king-zone/escape/check/line features). A learned phase router
    produces a sigmoid gate combining the per-stream logits, plus a
    small residual logit, to form the puzzle logit
    `gate * king_logit + (1 - gate) * exchange_logit + residual_logit`.
    """

    ALLOWED_ABLATIONS = ("none", "shared_stream_only", "fixed_half_gate", "king_only", "exchange_only")

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        gate_dim: int | None = None,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "ExchangeThenKingDualStreamNetwork supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError(
                "ExchangeThenKingDualStreamNetwork requires the simple_18 board tensor"
            )
        if str(ablation) not in self.ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(self.ALLOWED_ABLATIONS)}"
            )
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.ablation = str(ablation)
        self.gate_dim = int(gate_dim if gate_dim is not None else hidden_dim)

        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.feature_builder = DualStreamFeatureBuilder(input_channels=int(input_channels))

        if self.ablation == "shared_stream_only":
            shared_in = int(input_channels)
            self.exchange_encoder = StreamEncoder(
                shared_in, self.channels, self.depth, float(dropout), bool(use_batchnorm)
            )
            self.king_encoder = self.exchange_encoder
        else:
            ex_in = int(input_channels) + DualStreamFeatureBuilder.EXCHANGE_PLANES
            kg_in = int(input_channels) + DualStreamFeatureBuilder.KING_PLANES
            self.exchange_encoder = StreamEncoder(
                ex_in, self.channels, self.depth, float(dropout), bool(use_batchnorm)
            )
            self.king_encoder = StreamEncoder(
                kg_in, self.channels, self.depth, float(dropout), bool(use_batchnorm)
            )

        head_in = self.exchange_encoder.output_dim + DualStreamFeatureBuilder.SUMMARY_DIM
        self.exchange_head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, max(16, self.hidden_dim // 2)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(max(16, self.hidden_dim // 2), 1),
        )
        self.king_head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, max(16, self.hidden_dim // 2)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(max(16, self.hidden_dim // 2), 1),
        )

        joint_dim = 2 * self.exchange_encoder.output_dim + DualStreamFeatureBuilder.SUMMARY_DIM
        self.phase_router = nn.Sequential(
            nn.LayerNorm(joint_dim),
            nn.Linear(joint_dim, self.gate_dim),
            nn.GELU(),
            nn.Linear(self.gate_dim, 1),
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(joint_dim),
            nn.Linear(joint_dim, max(16, self.hidden_dim // 2)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(max(16, self.hidden_dim // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        feats = self.feature_builder(board)

        if self.ablation == "shared_stream_only":
            ex_input = board
            kg_input = board
        else:
            ex_input = torch.cat([board, feats.exchange], dim=1)
            kg_input = torch.cat([board, feats.king], dim=1)

        ex_h, ex_pool = self.exchange_encoder(ex_input)
        if self.ablation == "shared_stream_only":
            kg_h, kg_pool = ex_h, ex_pool
        else:
            kg_h, kg_pool = self.king_encoder(kg_input)

        ex_context = torch.cat([ex_pool, feats.summary], dim=1)
        kg_context = torch.cat([kg_pool, feats.summary], dim=1)
        exchange_logit = self.exchange_head(ex_context).view(-1)
        king_logit = self.king_head(kg_context).view(-1)

        joint = torch.cat([ex_pool, kg_pool, feats.summary], dim=1)
        gate_logit = self.phase_router(joint).view(-1)
        gate = torch.sigmoid(gate_logit)

        if self.ablation == "fixed_half_gate":
            gate = torch.full_like(gate, 0.5)
        elif self.ablation == "king_only":
            gate = torch.ones_like(gate)
        elif self.ablation == "exchange_only":
            gate = torch.zeros_like(gate)

        residual_logit = self.residual_head(joint).view(-1)
        logits = gate * king_logit + (1.0 - gate) * exchange_logit + residual_logit

        eps = 1.0e-6
        gate_clamped = gate.clamp(eps, 1.0 - eps)
        gate_entropy = -(gate_clamped * gate_clamped.log() + (1.0 - gate_clamped) * (1.0 - gate_clamped).log())
        stream_disagreement = (king_logit - exchange_logit).abs()
        proposal_strength = stream_disagreement * gate_entropy

        return {
            "logits": logits,
            "exchange_logit": exchange_logit,
            "king_logit": king_logit,
            "gate": gate,
            "gate_logit": gate_logit,
            "residual_logit": residual_logit,
            "gate_entropy": gate_entropy,
            "stream_disagreement": stream_disagreement,
            "exchange_pool_norm": ex_pool.pow(2).mean(dim=1),
            "king_pool_norm": kg_pool.pow(2).mean(dim=1),
            "mechanism_energy": joint.pow(2).mean(dim=1),
            "proposal_profile_strength": proposal_strength,
            "proposal_keyword_count": logits.new_full((board.shape[0],), 8.0),
        }


def build_exchange_then_king_dual_stream_from_config(
    config: dict[str, Any],
) -> ExchangeThenKingDualStreamNetwork:
    cfg = dict(config)
    return ExchangeThenKingDualStreamNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        channels=int(cfg.get("channels", cfg.get("stream_channels", 64))),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        gate_dim=int(cfg.get("gate_dim", cfg.get("hidden_dim", 96))),
        ablation=str(cfg.get("ablation", "none")),
    )

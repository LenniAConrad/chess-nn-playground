"""Traced Threat Motif Network for idea i088."""
from __future__ import annotations

import math
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
ROLE_CTRL = 0
ROLE_HIT = 1
ROLE_QUIET = 2
SQUARES = 64
K_RAW = 36
GROUP_NAMES = (
    "u_ctrl",
    "u_hit",
    "u_quiet",
    "t_ctrl",
    "t_hit",
    "t_quiet",
    "u_ray",
    "t_ray",
    "u_jump",
    "t_jump",
)
MOTIF_WORDS = (
    ("u_ctrl", "u_hit"),
    ("u_quiet", "u_hit"),
    ("u_ray", "u_hit"),
    ("u_jump", "u_hit"),
    ("t_hit", "u_hit"),
    ("t_ctrl", "u_hit"),
    ("u_quiet", "u_ctrl", "u_hit"),
    ("u_ctrl", "u_ctrl", "u_hit"),
    ("u_ray", "u_ctrl", "u_hit"),
    ("u_jump", "u_ctrl", "u_hit"),
    ("t_ctrl", "u_ctrl", "u_hit"),
    ("t_hit", "u_ctrl", "u_hit"),
    ("u_ctrl", "t_ctrl", "u_hit"),
    ("u_quiet", "t_ctrl", "u_hit"),
    ("u_ray", "t_ctrl", "u_hit"),
    ("u_jump", "t_ctrl", "u_hit"),
    ("u_quiet", "u_ctrl", "t_ctrl", "u_hit"),
    ("u_ctrl", "u_ray", "u_ctrl", "u_hit"),
    ("u_ctrl", "u_jump", "u_ctrl", "u_hit"),
    ("t_hit", "u_quiet", "u_ctrl", "u_hit"),
    ("t_ctrl", "u_quiet", "u_ctrl", "u_hit"),
    ("u_ray", "t_ctrl", "u_ctrl", "u_hit"),
    ("u_jump", "t_ctrl", "u_ctrl", "u_hit"),
    ("u_ctrl", "t_hit", "u_ctrl", "u_hit"),
)


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.view(-1) if num_classes == 1 else logits


def _square(row: int, file: int) -> int:
    return row * 8 + file


def _row_file(square: int) -> tuple[int, int]:
    return square // 8, square % 8


def _inside(row: int, file: int) -> bool:
    return 0 <= row < 8 and 0 <= file < 8


def _raw_index(color: int, piece: int, role: int) -> int:
    return ((color * 6 + piece) * 3) + role


def _piece_channel(color: int, piece: int) -> int:
    return piece if color == WHITE else 6 + piece


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _line_direction(source_row: int, source_file: int, target_row: int, target_file: int) -> tuple[int, int] | None:
    row_delta = target_row - source_row
    file_delta = target_file - source_file
    if row_delta == 0 and file_delta != 0:
        return 0, _sign(file_delta)
    if file_delta == 0 and row_delta != 0:
        return _sign(row_delta), 0
    if abs(row_delta) == abs(file_delta) and row_delta != 0:
        return _sign(row_delta), _sign(file_delta)
    return None


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.net(x))


class BoardStem(nn.Module):
    def __init__(self, in_channels: int, width: int = 96, blocks: int = 3) -> None:
        super().__init__()
        self.input = nn.Conv2d(in_channels, width, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([ResidualBlock(width) for _ in range(max(1, int(blocks)))])
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = F.gelu(self.input(x))
        for block in self.blocks:
            z = block(z)
        h = z.flatten(2).transpose(1, 2)
        return self.norm(h), z


class RelationMaskBuilder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        geom_ctrl, geom_quiet, between, pawn_mid = self._build_geometry()
        self.register_buffer("geom_ctrl", geom_ctrl, persistent=False)
        self.register_buffer("geom_quiet", geom_quiet, persistent=False)
        self.register_buffer("between", between, persistent=False)
        self.register_buffer("pawn_mid", pawn_mid, persistent=False)

    def forward(self, piece_planes: torch.Tensor) -> torch.Tensor:
        batch = piece_planes.shape[0]
        device = piece_planes.device
        dtype = piece_planes.dtype
        occ = piece_planes.sum(dim=1).clamp(0.0, 1.0)
        white_occ = piece_planes[:, :6].sum(dim=1).clamp(0.0, 1.0)
        black_occ = piece_planes[:, 6:12].sum(dim=1).clamp(0.0, 1.0)
        blocked_count = torch.einsum("ijk,bk->bij", self.between.to(dtype=dtype), occ)
        clear = (blocked_count <= 0).to(dtype=dtype)
        raw: list[torch.Tensor] = []
        for color in (WHITE, BLACK):
            enemy_occ = black_occ if color == WHITE else white_occ
            for piece in (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING):
                source = piece_planes[:, _piece_channel(color, piece)]
                ctrl_geom = self.geom_ctrl[piece, color].to(device=device, dtype=dtype)
                quiet_geom = self.geom_quiet[piece, color].to(device=device, dtype=dtype)
                line = clear if piece in {BISHOP, ROOK, QUEEN} else torch.ones_like(clear)
                ctrl = source[:, :, None] * ctrl_geom.unsqueeze(0) * line
                hit = ctrl * enemy_occ[:, None, :]
                quiet = source[:, :, None] * quiet_geom.unsqueeze(0) * line * (1.0 - occ[:, None, :])
                if piece == PAWN:
                    quiet = quiet * self._pawn_mid_clear(occ, color)
                raw.extend([ctrl, hit, quiet])
        return torch.stack(raw, dim=1).clamp(0.0, 1.0)

    def _pawn_mid_clear(self, occ: torch.Tensor, color: int) -> torch.Tensor:
        mid = self.pawn_mid[color].to(device=occ.device)
        valid = mid >= 0
        mid_safe = mid.clamp_min(0)
        mid_occ = occ[:, mid_safe.reshape(-1)].view(occ.shape[0], SQUARES, SQUARES)
        return torch.where(valid.unsqueeze(0), 1.0 - mid_occ, torch.ones_like(mid_occ))

    @staticmethod
    def _build_geometry() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        geom_ctrl = torch.zeros(6, 2, SQUARES, SQUARES, dtype=torch.float32)
        geom_quiet = torch.zeros(6, 2, SQUARES, SQUARES, dtype=torch.float32)
        between = torch.zeros(SQUARES, SQUARES, SQUARES, dtype=torch.float32)
        pawn_mid = torch.full((2, SQUARES, SQUARES), -1, dtype=torch.long)
        knight_offsets = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        king_offsets = [(r, f) for r in (-1, 0, 1) for f in (-1, 0, 1) if r != 0 or f != 0]
        bishop_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        rook_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for source in range(SQUARES):
            source_row, source_file = _row_file(source)
            for color in (WHITE, BLACK):
                pawn_forward = -1 if color == WHITE else 1
                for df in (-1, 1):
                    target_row, target_file = source_row + pawn_forward, source_file + df
                    if _inside(target_row, target_file):
                        geom_ctrl[PAWN, color, source, _square(target_row, target_file)] = 1.0
                for step_count in (1, 2):
                    target_row = source_row + step_count * pawn_forward
                    if _inside(target_row, source_file):
                        start_row = 6 if color == WHITE else 1
                        if step_count == 1 or source_row == start_row:
                            target = _square(target_row, source_file)
                            geom_quiet[PAWN, color, source, target] = 1.0
                            if step_count == 2:
                                pawn_mid[color, source, target] = _square(source_row + pawn_forward, source_file)
                for dr, df in knight_offsets:
                    target_row, target_file = source_row + dr, source_file + df
                    if _inside(target_row, target_file):
                        geom_ctrl[KNIGHT, color, source, _square(target_row, target_file)] = 1.0
                        geom_quiet[KNIGHT, color, source, _square(target_row, target_file)] = 1.0
                for dr, df in king_offsets:
                    target_row, target_file = source_row + dr, source_file + df
                    if _inside(target_row, target_file):
                        geom_ctrl[KING, color, source, _square(target_row, target_file)] = 1.0
                        geom_quiet[KING, color, source, _square(target_row, target_file)] = 1.0
                for piece, directions in ((BISHOP, bishop_dirs), (ROOK, rook_dirs), (QUEEN, bishop_dirs + rook_dirs)):
                    for dr, df in directions:
                        row, file = source_row + dr, source_file + df
                        while _inside(row, file):
                            target = _square(row, file)
                            geom_ctrl[piece, color, source, target] = 1.0
                            geom_quiet[piece, color, source, target] = 1.0
                            between_row, between_file = source_row + dr, source_file + df
                            while (between_row, between_file) != (row, file):
                                between[source, target, _square(between_row, between_file)] = 1.0
                                between_row += dr
                                between_file += df
                            row += dr
                            file += df
        return geom_ctrl, geom_quiet, between, pawn_mid


class RelationGate(nn.Module):
    def __init__(self, width: int = 96, gate_dim: int = 32, raw_count: int = K_RAW) -> None:
        super().__init__()
        self.wq = nn.Parameter(torch.randn(raw_count, width, gate_dim) * 0.02)
        self.wk = nn.Parameter(torch.randn(raw_count, width, gate_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(raw_count))

    def forward(self, h: torch.Tensor, mask_raw: torch.Tensor) -> torch.Tensor:
        q = torch.einsum("bid,kdg->bkig", h, self.wq)
        k = torch.einsum("bid,kdg->bkig", h, self.wk)
        score = torch.einsum("bkig,bkjg->bkij", q, k) / math.sqrt(q.shape[-1])
        score = score + self.bias.view(1, -1, 1, 1)
        a = mask_raw * F.softplus(score)
        return a / a.sum(dim=-1, keepdim=True).clamp_min(1.0)


class GroupMixer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.group_logits = nn.Parameter(torch.zeros(len(GROUP_NAMES), K_RAW))

    def forward(self, a_raw: torch.Tensor, stm: torch.Tensor) -> dict[str, torch.Tensor]:
        groups: dict[str, torch.Tensor] = {}
        white_u = {
            "ctrl": self._mix(a_raw, 0, self._indices(WHITE, role=ROLE_CTRL)),
            "hit": self._mix(a_raw, 1, self._indices(WHITE, role=ROLE_HIT)),
            "quiet": self._mix(a_raw, 2, self._indices(WHITE, role=ROLE_QUIET)),
            "ray": self._mix(a_raw, 6, self._indices(WHITE, pieces=(BISHOP, ROOK, QUEEN), roles=(ROLE_CTRL, ROLE_HIT))),
            "jump": self._mix(a_raw, 8, self._indices(WHITE, pieces=(PAWN, KNIGHT, KING), roles=(ROLE_CTRL,))),
        }
        black_u = {
            "ctrl": self._mix(a_raw, 0, self._indices(BLACK, role=ROLE_CTRL)),
            "hit": self._mix(a_raw, 1, self._indices(BLACK, role=ROLE_HIT)),
            "quiet": self._mix(a_raw, 2, self._indices(BLACK, role=ROLE_QUIET)),
            "ray": self._mix(a_raw, 6, self._indices(BLACK, pieces=(BISHOP, ROOK, QUEEN), roles=(ROLE_CTRL, ROLE_HIT))),
            "jump": self._mix(a_raw, 8, self._indices(BLACK, pieces=(PAWN, KNIGHT, KING), roles=(ROLE_CTRL,))),
        }
        white_t = {
            "ctrl": self._mix(a_raw, 3, self._indices(WHITE, role=ROLE_CTRL)),
            "hit": self._mix(a_raw, 4, self._indices(WHITE, role=ROLE_HIT)),
            "quiet": self._mix(a_raw, 5, self._indices(WHITE, role=ROLE_QUIET)),
            "ray": self._mix(a_raw, 7, self._indices(WHITE, pieces=(BISHOP, ROOK, QUEEN), roles=(ROLE_CTRL, ROLE_HIT))),
            "jump": self._mix(a_raw, 9, self._indices(WHITE, pieces=(PAWN, KNIGHT, KING), roles=(ROLE_CTRL,))),
        }
        black_t = {
            "ctrl": self._mix(a_raw, 3, self._indices(BLACK, role=ROLE_CTRL)),
            "hit": self._mix(a_raw, 4, self._indices(BLACK, role=ROLE_HIT)),
            "quiet": self._mix(a_raw, 5, self._indices(BLACK, role=ROLE_QUIET)),
            "ray": self._mix(a_raw, 7, self._indices(BLACK, pieces=(BISHOP, ROOK, QUEEN), roles=(ROLE_CTRL, ROLE_HIT))),
            "jump": self._mix(a_raw, 9, self._indices(BLACK, pieces=(PAWN, KNIGHT, KING), roles=(ROLE_CTRL,))),
        }
        groups["u_ctrl"] = self._stm_select(white_u["ctrl"], black_u["ctrl"], stm)
        groups["u_hit"] = self._stm_select(white_u["hit"], black_u["hit"], stm)
        groups["u_quiet"] = self._stm_select(white_u["quiet"], black_u["quiet"], stm)
        groups["t_ctrl"] = self._stm_select(black_t["ctrl"], white_t["ctrl"], stm)
        groups["t_hit"] = self._stm_select(black_t["hit"], white_t["hit"], stm)
        groups["t_quiet"] = self._stm_select(black_t["quiet"], white_t["quiet"], stm)
        groups["u_ray"] = self._stm_select(white_u["ray"], black_u["ray"], stm)
        groups["t_ray"] = self._stm_select(black_t["ray"], white_t["ray"], stm)
        groups["u_jump"] = self._stm_select(white_u["jump"], black_u["jump"], stm)
        groups["t_jump"] = self._stm_select(black_t["jump"], white_t["jump"], stm)
        return groups

    def _mix(self, a_raw: torch.Tensor, group_index: int, indices: tuple[int, ...]) -> torch.Tensor:
        selected = a_raw[:, list(indices)]
        weights = torch.softmax(self.group_logits[group_index, list(indices)], dim=0)
        return torch.einsum("k,bkij->bij", weights, selected)

    @staticmethod
    def _indices(
        color: int,
        piece: int | None = None,
        role: int | None = None,
        pieces: tuple[int, ...] | None = None,
        roles: tuple[int, ...] | None = None,
    ) -> tuple[int, ...]:
        piece_values = pieces if pieces is not None else (piece,) if piece is not None else (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)
        role_values = roles if roles is not None else (role,) if role is not None else (ROLE_CTRL, ROLE_HIT, ROLE_QUIET)
        return tuple(_raw_index(color, p, r) for p in piece_values for r in role_values)

    @staticmethod
    def _stm_select(white_tensor: torch.Tensor, black_tensor: torch.Tensor, stm: torch.Tensor) -> torch.Tensor:
        selector = stm.view(-1, 1, 1)
        return selector * white_tensor + (1.0 - selector) * black_tensor


class MotifComposer(nn.Module):
    def __init__(self, motif_words: tuple[tuple[str, ...], ...] = MOTIF_WORDS) -> None:
        super().__init__()
        self.motif_words = motif_words
        self.value_weight = nn.Parameter(torch.tensor([1.0, 3.0, 3.0, 5.0, 9.0, 10.0]) / 10.0)

    def forward(
        self,
        groups: dict[str, torch.Tensor],
        piece_planes: torch.Tensor,
        stm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        u_piece, enemy_king, enemy_value = self._boundary_vectors(piece_planes, stm)
        trace_scores: list[torch.Tensor] = []
        mass_scores: list[torch.Tensor] = []
        king_scores: list[torch.Tensor] = []
        value_scores: list[torch.Tensor] = []
        for word in self.motif_words:
            composed = self._compose(groups, word)
            trace_scores.append(self._trace(composed))
            mass_scores.append(torch.log1p(composed.sum(dim=(1, 2))))
            king_scores.append(torch.einsum("bi,bij,bj->b", u_piece, composed, enemy_king))
            value_scores.append(torch.einsum("bi,bij,bj->b", u_piece, composed, enemy_value))
        trace_vec = torch.stack(trace_scores, dim=1)
        mass_vec = torch.stack(mass_scores, dim=1)
        king_vec = torch.stack(king_scores, dim=1)
        value_vec = torch.stack(value_scores, dim=1)
        motif_features = torch.cat([trace_vec, mass_vec, king_vec, value_vec], dim=1)
        motif_scores = trace_vec + king_vec + value_vec + 0.1 * mass_vec
        u_loop2 = self._trace(torch.bmm(groups["u_ctrl"], groups["u_ctrl"]))
        t_loop2 = self._trace(torch.bmm(groups["t_ctrl"], groups["t_ctrl"]))
        interaction_loop = self._trace(torch.bmm(groups["u_ctrl"], groups["t_ctrl"]))
        monoidal = torch.stack([u_loop2, t_loop2, u_loop2 + t_loop2, interaction_loop], dim=1)
        extra = {
            "trace_closure": trace_vec.mean(dim=1),
            "open_king_mass": king_vec.mean(dim=1),
            "open_value_mass": value_vec.mean(dim=1),
            "monoidal_features": monoidal,
        }
        return torch.cat([motif_features, monoidal], dim=1), motif_scores, extra

    @staticmethod
    def _compose(groups: dict[str, torch.Tensor], word: tuple[str, ...]) -> torch.Tensor:
        composed = groups[word[0]]
        for name in word[1:]:
            composed = torch.bmm(composed, groups[name])
        return composed

    @staticmethod
    def _trace(matrix: torch.Tensor) -> torch.Tensor:
        return matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / float(matrix.shape[-1])

    def _boundary_vectors(self, piece_planes: torch.Tensor, stm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        white_piece = piece_planes[:, :6].sum(dim=1).clamp(0.0, 1.0)
        black_piece = piece_planes[:, 6:12].sum(dim=1).clamp(0.0, 1.0)
        selector = stm.view(-1, 1)
        u_piece = selector * white_piece + (1.0 - selector) * black_piece
        enemy_king = selector * piece_planes[:, _piece_channel(BLACK, KING)] + (1.0 - selector) * piece_planes[:, _piece_channel(WHITE, KING)]
        values = F.softplus(self.value_weight).to(device=piece_planes.device, dtype=piece_planes.dtype)
        values = values / values.max().clamp_min(1.0e-6)
        white_value = (piece_planes[:, :6] * values.view(1, 6, 1)).sum(dim=1)
        black_value = (piece_planes[:, 6:12] * values.view(1, 6, 1)).sum(dim=1)
        enemy_value = selector * black_value + (1.0 - selector) * white_value
        return u_piece, enemy_king, enemy_value


class ContestPullback(nn.Module):
    def forward(self, groups: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        incoming_u = groups["u_ctrl"].sum(dim=1)
        incoming_t = groups["t_ctrl"].sum(dim=1)
        contest = incoming_u * incoming_t
        contest_sum = contest.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        probs = contest / contest_sum
        entropy = -(probs * probs.clamp_min(1.0e-6).log()).sum(dim=1)
        top4 = contest.topk(4, dim=1).values.mean(dim=1)
        features = torch.stack([contest.mean(dim=1), top4, entropy], dim=1)
        return contest.view(-1, 8, 8), features


class TracedThreatMotifNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        d_model: int = 96,
        gate_dim: int = 32,
        stem_blocks: int = 3,
        head_hidden: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("TracedThreatMotifNet supports the puzzle_binary one-logit contract")
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.num_classes = int(num_classes)
        self.stem = BoardStem(int(input_channels), int(d_model), blocks=stem_blocks)
        self.mask_builder = RelationMaskBuilder()
        self.relation_gate = RelationGate(width=int(d_model), gate_dim=int(gate_dim), raw_count=K_RAW)
        self.group_mixer = GroupMixer()
        self.motif_composer = MotifComposer()
        self.contest_pullback = ContestPullback()
        motif_dim = len(MOTIF_WORDS) * 4 + 4
        head_dim = 2 * int(d_model) + motif_dim + 3
        self.head = nn.Sequential(
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, int(head_hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(int(head_hidden), max(32, int(head_hidden) // 4)),
            nn.GELU(),
            nn.Linear(max(32, int(head_hidden) // 4), 1),
        )

    def forward(self, x: torch.Tensor, *, return_diag: bool = False) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        if board.shape[1] < 13:
            raise ValueError("TracedThreatMotifNet requires at least 13 current-board channels")
        piece_planes = board[:, :12].flatten(2).clamp(0.0, 1.0)
        stm = board[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0)
        h, _ = self.stem(board)
        mask_raw = self.mask_builder(piece_planes)
        a_raw = self.relation_gate(h, mask_raw)
        groups = self.group_mixer(a_raw, stm)
        motif_features, motif_scores, motif_extra = self.motif_composer(groups, piece_planes, stm)
        contest_heatmap, contest_features = self.contest_pullback(groups)
        cnn_pool = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=1)
        logits = _format_logits(self.head(torch.cat([cnn_pool, motif_features, contest_features], dim=1)), self.num_classes)
        top_k = min(5, motif_scores.shape[1])
        output = {
            "logits": logits,
            "prob": torch.sigmoid(logits),
            "motif_scores": motif_scores,
            "top_motif_idx": motif_scores.topk(top_k, dim=1).indices,
            "contest_heatmap": contest_heatmap,
            "contest_features": contest_features,
            "trace_closure": motif_extra["trace_closure"],
            "open_king_mass": motif_extra["open_king_mass"],
            "open_value_mass": motif_extra["open_value_mass"],
            "monoidal_features": motif_extra["monoidal_features"],
            "parallel_loop2": motif_extra["monoidal_features"][:, 2],
            "interaction_loop": motif_extra["monoidal_features"][:, 3],
            "raw_relation_density": mask_raw.mean(dim=(1, 2, 3)),
            "gated_relation_density": a_raw.mean(dim=(1, 2, 3)),
            "mechanism_energy": motif_features.pow(2).mean(dim=1),
            "proposal_profile_strength": motif_scores.max(dim=1).values,
            "proposal_keyword_count": logits.new_full((board.shape[0],), 4.0),
        }
        if return_diag:
            output["group_ctrl_mass"] = torch.stack([groups["u_ctrl"].sum(dim=(1, 2)), groups["t_ctrl"].sum(dim=(1, 2))], dim=1)
        return output


def build_traced_threat_motif_network_from_config(config: dict[str, Any]) -> TracedThreatMotifNet:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    d_model = int(cfg.get("d_model", cfg.get("hidden_dim", cfg.get("channels", 96))))
    return TracedThreatMotifNet(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        d_model=d_model,
        gate_dim=int(cfg.get("gate_dim", 32)),
        stem_blocks=int(cfg.get("stem_blocks", cfg.get("depth", 3))),
        head_hidden=int(cfg.get("head_hidden", max(128, d_model * 2))),
        dropout=float(cfg.get("dropout", 0.1)),
    )

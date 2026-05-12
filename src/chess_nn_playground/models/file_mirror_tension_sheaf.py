"""File-Mirror Tension Sheaf Network for idea i028.

Bespoke implementation of the architecture described in
``ideas/registry/i028_file_mirror_tension_sheaf/architecture.md`` and
``math_thesis.md``. Builds a board-only typed directed sheaf over
pseudo-legal attack/defense/x-ray relations, runs sheaf coboundary
diffusion, reads out energy statistics, and combines them with the
statistics computed on the file-mirrored input through a learned
partial-equivariance gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


EDGE_TYPES: tuple[str, ...] = (
    "own_attack_enemy",
    "enemy_attack_own",
    "own_defense",
    "enemy_defense",
    "own_xray_enemy",
    "enemy_xray_own",
    "own_pawn_control",
    "enemy_pawn_control",
    "own_king_zone_pressure",
    "enemy_king_zone_pressure",
    "own_line_blocker_pressure",
    "enemy_line_blocker_pressure",
)
NUM_EDGE_TYPES = len(EDGE_TYPES)
TYPE_INDEX = {name: i for i, name in enumerate(EDGE_TYPES)}

ATTACK_TYPES = {
    TYPE_INDEX["own_attack_enemy"],
    TYPE_INDEX["enemy_attack_own"],
    TYPE_INDEX["own_xray_enemy"],
    TYPE_INDEX["enemy_xray_own"],
}
DEFENSE_TYPES = {TYPE_INDEX["own_defense"], TYPE_INDEX["enemy_defense"]}
KING_ZONE_TYPES = {
    TYPE_INDEX["own_king_zone_pressure"],
    TYPE_INDEX["enemy_king_zone_pressure"],
}
XRAY_TYPES = {TYPE_INDEX["own_xray_enemy"], TYPE_INDEX["enemy_xray_own"]}


@dataclass(frozen=True)
class BoardState:
    square_raw: torch.Tensor          # [B, 64, C]
    piece_type: torch.Tensor          # [B, 64] long, 0=empty 1..6
    piece_color: torch.Tensor         # [B, 64] long, 0=empty 1=white 2=black
    role: torch.Tensor                # [B, 64] long, 0=empty 1=own 2=enemy
    side_to_move: torch.Tensor        # [B] long, 1=white 2=black


@dataclass(frozen=True)
class SheafGraph:
    edge_src: torch.Tensor            # [B, E_max] long
    edge_dst: torch.Tensor            # [B, E_max] long
    edge_type: torch.Tensor           # [B, E_max] long
    edge_sign: torch.Tensor           # [B, E_max] float
    edge_mask: torch.Tensor           # [B, E_max] bool
    edge_count: torch.Tensor          # [B] float


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _masked_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    neg_large = torch.finfo(values.dtype).min / 4.0
    masked = torch.where(mask, values, values.new_full((), neg_large))
    out = masked.amax(dim=1)
    return torch.where(mask.any(dim=1), out, torch.zeros_like(out))


def _topk_pool(values: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    neg_large = torch.finfo(values.dtype).min / 4.0
    masked = torch.where(mask, values, values.new_full((), neg_large))
    k_eff = min(int(k), masked.shape[1])
    topk = masked.topk(k_eff, dim=1).values
    if k_eff < k:
        pad = values.new_zeros(values.shape[0], k - k_eff)
        topk = torch.cat([topk, pad], dim=1)
    return topk.clamp_min(0.0)


def _square_coords() -> torch.Tensor:
    square = torch.arange(64, dtype=torch.float32)
    rank = torch.div(square, 8, rounding_mode="floor")
    file = square.remainder(8)
    return torch.stack(
        [
            rank / 7.0,
            file / 7.0,
            (rank - 3.5) / 3.5,
            (file - 3.5) / 3.5,
            ((rank + file).remainder(2.0) * 2.0) - 1.0,
        ],
        dim=1,
    )


class Simple18Mirror:
    """File mirror with the simple_18 plane permutation.

    Piece planes (0..11) keep their plane identity; only files flip.
    Side-to-move plane (12) is unchanged.
    Castling planes swap kingside <-> queenside per color
    (13<->14 white, 15<->16 black). En passant plane (17) is file-flipped.
    """

    @staticmethod
    def apply(x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 18:
            raise ValueError(
                f"Simple18 file mirror expects 18 input channels, got {x.shape[1]}"
            )
        flipped = torch.flip(x, dims=[3])
        permuted = flipped.clone()
        permuted[:, 13] = flipped[:, 14]
        permuted[:, 14] = flipped[:, 13]
        permuted[:, 15] = flipped[:, 16]
        permuted[:, 16] = flipped[:, 15]
        return permuted


class EncodingAdapter(nn.Module):
    def __init__(self, input_channels: int = 18) -> None:
        super().__init__()
        if input_channels != 18:
            raise ValueError(
                "FileMirrorTensionSheafNet currently supports the simple_18 encoding."
            )
        self.input_channels = input_channels

    def forward(self, x: torch.Tensor) -> BoardState:
        batch = x.shape[0]
        device = x.device
        square_raw = x.flatten(2).transpose(1, 2)
        piece_planes = x[:, :12].clamp(0.0, 1.0)
        max_value, plane = piece_planes.max(dim=1)
        occupied = max_value >= 0.5
        piece_type_grid = (plane.remainder(6) + 1).where(occupied, torch.zeros_like(plane))
        piece_color_grid = torch.where(
            plane < 6,
            torch.ones_like(plane),
            torch.full_like(plane, 2),
        ).where(occupied, torch.zeros_like(plane))
        side_to_move = torch.where(
            x[:, 12].mean(dim=(1, 2)) >= 0.5,
            torch.ones(batch, device=device, dtype=torch.long),
            torch.full((batch,), 2, device=device, dtype=torch.long),
        )
        piece_type = piece_type_grid.flatten(1).long()
        piece_color = piece_color_grid.flatten(1).long()
        side_color = side_to_move.view(batch, 1)
        role = torch.where(
            piece_color == side_color,
            torch.ones_like(piece_color),
            torch.full_like(piece_color, 2),
        )
        role = role.where(piece_color > 0, torch.zeros_like(role))
        return BoardState(
            square_raw=square_raw,
            piece_type=piece_type,
            piece_color=piece_color,
            role=role,
            side_to_move=side_to_move,
        )


def _idx(rank: int, file: int) -> int:
    return rank * 8 + file


def _inside(rank: int, file: int) -> bool:
    return 0 <= rank < 8 and 0 <= file < 8


def _pawn_capture_steps(color: int) -> list[tuple[int, int]]:
    # tensor rank 0 is chess rank 8; whites move toward smaller tensor rank.
    if color == 1:
        return [(-1, -1), (-1, 1)]
    return [(1, -1), (1, 1)]


_KNIGHT_STEPS = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
_KING_STEPS = [(dr, df) for dr in (-1, 0, 1) for df in (-1, 0, 1) if not (dr == 0 and df == 0)]
_BISHOP_STEPS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
_ROOK_STEPS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_QUEEN_STEPS = _BISHOP_STEPS + _ROOK_STEPS


class AttackDefenseGraphBuilder(nn.Module):
    def __init__(
        self,
        max_edges: int = 2048,
        use_xray_edges: bool = True,
        use_king_zone_edges: bool = True,
    ) -> None:
        super().__init__()
        if max_edges < 1:
            raise ValueError("max_edges must be positive")
        self.max_edges = int(max_edges)
        self.use_xray_edges = bool(use_xray_edges)
        self.use_king_zone_edges = bool(use_king_zone_edges)

    @staticmethod
    def _enemy_king(piece_type: list[int], piece_color: list[int], own_color: int) -> int | None:
        enemy_color = 2 if own_color == 1 else 1
        for sq, p in enumerate(piece_type):
            if p == 6 and piece_color[sq] == enemy_color:
                return sq
        return None

    @staticmethod
    def _own_king(piece_type: list[int], piece_color: list[int], own_color: int) -> int | None:
        for sq, p in enumerate(piece_type):
            if p == 6 and piece_color[sq] == own_color:
                return sq
        return None

    @staticmethod
    def _is_king_neighbor(sq: int, king_sq: int) -> bool:
        if king_sq is None:
            return False
        sr, sf = divmod(sq, 8)
        kr, kf = divmod(king_sq, 8)
        return max(abs(sr - kr), abs(sf - kf)) <= 1

    def _classify_attack(
        self,
        source_color: int,
        source_role: int,
        target_piece: int,
        target_color: int,
    ) -> str | None:
        if target_piece == 0:
            return None
        if target_color == source_color:
            # own->own with attacking-style movement is a defense
            return "own_defense" if source_role == 1 else "enemy_defense"
        return "own_attack_enemy" if source_role == 1 else "enemy_attack_own"

    def _add(
        self,
        edges: list[tuple[int, int, int, float]],
        src: int,
        dst: int,
        type_id: int,
        sign: float,
    ) -> None:
        if src == dst or len(edges) >= self.max_edges:
            return
        edges.append((src, dst, type_id, sign))

    def _build_one(
        self,
        piece_type: list[int],
        piece_color: list[int],
        role: list[int],
    ) -> list[tuple[int, int, int, float]]:
        edges: list[tuple[int, int, int, float]] = []
        own_color = 1 if any(c == 1 and role[i] == 1 for i, c in enumerate(piece_color)) else (
            2 if any(c == 2 and role[i] == 1 for i, c in enumerate(piece_color)) else 1
        )
        # We don't know own color from role array alone if board has no own pieces;
        # pull it from any role==1 entry, otherwise fall back to side-to-move-derived color.
        for i, r in enumerate(role):
            if r == 1:
                own_color = piece_color[i]
                break

        priority: dict[str, int] = {
            "king_zone": 0,
            "attack": 1,
            "xray": 2,
            "defense": 3,
            "pawn_control": 4,
            "blocker": 5,
        }

        # Key the edges by priority bucket so we can prune deterministically when
        # the per-batch budget is exceeded.
        bucketed: list[list[tuple[int, int, int, float]]] = [[] for _ in range(6)]

        # ---- pseudo-legal pawn / knight / king attacks and slider rays ----
        for source, sp in enumerate(piece_type):
            if sp == 0:
                continue
            sc = piece_color[source]
            sr_role = role[source]
            rank, file = divmod(source, 8)

            if sp == 1:  # pawn capture geometry
                steps = _pawn_capture_steps(sc)
                for dr, df in steps:
                    rr, ff = rank + dr, file + df
                    if not _inside(rr, ff):
                        continue
                    target = _idx(rr, ff)
                    tp = piece_type[target]
                    tc = piece_color[target]
                    relation = self._classify_attack(sc, sr_role, tp, tc)
                    pawn_type = "own_pawn_control" if sr_role == 1 else "enemy_pawn_control"
                    bucketed[priority["pawn_control"]].append(
                        (source, target, TYPE_INDEX[pawn_type], +1.0)
                    )
                    if relation is not None:
                        sign = -1.0 if "defense" in relation else +1.0
                        bucketed[priority["attack"] if "attack" in relation else priority["defense"]].append(
                            (source, target, TYPE_INDEX[relation], sign)
                        )
                continue

            if sp == 2:  # knight
                for dr, df in _KNIGHT_STEPS:
                    rr, ff = rank + dr, file + df
                    if not _inside(rr, ff):
                        continue
                    target = _idx(rr, ff)
                    tp = piece_type[target]
                    tc = piece_color[target]
                    relation = self._classify_attack(sc, sr_role, tp, tc)
                    if relation is None:
                        continue
                    sign = -1.0 if "defense" in relation else +1.0
                    bucketed[priority["attack"] if "attack" in relation else priority["defense"]].append(
                        (source, target, TYPE_INDEX[relation], sign)
                    )
                continue

            if sp == 6:  # king
                for dr, df in _KING_STEPS:
                    rr, ff = rank + dr, file + df
                    if not _inside(rr, ff):
                        continue
                    target = _idx(rr, ff)
                    tp = piece_type[target]
                    tc = piece_color[target]
                    relation = self._classify_attack(sc, sr_role, tp, tc)
                    if relation is None:
                        continue
                    sign = -1.0 if "defense" in relation else +1.0
                    bucketed[priority["attack"] if "attack" in relation else priority["defense"]].append(
                        (source, target, TYPE_INDEX[relation], sign)
                    )
                continue

            # Sliders: bishop=3, rook=4, queen=5
            if sp == 3:
                steps = _BISHOP_STEPS
            elif sp == 4:
                steps = _ROOK_STEPS
            else:
                steps = _QUEEN_STEPS

            for dr, df in steps:
                rr, ff = rank + dr, file + df
                blockers = 0
                blocker_squares: list[int] = []
                while _inside(rr, ff):
                    target = _idx(rr, ff)
                    tp = piece_type[target]
                    tc = piece_color[target]
                    if blockers == 0:
                        relation = self._classify_attack(sc, sr_role, tp, tc)
                        if relation is not None:
                            sign = -1.0 if "defense" in relation else +1.0
                            bucketed[priority["attack"] if "attack" in relation else priority["defense"]].append(
                                (source, target, TYPE_INDEX[relation], sign)
                            )
                        if tp != 0:
                            blockers = 1
                            blocker_squares.append(target)
                            # line-blocker pressure: source -> blocker (always +1)
                            blocker_type = (
                                "own_line_blocker_pressure" if sr_role == 1 else "enemy_line_blocker_pressure"
                            )
                            bucketed[priority["blocker"]].append(
                                (source, target, TYPE_INDEX[blocker_type], +1.0)
                            )
                    elif blockers == 1 and self.use_xray_edges:
                        # x-ray pressure through exactly one blocker
                        xray_type = "own_xray_enemy" if sr_role == 1 else "enemy_xray_own"
                        sign_xray = +1.0
                        bucketed[priority["xray"]].append(
                            (source, target, TYPE_INDEX[xray_type], sign_xray)
                        )
                        if tp != 0:
                            blockers = 2
                            break
                    else:
                        break
                    rr += dr
                    ff += df

        # ---- king-zone pressure edges ----
        if self.use_king_zone_edges:
            enemy_king_white = self._enemy_king(piece_type, piece_color, 1)  # enemy of white = black king
            enemy_king_black = self._enemy_king(piece_type, piece_color, 2)
            own_king_sq = self._own_king(piece_type, piece_color, own_color)
            enemy_king_sq = self._enemy_king(piece_type, piece_color, own_color)
            for source, sp in enumerate(piece_type):
                if sp == 0:
                    continue
                sc = piece_color[source]
                sr_role = role[source]
                target_king = enemy_king_sq if sr_role == 1 else own_king_sq
                if target_king is None:
                    continue
                if not self._is_king_neighbor(source, target_king):
                    # only mark direct neighbors of the targeted king as king-zone
                    # pressure; this keeps the tensor small and matches the
                    # "king-adjacent attack imbalance" hypothesis.
                    continue
                ktype = (
                    "own_king_zone_pressure" if sr_role == 1 else "enemy_king_zone_pressure"
                )
                bucketed[priority["king_zone"]].append(
                    (source, target_king, TYPE_INDEX[ktype], +1.0)
                )

        # priority-ordered concatenation, capped at max_edges
        for bucket in bucketed:
            for edge in bucket:
                if len(edges) >= self.max_edges:
                    return edges
                edges.append(edge)
        return edges

    def forward(self, board: BoardState) -> SheafGraph:
        device = board.piece_type.device
        batch = board.piece_type.shape[0]
        edge_src = torch.zeros(batch, self.max_edges, dtype=torch.long, device=device)
        edge_dst = torch.zeros_like(edge_src)
        edge_type = torch.zeros_like(edge_src)
        edge_sign = torch.zeros(batch, self.max_edges, dtype=torch.float32, device=device)
        edge_mask = torch.zeros(batch, self.max_edges, dtype=torch.bool, device=device)
        edge_counts: list[int] = []

        piece_rows = board.piece_type.detach().cpu().tolist()
        color_rows = board.piece_color.detach().cpu().tolist()
        role_rows = board.role.detach().cpu().tolist()

        for b in range(batch):
            edges = self._build_one(piece_rows[b], color_rows[b], role_rows[b])
            count = min(len(edges), self.max_edges)
            edge_counts.append(count)
            if not count:
                continue
            srcs = torch.tensor([e[0] for e in edges[:count]], dtype=torch.long, device=device)
            dsts = torch.tensor([e[1] for e in edges[:count]], dtype=torch.long, device=device)
            types = torch.tensor([e[2] for e in edges[:count]], dtype=torch.long, device=device)
            signs = torch.tensor([e[3] for e in edges[:count]], dtype=torch.float32, device=device)
            edge_src[b, :count] = srcs
            edge_dst[b, :count] = dsts
            edge_type[b, :count] = types
            edge_sign[b, :count] = signs
            edge_mask[b, :count] = True

        return SheafGraph(
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_type=edge_type,
            edge_sign=edge_sign,
            edge_mask=edge_mask,
            edge_count=torch.tensor(edge_counts, dtype=torch.float32, device=device),
        )


class NodeInitializer(nn.Module):
    def __init__(self, input_channels: int, d_model: int, stalk_dim: int, dropout: float) -> None:
        super().__init__()
        self.register_buffer("square_coords", _square_coords(), persistent=False)
        in_dim = input_channels + 7 + 3 + 3 + 5 + 1
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(d_model, stalk_dim),
            nn.LayerNorm(stalk_dim),
        )

    def forward(self, board: BoardState) -> torch.Tensor:
        batch = board.square_raw.shape[0]
        dtype = board.square_raw.dtype
        device = board.square_raw.device
        piece = nn.functional.one_hot(board.piece_type.clamp(0, 6), num_classes=7).to(dtype=dtype)
        color = nn.functional.one_hot(board.piece_color.clamp(0, 2), num_classes=3).to(dtype=dtype)
        role = nn.functional.one_hot(board.role.clamp(0, 2), num_classes=3).to(dtype=dtype)
        coords = self.square_coords.to(device=device, dtype=dtype).unsqueeze(0).expand(batch, -1, -1)
        side = (
            (board.side_to_move == 1)
            .to(dtype=dtype)
            .view(batch, 1, 1)
            .expand(batch, 64, 1)
        )
        return self.proj(torch.cat([board.square_raw, piece, color, role, coords, side], dim=-1))


class TypedSheafDiffusionLayer(nn.Module):
    def __init__(
        self,
        stalk_dim: int,
        num_edge_types: int = NUM_EDGE_TYPES,
        eta_max: float = 0.2,
        use_edge_gate: bool = True,
        node_hidden: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.stalk_dim = int(stalk_dim)
        self.num_edge_types = int(num_edge_types)
        self.eta_max = float(eta_max)
        scale = float(stalk_dim) ** -0.5
        # Initialize near identity per the math thesis: A_src ~ I + small noise.
        eye = torch.eye(stalk_dim).expand(num_edge_types, stalk_dim, stalk_dim).clone()
        self.A_src = nn.Parameter(eye + scale * torch.randn(num_edge_types, stalk_dim, stalk_dim))
        self.A_dst = nn.Parameter(eye + scale * torch.randn(num_edge_types, stalk_dim, stalk_dim))
        self.use_edge_gate = bool(use_edge_gate)
        if self.use_edge_gate:
            self.edge_type_embed = nn.Embedding(num_edge_types, max(8, stalk_dim))
            self.edge_gate = nn.Sequential(
                nn.Linear(2 * stalk_dim + max(8, stalk_dim) + 1, max(16, stalk_dim)),
                nn.GELU(),
                nn.Linear(max(16, stalk_dim), 1),
            )
        hidden = int(node_hidden or max(stalk_dim * 2, 16))
        self.node_mlp = nn.Sequential(
            nn.Linear(stalk_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, stalk_dim),
        )
        self.norm = nn.LayerNorm(stalk_dim)
        # eta parameterized through sigmoid * eta_max for stability.
        self.eta_logit = nn.Parameter(torch.zeros(1))

    def _gather(self, h: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return h.gather(1, idx.unsqueeze(-1).expand(-1, -1, h.shape[-1]))

    def _scatter(self, vals: torch.Tensor, idx: torch.Tensor, n_nodes: int) -> torch.Tensor:
        out = vals.new_zeros(vals.shape[0], n_nodes, vals.shape[-1])
        return out.scatter_add(1, idx.unsqueeze(-1).expand(-1, -1, vals.shape[-1]), vals)

    def forward(self, h: torch.Tensor, graph: SheafGraph) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, n_nodes, d = h.shape
        e_max = graph.edge_src.shape[1]
        # edge restriction maps
        type_idx = graph.edge_type.clamp(0, self.num_edge_types - 1)
        A_src = self.A_src[type_idx]                  # [B, E, d, d]
        A_dst = self.A_dst[type_idx]                  # [B, E, d, d]
        h_s = self._gather(h, graph.edge_src)         # [B, E, d]
        h_t = self._gather(h, graph.edge_dst)         # [B, E, d]
        # coboundary residual: A_dst h_t - sigma(e) A_src h_s
        As_hs = torch.einsum("berd,bed->ber", A_src, h_s)
        At_ht = torch.einsum("berd,bed->ber", A_dst, h_t)
        residual = At_ht - graph.edge_sign.unsqueeze(-1) * As_hs
        mask_f = graph.edge_mask.to(dtype=residual.dtype).unsqueeze(-1)
        residual = residual * mask_f

        if self.use_edge_gate:
            type_emb = self.edge_type_embed(type_idx)
            sign_feat = graph.edge_sign.unsqueeze(-1)
            gate_in = torch.cat([h_s, h_t, type_emb, sign_feat], dim=-1)
            gate = torch.sigmoid(self.edge_gate(gate_in)).squeeze(-1)
        else:
            gate = torch.ones(batch, e_max, device=h.device, dtype=h.dtype)
        gate = gate * graph.edge_mask.to(dtype=h.dtype)

        # weighted residual
        weighted = residual * gate.unsqueeze(-1)
        # divergence: delta^T W delta
        # contribution to source: -sigma * A_src^T weighted
        div_src = -graph.edge_sign.unsqueeze(-1) * torch.einsum("berd,ber->bed", A_src, weighted)
        div_dst = torch.einsum("berd,ber->bed", A_dst, weighted)
        out_src = self._scatter(div_src, graph.edge_src, n_nodes)
        out_dst = self._scatter(div_dst, graph.edge_dst, n_nodes)
        # delta^T accumulates at both endpoints; total node-side gradient
        div = out_dst + out_src

        eta = torch.sigmoid(self.eta_logit) * self.eta_max
        h_next = self.norm(h - eta * div + self.node_mlp(h))

        edge_energy = (residual * residual).sum(dim=-1) * gate
        return h_next, edge_energy, gate


class SheafEnergyReadout(nn.Module):
    """Computes the sheaf-energy statistic vector ``s_F``.

    Layout is fixed and exposed via :pyattr:`output_dim` so the partial-mirror
    gate can re-use it. The vector has the structure described in
    architecture.md section "SheafEnergyReadout"::

        [ mean per type | max per type | top-k global pooled | king zone own/enemy
          | concentration ratio | divergence-norm mean/max | node mean | node max ]
    """

    def __init__(self, stalk_dim: int, top_k: int = 8) -> None:
        super().__init__()
        self.stalk_dim = int(stalk_dim)
        self.top_k = int(top_k)
        # stats: 2*T + k + 2 + 1 + 2 + 2*d
        self.output_dim = 2 * NUM_EDGE_TYPES + self.top_k + 2 + 1 + 2 + 2 * self.stalk_dim

    def forward(
        self,
        node_state: torch.Tensor,
        per_layer_energy: list[torch.Tensor],
        graph: SheafGraph,
    ) -> torch.Tensor:
        # Use the LAST layer's edge energies for the per-type stats; this
        # matches the math thesis where the readout reads "final" energies.
        energy = per_layer_energy[-1]
        mask = graph.edge_mask
        dtype = energy.dtype

        means = []
        maxes = []
        for t in range(NUM_EDGE_TYPES):
            type_mask = mask & (graph.edge_type == t)
            means.append(_masked_mean(energy, type_mask))
            maxes.append(_masked_max(energy, type_mask))
        type_means = torch.stack(means, dim=1)  # [B, T]
        type_maxes = torch.stack(maxes, dim=1)

        topk = _topk_pool(energy, mask, self.top_k)  # [B, k]

        own_kz_mask = mask & (graph.edge_type == TYPE_INDEX["own_king_zone_pressure"])
        enemy_kz_mask = mask & (graph.edge_type == TYPE_INDEX["enemy_king_zone_pressure"])
        own_kz = _masked_mean(energy, own_kz_mask).unsqueeze(-1)
        enemy_kz = _masked_mean(energy, enemy_kz_mask).unsqueeze(-1)

        total_energy = (energy * mask.to(dtype=dtype)).sum(dim=1)
        topk_sum = topk.sum(dim=1)
        concentration = (topk_sum / total_energy.clamp_min(1.0e-6)).unsqueeze(-1)

        # divergence norm proxy via per-type mean of energies (stable surrogate).
        div_norm = energy * mask.to(dtype=dtype)
        div_norm_mean = div_norm.mean(dim=1, keepdim=True)
        div_norm_max = div_norm.amax(dim=1, keepdim=True)

        node_mean = node_state.mean(dim=1)
        node_max = node_state.amax(dim=1)

        return torch.cat(
            [type_means, type_maxes, topk, own_kz, enemy_kz, concentration, div_norm_mean, div_norm_max, node_mean, node_max],
            dim=1,
        )


# index permutation Pi_M for the type-stat segments under file mirror.
def _type_mirror_index() -> list[int]:
    # The own_*/enemy_* split is preserved by file mirror (it does not flip
    # color). All edge types map to themselves, so Pi_M restricted to the
    # type-stat segments is identity. Statistics are therefore equivariant
    # (in fact invariant) under the file-mirror permutation.
    return list(range(NUM_EDGE_TYPES))


class FileMirrorPartialGate(nn.Module):
    def __init__(self, stat_dim: int, gate_mode: str = "scalar") -> None:
        super().__init__()
        self.gate_mode = str(gate_mode)
        if self.gate_mode not in {"scalar", "vector"}:
            raise ValueError("gate_mode must be 'scalar' or 'vector'")
        gate_out = 1 if self.gate_mode == "scalar" else stat_dim
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(3 * stat_dim),
            nn.Linear(3 * stat_dim, max(32, stat_dim)),
            nn.GELU(),
            nn.Linear(max(32, stat_dim), gate_out),
        )

    def forward(self, s: torch.Tensor, s_mirror: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delta = (s - s_mirror).abs()
        rho = torch.sigmoid(self.gate_mlp(torch.cat([s, s_mirror, delta], dim=-1)))
        if self.gate_mode == "scalar":
            gated = rho * delta
            rho_feature = rho
        else:
            gated = rho * delta
            rho_feature = rho
        return gated, rho_feature, delta


class FileMirrorTensionSheafNet(nn.Module):
    """Bespoke File-Mirror Tension Sheaf classifier.

    See ``ideas/registry/i028_file_mirror_tension_sheaf/architecture.md`` for the
    full component description.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        stalk_dim: int = 16,
        num_sheaf_layers: int = 3,
        e_max: int = 2048,
        dropout: float = 0.1,
        eta_max: float = 0.2,
        top_k_energy: int = 8,
        gate_mode: str = "scalar",
        use_xray_edges: bool = True,
        use_king_zone_edges: bool = True,
        use_file_mirror_gate: bool = True,
        head_hidden_dim: int = 96,
        d_model: int = 48,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.use_file_mirror_gate = bool(use_file_mirror_gate)
        self.adapter = EncodingAdapter(input_channels=input_channels)
        self.graph_builder = AttackDefenseGraphBuilder(
            max_edges=e_max,
            use_xray_edges=use_xray_edges,
            use_king_zone_edges=use_king_zone_edges,
        )
        self.node_init = NodeInitializer(
            input_channels=input_channels,
            d_model=d_model,
            stalk_dim=stalk_dim,
            dropout=dropout,
        )
        self.layers = nn.ModuleList(
            [
                TypedSheafDiffusionLayer(
                    stalk_dim=stalk_dim,
                    eta_max=eta_max,
                    dropout=dropout,
                )
                for _ in range(max(1, int(num_sheaf_layers)))
            ]
        )
        self.readout = SheafEnergyReadout(stalk_dim=stalk_dim, top_k=top_k_energy)
        self.gate = FileMirrorPartialGate(stat_dim=self.readout.output_dim, gate_mode=gate_mode)

        gate_dim = self.readout.output_dim if gate_mode == "vector" else 1
        feature_dim = (
            self.readout.output_dim                  # s
            + self.readout.output_dim                # rho * delta
            + gate_dim                               # rho
            + 2 * stalk_dim                          # pooled node features
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(head_hidden_dim, self.num_classes),
        )

    def _encode_once(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], SheafGraph]:
        board = self.adapter(x)
        graph = self.graph_builder(board)
        h = self.node_init(board)
        energies: list[torch.Tensor] = []
        for layer in self.layers:
            h, energy, _gate = layer(h, graph)
            energies.append(energy)
        s = self.readout(h, energies, graph)
        node_pool = torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=-1)
        return s, node_pool, energies, graph

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        s, node_pool, energies, graph = self._encode_once(x)
        if self.use_file_mirror_gate:
            x_m = Simple18Mirror.apply(x)
            s_m, _, _, _ = self._encode_once(x_m)
        else:
            s_m = s
        gated, rho, delta = self.gate(s, s_m)
        feat = torch.cat([s, gated, rho, node_pool], dim=-1)
        logits = _format_logits(self.classifier(feat), self.num_classes)

        last_energy = energies[-1]
        mask = graph.edge_mask
        dtype = x.dtype
        own_kz_mask = mask & (graph.edge_type == TYPE_INDEX["own_king_zone_pressure"])
        enemy_kz_mask = mask & (graph.edge_type == TYPE_INDEX["enemy_king_zone_pressure"])
        attack_mask = mask & (
            (graph.edge_type == TYPE_INDEX["own_attack_enemy"])
            | (graph.edge_type == TYPE_INDEX["enemy_attack_own"])
        )
        defense_mask = mask & (
            (graph.edge_type == TYPE_INDEX["own_defense"])
            | (graph.edge_type == TYPE_INDEX["enemy_defense"])
        )
        xray_mask = mask & (
            (graph.edge_type == TYPE_INDEX["own_xray_enemy"])
            | (graph.edge_type == TYPE_INDEX["enemy_xray_own"])
        )

        diagnostics = {
            "logits": logits,
            "sheaf_tension": _masked_mean(last_energy, mask),
            "transport_imbalance": (delta).mean(dim=1),
            "symmetry_residual": delta.mean(dim=1),
            "topology_pressure": _masked_max(last_energy, mask),
            "ray_language_energy": _masked_mean(last_energy, xray_mask),
            "information_surprisal": torch.log1p(_masked_mean(last_energy, mask)),
            "sparse_certificate_energy": _topk_pool(last_energy, mask, k=self.readout.top_k).sum(dim=1),
            "rank_file_imbalance": delta.mean(dim=1),
            "king_ring_pressure": _masked_mean(last_energy, own_kz_mask | enemy_kz_mask),
            "reply_pressure": _masked_mean(last_energy, attack_mask),
            "defense_gap": _masked_mean(last_energy, defense_mask),
            "mechanism_energy": torch.log1p(_masked_mean(last_energy, mask)),
            "proposal_profile_strength": graph.edge_count.to(dtype=dtype).clamp_min(1.0).log1p(),
            "proposal_keyword_count": logits.new_full((x.shape[0],), float(NUM_EDGE_TYPES)),
            "mirror_gate_rho": rho.mean(dim=-1) if rho.ndim > 1 else rho,
        }
        return diagnostics


def build_file_mirror_tension_sheaf_from_config(config: dict[str, Any]) -> FileMirrorTensionSheafNet:
    """Build a FileMirrorTensionSheafNet from a flat model config dict."""
    return FileMirrorTensionSheafNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        stalk_dim=int(config.get("stalk_dim", 16)),
        num_sheaf_layers=int(config.get("num_sheaf_layers", config.get("depth", 3))),
        e_max=int(config.get("e_max", config.get("max_edges", 2048))),
        dropout=float(config.get("dropout", 0.1)),
        eta_max=float(config.get("eta_max", 0.2)),
        top_k_energy=int(config.get("top_k_energy", config.get("topk_energy", 8))),
        gate_mode=str(config.get("gate_mode", "scalar")),
        use_xray_edges=bool(config.get("use_xray_edges", True)),
        use_king_zone_edges=bool(config.get("use_king_zone_edges", True)),
        use_file_mirror_gate=bool(config.get("use_file_mirror_gate", True)),
        head_hidden_dim=int(config.get("head_hidden_dim", config.get("hidden_dim", 96))),
        d_model=int(config.get("d_model", config.get("channels", 48))),
    )

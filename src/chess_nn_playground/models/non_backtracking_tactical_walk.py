from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


_RELATION_TYPES: tuple[str, ...] = (
    "enemy_piece_attack",
    "friendly_piece_protect",
    "enemy_king_zone_attack",
    "own_king_zone_protect",
)
NUM_RELATION_TYPES = len(_RELATION_TYPES)
NUM_TYPE_PAIRS = NUM_RELATION_TYPES * NUM_RELATION_TYPES

NUM_PIECE_TYPES = 6
F_EDGE = 32

_KNIGHT_OFFSETS: tuple[tuple[int, int], ...] = (
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1),
)
_KING_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)
_BISHOP_DIRS: tuple[tuple[int, int], ...] = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_ROOK_DIRS: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass
class _DecodedBoard:
    pieces: list[tuple[int, int, int, int]]  # (square_idx, color, piece_type, row, col) -- but tuple[4]
    occupancy: list[list[int | None]]
    side_to_move_white: bool
    king_squares: dict[int, tuple[int, int] | None]


@dataclass
class _BoardEdges:
    edge_features: torch.Tensor
    edge_mask: torch.Tensor
    edge_relation: torch.Tensor
    transition_src: torch.Tensor
    transition_dst: torch.Tensor
    transition_type_pair: torch.Tensor
    transition_mask: torch.Tensor
    edge_overflow: int = 0
    transition_overflow: int = 0


def _decode_simple18_board(x_b: torch.Tensor) -> tuple[list[list[tuple[int, int] | None]], bool, dict[int, tuple[int, int] | None]]:
    """Decode a single (18, 8, 8) tensor into a square-indexed occupancy grid.

    Returns (occupancy[r][c] = (color, piece_type) | None, side_to_move_white, king_squares).
    """
    pieces_planes = x_b[:12].detach().cpu()
    side_to_move_white = bool(x_b[12].detach().mean().item() > 0.5)
    occupancy: list[list[tuple[int, int] | None]] = [[None] * 8 for _ in range(8)]
    king_squares: dict[int, tuple[int, int] | None] = {0: None, 1: None}
    for plane in range(12):
        plane_t = pieces_planes[plane]
        nz = plane_t.nonzero(as_tuple=False)
        color = 0 if plane < 6 else 1
        ptype = plane if plane < 6 else plane - 6
        for entry in nz.tolist():
            r, c = entry[0], entry[1]
            if 0 <= r < 8 and 0 <= c < 8 and occupancy[r][c] is None:
                occupancy[r][c] = (color, ptype)
                if ptype == 5:
                    king_squares[color] = (r, c)
    return occupancy, side_to_move_white, king_squares


def _attacked_squares(
    r: int,
    c: int,
    color: int,
    piece_type: int,
    occupancy: list[list[tuple[int, int] | None]],
) -> list[tuple[int, int]]:
    """Return list of squares attacked or protected by the piece at (r,c).

    The attacker reaches a blocker square (i.e. attacks pieces it bumps into); the ray
    stops at the first occupied square in each direction for sliders.
    """
    attacks: list[tuple[int, int]] = []
    if piece_type == 0:
        # pawn: white (color=0) attacks dr=-1, black (color=1) attacks dr=+1
        dr = -1 if color == 0 else 1
        for dc in (-1, 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                attacks.append((nr, nc))
        return attacks
    if piece_type == 1:
        # knight
        for dr, dc in _KNIGHT_OFFSETS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                attacks.append((nr, nc))
        return attacks
    if piece_type == 5:
        # king
        for dr, dc in _KING_OFFSETS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                attacks.append((nr, nc))
        return attacks

    if piece_type == 2:
        dirs = _BISHOP_DIRS
    elif piece_type == 3:
        dirs = _ROOK_DIRS
    elif piece_type == 4:
        dirs = _BISHOP_DIRS + _ROOK_DIRS
    else:
        dirs = ()
    for dr, dc in dirs:
        nr, nc = r + dr, c + dc
        while 0 <= nr < 8 and 0 <= nc < 8:
            attacks.append((nr, nc))
            if occupancy[nr][nc] is not None:
                break
            nr += dr
            nc += dc
    return attacks


def _king_zone(king_square: tuple[int, int] | None) -> list[tuple[int, int]]:
    if king_square is None:
        return []
    kr, kc = king_square
    zone = [(kr, kc)]
    for dr, dc in _KING_OFFSETS:
        nr, nc = kr + dr, kc + dc
        if 0 <= nr < 8 and 0 <= nc < 8:
            zone.append((nr, nc))
    return zone


def _build_edge_feature(
    relation: int,
    src_color: int,
    src_piece_type: int,
    src_rc: tuple[int, int],
    tgt_rc: tuple[int, int],
    tgt_color: int,
    tgt_piece_type: int,
    tgt_is_virtual: bool,
    side_to_move_white: bool,
) -> list[float]:
    feat = [0.0] * F_EDGE
    # 0..3: relation one-hot
    feat[relation] = 1.0
    # 4..9: source piece type one-hot
    feat[4 + src_piece_type] = 1.0
    # 10..15: target piece type one-hot (zeroed for virtual)
    if not tgt_is_virtual:
        feat[10 + tgt_piece_type] = 1.0
    # 16: source color relative to side-to-move (1.0 if source is side-to-move)
    stm = 0 if side_to_move_white else 1
    feat[16] = 1.0 if src_color == stm else 0.0
    # 17: target color relative (only for occupied)
    feat[17] = (1.0 if (not tgt_is_virtual and tgt_color == stm) else 0.0)
    # 18: virtual target flag, 19: occupied target flag
    feat[18] = 1.0 if tgt_is_virtual else 0.0
    feat[19] = 0.0 if tgt_is_virtual else 1.0
    # 20..23: signed displacement and magnitudes (rank-relative to side-to-move)
    sr, sc = src_rc
    tr, tc = tgt_rc
    if not side_to_move_white:
        # mirror ranks so "forward" is consistent
        sr_eff = 7 - sr
        tr_eff = 7 - tr
    else:
        sr_eff = sr
        tr_eff = tr
    dr = (tr_eff - sr_eff) / 7.0
    dc = (tc - sc) / 7.0
    feat[20] = dr
    feat[21] = dc
    feat[22] = abs(dr)
    feat[23] = abs(dc)
    # 24: Chebyshev distance / 7
    feat[24] = max(abs(tr_eff - sr_eff), abs(tc - sc)) / 7.0
    # 25..28: side-to-move-relative source rank/file as bins
    feat[25] = sr_eff / 7.0
    feat[26] = sc / 7.0
    feat[27] = tr_eff / 7.0
    feat[28] = tc / 7.0
    # 29: source-target same color flag (only meaningful for occupied targets)
    feat[29] = 1.0 if (not tgt_is_virtual and tgt_color == src_color) else 0.0
    # 30: side-to-move bit (white=1)
    feat[30] = 1.0 if side_to_move_white else 0.0
    # 31: bias
    feat[31] = 1.0
    return feat


def build_board_edges(
    x_b: torch.Tensor,
    *,
    edge_max: int,
    transition_max: int,
    use_king_zone_virtual_nodes: bool,
    ablation_mode: str,
    rng: torch.Generator | None = None,
) -> _BoardEdges:
    """Construct attack/protection edges and non-backtracking transitions for one board.

    Pads to (edge_max,) and (transition_max,). Returns CPU tensors.
    """
    occupancy, side_to_move_white, king_squares = _decode_simple18_board(x_b)
    enemy_zone_squares: dict[tuple[int, int], int] = {}
    own_zone_squares: dict[tuple[int, int], int] = {}
    if use_king_zone_virtual_nodes:
        stm_color = 0 if side_to_move_white else 1
        opp_color = 1 - stm_color
        for sq in _king_zone(king_squares.get(opp_color)):
            enemy_zone_squares[sq] = 1
        for sq in _king_zone(king_squares.get(stm_color)):
            own_zone_squares[sq] = 1

    # Node identity: occupied squares get one node id. For non-backtracking, we need to
    # know the "origin" and "terminal" of each edge in the same node space.
    occupied_squares: list[tuple[int, int]] = []
    occ_node_id: dict[tuple[int, int], int] = {}
    for r in range(8):
        for c in range(8):
            if occupancy[r][c] is not None:
                occ_node_id[(r, c)] = len(occupied_squares)
                occupied_squares.append((r, c))
    next_node_id = len(occupied_squares)
    enemy_zone_node_id: dict[tuple[int, int], int] = {}
    own_zone_node_id: dict[tuple[int, int], int] = {}
    for sq in enemy_zone_squares:
        enemy_zone_node_id[sq] = next_node_id
        next_node_id += 1
    for sq in own_zone_squares:
        own_zone_node_id[sq] = next_node_id
        next_node_id += 1

    # Edge list: each entry (origin_node, terminal_node, terminal_is_virtual, relation, feature)
    edges_origin: list[int] = []
    edges_terminal: list[int] = []
    edges_relation: list[int] = []
    edges_features: list[list[float]] = []
    edges_dist: list[float] = []  # for safe truncation ordering

    for (r, c) in occupied_squares:
        cell = occupancy[r][c]
        if cell is None:
            continue
        src_color, src_ptype = cell
        attacked = _attacked_squares(r, c, src_color, src_ptype, occupancy)
        seen_virtual_enemy: set[tuple[int, int]] = set()
        seen_virtual_own: set[tuple[int, int]] = set()
        for (nr, nc) in attacked:
            tgt = occupancy[nr][nc]
            # occupied target edges
            if tgt is not None:
                tgt_color, tgt_ptype = tgt
                relation = 0 if tgt_color != src_color else 1
                feat = _build_edge_feature(
                    relation,
                    src_color,
                    src_ptype,
                    (r, c),
                    (nr, nc),
                    tgt_color,
                    tgt_ptype,
                    tgt_is_virtual=False,
                    side_to_move_white=side_to_move_white,
                )
                edges_origin.append(occ_node_id[(r, c)])
                edges_terminal.append(occ_node_id[(nr, nc)])
                edges_relation.append(relation)
                edges_features.append(feat)
                edges_dist.append(max(abs(nr - r), abs(nc - c)))
            # virtual king-zone target edges (always added if zone square is included,
            # even if occupied; this is by construction independent of occupancy)
            if (nr, nc) in enemy_zone_squares and (nr, nc) not in seen_virtual_enemy:
                seen_virtual_enemy.add((nr, nc))
                feat = _build_edge_feature(
                    2,
                    src_color,
                    src_ptype,
                    (r, c),
                    (nr, nc),
                    0,
                    0,
                    tgt_is_virtual=True,
                    side_to_move_white=side_to_move_white,
                )
                edges_origin.append(occ_node_id[(r, c)])
                edges_terminal.append(enemy_zone_node_id[(nr, nc)])
                edges_relation.append(2)
                edges_features.append(feat)
                edges_dist.append(max(abs(nr - r), abs(nc - c)))
            if (nr, nc) in own_zone_squares and (nr, nc) not in seen_virtual_own:
                seen_virtual_own.add((nr, nc))
                feat = _build_edge_feature(
                    3,
                    src_color,
                    src_ptype,
                    (r, c),
                    (nr, nc),
                    0,
                    0,
                    tgt_is_virtual=True,
                    side_to_move_white=side_to_move_white,
                )
                edges_origin.append(occ_node_id[(r, c)])
                edges_terminal.append(own_zone_node_id[(nr, nc)])
                edges_relation.append(3)
                edges_features.append(feat)
                edges_dist.append(max(abs(nr - r), abs(nc - c)))

    # Deterministic safe truncation order: occupied-target tactical edges first,
    # then king-zone edges, then by ascending distance, then origin id, then terminal id.
    indexed = list(range(len(edges_origin)))
    def _sort_key(i: int) -> tuple[int, float, int, int]:
        rel = edges_relation[i]
        # occupied-target = 0,1; king-zone = 2,3
        primary = 0 if rel < 2 else 1
        return (primary, edges_dist[i], edges_origin[i], edges_terminal[i])
    indexed.sort(key=_sort_key)
    edge_overflow = max(0, len(indexed) - edge_max)
    indexed = indexed[:edge_max]
    new_index_of: dict[int, int] = {old: new for new, old in enumerate(indexed)}

    n_edges = len(indexed)
    edge_features = torch.zeros((edge_max, F_EDGE), dtype=torch.float32)
    edge_mask = torch.zeros((edge_max,), dtype=torch.bool)
    edge_relation = torch.zeros((edge_max,), dtype=torch.long)
    for new_idx, old_idx in enumerate(indexed):
        edge_features[new_idx] = torch.tensor(edges_features[old_idx], dtype=torch.float32)
        edge_mask[new_idx] = True
        edge_relation[new_idx] = edges_relation[old_idx]

    # Build transitions: e->f valid iff terminal(e)==origin(f), terminal(e) is occupied
    # (i.e. not virtual), and origin(e) != terminal(f). Optionally apply ablations.
    # Group edges by their origin node.
    edges_by_origin: dict[int, list[int]] = {}
    for new_idx, old_idx in enumerate(indexed):
        edges_by_origin.setdefault(edges_origin[old_idx], []).append(new_idx)

    transitions: list[tuple[int, int, int]] = []  # (src_edge, dst_edge, type_pair)
    apply_backtracking = ablation_mode == "backtracking_allowed"
    for new_e_idx, old_e_idx in enumerate(indexed):
        terminal_e = edges_terminal[old_e_idx]
        origin_e = edges_origin[old_e_idx]
        # if terminal is virtual, no outgoing transition
        if terminal_e >= len(occupied_squares):
            continue
        rel_e = edges_relation[old_e_idx]
        # Look up edges starting at terminal_e
        for new_f_idx in edges_by_origin.get(terminal_e, ()):
            old_f_idx = indexed[new_f_idx]
            terminal_f = edges_terminal[old_f_idx]
            if not apply_backtracking and terminal_f == origin_e:
                continue
            rel_f = edges_relation[old_f_idx]
            type_pair = rel_e * NUM_RELATION_TYPES + rel_f
            transitions.append((new_e_idx, new_f_idx, type_pair))

    if ablation_mode == "randomized_transitions" and transitions and rng is not None:
        # Degree/type-preserving randomization: per source edge and outgoing type pair,
        # shuffle the destination among the valid set sharing the same source rel and dst rel.
        # Simpler approximation: keep src_edge, type_pair; permute dst_edge across the full
        # transition list. This preserves per-source out-degree only on average. We instead
        # do: bucket by (rel_e, rel_f), permute dst within bucket. That preserves per-bucket
        # type pair counts and degree marginals across edge ids approximately.
        bucket: dict[int, list[int]] = {}
        for tidx, (_, dst, tp) in enumerate(transitions):
            bucket.setdefault(tp, []).append(tidx)
        new_dsts = list(t[1] for t in transitions)
        for tp, idx_list in bucket.items():
            dsts = [transitions[i][1] for i in idx_list]
            perm = torch.randperm(len(dsts), generator=rng).tolist()
            for k, src_pos in enumerate(idx_list):
                new_dsts[src_pos] = dsts[perm[k]]
        transitions = [(t[0], new_dsts[i], t[2]) for i, t in enumerate(transitions)]

    # Truncate transitions deterministically
    transitions.sort(key=lambda triple: (triple[0], triple[1]))
    transition_overflow = max(0, len(transitions) - transition_max)
    transitions = transitions[:transition_max]

    transition_src = torch.zeros((transition_max,), dtype=torch.long)
    transition_dst = torch.zeros((transition_max,), dtype=torch.long)
    transition_type_pair = torch.zeros((transition_max,), dtype=torch.long)
    transition_mask = torch.zeros((transition_max,), dtype=torch.bool)
    for tidx, (src, dst, tp) in enumerate(transitions):
        transition_src[tidx] = src
        transition_dst[tidx] = dst
        transition_type_pair[tidx] = tp
        transition_mask[tidx] = True

    return _BoardEdges(
        edge_features=edge_features,
        edge_mask=edge_mask,
        edge_relation=edge_relation,
        transition_src=transition_src,
        transition_dst=transition_dst,
        transition_type_pair=transition_type_pair,
        transition_mask=transition_mask,
        edge_overflow=edge_overflow,
        transition_overflow=transition_overflow,
    )


class Simple18BoardParser(nn.Module):
    """Fail-closed simple_18 parser used by the deterministic edge builder."""

    def __init__(self, input_channels: int = 18, encoding: str = "simple_18", side_to_move_channel: int = 12) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.encoding = str(encoding)
        self.side_to_move_channel = int(side_to_move_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = require_board_tensor(x, self.spec)
        if self.encoding != "simple_18" or self.spec.input_channels != 18 or self.side_to_move_channel != 12:
            raise ValueError(
                "NonBacktrackingTacticalWalkNet supports simple_18 only; got "
                f"encoding={self.encoding!r}, channels={self.spec.input_channels}, "
                f"side_to_move_channel={self.side_to_move_channel}"
            )
        return x


class AttackProtectionEdgeBuilder(nn.Module):
    """Builds attack/protection directed edges from a simple_18 board batch (CPU)."""

    def __init__(
        self,
        edge_max: int = 192,
        use_king_zone_virtual_nodes: bool = True,
    ) -> None:
        super().__init__()
        self.edge_max = int(edge_max)
        self.use_king_zone_virtual_nodes = bool(use_king_zone_virtual_nodes)


class NonBacktrackingTransitionBuilder(nn.Module):
    """Builds non-backtracking edge-to-edge transitions from a board edge set."""

    def __init__(self, transition_max: int = 1024, ablation_mode: str = "none", seed: int = 42) -> None:
        super().__init__()
        self.transition_max = int(transition_max)
        self.ablation_mode = str(ablation_mode)
        self.seed = int(seed)


class TypedEdgeEncoder(nn.Module):
    def __init__(self, edge_dim: int = 64, dropout: float = 0.05) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(F_EDGE, int(edge_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(edge_dim), int(edge_dim)),
            nn.GELU(),
        )

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        return self.net(edge_features)


class NonBacktrackingEdgeBlock(nn.Module):
    """One typed-message non-backtracking propagation layer.

    For each transition (src_edge, dst_edge, type_pair) it computes a typed-linear
    transform of the source edge state and scatter-adds it into the destination edge
    state. Typed weights use a basis decomposition: a shared linear plus a sum of
    R_basis basis transforms gated by a learnable type-pair mixing matrix.
    """

    def __init__(self, edge_dim: int = 64, num_type_pairs: int = NUM_TYPE_PAIRS, r_basis: int = 4) -> None:
        super().__init__()
        self.edge_dim = int(edge_dim)
        self.num_type_pairs = int(num_type_pairs)
        self.r_basis = int(r_basis)
        self.self_linear = nn.Linear(self.edge_dim, self.edge_dim, bias=False)
        self.shared = nn.Linear(self.edge_dim, self.edge_dim, bias=False)
        self.basis = nn.Parameter(torch.randn(self.r_basis, self.edge_dim, self.edge_dim) * (1.0 / max(self.edge_dim, 1) ** 0.5))
        self.type_pair_mixer = nn.Embedding(self.num_type_pairs, self.r_basis)
        nn.init.zeros_(self.type_pair_mixer.weight)
        self.bias_per_type = nn.Embedding(NUM_RELATION_TYPES, self.edge_dim)
        nn.init.zeros_(self.bias_per_type.weight)
        self.norm = nn.LayerNorm(self.edge_dim)

    def forward(
        self,
        edge_state: torch.Tensor,
        transition_src: torch.Tensor,
        transition_dst: torch.Tensor,
        transition_type_pair: torch.Tensor,
        transition_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        edge_relation: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, edge_max, dim = edge_state.shape
        # Gather source edge states for each transition: (B, T_max, D)
        src_states = torch.gather(
            edge_state, 1, transition_src.unsqueeze(-1).expand(-1, -1, dim),
        )
        # Shared transform
        shared_msg = self.shared(src_states)
        # Basis transform: select per-transition mixing weights
        mixer = self.type_pair_mixer(transition_type_pair)  # (B, T, R_basis)
        # Apply each basis transform to src_states and combine
        # basis: (R_basis, D, D); src: (B, T, D)
        basis_msgs = torch.einsum("rij,btj->btri", self.basis, src_states)  # (B, T, R, D)
        basis_msg = (mixer.unsqueeze(-1) * basis_msgs).sum(dim=2)  # (B, T, D)
        msg = (shared_msg + basis_msg) * transition_mask.unsqueeze(-1).to(src_states.dtype)
        # Scatter-add into incoming_sum at destination edges
        incoming_sum = torch.zeros_like(edge_state)
        incoming_sum.scatter_add_(
            1, transition_dst.unsqueeze(-1).expand(-1, -1, dim), msg,
        )
        update = self.self_linear(edge_state) + incoming_sum
        # Add per-type bias
        bias = self.bias_per_type(edge_relation)  # (B, E_max, D)
        update = update + bias
        update = F.gelu(update) * edge_mask.unsqueeze(-1).to(edge_state.dtype)
        return self.norm(edge_state + update) * edge_mask.unsqueeze(-1).to(edge_state.dtype)


class EdgeMomentPooler(nn.Module):
    """Pool edge states with mean, max, log-sum-exp, per-relation mean, and energy."""

    def __init__(self, edge_dim: int = 64) -> None:
        super().__init__()
        self.edge_dim = int(edge_dim)

    @property
    def output_dim(self) -> int:
        # mean + max + lse + per-relation mean (NUM_RELATION_TYPES) + 1 energy scalar
        return self.edge_dim * (3 + NUM_RELATION_TYPES) + 1

    def forward(
        self,
        edge_state: torch.Tensor,
        edge_mask: torch.Tensor,
        edge_relation: torch.Tensor,
    ) -> torch.Tensor:
        mask = edge_mask.unsqueeze(-1).to(edge_state.dtype)
        masked = edge_state * mask
        denom = mask.sum(dim=1).clamp_min(1.0)
        mean = masked.sum(dim=1) / denom
        # max with -inf for invalid entries
        very_neg = torch.full_like(edge_state, float("-1e9"))
        max_input = torch.where(edge_mask.unsqueeze(-1), edge_state, very_neg)
        max_pooled = max_input.amax(dim=1)
        # If no edges, max would be -1e9; clamp to zero
        any_edge = edge_mask.any(dim=1, keepdim=True).to(edge_state.dtype)
        max_pooled = max_pooled * any_edge
        # log-sum-exp; subtract max for stability
        lse_input = torch.where(edge_mask.unsqueeze(-1), edge_state, very_neg)
        lse_pooled = torch.logsumexp(lse_input, dim=1)
        lse_pooled = torch.where(any_edge.expand_as(lse_pooled) > 0, lse_pooled, torch.zeros_like(lse_pooled))
        # per-relation mean
        rel_means: list[torch.Tensor] = []
        for rel in range(NUM_RELATION_TYPES):
            rel_mask = ((edge_relation == rel) & edge_mask).unsqueeze(-1).to(edge_state.dtype)
            rel_denom = rel_mask.sum(dim=1).clamp_min(1.0)
            rel_means.append((edge_state * rel_mask).sum(dim=1) / rel_denom)
        # energy scalar
        energy = (masked.square().sum(dim=(1, 2)) / denom.squeeze(-1).clamp_min(1.0)).unsqueeze(-1)
        return torch.cat([mean, max_pooled, lse_pooled, *rel_means, energy], dim=1)


class SmallBoardAdapter(nn.Module):
    """Compact CNN trunk used only as a small auxiliary, not the central claim."""

    def __init__(self, input_channels: int = 18, channels: int = 32, use_batchnorm: bool = True) -> None:
        super().__init__()
        norm_a = nn.BatchNorm2d(channels) if use_batchnorm else nn.GroupNorm(min(channels, 8), channels)
        norm_b = nn.BatchNorm2d(channels) if use_batchnorm else nn.GroupNorm(min(channels, 8), channels)
        self.conv = nn.Sequential(
            nn.Conv2d(int(input_channels), channels, kernel_size=3, padding=1, bias=not use_batchnorm),
            norm_a,
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm),
            norm_b,
            nn.GELU(),
        )
        self.output_dim = 2 * channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        pooled = torch.cat([feat.mean(dim=(-1, -2)), feat.amax(dim=(-1, -2))], dim=1)
        return pooled


class NonBacktrackingTacticalWalkNet(nn.Module):
    """Bespoke non-backtracking tactical walk classifier for puzzle-binary."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        encoding: str = "simple_18",
        side_to_move_channel: int = 12,
        edge_dim: int = 64,
        edge_layers: int = 4,
        edge_max: int = 192,
        transition_max: int = 1024,
        r_basis: int = 4,
        use_king_zone_virtual_nodes: bool = True,
        board_adapter_channels: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.05,
        use_batchnorm: bool = True,
        ablation_mode: str = "none",
        seed: int = 42,
    ) -> None:
        super().__init__()
        if int(num_classes) not in {1, 2}:
            raise ValueError("NonBacktrackingTacticalWalkNet supports num_classes in {1, 2}")
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.num_classes = int(num_classes)
        self.parser = Simple18BoardParser(
            input_channels=input_channels,
            encoding=encoding,
            side_to_move_channel=side_to_move_channel,
        )
        self.edge_builder = AttackProtectionEdgeBuilder(
            edge_max=edge_max,
            use_king_zone_virtual_nodes=use_king_zone_virtual_nodes,
        )
        self.transition_builder = NonBacktrackingTransitionBuilder(
            transition_max=transition_max,
            ablation_mode=ablation_mode,
            seed=seed,
        )
        self.edge_encoder = TypedEdgeEncoder(edge_dim=edge_dim, dropout=dropout)
        self.edge_blocks = nn.ModuleList(
            [
                NonBacktrackingEdgeBlock(edge_dim=edge_dim, r_basis=r_basis)
                for _ in range(int(edge_layers))
            ]
        )
        self.pooler = EdgeMomentPooler(edge_dim=edge_dim)
        self.board_adapter = SmallBoardAdapter(
            input_channels=input_channels,
            channels=int(board_adapter_channels),
            use_batchnorm=use_batchnorm,
        )
        self.edge_max = int(edge_max)
        self.transition_max = int(transition_max)
        self.use_king_zone_virtual_nodes = bool(use_king_zone_virtual_nodes)
        self.ablation_mode = str(ablation_mode)
        self.seed = int(seed)
        self._rng_state = torch.Generator()
        self._rng_state.manual_seed(int(seed))
        self.edge_layers = int(edge_layers)
        self.edge_dim = int(edge_dim)
        joint_dim = self.board_adapter.output_dim + self.pooler.output_dim * (1 + int(edge_layers))
        self.classifier = nn.Sequential(
            nn.Linear(joint_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 2),
        )

    def _build_batch_edges(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        per_board: list[_BoardEdges] = []
        for b in range(x.shape[0]):
            be = build_board_edges(
                x[b],
                edge_max=self.edge_max,
                transition_max=self.transition_max,
                use_king_zone_virtual_nodes=self.use_king_zone_virtual_nodes,
                ablation_mode=self.ablation_mode,
                rng=self._rng_state if self.ablation_mode == "randomized_transitions" else None,
            )
            per_board.append(be)
        edge_features = torch.stack([e.edge_features for e in per_board], dim=0).to(device=x.device, dtype=x.dtype)
        edge_mask = torch.stack([e.edge_mask for e in per_board], dim=0).to(device=x.device)
        edge_relation = torch.stack([e.edge_relation for e in per_board], dim=0).to(device=x.device)
        transition_src = torch.stack([e.transition_src for e in per_board], dim=0).to(device=x.device)
        transition_dst = torch.stack([e.transition_dst for e in per_board], dim=0).to(device=x.device)
        transition_type_pair = torch.stack([e.transition_type_pair for e in per_board], dim=0).to(device=x.device)
        transition_mask = torch.stack([e.transition_mask for e in per_board], dim=0).to(device=x.device)
        edge_overflow = torch.tensor([e.edge_overflow for e in per_board], dtype=torch.float32, device=x.device)
        transition_overflow = torch.tensor([e.transition_overflow for e in per_board], dtype=torch.float32, device=x.device)
        return {
            "edge_features": edge_features,
            "edge_mask": edge_mask,
            "edge_relation": edge_relation,
            "transition_src": transition_src,
            "transition_dst": transition_dst,
            "transition_type_pair": transition_type_pair,
            "transition_mask": transition_mask,
            "edge_overflow": edge_overflow,
            "transition_overflow": transition_overflow,
        }

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> dict[str, torch.Tensor]:
        x = self.parser(x)
        graph = self._build_batch_edges(x)
        edge_state = self.edge_encoder(graph["edge_features"])
        edge_state = edge_state * graph["edge_mask"].unsqueeze(-1).to(edge_state.dtype)
        pooled_states: list[torch.Tensor] = [
            self.pooler(edge_state, graph["edge_mask"], graph["edge_relation"]),
        ]
        for block in self.edge_blocks:
            edge_state = block(
                edge_state,
                graph["transition_src"],
                graph["transition_dst"],
                graph["transition_type_pair"],
                graph["transition_mask"],
                graph["edge_mask"],
                graph["edge_relation"],
            )
            pooled_states.append(self.pooler(edge_state, graph["edge_mask"], graph["edge_relation"]))
        edge_latent = torch.cat(pooled_states, dim=1)
        board_latent = self.board_adapter(x)
        joint = torch.cat([board_latent, edge_latent], dim=1)
        two_class = self.classifier(joint)
        if self.num_classes == 1:
            logits = two_class[:, 1] - two_class[:, 0]
        else:
            logits = two_class

        edge_count = graph["edge_mask"].to(edge_state.dtype).sum(dim=1)
        transition_count = graph["transition_mask"].to(edge_state.dtype).sum(dim=1)
        per_relation_counts = []
        for rel in range(NUM_RELATION_TYPES):
            rel_count = ((graph["edge_relation"] == rel) & graph["edge_mask"]).to(edge_state.dtype).sum(dim=1)
            per_relation_counts.append(rel_count)
        edge_state_energy = edge_state.square().mean(dim=(1, 2))
        rel_count_stack = torch.stack(per_relation_counts, dim=1)

        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "two_class_logits": two_class,
            "edge_count": edge_count,
            "transition_count": transition_count,
            "edge_overflow_count": graph["edge_overflow"].to(edge_state.dtype),
            "transition_overflow_count": graph["transition_overflow"].to(edge_state.dtype),
            "non_backtracking_walk_energy": edge_state_energy,
            "enemy_attack_edge_count": rel_count_stack[:, 0],
            "friendly_protect_edge_count": rel_count_stack[:, 1],
            "enemy_king_zone_edge_count": rel_count_stack[:, 2],
            "own_king_zone_edge_count": rel_count_stack[:, 3],
            "edge_state_mean_norm": edge_state.norm(dim=2).mean(dim=1),
            "edge_state_max_norm": edge_state.norm(dim=2).amax(dim=1),
            "mechanism_energy": edge_state_energy,
            "proposal_profile_strength": edge_state.norm(dim=2).amax(dim=1),
            "proposal_keyword_count": logits.new_full((x.shape[0],), float(NUM_RELATION_TYPES)),
            "defense_gap": rel_count_stack[:, 0] - rel_count_stack[:, 1],
            "king_ring_pressure": rel_count_stack[:, 2] + rel_count_stack[:, 3],
        }
        if return_aux:
            output.update(
                {
                    "edge_features": graph["edge_features"],
                    "edge_mask": graph["edge_mask"],
                    "edge_relation": graph["edge_relation"],
                    "transition_src": graph["transition_src"],
                    "transition_dst": graph["transition_dst"],
                    "transition_type_pair": graph["transition_type_pair"],
                    "transition_mask": graph["transition_mask"],
                    "edge_state": edge_state,
                    "edge_latent": edge_latent,
                    "board_latent": board_latent,
                }
            )
        return output


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("model", config))


def _data_config(config: dict[str, Any]) -> dict[str, Any]:
    data = config.get("data", {})
    return data if isinstance(data, dict) else {}


def build_non_backtracking_tactical_walk_network_from_config(
    config: dict[str, Any],
) -> NonBacktrackingTacticalWalkNet:
    cfg = _model_config(config)
    data_cfg = _data_config(config)
    return NonBacktrackingTacticalWalkNet(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        encoding=str(cfg.get("encoding", data_cfg.get("encoding", "simple_18"))),
        side_to_move_channel=int(cfg.get("side_to_move_channel", 12)),
        edge_dim=int(cfg.get("edge_dim", cfg.get("hidden_dim", 64))),
        edge_layers=int(cfg.get("edge_layers", cfg.get("depth", 4))),
        edge_max=int(cfg.get("edge_max", 192)),
        transition_max=int(cfg.get("transition_max", 1024)),
        r_basis=int(cfg.get("r_basis", 4)),
        use_king_zone_virtual_nodes=bool(cfg.get("use_king_zone_virtual_nodes", True)),
        board_adapter_channels=int(cfg.get("board_adapter_channels", 32)),
        hidden_dim=int(cfg.get("classifier_hidden_dim", cfg.get("hidden_dim", 128))),
        dropout=float(cfg.get("dropout", 0.05)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        ablation_mode=str(cfg.get("ablation_mode", cfg.get("ablation", "none"))),
        seed=int(cfg.get("seed", config.get("seed", 42))),
    )

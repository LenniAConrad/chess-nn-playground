"""Legal-Move-Graph Convolution (p009) — typed message passing primitive head.

Promoted from
``ideas/research/primitives/external_05_legal_move_graph_delta_accumulator.md``
(top-ranked proposal: LMGConv). LMGConv is a graph operator whose adjacency
is the current position's legal-move bitboard, with per-piece-type weight
matrices that route information along each piece's moves rather than the
8x8 grid.

For square tokens ``x`` and per-piece-type adjacency
``A_r ∈ {0,1}^{64×64}`` (1 where a piece of type ``r`` on square ``i`` can
move to ``j`` according to the rules and occlusion):

    y_i = sum_r mean_{j : A_r[i, j] = 1} (W_r x_j) + b_i

We normalise by the per-type degree so heads with very few edges (e.g. a
single rook on an open file) do not dominate the aggregate; this matches
the GraphSAGE-style normalisation flagged in the LMGConv failure-mode
catalogue. The adjacency tensor is treated as stop-gradient — the spec
specifies "no gradient w.r.t. A".

The "delta_accumulator" half of the file's slug refers to ΔAcc, the
stateful NNUE-style accumulator (#2 in the source file). Per the
implementation rule "implement the strongest or first-ranked proposal", we
implement LMGConv as the trainable head and document ΔAcc as a deferred
internal proposal in ``ablations.md``.

Architecture (additive, gated):

    base_logit       = i193_trunk(board)["logits"]
    tokens           = SquareTokenEmbedder(board)        # (B, 64, d)
    legal_per_type   = compute_typed_legal_graph(board)  # (B, R, 64, 64) stop-grad
    msgs             = einsum("brij,bjd,rdc->brio", legal_per_type, tokens, W) / degree
    routed           = LayerNorm( sum_r msgs )           # (B, 64, c)
    pooled           = mean_squares(routed)
    delta            = MLP(pooled)
    gate             = sigmoid(MLP(trunk_pool))
    final_logit      = base_logit + gate * delta
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.primitives.rule_graph_features import (
    BISHOP,
    KING,
    KNIGHT,
    NUM_PIECE_TYPES,
    PAWN,
    QUEEN,
    ROOK,
    SQUARES,
    SquareTokenEmbedder,
    compute_attack_relations,
    piece_planes_flat,
    rule_geometry,
    select_by_side_to_move,
    side_to_move_from_board,
)
from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    DualStreamFeatureBuilder,
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


PIECE_TYPE_NAMES: tuple[str, ...] = ("P", "N", "B", "R", "Q", "K")


ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    "random_typed_edges",   # per-type adjacency replaced by random mask of same density
    "shared_weight",         # collapse W_r to a single shared linear (kills typed channel)
    "no_normalization",     # remove GraphSAGE-style degree normalisation
    "zero_delta",
    "disable_gate",
    "trunk_only",
)


def _compute_typed_legal_edges(
    board: torch.Tensor,
    geometry,
) -> torch.Tensor:
    """Compute the per-piece-type legal-move graph ``(B, 6, 64, 64)``.

    For each piece type ``r`` (P, N, B, R, Q, K) the edge ``(i, j)`` is 1
    if the side-to-move has an own piece of type ``r`` on square ``i`` and
    that piece can move (attack-style; pseudo-legal) to square ``j`` with
    occlusion already factored in for sliding pieces. Targets occupied by
    own pieces are excluded.
    """
    device = board.device
    dtype = board.dtype
    batch = board.shape[0]
    piece_planes = piece_planes_flat(board)  # (B, 12, 64)
    stm = side_to_move_from_board(board)
    occ = piece_planes.sum(dim=1).clamp(0.0, 1.0)
    own_pieces_white = piece_planes[:, :6].sum(dim=1).clamp(0.0, 1.0)
    own_pieces_black = piece_planes[:, 6:12].sum(dim=1).clamp(0.0, 1.0)
    own_pieces = select_by_side_to_move(own_pieces_white, own_pieces_black, stm)
    target_open = (1.0 - own_pieces).unsqueeze(1)  # (B, 1, 64)

    between = geometry.between.to(device=device, dtype=dtype)
    geom = geometry.geom_attacks.to(device=device, dtype=dtype)
    blocked = torch.einsum("stk,bk->bst", between, occ)
    clear_slide = (blocked <= 0.5).to(dtype=dtype)
    clear_dense = torch.ones_like(clear_slide)

    edges = piece_planes.new_zeros(batch, NUM_PIECE_TYPES, SQUARES, SQUARES)
    for piece in range(NUM_PIECE_TYPES):
        white_source = piece_planes[:, piece]
        black_source = piece_planes[:, 6 + piece]
        own_source = select_by_side_to_move(white_source, black_source, stm)
        # Pick the geom_attacks pattern matching the side-to-move colour.
        own_geom = select_by_side_to_move(
            geom[piece, 0].unsqueeze(0).expand(batch, SQUARES, SQUARES),
            geom[piece, 1].unsqueeze(0).expand(batch, SQUARES, SQUARES),
            stm,
        )
        line_clear = clear_slide if piece in {BISHOP, ROOK, QUEEN} else clear_dense
        relation = own_source[:, :, None] * own_geom * line_clear * target_open
        edges[:, piece] = relation
    return edges.clamp(0.0, 1.0)


class LegalMoveGraphDelta(nn.Module):
    """p009 — Legal-Move-Graph Convolution head with per-piece-type messaging."""

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
        token_embed_dim: int = 32,
        token_hidden_dim: int = 0,
        message_dim: int = 32,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "LegalMoveGraphDelta supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError(
                "LegalMoveGraphDelta requires the simple_18 board tensor"
            )
        if str(ablation) not in ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(ALLOWED_ABLATIONS)}"
            )
        self.num_classes = 1
        self.ablation = str(ablation)
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self._geometry = rule_geometry()

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

        self.token_embed = SquareTokenEmbedder(
            input_channels=int(input_channels),
            embed_dim=int(token_embed_dim),
            hidden_dim=int(token_hidden_dim),
            dropout=float(head_dropout),
        )
        if self.ablation == "shared_weight":
            self.type_linear = nn.Linear(int(token_embed_dim), int(message_dim))
            self.type_linears = None
        else:
            self.type_linear = None
            self.type_linears = nn.ModuleList(
                [nn.Linear(int(token_embed_dim), int(message_dim)) for _ in range(NUM_PIECE_TYPES)]
            )
        self.message_norm = nn.LayerNorm(int(message_dim))

        trunk_pool_dim = (
            2 * self.trunk.exchange_encoder.output_dim
            + DualStreamFeatureBuilder.SUMMARY_DIM
        )
        self._trunk_pool_dim = trunk_pool_dim
        head_dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(int(message_dim)),
            nn.Linear(int(message_dim), int(head_hidden_dim)),
            nn.GELU(),
            head_dropout_module,
            nn.Linear(int(head_hidden_dim), 1),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(trunk_pool_dim),
            nn.Linear(trunk_pool_dim, int(head_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(head_hidden_dim), 1),
        )
        with torch.no_grad():
            final_layer = self.gate_head[-1]
            if isinstance(final_layer, nn.Linear):
                final_layer.bias.fill_(float(gate_init))

    def _trunk_joint(self, board: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        trunk_out = self.trunk(board)
        feats = self.trunk.feature_builder(board)
        if self.trunk.ablation == "shared_stream_only":
            ex_input = board
            kg_input = board
        else:
            ex_input = torch.cat([board, feats.exchange], dim=1)
            kg_input = torch.cat([board, feats.king], dim=1)
        _, ex_pool = self.trunk.exchange_encoder(ex_input)
        if self.trunk.ablation == "shared_stream_only":
            kg_pool = ex_pool
        else:
            _, kg_pool = self.trunk.king_encoder(kg_input)
        joint = torch.cat([ex_pool, kg_pool, feats.summary], dim=1)
        return trunk_out, joint

    @torch.no_grad()
    def _build_edges(self, board: torch.Tensor) -> torch.Tensor:
        edges = _compute_typed_legal_edges(board, self._geometry)
        if self.ablation == "random_typed_edges":
            density = edges.sum(dim=(2, 3), keepdim=True) / (SQUARES * SQUARES)
            rand = torch.rand_like(edges)
            edges = (rand < density).to(dtype=edges.dtype)
        return edges

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]
        trunk_out, trunk_joint = self._trunk_joint(board)
        base_logit = trunk_out["logits"]

        tokens = self.token_embed(board)  # (B, 64, d)
        edges_per_type = self._build_edges(board)  # (B, R, 64, 64)

        # Per-piece-type messages: each type uses its own linear projection.
        if self.type_linears is None:
            assert self.type_linear is not None
            projected = self.type_linear(tokens).unsqueeze(1).expand(
                batch, NUM_PIECE_TYPES, SQUARES, -1
            )
        else:
            projected = torch.stack(
                [linear(tokens) for linear in self.type_linears], dim=1
            )  # (B, R, 64, message_dim)
        # message at source i = mean_j edges_per_type[r, i, j] * projected[j]
        # Use bmm per type and stack.
        # For efficiency: collapse R into batch dim
        b, r, n, _ = projected.shape
        edges_flat = edges_per_type.view(batch * NUM_PIECE_TYPES, SQUARES, SQUARES)
        projected_flat = projected.reshape(batch * NUM_PIECE_TYPES, SQUARES, -1)
        msg_flat = torch.bmm(edges_flat, projected_flat)  # (B*R, 64, message_dim)
        msg_per_type = msg_flat.view(batch, NUM_PIECE_TYPES, SQUARES, -1)
        if self.ablation != "no_normalization":
            degree = edges_per_type.sum(dim=-1, keepdim=True).clamp_min(1.0)  # (B, R, 64, 1)
            msg_per_type = msg_per_type / degree
        msgs = msg_per_type.sum(dim=1)  # (B, 64, message_dim)
        msgs = self.message_norm(msgs)

        pooled = msgs.mean(dim=1)  # (B, message_dim)
        delta_raw = self.delta_head(pooled).view(-1)
        gate_logit = self.gate_head(trunk_joint.detach()).view(-1)
        gate = torch.sigmoid(gate_logit)
        if self.ablation == "disable_gate":
            gate = torch.ones_like(gate)
        if self.ablation in {"zero_delta", "trunk_only"}:
            primitive_delta = torch.zeros_like(base_logit)
        else:
            primitive_delta = gate * delta_raw
        if self.ablation == "trunk_only":
            primitive_delta = torch.zeros_like(primitive_delta)
        logits = base_logit + primitive_delta

        out: dict[str, torch.Tensor] = dict(trunk_out)
        out["logits"] = logits
        out["base_logit"] = base_logit
        out["primitive_delta"] = primitive_delta
        out["primitive_delta_raw"] = delta_raw
        out["primitive_gate"] = gate
        out["primitive_gate_logit"] = gate_logit
        out["lmgconv_edge_count"] = edges_per_type.sum(dim=(1, 2, 3))
        # Per-type edge contribution (mean magnitude per piece-type channel).
        type_norm = msg_per_type.pow(2).mean(dim=(2, 3)).sqrt()  # (B, R)
        for piece_idx, piece_name in enumerate(PIECE_TYPE_NAMES):
            out[f"lmgconv_msg_norm_{piece_name}"] = type_norm[:, piece_idx]
        return out


def build_legal_move_graph_delta_from_config(
    config: dict[str, Any],
) -> LegalMoveGraphDelta:
    cfg = dict(config)
    return LegalMoveGraphDelta(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim", None)),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        token_embed_dim=int(cfg.get("token_embed_dim", 32)),
        token_hidden_dim=int(cfg.get("token_hidden_dim", 0)),
        message_dim=int(cfg.get("message_dim", 32)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )


_ = (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)  # plane order anchor

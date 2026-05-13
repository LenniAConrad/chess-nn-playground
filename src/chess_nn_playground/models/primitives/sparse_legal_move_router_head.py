"""Sparse Legal-Move Router Head (p027).

Promotes the **Sparse Legal-Move Router (SLMR)** primitive (first-ranked
proposal of
``ideas/research/primitives/external_23_sparse_legal_move_router_kinematic_state_space.md``).
SLMR is a sparse graph-interaction operator whose connectivity is the
rule-derived legal-move adjacency of the current board state:

    Y_i = Agg({ phi(X_i, X_j, W) : (i, j) in LegalMoves })

We materialise the adjacency rule-exactly from the ``simple_18`` board using
the i193 geometry tables (``geom_attacks`` for jump-piece moves and
``between`` + occupancy for sliding-piece moves), so no ``python-chess``
call is needed in the forward pass. The aggregator is a soft attention over
just the legal targets — the dense ``(64, 64)`` mask is computed inside the
helper but the routing is masked everywhere outside the legal-move support,
matching the spec's "hard topological constraint" requirement.

Deferred external_23 proposals (research-only): ``KDS`` kinematic deformable
sampling (piece-type-conditioned convolution offsets — handled cleaner as a
trunk feature stack), ``ISSC`` incremental state-space cell (covered by
``p028`` / ``p030`` for the state-space family), ``CIF`` color-invariant
folding (an activation primitive on top of the trunk), and ``BPIP`` bilinear
piece-interaction pooling (a separate piece-pair primitive overlapping with
the DHPE/PPI family).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.primitives.primitive_heads import (
    SHARED_ABLATIONS,
    BoardTensorSpec,
    build_trunk_from_kwargs,
    extract_trunk_diagnostics,
    fuse_with_base_logit,
    require_board_tensor,
    small_mlp,
    standard_diagnostics_dict,
)
from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    BISHOP,
    KING,
    KNIGHT,
    PAWN,
    QUEEN,
    ROOK,
    SQUARES,
    _build_geometry,
    _piece_channel,
)


PIECE_PLANE_COUNT = 12
STM_CHANNEL = 12
NUM_PIECE_TYPES = 6


ALLOWED_ABLATIONS: tuple[str, ...] = SHARED_ABLATIONS + (
    "full_64x64_mask",       # ignore legal-move adjacency; route over all squares
    "self_loop_only",        # only allow each square to attend to itself
    "shuffle_adjacency",     # random permutation of the legal-move mask
    "zero_router_features",  # disable the router output entirely
)


def compute_legal_move_adjacency(board: torch.Tensor) -> torch.Tensor:
    """Compute the rule-exact legal-move adjacency from the simple_18 board.

    The adjacency we use is "own-piece on square i can target square j"
    under the standard piece movement rules. Sliding pieces honour blockers
    via the ``between`` mask + occupancy. Pawns use forward-only adjacency
    in addition to standard captures (this is conservative; we treat
    forward and diagonal pawn moves uniformly as the routing topology, not
    as a chess-policy estimator).

    Returns:
        ``(B, 64, 64)`` float tensor with 1.0 where a legal own-piece move
        connects square i to square j.
    """
    if board.shape[1] < PIECE_PLANE_COUNT + 1:
        raise ValueError("simple_18 board must contain at least 13 channels")
    device = board.device
    dtype = board.dtype
    batch = board.shape[0]
    piece_planes = board[:, :PIECE_PLANE_COUNT].flatten(2).clamp(0.0, 1.0)
    # ``(B,)`` indicator: 1.0 white-to-move, 0.0 black-to-move.
    stm = board[:, STM_CHANNEL].mean(dim=(1, 2)).clamp(0.0, 1.0)

    geom_attacks, between, _king_zone = _build_geometry()
    geom_attacks = geom_attacks.to(device=device, dtype=dtype)
    between = between.to(device=device, dtype=dtype)
    occ = piece_planes.sum(dim=1).clamp(0.0, 1.0)  # (B, 64)
    blocked_count = torch.einsum("stk,bk->bst", between, occ)
    clear = (blocked_count <= 0.5).to(dtype=dtype)  # (B, 64, 64)
    ones_clear = torch.ones_like(clear)

    adjacency = piece_planes.new_zeros(batch, SQUARES, SQUARES)
    # Stack indices: (piece_type, channels_for_white_or_black)
    piece_types = (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)
    selector = stm.view(-1, 1, 1)  # (B, 1, 1)
    for piece in piece_types:
        white_source = piece_planes[:, _piece_channel(0, piece)]  # (B, 64)
        black_source = piece_planes[:, _piece_channel(1, piece)]  # (B, 64)
        own_source = selector.squeeze(-1) * white_source + (1.0 - selector.squeeze(-1)) * black_source
        line_clear = clear if piece in {BISHOP, ROOK, QUEEN} else ones_clear
        # geom_attacks: (6, 2, 64, 64); pick the right color per sample.
        white_attacks = geom_attacks[piece, 0]  # (64, 64)
        black_attacks = geom_attacks[piece, 1]
        own_attacks = selector * white_attacks.unsqueeze(0) + (1.0 - selector) * black_attacks.unsqueeze(0)
        # (B, 64, 64)
        relation = own_source.unsqueeze(-1) * own_attacks * line_clear
        adjacency = adjacency + relation
    return adjacency.clamp(0.0, 1.0)


def _piece_type_per_square(board: torch.Tensor) -> torch.Tensor:
    """``(B, 64)`` long tensor: piece-type id (0..11) at each square, or 12 for empty.

    The label space is the same as ``simple_18`` piece plane indices
    concatenated with an "empty" class — useful for piece-type-aware
    embedding lookups inside the router head.
    """
    piece_planes = board[:, :PIECE_PLANE_COUNT].flatten(2).clamp(0.0, 1.0)
    # Per-square one-hot scores; argmax breaks ties to the lowest plane id,
    # but the simple_18 encoding has at most one piece per square.
    has_piece = piece_planes.sum(dim=1).clamp(0.0, 1.0)
    plane_argmax = piece_planes.argmax(dim=1)
    empty_label = piece_planes.new_full(plane_argmax.shape, PIECE_PLANE_COUNT)
    out = torch.where(has_piece > 0.5, plane_argmax.to(empty_label.dtype), empty_label)
    return out.long()


class SparseLegalMoveRouterHead(nn.Module):
    """p027 — Sparse Legal-Move Router primitive head over the i193 trunk.

    Forward pass:

    1. Embed each square with a piece-type-aware embedding (13 classes: 12
       piece types + empty) plus a learned per-square positional embedding.
    2. Compute the rule-exact legal-move adjacency from the simple_18 board.
    3. Apply optional ablation-mode mask transforms (full mask, self-loop,
       shuffled adjacency).
    4. Run one round of masked attention scaled by the adjacency mask and
       normalise across the (variable-length) legal targets per source.
    5. Mean-pool the per-square routed features, fuse with the trunk
       diagnostics, and produce the additive-gated primitive delta.
    """

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
        # SLMR hyper-parameters.
        square_embed_dim: int = 32,
        attn_dim: int = 32,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -1.5,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "SparseLegalMoveRouterHead supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError(
                "SparseLegalMoveRouterHead requires the simple_18 board tensor"
            )
        if str(ablation) not in ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(ALLOWED_ABLATIONS)}"
            )

        self.num_classes = 1
        self.ablation = str(ablation)
        self.square_embed_dim = int(square_embed_dim)
        self.attn_dim = int(attn_dim)
        self.spec = BoardTensorSpec(input_channels=int(input_channels))

        self.trunk = build_trunk_from_kwargs(
            input_channels=int(input_channels),
            trunk_channels=int(trunk_channels),
            trunk_hidden_dim=int(trunk_hidden_dim),
            trunk_depth=int(trunk_depth),
            trunk_dropout=float(trunk_dropout),
            trunk_use_batchnorm=bool(trunk_use_batchnorm),
            trunk_gate_dim=trunk_gate_dim,
            trunk_ablation=str(trunk_ablation),
        )

        # 13 = 12 piece planes + empty class.
        self.piece_type_embedding = nn.Embedding(PIECE_PLANE_COUNT + 1, self.square_embed_dim)
        self.square_position_embedding = nn.Embedding(SQUARES, self.square_embed_dim)
        self.q_proj = nn.Linear(self.square_embed_dim, self.attn_dim)
        self.k_proj = nn.Linear(self.square_embed_dim, self.attn_dim)
        self.v_proj = nn.Linear(self.square_embed_dim, self.attn_dim)
        self.routed_norm = nn.LayerNorm(self.attn_dim)
        self.pooled_norm = nn.LayerNorm(self.attn_dim)

        fusion_in = self.attn_dim + 4
        self._fusion_dim = fusion_in
        self.delta_mlp = small_mlp(
            fusion_in,
            int(head_hidden_dim),
            1,
            dropout=float(head_dropout),
        )
        self.gate_mlp = small_mlp(
            fusion_in,
            int(head_hidden_dim),
            1,
            dropout=float(head_dropout),
            final_bias_init=float(gate_init),
        )

    def _build_adjacency(self, board: torch.Tensor) -> torch.Tensor:
        if self.ablation == "full_64x64_mask":
            return board.new_ones(board.shape[0], SQUARES, SQUARES)
        if self.ablation == "self_loop_only":
            mask = board.new_zeros(board.shape[0], SQUARES, SQUARES)
            idx = torch.arange(SQUARES, device=board.device)
            mask[:, idx, idx] = 1.0
            return mask
        adjacency = compute_legal_move_adjacency(board)
        if self.ablation == "shuffle_adjacency":
            perm = torch.randperm(SQUARES, device=adjacency.device)
            adjacency = adjacency[:, perm][:, :, perm]
        return adjacency

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        trunk_output = self.trunk(board)
        base_logit = trunk_output["logits"]

        piece_types = _piece_type_per_square(board)  # (B, 64) long
        square_idx = torch.arange(SQUARES, device=board.device, dtype=torch.long).expand(
            board.shape[0], SQUARES
        )
        token_features = (
            self.piece_type_embedding(piece_types)
            + self.square_position_embedding(square_idx)
        )  # (B, 64, square_embed_dim)

        adjacency = self._build_adjacency(board)
        adjacency_sum = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)

        q = self.q_proj(token_features)  # (B, 64, attn_dim)
        k = self.k_proj(token_features)
        v = self.v_proj(token_features)

        scale = self.attn_dim ** 0.5
        attn_logits = torch.einsum("bid,bjd->bij", q, k) / scale  # (B, 64, 64)
        # Mask non-edges to -inf so softmax never spends mass on illegal targets.
        masked_logits = attn_logits.masked_fill(adjacency < 0.5, float("-inf"))
        # Some source squares may have zero legal targets (empty squares, or
        # pieces with no moves) — fall back to a self-loop so softmax doesn't
        # NaN. The corresponding row of the adjacency stays zero, so we mask
        # the *output* contribution downstream.
        source_has_target = (adjacency.sum(dim=-1) > 0.5).float()
        no_target_rows = source_has_target < 0.5
        if no_target_rows.any():
            # Build a self-loop fallback only on rows with no legal targets.
            eye = torch.eye(SQUARES, device=board.device, dtype=board.dtype).unsqueeze(0)
            fallback_logits = attn_logits.masked_fill(eye < 0.5, float("-inf"))
            masked_logits = torch.where(
                no_target_rows.unsqueeze(-1).expand_as(masked_logits),
                fallback_logits,
                masked_logits,
            )
        attn_weights = torch.softmax(masked_logits, dim=-1)
        routed = torch.einsum("bij,bjd->bid", attn_weights, v)  # (B, 64, attn_dim)
        routed = routed * source_has_target.unsqueeze(-1)
        routed_normed = self.routed_norm(routed)

        legal_move_count = adjacency.sum(dim=(1, 2))  # (B,)
        pooled = routed_normed.mean(dim=1)  # (B, attn_dim)
        pooled = self.pooled_norm(pooled)
        if self.ablation == "zero_router_features":
            pooled = torch.zeros_like(pooled)

        diagnostics = extract_trunk_diagnostics(trunk_output)
        fusion_in = torch.cat([pooled, diagnostics], dim=1)

        delta_raw = self.delta_mlp(fusion_in).view(-1)
        gate_logit = self.gate_mlp(fusion_in).view(-1)
        gate = torch.sigmoid(gate_logit)
        logits, primitive_delta, effective_gate = fuse_with_base_logit(
            base_logit,
            gate,
            delta_raw,
            zero_delta=self.ablation in {"zero_delta", "trunk_only", "zero_router_features"},
            force_gate_one=self.ablation == "disable_gate",
        )

        # Diagnostics: legal-move sparsity and attention entropy.
        eps = 1.0e-8
        attn_entropy = -(attn_weights.clamp_min(eps) * attn_weights.clamp_min(eps).log()).sum(dim=-1)
        # Normalise by the per-source entropy ceiling (log of available targets).
        ceiling = adjacency_sum.squeeze(-1).clamp_min(1.0).log().clamp_min(eps)
        norm_entropy = (attn_entropy / ceiling) * source_has_target  # (B, 64)
        attn_entropy_mean = norm_entropy.sum(dim=-1) / source_has_target.sum(dim=-1).clamp_min(1.0)

        extra: dict[str, torch.Tensor] = {
            "slmr_legal_move_edges": legal_move_count,
            "slmr_active_sources": source_has_target.sum(dim=-1),
            "slmr_attention_entropy": attn_entropy_mean,
            "slmr_routed_feature_norm": pooled.pow(2).mean(dim=-1).sqrt(),
        }
        return standard_diagnostics_dict(
            trunk_output=trunk_output,
            logits=logits,
            base_logit=base_logit,
            primitive_delta=primitive_delta,
            delta_raw=delta_raw,
            gate=effective_gate,
            gate_logit=gate_logit,
            extra=extra,
        )


def build_sparse_legal_move_router_head_from_config(
    config: dict[str, Any],
) -> SparseLegalMoveRouterHead:
    cfg = dict(config)
    return SparseLegalMoveRouterHead(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim", None)),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        square_embed_dim=int(cfg.get("square_embed_dim", 32)),
        attn_dim=int(cfg.get("attn_dim", 32)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        gate_init=float(cfg.get("gate_init", -1.5)),
        ablation=str(cfg.get("ablation", "none")),
    )

"""Move-Graph Router (p006) — content-routed gather-scatter primitive head.

Promoted from ``ideas/research/primitives/external_02_move_graph_router_delta_accumulator.md``
(top-ranked proposal: MoveGraphRouter / MGR). The MGR primitive is a
gather-scatter operator whose sparse adjacency is determined inside the op
by a deterministic discrete function of the input — here, the
pseudo-legal-move graph derived from the ``simple_18`` board.

Mathematically, for square tokens ``x_i`` and the rule-derived adjacency
``E = {(i, j) : own piece on i has a legal/attack move to j}``:

    y_i = mean_{(i, j) in E} φ_θ(x_i, x_j)

where ``φ_θ`` is a small two-layer MLP shared across edges. The mask ``E``
is treated as a stop-gradient discrete tensor, exactly matching the MGR
spec's "topology is a non-differentiable branch" claim. Per-square outputs
are gated by the trunk's pool features and reduced to a scalar logit delta,
matching the additive head pattern from i246 / i248.

Architecture:

    base_logit       = i193_trunk(board)["logits"]
    sq_tokens        = SquareTokenEmbedder(board)        # (B, 64, d)
    edges_mask       = compute_legal_move_graph(board)   # (B, 64, 64) stop-grad
    edge_messages    = MLP(concat(x_i, x_j)) * edges     # (B, 64, 64, d)
    routed           = edge_messages.sum(dim=2) / degree # (B, 64, d)
    delta            = scalar(MLP(global_mean(routed)))
    gate             = sigmoid(MLP(trunk_pool))
    final_logit      = base_logit + gate * delta

The deferred internal proposals from external_02 (IDA, KISB, TBL, EEP) are
not implemented in this head; see ``ablations.md`` for the reasoning.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.primitives.rule_graph_features import (
    SQUARES,
    SquareTokenEmbedder,
    compute_legal_move_graph,
    rule_geometry,
)
from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    DualStreamFeatureBuilder,
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    "random_edges",  # replace legal-edge mask with a random mask of same density
    "dense_edges",   # use a fully-connected mask (degenerates MGR to a dense gather-scatter)
    "zero_delta",    # primitive_delta forced to zero (matches trunk baseline)
    "disable_gate",  # primitive_gate held at 1
    "trunk_only",    # zero features + zero delta
)


class MoveGraphRouter(nn.Module):
    """p006 — MoveGraphRouter primitive head over the i193 dual-stream trunk."""

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
        edge_hidden_dim: int = 48,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "MoveGraphRouter supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError(
                "MoveGraphRouter requires the simple_18 board tensor"
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

        edge_in = 2 * int(token_embed_dim)
        edge_hidden = int(edge_hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.LayerNorm(edge_in),
            nn.Linear(edge_in, edge_hidden),
            nn.GELU(),
            nn.Linear(edge_hidden, int(token_embed_dim)),
        )

        # Trunk pool dimension for gate / context features.
        trunk_pool_dim = (
            2 * self.trunk.exchange_encoder.output_dim
            + DualStreamFeatureBuilder.SUMMARY_DIM
        )
        self._trunk_pool_dim = trunk_pool_dim

        head_dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(int(token_embed_dim)),
            nn.Linear(int(token_embed_dim), int(head_hidden_dim)),
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
    def _build_edge_mask(self, board: torch.Tensor) -> torch.Tensor:
        if self.ablation == "dense_edges":
            return board.new_ones(board.shape[0], SQUARES, SQUARES)
        edges = compute_legal_move_graph(board, self._geometry)
        if self.ablation == "random_edges":
            density = edges.sum(dim=(1, 2), keepdim=True) / (SQUARES * SQUARES)
            rand = torch.rand_like(edges)
            edges = (rand < density).to(dtype=edges.dtype)
        return edges

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]
        trunk_out, trunk_joint = self._trunk_joint(board)
        base_logit = trunk_out["logits"]

        tokens = self.token_embed(board)  # (B, 64, d)
        edge_mask = self._build_edge_mask(board)  # (B, 64, 64)

        # Build per-edge concat features [x_i, x_j] then run the shared MLP.
        d = tokens.shape[-1]
        x_i = tokens.unsqueeze(2).expand(batch, SQUARES, SQUARES, d)
        x_j = tokens.unsqueeze(1).expand(batch, SQUARES, SQUARES, d)
        edge_in = torch.cat([x_i, x_j], dim=-1)  # (B, 64, 64, 2d)
        edge_msg = self.edge_mlp(edge_in)  # (B, 64, 64, d)
        edge_msg = edge_msg * edge_mask.unsqueeze(-1)

        degree = edge_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)  # (B, 64, 1)
        routed = edge_msg.sum(dim=2) / degree  # (B, 64, d)

        # Global summary across squares -> delta scalar.
        routed_norm = routed.pow(2).sum(dim=-1).sqrt()
        pooled = routed.mean(dim=1)  # (B, d)
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
        out["mgr_edge_count"] = edge_mask.sum(dim=(1, 2))
        out["mgr_mean_routed_norm"] = routed_norm.mean(dim=1)
        out["mgr_pooled_norm"] = pooled.pow(2).sum(dim=-1).sqrt()
        return out


def build_move_graph_router_from_config(config: dict[str, Any]) -> MoveGraphRouter:
    cfg = dict(config)
    return MoveGraphRouter(
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
        edge_hidden_dim=int(cfg.get("edge_hidden_dim", 48)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )

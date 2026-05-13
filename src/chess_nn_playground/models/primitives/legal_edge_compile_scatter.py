"""Content-Compiled Legal Edge Scatter (p011) — typed gated message scatter head.

Promoted from
``ideas/research/primitives/external_14_ray_occlusion_legal_edge_compile_scatter.md``.
The file ranks Occlusion-Gated Ray Scan as #1 and Content-Compiled Legal
Edge Scatter as #2; since p010 already implements the ray-scan flavour
(via external_12's Ray-Occlusion Semiring Scan, which is essentially the
same operator), this entry implements the **legal-edge compile-scatter**
primitive — content-determined typed edge compilation followed by a sparse
gated message scatter.

For board symbols ``P`` and per-square features ``X``:

    E(P)   = compile_typed_edges(P)    # (B, E_max, 3) with (src, dst, type) entries
    s_e    = σ(a_τ . [x_src, x_dst])
    m_e    = W_τ x_src
    y_dst  = sum_{e=(*, dst, τ)} s_e * m_e

We use a dense ``(B, T, 64, 64)`` mask realisation rather than a packed
edge list because variable-length ragged tensors are a poor fit for
PyTorch eager mode at scout scale — the spec's "no n^2 mask" objection is
addressed by per-type masking that keeps the operator's structural meaning
(typed connectivity is content-determined) while avoiding ragged kernel
overhead. Per-edge gates are computed from the source/destination token
pair, matching the (σ-gated message) compute graph that the file's
self-audit highlights as "not standard masked attention".

Deferred internal proposals from external_14 (Occlusion-Gated Ray Scan,
Delta-Apply Linear) are covered by p010 and ΔAcc (deferred in p009/p010)
respectively; see ``ablations.md``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.primitives.legal_move_graph_delta import (
    PIECE_TYPE_NAMES,
    _compute_typed_legal_edges,
)
from chess_nn_playground.models.primitives.rule_graph_features import (
    NUM_PIECE_TYPES,
    SQUARES,
    SquareTokenEmbedder,
    rule_geometry,
)
from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    DualStreamFeatureBuilder,
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    "no_edge_gate",          # σ(a_τ . [x_src, x_dst]) replaced by 1
    "random_typed_edges",   # typed adjacency replaced by random mask of same density
    "shared_type_weight",   # collapse W_τ to one shared linear (deactivates typed channel)
    "zero_delta",
    "disable_gate",
    "trunk_only",
)


class LegalEdgeCompileScatter(nn.Module):
    """p011 — Legal-Edge Compile-Scatter head over the i193 trunk."""

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
        edge_gate_hidden: int = 16,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "LegalEdgeCompileScatter supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError(
                "LegalEdgeCompileScatter requires the simple_18 board tensor"
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

        # σ(a_τ . [x_src, x_dst]) per type. We share a small 2-layer MLP per
        # type; the spec only requires per-edge gating, not per-edge weights.
        edge_in = 2 * int(token_embed_dim)
        if self.ablation == "shared_type_weight":
            self.message_linear = nn.Linear(int(token_embed_dim), int(message_dim))
            self.message_linears = None
        else:
            self.message_linear = None
            self.message_linears = nn.ModuleList(
                [
                    nn.Linear(int(token_embed_dim), int(message_dim))
                    for _ in range(NUM_PIECE_TYPES)
                ]
            )
        self.edge_gate_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(edge_in),
                    nn.Linear(edge_in, int(edge_gate_hidden)),
                    nn.GELU(),
                    nn.Linear(int(edge_gate_hidden), 1),
                )
                for _ in range(NUM_PIECE_TYPES)
            ]
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
        self._message_dim = int(message_dim)
        self._token_dim = int(token_embed_dim)

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

    def _project_messages(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return ``(B, R, 64, message_dim)`` per-type W_τ projections."""
        if self.message_linears is None:
            assert self.message_linear is not None
            return self.message_linear(tokens).unsqueeze(1).expand(
                tokens.shape[0], NUM_PIECE_TYPES, SQUARES, -1
            )
        projected = torch.stack(
            [linear(tokens) for linear in self.message_linears], dim=1
        )
        return projected  # (B, R, 64, message_dim)

    def _edge_gates(
        self,
        tokens: torch.Tensor,
        edges_per_type: torch.Tensor,
    ) -> torch.Tensor:
        """Per-type per-edge gates ``(B, R, 64, 64)`` of σ(a_τ . [x_src, x_dst]).

        Computed only for edges where the typed adjacency is 1; off-edge
        entries are zero so the downstream scatter ignores them.
        """
        if self.ablation == "no_edge_gate":
            return edges_per_type

        batch, n_types, n_src, n_dst = edges_per_type.shape
        d = tokens.shape[-1]
        x_src = tokens.unsqueeze(2).expand(batch, n_src, n_dst, d)
        x_dst = tokens.unsqueeze(1).expand(batch, n_src, n_dst, d)
        edge_in = torch.cat([x_src, x_dst], dim=-1)  # (B, 64, 64, 2d)

        gates = []
        for piece in range(NUM_PIECE_TYPES):
            mask = edges_per_type[:, piece]  # (B, 64, 64)
            logits = self.edge_gate_mlps[piece](edge_in).squeeze(-1)  # (B, 64, 64)
            g = torch.sigmoid(logits) * mask
            gates.append(g)
        return torch.stack(gates, dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]
        trunk_out, trunk_joint = self._trunk_joint(board)
        base_logit = trunk_out["logits"]

        tokens = self.token_embed(board)  # (B, 64, d)
        edges_per_type = self._build_edges(board)  # (B, R, 64, 64)
        gates = self._edge_gates(tokens, edges_per_type)  # (B, R, 64, 64)
        projected = self._project_messages(tokens)  # (B, R, 64, message_dim)

        # Per-type scatter: y_dst = sum_src gates[type, src, dst] * projected[type, src]
        # gates.transpose(-1, -2) shape (B, R, 64_dst, 64_src) bmm projected (B, R, 64_src, m)
        b, r, n, _ = projected.shape
        gates_t = gates.transpose(-1, -2).reshape(b * r, n, n)
        projected_flat = projected.reshape(b * r, n, -1)
        msg_per_type = torch.bmm(gates_t, projected_flat).view(b, r, n, -1)
        # Normalise by typed in-degree (gate weight sum).
        gate_in_deg = gates.sum(dim=-2).unsqueeze(-1).clamp_min(1.0e-3)  # (B, R, 64, 1)
        msg_per_type = msg_per_type / gate_in_deg

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
        out["lecs_edge_count"] = edges_per_type.sum(dim=(1, 2, 3))
        out["lecs_gate_mean"] = gates.sum(dim=(1, 2, 3)) / edges_per_type.sum(dim=(1, 2, 3)).clamp_min(1.0)
        type_norm = msg_per_type.pow(2).mean(dim=(2, 3)).sqrt()
        for piece_idx, piece_name in enumerate(PIECE_TYPE_NAMES):
            out[f"lecs_msg_norm_{piece_name}"] = type_norm[:, piece_idx]
        return out


def build_legal_edge_compile_scatter_from_config(
    config: dict[str, Any],
) -> LegalEdgeCompileScatter:
    cfg = dict(config)
    return LegalEdgeCompileScatter(
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
        edge_gate_hidden=int(cfg.get("edge_gate_hidden", 16)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )

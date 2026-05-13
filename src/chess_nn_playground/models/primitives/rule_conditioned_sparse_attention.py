"""Rule-Conditioned Sparse Attention — MobScan flavour (p008).

Promoted from
``ideas/research/primitives/external_04_rule_conditioned_sparse_attention_mobscan.md``.
The file ranks RCSA (Ray-Cast Sparse Attention) and MobScan (Mobility-Scan
Selective SSM) as its top two proposals. Since p007 already implements the
ray-cast attention flavour as ARSA, this entry implements **MobScan** —
a selective-recurrence (Mamba/Mamba-2-style) sparse propagation along the
input-determined legal-move graph. The slug ``rule_conditioned_sparse_attention_mobscan``
explicitly calls out the MobScan variant; deferring the in-file RCSA
proposal to ablations / a future entry keeps the two batch primitives
substantively different.

MobScan replaces Mamba's implicit 1-D causal chain by a per-batch sparse
DAG defined by the rule-derived legal-move graph. For square tokens
``x_s``, learned input-conditioned gates ``A_s, B_s, C_s``, and inbound
edge set ``parents(s) = {p : (p, s) in E}``:

    h_s   = A_s ⊙ mean_{p in parents(s)} h_p   +   B_s ⊙ x_s
    y_s   = C_s ⊙ h_s

For a single-step (depth = 1) variant — sufficient for the scout-scale
falsifier — this collapses to a sparse weighted aggregation over inbound
edges and a learned input-conditioned gate. The implementation runs the
recurrence for ``num_iterations`` steps using the same gates each time
(weight-tied unrolled scan), then pools to a delta logit. The mask itself
is treated as ``stop_gradient`` exactly as MobScan specifies — the graph
is a deterministic discrete function of the board.

Architecture (additive, gated):

    base_logit       = i193_trunk(board)["logits"]
    tokens           = SquareTokenEmbedder(board)         # (B, 64, d)
    edges            = compute_legal_move_graph(board)    # (B, 64, 64) stop-grad
    A, B, C          = input_conditioned_gates(tokens)
    h_0              = B * tokens
    for t in 1..T:
        h_t = A * mean_inbound(h_{t-1}, edges) + B * tokens
    y_s              = C * h_T
    pooled           = mean_squares(y)
    delta            = MLP(pooled)
    gate             = sigmoid(MLP(trunk_pool))
    final_logit      = base_logit + gate * delta
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
    "random_edges",         # legal-move graph replaced by random mask of same density
    "dense_edges",           # legal-move graph replaced by all-ones (degenerates scan)
    "untied_state",          # disable selective gates (set A=0.5, B=0.5, C=1)
    "single_iteration",      # force num_iterations=1 regardless of config
    "zero_delta",
    "disable_gate",
    "trunk_only",
)


class RuleConditionedSparseAttention(nn.Module):
    """p008 — MobScan-style rule-conditioned sparse recurrence head."""

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
        state_dim: int = 32,
        num_iterations: int = 3,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "RuleConditionedSparseAttention supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError(
                "RuleConditionedSparseAttention requires the simple_18 board tensor"
            )
        if int(num_iterations) < 1 or int(num_iterations) > 8:
            raise ValueError("num_iterations must be between 1 and 8")
        if str(ablation) not in ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(ALLOWED_ABLATIONS)}"
            )

        self.num_classes = 1
        self.ablation = str(ablation)
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self._geometry = rule_geometry()
        self._configured_iterations = int(num_iterations)

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
        # Selective S6 input-conditioned gates: A controls retention from
        # parents, B controls input injection, C controls output read-out.
        self.gate_A = nn.Linear(int(token_embed_dim), int(state_dim))
        self.gate_B = nn.Linear(int(token_embed_dim), int(state_dim))
        self.gate_C = nn.Linear(int(token_embed_dim), int(state_dim))
        self.input_proj = nn.Linear(int(token_embed_dim), int(state_dim))

        trunk_pool_dim = (
            2 * self.trunk.exchange_encoder.output_dim
            + DualStreamFeatureBuilder.SUMMARY_DIM
        )
        self._trunk_pool_dim = trunk_pool_dim
        head_dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(int(state_dim)),
            nn.Linear(int(state_dim), int(head_hidden_dim)),
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

    @property
    def num_iterations(self) -> int:
        return 1 if self.ablation == "single_iteration" else int(self._configured_iterations)

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
        edges = self._build_edge_mask(board)  # (B, 64, 64)
        # Mean over inbound parents -- transpose along edges so columns index "child" side.
        in_count = edges.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B, 1, 64)
        norm_edges = edges / in_count  # incoming-normalised aggregator

        if self.ablation == "untied_state":
            a = tokens.new_full((batch, SQUARES, self.gate_A.out_features), 0.5)
            b = tokens.new_full((batch, SQUARES, self.gate_B.out_features), 0.5)
            c = tokens.new_ones((batch, SQUARES, self.gate_C.out_features))
        else:
            a = torch.sigmoid(self.gate_A(tokens))
            b = torch.sigmoid(self.gate_B(tokens))
            c = torch.sigmoid(self.gate_C(tokens))

        input_proj = self.input_proj(tokens)  # (B, 64, state_dim)
        h = b * input_proj

        for _ in range(self.num_iterations):
            # Aggregate inbound parents: rows of `h` indexed by parent square.
            # Use bmm with norm_edges (B, 64, 64): inbound[child] = sum_parent edges[parent, child] / in_count[child] * h[parent].
            # PyTorch broadcasting: norm_edges.transpose(-1,-2) @ h = (B, 64, state)
            inbound = torch.bmm(norm_edges.transpose(-1, -2), h)
            h = a * inbound + b * input_proj

        y = c * h  # (B, 64, state_dim)

        pooled = y.mean(dim=1)  # (B, state_dim)
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
        out["mobscan_edge_count"] = edges.sum(dim=(1, 2))
        out["mobscan_state_norm"] = pooled.pow(2).sum(dim=-1).sqrt()
        out["mobscan_gate_A_mean"] = a.mean(dim=(1, 2))
        out["mobscan_gate_B_mean"] = b.mean(dim=(1, 2))
        out["mobscan_gate_C_mean"] = c.mean(dim=(1, 2))
        return out


def build_rule_conditioned_sparse_attention_from_config(
    config: dict[str, Any],
) -> RuleConditionedSparseAttention:
    cfg = dict(config)
    return RuleConditionedSparseAttention(
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
        state_dim=int(cfg.get("state_dim", 32)),
        num_iterations=int(cfg.get("num_iterations", 3)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )

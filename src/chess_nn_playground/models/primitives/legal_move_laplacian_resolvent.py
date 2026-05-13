"""Legal-Move Laplacian Resolvent (p031, LM-LPP) primitive.

Source: ``ideas/research/primitives/external_06_high_risk_legal_graph_delta_state_primitives.md``
(Section 1, "Legal-Move Laplacian Pseudoinverse Propagation"; explicitly
identified as the single highest-ranked proposal in that file).

Architecture (additive, gated):

```
board (B, 18, 8, 8)
    -> i193 ExchangeThenKingDualStreamNetwork  -> base_logit, joint pool
    -> compute_legal_move_graph(board)         -> (B, 64, 64) stop-grad adjacency
    -> per-square features X (B, 64, d_sq)     from a small board-projection MLP
    -> piece-conditioned edge weight w(piece(i))
    -> signed Laplacian L = D - W * A          (W carries learned per-piece weights)
    -> Y = sum_{k=0..K} alpha^k * (L @ X) @ Theta  (truncated Neumann series)
    -> pool Y, gate with the i193 trunk diagnostics
    -> final_logit = base_logit + gate * primitive_delta
```

The Neumann truncation is the spec's defining choice: the closest existing
PyTorch op is ``GAT`` with a stop-gradient mask, which would correspond to
``K=1`` -- the K=1 case is documented as the explicit "rebrand of GAT"
failure mode in the source primitive. The default config uses ``K=4`` so
the operator is meaningfully different from a single hop. ``alpha`` is
constrained by ``alpha = alpha_init * tanh(alpha_logit)`` to keep its
absolute value bounded; spectral clipping by power iteration is documented
in the failure-mode catalogue and would be the next upgrade.

CRTK metadata, source labels, verification flags, and engine scores are
*not* consulted. The adjacency depends on the simple_18 piece planes,
side-to-move plane, and (transitively) blocker resolution -- all rule-
derived from the current FEN state.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.primitives.legal_move_graph import (
    SQUARES,
    LegalMoveGraph,
    compute_legal_move_graph,
)
from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    DualStreamFeatureBuilder,
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


# Piece weight bands: own pawn / knight / bishop / rook / queen / king / enemy.
# Per-piece-conditioned edge weight follows the spec's ``w(piece(i, b))`` term.
_PIECE_WEIGHT_TABLE = (
    "own_pawn",
    "own_knight",
    "own_bishop",
    "own_rook",
    "own_queen",
    "own_king",
    "enemy_pawn",
    "enemy_knight",
    "enemy_bishop",
    "enemy_rook",
    "enemy_queen",
    "enemy_king",
)

NUM_PIECE_CHANNELS = 12
ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    "k1_gat_rebrand",          # K := 1: collapse to single-hop legal-mask GAT control
    "uniform_piece_weights",   # disable per-piece w(piece) gating
    "shuffle_adjacency",       # in-batch permutation of the legal-move graph
    "zero_alpha",              # alpha forced to 0 -> Y = X * Theta only
    "zero_delta",              # primitive_delta forced to 0
    "trunk_only",              # equivalent to zero_delta
    "disable_gate",            # gate forced to 1.0
)


class LegalMoveLaplacianResolvent(nn.Module):
    """p031 -- Legal-Move Laplacian Pseudoinverse Propagation (LM-LPP).

    Wraps the i193 dual-stream trunk and computes an additive, gated logit
    delta from a truncated Neumann-series resolvent of the per-board legal-
    move Laplacian applied to per-square learned features.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        # Trunk hyper-parameters mirror the i193 builder.
        trunk_channels: int = 64,
        trunk_hidden_dim: int = 96,
        trunk_depth: int = 2,
        trunk_dropout: float = 0.1,
        trunk_use_batchnorm: bool = True,
        trunk_gate_dim: int | None = None,
        trunk_ablation: str = "none",
        # LM-LPP head hyper-parameters.
        feature_dim: int = 32,
        neumann_terms: int = 4,
        alpha_init: float = 0.25,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "LegalMoveLaplacianResolvent supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError(
                "LegalMoveLaplacianResolvent requires the simple_18 board tensor"
            )
        if int(neumann_terms) < 1 or int(neumann_terms) > 8:
            raise ValueError("neumann_terms must be between 1 and 8")
        if str(ablation) not in ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(ALLOWED_ABLATIONS)}"
            )

        self.num_classes = 1
        self.ablation = str(ablation)
        self.neumann_terms = int(neumann_terms)
        self.alpha_init = float(alpha_init)
        self.feature_dim = int(feature_dim)
        self.spec = BoardTensorSpec(input_channels=int(input_channels))

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

        # Per-square feature projector: take the 12 piece-presence planes
        # (and the side-to-move plane) and map each square's local descriptor
        # to a ``feature_dim`` vector that participates in the resolvent.
        # Using only the rule-derived ``simple_18`` slice keeps the per-square
        # features compatible with the stop-gradient adjacency.
        self.square_feature_proj = nn.Linear(NUM_PIECE_CHANNELS + 1, self.feature_dim)

        # Per-piece edge weight ``w(piece(i, b))`` -- 12 piece channels plus
        # an "empty" slot for completeness (own_mask handles gating). Init in
        # a narrow band to keep early Laplacian spectra small.
        self.piece_edge_weights = nn.Parameter(torch.empty(NUM_PIECE_CHANNELS))
        nn.init.normal_(self.piece_edge_weights, mean=1.0, std=0.1)

        # alpha is stored as a free scalar logit; the effective alpha is
        # ``alpha_init * tanh(alpha_logit)`` so |alpha| < alpha_init always.
        self.alpha_logit = nn.Parameter(torch.zeros(1))

        # Mixing matrix Theta in the spec, applied after the Neumann sum.
        self.theta = nn.Linear(self.feature_dim, self.feature_dim, bias=False)

        # Pool the (B, 64, feature_dim) tensor to a (B, head_hidden_dim) summary.
        self.pool_proj = nn.Sequential(
            nn.LayerNorm(self.feature_dim * 2),
            nn.Linear(self.feature_dim * 2, int(head_hidden_dim)),
            nn.GELU(),
        )

        # Delta head: turns the pooled resolvent summary into a scalar.
        dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(int(head_hidden_dim)),
            nn.Linear(int(head_hidden_dim), max(16, int(head_hidden_dim) // 2)),
            nn.GELU(),
            dropout_module,
            nn.Linear(max(16, int(head_hidden_dim) // 2), 1),
        )

        # Gate: trunk diagnostics + a few resolvent-spectral summaries.
        gate_in = 4 + 3  # trunk diagnostics + (mean_norm, max_norm, deg_mean)
        self.gate_head = nn.Sequential(
            nn.LayerNorm(gate_in),
            nn.Linear(gate_in, max(16, int(head_hidden_dim) // 2)),
            nn.GELU(),
            nn.Linear(max(16, int(head_hidden_dim) // 2), 1),
        )
        with torch.no_grad():
            final = self.gate_head[-1]
            if isinstance(final, nn.Linear):
                final.bias.fill_(float(gate_init))

    @staticmethod
    def _square_descriptor(board: torch.Tensor) -> torch.Tensor:
        """Per-square 13-d descriptor used to seed the resolvent features."""
        piece_planes = board[:, :NUM_PIECE_CHANNELS].flatten(2).clamp(0.0, 1.0)  # (B, 12, 64)
        batch = board.shape[0]
        stm_scalar = board[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0)  # (B,)
        stm_broadcast = stm_scalar.view(batch, 1, 1).expand(batch, 1, SQUARES)
        descriptor = torch.cat([piece_planes, stm_broadcast], dim=1)  # (B, 13, 64)
        return descriptor.transpose(1, 2).contiguous()  # (B, 64, 13)

    def _piece_weighted_adjacency(
        self,
        board: torch.Tensor,
        graph: LegalMoveGraph,
    ) -> torch.Tensor:
        """Multiply adjacency rows by w(piece(i, b)) per chess rule."""
        if self.ablation == "uniform_piece_weights":
            return graph.adjacency
        piece_planes = board[:, :NUM_PIECE_CHANNELS].flatten(2).clamp(0.0, 1.0)  # (B, 12, 64)
        # Per-square scalar weight: weighted sum of piece-existence * piece-weight.
        # Empty squares contribute 0, which is fine because the adjacency row of
        # an empty square is structurally zero.
        weights = (piece_planes * self.piece_edge_weights.view(1, NUM_PIECE_CHANNELS, 1)).sum(
            dim=1
        )  # (B, 64)
        weighted = graph.adjacency * weights.unsqueeze(-1)
        return weighted

    def _build_laplacian(self, weighted_adjacency: torch.Tensor) -> torch.Tensor:
        """L = D - A as a per-board (B, 64, 64) signed Laplacian."""
        degree = weighted_adjacency.sum(dim=-1)  # (B, 64)
        eye = torch.eye(SQUARES, device=weighted_adjacency.device, dtype=weighted_adjacency.dtype)
        d_diag = degree.unsqueeze(-1) * eye.unsqueeze(0)
        return d_diag - weighted_adjacency

    def _resolvent(self, laplacian: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Truncated Neumann series sum_{k=0..K} alpha^k L^k X."""
        alpha = self.alpha_init * torch.tanh(self.alpha_logit)
        if self.ablation == "zero_alpha":
            alpha = torch.zeros_like(alpha)
        K = 1 if self.ablation == "k1_gat_rebrand" else self.neumann_terms
        accumulator = features.clone()
        current = features
        alpha_value = alpha.view(-1)
        for _ in range(int(K)):
            current = torch.bmm(laplacian, current)
            accumulator = accumulator + alpha_value * current
        return accumulator

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]
        device = board.device
        dtype = board.dtype

        # 1. trunk forward (rule-aware base logit + diagnostics).
        trunk_out = self.trunk(board)
        base_logit = trunk_out["logits"].view(-1)

        # 2. legal-move graph (rule-derived, stop-gradient).
        with torch.no_grad():
            graph = compute_legal_move_graph(board)
            if self.ablation == "shuffle_adjacency" and batch > 1:
                perm = torch.randperm(batch, device=device)
                graph = LegalMoveGraph(
                    adjacency=graph.adjacency[perm],
                    own_piece_mask=graph.own_piece_mask[perm],
                    enemy_piece_mask=graph.enemy_piece_mask[perm],
                    move_type=graph.move_type[perm],
                    ray_direction=graph.ray_direction[perm],
                    degree=graph.degree[perm],
                    occupancy=graph.occupancy[perm],
                )

        # 3. weighted adjacency and Laplacian.
        weighted = self._piece_weighted_adjacency(board, graph)
        laplacian = self._build_laplacian(weighted)

        # 4. per-square features from the rule-derived descriptor.
        descriptor = self._square_descriptor(board).to(dtype=dtype)
        features = self.square_feature_proj(descriptor)  # (B, 64, feature_dim)

        # 5. Neumann series.
        propagated = self._resolvent(laplacian, features)
        mixed = self.theta(propagated)  # (B, 64, feature_dim)

        # 6. Pool weighted by own-piece mask + uniform pool.
        own_mask = graph.own_piece_mask.unsqueeze(-1)  # (B, 64, 1)
        own_weight = own_mask.sum(dim=1).clamp_min(1.0)
        own_pooled = (mixed * own_mask).sum(dim=1) / own_weight
        global_pooled = mixed.mean(dim=1)
        pooled = torch.cat([own_pooled, global_pooled], dim=1)  # (B, 2 * feature_dim)
        summary = self.pool_proj(pooled)
        delta_raw = self.delta_head(summary).view(-1)

        # 7. Gate from trunk diagnostics + spectral summaries.
        diag_keys = ("gate", "gate_entropy", "mechanism_energy", "stream_disagreement")
        diag = torch.stack(
            [trunk_out[k].detach() for k in diag_keys],
            dim=1,
        )
        mean_norm = mixed.pow(2).mean(dim=(1, 2)).clamp_min(0.0).sqrt()
        max_norm = mixed.pow(2).mean(dim=2).sqrt().amax(dim=1)
        deg_mean = graph.degree.mean(dim=1).to(dtype=dtype) / float(SQUARES)
        gate_input = torch.cat([diag, mean_norm.unsqueeze(-1), max_norm.unsqueeze(-1), deg_mean.unsqueeze(-1)], dim=1)
        gate_logit = self.gate_head(gate_input).view(-1)
        gate = torch.sigmoid(gate_logit)
        if self.ablation == "disable_gate":
            gate = torch.ones_like(gate)

        if self.ablation in {"zero_delta", "trunk_only"}:
            primitive_delta = torch.zeros_like(base_logit)
            gate_applied = torch.zeros_like(gate)
        else:
            primitive_delta = delta_raw
            gate_applied = gate

        contribution = gate_applied * primitive_delta
        logits = base_logit + contribution

        eps = 1.0e-6
        gate_clamped = gate.clamp(eps, 1.0 - eps)
        gate_entropy = -(
            gate_clamped * gate_clamped.log()
            + (1.0 - gate_clamped) * (1.0 - gate_clamped).log()
        )

        alpha_effective = self.alpha_init * torch.tanh(self.alpha_logit)
        if self.ablation == "zero_alpha":
            alpha_effective = torch.zeros_like(alpha_effective)
        alpha_scalar = alpha_effective.view(-1).repeat(batch)

        out: dict[str, torch.Tensor] = {}
        for key, value in trunk_out.items():
            if key in {"logits", "proposal_profile_strength", "proposal_keyword_count"}:
                continue
            out[f"trunk_{key}"] = value
        out["logits"] = logits
        out["base_logit"] = base_logit
        out["primitive_delta"] = primitive_delta
        out["primitive_delta_raw"] = delta_raw
        out["primitive_gate"] = gate
        out["primitive_gate_applied"] = gate_applied
        out["primitive_gate_logit"] = gate_logit
        out["primitive_gate_entropy"] = gate_entropy
        out["primitive_contribution"] = contribution
        out["lmlpp_alpha"] = alpha_scalar
        out["lmlpp_mean_feature_norm"] = mean_norm
        out["lmlpp_max_feature_norm"] = max_norm
        out["lmlpp_degree_mean"] = deg_mean
        out["lmlpp_neumann_terms"] = logits.new_full((batch,), float(self.neumann_terms))
        out["mechanism_energy"] = trunk_out["mechanism_energy"] + max_norm.detach()
        out["proposal_profile_strength"] = (
            primitive_delta.detach().abs() * gate_entropy
        )
        out["proposal_keyword_count"] = logits.new_full((batch,), float(self.feature_dim))
        return out


def build_legal_move_laplacian_resolvent_from_config(
    config: dict[str, Any],
) -> LegalMoveLaplacianResolvent:
    cfg = dict(config)
    return LegalMoveLaplacianResolvent(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim")),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        feature_dim=int(cfg.get("feature_dim", 32)),
        neumann_terms=int(cfg.get("neumann_terms", cfg.get("K", 4))),
        alpha_init=float(cfg.get("alpha_init", 0.25)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )


__all__ = (
    "ALLOWED_ABLATIONS",
    "LegalMoveLaplacianResolvent",
    "build_legal_move_laplacian_resolvent_from_config",
)


_ = DualStreamFeatureBuilder  # keep import for downstream cross-references

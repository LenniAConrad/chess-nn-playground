"""Move-Kernel Operator (p033, MKO) primitive.

Source: ``ideas/research/primitives/external_28_sparse_differential_accumulator_move_kernel.md``
(Section "primitive_mko"). The proposal defines a rule-informed
"convolution" whose kernel is indexed by the chess move-type relationship
between source and target rather than by spatial offset:

    Y_i = sum_{j in M(i)} W_{type(i, j)} @ X_j

with ``M(i)`` the set of squares reachable from ``i`` under *any* piece
type (knight, king, sliding pieces -- not blocker resolved) and
``type(i, j)`` the chess move relationship (rank, file, diagonal,
antidiagonal, knight, king-step). The operator captures the spec's
defining property: a knight at f3 and a knight at d4 share the same
geometric influence, so weights are tied across squares and across
specific source-target pairs that share a move type.

This is distinct from p032 DAG (which uses the per-board blocker-resolved
legal-move adjacency) in that the connectivity is a *fixed* chess-rule
geometric reach -- it does not depend on per-board occupancy. The MKO
kernel is therefore a static (64, 64) per-type mask combined with
learned per-type weights.

The deferred internal proposals from external_28 (SDA, IPN, SRA, MPC) are
documented in the idea registry notes. SDA (sparse differential
accumulator) overlaps with i248 TSDP and the existing
``one_ply_counterfactual_move_landscape_network`` / ``counterfactual_move_delta_spectrum_network``
delta family; IPN (involutional parity normalisation) is an orthogonal
symmetry primitive; SRA (selective ray attention) overlaps with p035 and
``ray_state_space_scan_network``; MPC (metamorphic piece convolution)
is a separable hyper-network primitive.

CRTK metadata, source labels, verification flags, and engine scores are
*not* consulted. The MKO kernel depends on chess-rule geometry only;
piece presence enters only through the per-square seed feature, which is
derived from the simple_18 piece planes.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.primitives.legal_move_graph import (
    MOVE_TYPE_ANTIDIAG,
    MOVE_TYPE_DIAG,
    MOVE_TYPE_FILE,
    MOVE_TYPE_KING,
    MOVE_TYPE_KNIGHT,
    MOVE_TYPE_RANK,
    SQUARES,
    _get_geometry,
)
from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


NUM_PIECE_CHANNELS = 12
# MKO uses the geometric reach mask: knight + king + sliding (no blocker
# resolution). The five sliding ray types plus knight gives six classes;
# we keep king-adjacency as a seventh slot for ``king_edges`` (distinct
# from sliding-step adjacency in MKO because the source primitive treats
# king vs sliding as different move types when both could apply).
MKO_MOVE_TYPES: tuple[tuple[str, int], ...] = (
    ("knight", MOVE_TYPE_KNIGHT),
    ("rank", MOVE_TYPE_RANK),
    ("file", MOVE_TYPE_FILE),
    ("diag", MOVE_TYPE_DIAG),
    ("antidiag", MOVE_TYPE_ANTIDIAG),
    ("king", MOVE_TYPE_KING),
)
NUM_MKO_TYPES = len(MKO_MOVE_TYPES)

ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    "shared_kernel",        # collapse all move types to one shared projection
    "scalar_per_type",      # per-type scalar instead of per-type matrix
    "shuffle_features",     # in-batch shuffle of seed features (rule-feature falsifier)
    "zero_delta",
    "trunk_only",
    "disable_gate",
)


def _build_static_move_type_masks() -> torch.Tensor:
    """Static (T, 64, 64) float mask for each MKO move type.

    The masks are NOT blocker-resolved: a queen at a1 reaches every square
    on its rank / file / diagonal regardless of occupancy. Knight and king
    masks are the standard leap masks. The masks include only the
    reachable cells (1.0) and zero elsewhere; diagonal entries (i, i) are
    zero by construction.
    """
    geometry = _get_geometry()
    knight = geometry.knight_edges
    king = geometry.king_edges
    move_type = geometry.move_type
    masks = torch.zeros(NUM_MKO_TYPES, SQUARES, SQUARES, dtype=torch.float32)
    for slot, (name, code) in enumerate(MKO_MOVE_TYPES):
        if name == "knight":
            masks[slot] = knight
        elif name == "king":
            masks[slot] = king
        else:
            masks[slot] = (move_type == code).to(dtype=torch.float32)
    return masks


class MoveKernelOperator(nn.Module):
    """p033 -- Move-Kernel Operator over the i193 trunk.

    The operator applies a per-move-type linear projection ``W_t`` to the
    per-square seed features and aggregates the result along a static
    move-type mask ``M_t``. Knight, king-step, and the four sliding ray
    directions each carry their own learned ``W_t``.

    Aggregation:

        Y[b, i, d] = sum_t sum_j M_t[i, j] * (W_t X[b])[j, d]

    Implementation collapses the inner sums to a single batched einsum
    after stacking the projected features.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        # Trunk hyper-parameters.
        trunk_channels: int = 64,
        trunk_hidden_dim: int = 96,
        trunk_depth: int = 2,
        trunk_dropout: float = 0.1,
        trunk_use_batchnorm: bool = True,
        trunk_gate_dim: int | None = None,
        trunk_ablation: str = "none",
        # MKO head hyper-parameters.
        feature_dim: int = 24,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError("MoveKernelOperator supports puzzle_binary one-logit contract")
        if int(input_channels) != 18:
            raise ValueError("MoveKernelOperator requires the simple_18 board tensor")
        if str(ablation) not in ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(ALLOWED_ABLATIONS)}"
            )
        self.num_classes = 1
        self.ablation = str(ablation)
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

        self.square_feature_proj = nn.Linear(NUM_PIECE_CHANNELS + 1, self.feature_dim)

        # Per-type matrix projections (used by the default operator).
        self.type_projections = nn.ModuleList(
            [nn.Linear(self.feature_dim, self.feature_dim) for _ in MKO_MOVE_TYPES]
        )
        # Shared projection (used by the ``shared_kernel`` ablation).
        self.shared_projection = nn.Linear(self.feature_dim, self.feature_dim)
        # Per-type scalar (used by the ``scalar_per_type`` ablation -- replaces
        # the matrix with a single learned scalar per type).
        self.type_scalars = nn.Parameter(torch.ones(NUM_MKO_TYPES))

        self.aggregator = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, int(head_hidden_dim)),
            nn.GELU(),
        )

        dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(int(head_hidden_dim) * 2),
            nn.Linear(int(head_hidden_dim) * 2, max(16, int(head_hidden_dim) // 2)),
            nn.GELU(),
            dropout_module,
            nn.Linear(max(16, int(head_hidden_dim) // 2), 1),
        )

        gate_in = 4 + NUM_MKO_TYPES
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

        masks = _build_static_move_type_masks()
        self.register_buffer("move_type_masks", masks, persistent=False)
        # Per-type row degree (for diagnostic and normalisation hints).
        self.register_buffer(
            "move_type_row_degree",
            masks.sum(dim=-1),  # (T, 64)
            persistent=False,
        )

    @staticmethod
    def _square_descriptor(board: torch.Tensor) -> torch.Tensor:
        piece_planes = board[:, :NUM_PIECE_CHANNELS].flatten(2).clamp(0.0, 1.0)
        batch = board.shape[0]
        stm_scalar = board[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0)
        stm = stm_scalar.view(batch, 1, 1).expand(batch, 1, SQUARES)
        return torch.cat([piece_planes, stm], dim=1).transpose(1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]
        dtype = board.dtype

        trunk_out = self.trunk(board)
        base_logit = trunk_out["logits"].view(-1)

        descriptor = self._square_descriptor(board).to(dtype=dtype)
        if self.ablation == "shuffle_features" and batch > 1:
            perm = torch.randperm(batch, device=descriptor.device)
            descriptor = descriptor[perm]
        x_features = self.square_feature_proj(descriptor)  # (B, 64, feature_dim)

        masks = self.move_type_masks.to(device=board.device, dtype=dtype)
        # masks shape: (T, 64, 64), broadcast to (1, T, 64, 64) for per-sample matmul.
        if self.ablation == "shared_kernel":
            shared = self.shared_projection(x_features)  # (B, 64, feature_dim)
            collapsed_mask = masks.sum(dim=0).clamp(0.0, 1.0)
            aggregated = torch.einsum("ij,bjd->bid", collapsed_mask, shared)
        elif self.ablation == "scalar_per_type":
            projected = []
            for slot in range(NUM_MKO_TYPES):
                projected.append(
                    self.type_scalars[slot].view(1, 1, 1) * x_features
                )
            projected_stack = torch.stack(projected, dim=1)  # (B, T, 64, feature_dim)
            aggregated = torch.einsum("tij,btjd->bid", masks, projected_stack)
        else:
            projected = []
            for slot, proj in enumerate(self.type_projections):
                projected.append(proj(x_features))
            projected_stack = torch.stack(projected, dim=1)  # (B, T, 64, feature_dim)
            aggregated = torch.einsum("tij,btjd->bid", masks, projected_stack)

        aggregated_hidden = self.aggregator(aggregated)  # (B, 64, head_hidden_dim)

        piece_planes = board[:, :NUM_PIECE_CHANNELS].flatten(2).clamp(0.0, 1.0)
        white_mask = piece_planes[:, :6].sum(dim=1).clamp(0.0, 1.0)
        black_mask = piece_planes[:, 6:].sum(dim=1).clamp(0.0, 1.0)
        stm = board[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0).view(-1, 1)
        own_mask = stm * white_mask + (1.0 - stm) * black_mask
        own_weight = own_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        own_pooled = (aggregated_hidden * own_mask.unsqueeze(-1)).sum(dim=1) / own_weight
        global_pooled = aggregated_hidden.mean(dim=1)
        delta_raw = self.delta_head(torch.cat([own_pooled, global_pooled], dim=1)).view(-1)

        diag_keys = ("gate", "gate_entropy", "mechanism_energy", "stream_disagreement")
        diag = torch.stack([trunk_out[k].detach() for k in diag_keys], dim=1)
        # Per-type aggregate activation magnitude as a "how loud is this type"
        # diagnostic; useful for inspecting which move type dominates the head.
        per_type_norm = torch.einsum(
            "tij,btjd->btid",
            masks,
            torch.stack(
                [
                    self.shared_projection(x_features)
                    if self.ablation == "shared_kernel"
                    else (
                        self.type_scalars[slot].view(1, 1, 1) * x_features
                        if self.ablation == "scalar_per_type"
                        else self.type_projections[slot](x_features)
                    )
                    for slot in range(NUM_MKO_TYPES)
                ],
                dim=1,
            ),
        ).pow(2).mean(dim=(2, 3)).sqrt()  # (B, T)
        gate_input = torch.cat([diag, per_type_norm], dim=1)
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
        for slot, (name, _) in enumerate(MKO_MOVE_TYPES):
            out[f"mko_norm_{name}"] = per_type_norm[:, slot]
        out["mechanism_energy"] = trunk_out["mechanism_energy"] + per_type_norm.mean(
            dim=1
        ).detach()
        out["proposal_profile_strength"] = primitive_delta.detach().abs() * gate_entropy
        out["proposal_keyword_count"] = logits.new_full((batch,), float(NUM_MKO_TYPES))
        return out


def build_move_kernel_operator_from_config(config: dict[str, Any]) -> MoveKernelOperator:
    cfg = dict(config)
    return MoveKernelOperator(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim")),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        feature_dim=int(cfg.get("feature_dim", 24)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )


__all__ = (
    "ALLOWED_ABLATIONS",
    "MKO_MOVE_TYPES",
    "MoveKernelOperator",
    "NUM_MKO_TYPES",
    "build_move_kernel_operator_from_config",
)

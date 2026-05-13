"""Reversible Delta Kernel Memory (p019).

Source: ``ideas/research/primitives/external_13_reversible_delta_kernel_occlusion_transport.md``
(rank-1 proposal ``primitive_01_reversible_delta_kernel_memory``).

The primitive maintains a linear-attention-like *set* memory ``(M, z)``
with exact signed insert/delete updates on bounded-change inputs:

    M = sum_i s_i * phi(u_i) nu(u_i)^T,  z = sum_i s_i * phi(u_i)
    Y_q = (phi(q)^T M) / (phi(q)^T z + eps)

At training time the model sees one static board at a time, so the
"stateful event-update API" is documented but not exercised here. The
forward pass computes the same memory state the insert/delete events
would produce, by aggregating over the active piece set on the current
board. Each occupied square emits one piece token; the kernel memory is
queried by pooled i193 trunk features and the resulting head value is
gated and added to the base logit.

The model wraps the bespoke i193 ``ExchangeThenKingDualStreamNetwork``
trunk. The primitive head is the new module. CRTK metadata, source
labels, verification flags, and engine evaluations are not consumed.

Deferred internal proposals from the same packet:

- ``primitive_03_incremental_pair_accumulator``: pair-cache update (sister
  to ``event_delta_bilinear_accumulator`` p022).
- ``primitive_04_alternating_soft_exchange_scan``: SEE-style alternating
  scan (not in this batch).
- ``primitive_05_signed_chess_orbit_norm``: orbit-tied normalisation
  (not in this batch).
- ``primitive_02_occlusion_scanned_move_transport``: occlusion ray
  primitive (this batch implements two ray formulations as p020 and
  p021).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.primitives.trunk_features import trunk_joint_features
from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    DualStreamFeatureBuilder,
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


SQUARES = 64
PIECE_PLANE_COUNT = 12
STM_CHANNEL = 12

ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    "shuffle_tokens",
    "zero_memory",
    "uniform_query",
    "zero_delta",
    "trunk_only",
)


def _phi(x: torch.Tensor) -> torch.Tensor:
    """Positive feature map (elu+1) used by linear/kernel attention."""
    return F.elu(x) + 1.0


def _piece_token_features(board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract per-square (piece-type, color) token labels and occupancy.

    Returns:
        piece_type_idx ``(B, 64)`` long in [0, 6]:
            0..5 = piece type for occupied squares, 6 = empty.
        side_idx ``(B, 64)`` long in [0, 2]:
            0 = white, 1 = black, 2 = empty.
        occupied ``(B, 64)`` float in {0, 1}.
    """
    piece_planes = board[:, :PIECE_PLANE_COUNT].clamp(0.0, 1.0).flatten(2)  # (B, 12, 64)
    occupied = piece_planes.sum(dim=1).clamp(0.0, 1.0)  # (B, 64)
    # Piece type 0..5 within colour
    type_logits_white = piece_planes[:, :6]  # (B, 6, 64)
    type_logits_black = piece_planes[:, 6:12]  # (B, 6, 64)
    type_logits = type_logits_white + type_logits_black  # (B, 6, 64)
    pad = (1.0 - occupied).unsqueeze(1)  # (B, 1, 64)
    type_full = torch.cat([type_logits, pad], dim=1)  # (B, 7, 64)
    piece_type_idx = type_full.argmax(dim=1)  # (B, 64), 6 for empty

    is_white = type_logits_white.sum(dim=1) > 0.5
    is_black = type_logits_black.sum(dim=1) > 0.5
    side_idx = torch.where(is_white, torch.zeros_like(piece_type_idx),
                           torch.where(is_black, torch.ones_like(piece_type_idx),
                                       torch.full_like(piece_type_idx, 2)))
    return piece_type_idx, side_idx, occupied


class ReversibleDeltaKernelMemory(nn.Module):
    """Reversible Delta Kernel Memory primitive head (p019).

    Forward pass on a single static board:

    1. Identify active pieces on the board via piece planes.
    2. Embed each (piece_type, side, square) tuple to a token ``u_i``.
    3. Compute the kernel memory ``M, z`` by summing over active tokens.
    4. Form ``Q`` queries from the pooled i193 trunk joint feature, query
       the memory, and project to a per-sample scalar.
    5. Sigmoid-gate the result on the trunk joint feature and add it to
       the i193 base logit.
    """

    ALLOWED_ABLATIONS = ALLOWED_ABLATIONS

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
        token_dim: int = 32,
        memory_heads: int = 16,
        memory_value_dim: int = 16,
        num_queries: int = 4,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError("ReversibleDeltaKernelMemory supports the puzzle_binary one-logit contract")
        if int(input_channels) != 18:
            raise ValueError("ReversibleDeltaKernelMemory requires the simple_18 board tensor")
        if str(ablation) not in self.ALLOWED_ABLATIONS:
            raise ValueError(f"Unknown ablation={ablation!r}; expected one of {sorted(self.ALLOWED_ABLATIONS)}")

        self.num_classes = 1
        self.ablation = str(ablation)
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.token_dim = int(token_dim)
        self.memory_heads = int(memory_heads)
        self.memory_value_dim = int(memory_value_dim)
        self.num_queries = int(num_queries)

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

        self.feature_dim = (
            2 * self.trunk.exchange_encoder.output_dim
            + DualStreamFeatureBuilder.SUMMARY_DIM
        )

        # 7-way piece type embedding (6 piece types + empty/pad token)
        self.piece_type_embed = nn.Embedding(7, self.token_dim, padding_idx=6)
        # 3-way side embedding (white, black, empty/pad)
        self.side_embed = nn.Embedding(3, self.token_dim, padding_idx=2)
        # Per-square (file/rank) embedding
        self.square_embed = nn.Embedding(SQUARES, self.token_dim)

        # phi/nu projections take a token and produce h-dim key and v-dim value.
        self.phi_proj = nn.Linear(self.token_dim, self.memory_heads)
        self.nu_proj = nn.Linear(self.token_dim, self.memory_value_dim)
        # Query projection from trunk joint feature to num_queries x memory_heads.
        self.query_proj = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.num_queries * self.memory_heads),
        )
        # Per-query value-pooling -> scalar.
        head_dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        readout_dim = self.num_queries * self.memory_value_dim
        self.value_readout = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, int(head_hidden_dim)),
            nn.GELU(),
            head_dropout_module,
            nn.Linear(int(head_hidden_dim), 1),
        )
        # Gate uses the trunk joint feature alone.
        self.gate_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, max(16, int(head_hidden_dim) // 2)),
            nn.GELU(),
            nn.Linear(max(16, int(head_hidden_dim) // 2), 1),
        )
        with torch.no_grad():
            final_layer = self.gate_head[-1]
            if isinstance(final_layer, nn.Linear):
                final_layer.bias.fill_(float(gate_init))

        # Cached per-square indices used by the embedding lookups.
        self.register_buffer(
            "square_index",
            torch.arange(SQUARES, dtype=torch.long),
            persistent=False,
        )

    def _build_piece_tokens(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-square piece tokens and occupancy mask."""
        piece_type_idx, side_idx, occupied = _piece_token_features(board)
        # When a square is empty we still embed but the mask zeroes the token.
        square_idx = self.square_index.to(device=board.device)
        type_emb = self.piece_type_embed(piece_type_idx)  # (B, 64, D)
        side_emb = self.side_embed(side_idx)              # (B, 64, D)
        sq_emb = self.square_embed(square_idx).unsqueeze(0).expand(board.shape[0], -1, -1)
        tokens = type_emb + side_emb + sq_emb
        return tokens, occupied

    def _maybe_shuffle_tokens(self, tokens: torch.Tensor, occupied: torch.Tensor) -> torch.Tensor:
        if self.ablation != "shuffle_tokens":
            return tokens
        batch = tokens.shape[0]
        if batch <= 1:
            return tokens
        perm = torch.randperm(batch, device=tokens.device)
        return tokens[perm]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]
        eps = 1.0e-6

        trunk_out = self.trunk(board)
        base_logit = trunk_out["logits"].view(-1)
        joint, _, _ = trunk_joint_features(self.trunk, board)

        tokens, occupied = self._build_piece_tokens(board)
        tokens = self._maybe_shuffle_tokens(tokens, occupied)
        # mask out tokens for empty squares so they do not contribute to memory.
        mask = occupied.to(dtype=tokens.dtype).unsqueeze(-1)  # (B, 64, 1)
        tokens = tokens * mask

        phi_tokens = _phi(self.phi_proj(tokens))   # (B, 64, h)
        nu_tokens = self.nu_proj(tokens)            # (B, 64, v)
        phi_tokens = phi_tokens * mask
        nu_tokens = nu_tokens * mask

        # Memory M (B, h, v) = sum_i phi_i nu_i^T;  z (B, h) = sum_i phi_i
        memory = torch.einsum("bnh,bnv->bhv", phi_tokens, nu_tokens)
        z = phi_tokens.sum(dim=1)  # (B, h)

        if self.ablation == "zero_memory":
            memory = torch.zeros_like(memory)
            z = torch.zeros_like(z)

        # Build queries from the trunk joint feature
        if self.ablation == "uniform_query":
            queries = phi_tokens.new_ones(batch, self.num_queries, self.memory_heads) / self.memory_heads
        else:
            q_raw = self.query_proj(joint).view(batch, self.num_queries, self.memory_heads)
            queries = _phi(q_raw)

        # y_q = (phi(q)^T M) / (phi(q)^T z + eps)
        num = torch.einsum("bqh,bhv->bqv", queries, memory)
        den = torch.einsum("bqh,bh->bq", queries, z).clamp_min(eps).unsqueeze(-1)
        per_query = num / den  # (B, num_queries, v)

        readout = per_query.flatten(1)  # (B, num_queries * v)
        delta_raw = self.value_readout(readout).view(-1)

        gate_logit = self.gate_head(joint).view(-1)
        gate = torch.sigmoid(gate_logit)
        if self.ablation in {"zero_delta", "trunk_only"}:
            primitive_delta = torch.zeros_like(base_logit)
        else:
            primitive_delta = gate * delta_raw

        logits = base_logit + primitive_delta

        gate_clamped = gate.clamp(eps, 1.0 - eps)
        gate_entropy = -(
            gate_clamped * gate_clamped.log()
            + (1.0 - gate_clamped) * (1.0 - gate_clamped).log()
        )
        active_count = occupied.sum(dim=1)
        memory_norm = memory.pow(2).flatten(1).mean(dim=1).sqrt()
        z_norm = z.pow(2).mean(dim=1).sqrt()

        diagnostics: dict[str, torch.Tensor] = {
            "logits": logits,
            "base_logit": base_logit,
            "primitive_delta": primitive_delta,
            "primitive_delta_raw": delta_raw,
            "primitive_gate": gate,
            "primitive_gate_logit": gate_logit,
            "primitive_gate_entropy": gate_entropy,
            "rdkm_active_count": active_count,
            "rdkm_memory_norm": memory_norm,
            "rdkm_z_norm": z_norm,
            "mechanism_energy": trunk_out["mechanism_energy"] + memory_norm.detach(),
            "proposal_profile_strength": (primitive_delta.abs() * gate_entropy).clamp(0.0, 20.0),
            "proposal_keyword_count": logits.new_full((batch,), float(self.memory_heads * self.memory_value_dim)),
        }
        for key, value in trunk_out.items():
            if key == "logits":
                continue
            diag_key = key if key.startswith("base_") or key in {"gate", "gate_logit"} else f"trunk_{key}"
            diagnostics.setdefault(diag_key, value)
        return diagnostics


def build_reversible_delta_kernel_memory_from_config(
    config: dict[str, Any],
) -> ReversibleDeltaKernelMemory:
    cfg = dict(config)
    return ReversibleDeltaKernelMemory(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim", None)),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        token_dim=int(cfg.get("token_dim", 32)),
        memory_heads=int(cfg.get("memory_heads", 16)),
        memory_value_dim=int(cfg.get("memory_value_dim", 16)),
        num_queries=int(cfg.get("num_queries", 4)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )

"""Forcing-Certificate Transformer for idea i177.

Faithful implementation of the markdown thesis under
``ideas/registry/i177_forcing_certificate_transformer/``. The model classifies a
position by trying to assemble a small, slot-structured tactical
certificate

    attacker / forcing piece
    target
    defender / escape resource
    blocker / pin / overload relation
    tempo side

instead of pooling a single global board embedding. A small set of
``num_slots`` learned certificate-slot queries cross-attend to square
tokens with chess relation biases (same line, knight reach, king-zone
adjacency, pawn attack), each slot emits a slot score, and the puzzle
logit is the log-sum-exp of slot scores plus a global residual logit.

The ablations exposed via ``ablation`` mirror the failure modes the
packet warns about:

* ``"none"`` -- main model.
* ``"no_relation_bias"`` -- drop the chess relation prior, leaving the
  slots to discover structure from positional features only.
* ``"no_global_residual"`` -- read out from the slots only; tests
  whether the certificate is doing the work or piggy-backing on a
  global head.
* ``"uniform_slot_attention"`` -- replace cross-attention with uniform
  attention; tests whether the slots actually pick out a few squares.
* ``"single_slot"`` -- collapse to ``num_slots = 1``; tests whether
  competition between certificate slots matters.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import (
    BoardConvStem,
    BoardTensorSpec,
    require_board_tensor,
)


_VALID_ABLATIONS = {
    "none",
    "no_relation_bias",
    "no_global_residual",
    "uniform_slot_attention",
    "single_slot",
}


_RELATION_NAMES = (
    "same_rank",
    "same_file",
    "same_diagonal_pos",
    "same_diagonal_neg",
    "knight_reach",
    "king_adjacent",
    "pawn_attack_white",
    "pawn_attack_black",
)


def _chess_relation_matrices() -> torch.Tensor:
    """Return ``(R, 64, 64)`` binary chess relation tensors.

    Each plane is the asymmetric ``i -> j`` adjacency for one of the
    fixed chess relations the packet calls for.
    """
    idx = torch.arange(64)
    rank = idx // 8
    file = idx % 8
    src_rank = rank.view(64, 1)
    src_file = file.view(64, 1)
    tgt_rank = rank.view(1, 64)
    tgt_file = file.view(1, 64)
    dr = tgt_rank - src_rank
    df = tgt_file - src_file
    eye = torch.eye(64, dtype=torch.bool)
    same_rank = (dr == 0) & (~eye)
    same_file = (df == 0) & (~eye)
    same_diag_pos = (dr == df) & (~eye)
    same_diag_neg = (dr == -df) & (~eye)
    knight = ((dr.abs() == 1) & (df.abs() == 2)) | ((dr.abs() == 2) & (df.abs() == 1))
    king = (dr.abs() <= 1) & (df.abs() <= 1) & (~eye)
    pawn_white = (dr == 1) & (df.abs() == 1)
    pawn_black = (dr == -1) & (df.abs() == 1)
    relations = torch.stack(
        [
            same_rank,
            same_file,
            same_diag_pos,
            same_diag_neg,
            knight,
            king,
            pawn_white,
            pawn_black,
        ],
        dim=0,
    ).to(torch.float32)
    return relations


def _square_coords() -> torch.Tensor:
    rank = torch.arange(8, dtype=torch.float32).view(8, 1).expand(8, 8)
    file = torch.arange(8, dtype=torch.float32).view(1, 8).expand(8, 8)
    centered_rank = (rank - 3.5) / 3.5
    centered_file = (file - 3.5) / 3.5
    edge = torch.minimum(
        torch.minimum(rank, 7.0 - rank),
        torch.minimum(file, 7.0 - file),
    ) / 3.5
    parity = ((rank + file).remainder(2.0) * 2.0) - 1.0
    coords = torch.stack(
        [rank / 7.0, file / 7.0, centered_rank, centered_file, edge, parity],
        dim=-1,
    ).reshape(64, 6)
    return coords


class CertificateSlotAttention(nn.Module):
    """Multi-head cross-attention from ``K`` slots to 64 square tokens.

    Adds a per-slot, per-token bias produced by mixing fixed chess
    relation matrices with learnable slot anchors:

        anchor_k     = softmax(anchor_logits_k)        # (K, 64)
        bias_{k, j}  = sum_r mix_{k,r} * sum_i anchor_{k,i} * R_r[i, j]

    The bias is broadcast across heads and added to the attention
    logits. Setting ``use_relation_bias=False`` disables the bias term
    (the ``no_relation_bias`` ablation). Setting ``uniform_attention``
    replaces softmax attention with uniform weights but still uses the
    learned value projection (the ``uniform_slot_attention`` ablation).
    """

    def __init__(
        self,
        token_dim: int,
        num_slots: int,
        num_heads: int,
        dropout: float,
        relation_count: int,
        use_relation_bias: bool,
        uniform_attention: bool,
    ) -> None:
        super().__init__()
        if token_dim < 1 or num_slots < 1 or num_heads < 1:
            raise ValueError("token_dim, num_slots, num_heads must be positive")
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        self.token_dim = int(token_dim)
        self.num_slots = int(num_slots)
        self.num_heads = int(num_heads)
        self.head_dim = self.token_dim // self.num_heads
        self.use_relation_bias = bool(use_relation_bias)
        self.uniform_attention = bool(uniform_attention)
        self.query_norm = nn.LayerNorm(self.token_dim)
        self.key_norm = nn.LayerNorm(self.token_dim)
        self.q_proj = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.k_proj = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.v_proj = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.out_proj = nn.Linear(self.token_dim, self.token_dim, bias=True)
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.relation_count = int(relation_count)
        if self.use_relation_bias:
            self.anchor_logits = nn.Parameter(torch.zeros(self.num_slots, 64))
            self.relation_mix = nn.Parameter(torch.zeros(self.num_slots, self.relation_count))
            nn.init.normal_(self.anchor_logits, std=0.5)
            nn.init.normal_(self.relation_mix, std=0.1)
        else:
            self.register_parameter("anchor_logits", None)
            self.register_parameter("relation_mix", None)

    def relation_bias(self, relations: torch.Tensor) -> torch.Tensor:
        """Compute ``(K, 64)`` slot-to-token relation bias."""
        if not self.use_relation_bias:
            return torch.zeros(self.num_slots, 64, device=relations.device, dtype=relations.dtype)
        anchor = F.softmax(self.anchor_logits, dim=-1)
        per_slot = torch.einsum("rij,ki->krj", relations, anchor)
        return torch.einsum("kr,krj->kj", self.relation_mix, per_slot)

    def forward(
        self,
        slots: torch.Tensor,
        tokens: torch.Tensor,
        relations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = tokens.shape[0]
        q = self.q_proj(self.query_norm(slots))
        k = self.k_proj(self.key_norm(tokens))
        v = self.v_proj(tokens)
        q = q.view(batch_size, self.num_slots, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, 64, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, 64, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.einsum("bhqd,bhnd->bhqn", q, k) / math.sqrt(float(self.head_dim))
        bias = self.relation_bias(relations)
        if self.use_relation_bias:
            scores = scores + bias.view(1, 1, self.num_slots, 64)
        attention = F.softmax(scores, dim=-1)
        if self.uniform_attention:
            attention = torch.full_like(attention, 1.0 / 64.0)
        attention = self.attn_dropout(attention)
        attended = torch.einsum("bhqn,bhnd->bhqd", attention, v)
        attended = attended.transpose(1, 2).reshape(batch_size, self.num_slots, self.token_dim)
        attended = self.out_proj(attended)
        return attended, attention.mean(dim=1), bias


class ForcingCertificateTransformer(nn.Module):
    """Slot-structured tactical-certificate classifier.

    Forward returns a dict with at least:

      - ``logits``: ``(B,)`` puzzle logit fed to the BCE-with-logits trainer
        (``(B, num_classes)`` if ``num_classes > 1``).
      - ``prob``: ``sigmoid(logits)`` when ``num_classes == 1``.
      - ``slot_scores``: ``(B, num_slots)`` per-slot scalar scores.
      - ``slot_logsumexp``: ``(B,)`` log-sum-exp aggregation of the slot
        scores -- the certificate side of the readout.
      - ``global_residual_logit``: ``(B,)`` residual logit from a mean-
        pooled MLP head over the square tokens.
      - ``slot_attention``: ``(B, num_slots, 64)`` slot attention map.
      - ``slot_attention_entropy``: ``(B, num_slots)`` per-slot entropy
        of the attention distribution (normalised by ``log 64``).
      - ``slot_attention_max``, ``slot_attention_margin``: ``(B, num_slots)``
        diagnostics of attention concentration.
      - ``slot_diversity``: ``(B,)`` mean pairwise total-variation
        distance between slot attention maps -- the slot diversity
        signal the packet calls out.
      - ``slot_features``: ``(B, num_slots, token_dim)`` final slot
        embeddings.
      - ``token_features``: ``(B, 64, token_dim)`` square-token sequence.
      - ``trunk_features``: ``(B, channels, 8, 8)`` CNN stem output.
      - ``relation_bias``: ``(num_slots, 64)`` (broadcast to ``(B,...)``)
        relation bias the slots saw.
      - ablation flag tensors ``ablation_active``,
        ``uses_relation_bias``, ``uses_global_residual``,
        ``uses_slot_attention``, ``num_slots_levels``, ``slot_iters_levels``.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        depth: int = 2,
        token_dim: int = 96,
        num_slots: int = 6,
        num_heads: int = 4,
        slot_iters: int = 2,
        head_hidden: int | None = None,
        residual_hidden: int | None = None,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if depth < 1 or channels < 1 or num_classes < 1:
            raise ValueError("depth, channels, num_classes must be >= 1")
        if token_dim < 1 or num_slots < 1 or num_heads < 1 or slot_iters < 1:
            raise ValueError("token_dim, num_slots, num_heads, slot_iters must be >= 1")
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        if ablation not in _VALID_ABLATIONS:
            raise ValueError(
                f"ablation must be one of {sorted(_VALID_ABLATIONS)}, got {ablation!r}"
            )

        if ablation == "single_slot":
            num_slots = 1
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.depth = int(depth)
        self.token_dim = int(token_dim)
        self.num_slots = int(num_slots)
        self.num_heads = int(num_heads)
        self.slot_iters = int(slot_iters)
        self.dropout = float(dropout)
        self.use_batchnorm = bool(use_batchnorm)
        self.ablation = str(ablation)

        self.uses_relation_bias = self.ablation != "no_relation_bias"
        self.uses_global_residual = self.ablation != "no_global_residual"
        self.uses_slot_attention = self.ablation != "uniform_slot_attention"

        head_hidden_dim = int(head_hidden) if head_hidden is not None else self.token_dim
        residual_hidden_dim = (
            int(residual_hidden) if residual_hidden is not None else self.token_dim
        )

        self.trunk = BoardConvStem(
            input_channels=self.input_channels,
            channels=self.channels,
            depth=self.depth,
            use_batchnorm=self.use_batchnorm,
        )
        # 64 square tokens of width token_dim. The trunk emits per-square
        # feature vectors of width ``channels``; we project to ``token_dim``
        # and add a learnable per-square positional embedding plus the
        # geometric coords used elsewhere in the codebase.
        self.token_proj = nn.Sequential(
            nn.LayerNorm(self.channels + 6),
            nn.Linear(self.channels + 6, self.token_dim),
        )
        self.position_embed = nn.Parameter(torch.zeros(64, self.token_dim))
        nn.init.normal_(self.position_embed, std=0.02)
        self.register_buffer("coords", _square_coords(), persistent=False)
        self.register_buffer(
            "relations", _chess_relation_matrices(), persistent=False
        )
        self.relation_count = int(self.relations.shape[0])
        # K certificate slot queries, learnable.
        slot_init = torch.randn(self.num_slots, self.token_dim) / math.sqrt(float(self.token_dim))
        self.slot_queries = nn.Parameter(slot_init)

        self.slot_layers = nn.ModuleList(
            [
                CertificateSlotAttention(
                    token_dim=self.token_dim,
                    num_slots=self.num_slots,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    relation_count=self.relation_count,
                    use_relation_bias=self.uses_relation_bias,
                    uniform_attention=not self.uses_slot_attention,
                )
                for _ in range(self.slot_iters)
            ]
        )
        # Per-iteration slot MLP refines the slot embedding (a small
        # post-attention block with GELU and residual). The classifier
        # head reads slot scores from the final slots.
        self.slot_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.token_dim),
                    nn.Linear(self.token_dim, head_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
                    nn.Linear(head_hidden_dim, self.token_dim),
                )
                for _ in range(self.slot_iters)
            ]
        )
        self.slot_score_head = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(head_hidden_dim, 1),
        )
        self.global_head = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, residual_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(residual_hidden_dim, 1),
        )
        # If a multi-class output is requested we re-use the same scalar
        # certificate logit and place it in the last column of a zero
        # logits tensor, mirroring the convention used by other puzzle
        # binary bespoke models in this repo.

    def _square_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feats = self.trunk(x)  # (B, channels, 8, 8)
        per_square = feats.flatten(2).transpose(1, 2)  # (B, 64, channels)
        coords = self.coords.to(dtype=per_square.dtype, device=per_square.device)
        coords = coords.unsqueeze(0).expand(per_square.shape[0], -1, -1)
        tokens = self.token_proj(torch.cat([per_square, coords], dim=-1))
        tokens = tokens + self.position_embed.unsqueeze(0)
        return feats, tokens

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats, tokens = self._square_tokens(x)
        batch_size = tokens.shape[0]
        slots = self.slot_queries.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        relations = self.relations.to(dtype=tokens.dtype, device=tokens.device)
        last_attention: torch.Tensor | None = None
        last_bias: torch.Tensor | None = None
        for layer, mlp in zip(self.slot_layers, self.slot_mlps):
            attended, attention, bias = layer(slots, tokens, relations)
            slots = slots + attended
            slots = slots + mlp(slots)
            last_attention = attention
            last_bias = bias
        assert last_attention is not None
        assert last_bias is not None

        slot_scores = self.slot_score_head(slots).squeeze(-1)  # (B, K)
        slot_logsumexp = torch.logsumexp(slot_scores, dim=-1)  # (B,)

        token_pool = tokens.mean(dim=1)
        global_residual_logit = self.global_head(token_pool).squeeze(-1)  # (B,)

        if self.uses_global_residual:
            scalar_logit = slot_logsumexp + global_residual_logit
        else:
            scalar_logit = slot_logsumexp

        if self.num_classes == 1:
            logits = scalar_logit
        else:
            logits = torch.zeros(
                batch_size, self.num_classes, device=tokens.device, dtype=tokens.dtype
            )
            logits[:, -1] = scalar_logit

        # Diagnostics. ``slot_attention`` is (B, K, 64).
        eps = 1.0e-8
        entropy = -(last_attention * last_attention.clamp_min(eps).log()).sum(dim=-1)
        entropy = entropy / math.log(64.0)
        sorted_attention = last_attention.sort(dim=-1, descending=True).values
        slot_attention_max = sorted_attention[..., 0]
        slot_attention_margin = sorted_attention[..., 0] - sorted_attention[..., 1]

        if self.num_slots > 1:
            # Mean pairwise total-variation distance between slot
            # attention maps. Higher = more diverse certificates.
            attention_a = last_attention.unsqueeze(2)  # (B, K, 1, 64)
            attention_b = last_attention.unsqueeze(1)  # (B, 1, K, 64)
            tv = 0.5 * (attention_a - attention_b).abs().sum(dim=-1)  # (B, K, K)
            mask = 1.0 - torch.eye(self.num_slots, device=tv.device, dtype=tv.dtype)
            denom = float(self.num_slots * (self.num_slots - 1))
            slot_diversity = (tv * mask).sum(dim=(-1, -2)) / denom
        else:
            slot_diversity = torch.zeros(batch_size, device=tokens.device, dtype=tokens.dtype)

        with torch.no_grad():
            ones = torch.ones(batch_size, device=tokens.device, dtype=tokens.dtype)
            ablation_active = ones * (0.0 if self.ablation == "none" else 1.0)
            uses_relation_bias = ones * (1.0 if self.uses_relation_bias else 0.0)
            uses_global_residual = ones * (1.0 if self.uses_global_residual else 0.0)
            uses_slot_attention = ones * (1.0 if self.uses_slot_attention else 0.0)
            num_slots_levels = ones * float(self.num_slots)
            slot_iters_levels = ones * float(self.slot_iters)

        relation_bias_b = last_bias.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "slot_scores": slot_scores,
            "slot_logsumexp": slot_logsumexp,
            "global_residual_logit": global_residual_logit,
            "slot_attention": last_attention,
            "slot_attention_entropy": entropy,
            "slot_attention_entropy_mean": entropy.mean(dim=-1),
            "slot_attention_max": slot_attention_max,
            "slot_attention_margin": slot_attention_margin,
            "slot_diversity": slot_diversity,
            "slot_features": slots,
            "token_features": tokens,
            "trunk_features": feats,
            "relation_bias": relation_bias_b,
            "ablation_active": ablation_active,
            "uses_relation_bias": uses_relation_bias,
            "uses_global_residual": uses_global_residual,
            "uses_slot_attention": uses_slot_attention,
            "num_slots_levels": num_slots_levels,
            "slot_iters_levels": slot_iters_levels,
        }
        if self.num_classes == 1:
            output["prob"] = torch.sigmoid(logits)
        return output


def build_forcing_certificate_transformer_from_config(
    config: dict[str, Any],
) -> ForcingCertificateTransformer:
    cfg = dict(config)
    cfg.pop("name", None)
    cfg.pop("mechanism_family", None)
    cfg.pop("packet_profile", None)
    hidden_dim = cfg.pop("hidden_dim", None)
    head_hidden = cfg.pop("head_hidden", hidden_dim)
    residual_hidden = cfg.pop("residual_hidden", hidden_dim)
    token_dim = cfg.pop("token_dim", hidden_dim if hidden_dim is not None else 96)
    return ForcingCertificateTransformer(
        input_channels=int(cfg.pop("input_channels", 18)),
        num_classes=int(cfg.pop("num_classes", 1)),
        channels=int(cfg.pop("channels", 64)),
        depth=int(cfg.pop("depth", 2)),
        token_dim=int(token_dim),
        num_slots=int(cfg.pop("num_slots", 6)),
        num_heads=int(cfg.pop("num_heads", 4)),
        slot_iters=int(cfg.pop("slot_iters", 2)),
        head_hidden=int(head_hidden) if head_hidden is not None else None,
        residual_hidden=int(residual_hidden) if residual_hidden is not None else None,
        dropout=float(cfg.pop("dropout", 0.1)),
        use_batchnorm=bool(cfg.pop("use_batchnorm", True)),
        ablation=str(cfg.pop("ablation", "none")),
    )


__all__ = [
    "CertificateSlotAttention",
    "ForcingCertificateTransformer",
    "build_forcing_certificate_transformer_from_config",
]

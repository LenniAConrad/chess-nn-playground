"""Sparse Expert Board Router (idea i123).

This bespoke model materialises the sparse mixture-of-experts thesis from
`ideas/registry/i123_sparse_expert_board_router/math_thesis.md`. A cheap routing
summary (material counts, king locations, side-to-move, coarse occupancy
quadrants and a small CNN stem pool) drives a router that produces logits
over ``E`` heterogeneous expert encoders. Top-``k`` experts are selected
per example with a softmax-renormalised gating, their hidden vectors are
fused, and a binary puzzle logit is produced.

Following the markdown architecture sketch:

- Six experts cover distinct inductive biases: ``local_cnn``, ``dilated_cnn``,
  ``token_mixer``, ``rank_file_mixer``, ``morphology_lite`` and
  ``compact_mlp_mixer``.
- The router consumes (a) a cheap deterministic summary derived from the
  ``simple_18`` planes and (b) a small CNN stem pool, mirroring the
  ``material_only`` and spatial-routing ablation framings in the packet.
- Selection is sparse: only the top ``k`` expert outputs contribute to the
  fused representation; non-selected experts get zero gate weight, but the
  evaluation cost is still ``E`` because experts are run in parallel for the
  small board budget. The model only fuses the top-``k`` outputs, matching
  the markdown's "selected_weights" tensor contract.
- Diagnostics include load-balance statistics, router entropy, top-1 mass,
  top-2 mass, expert usage histogram, and pairwise expert-logit
  disagreement, plus a ``load_balance_loss`` and ``router_entropy_loss``
  term so the trainer can add them as auxiliary regularisers.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.data.board_features import SIMPLE_18
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


EXPERT_NAMES: tuple[str, ...] = (
    "local_cnn",
    "dilated_cnn",
    "token_mixer",
    "rank_file_mixer",
    "morphology_lite",
    "compact_mlp_mixer",
)
NUM_EXPERTS = len(EXPERT_NAMES)
PIECE_VALUES = (1.0, 3.0, 3.0, 5.0, 9.0, 0.0)


class _LocalCNNExpert(nn.Module):
    def __init__(self, in_channels: int, channels: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.body(x)
        pooled = torch.cat([feats.mean(dim=(2, 3)), feats.amax(dim=(2, 3))], dim=-1)
        return self.head(pooled)


class _DilatedCNNExpert(nn.Module):
    def __init__(self, in_channels: int, channels: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, dilation=1),
            nn.GELU(),
        )
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
        )
        self.path3 = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=3, dilation=3),
            nn.GELU(),
        )
        self.merge = nn.Conv2d(3 * channels, channels, kernel_size=1)
        self.head = nn.Sequential(
            nn.Linear(2 * channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        merged = self.merge(torch.cat([self.path1(x), self.path2(x), self.path3(x)], dim=1))
        merged = F.gelu(merged)
        pooled = torch.cat([merged.mean(dim=(2, 3)), merged.amax(dim=(2, 3))], dim=-1)
        return self.head(pooled)


class _TokenMixerExpert(nn.Module):
    """Per-square token mixer: a small attention-free token MLP on 64 tokens."""

    def __init__(self, in_channels: int, token_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Linear(in_channels, token_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        self.channel_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.norm1 = nn.LayerNorm(token_dim)
        self.norm2 = nn.LayerNorm(token_dim)
        self.head = nn.Sequential(
            nn.Linear(2 * token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        tokens = x.flatten(2).transpose(1, 2)  # (B, 64, in_channels)
        tokens = self.proj(tokens)  # (B, 64, token_dim)
        residual = tokens
        tokens = self.norm1(tokens)
        tokens = tokens + self.token_mlp(tokens.transpose(1, 2)).transpose(1, 2)
        tokens = tokens + self.channel_mlp(self.norm2(tokens))
        tokens = tokens + residual
        pooled = torch.cat([tokens.mean(dim=1), tokens.amax(dim=1)], dim=-1)
        return self.head(pooled)


class _RankFileMixerExpert(nn.Module):
    """Mixes rank and file aggregates separately."""

    def __init__(self, in_channels: int, channels: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, channels, kernel_size=1)
        self.rank_mix = nn.Sequential(
            nn.Linear(channels * 8, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels * 8),
        )
        self.file_mix = nn.Sequential(
            nn.Linear(channels * 8, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels * 8),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * channels * 8, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.proj(x)  # (B, C, 8, 8)
        rank_summary = feats.mean(dim=3).flatten(1)  # (B, C*8)
        file_summary = feats.mean(dim=2).flatten(1)
        rank_mixed = self.rank_mix(rank_summary)
        file_mixed = self.file_mix(file_summary)
        return self.head(torch.cat([rank_mixed, file_mixed], dim=-1))


class _MorphologyLiteExpert(nn.Module):
    """Approximates dilation/erosion via max- and min-pool stencils."""

    def __init__(self, in_channels: int, channels: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, channels, kernel_size=1)
        self.smooth = nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1)
        self.head = nn.Sequential(
            nn.Linear(2 * channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.proj(x)
        dilation = F.max_pool2d(feats, kernel_size=3, stride=1, padding=1)
        erosion = -F.max_pool2d(-feats, kernel_size=3, stride=1, padding=1)
        gradient = dilation - erosion
        merged = self.smooth(torch.cat([dilation, gradient], dim=1))
        merged = F.gelu(merged)
        pooled = torch.cat([merged.mean(dim=(2, 3)), merged.amax(dim=(2, 3))], dim=-1)
        return self.head(pooled)


class _CompactMLPExpert(nn.Module):
    """Pixel-wise MLP plus global pooling — the simplest expert."""

    def __init__(self, in_channels: int, channels: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.body(x)
        pooled = torch.cat([feats.mean(dim=(2, 3)), feats.amax(dim=(2, 3))], dim=-1)
        return self.head(pooled)


class _RouterStem(nn.Module):
    """Cheap CNN stem feeding the router. Kept small per the packet budget."""

    def __init__(self, in_channels: int, channels: int, use_batchnorm: bool) -> None:
        super().__init__()
        norm = nn.BatchNorm2d if use_batchnorm else nn.Identity
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm),
            norm(channels) if use_batchnorm else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm),
            norm(channels) if use_batchnorm else nn.Identity(),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class SparseExpertBoardRouter(nn.Module):
    """Sparse mixture of small board experts for the puzzle_binary contract."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        num_experts: int = NUM_EXPERTS,
        top_k: int = 2,
        router_temperature: float = 1.0,
        load_balance_weight: float = 0.01,
        router_entropy_weight: float = 0.001,
        encoding_adapter: str = SIMPLE_18,
    ) -> None:
        super().__init__()
        if input_channels != 18 or encoding_adapter != SIMPLE_18:
            raise ValueError("SparseExpertBoardRouter supports simple_18 with 18 input channels")
        if num_classes != 1:
            raise ValueError("SparseExpertBoardRouter supports the puzzle_binary one-logit contract")
        if num_experts != NUM_EXPERTS:
            raise ValueError(f"num_experts must be {NUM_EXPERTS} for this architecture")
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        if depth < 1:
            raise ValueError("depth must be >= 1 for the router stem")
        del depth  # depth is consumed by the stem below; kept for config compatibility

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.router_temperature = float(router_temperature)
        self.load_balance_weight = float(load_balance_weight)
        self.router_entropy_weight = float(router_entropy_weight)

        self.register_buffer("piece_values", torch.tensor(PIECE_VALUES), persistent=False)

        self.stem = _RouterStem(input_channels, channels, use_batchnorm=use_batchnorm)

        # Routing summary: 12 deterministic features below + small CNN stem pool
        # (mean + max -> 2 * channels).
        self._summary_dim = 12
        router_input_dim = self._summary_dim + 2 * channels
        self.router_summary_norm = nn.LayerNorm(router_input_dim)
        self.router = nn.Sequential(
            nn.Linear(router_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_experts),
        )

        expert_channels = max(channels // 2, 8)
        token_dim = max(channels // 2, 16)
        self.experts = nn.ModuleList(
            [
                _LocalCNNExpert(input_channels, expert_channels, hidden_dim, dropout),
                _DilatedCNNExpert(input_channels, expert_channels, hidden_dim, dropout),
                _TokenMixerExpert(input_channels, token_dim, hidden_dim, dropout),
                _RankFileMixerExpert(input_channels, max(expert_channels // 2, 4), hidden_dim, dropout),
                _MorphologyLiteExpert(input_channels, expert_channels, hidden_dim, dropout),
                _CompactMLPExpert(input_channels, expert_channels, hidden_dim, dropout),
            ]
        )

        # Per-expert binary logit head plus a fused classifier head. The
        # mixture logit is a router-weighted combination of expert logits,
        # while the diagnostic head reads the fused representation directly.
        self.expert_logit_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(num_experts)]
        )
        self.fused_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )
        self.fusion_gate = nn.Parameter(torch.zeros(1))  # blend mixture logit with fused-head logit

    # ------------------------------------------------------------------
    # Routing summary features
    # ------------------------------------------------------------------
    def _routing_summary(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        # Material counts (us / them).
        white = x[:, 0:6].clamp(0.0, 1.0)
        black = x[:, 6:12].clamp(0.0, 1.0)
        side = x[:, 12:13].clamp(0.0, 1.0).flatten(1).mean(dim=1, keepdim=True)
        white_material = (white * self.piece_values.view(1, -1, 1, 1)).sum(dim=(1, 2, 3), keepdim=False)
        black_material = (black * self.piece_values.view(1, -1, 1, 1)).sum(dim=(1, 2, 3), keepdim=False)

        # King locations (rank / file proxies).
        white_king = white[:, 5]
        black_king = black[:, 5]
        ranks = torch.arange(8, device=x.device, dtype=x.dtype).view(1, 8, 1) / 7.0
        files = torch.arange(8, device=x.device, dtype=x.dtype).view(1, 1, 8) / 7.0
        white_king_rank = (white_king * ranks).sum(dim=(1, 2)) / white_king.sum(dim=(1, 2)).clamp_min(1.0)
        white_king_file = (white_king * files).sum(dim=(1, 2)) / white_king.sum(dim=(1, 2)).clamp_min(1.0)
        black_king_rank = (black_king * ranks).sum(dim=(1, 2)) / black_king.sum(dim=(1, 2)).clamp_min(1.0)
        black_king_file = (black_king * files).sum(dim=(1, 2)) / black_king.sum(dim=(1, 2)).clamp_min(1.0)

        # Coarse occupancy quadrants.
        occ = (white + black).sum(dim=1)  # (B, 8, 8)
        q_tl = occ[:, :4, :4].mean(dim=(1, 2))
        q_tr = occ[:, :4, 4:].mean(dim=(1, 2))
        q_bl = occ[:, 4:, :4].mean(dim=(1, 2))
        q_br = occ[:, 4:, 4:].mean(dim=(1, 2))

        summary = torch.stack(
            [
                white_material / 39.0,
                black_material / 39.0,
                side.squeeze(-1),
                white_king_rank,
                white_king_file,
                black_king_rank,
                black_king_file,
                q_tl,
                q_tr,
                q_bl,
                q_br,
                (white_material + black_material) / 78.0,
            ],
            dim=1,
        )
        assert summary.shape == (batch, self._summary_dim)
        return summary

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        batch = x.shape[0]

        stem_feats = self.stem(x)
        stem_pool = torch.cat([stem_feats.mean(dim=(2, 3)), stem_feats.amax(dim=(2, 3))], dim=-1)
        deterministic = self._routing_summary(x)
        router_input = self.router_summary_norm(torch.cat([deterministic, stem_pool], dim=-1))
        router_logits = self.router(router_input) / max(self.router_temperature, 1.0e-6)
        router_probs = F.softmax(router_logits, dim=-1)

        # Sparse top-k gating: keep top-k logits, set others to -inf, renormalise.
        top_vals, top_idx = router_logits.topk(self.top_k, dim=-1)
        sparse_mask = router_logits.new_full(router_logits.shape, float("-inf"))
        sparse_mask.scatter_(1, top_idx, top_vals)
        gate = F.softmax(sparse_mask, dim=-1)  # zeros for non-selected experts

        # Run all experts for this small board budget; gating is sparse.
        expert_hidden = torch.stack([expert(x) for expert in self.experts], dim=1)  # (B, E, H)
        fused = (gate.unsqueeze(-1) * expert_hidden).sum(dim=1)  # (B, H)

        # Per-expert binary logits and the router-weighted mixture logit.
        per_expert_logits = torch.stack(
            [head(expert_hidden[:, i]).squeeze(-1) for i, head in enumerate(self.expert_logit_heads)],
            dim=1,
        )  # (B, E)
        mixture_logit = (gate * per_expert_logits).sum(dim=1)
        fused_logit = self.fused_classifier(fused).squeeze(-1)
        blend = torch.sigmoid(self.fusion_gate)
        logits = blend * mixture_logit + (1.0 - blend) * fused_logit

        # Diagnostics --------------------------------------------------
        log_probs = router_probs.clamp_min(1.0e-8).log()
        router_entropy = -(router_probs * log_probs).sum(dim=1)
        gate_safe = gate.clamp_min(1.0e-8)
        sparse_entropy = -(gate * gate_safe.log()).sum(dim=1)

        # Per-batch load balance: mean gate per expert vs uniform.
        mean_gate = gate.mean(dim=0)
        mean_router = router_probs.mean(dim=0)
        uniform = mean_gate.new_full(mean_gate.shape, 1.0 / self.num_experts)
        load_balance_loss = ((mean_gate - uniform) ** 2).sum() * float(self.num_experts)
        # Switch-style auxiliary loss using soft probs and selection counts.
        selected_one_hot = F.one_hot(top_idx, num_classes=self.num_experts).float()
        selection_fraction = selected_one_hot.sum(dim=(0, 1)) / float(batch * self.top_k)
        switch_aux_loss = float(self.num_experts) * (selection_fraction * mean_router).sum()

        load_balance_term = self.load_balance_weight * (load_balance_loss + switch_aux_loss)
        router_entropy_term = self.router_entropy_weight * (-router_entropy.mean())
        aux_loss = load_balance_term + router_entropy_term

        # Pairwise expert-logit disagreement summary.
        diffs = per_expert_logits.unsqueeze(2) - per_expert_logits.unsqueeze(1)
        pairwise_disagreement = diffs.abs().mean(dim=(1, 2))

        top1_mass = gate.amax(dim=1)
        top2_mass = gate.topk(min(2, self.num_experts), dim=1).values.sum(dim=1)
        dominant_expert = gate.argmax(dim=1).to(logits.dtype)
        selection_counts = selected_one_hot.sum(dim=(0, 1))  # per-expert select count

        return {
            "logits": logits,
            "mixture_logit": mixture_logit,
            "fused_logit": fused_logit,
            "router_logits": router_logits,
            "router_probs": router_probs,
            "router_gate": gate,
            "router_entropy": router_entropy,
            "sparse_gate_entropy": sparse_entropy,
            "load_balance_loss": load_balance_loss,
            "switch_aux_loss": switch_aux_loss,
            "router_entropy_loss": -router_entropy.mean(),
            "auxiliary_loss": aux_loss,
            "expert_logits": per_expert_logits,
            "pairwise_expert_disagreement": pairwise_disagreement,
            "top1_gate_mass": top1_mass,
            "top2_gate_mass": top2_mass,
            "dominant_expert": dominant_expert,
            "expert_selection_counts": selection_counts,
            "expert_usage": mean_gate,
        }


def build_sparse_expert_board_router_from_config(config: dict[str, Any]) -> SparseExpertBoardRouter:
    cfg = dict(config)
    return SparseExpertBoardRouter(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        num_experts=int(cfg.get("num_experts", NUM_EXPERTS)),
        top_k=int(cfg.get("top_k", 2)),
        router_temperature=float(cfg.get("router_temperature", 1.0)),
        load_balance_weight=float(cfg.get("load_balance_weight", 0.01)),
        router_entropy_weight=float(cfg.get("router_entropy_weight", 0.001)),
        encoding_adapter=str(cfg.get("encoding_adapter", cfg.get("encoding", SIMPLE_18))),
    )

"""Permanent Ryser Coupling Network (idea i239).

Builds a small (k x k, k = 6 by default) learned bipartite interaction matrix M
between top-k attacker squares and top-k defender squares; computes its
*permanent* via Ryser's formula. The permanent counts unsigned perfect matchings
of attackers to defenders (#P-hard in general but trivial at k <= 8 via Ryser).
Distinct from i058 DPP (uses det) and i226 Pfaffian (signed matchings on
skew-symmetric); the permanent is the unsigned dual.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardConvStem


def _ryser_permanent(M: torch.Tensor) -> torch.Tensor:
    """Compute permanent via Ryser's formula. M: (B, k, k), k <= 8 recommended."""
    B, k, _ = M.shape
    device = M.device
    total = torch.zeros(B, device=device, dtype=M.dtype)
    sign = (-1.0) ** k
    # Iterate over all 2^k subsets (skip the empty set in Ryser variant; here we
    # use the standard form with full inclusion-exclusion).
    for s in range(1, 1 << k):
        # Bitmask -> subset indicator (k,).
        bits = torch.tensor([(s >> i) & 1 for i in range(k)], dtype=M.dtype, device=device)
        popcount = int(bits.sum().item())
        # Row sums over selected columns: (B, k).
        row_sums = (M * bits.view(1, 1, k)).sum(dim=-1)
        prod = row_sums.prod(dim=-1)
        total = total + ((-1.0) ** (k - popcount)) * prod
    return sign * total


class PermanentRyserNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 32,
        stem_depth: int = 2,
        topk_k: int = 6,
        hidden_dim: int = 96,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if topk_k > 8:
            raise ValueError("topk_k must be <= 8 for Ryser tractability")
        self.stem = BoardConvStem(
            input_channels=input_channels, channels=channels, depth=stem_depth
        )
        self.topk_k = topk_k
        # Score each square as attacker / defender / interaction-feature source.
        self.attacker_score = nn.Conv2d(channels, 1, kernel_size=1)
        self.defender_score = nn.Conv2d(channels, 1, kernel_size=1)
        self.interaction_proj = nn.Conv2d(channels, channels, kernel_size=1)
        # Bilinear interaction projector to scalar from a pair of feature vectors.
        self.bilinear = nn.Bilinear(channels, channels, 1)
        feature_dim = 4 + channels  # log|perm|, sign, ||M||_F, mean(M), pooled
        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def soft_topk(self, scores: torch.Tensor, k: int) -> torch.Tensor:
        # scores: (B, 64). Return top-k indices (B, k); straight-through gradient.
        return scores.topk(k, dim=-1).indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)                              # (B, C, 8, 8)
        B, C, H, W = feat.shape
        a_scores = self.attacker_score(feat).view(B, H * W)   # (B, 64)
        d_scores = self.defender_score(feat).view(B, H * W)
        att_idx = self.soft_topk(a_scores, self.topk_k)       # (B, k)
        def_idx = self.soft_topk(d_scores, self.topk_k)
        feats_flat = self.interaction_proj(feat).view(B, C, H * W).transpose(1, 2)  # (B, 64, C)
        att_feats = torch.gather(feats_flat, 1, att_idx.unsqueeze(-1).expand(-1, -1, C))  # (B, k, C)
        def_feats = torch.gather(feats_flat, 1, def_idx.unsqueeze(-1).expand(-1, -1, C))  # (B, k, C)
        # Bilinear pairwise: M[i, j] = bilinear(att_feats[i], def_feats[j]).
        # Expand to (B, k, k, C) pairs.
        af = att_feats.unsqueeze(2).expand(-1, -1, self.topk_k, -1).reshape(-1, C)
        df = def_feats.unsqueeze(1).expand(-1, self.topk_k, -1, -1).reshape(-1, C)
        pair = self.bilinear(af, df).view(B, self.topk_k, self.topk_k)  # (B, k, k)
        # Pass through sigmoid to keep entries in [0, 1] -> permanent stays bounded.
        M = torch.sigmoid(pair)
        perm = _ryser_permanent(M)                                       # (B,)
        log_abs_perm = torch.log(perm.abs() + 1e-6)
        sign_perm = torch.tanh(perm * 4.0)  # smooth sign
        fro = M.flatten(1).norm(dim=-1)
        mean_M = M.flatten(1).mean(dim=-1)
        pooled = feat.mean(dim=(-2, -1))
        feature = torch.cat(
            [
                log_abs_perm.unsqueeze(-1),
                sign_perm.unsqueeze(-1),
                fro.unsqueeze(-1),
                mean_M.unsqueeze(-1),
                pooled,
            ],
            dim=-1,
        )
        out = self.head(feature)
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out


def build_permanent_ryser_from_config(config: dict[str, Any]) -> PermanentRyserNetwork:
    return PermanentRyserNetwork(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 32)),
        stem_depth=int(config.get("stem_depth", 2)),
        topk_k=int(config.get("topk_k", 6)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        dropout=float(config.get("dropout", 0.1)),
    )

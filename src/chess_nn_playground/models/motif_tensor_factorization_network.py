"""Motif Tensor Factorization Network for idea i182.

Faithful implementation of the markdown thesis under
``ideas/registry/i182_motif_tensor_factorization_network/``. The packet thesis
is that puzzle signal is a *multiplicative* relation among typed roles:

    attacker x target x defender x line-relation x tempo

A plain CNN learns these implicitly. This network instead represents
each of those role candidates explicitly and scores their conjunction
through a low-rank CP factorization on a 4-way motif tensor::

    score(i, j, k) = sum_r a_r(A_i) * t_r(T_j) * d_r(D_k) * rel_r(R_ij)

Forward pass:

* A compact convolutional trunk produces per-square features ``H``.
* Three role-specific 1x1 conv heads emit attacker / target / defender
  selection logits over the 64 squares; the top ``P`` candidates per
  role are gathered.
* Three role-specific token MLPs project the gathered candidate
  features into rank-``R`` factor vectors.
* A small relation MLP, conditioned on attacker / target factors and
  a learned signed (delta_rank, delta_file) positional embedding,
  produces a rank-``R`` line-relation factor for every (attacker_i,
  target_j) pair.
* The motif score tensor ``M[i, j, k] = sum_r a_r(i) * t_r(j) *
  d_r(k) * rel_r(i, j)`` is computed as an einsum.
* The final puzzle logit reads from pooled motif features:
  ``top_motif_scores``, ``motif_entropy``, an own/opponent motif
  contrast obtained by flipping the side-to-move plane and re-running
  the head, and a ``near_disproof_score`` defined as the smallest
  per-component magnitude of the top motif (the multiplicative
  motif's weakest leg).

The packet's required ablations are wired through the ``ablation``
flag:

* ``"none"`` -- main multiplicative motif tensor.
* ``"additive_motif_score"`` -- replace
  ``a * t * d * rel`` with ``a + t + d + rel``, killing the
  conjunction.
* ``"no_relation_embedding"`` -- set ``rel`` to ones, so the line
  relation factor cannot contribute.

The ``"rank_8_24_64"`` ablation is the ``rank`` config knob and is
not a separate code path.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import (
    BoardConvStem,
    BoardTensorSpec,
    require_board_tensor,
)


_VALID_ABLATIONS = {
    "none",
    "additive_motif_score",
    "no_relation_embedding",
}


def _square_coords(device: torch.device) -> torch.Tensor:
    """Return (64, 2) tensor of (rank, file) per square in row-major order."""
    rank = torch.arange(8, device=device).repeat_interleave(8)
    file = torch.arange(8, device=device).repeat(8)
    return torch.stack([rank, file], dim=-1).to(torch.float32)


class MotifTensorFactorizationNetwork(nn.Module):
    """CP-factorized attacker x target x defender x relation motif network.

    Forward returns a dict with at least:

      - ``logits``: ``(B,)`` puzzle logit fed to BCE-with-logits
        (``(B, num_classes)`` if ``num_classes > 1``).
      - ``prob``: ``sigmoid(logits)`` when ``num_classes == 1``.
      - ``motif_score_tensor``: ``(B, P, P, P)`` full motif scores over
        the selected (attacker, target, defender) triples.
      - ``top_motif_scores``: ``(B, top_motifs)`` top motif values.
      - ``top_motif_indices``: ``(B, top_motifs)`` flattened indices of
        the top motifs in the (P, P, P) tensor.
      - ``motif_entropy``: ``(B,)`` entropy of softmax(motif_score_tensor)
        across the flattened motif axis.
      - ``own_motif_score``: ``(B,)`` mean of the top motifs at the
        actual side-to-move plane.
      - ``opponent_motif_score``: ``(B,)`` mean of the top motifs after
        the side-to-move plane is flipped.
      - ``motif_contrast``: ``(B,)`` ``own_motif_score -
        opponent_motif_score``.
      - ``near_disproof_score``: ``(B,)`` smallest per-component
        magnitude of the top motif's CP factors -- the multiplicative
        motif's weakest leg.
      - ``attacker_top_indices``, ``target_top_indices``,
        ``defender_top_indices``: ``(B, P)`` selected square indices
        per role.
      - ``trunk_features``: ``(B, channels, 8, 8)``.
      - ``ablation_active``, ``uses_multiplicative_motif``,
        ``uses_relation_embedding``, ``rank``, ``num_top_candidates``,
        ``num_top_motifs``: ``(B,)`` flags exposing the running
        ablation.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        depth: int = 2,
        token_dim: int = 64,
        rank: int = 24,
        top_candidates: int = 8,
        top_motifs: int = 16,
        relation_hidden: int = 64,
        head_hidden: int = 96,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        ablation: str = "none",
        side_to_move_plane_index: int = 12,
    ) -> None:
        super().__init__()
        if depth < 1 or channels < 1 or num_classes < 1:
            raise ValueError("depth, channels, num_classes must be >= 1")
        if token_dim < 1 or rank < 1:
            raise ValueError("token_dim, rank must be >= 1")
        if top_candidates < 1 or top_candidates > 64:
            raise ValueError("top_candidates must be in [1, 64]")
        if top_motifs < 1 or top_motifs > top_candidates ** 3:
            raise ValueError("top_motifs must be in [1, top_candidates**3]")
        if relation_hidden < 1 or head_hidden < 1:
            raise ValueError("relation_hidden, head_hidden must be >= 1")
        if not 0 <= side_to_move_plane_index < input_channels:
            raise ValueError("side_to_move_plane_index must be in [0, input_channels)")
        if ablation not in _VALID_ABLATIONS:
            raise ValueError(
                f"ablation must be one of {sorted(_VALID_ABLATIONS)}, got {ablation!r}"
            )

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.depth = int(depth)
        self.token_dim = int(token_dim)
        self.rank = int(rank)
        self.top_candidates = int(top_candidates)
        self.top_motifs = int(top_motifs)
        self.relation_hidden = int(relation_hidden)
        self.head_hidden = int(head_hidden)
        self.dropout = float(dropout)
        self.use_batchnorm = bool(use_batchnorm)
        self.ablation = str(ablation)
        self.side_to_move_plane_index = int(side_to_move_plane_index)

        self.uses_multiplicative_motif = self.ablation != "additive_motif_score"
        self.uses_relation_embedding = self.ablation != "no_relation_embedding"

        self.trunk = BoardConvStem(
            input_channels=self.input_channels,
            channels=self.channels,
            depth=self.depth,
            use_batchnorm=self.use_batchnorm,
        )

        # Per-role selectors: 1x1 conv giving one selection logit per
        # square. The top-K squares per role become the role candidates.
        self.attacker_selector = nn.Conv2d(self.channels, 1, kernel_size=1)
        self.target_selector = nn.Conv2d(self.channels, 1, kernel_size=1)
        self.defender_selector = nn.Conv2d(self.channels, 1, kernel_size=1)

        # Per-role token projections to rank-R CP factors.
        self.attacker_token = nn.Sequential(
            nn.Linear(self.channels, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.rank),
        )
        self.target_token = nn.Sequential(
            nn.Linear(self.channels, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.rank),
        )
        self.defender_token = nn.Sequential(
            nn.Linear(self.channels, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.rank),
        )

        # Relative-position embedding for line relations. We embed the
        # signed (delta_rank, delta_file) pair via a small MLP -- this
        # captures same-rank/file/diagonal/knight-jump style relations.
        # Indices range over [-7, 7], offset by +7.
        self.relation_pos_embed = nn.Embedding(15 * 15, self.relation_hidden)
        self.relation_mlp = nn.Sequential(
            nn.Linear(self.rank * 2 + self.relation_hidden, self.relation_hidden),
            nn.GELU(),
            nn.Linear(self.relation_hidden, self.rank),
        )

        # Final head consumes:
        #   top_motif_scores (top_motifs), motif_entropy (1),
        #   own/opponent/contrast/near_disproof scalars (4)
        head_input_dim = self.top_motifs + 1 + 4
        self.head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, self.head_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(self.head_hidden, 1),
        )

    def _gather_role_candidates(
        self,
        feats_flat: torch.Tensor,
        selector_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (top_features, top_indices) shaped (B, P, C) and (B, P)."""
        top_logits, top_idx = torch.topk(selector_logits, self.top_candidates, dim=-1)
        # Use a soft gating over the top-P so the selector is differentiable
        # while still highlighting only a handful of squares.
        gate = F.softmax(top_logits, dim=-1).unsqueeze(-1)
        gathered = torch.gather(
            feats_flat, 1, top_idx.unsqueeze(-1).expand(-1, -1, feats_flat.shape[-1])
        )
        return gathered * (1.0 + gate), top_idx

    def _relation_factor(
        self,
        attacker_factors: torch.Tensor,
        target_factors: torch.Tensor,
        attacker_idx: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute rank-R relation factor per (attacker_i, target_j) pair.

        Shape: ``(B, P, P, R)``.
        """
        batch_size = attacker_factors.shape[0]
        device = attacker_factors.device
        coords = _square_coords(device)  # (64, 2)
        attacker_pos = coords[attacker_idx]  # (B, P, 2)
        target_pos = coords[target_idx]  # (B, P, 2)
        delta = (attacker_pos.unsqueeze(2) - target_pos.unsqueeze(1)).long()  # (B, P, P, 2)
        # Map [-7, 7] -> [0, 14], then 2D -> flat index in [0, 224].
        delta_idx = (delta[..., 0] + 7) * 15 + (delta[..., 1] + 7)
        rel_pos = self.relation_pos_embed(delta_idx)  # (B, P, P, relation_hidden)

        a_expand = attacker_factors.unsqueeze(2).expand(-1, -1, self.top_candidates, -1)
        t_expand = target_factors.unsqueeze(1).expand(-1, self.top_candidates, -1, -1)
        rel_input = torch.cat([a_expand, t_expand, rel_pos], dim=-1)  # (B, P, P, _)
        rel_factor = self.relation_mlp(rel_input)  # (B, P, P, R)
        if not self.uses_relation_embedding:
            rel_factor = torch.ones_like(rel_factor)
        return rel_factor

    def _score_with_features(
        self,
        feats: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size = feats.shape[0]
        feats_flat = feats.flatten(2).transpose(1, 2)  # (B, 64, C)

        attacker_logits = self.attacker_selector(feats).flatten(1)  # (B, 64)
        target_logits = self.target_selector(feats).flatten(1)
        defender_logits = self.defender_selector(feats).flatten(1)

        attacker_feats, attacker_idx = self._gather_role_candidates(feats_flat, attacker_logits)
        target_feats, target_idx = self._gather_role_candidates(feats_flat, target_logits)
        defender_feats, defender_idx = self._gather_role_candidates(feats_flat, defender_logits)

        a_factors = self.attacker_token(attacker_feats)  # (B, P, R)
        t_factors = self.target_token(target_feats)
        d_factors = self.defender_token(defender_feats)

        rel_factors = self._relation_factor(
            a_factors, t_factors, attacker_idx, target_idx
        )  # (B, P, P, R)

        if self.uses_multiplicative_motif:
            # M[b, i, j, k] = sum_r a[b,i,r] * t[b,j,r] * d[b,k,r] * rel[b,i,j,r]
            attacker_target = a_factors.unsqueeze(2) * t_factors.unsqueeze(1)  # (B, P, P, R)
            attacker_target_rel = attacker_target * rel_factors  # (B, P, P, R)
            motif_tensor = torch.einsum(
                "bijr,bkr->bijk", attacker_target_rel, d_factors
            )  # (B, P, P, P)
        else:
            # Additive ablation: M[b,i,j,k] = sum_r (a + t + d + rel) reduces to
            # (sum a) + (sum t) + (sum d) + sum rel, broadcast across triples.
            a_sum = a_factors.sum(dim=-1)  # (B, P)
            t_sum = t_factors.sum(dim=-1)
            d_sum = d_factors.sum(dim=-1)
            rel_sum = rel_factors.sum(dim=-1)  # (B, P, P)
            motif_tensor = (
                a_sum.unsqueeze(-1).unsqueeze(-1)
                + t_sum.unsqueeze(1).unsqueeze(-1)
                + d_sum.unsqueeze(1).unsqueeze(1)
                + rel_sum.unsqueeze(-1)
            )

        motif_flat = motif_tensor.reshape(batch_size, -1)  # (B, P^3)
        top_values, top_idx = torch.topk(motif_flat, self.top_motifs, dim=-1)

        log_probs = F.log_softmax(motif_flat, dim=-1)
        probs = log_probs.exp()
        motif_entropy = -(probs * log_probs).sum(dim=-1)  # (B,)

        # Near-disproof score: for the single best motif, the smallest
        # absolute factor magnitude across the four CP legs (attacker,
        # target, defender, relation). A small leg means the motif is
        # held up by one weak component -- the multiplicative form
        # would collapse if that leg were missing.
        best_flat_idx = top_idx[:, 0]  # (B,)
        p = self.top_candidates
        best_i = (best_flat_idx // (p * p)).long()
        best_j = ((best_flat_idx // p) % p).long()
        best_k = (best_flat_idx % p).long()
        b_range = torch.arange(batch_size, device=motif_tensor.device)
        a_norm = a_factors[b_range, best_i].abs().mean(dim=-1)
        t_norm = t_factors[b_range, best_j].abs().mean(dim=-1)
        d_norm = d_factors[b_range, best_k].abs().mean(dim=-1)
        rel_norm = rel_factors[b_range, best_i, best_j].abs().mean(dim=-1)
        leg_strengths = torch.stack([a_norm, t_norm, d_norm, rel_norm], dim=-1)
        near_disproof_score = leg_strengths.min(dim=-1).values  # (B,)

        return {
            "motif_tensor": motif_tensor,
            "top_values": top_values,
            "top_idx": top_idx,
            "motif_entropy": motif_entropy,
            "near_disproof_score": near_disproof_score,
            "attacker_idx": attacker_idx,
            "target_idx": target_idx,
            "defender_idx": defender_idx,
            "a_factors": a_factors,
            "t_factors": t_factors,
            "d_factors": d_factors,
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        batch_size = x.shape[0]
        feats = self.trunk(x)  # (B, C, 8, 8)
        own = self._score_with_features(feats)

        # Own/opponent contrast: re-run the head on the same trunk
        # features but with the side-to-move plane flipped. The trunk
        # has already consumed the original plane, so the opponent
        # branch perturbs the *input* and re-trunks. This makes the
        # contrast a faithful intervention rather than a no-op.
        x_flipped = x.clone()
        x_flipped[:, self.side_to_move_plane_index] = (
            1.0 - x_flipped[:, self.side_to_move_plane_index]
        )
        feats_flipped = self.trunk(x_flipped)
        opponent = self._score_with_features(feats_flipped)

        own_top_mean = own["top_values"].mean(dim=-1)  # (B,)
        opponent_top_mean = opponent["top_values"].mean(dim=-1)
        motif_contrast = own_top_mean - opponent_top_mean

        head_input = torch.cat(
            [
                own["top_values"],
                own["motif_entropy"].unsqueeze(-1),
                own_top_mean.unsqueeze(-1),
                opponent_top_mean.unsqueeze(-1),
                motif_contrast.unsqueeze(-1),
                own["near_disproof_score"].unsqueeze(-1),
            ],
            dim=-1,
        )
        scalar_logit = self.head(head_input).squeeze(-1)  # (B,)

        if self.num_classes == 1:
            logits = scalar_logit
        else:
            logits = torch.zeros(
                batch_size, self.num_classes, device=feats.device, dtype=feats.dtype
            )
            logits[:, -1] = scalar_logit

        with torch.no_grad():
            ones = torch.ones(batch_size, device=feats.device, dtype=feats.dtype)
            ablation_active = ones * (0.0 if self.ablation == "none" else 1.0)
            uses_mult = ones * (1.0 if self.uses_multiplicative_motif else 0.0)
            uses_rel = ones * (1.0 if self.uses_relation_embedding else 0.0)
            rank_flag = ones * float(self.rank)
            top_cands_flag = ones * float(self.top_candidates)
            top_motifs_flag = ones * float(self.top_motifs)

        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "motif_score_tensor": own["motif_tensor"],
            "top_motif_scores": own["top_values"],
            "top_motif_indices": own["top_idx"].to(dtype=feats.dtype),
            "motif_entropy": own["motif_entropy"],
            "own_motif_score": own_top_mean,
            "opponent_motif_score": opponent_top_mean,
            "motif_contrast": motif_contrast,
            "near_disproof_score": own["near_disproof_score"],
            "attacker_top_indices": own["attacker_idx"].to(dtype=feats.dtype),
            "target_top_indices": own["target_idx"].to(dtype=feats.dtype),
            "defender_top_indices": own["defender_idx"].to(dtype=feats.dtype),
            "trunk_features": feats,
            "ablation_active": ablation_active,
            "uses_multiplicative_motif": uses_mult,
            "uses_relation_embedding": uses_rel,
            "rank": rank_flag,
            "num_top_candidates": top_cands_flag,
            "num_top_motifs": top_motifs_flag,
        }
        if self.num_classes == 1:
            output["prob"] = torch.sigmoid(logits)
        return output


def build_motif_tensor_factorization_network_from_config(
    config: dict[str, Any],
) -> MotifTensorFactorizationNetwork:
    cfg = dict(config)
    cfg.pop("name", None)
    cfg.pop("mechanism_family", None)
    cfg.pop("packet_profile", None)
    return MotifTensorFactorizationNetwork(
        input_channels=int(cfg.pop("input_channels", 18)),
        num_classes=int(cfg.pop("num_classes", 1)),
        channels=int(cfg.pop("channels", 64)),
        depth=int(cfg.pop("depth", 2)),
        token_dim=int(cfg.pop("token_dim", 64)),
        rank=int(cfg.pop("rank", 24)),
        top_candidates=int(cfg.pop("top_candidates", 8)),
        top_motifs=int(cfg.pop("top_motifs", 16)),
        relation_hidden=int(cfg.pop("relation_hidden", 64)),
        head_hidden=int(cfg.pop("head_hidden", cfg.pop("hidden_dim", 96))),
        dropout=float(cfg.pop("dropout", 0.1)),
        use_batchnorm=bool(cfg.pop("use_batchnorm", True)),
        ablation=str(cfg.pop("ablation", "none")),
        side_to_move_plane_index=int(cfg.pop("side_to_move_plane_index", 12)),
    )


__all__ = [
    "MotifTensorFactorizationNetwork",
    "build_motif_tensor_factorization_network_from_config",
]

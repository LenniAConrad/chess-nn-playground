"""Cayley Orthogonal Map Network (idea i237).

Builds a skew-symmetric matrix A from board features and forms the Cayley map
Q = (I - A) (I + A)^{-1}, which is in SO(r) for any skew A with no eigenvalue at
-1 (guaranteed by spectral-clipping ||A||). Q acts on a learned r-dim feature
basis pooled from the board, producing rotated embeddings whose deviation from
the identity action is the central feature. Distinct from i063 Polar-Procrustes
(which uses polar/SVD decomposition) and from QR.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardConvStem


class CayleyOrthogonalNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 32,
        stem_depth: int = 2,
        rank_r: int = 12,
        hidden_dim: int = 96,
        dropout: float = 0.1,
        spectral_clip: float = 0.5,
    ) -> None:
        super().__init__()
        self.stem = BoardConvStem(
            input_channels=input_channels, channels=channels, depth=stem_depth
        )
        self.rank_r = rank_r
        self.spectral_clip = spectral_clip
        # Project pooled stem features to two r-dim vectors -> outer-product builds A.
        # A is r x r; we make it skew via A = U V^T - V U^T.
        self.proj_U = nn.Linear(channels, rank_r * rank_r // 2)
        # Reference basis to be rotated.
        self.basis = nn.Parameter(torch.randn(rank_r, channels) / channels ** 0.5)
        # Identity register.
        self.register_buffer("eye_r", torch.eye(rank_r), persistent=False)
        feature_dim = rank_r * 3 + 2  # rotated basis trace per row + diag(Q) + 2 scalars
        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def cayley(self, A: torch.Tensor) -> torch.Tensor:
        # A: (B, r, r) skew. Q = (I - A) @ (I + A)^{-1}.
        I = self.eye_r.expand_as(A)
        return torch.linalg.solve(I + A, I - A)

    def build_skew(self, pooled: torch.Tensor) -> torch.Tensor:
        # pooled: (B, channels). Produce (B, r, r) skew with bounded ||A||_F.
        r = self.rank_r
        flat = self.proj_U(pooled)                         # (B, r*r/2)
        # Pack into upper-triangular and form skew.
        B = pooled.shape[0]
        n_upper = r * (r - 1) // 2
        flat = flat[:, :n_upper]
        triu = torch.zeros(B, r, r, device=pooled.device, dtype=pooled.dtype)
        ii, jj = torch.triu_indices(r, r, offset=1, device=pooled.device).unbind(0)
        triu[:, ii, jj] = flat
        skew = triu - triu.transpose(-1, -2)
        # Spectral clip: scale so largest singular value <= spectral_clip.
        # Cheap upper bound via Frobenius: ||A||_2 <= ||A||_F.
        fro = skew.flatten(1).norm(dim=1).clamp(min=1e-6)
        scale = torch.minimum(torch.ones_like(fro), self.spectral_clip / fro)
        return skew * scale.view(-1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)                                 # (B, C, 8, 8)
        pooled = feat.mean(dim=(-2, -1))                    # (B, C)
        A = self.build_skew(pooled)                         # (B, r, r) skew
        Q = self.cayley(A)                                  # (B, r, r) in SO(r)
        # Rotate the learned basis. basis: (r, C); produce per-row rotated trace.
        # rotated = Q @ basis  -> (B, r, C); pool over feature dim.
        rotated = torch.einsum("brs,sc->brc", Q, self.basis)
        # Identity-deviation features.
        I = self.eye_r.expand_as(Q)
        dev = Q - I
        diag_Q = torch.diagonal(Q, dim1=-2, dim2=-1)        # (B, r)
        per_row_norm = rotated.norm(dim=-1)                 # (B, r)
        sym_part = ((Q + Q.transpose(-1, -2)) ** 2).sum(dim=(-1, -2))  # (B,)
        det_proxy = torch.diagonal(rotated @ rotated.transpose(-1, -2), dim1=-2, dim2=-1).sum(-1)  # (B,)
        feature = torch.cat(
            [diag_Q, per_row_norm, dev.flatten(1)[:, : self.rank_r], sym_part.unsqueeze(-1), det_proxy.unsqueeze(-1)],
            dim=-1,
        )
        out = self.head(feature)
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out


def build_cayley_orthogonal_from_config(config: dict[str, Any]) -> CayleyOrthogonalNetwork:
    return CayleyOrthogonalNetwork(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 32)),
        stem_depth=int(config.get("stem_depth", 2)),
        rank_r=int(config.get("rank_r", 12)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        dropout=float(config.get("dropout", 0.1)),
        spectral_clip=float(config.get("spectral_clip", 0.5)),
    )

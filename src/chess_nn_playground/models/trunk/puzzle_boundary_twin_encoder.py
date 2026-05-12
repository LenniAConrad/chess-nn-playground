"""Puzzle Boundary Twin Encoder for idea i184.

Faithful implementation of the markdown thesis under
``ideas/registry/i184_puzzle_boundary_twin_encoder/``. The architecture is a
*siamese* board encoder that is applied identically to every position
in a mini-batch (puzzle / near-puzzle / random), and a single linear
*boundary surface* that turns the encoder latent into one puzzle logit:

    z      = encoder(board)         # shared "twin" encoder
    e      = boundary_embedding(z)  # margin-friendly latent
    logit  = <e_unit, w_unit> * scale + bias

At inference the model is a single-board single-logit puzzle classifier.
The "twin" structure is the training contract: the same encoder is
shared across the in-batch pair triples (puzzle, near, random) so the
trainer can apply the packet's pair-margin objective on top of BCE,

    logit(puzzle) >= logit(near)   + margin_near
    logit(near)   >= logit(random) + margin_random_surface

The forward pass exposes the raw boundary score, the unit-norm boundary
embedding, the pre-normalisation embedding norm and the trunk energy so
a margin trainer with reliable group ids can attach pair losses without
any extra compute.

Distinct from idea i172 (Near-Puzzle Margin Twin Network), which uses
two parallel readout projectors over a shared encoder. Here the twin
structure lives across batch items rather than across head branches:
one encoder, one boundary surface, one logit.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardConvStem, BoardTensorSpec, require_board_tensor


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(self.norm(x)))))


class PuzzleBoundaryTwinEncoder(nn.Module):
    """Siamese board encoder + linear boundary surface.

    Forward returns a dict with at least:

      - ``logits``: ``(B,)`` puzzle logit for the BCE-with-logits trainer
        (``(B, num_classes)`` if ``num_classes > 1``).
      - ``prob``: ``sigmoid(logits)`` when ``num_classes == 1``.
      - ``boundary_score``: ``(B,)`` raw signed margin to the decision
        surface; equals ``logits`` when ``num_classes == 1``. The
        trainer reads this for in-batch pair-margin losses
        (``boundary_score(puzzle) >= boundary_score(near) + m`` etc.).
      - ``boundary_distance``: ``(B,)`` ``|boundary_score|``.
      - ``boundary_embedding``: ``(B, embedding_dim)`` unit-norm
        embedding consumed by the linear boundary surface.
      - ``z_shared``: ``(B, shared_dim)`` post-pool descriptor before the
        margin projector.
      - ``embedding_norm``: ``(B,)`` pre-normalisation L2 norm of the
        boundary embedding (representational-collapse monitor).
      - ``trunk_energy``: ``(B,)`` mean-square trunk activation.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        depth: int = 2,
        shared_dim: int = 128,
        embedding_dim: int = 96,
        projector_hidden: int = 96,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        score_scale: float = 4.0,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if shared_dim < 1 or embedding_dim < 1 or projector_hidden < 1:
            raise ValueError("latent dims must be >= 1")
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        if score_scale <= 0.0:
            raise ValueError("score_scale must be > 0")

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.depth = int(depth)
        self.shared_dim = int(shared_dim)
        self.embedding_dim = int(embedding_dim)
        self.projector_hidden = int(projector_hidden)
        self.dropout = float(dropout)
        self.use_batchnorm = bool(use_batchnorm)
        self.score_scale = float(score_scale)

        self.trunk = BoardConvStem(
            input_channels=self.input_channels,
            channels=self.channels,
            depth=self.depth,
            use_batchnorm=self.use_batchnorm,
        )
        # Mean + max + std pooling -> richer pooled descriptor than the
        # plain CNN baseline; std is a cheap second-moment cue useful
        # for puzzle/near-puzzle separation.
        pooled_dim = self.channels * 3
        self.shared_proj = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, self.shared_dim),
            nn.GELU(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
        )

        self.embedding_projector = _MLP(
            in_dim=self.shared_dim,
            hidden_dim=self.projector_hidden,
            out_dim=self.embedding_dim,
            dropout=self.dropout,
        )

        # Boundary surface: a single learned direction in the embedding
        # space. The puzzle logit is its signed cosine distance times a
        # learned scale, plus a learned bias. This is the explicit
        # "margin surface" the packet calls for.
        self.boundary_direction = nn.Parameter(torch.randn(self.embedding_dim) * 0.1)
        self.boundary_scale = nn.Parameter(torch.tensor(float(self.score_scale)))
        self.boundary_bias = nn.Parameter(torch.zeros(1))

        # Auxiliary logit head for num_classes > 1; falls back to the
        # boundary score for the binary case.
        if self.num_classes > 1:
            self.aux_head = nn.Sequential(
                nn.LayerNorm(self.embedding_dim),
                nn.Linear(self.embedding_dim, self.projector_hidden),
                nn.GELU(),
                nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
                nn.Linear(self.projector_hidden, self.num_classes),
            )
        else:
            self.aux_head = None

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feats = self.trunk(x)  # (B, C, 8, 8)
        flat = feats.flatten(2)  # (B, C, 64)
        mean_pool = flat.mean(dim=2)
        max_pool = flat.amax(dim=2)
        # Use unbiased=False so single-element rows degrade gracefully
        # (unused in practice but keeps the op safe).
        std_pool = flat.std(dim=2, unbiased=False)
        pooled = torch.cat([mean_pool, max_pool, std_pool], dim=1)
        return feats, self.shared_proj(pooled)

    def _boundary_score(self, embedding_unit: torch.Tensor) -> torch.Tensor:
        direction_unit = F.normalize(self.boundary_direction, dim=0, eps=1e-8)
        cosine = embedding_unit @ direction_unit
        return cosine * self.boundary_scale + self.boundary_bias

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats, z_shared = self._encode(x)
        embedding_raw = self.embedding_projector(z_shared)
        embedding_norm = embedding_raw.norm(dim=1)
        embedding_unit = F.normalize(embedding_raw, dim=1, eps=1e-8)

        boundary_score = self._boundary_score(embedding_unit)

        if self.num_classes == 1:
            logits = boundary_score
        else:
            assert self.aux_head is not None
            logits = self.aux_head(embedding_raw)
            logits = _format_logits(logits, self.num_classes)

        with torch.no_grad():
            trunk_energy = feats.square().mean(dim=(1, 2, 3))
            boundary_distance = boundary_score.detach().abs()

        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "boundary_score": boundary_score,
            "boundary_distance": boundary_distance,
            "boundary_embedding": embedding_unit,
            "z_shared": z_shared,
            "embedding_norm": embedding_norm,
            "trunk_energy": trunk_energy,
        }
        if self.num_classes == 1:
            output["prob"] = torch.sigmoid(logits)
        return output


def build_puzzle_boundary_twin_encoder_from_config(
    config: dict[str, Any],
) -> PuzzleBoundaryTwinEncoder:
    cfg = dict(config)
    cfg.pop("name", None)
    cfg.pop("mechanism_family", None)
    cfg.pop("packet_profile", None)
    hidden_dim = cfg.pop("hidden_dim", 96)
    embedding_dim = cfg.pop("embedding_dim", hidden_dim)
    projector_hidden = cfg.pop("projector_hidden", hidden_dim)
    shared_dim = cfg.pop("shared_dim", 128)
    return PuzzleBoundaryTwinEncoder(
        input_channels=int(cfg.pop("input_channels", 18)),
        num_classes=int(cfg.pop("num_classes", 1)),
        channels=int(cfg.pop("channels", 64)),
        depth=int(cfg.pop("depth", 2)),
        shared_dim=int(shared_dim),
        embedding_dim=int(embedding_dim),
        projector_hidden=int(projector_hidden),
        dropout=float(cfg.pop("dropout", 0.1)),
        use_batchnorm=bool(cfg.pop("use_batchnorm", True)),
        score_scale=float(cfg.pop("score_scale", 4.0)),
    )


__all__ = [
    "PuzzleBoundaryTwinEncoder",
    "build_puzzle_boundary_twin_encoder_from_config",
]

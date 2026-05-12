"""Capsule Motif BoardNet for idea i155.

A capsule-style architecture for the puzzle_binary contract. The model
encodes local board patterns as small *primary* capsule vectors -- one
per (square, primary-channel) cell -- and routes them into a small set
of *motif* capsules by iterative agreement (a la Sabour et al.,
"Dynamic Routing Between Capsules"). Motif capsule lengths and the
pooled trunk feature drive the puzzle logit.

The model deliberately keeps the trunk small. The capsule head is the
distinctive piece: each primary capsule predicts every motif capsule
through its own learned transformation matrix `W_m`, then `T` rounds
of softmax-normalised agreement compute motif activations.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


def _squash(s: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    sq_norm = s.pow(2).sum(dim=dim, keepdim=True)
    scale = sq_norm / (1.0 + sq_norm)
    return scale * s / torch.sqrt(sq_norm + eps)


class _ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.norm = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.norm(self.conv(x)), inplace=True)


class CapsuleMotifBoardNet(nn.Module):
    """Capsule motif network for the puzzle_binary contract.

    Pipeline:

    1. **Stem.** A small ConvStack lifts the `(B, input_channels, 8, 8)`
       board, augmented with two coordinate channels (rank, file), to
       a `(B, channels, 8, 8)` board representation.
    2. **Primary capsules.** A `3x3` `Conv2d(channels -> num_primary_caps * primary_capsule_dim)`
       projects the trunk into primary capsule slots. Reshaped to
       `(B, N_caps, D_caps)` with `N_caps = 8 * 8 * num_primary_caps`,
       each row is a small vector encoding pose/type of a local pattern
       at one square. The per-capsule vectors are squashed to live on a
       bounded manifold.
    3. **Motif transforms.** Each primary capsule `i` predicts each
       motif `m` through a learned matrix `W_m`:
       `u_hat[b, i, m] = W_m * u[b, i]`. Implemented as a single
       `Linear(D_caps, num_motif_caps * motif_capsule_dim)` per capsule
       index family (shared across primary positions) so the parameter
       count is `D_caps * num_motif_caps * motif_capsule_dim`.
    4. **Routing-by-agreement.** Routing logits `b[b, i, m]` start at
       zero. For `routing_iterations` iterations:

       ```
       c = softmax(b, dim=motif)
       s[b, m] = sum_i c[b, i, m] * u_hat[b, i, m]
       v[b, m] = squash(s[b, m])
       b += <u_hat[b, i, m], v[b, m]>
       ```

       The final motif capsule activations `v` carry both magnitude
       (agreement strength) and direction (pose).
    5. **Readout.** Motif capsule lengths `||v_m||` and the global
       average-pooled trunk feature are concatenated, run through a
       small MLP, and produce one puzzle logit. Routing entropies and
       motif activation norms are exposed as diagnostics.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        num_primary_caps: int = 8,
        primary_capsule_dim: int = 8,
        num_motif_caps: int = 16,
        motif_capsule_dim: int = 16,
        routing_iterations: int = 3,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "CapsuleMotifBoardNet supports the puzzle_binary one-logit contract"
            )
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if channels < 1 or hidden_dim < 1:
            raise ValueError("channels and hidden_dim must be positive")
        if num_primary_caps < 1 or primary_capsule_dim < 1:
            raise ValueError("num_primary_caps and primary_capsule_dim must be positive")
        if num_motif_caps < 1 or motif_capsule_dim < 1:
            raise ValueError("num_motif_caps and motif_capsule_dim must be positive")
        if routing_iterations < 1:
            raise ValueError("routing_iterations must be >= 1")

        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.num_primary_caps = int(num_primary_caps)
        self.primary_capsule_dim = int(primary_capsule_dim)
        self.num_motif_caps = int(num_motif_caps)
        self.motif_capsule_dim = int(motif_capsule_dim)
        self.routing_iterations = int(routing_iterations)
        self.dropout_p = float(dropout)

        # Trunk: stem (with 2 coordinate planes) + (depth) ConvNormAct.
        stem_in = self.input_channels + 2
        layers: list[nn.Module] = [_ConvNormAct(stem_in, self.channels, use_batchnorm=use_batchnorm)]
        for _ in range(self.depth - 1):
            layers.append(_ConvNormAct(self.channels, self.channels, use_batchnorm=use_batchnorm))
        self.trunk = nn.Sequential(*layers)

        # Primary capsule projection: a single 3x3 conv that produces
        # num_primary_caps * primary_capsule_dim channels per square.
        self.primary_conv = nn.Conv2d(
            self.channels,
            self.num_primary_caps * self.primary_capsule_dim,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        # Motif transforms: a single learned tensor W of shape
        # (num_motif_caps, motif_capsule_dim, primary_capsule_dim).
        # u_hat[i, m] = W_m @ u_i (broadcast over primary capsule index).
        # Initialised with Kaiming uniform so that early-training motif
        # predictions are not all-zero and routing has signal to lock on.
        self.motif_transform = nn.Parameter(
            torch.empty(self.num_motif_caps, self.motif_capsule_dim, self.primary_capsule_dim)
        )
        nn.init.kaiming_uniform_(self.motif_transform, a=5 ** 0.5)

        self.pool = nn.AdaptiveAvgPool2d(1)
        head_in = self.channels + self.num_motif_caps
        head_layers: list[nn.Module] = [
            nn.Linear(head_in, self.hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if self.dropout_p > 0:
            head_layers.append(nn.Dropout(self.dropout_p))
        head_layers.append(nn.Linear(self.hidden_dim, 1))
        self.head = nn.Sequential(*head_layers)

    # -- helpers ----------------------------------------------------

    @staticmethod
    def _coordinate_channels(x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        device = x.device
        dtype = x.dtype
        rank = (
            torch.linspace(-1.0, 1.0, steps=8, device=device, dtype=dtype)
            .view(1, 1, 8, 1)
            .expand(batch, 1, 8, 8)
        )
        file = (
            torch.linspace(-1.0, 1.0, steps=8, device=device, dtype=dtype)
            .view(1, 1, 1, 8)
            .expand(batch, 1, 8, 8)
        )
        return torch.cat([rank, file], dim=1)

    def _primary_capsules(self, trunk: torch.Tensor) -> torch.Tensor:
        """Return primary capsules of shape (B, N_caps, D_caps)."""
        batch = trunk.shape[0]
        primary = self.primary_conv(trunk)  # (B, C_p * D_caps, 8, 8)
        primary = primary.view(
            batch, self.num_primary_caps, self.primary_capsule_dim, 8, 8
        )
        # Move spatial dims next to capsule channel: (B, C_p, 8, 8, D_caps).
        primary = primary.permute(0, 1, 3, 4, 2).contiguous()
        primary = primary.view(batch, self.num_primary_caps * 8 * 8, self.primary_capsule_dim)
        return _squash(primary, dim=-1)

    def _route(
        self, u_hat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Routing-by-agreement.

        Args:
          u_hat: (B, N_caps, num_motif_caps, motif_capsule_dim) predictions.

        Returns:
          v: (B, num_motif_caps, motif_capsule_dim) motif capsule activations.
          coupling: (B, N_caps, num_motif_caps) final coupling coefficients.
          routing_logits: (B, N_caps, num_motif_caps) final logits.
        """
        batch, n_caps, num_motif, _ = u_hat.shape
        b = u_hat.new_zeros(batch, n_caps, num_motif)

        v: torch.Tensor | None = None
        for step in range(self.routing_iterations):
            c = F.softmax(b, dim=2)  # softmax across motifs
            # weighted sum over primary capsules: s[b, m, d] = sum_i c[b,i,m] * u_hat[b,i,m,d]
            s = (c.unsqueeze(-1) * u_hat).sum(dim=1)
            v = _squash(s, dim=-1)
            if step + 1 < self.routing_iterations:
                # Update routing logits with agreement; detach v to keep
                # the routing update from differentiating through earlier
                # routing iterations (standard dynamic-routing recipe).
                v_detached = v.detach()
                agreement = (u_hat * v_detached.unsqueeze(1)).sum(dim=-1)
                b = b + agreement
        assert v is not None
        coupling = F.softmax(b, dim=2)
        return v, coupling, b

    # -- forward ----------------------------------------------------

    def forward(self, board: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(board, self.spec)
        coords = self._coordinate_channels(x)
        h = self.trunk(torch.cat([x, coords], dim=1))

        primary = self._primary_capsules(h)  # (B, N_caps, D_caps)
        # Predict motif capsules via the shared motif transform tensor.
        # u_hat[b, i, m, d] = sum_e W[m, d, e] * primary[b, i, e].
        # einsum keeps this exact and memory-efficient.
        u_hat = torch.einsum("mde,bie->bimd", self.motif_transform, primary)

        v, coupling, routing_logits = self._route(u_hat)

        motif_norms = v.norm(dim=-1)  # (B, num_motif_caps)
        pooled = self.pool(h).flatten(1)
        head_input = torch.cat([pooled, motif_norms], dim=1)
        logits = self.head(head_input).view(-1)

        # Diagnostics: routing entropy across motifs, averaged over
        # primary capsules. Detached so they are reportable without
        # biasing the loss.
        eps = 1e-12
        coupling_detached = coupling.detach()
        routing_entropy = -(coupling_detached * (coupling_detached + eps).log()).sum(dim=2)
        mean_routing_entropy = routing_entropy.mean(dim=1)
        max_motif_norm = motif_norms.detach().max(dim=1).values
        mean_motif_norm = motif_norms.detach().mean(dim=1)

        return {
            "logits": logits,
            "logit": logits,
            "prob": torch.sigmoid(logits),
            "latent": h,
            "primary_capsules": primary,
            "motif_capsules": v,
            "motif_norms": motif_norms,
            "routing_coupling": coupling,
            "routing_logits": routing_logits,
            "routing_entropy": mean_routing_entropy,
            "max_motif_norm": max_motif_norm,
            "mean_motif_norm": mean_motif_norm,
        }


def build_capsule_motif_boardnet_from_config(config: dict[str, Any]) -> CapsuleMotifBoardNet:
    cfg = dict(config)
    cfg.setdefault("input_channels", 18)
    cfg.setdefault("num_classes", 1)
    return CapsuleMotifBoardNet(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        num_primary_caps=int(cfg.get("num_primary_caps", 8)),
        primary_capsule_dim=int(cfg.get("primary_capsule_dim", 8)),
        num_motif_caps=int(cfg.get("num_motif_caps", 16)),
        motif_capsule_dim=int(cfg.get("motif_capsule_dim", 16)),
        routing_iterations=int(cfg.get("routing_iterations", 3)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
    )

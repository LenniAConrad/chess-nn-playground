"""Prototype Patch Dictionary Network for idea i117.

Working thesis (from
``ideas/registry/i117_prototype_patch_dictionary_network``): puzzle-like
positions may contain local motifs, but a standard CNN may hide them in
distributed filters. A learned patch dictionary can expose motif
assignments, reconstruction residuals, and prototype activation
histograms.

Concretely, this model:

1.  Embeds each board square together with its local neighborhood into
    a patch vector ``p_{b, s} in R^D`` via a small unfolded-conv
    "patch encoder". The neighborhood radius is set by the ``patch_kernel``
    config knob.
2.  Holds a learned dictionary ``D = [d_1, ..., d_K] in R^{K x D}`` of
    motif prototypes. Each row is L2-normalised before use so
    similarities are bounded.
3.  Computes per-square soft motif assignments by cosine similarity:
    ``alpha_{b, s, k} = softmax_k( <p_{b, s}, d_k> / (||p_{b,s}|| * tau) )``.
    ``tau`` is a learned positive temperature; ``alpha`` is the
    sparse-coding-style assignment map, shape ``(B, K, 8, 8)``.
4.  Reconstructs each patch as a convex combination of prototypes:
    ``p_hat_{b, s} = sum_k alpha_{b, s, k} * d_k``.
5.  Reads three readouts that are *exactly* the diagnostics the thesis
    calls out:
      - **Motif assignment map** -- the soft assignment ``alpha`` and
        its argmax (top-1 prototype id per square).
      - **Reconstruction residual** -- ``r_{b, s} = p_{b, s} - p_hat_{b,s}``
        and the per-square residual energy ``||r_{b, s}||^2``.
      - **Prototype activation histogram** -- spatial sum of
        ``alpha`` over the 8x8 grid, ``h_{b, k} = sum_s alpha_{b, s, k}``,
        normalised to a histogram so it sums to 1 across prototypes.
6.  The puzzle-binary classifier reads ``[h, residual_pooled,
    pooled_p, pooled_p_hat]`` (i.e. the histogram, the pooled residual
    energy per dictionary direction, and pooled patch / reconstruction
    statistics). It returns one logit plus the diagnostic outputs.

This is materially distinct from:

*   The shared ``ResearchPacketProbe`` scaffold: there is no learned
    dictionary, no soft-assignment map, no reconstruction residual,
    and no prototype histogram in that scaffold.
*   A vanilla CNN: removing the dictionary, or replacing the
    softmax-cosine assignment with a generic Conv2d, deletes the
    motif-assignment / residual / histogram diagnostics that the
    classifier consumes. The dictionary directions ``d_k`` are exposed
    parameters that participate in both the assignment soft-max and
    the reconstruction.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import (
    BoardTensorSpec,
    require_board_tensor,
)


class _PatchEncoder(nn.Module):
    """Embeds each square together with its local neighborhood into a
    patch vector of dimension ``patch_dim``.

    The encoder is a tiny convolutional stack that preserves the 8x8
    grid. The kernel size sets the neighborhood radius; ``patch_kernel=3``
    is the natural choice for "the square plus its 8 neighbors" motifs.
    """

    def __init__(
        self,
        input_channels: int,
        patch_dim: int,
        patch_kernel: int,
        depth: int,
        dropout: float,
        use_batchnorm: bool,
    ) -> None:
        super().__init__()
        if patch_dim < 1:
            raise ValueError("patch_dim must be >= 1")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if patch_kernel < 1 or patch_kernel % 2 == 0:
            raise ValueError("patch_kernel must be a positive odd integer")
        layers: list[nn.Module] = []
        in_ch = input_channels
        padding = patch_kernel // 2
        for layer_idx in range(depth):
            kernel = patch_kernel if layer_idx == 0 else 3
            pad = kernel // 2
            layers.append(
                nn.Conv2d(in_ch, patch_dim, kernel_size=kernel, padding=pad, bias=not use_batchnorm)
            )
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(patch_dim))
            else:
                layers.append(nn.GroupNorm(1, patch_dim))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout2d(dropout))
            in_ch = patch_dim
        self.body = nn.Sequential(*layers)
        # Suppress an unused-attribute warning for the explicit padding
        # value computed above; the actual padding lives inside the
        # individual Conv2d layers.
        self._neighborhood_padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class PrototypePatchDictionaryNetwork(nn.Module):
    """Bespoke patch-dictionary classifier for puzzle_binary.

    Holds a learned dictionary of ``num_prototypes`` motif prototypes,
    decomposes each board into per-square soft-assignment maps,
    reconstruction residuals, and a prototype activation histogram, and
    reads those diagnostics from the classifier head.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        num_prototypes: int = 32,
        patch_kernel: int = 3,
        height: int = 8,
        width: int = 8,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "PrototypePatchDictionaryNetwork supports the puzzle_binary one-logit contract"
            )
        if input_channels < 1:
            raise ValueError("input_channels must be >= 1")
        if num_prototypes < 2:
            raise ValueError("num_prototypes must be >= 2")
        self.spec = BoardTensorSpec(
            input_channels=int(input_channels), height=int(height), width=int(width)
        )
        self.input_channels = int(input_channels)
        self.height = int(height)
        self.width = int(width)
        self.patch_dim = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.dropout_p = float(dropout)
        self.num_prototypes = int(num_prototypes)
        self.patch_kernel = int(patch_kernel)

        self.patch_encoder = _PatchEncoder(
            input_channels=self.input_channels,
            patch_dim=self.patch_dim,
            patch_kernel=self.patch_kernel,
            depth=self.depth,
            dropout=self.dropout_p,
            use_batchnorm=bool(use_batchnorm),
        )

        # Learned dictionary D in R^{K x patch_dim}. We initialise it
        # with orthogonal vectors so the prototypes start well-separated.
        prototypes = torch.empty(self.num_prototypes, self.patch_dim)
        nn.init.orthogonal_(prototypes)
        self.prototypes = nn.Parameter(prototypes)

        # Learned positive temperature ``tau`` for the soft assignment.
        # We parameterise log_tau so tau stays positive without clamping.
        self.log_tau = nn.Parameter(torch.zeros(()))

        head_input_dim = self.num_prototypes + self.patch_dim + self.patch_dim + self.patch_dim
        head_layers: list[nn.Module] = [
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, self.hidden_dim),
            nn.GELU(),
        ]
        if self.dropout_p > 0.0:
            head_layers.append(nn.Dropout(self.dropout_p))
        head_layers.append(nn.Linear(self.hidden_dim, 1))
        self.classifier = nn.Sequential(*head_layers)

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_tau)

    def normalised_prototypes(self) -> torch.Tensor:
        return F.normalize(self.prototypes, dim=-1, eps=1.0e-8)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        batch = x.shape[0]
        cells = self.height * self.width

        # (B, D, 8, 8) -> (B, 64, D)
        patch_map = self.patch_encoder(x)
        patches = patch_map.permute(0, 2, 3, 1).reshape(batch, cells, self.patch_dim)

        # Cosine similarity between every patch and every prototype.
        patches_normed = F.normalize(patches, dim=-1, eps=1.0e-8)
        prototypes_normed = self.normalised_prototypes()
        # (B, S, K)
        similarity = torch.matmul(patches_normed, prototypes_normed.transpose(0, 1))
        tau = self.temperature
        soft_assignment = F.softmax(similarity / tau, dim=-1)

        # Reconstruction in the *unnormalised* patch space using the
        # original prototype directions.
        reconstruction = torch.matmul(soft_assignment, self.prototypes)
        residual = patches - reconstruction

        # Prototype activation histogram h_{b, k}: spatial mass of the
        # soft-assignment for prototype k on board b. We normalise to a
        # histogram (sums to 1 across prototypes).
        prototype_mass = soft_assignment.sum(dim=1)
        prototype_histogram = prototype_mass / float(cells)

        # Per-prototype residual energy: how much residual mass is
        # *attributed* to each prototype direction by the assignment.
        # (B, S, K) * (B, S, 1) -> (B, K)
        residual_norm_sq = residual.pow(2).sum(dim=-1, keepdim=True)
        residual_per_prototype = (soft_assignment * residual_norm_sq).sum(dim=1) / float(cells)

        residual_energy_per_square = residual.pow(2).sum(dim=-1)
        residual_energy = residual_energy_per_square.mean(dim=-1)

        pooled_patch = patches.mean(dim=1)
        pooled_reconstruction = reconstruction.mean(dim=1)
        pooled_residual = residual.abs().mean(dim=1)

        head_input = torch.cat(
            [prototype_histogram, pooled_residual, pooled_patch, pooled_reconstruction],
            dim=-1,
        )
        logits = self.classifier(head_input).view(-1)

        # Reshape soft_assignment back to a spatial (B, K, 8, 8) map for
        # visualisation. Argmax gives the top-1 motif id per square.
        assignment_map = soft_assignment.permute(0, 2, 1).reshape(
            batch, self.num_prototypes, self.height, self.width
        )
        top1_motif = assignment_map.argmax(dim=1)

        residual_map = residual.reshape(batch, self.height, self.width, self.patch_dim).permute(
            0, 3, 1, 2
        )
        reconstruction_map = reconstruction.reshape(
            batch, self.height, self.width, self.patch_dim
        ).permute(0, 3, 1, 2)

        # Prototype usage entropy across the board (per-batch). Higher
        # entropy = motifs are spread out, lower = a few prototypes
        # dominate. We compute it from the histogram so it is cheap.
        eps = 1.0e-8
        prototype_entropy = -(prototype_histogram * (prototype_histogram + eps).log()).sum(dim=-1)

        return {
            "logits": logits,
            "patch_map": patch_map,
            "patches": patches,
            "assignment_map": assignment_map,
            "soft_assignment": soft_assignment,
            "top1_motif_map": top1_motif,
            "reconstruction_map": reconstruction_map,
            "residual_map": residual_map,
            "prototype_histogram": prototype_histogram,
            "residual_per_prototype": residual_per_prototype,
            "residual_energy_per_square": residual_energy_per_square.reshape(
                batch, self.height, self.width
            ),
            "residual_energy": residual_energy,
            "prototype_entropy": prototype_entropy,
            "temperature": tau.expand(batch),
            "pooled_patch": pooled_patch,
            "pooled_reconstruction": pooled_reconstruction,
            "pooled_residual": pooled_residual,
        }


def build_prototype_patch_dictionary_network_from_config(
    config: dict[str, Any],
) -> PrototypePatchDictionaryNetwork:
    cfg = dict(config)
    return PrototypePatchDictionaryNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        num_prototypes=int(cfg.get("num_prototypes", 32)),
        patch_kernel=int(cfg.get("patch_kernel", 3)),
        height=int(cfg.get("height", 8)),
        width=int(cfg.get("width", 8)),
    )

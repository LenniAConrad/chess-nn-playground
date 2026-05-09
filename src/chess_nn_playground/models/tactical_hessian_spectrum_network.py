"""Tactical Hessian Spectrum Network for idea i199.

Working thesis: a real puzzle is a sharp local maximum of tactical
evidence under small legal perturbations of the board, while a near-
puzzle has high raw evidence but flatter or less stable local
geometry. We turn that thesis into an explicit differentiable
computation: build a scalar tactical-evidence field ``E(x)`` over the
board, probe its second-order behaviour along ``K`` chess-meaningful
perturbation directions ``D_1, ..., D_K`` via finite differences,
assemble a small symmetric ``K x K`` reduced Hessian, and read its
spectrum into the puzzle classifier.

Mechanism:

1. **Compact convolutional trunk.** ``feats = trunk(x)`` produces a
   ``(B, channels, 8, 8)`` feature map.
2. **Tactical evidence field.** A ``1x1`` projection over ``feats``
   yields a per-square scalar ``S(x) in R^(B, 8, 8)``. The scalar
   evidence is ``E(x) = sum_{r, f} S(x)[r, f]``.
3. **Perturbation directions.** ``K`` board-shaped tensors
   ``D_k in R^(input_channels, 8, 8)`` defined by chess-aware patterns:
   a piece-color flip, a side-to-move flip, a one-rank vertical shift,
   a one-file horizontal shift, plus learnable refinements. Each ``D_k``
   has unit Frobenius norm so the finite-difference probes are scale
   matched.
4. **Reduced Hessian by finite differences.** With step ``eps``,
   we evaluate ``E`` on the board ``x`` plus the small set
   ``{x + eps D_k, x - eps D_k, x + eps (D_i + D_j) for i < j}``.
   These are stacked along the batch and run through the encoder once
   so the cost is one large forward. From the variant evidence
   values we form
   ``g_k = (E(x + eps D_k) - E(x - eps D_k)) / (2 eps)``,
   ``H_kk = (E(x + eps D_k) + E(x - eps D_k) - 2 E(x)) / eps^2``,
   ``H_ij = (E(x + eps (D_i + D_j)) - E(x + eps D_i) - E(x + eps D_j) + E(x)) / eps^2``,
   then symmetrise to obtain a real ``K x K`` matrix.
5. **Spectral readout.** ``torch.linalg.eigvalsh`` produces sorted
   real eigenvalues. The classifier reads the eigenvalue vector plus
   sharpness scalars: top eigenvalue, smallest eigenvalue, spectral
   gap, trace, sum of positive eigenvalues, sum of negative
   eigenvalues (concavity), spectral radius, gradient norm
   ``||g||``, evidence value ``E(x)``, and pooled trunk features.
6. **Head.** A small MLP turns this readout into a single puzzle
   logit. Sharp negative curvature (``concavity`` large in
   magnitude) pushes the position toward the puzzle class; flat or
   indefinite curvature pushes it toward non-puzzle.

The model is strictly board-only and never reads engine, source,
verification, or CRTK metadata.

Tensor contract (``input_channels = 18``):

* input ``x``                      shape ``(B, 18, 8, 8)``
* trunk feats                      shape ``(B, channels, 8, 8)``
* evidence field ``S(x)``          shape ``(B, 8, 8)``
* reduced Hessian                  shape ``(B, K, K)``
* eigenvalues                      shape ``(B, K)``
* puzzle ``logits``                shape ``(B,)``
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


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.view(-1) if num_classes == 1 else logits


def _build_chess_perturbation_basis(input_channels: int) -> torch.Tensor:
    """Return ``(4, input_channels, 8, 8)`` deterministic chess-meaningful
    perturbation directions.

    Each direction is unit-Frobenius normalised so finite-difference
    probes share a common scale.
    """
    directions: list[torch.Tensor] = []
    if input_channels >= 12:
        # Color-flip: pushes mass between white and black piece planes.
        flip = torch.zeros(input_channels, 8, 8)
        flip[0:6] = 1.0
        flip[6:12] = -1.0
        directions.append(flip)
    if input_channels >= 13:
        # Side-to-move toggle.
        stm = torch.zeros(input_channels, 8, 8)
        stm[12] = 1.0
        directions.append(stm)
    # Rank shift: positive on top half, negative on bottom half.
    rank = torch.zeros(input_channels, 8, 8)
    band = max(1, min(6, input_channels))
    for r in range(8):
        rank[:band, r, :] = 1.0 if r < 4 else -1.0
    directions.append(rank)
    # File shift: positive on king-side files, negative on queen-side.
    file_dir = torch.zeros(input_channels, 8, 8)
    for f in range(8):
        file_dir[:band, :, f] = 1.0 if f >= 4 else -1.0
    directions.append(file_dir)
    if not directions:
        directions.append(torch.eye(8).expand(input_channels, 8, 8).contiguous())

    stacked = torch.stack(directions[:4], dim=0)
    norms = stacked.flatten(1).norm(dim=-1).clamp_min(1.0e-6)
    return stacked / norms.view(-1, 1, 1, 1)


class _EvidenceTrunk(nn.Module):
    """Compact convolutional trunk plus per-square evidence head."""

    def __init__(
        self,
        input_channels: int,
        channels: int,
        depth: int,
        dropout: float,
        use_batchnorm: bool,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if channels < 1:
            raise ValueError("channels must be >= 1")
        layers: list[nn.Module] = []
        in_ch = input_channels
        for _ in range(depth):
            layers.append(
                nn.Conv2d(in_ch, channels, kernel_size=3, padding=1, bias=not use_batchnorm)
            )
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(channels))
            else:
                layers.append(nn.GroupNorm(1, channels))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout2d(dropout))
            in_ch = channels
        self.body = nn.Sequential(*layers)
        self.evidence_proj = nn.Conv2d(channels, 1, kernel_size=1)
        self.output_channels = channels

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)

    def evidence_field(self, feats: torch.Tensor) -> torch.Tensor:
        # (B, 1, 8, 8) -> (B, 8, 8)
        return self.evidence_proj(feats).squeeze(1)


class TacticalHessianSpectrumNetwork(nn.Module):
    """Bespoke implementation of idea i199.

    Forward output dict (board-only inputs):

    * ``logits``: ``(B,)`` puzzle logit for the BCE-with-logits trainer
      (``(B, num_classes)`` if ``num_classes > 1``).
    * ``prob``: ``sigmoid(logits)`` when ``num_classes == 1``.
    * ``evidence_field``: ``(B, 8, 8)`` per-square tactical evidence.
    * ``evidence_total``: ``(B,)`` scalar evidence ``E(x)``.
    * ``perturbation_directions``: ``(K, input_channels, 8, 8)`` unit
      directions actually used for the finite-difference probe (after
      learnable scaling has been applied; see ``direction_log_scales``).
    * ``directional_gradient``: ``(B, K)`` first-order responses
      ``g_k = (E(x + eps d_k) - E(x - eps d_k)) / (2 eps)``.
    * ``hessian``: ``(B, K, K)`` symmetric reduced Hessian.
    * ``hessian_eigenvalues``: ``(B, K)`` eigenvalues sorted ascending.
    * ``top_eigenvalue``, ``min_eigenvalue``: ``(B,)`` largest /
      smallest eigenvalue.
    * ``spectral_gap``: ``(B,)`` ``lambda_max - second_max``.
    * ``trace``: ``(B,)`` Hessian trace.
    * ``concavity``: ``(B,)`` sum of negative eigenvalues (negated for
      sharpness): ``-sum(lambda_i where lambda_i < 0)``.
    * ``positive_curvature``: ``(B,)`` sum of positive eigenvalues.
    * ``spectral_radius``: ``(B,)`` ``max(|lambda_i|)``.
    * ``gradient_norm``: ``(B,)`` ``||g||_2``.
    * ``trunk_energy``: ``(B,)`` mean-square trunk activation.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        eps: float = 0.5,
        num_directions: int = 4,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        height: int = 8,
        width: int = 8,
        **_: Any,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        if num_directions < 2:
            raise ValueError("num_directions must be >= 2 to define a non-degenerate Hessian")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.spec = BoardTensorSpec(
            input_channels=int(input_channels), height=int(height), width=int(width)
        )
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.eps = float(eps)
        self.num_directions = int(num_directions)
        self.dropout_p = float(dropout)
        self.use_batchnorm = bool(use_batchnorm)

        self.trunk = _EvidenceTrunk(
            input_channels=self.input_channels,
            channels=self.channels,
            depth=self.depth,
            dropout=self.dropout_p,
            use_batchnorm=self.use_batchnorm,
        )

        # Deterministic chess-meaningful directions, unit Frobenius norm.
        base = _build_chess_perturbation_basis(self.input_channels)
        if base.shape[0] < self.num_directions:
            # Pad with axis-aligned directions (rank/file stripes shifted).
            extra: list[torch.Tensor] = []
            for k in range(self.num_directions - base.shape[0]):
                d = torch.zeros(self.input_channels, 8, 8)
                rank = k % 8
                d[: max(1, min(6, self.input_channels)), rank, :] = 1.0
                d = d / d.norm().clamp_min(1.0e-6)
                extra.append(d)
            base = torch.cat([base, torch.stack(extra, dim=0)], dim=0)
        base = base[: self.num_directions]
        self.register_buffer("_perturbation_basis", base, persistent=False)

        # Learnable per-direction log-scale so the model can attenuate
        # or amplify each chess-meaningful probe; initialised to zero
        # (multiplier = 1) so the basis is preserved at init.
        self.direction_log_scales = nn.Parameter(torch.zeros(self.num_directions))

        # Head feature pack:
        #   eigenvalues (K)
        #   top, min, spectral_gap, trace, concavity, positive_curvature, spectral_radius (7)
        #   evidence_total, gradient_norm (2)
        #   trunk_mean, trunk_max, trunk_energy (3)
        head_in = self.num_directions + 7 + 2 + 3
        self.head_norm = nn.LayerNorm(head_in)
        head_layers: list[nn.Module] = [
            nn.Linear(head_in, self.hidden_dim),
            nn.GELU(),
        ]
        if self.dropout_p > 0.0:
            head_layers.append(nn.Dropout(self.dropout_p))
        head_layers.append(nn.Linear(self.hidden_dim, self.num_classes))
        self.head = nn.Sequential(*head_layers)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def perturbation_directions(self) -> torch.Tensor:
        scales = torch.exp(self.direction_log_scales).view(-1, 1, 1, 1)
        scaled = self._perturbation_basis * scales
        norms = scaled.flatten(1).norm(dim=-1).clamp_min(1.0e-6)
        return scaled / norms.view(-1, 1, 1, 1)

    def _evidence_for(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (trunk feats, evidence field, evidence scalar)."""
        feats = self.trunk.trunk(x)
        field = self.trunk.evidence_field(feats)
        scalar = field.flatten(1).sum(dim=-1)
        return feats, field, scalar

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        batch = x.shape[0]
        K = self.num_directions
        eps = self.eps

        directions = self.perturbation_directions()  # (K, C, 8, 8)

        # Stack the variant set so we run the trunk once over a single
        # batched tensor: [base, +d_k, -d_k, +d_i + d_j (i<j)].
        pair_indices = [(i, j) for i in range(K) for j in range(i + 1, K)]
        num_pairs = len(pair_indices)
        variants_per_sample = 1 + 2 * K + num_pairs

        # base
        chunks = [x.unsqueeze(1)]  # (B, 1, C, 8, 8)
        # plus / minus single-direction probes
        d_plus = (x.unsqueeze(1) + eps * directions.unsqueeze(0))  # (B, K, C, 8, 8)
        d_minus = (x.unsqueeze(1) - eps * directions.unsqueeze(0))
        chunks.append(d_plus)
        chunks.append(d_minus)
        # pair probes
        if num_pairs > 0:
            pair_dirs = torch.stack(
                [directions[i] + directions[j] for (i, j) in pair_indices], dim=0
            )  # (P, C, 8, 8)
            d_pair = x.unsqueeze(1) + eps * pair_dirs.unsqueeze(0)  # (B, P, C, 8, 8)
            chunks.append(d_pair)
        variants = torch.cat(chunks, dim=1)  # (B, V, C, 8, 8)
        flat = variants.reshape(batch * variants_per_sample, *x.shape[1:])

        feats_flat, field_flat, scalar_flat = self._evidence_for(flat)
        scalar = scalar_flat.view(batch, variants_per_sample)

        E_base = scalar[:, 0]
        E_plus = scalar[:, 1 : 1 + K]
        E_minus = scalar[:, 1 + K : 1 + 2 * K]
        if num_pairs > 0:
            E_pair = scalar[:, 1 + 2 * K :]
        else:
            E_pair = scalar.new_zeros(batch, 0)

        # Directional first derivatives via central differences.
        directional_gradient = (E_plus - E_minus) / (2.0 * eps)
        gradient_norm = directional_gradient.norm(dim=-1)

        # Diagonal Hessian entries.
        diag = (E_plus + E_minus - 2.0 * E_base.unsqueeze(-1)) / (eps * eps)

        # Off-diagonal Hessian entries from forward-difference pair probes.
        H = scalar.new_zeros(batch, K, K)
        diag_idx = torch.arange(K, device=H.device)
        H[:, diag_idx, diag_idx] = diag
        if num_pairs > 0:
            for p, (i, j) in enumerate(pair_indices):
                off = (
                    E_pair[:, p]
                    - E_plus[:, i]
                    - E_plus[:, j]
                    + E_base
                ) / (eps * eps)
                H[:, i, j] = off
                H[:, j, i] = off
        H = 0.5 * (H + H.transpose(1, 2))

        # Spectrum.
        eigenvalues = torch.linalg.eigvalsh(H)  # ascending order, real

        top_eig = eigenvalues[:, -1]
        min_eig = eigenvalues[:, 0]
        if K >= 2:
            spectral_gap = eigenvalues[:, -1] - eigenvalues[:, -2]
        else:
            spectral_gap = scalar.new_zeros(batch)
        trace = eigenvalues.sum(dim=-1)
        positive_curvature = eigenvalues.clamp_min(0.0).sum(dim=-1)
        # ``concavity`` = sharpness of negative curvature, sign-flipped
        # so it is a non-negative measure of how concave the local
        # tactical-evidence surface is.
        concavity = (-eigenvalues.clamp_max(0.0)).sum(dim=-1)
        spectral_radius = eigenvalues.abs().amax(dim=-1)

        # Pooled trunk features come from the un-perturbed trunk pass,
        # which lives at slot 0 of the variant axis.
        feats = feats_flat.view(batch, variants_per_sample, self.channels, 8, 8)[:, 0]
        trunk_mean = feats.mean(dim=(1, 2, 3))
        trunk_max = feats.amax(dim=(2, 3)).mean(dim=1)
        trunk_energy = feats.square().mean(dim=(1, 2, 3))

        head_input = torch.cat(
            [
                eigenvalues,
                top_eig.unsqueeze(-1),
                min_eig.unsqueeze(-1),
                spectral_gap.unsqueeze(-1),
                trace.unsqueeze(-1),
                concavity.unsqueeze(-1),
                positive_curvature.unsqueeze(-1),
                spectral_radius.unsqueeze(-1),
                E_base.unsqueeze(-1),
                gradient_norm.unsqueeze(-1),
                trunk_mean.unsqueeze(-1),
                trunk_max.unsqueeze(-1),
                trunk_energy.unsqueeze(-1),
            ],
            dim=-1,
        )
        head_input = self.head_norm(head_input)
        raw_logits = self.head(head_input)
        logits = _format_logits(raw_logits, self.num_classes)

        # Recover the un-perturbed evidence field for diagnostics.
        evidence_field = field_flat.view(batch, variants_per_sample, 8, 8)[:, 0]

        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "evidence_field": evidence_field,
            "evidence_total": E_base,
            "perturbation_directions": directions,
            "directional_gradient": directional_gradient,
            "hessian": H,
            "hessian_eigenvalues": eigenvalues,
            "top_eigenvalue": top_eig,
            "min_eigenvalue": min_eig,
            "spectral_gap": spectral_gap,
            "trace": trace,
            "concavity": concavity,
            "positive_curvature": positive_curvature,
            "spectral_radius": spectral_radius,
            "gradient_norm": gradient_norm,
            "trunk_energy": trunk_energy,
        }
        if self.num_classes == 1:
            output["prob"] = torch.sigmoid(logits)
        return output


def build_tactical_hessian_spectrum_network_from_config(
    config: dict[str, Any],
) -> TacticalHessianSpectrumNetwork:
    cfg = dict(config)
    cfg.pop("name", None)
    cfg.pop("mechanism_family", None)
    cfg.pop("packet_profile", None)
    return TacticalHessianSpectrumNetwork(
        input_channels=int(cfg.pop("input_channels", 18)),
        num_classes=int(cfg.pop("num_classes", 1)),
        channels=int(cfg.pop("channels", 64)),
        hidden_dim=int(cfg.pop("hidden_dim", 96)),
        depth=int(cfg.pop("depth", 2)),
        eps=float(cfg.pop("eps", 0.5)),
        num_directions=int(cfg.pop("num_directions", 4)),
        dropout=float(cfg.pop("dropout", 0.1)),
        use_batchnorm=bool(cfg.pop("use_batchnorm", True)),
        height=int(cfg.pop("height", 8)),
        width=int(cfg.pop("width", 8)),
    )


__all__ = [
    "TacticalHessianSpectrumNetwork",
    "build_tactical_hessian_spectrum_network_from_config",
]

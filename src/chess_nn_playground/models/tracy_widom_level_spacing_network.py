"""Tracy-Widom Level-Spacing Network for idea i233.

Builds a learned chess Hermitian operator ``H = (M + M^T) / 2`` of size
``hermitian_n``, computes its eigenvalues, unfolds the spectrum via a
local 5-point smoothing of the cumulative spectral count, and reads off
the random-matrix-theory invariants that distinguish quantum-chaotic
(Wigner-Dyson) from integrable (Poisson) regimes:

- mean nearest-neighbour level-spacing ratio ``<r>``
- soft spacing histogram (8 bins)
- spectral form factor ``K(t_k) = |sum_i exp(2 pi i tilde_lambda_i t_k)|^2``
- per-spacing log-likelihoods under the Poisson, GOE, and GUE level
  distributions, summed and softmaxed into a 3-way regime score

The puzzle head consumes a pooled board representation together with
those bulk-spectrum invariants. The mechanism is materially distinct
from any spectrum-position baseline (i062, i076, i077, i199, i228)
which use eigenvalues themselves rather than their *correlations*.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


class _Trunk(nn.Module):
    def __init__(
        self,
        input_channels: int,
        channels: int,
        depth: int,
        dropout: float,
        use_batchnorm: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = input_channels
        for _ in range(max(1, depth)):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    channels,
                    kernel_size=3,
                    padding=1,
                    bias=not use_batchnorm,
                )
            )
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            in_channels = channels
        self.trunk = nn.Sequential(*layers)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


def _wigner_dyson_log_pdf(s: torch.Tensor, beta: int) -> torch.Tensor:
    """Log of the Wigner-Dyson surmise for symmetry index ``beta in {1, 2}``.

    GOE (beta = 1): ``P(s) = (pi / 2) s exp(-pi s^2 / 4)``.
    GUE (beta = 2): ``P(s) = (32 / pi^2) s^2 exp(-4 s^2 / pi)``.
    """
    safe_s = s.clamp_min(1.0e-8)
    if beta == 1:
        log_norm = math.log(math.pi / 2.0)
        return log_norm + safe_s.log() - (math.pi / 4.0) * safe_s * safe_s
    if beta == 2:
        log_norm = math.log(32.0 / (math.pi * math.pi))
        return log_norm + 2.0 * safe_s.log() - (4.0 / math.pi) * safe_s * safe_s
    raise ValueError(f"Unsupported Wigner-Dyson beta={beta}")


class TracyWidomLevelSpacingNetwork(nn.Module):
    """Bespoke implementation of idea i233.

    The model is intentionally board-only; CRTK / source metadata is
    reporting-only and never consumed as input.
    """

    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        hermitian_n: int = 64,
        embedding_dim: int = 32,
        spacing_histogram_bins: int = 8,
        num_form_factor_taps: int = 16,
        unfolding_window: int = 5,
        form_factor_t_min: float = 0.05,
        form_factor_t_max: float = 1.0,
        spacing_histogram_max: float = 4.0,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "TracyWidomLevelSpacingNetwork supports the puzzle_binary one-logit contract"
            )
        if hermitian_n != 64:
            raise ValueError(
                "TracyWidomLevelSpacingNetwork operates on the 8x8 board: hermitian_n must be 64"
            )
        if spacing_histogram_bins < 2:
            raise ValueError("spacing_histogram_bins must be >= 2")
        if num_form_factor_taps < 1:
            raise ValueError("num_form_factor_taps must be >= 1")
        if unfolding_window < 1 or unfolding_window % 2 == 0:
            raise ValueError("unfolding_window must be a positive odd integer")
        if form_factor_t_max <= form_factor_t_min:
            raise ValueError("form_factor_t_max must be > form_factor_t_min")
        if spacing_histogram_max <= 0:
            raise ValueError("spacing_histogram_max must be > 0")

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.channels = int(channels)
        self.hermitian_n = int(hermitian_n)
        self.embedding_dim = int(embedding_dim)
        self.spacing_histogram_bins = int(spacing_histogram_bins)
        self.num_form_factor_taps = int(num_form_factor_taps)
        self.unfolding_window = int(unfolding_window)
        self.spacing_histogram_max = float(spacing_histogram_max)

        self.trunk = _Trunk(input_channels, channels, depth, dropout, use_batchnorm)
        # Two heads project per-square trunk features to "left" and "right"
        # operator embeddings; the asymmetric outer product is symmetrised
        # into the Hermitian operator H.
        self.proj_left = nn.Conv2d(channels, embedding_dim, kernel_size=1)
        self.proj_right = nn.Conv2d(channels, embedding_dim, kernel_size=1)

        # Soft histogram bin centres in unfolded-spacing units.
        bin_centres = torch.linspace(0.0, spacing_histogram_max, self.spacing_histogram_bins)
        bin_width = float(bin_centres[1] - bin_centres[0]) if self.spacing_histogram_bins > 1 else 1.0
        self.register_buffer("hist_centres", bin_centres, persistent=False)
        self.hist_bandwidth = max(bin_width * 0.5, 1.0e-3)

        # Form factor sample taps t_k.
        taps = torch.linspace(form_factor_t_min, form_factor_t_max, self.num_form_factor_taps)
        self.register_buffer("form_factor_taps", taps, persistent=False)

        # 5-point smoothing kernel (uniform local average), applied to the
        # cumulative empirical staircase to produce the unfolded spectrum.
        kernel = torch.ones(self.unfolding_window, dtype=torch.float32) / float(
            self.unfolding_window
        )
        self.register_buffer("unfolding_kernel", kernel, persistent=False)

        pooled_trunk_dim = 2 * channels
        # head input: pooled trunk + spacing histogram + mean-r + form-factor + 3 logliks + 3-way softmax
        head_in = (
            pooled_trunk_dim
            + self.spacing_histogram_bins
            + 1
            + self.num_form_factor_taps
            + 3
            + 3
        )
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def _build_hermitian(self, feat: torch.Tensor) -> torch.Tensor:
        """Project the trunk feature map to a Hermitian 64x64 operator."""
        bsz = feat.shape[0]
        left = self.proj_left(feat).reshape(bsz, self.embedding_dim, -1).transpose(1, 2)
        right = self.proj_right(feat).reshape(bsz, self.embedding_dim, -1).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.embedding_dim)
        m = torch.matmul(left, right.transpose(-1, -2)) * scale
        h = 0.5 * (m + m.transpose(-1, -2))
        return h

    def _unfold_eigvals(self, eigvals: torch.Tensor) -> torch.Tensor:
        """Smooth the empirical staircase ``N(lambda_i) = i + 1`` with a
        ``unfolding_window``-point boxcar; the result is the unfolded
        spectrum ``tilde_lambda``.

        For a Wigner-like operator the local mean spacing is
        ``rho(lambda)^{-1}``; convolving the trivial staircase with a
        local average and then renormalising to unit total range gives
        a smooth, differentiable proxy.
        """
        bsz, n = eigvals.shape
        # Empirical staircase normalised to [0, n-1] (the trivial unfolding).
        staircase = torch.arange(n, dtype=eigvals.dtype, device=eigvals.device)
        staircase = staircase.unsqueeze(0).expand(bsz, -1).contiguous()
        kernel = self.unfolding_kernel.view(1, 1, -1)
        pad = self.unfolding_window // 2
        # Convolution of the staircase weighted by raw eigvalue spacings
        # gives a smoothed unfolded position. Concretely we treat
        # ``eigvals - mean(eigvals)`` as a perturbation to the trivial
        # staircase; smoothing both yields a stable unfolding even when
        # the operator has only mild structure.
        centred = eigvals - eigvals.mean(dim=-1, keepdim=True)
        smoothed_eigvals = F.conv1d(
            F.pad(centred.unsqueeze(1), (pad, pad), mode="replicate"),
            kernel,
        ).squeeze(1)
        # Combine: rank-based staircase + smoothed eigvalue offset, then
        # rescale so unfolded spacings have mean 1.
        unfolded = staircase + smoothed_eigvals
        spacings = unfolded[:, 1:] - unfolded[:, :-1]
        mean_spacing = spacings.mean(dim=-1, keepdim=True).clamp_min(1.0e-6)
        unfolded = unfolded / mean_spacing
        return unfolded

    def _spacing_ratios(self, spacings: torch.Tensor) -> torch.Tensor:
        s_curr = spacings[:, 1:]
        s_prev = spacings[:, :-1]
        eps = 1.0e-8
        num = torch.minimum(s_curr, s_prev)
        den = torch.maximum(s_curr, s_prev).clamp_min(eps)
        return num / den

    def _soft_histogram(self, spacings: torch.Tensor) -> torch.Tensor:
        # Soft (RBF) histogram so the gradient flows through the head.
        s = spacings.unsqueeze(-1)  # (B, M, 1)
        centres = self.hist_centres.view(1, 1, -1)
        weights = torch.exp(-0.5 * ((s - centres) / self.hist_bandwidth) ** 2)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        return weights.mean(dim=1)  # (B, bins)

    def _form_factor(self, unfolded: torch.Tensor) -> torch.Tensor:
        # K(t) = | sum_i exp(2 pi i tilde_lambda_i t) |^2
        taps = self.form_factor_taps.view(1, 1, -1)
        phases = 2.0 * math.pi * unfolded.unsqueeze(-1) * taps  # (B, n, K)
        cos_sum = phases.cos().sum(dim=1)
        sin_sum = phases.sin().sum(dim=1)
        return cos_sum * cos_sum + sin_sum * sin_sum

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feat = self.trunk(x)  # (B, C, 8, 8)

        h = self._build_hermitian(feat)  # (B, 64, 64)
        eigvals = torch.linalg.eigvalsh(h)  # ascending

        unfolded = self._unfold_eigvals(eigvals)
        spacings = unfolded[:, 1:] - unfolded[:, :-1]
        # Negative spacings should not arise after sorting + boxcar smoothing
        # but we clamp to keep log-likelihoods finite.
        spacings = spacings.clamp_min(1.0e-6)

        ratios = self._spacing_ratios(spacings)
        mean_ratio = ratios.mean(dim=-1)

        histogram = self._soft_histogram(spacings)
        form_factor = self._form_factor(unfolded)
        # Normalise the form factor by N so the values are O(1) and
        # comparable across batch elements.
        form_factor = form_factor / float(self.hermitian_n)

        # Per-spacing log-likelihoods under the three reference regimes.
        log_p_poisson = -spacings  # log e^{-s} = -s
        log_p_goe = _wigner_dyson_log_pdf(spacings, beta=1)
        log_p_gue = _wigner_dyson_log_pdf(spacings, beta=2)
        nu_poisson = log_p_poisson.sum(dim=-1)
        nu_goe = log_p_goe.sum(dim=-1)
        nu_gue = log_p_gue.sum(dim=-1)
        regime_logits = torch.stack([nu_poisson, nu_goe, nu_gue], dim=-1)
        regime_softmax = torch.softmax(regime_logits, dim=-1)

        pooled_trunk = torch.cat([feat.mean(dim=(2, 3)), feat.amax(dim=(2, 3))], dim=1)

        feat_vec = torch.cat(
            [
                pooled_trunk,
                histogram,
                mean_ratio.unsqueeze(-1),
                form_factor,
                regime_logits,
                regime_softmax,
            ],
            dim=-1,
        )
        logits = self.head(feat_vec).view(-1)

        return {
            "logits": logits,
            "tracy_widom_mean_spacing_ratio": mean_ratio,
            "tracy_widom_spacing_histogram": histogram,
            "tracy_widom_spectral_form_factor": form_factor,
            "tracy_widom_regime_softmax": regime_softmax,
            "tracy_widom_poisson_loglik": nu_poisson,
            "tracy_widom_goe_loglik": nu_goe,
            "tracy_widom_gue_loglik": nu_gue,
        }


def build_tracy_widom_level_spacing_network_from_config(
    config: dict[str, Any],
) -> TracyWidomLevelSpacingNetwork:
    cfg = dict(config)
    return TracyWidomLevelSpacingNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        hermitian_n=int(cfg.get("hermitian_n", 64)),
        embedding_dim=int(cfg.get("embedding_dim", 32)),
        spacing_histogram_bins=int(cfg.get("spacing_histogram_bins", 8)),
        num_form_factor_taps=int(cfg.get("num_form_factor_taps", 16)),
        unfolding_window=int(cfg.get("unfolding_window", 5)),
        form_factor_t_min=float(cfg.get("form_factor_t_min", 0.05)),
        form_factor_t_max=float(cfg.get("form_factor_t_max", 1.0)),
        spacing_histogram_max=float(cfg.get("spacing_histogram_max", 4.0)),
        num_classes=int(cfg.get("num_classes", 1)),
    )

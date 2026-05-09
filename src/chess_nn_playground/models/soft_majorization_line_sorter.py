"""Soft Majorization Line Sorter for idea i139.

The model learns ``K`` scalar salience fields over the 8x8 board, gathers
their values along chess lines (ranks, files, diagonals, anti-diagonals),
soft-sorts each line descending, and reads tactical content from
majorization-style descriptors of the sorted profiles: top values,
adjacent gaps, top-j concentration ratios, and per-line entropy of the
soft-sorted scores.

Mechanism: instead of pooling the salience field into a bag-of-line
statistic (sum/mean/max) or routing tokens with attention, the model
classifies from how *uneven* and *front-loaded* each line is once its
scalar profile is sorted.  The soft-sort of Prillo & Eisenschlos
(`https://arxiv.org/abs/2006.16038`) is used so the ordering itself is
differentiable: gradients flow from majorization descriptors back through
both the ranks and the salience heads.

The model is intentionally board-only.  CRTK / source / engine metadata
is reporting-only and never consumed as input.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


_LINE_TYPE_NAMES: tuple[str, ...] = ("rank", "file", "diagonal", "anti_diagonal")


def _build_line_index(board_size: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build flat-square indices for every chess line.

    Returns
    -------
    flat_index : LongTensor of shape ``(num_lines, board_size)``
        Square indices (in the flat 64-square layout) for each line.  Each
        line is padded out to length ``board_size`` by repeating the last
        valid square; padded slots are masked out by ``valid_mask``.
    valid_mask : BoolTensor of shape ``(num_lines, board_size)``
        ``True`` for real line squares, ``False`` for padding.
    line_type : LongTensor of shape ``(num_lines,)``
        0=rank, 1=file, 2=diagonal, 3=anti-diagonal.
    """
    bs = int(board_size)
    lines: list[list[int]] = []
    types: list[int] = []
    # Ranks (rows).
    for r in range(bs):
        line = [r * bs + c for c in range(bs)]
        lines.append(line)
        types.append(0)
    # Files (columns).
    for c in range(bs):
        line = [r * bs + c for r in range(bs)]
        lines.append(line)
        types.append(1)
    # Diagonals (constant r - c).  Lengths 1..bs..1.
    for d in range(-(bs - 1), bs):
        line = [r * bs + (r - d) for r in range(bs) if 0 <= r - d < bs]
        lines.append(line)
        types.append(2)
    # Anti-diagonals (constant r + c).  Lengths 1..bs..1.
    for d in range(2 * bs - 1):
        line = [r * bs + (d - r) for r in range(bs) if 0 <= d - r < bs]
        lines.append(line)
        types.append(3)

    num_lines = len(lines)
    flat_index = torch.zeros(num_lines, bs, dtype=torch.long)
    valid_mask = torch.zeros(num_lines, bs, dtype=torch.bool)
    for li, sq in enumerate(lines):
        L = len(sq)
        for j in range(bs):
            flat_index[li, j] = sq[j] if j < L else sq[L - 1]
            valid_mask[li, j] = j < L
    line_type = torch.tensor(types, dtype=torch.long)
    return flat_index, valid_mask, line_type


def _soft_sort_descending(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Differentiable descending sort along the last dimension.

    Implements the SoftSort operator (Prillo & Eisenschlos, 2020):

    ``P = softmax(-|sort(s) - s| / tau)``  with the hard sort acting as a
    no-grad reference.  We then return ``P @ s`` so gradients flow back
    through ``s`` while the ordering itself is differentiable in tau.

    ``valid_mask`` (``True`` for real entries) suppresses padded slots so
    they sink to the end of the descending sort.
    """
    masked = values.masked_fill(~valid_mask, torch.finfo(values.dtype).min)
    hard_sorted, _ = torch.sort(masked, dim=-1, descending=True)
    # Keep the hard-sorted reference free of gradient (SoftSort uses the
    # *positions* of the hard sort, not its values, as anchors).
    hard_ref = hard_sorted.detach()
    diff = (hard_ref.unsqueeze(-1) - masked.unsqueeze(-2)).abs()
    perm = torch.softmax(-diff / max(tau, 1.0e-6), dim=-1)  # (..., L, L)
    soft_sorted = torch.matmul(perm, masked.unsqueeze(-1)).squeeze(-1)
    # After sorting descending, padded entries collapse to the right; zero
    # them so they cannot leak into majorization sums.
    valid_lengths = valid_mask.sum(dim=-1, keepdim=True)
    arange = torch.arange(values.shape[-1], device=values.device)
    sort_mask = arange < valid_lengths  # broadcasts against trailing dim
    soft_sorted = soft_sorted * sort_mask.to(soft_sorted.dtype)
    return soft_sorted, sort_mask


class _SalienceTrunk(nn.Module):
    """Compact CNN that lifts the board tensor to ``K`` salience fields.

    The trunk also exposes a pooled board-context vector that the head
    consumes alongside the per-line majorization descriptors.
    """

    def __init__(
        self,
        input_channels: int,
        channels: int,
        depth: int,
        num_salience_heads: int,
        dropout: float,
        use_batchnorm: bool,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if channels < num_salience_heads:
            raise ValueError("channels must be >= num_salience_heads")

        layers: list[nn.Module] = []
        in_channels = input_channels
        for _ in range(depth):
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
        self.salience_proj = nn.Conv2d(channels, num_salience_heads, kernel_size=1)
        self.context_dim = 2 * channels

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.trunk(x)  # (B, C, 8, 8)
        salience = self.salience_proj(feat)  # (B, K, 8, 8)
        context = torch.cat([feat.mean(dim=(2, 3)), feat.amax(dim=(2, 3))], dim=1)
        return salience, context


class SoftMajorizationLineSorter(nn.Module):
    """Bespoke implementation of idea i139.

    The model produces ``K`` scalar salience fields, gathers them along
    every chess line, soft-sorts each line descending with temperature
    ``tau``, and reads out a fixed-dimensional set of majorization
    descriptors per (salience-head, line-type) bucket which are pooled
    (mean + max over lines in the bucket) and fed to a small MLP head.
    """

    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        num_salience_heads: int = 5,
        sort_temperature: float = 0.5,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "SoftMajorizationLineSorter supports the puzzle_binary one-logit contract"
            )
        if num_salience_heads < 1:
            raise ValueError("num_salience_heads must be >= 1")
        if sort_temperature <= 0:
            raise ValueError("sort_temperature must be positive")
        if input_channels < 12:
            raise ValueError("SoftMajorizationLineSorter requires input_channels >= 12")

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_salience_heads = int(num_salience_heads)
        self.sort_temperature = float(sort_temperature)
        self.board_size = 8

        flat_index, valid_mask, line_type = _build_line_index(self.board_size)
        # Persistent buffers so the line geometry is shipped with the model.
        self.register_buffer("line_flat_index", flat_index, persistent=False)
        self.register_buffer("line_valid_mask", valid_mask, persistent=False)
        self.register_buffer("line_type", line_type, persistent=False)
        self.num_lines = int(flat_index.shape[0])
        self.num_line_types = len(_LINE_TYPE_NAMES)

        self.trunk = _SalienceTrunk(
            input_channels=input_channels,
            channels=channels,
            depth=depth,
            num_salience_heads=self.num_salience_heads,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )

        # Per-line majorization descriptor:
        #   - top1, top2, top3
        #   - gap01, gap12
        #   - line mean, line sum, line max-minus-mean
        #   - top1_concentration, top2_concentration
        #   - normalized entropy of softmax(sorted)
        # = 11 features per (sample, salience_head, line)
        self.descriptors_per_line = 11
        # After pooling within each line-type bucket we keep mean and max,
        # so each (salience_head, line_type) contributes
        # 2 * descriptors_per_line features.
        per_bucket = 2 * self.descriptors_per_line
        bucket_dim = self.num_salience_heads * self.num_line_types * per_bucket
        self.bucket_dim = int(bucket_dim)

        head_in = self.trunk.context_dim + self.bucket_dim
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    # ------------------------------------------------------------------
    # Line-level statistics
    # ------------------------------------------------------------------
    def _gather_lines(self, salience: torch.Tensor) -> torch.Tensor:
        """Gather salience along every line.

        Parameters
        ----------
        salience : (B, K, 8, 8)

        Returns
        -------
        (B, K, num_lines, board_size)
        """
        bsz, k = salience.shape[0], salience.shape[1]
        flat = salience.reshape(bsz, k, self.board_size * self.board_size)
        # line_flat_index : (num_lines, board_size); broadcast into (1, 1, L, S).
        idx = self.line_flat_index.view(1, 1, self.num_lines, self.board_size)
        idx = idx.expand(bsz, k, self.num_lines, self.board_size)
        gathered = torch.gather(
            flat.unsqueeze(2).expand(-1, -1, self.num_lines, -1),
            dim=-1,
            index=idx,
        )
        return gathered

    def _compute_descriptors(
        self,
        sorted_scores: torch.Tensor,
        sort_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the 11 per-line majorization descriptors.

        ``sorted_scores`` : (B, K, num_lines, S)  soft-sorted descending
        ``sort_mask`` : Bool (B, K, num_lines, S)  True where positions are
            real (not padding); padded positions are 0 in ``sorted_scores``.
        """
        eps = 1.0e-6
        bsz, k, nlines, S = sorted_scores.shape
        valid_count = sort_mask.sum(dim=-1).clamp_min(1).to(sorted_scores.dtype)

        top1 = sorted_scores[..., 0]
        # Pull the second/third sorted entries when present, else zero.
        if S >= 2:
            top2 = torch.where(sort_mask[..., 1], sorted_scores[..., 1], torch.zeros_like(top1))
        else:
            top2 = torch.zeros_like(top1)
        if S >= 3:
            top3 = torch.where(sort_mask[..., 2], sorted_scores[..., 2], torch.zeros_like(top1))
        else:
            top3 = torch.zeros_like(top1)

        gap01 = top1 - top2
        gap12 = top2 - top3

        line_sum = sorted_scores.sum(dim=-1)
        line_mean = line_sum / valid_count
        line_max_minus_mean = top1 - line_mean
        sum_abs = sorted_scores.abs().sum(dim=-1).clamp_min(eps)
        top1_conc = top1.abs() / sum_abs
        top2_conc = (top1.abs() + top2.abs()) / sum_abs

        # Normalized entropy of the softmax-distributed sorted scores.
        # Softmax with masked-out padding to keep entropy comparable across
        # variable-length lines.
        very_neg = torch.finfo(sorted_scores.dtype).min
        masked_for_softmax = torch.where(
            sort_mask, sorted_scores, sorted_scores.new_full((), very_neg)
        )
        probs = torch.softmax(masked_for_softmax, dim=-1)
        log_probs = torch.log(probs.clamp_min(eps))
        entropy = -(probs * log_probs).sum(dim=-1)
        # Normalize by log(L) so the descriptor is in [0, 1].
        norm_entropy = entropy / torch.log(valid_count.clamp_min(2.0))

        feats = torch.stack(
            [
                top1,
                top2,
                top3,
                gap01,
                gap12,
                line_mean,
                line_sum,
                line_max_minus_mean,
                top1_conc,
                top2_conc,
                norm_entropy,
            ],
            dim=-1,
        )
        return feats  # (B, K, num_lines, descriptors_per_line)

    def _bucket_pool(self, descriptors: torch.Tensor) -> torch.Tensor:
        """Pool per-line descriptors within each line-type bucket.

        Returns
        -------
        (B, K, num_line_types, 2 * descriptors_per_line)
            mean and max within each (salience_head, line_type) bucket.
        """
        bsz, k, nlines, d = descriptors.shape
        # Build per-line-type one-hot mask: (num_lines, num_line_types).
        type_mask = F.one_hot(self.line_type, num_classes=self.num_line_types).to(
            descriptors.dtype
        )  # (num_lines, T)
        # Sum within each bucket.
        # weighted: (B, K, num_lines, T, d)
        weighted = descriptors.unsqueeze(-2) * type_mask.view(1, 1, nlines, self.num_line_types, 1)
        bucket_sum = weighted.sum(dim=2)  # (B, K, T, d)
        bucket_count = type_mask.sum(dim=0).clamp_min(1.0)  # (T,)
        bucket_mean = bucket_sum / bucket_count.view(1, 1, self.num_line_types, 1)

        # Max within each bucket: replace non-bucket slots with very-neg
        # before reducing.
        very_neg = torch.finfo(descriptors.dtype).min
        type_mask_bool = type_mask > 0  # (num_lines, T)
        # Expand to (B, K, num_lines, T, d).
        replaced = torch.where(
            type_mask_bool.view(1, 1, nlines, self.num_line_types, 1),
            descriptors.unsqueeze(-2),
            descriptors.new_full((), very_neg),
        )
        bucket_max = replaced.amax(dim=2)
        # Replace very-neg (no lines in bucket) with 0; this never happens for
        # the standard 4 chess line types but keeps things finite.
        bucket_max = torch.where(
            torch.isfinite(bucket_max), bucket_max, torch.zeros_like(bucket_max)
        )
        return torch.cat([bucket_mean, bucket_max], dim=-1)  # (B, K, T, 2d)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        salience, context = self.trunk(x)

        line_values = self._gather_lines(salience)  # (B, K, num_lines, S)
        bsz, k, nlines, S = line_values.shape
        valid = self.line_valid_mask.view(1, 1, nlines, S).expand(bsz, k, nlines, S)

        # Soft-sort each line descending; padded entries are zeroed after sorting.
        sorted_scores, sort_mask = _soft_sort_descending(
            line_values, valid, self.sort_temperature
        )
        descriptors = self._compute_descriptors(sorted_scores, sort_mask)
        bucket = self._bucket_pool(descriptors)  # (B, K, T, 2d)
        bucket_flat = bucket.reshape(bsz, -1)

        feat_vec = torch.cat([context, bucket_flat], dim=-1)
        logits = self.head(feat_vec).view(-1)

        # Convenience aggregates for diagnostic logging / reports.
        per_line_top1 = sorted_scores[..., 0]  # (B, K, num_lines)
        per_line_concentration = descriptors[..., 8]  # top1 / sum|.|
        per_line_gap01 = descriptors[..., 3]
        per_line_norm_entropy = descriptors[..., 10]

        # Most-active line type per sample (averaged over salience heads):
        # mean concentration per line-type bucket (B, T).
        mean_conc_per_type = bucket[..., 8].mean(dim=1)  # mean across K of bucket-mean top1_conc
        most_active_line_type = mean_conc_per_type.argmax(dim=-1)

        return {
            "logits": logits,
            "smls_salience_fields": salience,
            "smls_line_values": line_values,
            "smls_sorted_scores": sorted_scores,
            "smls_line_descriptors": descriptors,
            "smls_bucket_descriptors": bucket,
            "smls_per_line_top1": per_line_top1,
            "smls_per_line_concentration": per_line_concentration,
            "smls_per_line_gap01": per_line_gap01,
            "smls_per_line_normalized_entropy": per_line_norm_entropy,
            "smls_mean_concentration_per_line_type": mean_conc_per_type,
            "smls_most_active_line_type": most_active_line_type,
            "smls_board_context": context,
        }


def build_soft_majorization_line_sorter_from_config(
    config: dict[str, Any],
) -> SoftMajorizationLineSorter:
    cfg = dict(config)
    return SoftMajorizationLineSorter(
        input_channels=int(cfg.get("input_channels", 18)),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        num_salience_heads=int(cfg.get("num_salience_heads", 5)),
        sort_temperature=float(cfg.get("sort_temperature", 0.5)),
        num_classes=int(cfg.get("num_classes", 1)),
    )

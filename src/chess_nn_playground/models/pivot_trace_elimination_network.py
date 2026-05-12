"""Pivot Trace Elimination Network for idea i142.

Implements the markdown architecture from
``ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2204_friday_shanghai_architecture_batch_8.md``
(Candidate 6: Pivot Trace Elimination Network).

The board is encoded into ``K`` group summaries ``g in R^{K x D}``
covering piece-type groups, side roles, line groups, king-region groups,
and center/edge groups. A small bilinear builds an interaction matrix
``M_ij = small_bilinear(g_i, g_j)``. After adding a diagonal stabilizer
``lambda I``, a fixed-order differentiable Gaussian elimination is run,

    pivot_t = softplus(M_tt) + eps
    row_update = M_{t+1:, t} / pivot_t
    M_{t+1:, t+1:} -= row_update outer M_{t, t+1:}

recording the pivot, off-diagonal update norm, residual norm of the
remaining trailing block, and a condition-like ratio per step. The
classifier reads out the log-pivot trace, update norms, residual decay
curve, final residual norm, and condition-like ratio.

Central ablations from the packet:

  ``none``, ``raw_matrix_pool``, ``random_elimination_order``,
  ``diagonal_matrix_only``, ``determinant_only``, ``matrix_pencil_control``.

Following the packet's implementation note, learned pivoting is *not*
used: the elimination order is fixed (canonical group order) so the
``random_elimination_order`` ablation isolates whether semantic order
matters.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models._packet_bespoke_base import (
    BoardConvStem,
    BoardTensorSpec,
    format_logits,
    require_board_tensor,
    side_to_move_field,
    us_them_piece_planes,
)


VALID_ABLATIONS: frozenset[str] = frozenset(
    {
        "none",
        "raw_matrix_pool",
        "random_elimination_order",
        "diagonal_matrix_only",
        "determinant_only",
        "matrix_pencil_control",
    }
)


GROUP_NAMES: tuple[str, ...] = (
    "us_pawn_pieces",
    "us_minor_pieces",
    "us_major_pieces",
    "us_king_region",
    "them_pawn_pieces",
    "them_minor_pieces",
    "them_major_pieces",
    "them_king_region",
    "ranks_lines",
    "files_lines",
    "center",
    "edge",
)


def _build_group_masks() -> torch.Tensor:
    """Return a (12, 8, 8) tensor of fixed spatial masks for the line/region/zone groups.

    Indices follow ``GROUP_NAMES``. The first eight (piece-type and king-region
    groups) are dynamic (filled in forward) so this tensor only carries the
    static masks (lines / center / edge) at their respective indices and zeros
    elsewhere.
    """
    masks = torch.zeros(len(GROUP_NAMES), 8, 8)
    rows = torch.arange(8).view(8, 1).expand(8, 8).float()
    cols = torch.arange(8).view(1, 8).expand(8, 8).float()
    rank_weight = torch.cos(rows * torch.pi / 7.0)
    file_weight = torch.cos(cols * torch.pi / 7.0)
    masks[GROUP_NAMES.index("ranks_lines")] = rank_weight
    masks[GROUP_NAMES.index("files_lines")] = file_weight
    center_mask = torch.zeros(8, 8)
    center_mask[2:6, 2:6] = 1.0
    center_mask[3:5, 3:5] = 2.0
    masks[GROUP_NAMES.index("center")] = center_mask
    edge_mask = torch.ones(8, 8)
    edge_mask[1:7, 1:7] = 0.0
    masks[GROUP_NAMES.index("edge")] = edge_mask
    return masks


def _king_region_field(king_plane: torch.Tensor) -> torch.Tensor:
    """Soft 3x3 dilation of a king plane producing a king-region weight map."""
    kernel = king_plane.new_ones(1, 1, 3, 3)
    return F.conv2d(king_plane.unsqueeze(1), kernel, padding=1).squeeze(1).clamp(0.0, 9.0)


class _GroupEncoder(nn.Module):
    """Turn a (B, C, 8, 8) feature map into K group summaries (B, K, D).

    Each group ``k`` is summarised as a soft-mask weighted average of trunk
    features, mass-normalised so it remains stable when a group is empty.
    """

    def __init__(self, channels: int, group_dim: int, num_groups: int = len(GROUP_NAMES)) -> None:
        super().__init__()
        self.num_groups = int(num_groups)
        self.group_dim = int(group_dim)
        self.proj = nn.Linear(channels, group_dim)
        self.register_buffer("static_masks", _build_group_masks(), persistent=False)
        self.eps = 1.0e-6

    def forward(self, feat: torch.Tensor, board: torch.Tensor, input_channels: int) -> torch.Tensor:
        b, c, h, w = feat.shape
        us, them = us_them_piece_planes(board, input_channels)
        side = side_to_move_field(board, input_channels)
        us_king = (us[:, 5] if us.shape[1] >= 6 else us.new_zeros(b, h, w))
        them_king = (them[:, 5] if them.shape[1] >= 6 else them.new_zeros(b, h, w))
        us_pawn = us[:, 0]
        us_minor = us[:, 1:3].sum(dim=1)
        us_major = us[:, 3:5].sum(dim=1)
        them_pawn = them[:, 0]
        them_minor = them[:, 1:3].sum(dim=1)
        them_major = them[:, 3:5].sum(dim=1)
        us_king_region = _king_region_field(us_king)
        them_king_region = _king_region_field(them_king)

        side_scalar = side.amax(dim=(1, 2, 3), keepdim=False).clamp(0.0, 1.0).view(b, 1, 1, 1)
        rank_mask = self.static_masks[GROUP_NAMES.index("ranks_lines")].abs()
        file_mask = self.static_masks[GROUP_NAMES.index("files_lines")].abs()
        center_mask = self.static_masks[GROUP_NAMES.index("center")]
        edge_mask = self.static_masks[GROUP_NAMES.index("edge")]
        rank_signed = self.static_masks[GROUP_NAMES.index("ranks_lines")]
        rank_signed = (1.0 - 2.0 * (1.0 - side_scalar.view(b, 1, 1))) * rank_signed.view(1, 8, 8)

        masks_per_batch = [
            us_pawn,
            us_minor,
            us_major,
            us_king_region,
            them_pawn,
            them_minor,
            them_major,
            them_king_region,
            rank_signed.abs().expand(b, 8, 8),
            file_mask.view(1, 8, 8).expand(b, 8, 8),
            center_mask.view(1, 8, 8).expand(b, 8, 8),
            edge_mask.view(1, 8, 8).expand(b, 8, 8),
        ]
        masks = torch.stack(masks_per_batch, dim=1)
        masks = masks.clamp_min(0.0)
        flat_feat = feat.permute(0, 2, 3, 1).reshape(b, h * w, c)
        flat_masks = masks.view(b, self.num_groups, h * w)
        weight_sum = flat_masks.sum(dim=2, keepdim=True).clamp_min(self.eps)
        weighted = torch.bmm(flat_masks, flat_feat) / weight_sum
        groups = self.proj(weighted)
        return groups, masks


class _BilinearMatrix(nn.Module):
    """``M_ij = phi(g_i)^T A phi(g_j)`` with a learned bilinear form."""

    def __init__(self, group_dim: int) -> None:
        super().__init__()
        self.left = nn.Linear(group_dim, group_dim, bias=False)
        self.right = nn.Linear(group_dim, group_dim, bias=False)

    def forward(self, groups: torch.Tensor) -> torch.Tensor:
        left = self.left(groups)
        right = self.right(groups)
        return torch.matmul(left, right.transpose(-1, -2))


def _fixed_order_elimination(
    matrix: torch.Tensor,
    *,
    eps: float = 1.0e-4,
) -> dict[str, torch.Tensor]:
    """Fixed-order differentiable Gaussian elimination.

    For ``t = 0, ..., K-1`` we record:
      - log of softplus pivot,
      - L1 norm of the row update factor (``M_{t+1:, t} / pivot``),
      - Frobenius norm of the trailing residual after the update,
      - condition-like ratio (current pivot vs. running min pivot).

    Returns tensors of shape ``(B, K)`` keyed in the dict by name.
    """
    b = matrix.shape[0]
    k = matrix.shape[-1]
    work = matrix.clone()
    log_pivots = matrix.new_zeros(b, k)
    update_norms = matrix.new_zeros(b, k)
    residual_norms = matrix.new_zeros(b, k)
    cond_ratio = matrix.new_zeros(b, k)
    running_min = matrix.new_full((b,), float("inf"))
    running_max = matrix.new_zeros(b)

    for t in range(k):
        diag = work[:, t, t]
        pivot = F.softplus(diag) + eps
        log_pivots[:, t] = pivot.log()
        running_min = torch.minimum(running_min, pivot)
        running_max = torch.maximum(running_max, pivot)
        cond_ratio[:, t] = (running_max / running_min.clamp_min(eps)).log()

        if t + 1 >= k:
            residual_norms[:, t] = matrix.new_zeros(b)
            update_norms[:, t] = matrix.new_zeros(b)
            continue

        col_below = work[:, t + 1 :, t]
        row_right = work[:, t, t + 1 :]
        update = col_below / pivot.unsqueeze(1)
        update_norms[:, t] = update.abs().mean(dim=1)
        outer = update.unsqueeze(2) * row_right.unsqueeze(1)
        trailing = work[:, t + 1 :, t + 1 :] - outer
        # Write back without breaking autograd graph (rebuild work).
        new_work = work.clone()
        new_work[:, t + 1 :, t + 1 :] = trailing
        work = new_work
        residual_norms[:, t] = trailing.reshape(b, -1).norm(dim=1) / float((k - t - 1) ** 0.5 + 1.0)

    final_residual = work[:, -1:, -1:].reshape(b, 1).abs().squeeze(1)
    return {
        "log_pivots": log_pivots,
        "update_norms": update_norms,
        "residual_norms": residual_norms,
        "cond_ratio": cond_ratio,
        "final_residual": final_residual,
    }


class PivotTraceEliminationNetwork(nn.Module):
    """Bespoke implementation of the Pivot Trace Elimination Network.

    Forward output dict keys:
      - ``logits``: ``(B,)`` puzzle logit.
      - ``log_pivots``, ``update_norms``, ``residual_norms``,
        ``cond_ratio``: ``(B, K)`` per-step elimination diagnostics.
      - ``final_residual``: ``(B,)`` final trailing diagonal magnitude.
      - ``log_determinant``: ``(B,)`` sum of log pivots (= log-det of ``M``).
      - ``matrix``: ``(B, K, K)`` constructed interaction matrix.
      - ``group_summaries``: ``(B, K, D)`` group encodings.
      - ``group_masses``: ``(B, K)`` per-group total mass on the board.
      - ``ablation_*``: indicator scalars.
    """

    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        group_dim: int = 32,
        num_classes: int = 1,
        diagonal_stabilizer: float = 0.1,
        elimination_eps: float = 1.0e-4,
        ablation: str = "none",
        # Optional knobs accepted for config-symmetry but not used by mechanism:
        mechanism_family: str | None = None,
        packet_profile: str | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("PivotTraceEliminationNetwork supports the puzzle_binary one-logit contract")
        if ablation not in VALID_ABLATIONS:
            raise ValueError(
                f"Unknown ablation {ablation!r}; expected one of {sorted(VALID_ABLATIONS)}"
            )
        self.input_channels = int(input_channels)
        self.spec = BoardTensorSpec(input_channels=self.input_channels)
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.group_dim = int(group_dim)
        self.diagonal_stabilizer = float(diagonal_stabilizer)
        self.elimination_eps = float(elimination_eps)
        self.ablation = str(ablation)
        self.k = len(GROUP_NAMES)

        self.trunk = BoardConvStem(
            input_channels=self.input_channels,
            channels=self.channels,
            depth=int(depth),
            use_batchnorm=bool(use_batchnorm),
        )
        self.encoder = _GroupEncoder(self.channels, self.group_dim, num_groups=self.k)
        self.matrix_builder = _BilinearMatrix(self.group_dim)

        # Fixed canonical order (identity permutation). For the
        # ``random_elimination_order`` ablation we pre-sample one fixed but
        # non-identity order so the ablation isolates "semantic order matters"
        # without injecting batch-time randomness.
        self.register_buffer("canonical_order", torch.arange(self.k), persistent=False)
        rng = torch.Generator()
        rng.manual_seed(0xE142)
        random_perm = torch.randperm(self.k, generator=rng)
        if torch.equal(random_perm, self.canonical_order):
            random_perm = random_perm.flip(0)
        self.register_buffer("random_order", random_perm, persistent=False)

        # Head feature size depends on the ablation. We unify all paths to
        # produce a (B, F) feature where F is constant for a given ablation.
        feat_dim = self._feature_dim()
        layers: list[nn.Module] = [nn.LayerNorm(feat_dim), nn.Linear(feat_dim, self.hidden_dim), nn.GELU()]
        if self.dropout > 0:
            layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Linear(self.hidden_dim, 1))
        self.head = nn.Sequential(*layers)

    def _feature_dim(self) -> int:
        if self.ablation == "raw_matrix_pool":
            return self.k * self.k + self.k
        if self.ablation == "determinant_only":
            return 1 + self.k
        if self.ablation == "matrix_pencil_control":
            return self.k + self.k
        # default elim path: log_pivots, update_norms, residual_norms,
        # cond_ratio, final_residual, log_det, plus group_masses
        return self.k * 4 + 2 + self.k

    def _build_matrix(self, groups: torch.Tensor) -> torch.Tensor:
        m = self.matrix_builder(groups)
        m = 0.5 * (m + m.transpose(-1, -2))
        eye = torch.eye(self.k, dtype=m.dtype, device=m.device)
        m = m + self.diagonal_stabilizer * eye
        if self.ablation == "diagonal_matrix_only":
            diag = m.diagonal(dim1=-2, dim2=-1)
            m = torch.diag_embed(diag)
        return m

    def _permute_for_elimination(self, m: torch.Tensor) -> torch.Tensor:
        order = self.random_order if self.ablation == "random_elimination_order" else self.canonical_order
        permuted = m.index_select(-2, order).index_select(-1, order)
        return permuted

    def _ablation_flags(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        ones = x.new_ones(x.shape[0])
        zeros = x.new_zeros(x.shape[0])
        flags = {f"ablation_{name}": (ones if self.ablation == name else zeros) for name in sorted(VALID_ABLATIONS)}
        return flags

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feat = self.trunk(x)
        groups, masks = self.encoder(feat, x, self.input_channels)
        group_masses = masks.sum(dim=(2, 3))
        matrix = self._build_matrix(groups)

        if self.ablation == "raw_matrix_pool":
            pooled = matrix.reshape(matrix.shape[0], -1)
            features = torch.cat([pooled, group_masses], dim=1)
            log_pivots = matrix.new_zeros(matrix.shape[0], self.k)
            update_norms = matrix.new_zeros(matrix.shape[0], self.k)
            residual_norms = matrix.new_zeros(matrix.shape[0], self.k)
            cond_ratio = matrix.new_zeros(matrix.shape[0], self.k)
            final_residual = matrix.new_zeros(matrix.shape[0])
            log_det = matrix.new_zeros(matrix.shape[0])
        elif self.ablation == "matrix_pencil_control":
            sym = 0.5 * (matrix + matrix.transpose(-1, -2))
            eigvals = torch.linalg.eigvalsh(sym)
            features = torch.cat([eigvals, group_masses], dim=1)
            log_pivots = eigvals.clamp_min(self.elimination_eps).log()
            update_norms = matrix.new_zeros(matrix.shape[0], self.k)
            residual_norms = matrix.new_zeros(matrix.shape[0], self.k)
            cond_ratio = matrix.new_zeros(matrix.shape[0], self.k)
            final_residual = eigvals.amin(dim=1)
            log_det = log_pivots.sum(dim=1)
        else:
            permuted = self._permute_for_elimination(matrix)
            elim = _fixed_order_elimination(permuted, eps=self.elimination_eps)
            log_pivots = elim["log_pivots"]
            update_norms = elim["update_norms"]
            residual_norms = elim["residual_norms"]
            cond_ratio = elim["cond_ratio"]
            final_residual = elim["final_residual"]
            log_det = log_pivots.sum(dim=1)
            if self.ablation == "determinant_only":
                features = torch.cat([log_det.unsqueeze(1), group_masses], dim=1)
            else:
                features = torch.cat(
                    [
                        log_pivots,
                        update_norms,
                        residual_norms,
                        cond_ratio,
                        final_residual.unsqueeze(1),
                        log_det.unsqueeze(1),
                        group_masses,
                    ],
                    dim=1,
                )

        logits = self.head(features).view(-1)

        outputs: dict[str, torch.Tensor] = {
            "logits": format_logits(logits, num_classes=1),
            "log_pivots": log_pivots,
            "update_norms": update_norms,
            "residual_norms": residual_norms,
            "cond_ratio": cond_ratio,
            "final_residual": final_residual,
            "log_determinant": log_det,
            "matrix": matrix,
            "group_summaries": groups,
            "group_masses": group_masses,
        }
        outputs.update(self._ablation_flags(x))
        return outputs


def build_pivot_trace_elimination_network_from_config(
    config: dict[str, Any],
) -> PivotTraceEliminationNetwork:
    cfg = dict(config)
    return PivotTraceEliminationNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        group_dim=int(cfg.get("group_dim", 32)),
        diagonal_stabilizer=float(cfg.get("diagonal_stabilizer", 0.1)),
        elimination_eps=float(cfg.get("elimination_eps", 1.0e-4)),
        ablation=str(cfg.get("ablation", "none")),
        num_classes=int(cfg.get("num_classes", 1)),
        mechanism_family=cfg.get("mechanism_family"),
        packet_profile=cfg.get("packet_profile"),
        name=cfg.get("name"),
    )

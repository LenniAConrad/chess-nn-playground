"""Critical-Square Budget Network for idea i185.

Faithful implementation of the markdown thesis under
``ideas/all_ideas/registry/i185_critical_square_budget_network/``. The packet's working
thesis is that puzzles often hinge on a *small number of critical
squares*: king escape squares, line intersections, pinned-piece
squares, promotion squares, or overloaded defender squares. The
architecture turns that thesis into an explicit *budget*: the model
must produce a soft mask over the 64 squares whose total mass equals a
fixed budget ``K`` and route the puzzle logit through that masked
pool.

The pipeline is:

    feats         = trunk(board)                       # (B, C, 8, 8)
    priors        = critical_square_priors(board)      # (B, P, 8, 8)
    saliency_logits = saliency_head(feats, priors)     # (B, 8, 8)
    mask          = budget * softmax(saliency_logits)  # (B, 8, 8)
    pooled        = sum_squares(mask * feats)          # (B, C)
    summary       = budget_summary(mask, priors)       # (B, S)
    logit         = head([pooled, summary])

The mask sums to ``budget`` per batch row, which is the explicit
"critical square budget" the packet calls for. Lowering the
``saliency_temperature`` makes the mask sparser; ``budget`` controls
how many squares the head is allowed to read from. The deterministic
priors -- king zones, promotion ranks, and line-piece ray landmarks --
correspond directly to the packet's enumerated critical-square
families and are concatenated to the trunk features so the saliency
head can learn to weigh them.

Distinct from idea i174 (King-Zone Evidence Ledger), which keeps a
king-anchored slot bank and reads from it; here the model has a single
global budget mask over all 64 squares and the diagnostics report how
much of that budget lands inside each prior region.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import (
    BoardConvStem,
    BoardTensorSpec,
    require_board_tensor,
)


WHITE_PIECE_PLANES = (0, 1, 2, 3, 4, 5)
BLACK_PIECE_PLANES = (6, 7, 8, 9, 10, 11)
WHITE_KING_PLANE = 5
BLACK_KING_PLANE = 11
SIDE_TO_MOVE_PLANE = 12
WHITE_LINE_PIECE_PLANES = (2, 3, 4)  # bishop, rook, queen
BLACK_LINE_PIECE_PLANES = (8, 9, 10)
NUM_CRITICAL_PRIORS = 6


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.view(-1) if num_classes == 1 else logits


def _dilate3(mask: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)


def _line_landmarks(line_mask: torch.Tensor) -> torch.Tensor:
    """Return a coarse line-intersection landmark for a (B, 1, 8, 8) mask
    of long-range pieces. The landmark is the elementwise product of
    per-square rank and file occupancy tallies, which lights up on the
    intersections of files and ranks containing line pieces. This is a
    deterministic stand-in for the packet's ``line intersections``
    critical-square family.
    """
    rank_count = line_mask.sum(dim=3, keepdim=True).expand_as(line_mask)
    file_count = line_mask.sum(dim=2, keepdim=True).expand_as(line_mask)
    return rank_count * file_count


def _critical_square_priors(board: torch.Tensor) -> torch.Tensor:
    """Compute deterministic per-square critical-square prior planes
    enumerated by the markdown thesis.

    Returns ``(B, NUM_CRITICAL_PRIORS, 8, 8)``:
        0 -- own king zone (3x3 dilation of own king square)
        1 -- opponent king zone (3x3 dilation of opp king square)
        2 -- promotion ranks for the side to move
        3 -- own pinned/long-range piece field landmarks
        4 -- opponent pinned/long-range piece field landmarks
        5 -- empty-square indicator (no piece on the square)
    """
    batch = board.shape[0]
    device = board.device
    dtype = board.dtype

    white = board[:, WHITE_KING_PLANE : WHITE_KING_PLANE + 1]
    black = board[:, BLACK_KING_PLANE : BLACK_KING_PLANE + 1]
    side = board[:, SIDE_TO_MOVE_PLANE : SIDE_TO_MOVE_PLANE + 1]
    white_to_move = side.amax(dim=(2, 3), keepdim=True).clamp(0.0, 1.0)
    black_to_move = 1.0 - white_to_move

    own_king = white_to_move * white + black_to_move * black
    opp_king = white_to_move * black + black_to_move * white
    own_king_zone = _dilate3(own_king)
    opp_king_zone = _dilate3(opp_king)

    rank_eight = torch.zeros(1, 1, 8, 8, device=device, dtype=dtype)
    rank_eight[..., 7, :] = 1.0
    rank_one = torch.zeros(1, 1, 8, 8, device=device, dtype=dtype)
    rank_one[..., 0, :] = 1.0
    promotion = white_to_move * rank_eight + black_to_move * rank_one
    promotion = promotion.expand(batch, 1, 8, 8)

    own_line = (
        white_to_move * board[:, WHITE_LINE_PIECE_PLANES, :, :].sum(dim=1, keepdim=True)
        + black_to_move * board[:, BLACK_LINE_PIECE_PLANES, :, :].sum(dim=1, keepdim=True)
    )
    opp_line = (
        white_to_move * board[:, BLACK_LINE_PIECE_PLANES, :, :].sum(dim=1, keepdim=True)
        + black_to_move * board[:, WHITE_LINE_PIECE_PLANES, :, :].sum(dim=1, keepdim=True)
    )
    own_landmarks = _line_landmarks(own_line.clamp(0.0, 1.0))
    opp_landmarks = _line_landmarks(opp_line.clamp(0.0, 1.0))
    scale = float(8 * 8)
    own_landmarks = own_landmarks / scale
    opp_landmarks = opp_landmarks / scale

    occupancy = board[:, 0:12, :, :].sum(dim=1, keepdim=True).clamp(0.0, 1.0)
    empty = 1.0 - occupancy

    priors = torch.cat(
        [own_king_zone, opp_king_zone, promotion, own_landmarks, opp_landmarks, empty],
        dim=1,
    )
    return priors


class _SaliencyHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.drop(self.act(self.conv1(x))))


class CriticalSquareBudgetNetwork(nn.Module):
    """Critical-Square Budget Network.

    Forward returns a dict with at least:

      - ``logits``: ``(B,)`` puzzle logit (or ``(B, num_classes)`` for
        ``num_classes > 1``).
      - ``prob``: ``sigmoid(logits)`` when ``num_classes == 1``.
      - ``saliency_logits``: ``(B, 8, 8)`` raw per-square saliency
        before the budgeted softmax.
      - ``saliency_mask``: ``(B, 8, 8)`` soft mask whose total mass per
        batch row equals ``budget``. This is the explicit critical-
        square budget the packet describes.
      - ``budget_used``: ``(B,)`` total mask mass per batch row (equals
        ``budget`` up to floating-point error; exposed for diagnostics
        and so the trainer can sanity-check the budget contract).
      - ``saliency_entropy``: ``(B,)`` entropy of the normalised mask
        across the 64 squares; small entropy means the model has
        concentrated its budget on a few squares.
      - ``top_k_mass``: ``(B,)`` sum of the largest ``budget`` mask
        entries (a sparsity proxy in [0, ``budget``]).
      - ``own_king_zone_mass`` / ``opp_king_zone_mass`` /
        ``promotion_mass`` / ``line_intersection_mass`` /
        ``empty_square_mass``: ``(B,)`` fraction of the mask mass that
        lands inside each named critical-square prior region. These
        let downstream reports check whether the model is actually
        using the prior regions enumerated by the packet.
      - ``trunk_energy``: ``(B,)`` mean-square trunk activation.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        depth: int = 2,
        hidden_dim: int = 96,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        budget: float = 6.0,
        saliency_temperature: float = 1.0,
        saliency_hidden: int = 0,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        if budget <= 0.0:
            raise ValueError("budget must be > 0")
        if saliency_temperature <= 0.0:
            raise ValueError("saliency_temperature must be > 0")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.depth = int(depth)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.use_batchnorm = bool(use_batchnorm)
        self.budget = float(budget)
        self.saliency_temperature = float(saliency_temperature)
        self.saliency_hidden = int(saliency_hidden) if saliency_hidden else max(self.channels // 2, 8)

        self.trunk = BoardConvStem(
            input_channels=self.input_channels,
            channels=self.channels,
            depth=self.depth,
            use_batchnorm=self.use_batchnorm,
        )

        self.saliency_head = _SaliencyHead(
            in_channels=self.channels + NUM_CRITICAL_PRIORS,
            hidden_channels=self.saliency_hidden,
            dropout=self.dropout,
        )

        # 5 prior masses + budget_used + entropy + top_k_mass.
        summary_dim = 5 + 3
        head_in = self.channels + summary_dim
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def _saliency_mask(self, saliency_logits: torch.Tensor) -> torch.Tensor:
        """Convert per-square saliency logits to a budget-K soft mask.

        ``saliency_logits`` has shape ``(B, 8, 8)``. The output is a
        non-negative mask of the same shape whose elements sum to
        ``budget`` per batch row.
        """
        flat = saliency_logits.flatten(1) / self.saliency_temperature
        weights = F.softmax(flat, dim=1)
        weights = weights * self.budget
        return weights.view(saliency_logits.shape)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feats = self.trunk(x)  # (B, C, 8, 8)
        priors = _critical_square_priors(x)  # (B, P, 8, 8)
        saliency_in = torch.cat([feats, priors], dim=1)
        saliency_logits = self.saliency_head(saliency_in).squeeze(1)  # (B, 8, 8)
        mask = self._saliency_mask(saliency_logits)  # (B, 8, 8) summing to budget

        mask_4d = mask.unsqueeze(1)  # (B, 1, 8, 8)
        pooled = (feats * mask_4d).flatten(2).sum(dim=2)  # (B, C)

        budget_used = mask.flatten(1).sum(dim=1)
        norm_mask = mask / budget_used.clamp_min(1e-8).view(-1, 1, 1)
        flat_norm = norm_mask.flatten(1).clamp_min(1e-12)
        saliency_entropy = -(flat_norm * flat_norm.log()).sum(dim=1)

        top_k = max(1, int(round(self.budget)))
        top_k = min(top_k, mask.flatten(1).shape[1])
        top_k_mass = mask.flatten(1).topk(top_k, dim=1).values.sum(dim=1)

        prior_masses = (priors * mask_4d).flatten(2).sum(dim=2)  # (B, P)
        own_king_zone_mass = prior_masses[:, 0]
        opp_king_zone_mass = prior_masses[:, 1]
        promotion_mass = prior_masses[:, 2]
        line_intersection_mass = prior_masses[:, 3] + prior_masses[:, 4]
        empty_square_mass = prior_masses[:, 5]

        summary = torch.stack(
            [
                own_king_zone_mass,
                opp_king_zone_mass,
                promotion_mass,
                line_intersection_mass,
                empty_square_mass,
                budget_used,
                saliency_entropy,
                top_k_mass,
            ],
            dim=1,
        )

        head_in = torch.cat([pooled, summary], dim=1)
        logits = self.head(head_in)
        logits = _format_logits(logits, self.num_classes)

        with torch.no_grad():
            trunk_energy = feats.square().mean(dim=(1, 2, 3))

        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "saliency_logits": saliency_logits,
            "saliency_mask": mask,
            "budget_used": budget_used,
            "saliency_entropy": saliency_entropy,
            "top_k_mass": top_k_mass,
            "own_king_zone_mass": own_king_zone_mass,
            "opp_king_zone_mass": opp_king_zone_mass,
            "promotion_mass": promotion_mass,
            "line_intersection_mass": line_intersection_mass,
            "empty_square_mass": empty_square_mass,
            "trunk_energy": trunk_energy,
        }
        if self.num_classes == 1:
            output["prob"] = torch.sigmoid(logits)
        return output


def build_critical_square_budget_network_from_config(
    config: dict[str, Any],
) -> CriticalSquareBudgetNetwork:
    cfg = dict(config)
    cfg.pop("name", None)
    cfg.pop("mechanism_family", None)
    cfg.pop("packet_profile", None)
    return CriticalSquareBudgetNetwork(
        input_channels=int(cfg.pop("input_channels", 18)),
        num_classes=int(cfg.pop("num_classes", 1)),
        channels=int(cfg.pop("channels", 64)),
        depth=int(cfg.pop("depth", 2)),
        hidden_dim=int(cfg.pop("hidden_dim", 96)),
        dropout=float(cfg.pop("dropout", 0.1)),
        use_batchnorm=bool(cfg.pop("use_batchnorm", True)),
        budget=float(cfg.pop("budget", 6.0)),
        saliency_temperature=float(cfg.pop("saliency_temperature", 1.0)),
        saliency_hidden=int(cfg.pop("saliency_hidden", 0)),
    )


__all__ = [
    "CriticalSquareBudgetNetwork",
    "build_critical_square_budget_network_from_config",
]

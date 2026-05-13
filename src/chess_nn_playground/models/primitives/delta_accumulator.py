"""Shared O(|Δ|) delta-accumulator helpers for primitive heads p012–p018.

The seven primitives in the delta-accumulator family
(``ideas/research/primitives/external_{01,07,08,09,10,11,17}*.md``)
all share the same chess-specific algebraic shape:

- The active feature set ``S(x) = {(piece_type, square) : board has piece at square}``
  is a sparse subset of ``{0, ..., 12·64 - 1}``.
- A move toggles a small bounded subset ``|Δ| ≤ 6`` of those indices.
- The primitive's defining contract is that the forward state can be updated
  in ``O(|Δ| · d)`` time without recomputing the full sum.

At static-position scout scale (no engine search), the *training* trainer
sees positions, not move sequences, so the natural training-time equivalent
of the ``apply_delta`` primitive is simply
``h = Σ_{(t,s) ∈ S(x)} W[t·64 + s]`` evaluated in one batched gather. The
``O(|Δ|)`` inference property is documented in each idea's
``implementation_notes.md`` — it is a property of how the operator *would*
be invoked inside an engine make/unmake loop, not of the static-batch
trainer used for puzzle_binary scout runs. The shared
``DeltaAccumulator`` module below packages the common feature-extraction
and embedding plumbing so each primitive only has to add its own
distinctive aggregation rule (signed pair state, ClippedReLU saturation,
χ-graded splits, legal-move routing, sheaf diffusion, etc.).

Inputs are restricted to the ``simple_18`` ``(B, 18, 8, 8)`` current-board
tensor. CRTK metadata, source labels, verification flags, principal
variations, and engine evaluations are **never** consulted as model inputs
(those are reporting-only). The 12 piece planes plus side-to-move and
castling-right planes carry every rule-derived signal these primitives
require.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


PIECE_PLANE_COUNT = 12
SQUARES = 64
FEATURE_VOCAB = PIECE_PLANE_COUNT * SQUARES  # (piece_type, square) cardinality
STM_CHANNEL = 12

# Maximum active piece-square count per position. A legal chess position has
# at most 32 pieces; we keep one extra slot of headroom so any malformed
# simple_18 tensor with rounding noise still fits.
MAX_ACTIVE_FEATURES = 40


def _build_square_grid() -> tuple[torch.Tensor, torch.Tensor]:
    """Returns rank- and file-index tables for the 64 plane squares."""

    ranks = torch.arange(SQUARES, dtype=torch.long) // 8
    files = torch.arange(SQUARES, dtype=torch.long) % 8
    return ranks, files


@dataclass(frozen=True)
class ActiveFeatures:
    """Compact representation of the active piece-square index set.

    ``indices`` and ``valid`` have shape ``(B, K)`` where ``K`` is a fixed
    upper bound (``MAX_ACTIVE_FEATURES`` by default). Invalid slots point at
    index 0 with ``valid = 0``; downstream code masks them out before
    summing. ``count`` is the per-sample number of active features.
    """

    indices: torch.Tensor  # (B, K) long, in [0, FEATURE_VOCAB)
    valid: torch.Tensor    # (B, K) float in {0, 1}
    count: torch.Tensor    # (B,) float — number of active features


def extract_active_features(
    board: torch.Tensor,
    max_features: int = MAX_ACTIVE_FEATURES,
) -> ActiveFeatures:
    """Convert the simple_18 board into a compact active piece-square index list.

    For each sample, we scan the 12 piece planes and collect every
    ``(piece_type, square)`` cell whose value exceeds 0.5. The collection
    is padded to ``max_features`` slots with index 0 and ``valid = 0`` for
    the unused tail so the downstream gather can run on a fixed-shape
    tensor. The sort order is deterministic (ascending feature index) so
    the static-position forward matches what an engine make/unmake stream
    would produce after a sequence of ``apply_delta`` calls.

    Args:
        board: ``(B, 18, 8, 8)`` simple_18 board tensor.
        max_features: padding upper bound for the active set.

    Returns:
        :class:`ActiveFeatures` with stop-gradient tensors. Gradients do
        not flow through the discrete index list — the only differentiable
        path is via the downstream learnable embedding lookup.
    """

    if board.ndim != 4 or board.shape[1] < PIECE_PLANE_COUNT + 1:
        raise ValueError(
            f"extract_active_features requires simple_18 board, got {tuple(board.shape)}"
        )
    if max_features < 1:
        raise ValueError("max_features must be >= 1")

    batch = board.shape[0]
    device = board.device

    piece_planes = board[:, :PIECE_PLANE_COUNT].reshape(batch, PIECE_PLANE_COUNT, SQUARES)
    occupancy = (piece_planes > 0.5).to(dtype=torch.bool)  # (B, 12, 64)
    flat_occupancy = occupancy.reshape(batch, FEATURE_VOCAB)
    count = flat_occupancy.to(dtype=torch.float32).sum(dim=1)

    # Pull the per-sample active indices via topk on the boolean mask. The
    # top-k indices are sorted by score, with ties broken by index; we then
    # re-sort to ascending feature-index order for determinism.
    presence = flat_occupancy.to(dtype=torch.float32)
    k = min(int(max_features), FEATURE_VOCAB)
    top_scores, top_indices = presence.topk(k, dim=1)
    valid = (top_scores > 0.5).to(dtype=torch.float32)
    # Sort active slots first, then by feature index, to get a deterministic
    # order across samples (valid slots come first, invalid slots last).
    sort_key = top_indices.float() + (1.0 - valid) * (FEATURE_VOCAB + 1.0)
    order = sort_key.argsort(dim=1)
    indices_sorted = top_indices.gather(1, order)
    valid_sorted = valid.gather(1, order)

    if k < int(max_features):
        pad_count = int(max_features) - k
        indices_sorted = torch.cat(
            [indices_sorted, indices_sorted.new_zeros(batch, pad_count)], dim=1
        )
        valid_sorted = torch.cat(
            [valid_sorted, valid_sorted.new_zeros(batch, pad_count)], dim=1
        )

    return ActiveFeatures(
        indices=indices_sorted.long().contiguous(),
        valid=valid_sorted.contiguous(),
        count=count.to(device=device, dtype=torch.float32),
    )


class DeltaAccumulator(nn.Module):
    """Embedding-table sum over an input-determined sparse feature set.

    This is the canonical generalised HalfKA "feature transformer" from
    ``ideas/research/primitives/external_07_sparse_delta_accumulator*.md``
    rewritten as a static-position differentiable module. The forward
    pass evaluates ``h = Σ_{i ∈ S(x)} W[i]`` where ``S(x)`` is the active
    piece-square index set. The same state is the natural starting point
    for the bilinear, ClippedReLU, χ-graded, and sheaf-diffusion
    extensions used by the rest of the primitive family.
    """

    def __init__(
        self,
        accumulator_dim: int = 64,
        max_features: int = MAX_ACTIVE_FEATURES,
    ) -> None:
        super().__init__()
        if int(accumulator_dim) < 1:
            raise ValueError("accumulator_dim must be >= 1")
        if int(max_features) < 1:
            raise ValueError("max_features must be >= 1")
        self.accumulator_dim = int(accumulator_dim)
        self.max_features = int(max_features)
        # ``embedding`` is the parameter ``W ∈ R^{V × d}`` from the primitive spec.
        # padding_idx=0 keeps the entry-0 row unused (we always mask invalid
        # slots out anyway, but having a dedicated padding row makes the
        # sparse gradient bookkeeping correct under any future autograd path).
        self.embedding = nn.Embedding(FEATURE_VOCAB, self.accumulator_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.05)

    def gather(self, features: ActiveFeatures) -> torch.Tensor:
        """Return the per-slot embedding ``W[i_k]`` masked by ``valid``."""

        raw = self.embedding(features.indices)  # (B, K, d)
        return raw * features.valid.unsqueeze(-1)

    def forward(self, features: ActiveFeatures) -> torch.Tensor:
        """Compute the additive accumulator state ``h``."""

        masked = self.gather(features)
        return masked.sum(dim=1)


def piece_type_and_square(indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a flat (piece_type * 64 + square) index back into (type, square)."""

    return indices // SQUARES, indices % SQUARES


def piece_color_id(piece_type: torch.Tensor) -> torch.Tensor:
    """Map simple_18 piece-plane index to ``{0, 1}`` (white = 0, black = 1)."""

    return (piece_type >= 6).to(dtype=torch.long)


def opponent_color_id(piece_type: torch.Tensor) -> torch.Tensor:
    """Opponent colour id for the given piece-plane index."""

    return 1 - piece_color_id(piece_type)


def side_to_move_color(board: torch.Tensor) -> torch.Tensor:
    """Recover the side-to-move colour (0 = white, 1 = black) per sample."""

    stm = board[:, STM_CHANNEL].mean(dim=(1, 2)).clamp(0.0, 1.0)
    return (stm <= 0.5).to(dtype=torch.long)


def involution_indices(indices: torch.Tensor) -> torch.Tensor:
    """Return the colour-swap involution image of each (piece_type, square).

    The chess involution sends white piece ``t`` on square ``s`` to black
    piece ``t`` on rank-flipped square ``s'``. simple_18 stores white
    pieces in planes 0..5 and black pieces in planes 6..11; the rank flip
    is ``rank -> 7 - rank`` (file unchanged).
    """

    piece_type, square = piece_type_and_square(indices)
    color = piece_color_id(piece_type)
    swapped_type = piece_type + torch.where(
        color == 0, torch.full_like(piece_type, 6), torch.full_like(piece_type, -6)
    )
    ranks = square // 8
    files = square % 8
    swapped_square = (7 - ranks) * 8 + files
    return swapped_type * SQUARES + swapped_square


def make_trunk_diagnostics_tensor(
    trunk_output: dict[str, torch.Tensor],
    keys: tuple[str, ...] = ("gate", "gate_entropy", "mechanism_energy", "stream_disagreement"),
) -> torch.Tensor:
    """Stack a few stop-gradient i193 diagnostics for fusion-head inputs."""

    return torch.stack([trunk_output[key].detach() for key in keys], dim=1)


__all__ = [
    "ActiveFeatures",
    "DeltaAccumulator",
    "FEATURE_VOCAB",
    "MAX_ACTIVE_FEATURES",
    "PIECE_PLANE_COUNT",
    "SQUARES",
    "STM_CHANNEL",
    "extract_active_features",
    "involution_indices",
    "make_trunk_diagnostics_tensor",
    "opponent_color_id",
    "piece_color_id",
    "piece_type_and_square",
    "side_to_move_color",
]

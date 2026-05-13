"""Shared ray geometry for blocker-aware occlusion-ray primitives.

Used by p020 (blocker_reset_ray_scan), p021 (occlusion_semiring_ray_scan),
and p023 (occlusion_semiring_delta_bilinear_hyperedge). These primitives
share a fixed deterministic ray structure: from every square on the 8x8
board, walk up to ``RAY_MAX_LEN = 7`` steps along each of 8 queen-style
directions. The walk is rule-derived from the board geometry (no
learnable parameters) and is independent of the position content; only
the ``transmittance`` factor (a function of occupancy along the ray)
depends on the board.

Two lookup tensors are provided:

- ``RAY_STEP_INDEX (8, 64, 7)`` long: ``ray_step_index[d, s, l]`` is the
  flat square index visited at step ``l + 1`` starting from square ``s``
  along direction ``d``. Off-board entries are clamped to ``0`` and
  masked out by ``RAY_STEP_MASK`` so they are safe to ``gather``.
- ``RAY_STEP_MASK (8, 64, 7)`` float: 1.0 if the step is on-board, 0.0
  otherwise. Multiply per-step features by this mask before reducing.

The ``ray_features`` helper applies both at once, returning a
``(B, 8, 64, 7, C)`` tensor whose off-board slots are guaranteed zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

SQUARES = 64
NUM_DIRECTIONS = 8
RAY_MAX_LEN = 7

# Eight queen directions as (drow, dfile). simple_18 uses row 0 at the top
# of the board (rank 8 for white), so "north" here means decreasing row.
DIRECTIONS: tuple[tuple[int, int], ...] = (
    (-1, 0),   # N
    (-1, 1),   # NE
    (0, 1),    # E
    (1, 1),    # SE
    (1, 0),    # S
    (1, -1),   # SW
    (0, -1),   # W
    (-1, -1),  # NW
)


def _row_file(square: int) -> tuple[int, int]:
    return square // 8, square % 8


def _inside(row: int, file: int) -> bool:
    return 0 <= row < 8 and 0 <= file < 8


def build_ray_step_index() -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(ray_step_index, ray_step_mask)``.

    Shapes: both ``(NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN)``. ``ray_step_index``
    is ``long``; off-board slots are clamped to ``0`` and must be masked by
    ``ray_step_mask`` (``float32``, 1.0 / 0.0) before being used as gather
    targets.
    """
    idx = torch.zeros(NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN, dtype=torch.long)
    mask = torch.zeros(NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN, dtype=torch.float32)
    for d, (dr, df) in enumerate(DIRECTIONS):
        for s in range(SQUARES):
            sr, sf = _row_file(s)
            for l in range(RAY_MAX_LEN):
                r = sr + dr * (l + 1)
                f = sf + df * (l + 1)
                if _inside(r, f):
                    idx[d, s, l] = r * 8 + f
                    mask[d, s, l] = 1.0
                else:
                    idx[d, s, l] = 0
                    mask[d, s, l] = 0.0
    return idx, mask


@dataclass(frozen=True)
class RayGeometry:
    """Cached ray index and mask tensors."""

    step_index: torch.Tensor   # (8, 64, 7) long
    step_mask: torch.Tensor    # (8, 64, 7) float

    @classmethod
    def build(cls) -> "RayGeometry":
        idx, mask = build_ray_step_index()
        return cls(step_index=idx, step_mask=mask)


def gather_along_rays(
    features: torch.Tensor,
    step_index: torch.Tensor,
    step_mask: torch.Tensor,
) -> torch.Tensor:
    """Gather per-square features along all rays.

    Args:
        features: ``(B, 64, C)`` per-square feature tensor.
        step_index: ``(8, 64, 7)`` long ray index table from ``build_ray_step_index``.
        step_mask: ``(8, 64, 7)`` float mask of valid steps.

    Returns:
        ``(B, 8, 64, 7, C)`` tensor with off-board steps zeroed.
    """
    if features.ndim != 3:
        raise ValueError(f"Expected features of shape (B, 64, C), got {tuple(features.shape)}")
    batch, n_squares, channels = features.shape
    if n_squares != SQUARES:
        raise ValueError(f"Expected 64 squares, got {n_squares}")
    if step_index.shape != (NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN):
        raise ValueError(f"Unexpected step_index shape {tuple(step_index.shape)}")
    if step_mask.shape != (NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN):
        raise ValueError(f"Unexpected step_mask shape {tuple(step_mask.shape)}")

    flat_idx = step_index.reshape(-1)  # (8*64*7,)
    gathered = features[:, flat_idx, :].reshape(
        batch, NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN, channels
    )
    mask = step_mask.to(device=features.device, dtype=features.dtype).view(
        1, NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN, 1
    )
    return gathered * mask


def gather_scalar_along_rays(
    scalar: torch.Tensor,
    step_index: torch.Tensor,
    step_mask: torch.Tensor,
) -> torch.Tensor:
    """Gather a per-square scalar tensor along all rays.

    Args:
        scalar: ``(B, 64)`` per-square scalar tensor (e.g., occupancy).
        step_index: ``(8, 64, 7)`` long ray index table.
        step_mask: ``(8, 64, 7)`` float mask of valid steps.

    Returns:
        ``(B, 8, 64, 7)`` tensor with off-board steps zeroed.
    """
    if scalar.ndim != 2:
        raise ValueError(f"Expected scalar of shape (B, 64), got {tuple(scalar.shape)}")
    batch, n_squares = scalar.shape
    if n_squares != SQUARES:
        raise ValueError(f"Expected 64 squares, got {n_squares}")
    flat_idx = step_index.reshape(-1)
    gathered = scalar[:, flat_idx].reshape(
        batch, NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN
    )
    mask = step_mask.to(device=scalar.device, dtype=scalar.dtype).view(
        1, NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN
    )
    return gathered * mask

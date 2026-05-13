"""Tests for the shared ray geometry used by p020/p021/p023."""

from __future__ import annotations

import torch

from chess_nn_playground.models.primitives.ray_geometry import (
    DIRECTIONS,
    NUM_DIRECTIONS,
    RAY_MAX_LEN,
    SQUARES,
    RayGeometry,
    gather_along_rays,
    gather_scalar_along_rays,
)


def test_geometry_shapes() -> None:
    g = RayGeometry.build()
    assert g.step_index.shape == (NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN)
    assert g.step_mask.shape == (NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN)
    assert g.step_index.dtype == torch.long
    assert g.step_mask.dtype == torch.float32


def test_eight_distinct_directions() -> None:
    assert len(set(DIRECTIONS)) == NUM_DIRECTIONS == 8


def test_a1_north_ray_visits_a_file() -> None:
    g = RayGeometry.build()
    a1 = 7 * 8 + 0  # row 7, file 0 -> square 56
    # Direction N is (drow=-1, dfile=0) -> direction index 0.
    idx_N = g.step_index[0, a1]
    mask_N = g.step_mask[0, a1]
    assert mask_N.sum().item() == RAY_MAX_LEN
    # Steps from a1 going N: a2 (48), a3 (40), a4 (32), a5 (24), a6 (16), a7 (8), a8 (0).
    assert idx_N.tolist() == [48, 40, 32, 24, 16, 8, 0]


def test_a8_north_ray_is_empty() -> None:
    g = RayGeometry.build()
    a8 = 0
    # From a8 going N (off-board immediately).
    mask_N = g.step_mask[0, a8]
    assert mask_N.sum().item() == 0


def test_gather_along_rays_zeroes_off_board() -> None:
    g = RayGeometry.build()
    feat = torch.arange(SQUARES).float().unsqueeze(0).unsqueeze(-1).repeat(2, 1, 3)
    gathered = gather_along_rays(feat, g.step_index, g.step_mask)
    # Off-board entries must be 0 even if the underlying gather pulled index 0.
    inverted_mask = (1.0 - g.step_mask).bool().unsqueeze(0).unsqueeze(-1)
    masked_out = gathered.masked_select(inverted_mask)
    assert torch.all(masked_out == 0.0)


def test_gather_scalar_along_rays_matches_indexing() -> None:
    g = RayGeometry.build()
    scalar = torch.linspace(0.0, 1.0, SQUARES).unsqueeze(0)  # (1, 64)
    gathered = gather_scalar_along_rays(scalar, g.step_index, g.step_mask)
    # On-board step indices must produce scalar[index].
    a4 = 4 * 8 + 0  # left-edge centre, square 32
    east_idx = g.step_index[2, a4]  # direction 2 = E
    east_mask = g.step_mask[2, a4]
    for l in range(RAY_MAX_LEN):
        if east_mask[l] > 0.5:
            assert torch.isclose(gathered[0, 2, a4, l], scalar[0, east_idx[l]])
        else:
            assert gathered[0, 2, a4, l].item() == 0.0

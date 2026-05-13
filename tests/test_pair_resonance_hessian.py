"""Tests for the DHPE primitive (i245 Pair-Resonance Hessian Network)."""
from __future__ import annotations

import itertools

import pytest
import torch

from chess_nn_playground.models.primitives.pair_resonance_hessian_network import (
    DHPEPrimitiveHead,
    PairResonanceHessianNetwork,
    PhiScorer,
    assemble_variant_boards,
    build_pair_resonance_hessian_network_from_config,
    piece_value_saliency,
    select_top_k_positions,
)
from chess_nn_playground.models.registry import available_models, build_model


def _toy_board(batch: int = 2) -> torch.Tensor:
    board = torch.zeros(batch, 18, 8, 8)
    # white queen on d4 (rank 4, file 3) -> piece plane 4 (white queen)
    # simple_18 rank index is (7 - rank); file index is file
    # so d4 -> rank index 7-3=4, file=3
    board[:, 4, 4, 3] = 1.0
    # white rook on a1 -> rank index 7-0=7, file=0
    board[:, 3, 7, 0] = 1.0
    # white knight on f3 -> rank index 7-2=5, file=5
    board[:, 1, 5, 5] = 1.0
    # black king on h8 -> rank index 7-7=0, file=7; plane 11 (black king)
    board[:, 11, 0, 7] = 1.0
    # black pawn on e7 -> plane 6 (black pawn) rank index 7-6=1, file=4
    board[:, 6, 1, 4] = 1.0
    # side-to-move = white
    board[:, 12, :, :] = 1.0
    return board


def test_piece_value_saliency_zero_for_king_and_empty():
    board = _toy_board(batch=1)
    saliency = piece_value_saliency(board[:, :12])
    # Saliency for black king position (plane 11, rank 0, file 7)
    flat_idx_king = 11 * 64 + 0 * 8 + 7
    assert saliency[0, flat_idx_king] == 0.0  # king prior is 0
    # White queen at plane 4, rank 4, file 3
    flat_idx_queen = 4 * 64 + 4 * 8 + 3
    assert saliency[0, flat_idx_queen] == 9.0
    # All other entries are zero except the 4 non-king pieces
    nonzero = (saliency[0] > 0).sum().item()
    assert nonzero == 4  # P+N+R+Q (king excluded)


def test_select_top_k_picks_high_value_pieces():
    board = _toy_board(batch=1)
    indices, valid = select_top_k_positions(board[:, :12], top_k=3)
    assert indices.shape == (1, 3)
    assert valid.shape == (1, 3)
    saliency = piece_value_saliency(board[:, :12])
    # Each picked slot must point at a non-empty piece-plane position
    for slot in range(3):
        idx = int(indices[0, slot].item())
        if valid[0, slot] > 0:
            assert saliency[0, idx] > 0
    # Top-1 should be the queen
    assert int(indices[0, 0].item()) == 4 * 64 + 4 * 8 + 3
    assert valid[0, 0] == 1.0


def test_select_top_k_marks_invalid_when_too_few_pieces():
    board = torch.zeros(1, 18, 8, 8)
    # only one piece on the board
    board[:, 4, 4, 3] = 1.0  # white queen on d4
    board[:, 12] = 1.0
    indices, valid = select_top_k_positions(board[:, :12], top_k=4)
    # Slot 0 should be valid (the queen) and the rest should be marked invalid.
    assert valid[0, 0] == 1.0
    assert (valid[0, 1:] == 0).all()


def test_assemble_variant_boards_unperturbed_first():
    board = _toy_board(batch=1)
    top_k = 3
    pair_count = top_k * (top_k - 1) // 2
    indices, valid = select_top_k_positions(board[:, :12], top_k=top_k)
    variants = assemble_variant_boards(board, indices, valid, top_k, pair_count)
    assert variants.shape == (1, 1 + top_k + pair_count, 18, 8, 8)
    # Slot 0 must equal the original board
    assert torch.allclose(variants[0, 0], board[0])
    # Slot 1 must have one piece zeroed (relative to slot 0)
    diff = (board[0, :12] - variants[0, 1, :12]).abs()
    assert diff.sum() == 1.0
    # Slot 1 + top_k must zero exactly two pieces vs the original
    pair_diff = (board[0, :12] - variants[0, 1 + top_k, :12]).abs()
    assert pair_diff.sum() == 2.0


def test_dhpe_primitive_head_forward_shapes():
    head = DHPEPrimitiveHead(top_k=3)
    board = _toy_board(batch=4)
    out = head(board)
    assert out["delta_phi"].shape == (4,)
    assert out["dhpe_z_pos"].shape == (4,)
    assert out["dhpe_z_neg"].shape == (4,)
    assert out["dhpe_z_total"].shape == (4,)
    assert out["dhpe_valid_count"].shape == (4,)
    assert out["dhpe_hessian"].shape == (4, 3)  # pair_count = 3
    assert out["dhpe_top_indices"].shape == (4, 3)


def test_dhpe_signed_hessian_recovers_planted_pair_interaction():
    """If phi has an additive piece-value model plus a pairwise bonus on the
    queen-rook pair, the signed Hessian on that pair must be the bonus."""
    head = DHPEPrimitiveHead(top_k=2, phi_channels=8, phi_depth=1)

    class PlantedPhi(torch.nn.Module):
        def __init__(self, bonus: float, idx_q: int, idx_r: int) -> None:
            super().__init__()
            self.bonus = float(bonus)
            self.idx_q = int(idx_q)
            self.idx_r = int(idx_r)

        def forward(self, board: torch.Tensor) -> torch.Tensor:
            flat = board[:, :12].reshape(board.shape[0], -1)
            # additive piece values via the deterministic prior
            values = piece_value_saliency(board[:, :12]).sum(dim=1)
            # pairwise bonus when both queen and rook positions are present
            qr_present = (flat[:, self.idx_q] * flat[:, self.idx_r]).clamp(0.0, 1.0)
            return values + self.bonus * qr_present

    idx_q = 4 * 64 + 4 * 8 + 3  # white queen on d4
    idx_r = 3 * 64 + 7 * 8 + 0  # white rook on a1
    head.phi = PlantedPhi(bonus=7.0, idx_q=idx_q, idx_r=idx_r)
    board = _toy_board(batch=1)
    out = head(board)
    # The top-2 saliency pieces are the queen (saliency 9) and rook (5), so the
    # Hessian on their pair should equal the planted bonus exactly.
    assert out["dhpe_hessian"].shape == (1, 1)
    assert torch.isclose(out["dhpe_hessian"][0, 0], torch.tensor(7.0), atol=1.0e-5)
    assert out["dhpe_z_pos"][0] > 0.0
    assert out["dhpe_z_neg"][0] == 0.0


def test_dhpe_signed_hessian_negative_for_substitutive_pair():
    """Sub-additive scorers must produce a negative Hessian entry."""
    head = DHPEPrimitiveHead(top_k=2)

    class SubAdditivePhi(torch.nn.Module):
        def __init__(self, idx_q: int, idx_r: int) -> None:
            super().__init__()
            self.idx_q = int(idx_q)
            self.idx_r = int(idx_r)

        def forward(self, board: torch.Tensor) -> torch.Tensor:
            flat = board[:, :12].reshape(board.shape[0], -1)
            values = piece_value_saliency(board[:, :12]).sum(dim=1)
            # Sub-additive: bonus only when EXACTLY ONE of the two pieces is present.
            q = flat[:, self.idx_q].clamp(0.0, 1.0)
            r = flat[:, self.idx_r].clamp(0.0, 1.0)
            exactly_one = q + r - 2.0 * q * r
            return values + 4.0 * exactly_one

    idx_q = 4 * 64 + 4 * 8 + 3
    idx_r = 3 * 64 + 7 * 8 + 0
    head.phi = SubAdditivePhi(idx_q=idx_q, idx_r=idx_r)
    board = _toy_board(batch=1)
    out = head(board)
    assert out["dhpe_hessian"][0, 0] < -0.5
    assert out["dhpe_z_neg"][0] > 0.0


def test_no_dhpe_ablation_zeros_pair_features():
    head = DHPEPrimitiveHead(top_k=3, ablation="no_dhpe")
    board = _toy_board(batch=2)
    out = head(board)
    assert torch.all(out["dhpe_z_pos"] == 0)
    assert torch.all(out["dhpe_z_neg"] == 0)
    assert torch.all(out["dhpe_z_total"] == 0)


def test_unsigned_ablation_drops_sign_distinction():
    base_head = DHPEPrimitiveHead(top_k=2)
    abl_head = DHPEPrimitiveHead(top_k=2, ablation="unsigned")
    # Copy weights from base into abl_head to keep the comparison apples-to-apples.
    abl_head.load_state_dict(base_head.state_dict())
    board = _toy_board(batch=3)
    base_out = base_head(board)
    abl_out = abl_head(board)
    # With unsigned aggregation, z_neg must be zero.
    assert torch.all(abl_out["dhpe_z_neg"] == 0)
    # z_pos for unsigned should equal |H| summed.
    assert torch.allclose(
        abl_out["dhpe_z_pos"], base_out["dhpe_hessian"].abs().sum(dim=1), atol=1.0e-5
    )


def test_full_network_forward_shape():
    model = PairResonanceHessianNetwork(top_k=3)
    board = _toy_board(batch=3)
    out = model(board)
    assert out["logits"].shape == (3,)
    assert out["base_logit"].shape == (3,)
    assert out["primitive_delta"].shape == (3,)
    assert out["primitive_gate"].shape == (3,)
    assert out["primitive_gate_logit"].shape == (3,)


def test_trunk_only_ablation_returns_base_logit():
    model = PairResonanceHessianNetwork(top_k=3, ablation="trunk_only")
    board = _toy_board(batch=2)
    out = model(board)
    assert torch.allclose(out["logits"], out["base_logit"], atol=1.0e-5)


def test_zero_gate_keeps_primitive_alive_but_silences_contribution():
    model = PairResonanceHessianNetwork(top_k=3, ablation="zero_gate")
    board = _toy_board(batch=2)
    out = model(board)
    assert torch.allclose(out["logits"], out["base_logit"], atol=1.0e-5)
    # primitive_delta_raw can still be non-zero
    assert out["primitive_delta_raw"].shape == (2,)


def test_registry_build_resolves_pair_resonance_model():
    assert "pair_resonance_hessian_network" in available_models()
    model = build_model(
        "pair_resonance_hessian_network",
        {
            "input_channels": 18,
            "num_classes": 1,
            "trunk_channels": 16,
            "trunk_hidden_dim": 16,
            "trunk_depth": 1,
            "trunk_use_batchnorm": False,
            "phi_channels": 8,
            "phi_depth": 1,
            "dhpe_hidden_dim": 8,
            "gate_hidden_dim": 8,
            "top_k": 2,
        },
    )
    board = _toy_board(batch=2)
    out = model(board)
    assert out["logits"].shape == (2,)


def test_invalid_ablation_raises():
    with pytest.raises(ValueError):
        PairResonanceHessianNetwork(ablation="unknown")
    with pytest.raises(ValueError):
        DHPEPrimitiveHead(ablation="unknown")


def test_invalid_top_k_raises():
    with pytest.raises(ValueError):
        DHPEPrimitiveHead(top_k=1)


def test_phi_scorer_is_real_valued_scalar():
    phi = PhiScorer(channels=8, depth=2)
    out = phi(torch.zeros(5, 18, 8, 8))
    assert out.shape == (5,)
    assert out.dtype == torch.float32


def test_pair_count_matches_combinations():
    head = DHPEPrimitiveHead(top_k=5)
    assert head.pair_count == sum(1 for _ in itertools.combinations(range(5), 2))

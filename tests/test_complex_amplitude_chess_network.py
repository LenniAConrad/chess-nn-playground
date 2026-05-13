"""Tests for the CAIO primitive (i247 Complex-Amplitude Chess Network)."""
from __future__ import annotations

import math

import pytest
import torch

from chess_nn_playground.models.primitives.complex_amplitude_chess_network import (
    CAIOPrimitiveHead,
    ComplexAmplitudeChessNetwork,
    NUM_RELATIONS,
    build_complex_amplitude_chess_network_from_config,
    build_relation_masks,
    color_flip_simple_18,
)
from chess_nn_playground.models.registry import available_models, build_model


def _toy_board(batch: int = 2) -> torch.Tensor:
    board = torch.zeros(batch, 18, 8, 8)
    # white queen on d4, plane 4, rank 4, file 3
    board[:, 4, 4, 3] = 1.0
    # black rook on d8, plane 9, rank 0, file 3
    board[:, 9, 0, 3] = 1.0
    # white king on g1, plane 5, rank 7, file 6
    board[:, 5, 7, 6] = 1.0
    # black king on g8, plane 11, rank 0, file 6
    board[:, 11, 0, 6] = 1.0
    # side-to-move = white
    board[:, 12, :, :] = 1.0
    # castling rights all enabled
    board[:, 13:17, :, :] = 1.0
    return board


def test_relation_masks_shape_and_no_self_edges():
    masks = build_relation_masks()
    assert masks.shape == (NUM_RELATIONS, 64, 64)
    # Diagonal should be zero for every relation (no self-edges).
    for r in range(NUM_RELATIONS):
        diag = masks[r].diag()
        assert torch.all(diag == 0)


def test_king_zone_mask_has_eight_neighbours_for_central_square():
    masks = build_relation_masks()
    king_mask = masks[0]
    # Center square e4 = rank 4 from bottom = simple_18 row 7-3=4, file 4 -> idx 4*8+4 = 36
    center = 4 * 8 + 4
    assert king_mask[center].sum() == 8.0


def test_color_flip_swaps_pieces_and_castling_rights():
    board = _toy_board(batch=1)
    flipped = color_flip_simple_18(board)
    # White queen at plane 4 ⇒ black queen at plane 10 on flipped board
    assert flipped[0, 10, 4, 3] == 1.0
    # Black rook at plane 9 ⇒ white rook at plane 3 on flipped board
    assert flipped[0, 3, 0, 3] == 1.0
    # Side-to-move plane flipped
    assert torch.allclose(flipped[0, 12], 1.0 - board[0, 12])
    # Castling rights swapped
    assert torch.allclose(flipped[0, 13], board[0, 15])
    assert torch.allclose(flipped[0, 14], board[0, 16])
    assert torch.allclose(flipped[0, 15], board[0, 13])
    assert torch.allclose(flipped[0, 16], board[0, 14])
    # En-passant left alone
    assert torch.allclose(flipped[0, 17], board[0, 17])


def test_caio_primitive_head_forward_shapes():
    head = CAIOPrimitiveHead(amplitude_dim=4, feature_channels=8, feature_depth=1)
    board = _toy_board(batch=3)
    out = head(board)
    assert out["delta_phi"].shape == (3,)
    assert out["caio_constructive"].shape == (3, NUM_RELATIONS)
    assert out["caio_destructive"].shape == (3, NUM_RELATIONS)
    assert out["caio_curl"].shape == (3, NUM_RELATIONS)
    assert out["caio_conjugacy_error"].shape == (3,)


def test_caio_no_complex_dtype_leakage_in_output_dict():
    head = CAIOPrimitiveHead(amplitude_dim=4, feature_channels=8, feature_depth=1)
    board = _toy_board(batch=2)
    out = head(board)
    for key, value in out.items():
        if isinstance(value, torch.Tensor):
            assert value.dtype.is_floating_point or value.dtype in {torch.long}, (
                f"output dict key {key!r} has dtype {value.dtype} which is not real-valued"
            )
            assert not value.is_complex(), (
                f"output dict key {key!r} leaks a complex tensor to downstream consumers"
            )


def test_caio_autograd_flows_through_complex_layer():
    head = CAIOPrimitiveHead(amplitude_dim=4, feature_channels=8, feature_depth=1)
    board = _toy_board(batch=2).requires_grad_(False)
    out = head(board)
    loss = out["delta_phi"].sum() + out["caio_constructive"].sum()
    loss.backward()
    grads = [p.grad for p in head.parameters() if p.grad is not None]
    assert grads
    total_norm = torch.stack([g.norm() for g in grads]).norm().item()
    assert total_norm > 0.0


def test_caio_random_phase_ablation_changes_output_against_full():
    base = CAIOPrimitiveHead(amplitude_dim=4, feature_channels=8, feature_depth=1)
    abl = CAIOPrimitiveHead(amplitude_dim=4, feature_channels=8, feature_depth=1, ablation="random_phase")
    abl.load_state_dict(base.state_dict())
    torch.manual_seed(0)
    board = _toy_board(batch=2)
    full = base(board)["delta_phi"]
    torch.manual_seed(0)
    randomised = abl(board)["delta_phi"]
    # With random phase, the output should generally differ from the base.
    assert not torch.allclose(full, randomised, atol=1.0e-4)


def test_caio_real_only_ablation_yields_zero_curl():
    head = CAIOPrimitiveHead(amplitude_dim=4, feature_channels=8, feature_depth=1, ablation="real_only")
    board = _toy_board(batch=2)
    out = head(board)
    # When theta is forced to 0, sin(theta)=0 so amplitudes are real-valued and
    # the curl (sum of imag parts) collapses to zero.
    assert torch.allclose(out["caio_curl"], torch.zeros_like(out["caio_curl"]), atol=1.0e-5)


def test_caio_no_conjugacy_ablation_zeros_conjugacy_error():
    head = CAIOPrimitiveHead(amplitude_dim=4, feature_channels=8, feature_depth=1, ablation="no_conjugacy")
    board = _toy_board(batch=2)
    out = head(board)
    assert torch.all(out["caio_conjugacy_error"] == 0)


def test_caio_constructive_only_zeros_destructive_features():
    head = CAIOPrimitiveHead(amplitude_dim=4, feature_channels=8, feature_depth=1, ablation="constructive_only")
    board = _toy_board(batch=2)
    out = head(board)
    assert torch.all(out["caio_destructive"] == 0)
    assert torch.all(out["caio_curl"] == 0)


def test_caio_no_caio_ablation_zeros_everything():
    head = CAIOPrimitiveHead(amplitude_dim=4, feature_channels=8, feature_depth=1, ablation="no_caio")
    board = _toy_board(batch=2)
    out = head(board)
    assert torch.all(out["caio_constructive"] == 0)
    assert torch.all(out["caio_destructive"] == 0)
    assert torch.all(out["caio_curl"] == 0)
    assert torch.all(out["caio_conjugacy_error"] == 0)


def test_caio_full_network_forward_shape():
    model = ComplexAmplitudeChessNetwork(
        amplitude_dim=4,
        feature_channels=8,
        feature_depth=1,
        trunk_channels=16,
        trunk_hidden_dim=16,
        trunk_depth=1,
        trunk_use_batchnorm=False,
        caio_hidden_dim=16,
        gate_hidden_dim=8,
    )
    board = _toy_board(batch=3)
    out = model(board)
    assert out["logits"].shape == (3,)
    assert out["base_logit"].shape == (3,)
    assert out["primitive_delta"].shape == (3,)
    assert out["primitive_gate"].shape == (3,)


def test_caio_trunk_only_returns_base_logit():
    model = ComplexAmplitudeChessNetwork(
        ablation="trunk_only",
        amplitude_dim=4,
        feature_channels=8,
        feature_depth=1,
        trunk_channels=16,
        trunk_hidden_dim=16,
        trunk_depth=1,
        trunk_use_batchnorm=False,
        caio_hidden_dim=16,
        gate_hidden_dim=8,
    )
    board = _toy_board(batch=2)
    out = model(board)
    assert torch.allclose(out["logits"], out["base_logit"], atol=1.0e-5)


def test_caio_zero_gate_silences_primitive_contribution():
    model = ComplexAmplitudeChessNetwork(
        ablation="zero_gate",
        amplitude_dim=4,
        feature_channels=8,
        feature_depth=1,
        trunk_channels=16,
        trunk_hidden_dim=16,
        trunk_depth=1,
        trunk_use_batchnorm=False,
        caio_hidden_dim=16,
        gate_hidden_dim=8,
    )
    board = _toy_board(batch=2)
    out = model(board)
    assert torch.allclose(out["logits"], out["base_logit"], atol=1.0e-5)
    # primitive_delta_raw is still computed
    assert out["primitive_delta_raw"].shape == (2,)


def test_caio_registry_build_resolves_complex_amplitude_model():
    assert "complex_amplitude_chess_network" in available_models()
    model = build_model(
        "complex_amplitude_chess_network",
        {
            "input_channels": 18,
            "num_classes": 1,
            "trunk_channels": 16,
            "trunk_hidden_dim": 16,
            "trunk_depth": 1,
            "trunk_use_batchnorm": False,
            "amplitude_dim": 4,
            "feature_channels": 8,
            "feature_depth": 1,
            "caio_hidden_dim": 16,
            "gate_hidden_dim": 8,
        },
    )
    board = _toy_board(batch=2)
    out = model(board)
    assert out["logits"].shape == (2,)


def test_caio_invalid_ablation_raises():
    with pytest.raises(ValueError):
        ComplexAmplitudeChessNetwork(ablation="unknown")
    with pytest.raises(ValueError):
        CAIOPrimitiveHead(ablation="unknown")


def test_caio_invalid_amplitude_dim_raises():
    with pytest.raises(ValueError):
        CAIOPrimitiveHead(amplitude_dim=0)


def test_caio_invariance_under_repeated_call():
    """Same board fed twice must give the same output (deterministic forward)."""
    head = CAIOPrimitiveHead(amplitude_dim=4, feature_channels=8, feature_depth=1)
    head.eval()
    board = _toy_board(batch=2)
    with torch.no_grad():
        a = head(board)["delta_phi"]
        b = head(board)["delta_phi"]
    assert torch.allclose(a, b, atol=1.0e-6)


def test_caio_torch_compile_disabled_baseline_works():
    """Eager forward (no torch.compile) must complete without complex-dtype crashes."""
    head = CAIOPrimitiveHead(amplitude_dim=4, feature_channels=8, feature_depth=1)
    board = _toy_board(batch=2)
    out = head(board)
    assert torch.all(torch.isfinite(out["delta_phi"]))
    assert torch.all(torch.isfinite(out["caio_constructive"]))
    assert torch.all(torch.isfinite(out["caio_destructive"]))
    assert torch.all(torch.isfinite(out["caio_curl"]))
    assert torch.all(torch.isfinite(out["caio_conjugacy_error"]))


_ = math  # keep math import in case future tests need it

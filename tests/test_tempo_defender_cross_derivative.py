"""Focused tests for the Tempo-Defender Cross-Derivative primitive (idea i244)."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import yaml

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.registry import available_models, build_model
from chess_nn_playground.models.trunk.tempo_defender_cross_derivative_network import (
    EN_PASSANT_CHANNEL,
    STM_CHANNEL,
    BLACK_PIECE_CHANNELS,
    WHITE_PIECE_CHANNELS,
    SaliencyHead,
    TempoDefenderCrossDerivativeNetwork,
    apply_square_removal,
    build_tempo_defender_cross_derivative_network_from_config,
    tempo_flip_board,
)


REGISTRY_KEY = "tempo_defender_cross_derivative_network"
IDEA_DIR = Path("ideas/registry/i244_tempo_defender_cross_derivative_network")


def _fen_to_tensor(fen: str) -> torch.Tensor:
    return torch.from_numpy(fen_to_simple_18(fen)).unsqueeze(0)


def test_registry_key_is_present():
    assert REGISTRY_KEY in available_models()


def test_tempo_flip_is_involution():
    fen = "r1bqk2r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5"
    board = _fen_to_tensor(fen)
    flipped = tempo_flip_board(board)
    twice = tempo_flip_board(flipped)
    assert torch.allclose(twice, board)
    # Only the stm plane should differ between board and flipped.
    diff = (board != flipped).any(dim=(2, 3))
    assert diff[0, STM_CHANNEL].item() is True
    diff_other = diff.clone()
    diff_other[0, STM_CHANNEL] = False
    assert not diff_other.any().item()


def test_apply_square_removal_zeros_enemy_planes_only():
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    board = _fen_to_tensor(fen)
    # White-to-move, so enemy is black; remove the piece on a8 (rank 0 file 0).
    mask = torch.zeros(1, 64)
    mask[0, 0] = 1.0
    removed = apply_square_removal(board, mask, remove_own=False)
    # The black rook at a8 (channel 6 + ROOK=3 -> channel 9? no - we use 0=P,
    # 1=N, 2=B, 3=R, 4=Q, 5=K; black starts at channel 6). The rook is at
    # plane 6+3=9 in the simple_18 layout used by board_features.
    # We instead check that *all* black piece planes are zero at the square,
    # and that white planes are untouched.
    assert removed[0, BLACK_PIECE_CHANNELS, 0, 0].abs().sum().item() == 0.0
    assert torch.equal(removed[0, WHITE_PIECE_CHANNELS], board[0, WHITE_PIECE_CHANNELS])
    # Other black squares untouched.
    assert removed[0, BLACK_PIECE_CHANNELS, 0, 1].sum().item() == board[0, BLACK_PIECE_CHANNELS, 0, 1].sum().item()


def test_apply_square_removal_respects_side_to_move_when_removing_own():
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
    board = _fen_to_tensor(fen)
    # Black-to-move, so "own" = black; with remove_own=True we should zero
    # black planes at the masked square and leave white untouched.
    mask = torch.zeros(1, 64)
    mask[0, 0] = 1.0
    removed = apply_square_removal(board, mask, remove_own=True)
    assert removed[0, BLACK_PIECE_CHANNELS, 0, 0].abs().sum().item() == 0.0
    assert torch.equal(removed[0, WHITE_PIECE_CHANNELS], board[0, WHITE_PIECE_CHANNELS])


def test_saliency_head_masks_to_enemy_pieces():
    head = SaliencyHead(input_channels=18, hidden=8, topk=3)
    fen = "8/8/8/3k4/8/8/4K3/8 w - - 0 1"
    board = _fen_to_tensor(fen)
    out = head(board)
    # Only one enemy piece (black king on d5 -> rank 3, file 3 -> idx 27).
    assert out.enemy_mask.sum().item() == 1.0
    assert int(out.top_indices[0, 0].item()) == 27
    # Validity mask should mark only the first slot as valid.
    assert out.top_valid[0, 0].item() == 1.0
    assert out.top_valid[0, 1].item() == 0.0


def test_model_forward_shape_and_diagnostics_keys():
    model = build_model(
        REGISTRY_KEY,
        {
            "input_channels": 18,
            "num_classes": 1,
            "topk": 3,
            "tdcd_channels": 16,
            "tdcd_depth": 1,
            "trunk_channels": 16,
            "trunk_hidden_dim": 24,
            "trunk_depth": 1,
        },
    )
    model.eval()
    fens = [
        "r1bqk2r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
    ]
    batch = torch.cat([_fen_to_tensor(fen) for fen in fens], dim=0)
    with torch.no_grad():
        out = model(batch)
    assert out["logits"].shape == (2,)
    expected_keys = {
        "logits",
        "base_logit",
        "primitive_delta",
        "primitive_gate",
        "primitive_gate_entropy",
        "primitive_logit_contribution",
        "saliency_entropy",
        "saliency_concentration",
        "saliency_top_valid_count",
        "g_T_norm",
        "max_dd",
        "mean_dd",
        "std_dd",
        "topk_dd_ratio",
        "valid_slot_count",
        "mechanism_energy",
        "proposal_profile_strength",
        "proposal_keyword_count",
        "delta_delta_per_slot",
    }
    missing = expected_keys - set(out.keys())
    assert not missing, f"Missing diagnostics keys: {sorted(missing)}"


def test_model_initial_gate_keeps_primitive_near_zero():
    # gate_init=-2.0 -> sigmoid(-2) ~= 0.119; the contribution should remain
    # small at init since head weights are zero-mean and the fingerprint
    # passes through a LayerNorm.
    torch.manual_seed(0)
    model = build_model(
        REGISTRY_KEY,
        {
            "topk": 3,
            "gate_init": -2.0,
            "tdcd_channels": 16,
            "tdcd_depth": 1,
            "trunk_channels": 16,
            "trunk_hidden_dim": 24,
            "trunk_depth": 1,
        },
    )
    model.eval()
    fen = "r1bqk2r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5"
    with torch.no_grad():
        out = model(_fen_to_tensor(fen))
    gate = float(out["primitive_gate"].item())
    contribution = float(out["primitive_logit_contribution"].abs().item())
    assert gate < 0.25, f"initial gate too open: {gate}"
    assert contribution < 0.5, f"initial primitive contribution too large: {contribution}"


def test_mixed_partial_zero_for_irrelevant_defender():
    """When sigma_T and delta_k commute on the input *and* the encoder is
    affine, the mixed partial must vanish. We swap the encoder for an
    `nn.Identity`-like pooling so the test isolates the algebraic claim.
    """
    model = build_model(
        REGISTRY_KEY,
        {
            "topk": 2,
            "tdcd_channels": 8,
            "tdcd_depth": 1,
            "trunk_channels": 8,
            "trunk_hidden_dim": 16,
            "trunk_depth": 1,
        },
    )

    # Replace TDCD encoder with mean+max pooling so the feature is a
    # closed-form linear function of the board, ensuring the mixed partial
    # is mathematically zero.
    class _LinearPool(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.output_dim = 18 * 2

        def forward(self, board: torch.Tensor) -> torch.Tensor:
            return torch.cat([board.mean(dim=(2, 3)), board.amax(dim=(2, 3))], dim=1)

    model.encoder = _LinearPool()
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    board = _fen_to_tensor(fen)
    with torch.no_grad():
        out = model(board)
    # For a closed-form linear feature, ||tau_k|| should equal ||g_T|| since
    # sigma_T flips only the stm plane and delta_k zeros disjoint piece
    # planes; the cross-derivative term vanishes.
    dd = out["delta_delta_per_slot"][0]
    assert dd.abs().max().item() < 1.0e-3, f"mixed partial should be ~0 for linear pool, got {dd}"


def test_skip_cross_derivative_matches_baseline_within_gate_offset():
    """With ablation='skip_cross_derivative' the head's gate is forced to
    zero, so the final logit should equal the trunk's base logit."""
    torch.manual_seed(0)
    model = build_model(
        REGISTRY_KEY,
        {
            "topk": 3,
            "tdcd_channels": 16,
            "tdcd_depth": 1,
            "trunk_channels": 16,
            "trunk_hidden_dim": 24,
            "trunk_depth": 1,
            "ablation": "skip_cross_derivative",
        },
    )
    model.eval()
    fen = "r1bqk2r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5"
    with torch.no_grad():
        out = model(_fen_to_tensor(fen))
    assert torch.allclose(out["logits"], out["base_logit"], atol=1.0e-6)


def test_attacker_perturbation_changes_grid_relative_to_default():
    """The `attacker_perturbation` ablation should produce different
    primitive features from the default `none` ablation on a position with
    at least one defender."""
    torch.manual_seed(0)
    config = {
        "topk": 2,
        "tdcd_channels": 12,
        "tdcd_depth": 1,
        "trunk_channels": 12,
        "trunk_hidden_dim": 16,
        "trunk_depth": 1,
    }
    model_default = build_model(REGISTRY_KEY, {**config, "ablation": "none"}).eval()
    model_attacker = build_model(REGISTRY_KEY, {**config, "ablation": "attacker_perturbation"}).eval()
    # Force identical weights so the only difference is the perturbation.
    model_attacker.load_state_dict(model_default.state_dict())
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/3P4/2N5/PPP1PPPP/R1BQKBNR w KQkq - 0 1"
    board = _fen_to_tensor(fen)
    with torch.no_grad():
        out_default = model_default(board)
        out_attacker = model_attacker(board)
    delta_diff = (out_default["delta_delta_per_slot"] - out_attacker["delta_delta_per_slot"]).abs()
    assert delta_diff.max().item() > 1.0e-4, "attacker ablation should perturb the grid differently"


def test_handles_position_with_no_enemy_pieces():
    """Edge case: empty enemy side. Saliency must not crash and the head
    must produce finite output."""
    model = build_model(
        REGISTRY_KEY,
        {
            "topk": 3,
            "tdcd_channels": 12,
            "tdcd_depth": 1,
            "trunk_channels": 12,
            "trunk_hidden_dim": 16,
            "trunk_depth": 1,
        },
    ).eval()
    fen = "8/8/8/8/8/8/4K3/8 w - - 0 1"  # white king only, no enemy pieces
    with torch.no_grad():
        out = model(_fen_to_tensor(fen))
    assert torch.isfinite(out["logits"]).all().item()
    # All saliency slots should be invalid.
    assert out["saliency_top_valid_count"][0].item() == 0.0


def test_config_yaml_passes_static_validation():
    """The promoted config must use the simple_18 split contract."""
    config_path = IDEA_DIR / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["idea_id"] == "i244"
    assert config["mode"] == "puzzle_binary"
    assert config["device"] == "nvidia"
    assert config["data"]["encoding"] == "simple_18"
    assert config["model"]["name"] == REGISTRY_KEY
    assert config["model"]["num_classes"] == 1
    assert config["model"]["input_channels"] == 18


def test_backward_pass_runs_under_bce_loss():
    """A short forward+backward cycle confirms gradients flow through the
    cross-derivative head and the trunk."""
    torch.manual_seed(0)
    model = build_model(
        REGISTRY_KEY,
        {
            "topk": 2,
            "tdcd_channels": 16,
            "tdcd_depth": 1,
            "trunk_channels": 16,
            "trunk_hidden_dim": 24,
            "trunk_depth": 1,
        },
    ).train()
    fens = [
        "r1bqk2r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
    ]
    batch = torch.cat([_fen_to_tensor(fen) for fen in fens], dim=0)
    target = torch.tensor([1.0, 0.0])
    out = model(batch)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out["logits"], target)
    loss.backward()
    # Ensure at least one parameter received a non-zero gradient in the head
    # and in the trunk.
    head_grad = any(
        param.grad is not None and param.grad.abs().sum() > 0
        for param in model.head.parameters()
    )
    trunk_grad = any(
        param.grad is not None and param.grad.abs().sum() > 0
        for param in model.trunk.parameters()
    )
    assert head_grad, "discriminator head received no gradient"
    assert trunk_grad, "i193 trunk received no gradient"

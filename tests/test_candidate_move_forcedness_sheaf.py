from __future__ import annotations

import torch

from chess_nn_playground.models.registry import build_model
from chess_nn_playground.models.trunk.candidate_move_forcedness_sheaf import (
    CandidateMoveForcednessSheafNet,
    MOVE_FLAG_COUNT,
    MOVE_KIND_COUNT,
)


def _config(**overrides):
    cfg = {
        "input_channels": 18,
        "num_classes": 1,
        "channels": 64,
        "hidden_dim": 96,
        "depth": 2,
        "stalk_dim": 8,
        "dropout": 0.0,
        "encoding": "simple_18",
        "max_candidates": 96,
        "top_k": 8,
    }
    cfg.update(overrides)
    return cfg


def _sample(batch_size: int) -> torch.Tensor:
    x = torch.rand(batch_size, 18, 8, 8)
    x[:, :12] = (x[:, :12] > 0.92).float()
    x[:, 12] = 1.0
    return x


def _structured_sample(batch_size: int) -> torch.Tensor:
    x = torch.zeros(batch_size, 18, 8, 8)
    x[:, 12] = 1.0
    x[:, 5, 7, 4] = 1.0      # our queen on h1 (canonical mover)
    x[:, 11, 0, 4] = 1.0     # their queen on h8
    x[:, 0, 6, 4] = 1.0      # our pawn on g7
    x[:, 6, 1, 4] = 1.0      # our king on g1
    x[:, 3, 7, 0] = 1.0      # our bishop on a1
    x[:, 10, 5, 0] = 1.0     # their rook on a3
    return x


def test_i251_builds_through_registry() -> None:
    model = build_model("candidate_move_forcedness_sheaf", _config())
    assert isinstance(model, CandidateMoveForcednessSheafNet)


def test_i251_forward_returns_logits_and_candidate_diagnostics() -> None:
    torch.manual_seed(0)
    model = build_model("candidate_move_forcedness_sheaf", _config()).eval()
    x = _structured_sample(3)
    with torch.no_grad():
        out = model(x)

    assert isinstance(out, dict)
    assert out["logits"].shape == (3,)
    assert torch.isfinite(out["logits"]).all()
    for key in (
        "candidate_base_logits",
        "candidate_delta_logits",
        "candidate_gate",
        "candidate_entropy",
        "candidate_top1_mass",
        "candidate_gap",
        "candidate_check_mass",
        "candidate_capture_mass",
        "candidate_promotion_mass",
        "candidate_underpromotion_mass",
        "candidate_pin_mass",
        "candidate_king_zone_mass",
        "candidate_overflow_count",
        "candidate_count",
        "sheaf_tension",
        "pin_pressure",
    ):
        assert key in out, f"missing diagnostic key {key}"
        assert torch.isfinite(out[key]).all()


def test_i251_zero_init_matches_i018_exactly() -> None:
    """Default head + zero init should produce final_logit == base_logit at init."""
    torch.manual_seed(42)
    base = build_model("oriented_tactical_sheaf_laplacian", _config()).eval()
    i251 = build_model("candidate_move_forcedness_sheaf", _config()).eval()

    i251_state = i251.state_dict()
    copied = 0
    for k, v in base.state_dict().items():
        if k in i251_state and i251_state[k].shape == v.shape:
            i251_state[k] = v
            copied += 1
    i251.load_state_dict(i251_state, strict=False)
    assert copied >= 50, f"expected to copy all i018 tensors, copied {copied}"

    x = _structured_sample(4)
    with torch.no_grad():
        base_logits = base(x)["logits"]
        i251_out = i251(x)
        i251_logits = i251_out["logits"]
    max_diff = (base_logits - i251_logits).abs().max().item()
    assert max_diff < 1.0e-5, f"zero-init i251 should match i018 within FP noise, got {max_diff:.3e}"
    # candidate_base_logits should equal i018 base output and final logits
    base_diff = (base_logits - i251_out["candidate_base_logits"]).abs().max().item()
    assert base_diff < 1.0e-5, f"candidate_base_logits should equal i018 logits, got {base_diff:.3e}"


def test_i251_disable_move_branch_matches_i018_exactly() -> None:
    """`disable_move_branch=True` is the i018-equivalent path."""
    torch.manual_seed(7)
    base = build_model("oriented_tactical_sheaf_laplacian", _config()).eval()
    disabled = build_model(
        "candidate_move_forcedness_sheaf", _config(disable_move_branch=True)
    ).eval()

    disabled_state = disabled.state_dict()
    for k, v in base.state_dict().items():
        if k in disabled_state and disabled_state[k].shape == v.shape:
            disabled_state[k] = v
    disabled.load_state_dict(disabled_state, strict=False)

    x = _structured_sample(4)
    with torch.no_grad():
        base_logits = base(x)["logits"]
        disabled_logits = disabled(x)["logits"]
    diff = (base_logits - disabled_logits).abs().max().item()
    assert diff == 0.0, f"disable_move_branch path must equal i018 exactly, got {diff:.3e}"


def test_i251_pool_weights_sum_to_one_when_candidates_exist() -> None:
    """top1_mass should be <= 1 and entropy non-negative on valid pools."""
    torch.manual_seed(11)
    model = build_model("candidate_move_forcedness_sheaf", _config()).eval()
    x = _structured_sample(2)
    with torch.no_grad():
        out = model(x)
    assert (out["candidate_top1_mass"] <= 1.0 + 1e-5).all()
    assert (out["candidate_top1_mass"] >= 0.0 - 1e-5).all()
    assert (out["candidate_entropy"] >= -1e-5).all()
    # At zero-init the score head is zero so the top-k softmax is uniform;
    # entropy should be log(top_k) approximately.
    expected_entropy = torch.log(torch.tensor(float(model.top_k)))
    assert torch.allclose(
        out["candidate_entropy"], expected_entropy.expand_as(out["candidate_entropy"]),
        atol=1e-4,
    )


def test_i251_move_branch_parameters_are_trainable() -> None:
    """Loss must produce non-zero gradient on the delta head's output layer.

    At zero-init the chain rule kills gradient on the *hidden* delta and gate
    weights -- the residual is exactly 0 there by design. But the delta head's
    *output* layer must still see gradient via
    `dL/dW_delta_out = dL/dlogit * sigmoid(gate) * hidden_activations`, which
    is non-zero because `sigmoid(0) = 0.5` and the trunk produces non-zero
    activations.
    """
    torch.manual_seed(13)
    model = build_model("candidate_move_forcedness_sheaf", _config()).train()
    x = _structured_sample(3)
    y = torch.tensor([1.0, 0.0, 1.0])
    out = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out["logits"], y)
    loss.backward()

    delta_out_grad = model.delta_head[-1].weight.grad
    assert delta_out_grad is not None
    assert delta_out_grad.abs().sum().item() > 0.0, (
        "delta_head output layer must receive nonzero gradient at zero init"
    )
    # After a single optimizer step from zero init, hidden gradients become
    # non-zero too -- but we don't run training here. The output-layer
    # gradient alone proves the chain to the move branch is connected.


def test_i251_scramble_relations_does_not_break_forward() -> None:
    torch.manual_seed(17)
    model = build_model(
        "candidate_move_forcedness_sheaf", _config(scramble_relations=True)
    ).eval()
    x = _structured_sample(2)
    with torch.no_grad():
        out = model(x)
    assert torch.isfinite(out["logits"]).all()
    assert torch.isfinite(out["candidate_entropy"]).all()


def test_i251_flat_move_pool_runs_and_emits_diagnostics() -> None:
    torch.manual_seed(19)
    model = build_model(
        "candidate_move_forcedness_sheaf", _config(flat_move_pool=True)
    ).eval()
    x = _structured_sample(2)
    with torch.no_grad():
        out = model(x)
    assert torch.isfinite(out["logits"]).all()
    # With uniform scores the top1_mass equals 1/top_k.
    expected = 1.0 / float(model.top_k)
    assert torch.allclose(
        out["candidate_top1_mass"],
        torch.full_like(out["candidate_top1_mass"], expected),
        atol=1e-4,
    )


def test_i251_candidate_count_does_not_exceed_max() -> None:
    torch.manual_seed(23)
    model = build_model("candidate_move_forcedness_sheaf", _config(max_candidates=32)).eval()
    x = _sample(4)
    with torch.no_grad():
        out = model(x)
    assert (out["candidate_count"] <= 32.0 + 1e-6).all()
    assert (out["candidate_count"] >= 0.0).all()


def test_i251_flag_count_matches_module_constant() -> None:
    assert MOVE_FLAG_COUNT == 9
    assert MOVE_KIND_COUNT >= 2

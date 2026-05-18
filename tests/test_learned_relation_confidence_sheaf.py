from __future__ import annotations

import torch

from chess_nn_playground.models.registry import build_model
from chess_nn_playground.models.trunk.learned_relation_confidence_sheaf import (
    LearnedRelationConfidenceSheafNet,
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
    }
    cfg.update(overrides)
    return cfg


def _sample(batch_size: int) -> torch.Tensor:
    x = torch.rand(batch_size, 18, 8, 8)
    x[:, :12] = (x[:, :12] > 0.92).float()
    x[:, 12] = 1.0
    return x


def test_i250_builds_through_registry() -> None:
    model = build_model("learned_relation_confidence_sheaf", _config())
    assert isinstance(model, LearnedRelationConfidenceSheafNet)


def test_i250_forward_returns_logits_and_confidence_diagnostics() -> None:
    torch.manual_seed(0)
    model = build_model("learned_relation_confidence_sheaf", _config()).eval()
    x = _sample(3)
    with torch.no_grad():
        out = model(x)

    assert isinstance(out, dict)
    assert out["logits"].shape == (3,)
    assert torch.isfinite(out["logits"]).all()
    for key in (
        "confidence_mean",
        "confidence_max",
        "confidence_std",
        "pin_edge_confidence",
        "king_zone_confidence",
        "sheaf_tension",
        "pin_pressure",
    ):
        assert key in out, f"missing diagnostic key {key}"
        assert torch.isfinite(out[key]).all()


def test_i250_zero_init_matches_i018_within_fp_noise() -> None:
    """Default head + zero init + relation-wise normalization should be i018 at init."""
    torch.manual_seed(42)
    base = build_model("oriented_tactical_sheaf_laplacian", _config()).eval()
    i250 = build_model("learned_relation_confidence_sheaf", _config()).eval()

    # Copy the i018 weights into the shared parameters of i250.
    i250_state = i250.state_dict()
    copied = 0
    for k, v in base.state_dict().items():
        if k in i250_state and i250_state[k].shape == v.shape:
            i250_state[k] = v
            copied += 1
    i250.load_state_dict(i250_state, strict=False)
    assert copied >= 50, f"expected to copy all i018 tensors, copied {copied}"

    x = _sample(4)
    with torch.no_grad():
        base_logits = base(x)["logits"]
        i250_logits = i250(x)["logits"]
    max_diff = (base_logits - i250_logits).abs().max().item()
    assert max_diff < 1.0e-5, f"zero-init i250 should match i018 within FP noise, got {max_diff:.3e}"


def test_i250_flat_confidence_matches_i018_exactly() -> None:
    """flat_confidence=True should reproduce i018 logits exactly with shared weights."""
    torch.manual_seed(7)
    base = build_model("oriented_tactical_sheaf_laplacian", _config()).eval()
    flat = build_model(
        "learned_relation_confidence_sheaf", _config(flat_confidence=True)
    ).eval()

    flat_state = flat.state_dict()
    for k, v in base.state_dict().items():
        if k in flat_state and flat_state[k].shape == v.shape:
            flat_state[k] = v
    flat.load_state_dict(flat_state, strict=False)

    x = _sample(4)
    with torch.no_grad():
        base_logits = base(x)["logits"]
        flat_logits = flat(x)["logits"]
    diff = (base_logits - flat_logits).abs().max().item()
    assert diff == 0.0, f"flat_confidence path must equal i018 exactly, got {diff:.3e}"


def test_i250_normalized_confidence_has_unit_mean_at_init() -> None:
    """The normalization step should make alpha_hat have mean 1 over active edges."""
    torch.manual_seed(11)
    model = build_model("learned_relation_confidence_sheaf", _config()).eval()
    x = _sample(2)
    with torch.no_grad():
        out = model(x)
    # confidence_mean is the per-batch mean over relations of the per-relation mean
    # over active edges of alpha_hat. At init alpha_hat == 1, so the value must be 1.
    assert torch.allclose(out["confidence_mean"], torch.ones_like(out["confidence_mean"]))


def test_i250_confidence_head_is_trainable() -> None:
    """Loss must produce non-zero gradient on the confidence head parameters."""
    torch.manual_seed(13)
    model = build_model("learned_relation_confidence_sheaf", _config()).train()
    x = _sample(3)
    y = torch.tensor([1.0, 0.0, 1.0])
    out = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out["logits"], y)
    loss.backward()

    confidence_named = [
        (name, param) for name, param in model.named_parameters() if "confidence" in name
    ]
    assert confidence_named, "expected at least one parameter in the confidence head"
    # The relation_bias should always see gradient. The zero-initialized output
    # heads see gradient through the normalized alpha_hat, but the normalization
    # divides out the constant init, so we only require relation_bias.
    bias_grad = None
    for name, param in confidence_named:
        if "relation_bias" in name:
            bias_grad = param.grad
            break
    assert bias_grad is not None, "relation_bias should be present"
    assert bias_grad.abs().max().item() > 0.0, "relation_bias must receive nonzero gradient"


def test_i250_scramble_relations_does_not_break_forward() -> None:
    torch.manual_seed(17)
    model = build_model(
        "learned_relation_confidence_sheaf", _config(scramble_relations=True)
    ).eval()
    x = _sample(2)
    with torch.no_grad():
        out = model(x)
    assert torch.isfinite(out["logits"]).all()

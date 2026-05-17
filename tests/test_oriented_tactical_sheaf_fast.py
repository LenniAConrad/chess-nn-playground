from __future__ import annotations

import torch

from chess_nn_playground.models.registry import build_model


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
        "compile_model": False,
    }
    cfg.update(overrides)
    return cfg


def _sample(batch_size: int) -> torch.Tensor:
    x = torch.rand(batch_size, 18, 8, 8)
    x[:, :12] = (x[:, :12] > 0.92).float()
    x[:, 12] = 1.0
    return x


def test_i249_loads_i018_state_and_matches_logits_and_gradients() -> None:
    torch.manual_seed(1234)
    base = build_model("oriented_tactical_sheaf_laplacian", _config()).eval()
    fast = build_model("oriented_tactical_sheaf_fast", _config(return_diagnostics=True)).eval()
    fast.load_state_dict(base.state_dict(), strict=True)

    x = _sample(4)
    with torch.no_grad():
        base_out = base(x)
        fast_out = fast(x)

    assert (base_out["logits"] - fast_out["logits"]).abs().max().item() < 1e-5
    assert (base_out["sheaf_tension"] - fast_out["sheaf_tension"]).abs().max().item() < 1e-5
    assert (base_out["pin_pressure"] - fast_out["pin_pressure"]).abs().max().item() < 1e-5

    base.train()
    fast.train()
    fast.load_state_dict(base.state_dict(), strict=True)
    y = torch.rand(4)
    base_loss = torch.nn.functional.binary_cross_entropy_with_logits(base(x)["logits"].view(-1), y)
    fast_loss = torch.nn.functional.binary_cross_entropy_with_logits(fast(x)["logits"].view(-1), y)
    base_loss.backward()
    fast_loss.backward()

    max_grad_diff = 0.0
    for (_, base_param), (_, fast_param) in zip(base.named_parameters(), fast.named_parameters()):
        if base_param.grad is None or fast_param.grad is None:
            continue
        max_grad_diff = max(max_grad_diff, (base_param.grad - fast_param.grad).abs().max().item())
    assert max_grad_diff < 1e-6


def test_i249_logits_only_mode_keeps_logits() -> None:
    torch.manual_seed(4321)
    full = build_model("oriented_tactical_sheaf_fast", _config(return_diagnostics=True)).eval()
    logits_only = build_model("oriented_tactical_sheaf_fast", _config(return_diagnostics=False)).eval()
    logits_only.load_state_dict(full.state_dict(), strict=True)

    x = _sample(3)
    with torch.no_grad():
        full_out = full(x)
        logits_only_out = logits_only(x)

    assert set(logits_only_out) == {"logits"}
    assert (full_out["logits"] - logits_only_out["logits"]).abs().max().item() < 1e-6

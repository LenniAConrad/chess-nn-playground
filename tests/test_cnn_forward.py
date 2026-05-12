from __future__ import annotations

import torch

from chess_nn_playground.models.trunk.cnn import SimpleChessCNN
from chess_nn_playground.models.registry import available_models, build_model


def test_cnn_forward_shape():
    model = SimpleChessCNN(input_channels=18, num_classes=2, channels=8, num_blocks=1)
    logits = model(torch.zeros(3, 18, 8, 8))
    assert logits.shape == (3, 2)


def test_registered_residual_cnn_forward_shape():
    assert "residual_cnn" in available_models()
    model = build_model(
        "residual_cnn",
        {"input_channels": 18, "num_classes": 3, "channels": 8, "num_blocks": 1},
    )
    logits = model(torch.zeros(3, 18, 8, 8))
    assert logits.shape == (3, 3)

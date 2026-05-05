from __future__ import annotations

import torch

from chess_nn_playground.models.registry import available_models, build_model


def test_signal_benchmark_models_are_registered():
    expected = {
        "stockfish_nnue",
        "mlp",
        "simple_cnn",
        "lc0_bt4_classifier",
    }
    assert expected.issubset(set(available_models()))


def test_mlp_forward_shape():
    model = build_model(
        "mlp",
        {"input_channels": 18, "num_classes": 2, "hidden_dims": [32, 16], "dropout": 0.0},
    )
    logits = model(torch.zeros(4, 18, 8, 8))
    assert logits.shape == (4, 2)


def test_stockfish_style_nnue_forward_shape():
    x = torch.zeros(4, 18, 8, 8)
    x[:, 12] = 1.0
    model = build_model(
        "stockfish_nnue",
        {
            "input_channels": 18,
            "num_classes": 3,
            "accumulator_size": 32,
            "hidden_dims": [16],
            "dropout": 0.0,
        },
    )
    logits = model(x)
    assert logits.shape == (4, 3)


def test_lc0_bt4_classifier_forward_shape():
    model = build_model(
        "lc0_bt4_classifier",
        {
            "input_channels": 112,
            "num_classes": 2,
            "channels": 16,
            "num_blocks": 1,
            "value_channels": 4,
            "value_hidden": 16,
            "se_channels": 4,
            "dropout": 0.0,
        },
    )
    logits = model(torch.zeros(2, 112, 8, 8))
    assert logits.shape == (2, 2)

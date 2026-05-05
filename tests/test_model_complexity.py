from __future__ import annotations

from chess_nn_playground.models.complexity import estimate_model_complexity_from_config


def test_model_complexity_estimate_counts_simple_cnn_flops():
    config = {
        "model": {
            "name": "simple_cnn",
            "input_channels": 18,
            "num_classes": 1,
            "channels": 8,
            "num_blocks": 2,
            "use_batchnorm": False,
            "dropout": 0.0,
        }
    }

    estimate = estimate_model_complexity_from_config(config)

    assert estimate["status"] == "estimated"
    assert estimate["estimated_macs_per_position"] > 0
    assert estimate["estimated_flops_per_position"] >= 2 * estimate["estimated_macs_per_position"]
    assert estimate["estimated_mflops_per_position"] > 0
    assert estimate["trainable_parameters"] > 0
    assert estimate["counted_modules"]["Conv2d"] == 2
    assert estimate["counted_modules"]["Linear"] == 1

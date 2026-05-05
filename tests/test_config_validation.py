from __future__ import annotations

import pandas as pd

from chess_nn_playground.training.config_validation import validate_training_config


def _base_config(train_path, val_path):
    return {
        "run": {"name": "validation_test", "output_dir": "results"},
        "mode": "puzzle_binary",
        "device": "cpu",
        "data": {
            "train_path": str(train_path),
            "val_path": str(val_path),
            "encoding": "simple_18",
        },
        "model": {"name": "simple_cnn", "input_channels": 18, "num_classes": 1},
        "training": {"epochs": 1, "batch_size": 2},
    }


def test_config_validation_rejects_empty_training_split(tmp_path):
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    pd.DataFrame(columns=["normalized_fen", "fine_label"]).to_parquet(train_path, index=False)
    pd.DataFrame(
        [{"normalized_fen": "8/8/8/8/8/8/8/K6k w - - 0 1", "fine_label": 0}]
    ).to_parquet(val_path, index=False)

    messages = validate_training_config(
        _base_config(train_path, val_path),
        tmp_path / "config.yaml",
        require_device_available=False,
    )

    assert any("data.train_path is empty" in message for message in messages)


def test_config_validation_rejects_missing_label_column(tmp_path):
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    frame = pd.DataFrame([{"normalized_fen": "8/8/8/8/8/8/8/K6k w - - 0 1"}])
    frame.to_parquet(train_path, index=False)
    frame.to_parquet(val_path, index=False)

    messages = validate_training_config(
        _base_config(train_path, val_path),
        tmp_path / "config.yaml",
        require_device_available=False,
    )

    assert any("missing required column" in message for message in messages)

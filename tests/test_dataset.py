from __future__ import annotations

import pandas as pd

from chess_nn_playground.data.dataset import ChessPositionDataset, collate_positions


def test_dataset_item_loading(tmp_path):
    path = tmp_path / "split.parquet"
    pd.DataFrame(
        [
            {
                "sample_id": "a",
                "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                "normalized_fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                "coarse_label": 0,
                "fine_label": 0,
                "label_status": "known_non_puzzle",
                "source_group_id": None,
                "sister_group_id": None,
                "split_group_id": "a",
                "source_file": "test",
            }
        ]
    ).to_parquet(path, index=False)
    dataset = ChessPositionDataset(path, mode="coarse_binary")
    item = dataset[0]
    assert item["x"].shape[1:] == (8, 8)
    assert item["y"].item() == 0
    assert item["sample_id"] == "a"


def test_dataset_item_loading_lc0_bt4_encoding(tmp_path):
    path = tmp_path / "split.parquet"
    pd.DataFrame(
        [
            {
                "sample_id": "a",
                "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                "normalized_fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                "coarse_label": 0,
                "fine_label": 0,
                "label_status": "known_non_puzzle",
                "source_group_id": None,
                "sister_group_id": None,
                "split_group_id": "a",
                "source_file": "test",
            }
        ]
    ).to_parquet(path, index=False)
    dataset = ChessPositionDataset(path, mode="coarse_binary", encoding="lc0_bt4_112")
    item = dataset[0]
    assert item["x"].shape == (112, 8, 8)
    assert item["y"].item() == 0


def test_dataset_preserves_optional_crtk_metadata(tmp_path):
    path = tmp_path / "split.parquet"
    pd.DataFrame(
        [
            {
                "sample_id": "a",
                "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                "normalized_fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                "coarse_label": 0,
                "fine_label": 0,
                "label_status": "known_non_puzzle",
                "source_group_id": None,
                "sister_group_id": None,
                "split_group_id": "a",
                "source_file": "test",
                "crtk_difficulty": "easy",
                "crtk_phase": "endgame",
                "crtk_tactic_motifs": "pin|fork",
                "crtk_eval_cp": 42,
            }
        ]
    ).to_parquet(path, index=False)
    dataset = ChessPositionDataset(path, mode="coarse_binary")
    item = dataset[0]
    assert item["metadata"]["crtk_difficulty"] == "easy"
    assert item["metadata"]["crtk_phase"] == "endgame"
    assert item["metadata"]["crtk_tactic_motifs"] == "pin|fork"
    assert item["metadata"]["crtk_eval_cp"] == 42


def test_puzzle_binary_mode_maps_only_fine_label_two_to_positive(tmp_path):
    path = tmp_path / "split.parquet"
    pd.DataFrame(
        [
            {
                "sample_id": f"s{i}",
                "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                "normalized_fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                "coarse_label": 1 if fine_label else 0,
                "fine_label": fine_label,
                "label_status": "test",
                "source_group_id": None,
                "sister_group_id": None,
                "split_group_id": f"s{i}",
                "source_file": "test",
            }
            for i, fine_label in enumerate([0, 1, 2])
        ]
    ).to_parquet(path, index=False)
    dataset = ChessPositionDataset(path, mode="puzzle_binary")
    assert [dataset[i]["y"].item() for i in range(len(dataset))] == [0, 0, 1]


def test_dataset_can_emit_rule_texture(tmp_path):
    path = tmp_path / "split.parquet"
    pd.DataFrame(
        [
            {
                "sample_id": "a",
                "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                "normalized_fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                "coarse_label": 0,
                "fine_label": 0,
                "label_status": "known_non_puzzle",
                "source_group_id": None,
                "sister_group_id": None,
                "split_group_id": "a",
                "source_file": "test",
            }
        ]
    ).to_parquet(path, index=False)

    dataset = ChessPositionDataset(path, mode="coarse_binary", include_rule_texture=True)
    item = dataset[0]
    assert 0.0 <= item["rule_texture"].item() <= 1.0

    batch = collate_positions([item])
    assert batch["rule_texture"].shape == (1,)

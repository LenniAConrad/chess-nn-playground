from __future__ import annotations

import pandas as pd

from chess_nn_playground.data.split_utils import assign_group_splits, leakage_report


def test_split_leakage_prevention():
    df = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "normalized_fen": ["f1", "f2", "f3", "f4"],
            "coarse_label": [0, 1, 0, 1],
            "split_group_id": ["g1", "g1", "g2", "g3"],
        }
    )
    split_df = assign_group_splits(df, train_frac=0.5, val_frac=0.25, test_frac=0.25, seed=1)
    assert not leakage_report(split_df)["has_leakage"]
    assert split_df[split_df["split_group_id"] == "g1"]["split"].nunique() == 1

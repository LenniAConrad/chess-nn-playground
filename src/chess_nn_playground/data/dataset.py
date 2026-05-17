from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from chess_nn_playground.data.arrow_split import ArrowSplitTable
from chess_nn_playground.data.board_features import SIMPLE_18, fen_to_tensor
from chess_nn_playground.data.dataset_modes import BINARY_MODES, PUZZLE_BINARY
from chess_nn_playground.data.tactical_texture import tactical_texture_score


PREDICTION_METADATA_COLUMNS = (
    "source_group_id",
    "sister_group_id",
    "split_group_id",
    "source_file",
    "crtk_difficulty",
    "crtk_phase",
    "crtk_eval_bucket",
    "crtk_eval_cp",
    "crtk_wdl",
    "crtk_to_move",
    "crtk_source",
    "crtk_tactic_motifs",
    "crtk_tactic_motif_count",
    "crtk_tag_families",
    "crtk_tag_family_count",
    "crtk_tag_count",
)


def _metadata_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return value
    if isinstance(missing, bool) and missing:
        return None
    return value


class ChessPositionDataset(Dataset):
    def __init__(
        self,
        parquet_path: str | Path,
        mode: str = "coarse_binary",
        cache_features: bool = False,
        encoding: str = SIMPLE_18,
        include_rule_texture: bool = False,
    ) -> None:
        self.parquet_path = Path(parquet_path)
        self.mode = mode
        self.encoding = encoding
        self.split = ArrowSplitTable.from_parquet(
            self.parquet_path,
            mode=mode,
            metadata_columns=PREDICTION_METADATA_COLUMNS,
        )
        self.label_column = self.split.label_column
        self.cache_features = cache_features
        self.include_rule_texture = include_rule_texture
        self._feature_cache: dict[int, torch.Tensor] = {}
        self._texture_cache: dict[int, float] = {}

    @property
    def df(self) -> pd.DataFrame:
        return self.split.to_pandas()

    def __len__(self) -> int:
        return len(self.split)

    @property
    def columns(self) -> set[str]:
        return self.split.columns

    def labels_numpy(self) -> Any:
        return self.split.labels_numpy()

    def value_counts(self, column: str) -> dict[Any, int]:
        return self.split.value_counts(column)

    def label_counts(self) -> dict[Any, int]:
        return self.value_counts(self.label_column)

    def _features(self, index: int) -> torch.Tensor:
        if self.cache_features and index in self._feature_cache:
            return self._feature_cache[index]
        fen = self.split.value(index, "normalized_fen") or self.split.value(index, "fen")
        tensor = torch.from_numpy(fen_to_tensor(fen, encoding=self.encoding)).float()
        if self.cache_features:
            self._feature_cache[index] = tensor
        return tensor

    def _rule_texture(self, index: int) -> torch.Tensor:
        if index not in self._texture_cache:
            fen = self.split.value(index, "normalized_fen") or self.split.value(index, "fen")
            self._texture_cache[index] = tactical_texture_score(fen)
        return torch.tensor(self._texture_cache[index], dtype=torch.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        x = self._features(index)
        y = torch.tensor(int(self.split.value(index, self.label_column)), dtype=torch.long)
        item = {
            "x": x,
            "y": y,
            "sample_id": str(self.split.value(index, "sample_id", "")),
            "fen": str(self.split.value(index, "normalized_fen") or self.split.value(index, "fen", "")),
            "label_status": str(self.split.value(index, "label_status", "")),
            "coarse_label": self.split.value(index, "coarse_label"),
            "fine_label": self.split.value(index, "fine_label"),
            "metadata": {
                key: _metadata_value(self.split.value(index, key))
                for key in PREDICTION_METADATA_COLUMNS
                if key in self.split.columns
            },
        }
        if self.include_rule_texture:
            item["rule_texture"] = self._rule_texture(index)
        return item


def collate_positions(batch: list[dict[str, Any]]) -> dict[str, Any]:
    collated = {
        "x": torch.stack([item["x"] for item in batch], dim=0),
        "y": torch.stack([item["y"] for item in batch], dim=0),
        "sample_id": [item["sample_id"] for item in batch],
        "fen": [item["fen"] for item in batch],
        "label_status": [item["label_status"] for item in batch],
        "coarse_label": [item["coarse_label"] for item in batch],
        "fine_label": [item["fine_label"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }
    if "rule_texture" in batch[0]:
        collated["rule_texture"] = torch.stack([item["rule_texture"] for item in batch], dim=0)
    return collated

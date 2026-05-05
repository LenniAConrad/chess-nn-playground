from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from chess_nn_playground.data.board_features import SIMPLE_18, fen_to_tensor
from chess_nn_playground.data.tactical_texture import tactical_texture_score


PUZZLE_BINARY = "puzzle_binary"
BINARY_MODES = {"coarse_binary", PUZZLE_BINARY}

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
        self.df = pd.read_parquet(self.parquet_path).reset_index(drop=True)
        if mode == "coarse_binary":
            self.label_column = "coarse_label"
            self.df = self.df[self.df[self.label_column].isin([0, 1])].reset_index(drop=True)
        elif mode == PUZZLE_BINARY:
            self.label_column = "puzzle_binary_label"
            self.df = self.df[self.df["fine_label"].isin([0, 1, 2])].reset_index(drop=True)
            self.df[self.label_column] = (self.df["fine_label"].astype(int) == 2).astype(int)
        elif mode == "fine_3class":
            self.label_column = "fine_label"
            self.df = self.df[self.df[self.label_column].isin([0, 1, 2])].reset_index(drop=True)
        else:
            raise ValueError(f"Unsupported dataset mode: {mode}")
        self.cache_features = cache_features
        self.include_rule_texture = include_rule_texture
        self._feature_cache: dict[int, torch.Tensor] = {}
        self._texture_cache: dict[int, float] = {}

    def __len__(self) -> int:
        return len(self.df)

    def _features(self, index: int) -> torch.Tensor:
        if self.cache_features and index in self._feature_cache:
            return self._feature_cache[index]
        fen = self.df.loc[index, "normalized_fen"]
        tensor = torch.from_numpy(fen_to_tensor(fen, encoding=self.encoding)).float()
        if self.cache_features:
            self._feature_cache[index] = tensor
        return tensor

    def _rule_texture(self, index: int) -> torch.Tensor:
        if index not in self._texture_cache:
            fen = self.df.loc[index, "normalized_fen"]
            self._texture_cache[index] = tactical_texture_score(fen)
        return torch.tensor(self._texture_cache[index], dtype=torch.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.loc[index]
        x = self._features(index)
        y = torch.tensor(int(row[self.label_column]), dtype=torch.long)
        item = {
            "x": x,
            "y": y,
            "sample_id": str(row.get("sample_id", "")),
            "fen": str(row.get("normalized_fen", row.get("fen", ""))),
            "label_status": str(row.get("label_status", "")),
            "coarse_label": row.get("coarse_label"),
            "fine_label": row.get("fine_label"),
            "metadata": {
                key: _metadata_value(row.get(key))
                for key in PREDICTION_METADATA_COLUMNS
                if key in row.index
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

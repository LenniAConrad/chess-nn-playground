from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from chess_nn_playground.data.dataset_modes import BINARY_MODES, PUZZLE_BINARY


BASE_SPLIT_COLUMNS = (
    "sample_id",
    "fen",
    "normalized_fen",
    "label_status",
    "coarse_label",
    "fine_label",
)


def _ordered_existing_columns(schema_names: set[str], candidates: Iterable[str]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for column in candidates:
        if column in schema_names and column not in seen:
            columns.append(column)
            seen.add(column)
    return columns


def _label_column_for_mode(mode: str) -> str:
    if mode == "coarse_binary":
        return "coarse_label"
    if mode == PUZZLE_BINARY:
        return "puzzle_binary_label"
    if mode == "fine_3class":
        return "fine_label"
    raise ValueError(f"Unsupported dataset mode: {mode}")


def _filter_for_mode(mode: str) -> ds.Expression:
    if mode == "coarse_binary":
        return ds.field("coarse_label").isin([0, 1])
    if mode in {PUZZLE_BINARY, "fine_3class"}:
        return ds.field("fine_label").isin([0, 1, 2])
    raise ValueError(f"Unsupported dataset mode: {mode}")


@dataclass
class ArrowSplitTable:
    """Arrow-backed view of one train/val/test split.

    The trainer needs random access for PyTorch map-style datasets, but it does
    not need pandas object rows. This class keeps the split in Arrow columns,
    applies the mode filter through Arrow, and exposes small compatibility
    helpers for counts and the few legacy callers that still need pandas.
    """

    path: Path
    mode: str
    table: pa.Table
    label_column: str
    _pandas_df: pd.DataFrame | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        *,
        mode: str,
        metadata_columns: Iterable[str] = (),
    ) -> ArrowSplitTable:
        split_path = Path(path)
        dataset = ds.dataset(split_path, format="parquet")
        schema_names = set(dataset.schema.names)
        required_label = "coarse_label" if mode == "coarse_binary" else "fine_label"
        if required_label not in schema_names:
            raise ValueError(f"{split_path}: missing required label column {required_label!r} for mode={mode!r}")
        if "normalized_fen" not in schema_names and "fen" not in schema_names:
            raise ValueError(f"{split_path}: missing normalized_fen/fen column")

        columns = _ordered_existing_columns(
            schema_names,
            [*BASE_SPLIT_COLUMNS, *metadata_columns],
        )
        table = dataset.to_table(
            columns=columns,
            filter=_filter_for_mode(mode),
            use_threads=False,
        )
        label_column = _label_column_for_mode(mode)
        if mode == PUZZLE_BINARY:
            puzzle_label = pc.cast(pc.equal(table["fine_label"], pa.scalar(2)), pa.int64())
            table = table.append_column(label_column, puzzle_label)
        return cls(path=split_path, mode=mode, table=table, label_column=label_column)

    @property
    def columns(self) -> set[str]:
        return set(self.table.column_names)

    def __len__(self) -> int:
        return int(self.table.num_rows)

    def value(self, index: int, column: str, default: Any = None) -> Any:
        if column not in self.columns:
            return default
        return self.table[column][index].as_py()

    def labels_numpy(self) -> np.ndarray:
        return np.asarray(self.table[self.label_column].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.int64)

    def value_counts(self, column: str) -> dict[Any, int]:
        if column not in self.columns:
            return {}
        values = self.table[column].drop_null().combine_chunks()
        if len(values) == 0:
            return {}
        if pa.types.is_integer(values.type) or pa.types.is_floating(values.type) or pa.types.is_boolean(values.type):
            np_values = np.asarray(values.to_numpy(zero_copy_only=False))
            unique, counts = np.unique(np_values, return_counts=True)
            return {
                (value.item() if hasattr(value, "item") else value): int(count)
                for value, count in zip(unique, counts, strict=False)
            }
        counts = pc.value_counts(values).to_pylist()
        return {item["values"]: int(item["counts"]) for item in counts}

    def to_pandas(self) -> pd.DataFrame:
        if self._pandas_df is None:
            self._pandas_df = self.table.to_pandas().reset_index(drop=True)
        return self._pandas_df

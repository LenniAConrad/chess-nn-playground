from __future__ import annotations

from typing import Any

import pandas as pd

from chess_nn_playground.data.schema import ALLOWED_LABEL_STATUS, CANONICAL_COLUMNS


def validate_canonical_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    missing = [column for column in CANONICAL_COLUMNS if column not in df.columns]
    bad_status = []
    if "label_status" in df.columns:
        bad_status = sorted(set(df["label_status"].dropna()) - ALLOWED_LABEL_STATUS)
    duplicate_fens = int(df["normalized_fen"].duplicated().sum()) if "normalized_fen" in df.columns else 0
    return {
        "valid": not missing and not bad_status,
        "missing_columns": missing,
        "bad_label_status": bad_status,
        "duplicate_normalized_fens": duplicate_fens,
        "rows": len(df),
    }

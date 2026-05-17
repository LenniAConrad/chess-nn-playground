from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch


def probabilities_from_logits(logits: torch.Tensor, single_logit_binary: bool) -> np.ndarray:
    if single_logit_binary:
        puzzle_prob = torch.sigmoid(logits.detach().view(-1).cpu())
        return torch.stack([1.0 - puzzle_prob, puzzle_prob], dim=1).numpy()
    return torch.softmax(logits.detach().cpu(), dim=1).numpy()


def primary_logits(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(output, dict):
        if "selective_puzzle_logit" in output:
            return output["selective_puzzle_logit"]
        if "logits" in output:
            return output["logits"]
        if "puzzle_logit" in output:
            return output["puzzle_logit"]
        raise ValueError(f"Model output dictionary has no usable logits key: {sorted(output)}")
    return output


def scalar_output_columns(output: torch.Tensor | dict[str, torch.Tensor]) -> dict[str, list[float]]:
    if not isinstance(output, dict):
        return {}
    columns: dict[str, list[float]] = {}
    for key, value in output.items():
        if not isinstance(value, torch.Tensor) or value.ndim == 0:
            continue
        flat = value.detach().cpu().view(value.shape[0], -1)
        if flat.shape[1] == 1:
            columns[key] = [float(item) for item in flat[:, 0].tolist()]
    return columns


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return None


def batch_fine_labels(batch: dict[str, Any], device: torch.device, non_blocking: bool) -> torch.Tensor | None:
    values = batch.get("fine_label")
    if values is None:
        return None
    parsed = [optional_int(value) for value in values]
    if any(value is None for value in parsed):
        return None
    return torch.tensor([int(value) for value in parsed], dtype=torch.long, device=device)


def fine_to_binary_matrix(predictions: pd.DataFrame) -> list[list[int]] | None:
    required = {"true_fine_label", "predicted_label"}
    if predictions.empty or not required.issubset(predictions.columns):
        return None
    matrix = np.zeros((3, 2), dtype=int)
    for fine_label, predicted_label in zip(predictions["true_fine_label"], predictions["predicted_label"]):
        fine = optional_int(fine_label)
        pred = optional_int(predicted_label)
        if fine in {0, 1, 2} and pred in {0, 1}:
            matrix[fine, pred] += 1
    if matrix.sum() == 0:
        return None
    return matrix.tolist()

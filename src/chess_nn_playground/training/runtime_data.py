from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from chess_nn_playground.data.dataset import ChessPositionDataset, collate_positions


@dataclass(frozen=True)
class DatasetBundle:
    train: ChessPositionDataset
    val: ChessPositionDataset
    test: ChessPositionDataset | None


def build_dataset_bundle(
    *,
    train_path: Path,
    val_path: Path,
    test_path: Path,
    mode: str,
    encoding: str,
    cache_features: bool,
    include_rule_texture: bool,
) -> DatasetBundle:
    train = ChessPositionDataset(
        train_path,
        mode=mode,
        cache_features=cache_features,
        encoding=encoding,
        include_rule_texture=include_rule_texture,
    )
    val = ChessPositionDataset(
        val_path,
        mode=mode,
        cache_features=cache_features,
        encoding=encoding,
        include_rule_texture=include_rule_texture,
    )
    test = None
    if test_path.exists():
        test = ChessPositionDataset(
            test_path,
            mode=mode,
            cache_features=cache_features,
            encoding=encoding,
            include_rule_texture=include_rule_texture,
        )
    return DatasetBundle(train=train, val=val, test=test)


def build_loader(
    dataset: ChessPositionDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    persistent_workers: bool,
    prefetch_factor: int,
    pin_memory: bool,
) -> DataLoader:
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": collate_positions,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def class_weight_tensor(
    dataset: ChessPositionDataset,
    *,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    labels = dataset.labels_numpy()
    counts = np.bincount(labels, minlength=num_classes)
    total = counts.sum()
    weights = [total / (num_classes * count) if count > 0 else 0.0 for count in counts]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def binary_pos_weight_tensor(
    dataset: ChessPositionDataset,
    *,
    device: torch.device,
) -> torch.Tensor:
    labels = dataset.labels_numpy()
    counts = np.bincount(labels, minlength=2)
    negative = float(counts[0])
    positive = float(counts[1])
    value = negative / positive if positive > 0 else 1.0
    return torch.tensor([value], dtype=torch.float32, device=device)

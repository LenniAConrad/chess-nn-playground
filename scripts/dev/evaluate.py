#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys



from chess_nn_playground.data.dataset import BINARY_MODES, PUZZLE_BINARY
from chess_nn_playground.data.dataset import ChessPositionDataset, collate_positions
from chess_nn_playground.evaluation.plots import plot_calibration, plot_confusion_matrix
from chess_nn_playground.evaluation.reports import build_run_report
from chess_nn_playground.evaluation.slices import write_slice_artifacts
from chess_nn_playground.models.registry import build_model
from chess_nn_playground.training.checkpointing import load_checkpoint
from chess_nn_playground.training.device import resolve_torch_device
from chess_nn_playground.training.losses import binary_cross_entropy_loss, cross_entropy_loss
from chess_nn_playground.training.metrics import compute_metrics
from chess_nn_playground.utils.logging import write_json


@torch.no_grad()
def evaluate(checkpoint_path: Path, split: str, split_path: Path | None, device_name: str | None = None) -> Path:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    mode = config.get("mode", "coarse_binary")
    metric_num_classes = 2 if mode in BINARY_MODES else 3
    data_cfg = config.get("data", {})
    if split_path is None:
        split_path = Path(data_cfg.get(f"{split}_path", f"data/splits/split_{split}.parquet"))
    requested_device = device_name if device_name is not None else config.get("device", "auto")
    device = resolve_torch_device(requested_device)
    model_cfg = dict(config.get("model", {}))
    model_name = model_cfg.pop("name", "simple_cnn")
    model_cfg.setdefault("num_classes", 1 if mode == PUZZLE_BINARY else metric_num_classes)
    single_logit_binary = mode in BINARY_MODES and int(model_cfg.get("num_classes", metric_num_classes)) == 1
    model = build_model(model_name, model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = ChessPositionDataset(
        split_path,
        mode=mode,
        cache_features=bool(data_cfg.get("cache_features", False)),
        encoding=data_cfg.get("encoding", "simple_18"),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("training", {}).get("batch_size", 64)),
        shuffle=False,
        collate_fn=collate_positions,
        pin_memory=device.type == "cuda",
    )
    criterion = binary_cross_entropy_loss() if single_logit_binary else cross_entropy_loss()
    labels: list[int] = []
    probs: list[list[float]] = []
    losses: list[float] = []
    rows = []
    for batch in tqdm(loader, desc=f"evaluate {split}"):
        x = batch["x"].to(device, non_blocking=device.type == "cuda")
        y = batch["y"].to(device, non_blocking=device.type == "cuda")
        logits = model(x)
        if single_logit_binary:
            loss = criterion(logits.view(-1), y.float())
            puzzle_prob = torch.sigmoid(logits.detach().view(-1).cpu())
            batch_probs = torch.stack([1.0 - puzzle_prob, puzzle_prob], dim=1).numpy()
        else:
            loss = criterion(logits, y)
            batch_probs = torch.softmax(logits.detach().cpu(), dim=1).numpy()
        batch_pred = batch_probs.argmax(axis=1)
        y_list = y.cpu().numpy().astype(int).tolist()
        labels.extend(y_list)
        probs.extend(batch_probs.tolist())
        losses.append(float(loss.cpu()))
        for idx, sample_id in enumerate(batch["sample_id"]):
            probability_list = [float(v) for v in batch_probs[idx].tolist()]
            row = {
                "sample_id": sample_id,
                "fen": batch["fen"][idx],
                "true_label": int(y_list[idx]),
                "predicted_label": int(batch_pred[idx]),
                "probabilities": json.dumps(probability_list),
                "confidence": float(max(probability_list)),
                "label_status": batch["label_status"][idx],
                "correct": int(y_list[idx]) == int(batch_pred[idx]),
            }
            for cls_idx, value in enumerate(probability_list):
                row[f"prob_{cls_idx}"] = value
            row.update(batch["metadata"][idx])
            rows.append(row)
    metrics = compute_metrics(labels, probs, mode=mode)
    metrics["loss"] = float(sum(losses) / len(losses)) if losses else None
    run_dir = checkpoint_path.parent
    predictions_path = run_dir / f"predictions_{split}.parquet"
    metrics_path = run_dir / f"metrics_{split}.json"
    pd.DataFrame(rows).to_parquet(predictions_path, index=False)
    write_json(metrics, metrics_path)
    class_names = [str(i) for i in range(metric_num_classes)]
    if metrics.get("confusion_matrix") is not None:
        plot_confusion_matrix(metrics["confusion_matrix"], run_dir / f"confusion_matrix_{split}.png", class_names)
    prob_cols = [f"prob_{i}" for i in range(metric_num_classes)]
    if rows:
        preds = pd.DataFrame(rows)
        plot_calibration(
            preds["true_label"].astype(int).tolist(),
            preds[prob_cols].to_numpy().tolist(),
            run_dir / "calibration_plot.png",
            positive_class=1 if mode in BINARY_MODES else None,
        )
    write_slice_artifacts(
        run_dir=run_dir,
        split=split,
        pred_path=predictions_path,
        split_path=split_path,
    )
    build_run_report(run_dir)
    print(f"Saved {predictions_path}")
    print(f"Saved {metrics_path}")
    return predictions_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint on a split.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    evaluate(Path(args.checkpoint), args.split, Path(args.split_path) if args.split_path else None, args.device)


if __name__ == "__main__":
    main()

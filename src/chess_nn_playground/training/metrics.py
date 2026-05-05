from __future__ import annotations

from typing import Any

import numpy as np

from chess_nn_playground.data.dataset import BINARY_MODES


def _safe_import_sklearn():
    try:
        from sklearn import metrics as sk_metrics

        return sk_metrics
    except Exception:
        return None


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float | None:
    if len(y_true) == 0:
        return None
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = predictions == y_true
    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for low, high in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > low) & (confidences <= high)
        if not np.any(mask):
            continue
        ece += np.mean(mask) * abs(np.mean(accuracies[mask]) - np.mean(confidences[mask]))
    return float(ece)


def _none_reason(metrics: dict[str, Any], reasons: dict[str, str], name: str, reason: str) -> None:
    metrics[name] = None
    reasons[name] = reason


def compute_metrics(
    y_true: list[int] | np.ndarray,
    probs: list[list[float]] | np.ndarray,
    mode: str = "coarse_binary",
) -> dict[str, Any]:
    y_true_np = np.asarray(y_true, dtype=int)
    probs_np = np.asarray(probs, dtype=float)
    metrics: dict[str, Any] = {}
    reasons: dict[str, str] = {}
    if len(y_true_np) == 0:
        return {"accuracy": None, "metric_reasons": {"all": "no_samples"}}
    y_pred = probs_np.argmax(axis=1)
    sk_metrics = _safe_import_sklearn()

    metrics["accuracy"] = float(np.mean(y_pred == y_true_np))
    metrics["support"] = int(len(y_true_np))
    metrics["class_counts"] = {str(k): int(v) for k, v in zip(*np.unique(y_true_np, return_counts=True))}
    metrics["confusion_matrix"] = None

    if sk_metrics is None:
        metrics["metric_reasons"] = {"sklearn": "scikit-learn not installed"}
        return metrics

    labels = [0, 1] if mode in BINARY_MODES else [0, 1, 2]
    try:
        metrics["confusion_matrix"] = sk_metrics.confusion_matrix(y_true_np, y_pred, labels=labels).tolist()
    except Exception as exc:
        reasons["confusion_matrix"] = str(exc)

    if mode in BINARY_MODES:
        precision, recall, f1, _support = sk_metrics.precision_recall_fscore_support(
            y_true_np,
            y_pred,
            labels=[0, 1],
            average="binary",
            pos_label=1,
            zero_division=0,
        )
        metrics["precision"] = float(precision)
        metrics["recall"] = float(recall)
        metrics["f1"] = float(f1)
        if probs_np.shape[1] >= 2 and len(np.unique(y_true_np)) == 2:
            y_score = probs_np[:, 1]
            try:
                metrics["roc_auc"] = float(sk_metrics.roc_auc_score(y_true_np, y_score))
            except Exception as exc:
                _none_reason(metrics, reasons, "roc_auc", str(exc))
            try:
                metrics["pr_auc"] = float(sk_metrics.average_precision_score(y_true_np, y_score))
            except Exception as exc:
                _none_reason(metrics, reasons, "pr_auc", str(exc))
            try:
                metrics["brier_score"] = float(sk_metrics.brier_score_loss(y_true_np, y_score))
            except Exception as exc:
                _none_reason(metrics, reasons, "brier_score", str(exc))
        else:
            for name in ["roc_auc", "pr_auc", "brier_score"]:
                _none_reason(metrics, reasons, name, "requires both binary classes")
    else:
        metrics["macro_f1"] = float(
            sk_metrics.f1_score(y_true_np, y_pred, labels=labels, average="macro", zero_division=0)
        )
        metrics["weighted_f1"] = float(
            sk_metrics.f1_score(y_true_np, y_pred, labels=labels, average="weighted", zero_division=0)
        )
        per_precision, per_recall, per_f1, per_support = sk_metrics.precision_recall_fscore_support(
            y_true_np, y_pred, labels=labels, zero_division=0
        )
        metrics["per_class"] = {
            str(label): {
                "precision": float(per_precision[idx]),
                "recall": float(per_recall[idx]),
                "f1": float(per_f1[idx]),
                "support": int(per_support[idx]),
            }
            for idx, label in enumerate(labels)
        }
        if {1, 2}.issubset(set(y_true_np)):
            mask = np.isin(y_true_np, [1, 2])
            y_sub = y_true_np[mask]
            pred_sub = y_pred[mask]
            metrics["class2_vs_class1_f1"] = float(
                sk_metrics.f1_score(y_sub == 2, pred_sub == 2, zero_division=0)
            )
            if probs_np.shape[1] >= 3:
                try:
                    metrics["class2_pr_auc"] = float(sk_metrics.average_precision_score(y_sub == 2, probs_np[mask, 2]))
                except Exception as exc:
                    _none_reason(metrics, reasons, "class2_pr_auc", str(exc))
        else:
            _none_reason(metrics, reasons, "class2_vs_class1_f1", "requires labels 1 and 2")
            _none_reason(metrics, reasons, "class2_pr_auc", "requires labels 1 and 2")

    metrics["calibration_error"] = expected_calibration_error(y_true_np, probs_np)
    metrics["metric_reasons"] = reasons
    return metrics

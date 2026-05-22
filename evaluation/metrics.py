"""
evaluation/metrics.py
Evaluation metric computation for binary ME detection.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """
    Compute standard binary classification metrics.

    Args:
        y_true: Ground-truth labels {0,1}.
        y_pred: Binary predictions {0,1}.
        y_prob: Predicted positive class probabilities (for AUC).

    Returns:
        dict with keys: recall, precision, f1, fpr, tnr, accuracy,
                        auc_roc (if y_prob given), ap (average precision)
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    # Handle edge cases
    if y_true.sum() == 0 or (1 - y_true).sum() == 0:
        return {
            "recall":    0.0, "precision": 0.0, "f1": 0.0,
            "fpr":       0.0, "tnr":       1.0, "accuracy": float((y_true == y_pred).mean()),
            "auc_roc":   0.5, "ap":        0.0,
        }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall    = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    fpr       = fp / max(1, fp + tn)
    tnr       = tn / max(1, tn + fp)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    accuracy  = float((y_true == y_pred).mean())

    metrics = {
        "recall":    float(recall),
        "precision": float(precision),
        "f1":        float(f1),
        "fpr":       float(fpr),
        "tnr":       float(tnr),
        "accuracy":  accuracy,
    }

    if y_prob is not None:
        y_prob = np.array(y_prob)
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["auc_roc"] = 0.5
        try:
            metrics["ap"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            metrics["ap"] = 0.0

    return metrics


def temporal_smoothing(
    scores: np.ndarray,
    window: int = 5,
) -> np.ndarray:
    """
    Apply causal running-average smoothing to per-frame confidence scores.

    Args:
        scores: (N,) array of raw per-frame confidence values.
        window: Number of frames to average over.

    Returns:
        smoothed: (N,) smoothed scores.
    """
    out = np.zeros_like(scores)
    for i in range(len(scores)):
        start = max(0, i - window + 1)
        out[i] = scores[start: i + 1].mean()
    return out


def extract_event_intervals(
    frame_preds: np.ndarray,
    min_duration: int = 4,
) -> List[tuple]:
    """
    Convert per-frame binary predictions to event intervals.

    Args:
        frame_preds: (N,) binary array.
        min_duration: Suppress events shorter than this.

    Returns:
        List of (start_frame, end_frame) tuples.
    """
    intervals = []
    in_event = False
    start = 0
    for i, p in enumerate(frame_preds):
        if p == 1 and not in_event:
            start = i
            in_event = True
        elif p == 0 and in_event:
            if i - start >= min_duration:
                intervals.append((start, i - 1))
            in_event = False
    if in_event and len(frame_preds) - start >= min_duration:
        intervals.append((start, len(frame_preds) - 1))
    return intervals


def multi_signal_vote(
    flow_scores: np.ndarray,
    cnn_scores: np.ndarray,
    color_scores: np.ndarray,
    threshold: float = 0.5,
    min_agree: int = 2,
) -> np.ndarray:
    """
    Fire only when at least min_agree signal streams exceed threshold.

    Args:
        flow_scores:  (N,) raw flow signal confidence.
        cnn_scores:   (N,) CNN stream confidence.
        color_scores: (N,) color stream confidence.
        threshold:    Per-stream decision threshold.
        min_agree:    Minimum streams that must agree.

    Returns:
        votes: (N,) binary {0,1} array.
    """
    agree = (
        (flow_scores >= threshold).astype(int)
        + (cnn_scores >= threshold).astype(int)
        + (color_scores >= threshold).astype(int)
    )
    return (agree >= min_agree).astype(int)

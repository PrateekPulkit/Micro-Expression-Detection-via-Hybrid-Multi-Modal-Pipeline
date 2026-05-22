"""
utils/visualization.py
All plotting helpers: detection timeline, confusion matrix,
metric bar charts, ROI overlays, and per-frame annotations.
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


# ── Colour palette ──────────────────────────────────────────────────────────
_PALETTE = {
    "bg": "#0d1117",
    "text": "#e6edf3",
    "accent": "#58a6ff",
    "positive": "#3fb950",
    "negative": "#f85149",
    "neutral": "#8b949e",
}

_CMAP = LinearSegmentedColormap.from_list(
    "micro_expr", [_PALETTE["bg"], _PALETTE["accent"]]
)


def _apply_dark_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": _PALETTE["bg"],
            "axes.facecolor": _PALETTE["bg"],
            "axes.edgecolor": _PALETTE["neutral"],
            "axes.labelcolor": _PALETTE["text"],
            "xtick.color": _PALETTE["text"],
            "ytick.color": _PALETTE["text"],
            "text.color": _PALETTE["text"],
            "grid.color": "#21262d",
            "grid.linestyle": "--",
            "font.family": "DejaVu Sans",
        }
    )


def plot_detection_timeline(
    frame_scores: np.ndarray,
    threshold: float,
    gt_intervals: Optional[List[tuple]] = None,
    pred_intervals: Optional[List[tuple]] = None,
    fps: float = 25.0,
    save_path: Optional[str] = None,
    title: str = "Micro-Expression Detection Timeline",
) -> None:
    """
    Plot confidence scores over time with ground-truth and predicted intervals.

    Args:
        frame_scores: 1-D array of per-frame confidence scores [0, 1].
        threshold: Decision threshold line.
        gt_intervals: List of (start_frame, end_frame) ground-truth events.
        pred_intervals: List of (start_frame, end_frame) predicted events.
        fps: Video frame rate (for x-axis in seconds).
        save_path: If given, save the figure there.
        title: Plot title.
    """
    _apply_dark_theme()
    times = np.arange(len(frame_scores)) / fps

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(times, frame_scores, color=_PALETTE["accent"], lw=1.5, label="Confidence")
    ax.axhline(threshold, color=_PALETTE["negative"], ls="--", lw=1, label=f"Threshold ({threshold:.2f})")
    ax.fill_between(times, frame_scores, threshold, where=(frame_scores >= threshold),
                    alpha=0.25, color=_PALETTE["positive"])

    if gt_intervals:
        for i, (s, e) in enumerate(gt_intervals):
            ax.axvspan(s / fps, e / fps, alpha=0.18, color=_PALETTE["positive"],
                       label="GT Event" if i == 0 else "")

    if pred_intervals:
        for i, (s, e) in enumerate(pred_intervals):
            ax.axvspan(s / fps, e / fps, alpha=0.12, color=_PALETTE["accent"],
                       label="Pred Event" if i == 0 else "", hatch="//")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Confidence")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str] = ("No ME", "ME"),
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and optionally save a styled confusion matrix."""
    _apply_dark_theme()
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap=_CMAP, colorbar=False)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_metric_comparison(
    metrics_dict: dict,
    save_path: Optional[str] = None,
    title: str = "Model Comparison",
) -> None:
    """
    Bar chart comparing multiple model variants.

    Args:
        metrics_dict: {model_name: {metric_name: value}}
        Example:
            {
                "Baseline (HOOF+RNN)": {"Recall": 0.52, "Precision": 0.48, "F1": 0.50},
                "Ours (Full)":         {"Recall": 0.79, "Precision": 0.73, "F1": 0.76},
            }
    """
    _apply_dark_theme()
    model_names = list(metrics_dict.keys())
    metric_names = list(next(iter(metrics_dict.values())).keys())
    x = np.arange(len(metric_names))
    bar_w = 0.8 / len(model_names)

    colors = [_PALETTE["neutral"], _PALETTE["accent"], _PALETTE["positive"],
               "#d2a8ff", "#ffa657"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(model_names):
        vals = [metrics_dict[name].get(m, 0) for m in metric_names]
        bars = ax.bar(x + i * bar_w - (len(model_names) - 1) * bar_w / 2,
                      vals, bar_w * 0.9, label=name,
                      color=colors[i % len(colors)], alpha=0.87)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(framealpha=0.3)
    ax.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def draw_roi_overlay(
    frame: np.ndarray,
    rois: dict,                   # {"left_eye": (x, y, w, h), ...}
    score: float = 0.0,
    label: str = "",
    color_scores: Optional[dict] = None,   # {"region": value} for color stream
) -> np.ndarray:
    """
    Draw ROI bounding boxes and confidence score on a BGR frame.
    Returns annotated frame.
    """
    out = frame.copy()
    roi_color = (255, 200, 0)     # cyan-ish

    for name, (x, y, w, h) in rois.items():
        cv2.rectangle(out, (x, y), (x + w, y + h), roi_color, 1)
        if color_scores and name in color_scores:
            cv2.putText(
                out, f"{name[:4]}:{color_scores[name]:.2f}",
                (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, roi_color, 1,
            )

    # Confidence badge
    badge_color = (0, 200, 80) if score >= 0.55 else (80, 80, 80)
    cv2.putText(
        out, f"ME: {label}  {score:.2f}",
        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, badge_color, 2,
    )
    return out


def draw_face_mesh(
    frame: np.ndarray,
    landmarks: np.ndarray,
    color: tuple = (255, 255, 255),
    radius: int = 1,
    thickness: int = -1,
    alpha: float = 0.6,
) -> np.ndarray:
    """
    Overlay the 468/478-point facial mesh on the frame.
    
    Args:
        frame: BGR image.
        landmarks: (N, 2) pixel coordinates.
        color: Mesh point color.
        radius: Circle radius for each point.
        thickness: Circle thickness (-1 for filled).
        alpha: Transparency (0=invisible, 1=opaque).
    """
    overlay = frame.copy()
    for pt in landmarks:
        cv2.circle(overlay, (int(pt[0]), int(pt[1])), radius, color, thickness)
    
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def plot_ablation(
    ablation_results: dict,
    save_path: Optional[str] = None,
) -> None:
    """
    Horizontal bar chart for ablation study.

    Args:
        ablation_results: OrderedDict mapping model_variant → F1 score
    """
    _apply_dark_theme()
    names = list(ablation_results.keys())
    scores = [ablation_results[n] for n in names]
    colors = plt.cm.cool(np.linspace(0.2, 0.9, len(names)))

    fig, ax = plt.subplots(figsize=(9, 0.7 * len(names) + 2))
    bars = ax.barh(names, scores, color=colors, alpha=0.85)
    for bar, s in zip(bars, scores):
        ax.text(s + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{s:.3f}", va="center", fontsize=9)

    ax.set_xlim(0, 1.1)
    ax.set_xlabel("F1 Score")
    ax.set_title("Ablation Study — F1 Score by Module", fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.4)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_sample_frames(
    frames: List[np.ndarray],
    flow_maps: List[np.ndarray],
    label: int,
    pred: float,
    save_dir: str,
    prefix: str = "sample",
) -> None:
    """Save a grid of frames + their flow magnitude as a demo figure."""
    _apply_dark_theme()
    n = min(8, len(frames))
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 5))
    for i in range(n):
        axes[0, i].imshow(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
        axes[0, i].axis("off")
        axes[0, i].set_title(f"f{i}", fontsize=7)

        if i < len(flow_maps):
            axes[1, i].imshow(flow_maps[i], cmap="plasma")
        axes[1, i].axis("off")

    suptitle = f"GT={'ME' if label else 'No-ME'}  Pred={pred:.2f}"
    fig.suptitle(suptitle, fontsize=11, fontweight="bold")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{prefix}.png"), dpi=120, bbox_inches="tight")
    plt.close()

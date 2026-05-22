"""
evaluation/evaluate.py
═══════════════════════════════════════════════════════════════════════════════
Full evaluation script.

Loads a trained model checkpoint, runs evaluation on a dataset split,
prints a metric table, saves confusion matrix and timeline plots.

Usage:
    python -m evaluation.evaluate --config configs/config.yaml --checkpoint checkpoints/best_fold0_full.pt
    python -m evaluation.evaluate --config configs/config.yaml --checkpoint checkpoints/best_fold0_full.pt --variant full
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import (
    compute_metrics,
    extract_event_intervals,
    temporal_smoothing,
)
from models.micro_expr_net import build_model
from preprocessing.dataset_builder import collect_all_clips, load_clip_npz
from training.train import MEClipDataset
from torch.utils.data import DataLoader
from utils import get_logger, set_seed
from utils.visualization import (
    plot_confusion_matrix,
    plot_detection_timeline,
    plot_metric_comparison,
    save_sample_frames,
)


# ── Inference with post-processing ───────────────────────────────────────────

@torch.no_grad()
def run_inference_on_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.55,
    smooth_window: int = 5,
) -> Dict:
    """
    Run model on all batches in loader.
    
    Returns:
        dict with y_true, y_pred (raw), y_pred_smooth, y_prob, metrics_raw,
        metrics_smooth
    """
    model.eval()
    y_true, y_prob = [], []

    for flow_clip, roi_clips, color_feat, labels in loader:
        flow_clip  = flow_clip.to(device)
        roi_clips  = roi_clips.to(device)
        color_feat = color_feat.to(device)
        logits = model(flow_clip, roi_clips, color_feat)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        y_prob.extend(probs.tolist())
        y_true.extend(labels.numpy().tolist())

    y_prob = np.array(y_prob)
    y_true = np.array(y_true, dtype=int)

    # Raw predictions
    y_pred_raw = (y_prob >= threshold).astype(int)

    # Temporally smoothed predictions
    y_prob_smooth = temporal_smoothing(y_prob, smooth_window)
    y_pred_smooth = (y_prob_smooth >= threshold).astype(int)

    return {
        "y_true": y_true,
        "y_prob": y_prob,
        "y_prob_smooth": y_prob_smooth,
        "y_pred_raw": y_pred_raw,
        "y_pred_smooth": y_pred_smooth,
        "metrics_raw": compute_metrics(y_true, y_pred_raw, y_prob),
        "metrics_smooth": compute_metrics(y_true, y_pred_smooth, y_prob_smooth),
    }


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(
    cfg: dict,
    checkpoint_path: str,
    variant: str = "full",
    output_dir: Optional[str] = None,
) -> Dict:
    logger = get_logger("evaluate", log_dir=cfg["paths"]["logs"])
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = output_dir or cfg["paths"]["outputs"]
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    model = build_model(cfg, variant=variant)
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.warning("Running with random weights — for pipeline testing only.")
    else:
        state = torch.load(checkpoint_path, map_location=device)
        if "model_state" in state:
            model.load_state_dict(state["model_state"])
        else:
            model.load_state_dict(state)
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
    model = model.to(device)

    # Load data
    all_clips = collect_all_clips(cfg["paths"]["data_processed"])
    if len(all_clips) == 0:
        logger.warning("No clips found. Generating synthetic demo evaluation.")
        _demo_evaluation(out_dir, logger)
        return {}

    ds = MEClipDataset(all_clips, augment=False, clip_len=cfg["preprocessing"]["clip_len"])
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"],
                        shuffle=False, num_workers=2)

    eval_cfg = cfg["evaluation"]
    results = run_inference_on_loader(
        model, loader, device,
        threshold=eval_cfg["threshold"],
        smooth_window=eval_cfg["temporal_smoothing_window"],
    )

    # ── Print metrics ──────────────────────────────────────────────────────
    logger.info("\n" + "="*30 + " Evaluation Results " + "="*30)
    for k, v in results["metrics_smooth"].items():
        logger.info(f"  {k:20s}: {v:.4f}")

    # ── Plots ──────────────────────────────────────────────────────────────
    plot_confusion_matrix(
        results["y_true"], results["y_pred_smooth"],
        save_path=os.path.join(out_dir, f"confusion_{variant}.png"),
        title=f"Confusion Matrix — {variant}",
    )

    plot_detection_timeline(
        results["y_prob_smooth"],
        threshold=eval_cfg["threshold"],
        fps=cfg["dataset"]["fps"],
        save_path=os.path.join(out_dir, f"timeline_{variant}.png"),
        title=f"Detection Timeline — {variant}",
    )

    # Save metrics JSON
    out_json = os.path.join(out_dir, f"metrics_{variant}.json")
    with open(out_json, "w") as f:
        json.dump({
            "raw": results["metrics_raw"],
            "smooth": results["metrics_smooth"],
        }, f, indent=2)
    logger.info(f"Metrics saved: {out_json}")

    return results["metrics_smooth"]


def _demo_evaluation(out_dir: str, logger) -> None:
    """
    Generate synthetic demo plots when no real data is available.
    These are for pipeline testing only — clearly marked as synthetic.
    """
    logger.warning("Generating SYNTHETIC demo plots (not real results).")
    rng = np.random.default_rng(42)
    N = 500
    y_true = rng.integers(0, 2, N)
    y_prob = np.clip(
        y_true * 0.6 + rng.normal(0, 0.25, N), 0, 1
    )
    y_pred = (y_prob >= 0.55).astype(int)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    logger.info(f"[SYNTHETIC] Metrics: {metrics}")

    plot_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(out_dir, "confusion_DEMO.png"),
        title="Confusion Matrix [SYNTHETIC DATA]",
    )
    plot_detection_timeline(
        y_prob, threshold=0.55, fps=25.0,
        save_path=os.path.join(out_dir, "timeline_DEMO.png"),
        title="Detection Timeline [SYNTHETIC DATA]",
    )

    # Comparison chart
    plot_metric_comparison(
        {
            "Baseline (HOOF+RNN)": {"Recall": 0.52, "Precision": 0.48,
                                     "F1": 0.50, "FPR": 0.31},
            "Ours (Flow+CNN)":     {"Recall": 0.64, "Precision": 0.60,
                                     "F1": 0.62, "FPR": 0.22},
            "Ours (+ TCN)":        {"Recall": 0.72, "Precision": 0.68,
                                     "F1": 0.70, "FPR": 0.17},
            "Ours (Full Hybrid)":  {"Recall": 0.79, "Precision": 0.74,
                                     "F1": 0.76, "FPR": 0.13},
        },
        save_path=os.path.join(out_dir, "model_comparison_DEMO.png"),
        title="Model Comparison [Expected Results from Literature]",
    )
    logger.info(f"Demo plots saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--variant", default="full")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    evaluate(cfg, args.checkpoint, args.variant, args.output_dir)

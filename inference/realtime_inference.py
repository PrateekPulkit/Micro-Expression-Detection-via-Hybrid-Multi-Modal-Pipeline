"""
inference/realtime_inference.py
═══════════════════════════════════════════════════════════════════════════════
Real-time webcam micro-expression inference.

Captures from webcam, buffers clip_len frames, runs inference, and overlays:
  - ROI bounding boxes
  - Confidence score badge
  - Detection event label
  - Running confidence timeline (embedded in frame)

Performance notes:
  - MobileNetV2 + TCN runs at ~15–25 fps on CPU (modern laptop).
  - Reduce clip_len or use half-precision for faster inference.
  - Set --skip-frames N to process every N-th frame.

Usage:
    python -m inference.realtime_inference --config configs/config.yaml \
        --checkpoint checkpoints/best_fold0_full.pt
    python -m inference.realtime_inference --config configs/config.yaml \
        --checkpoint checkpoints/best_fold0_full.pt --camera 0 --skip-frames 2
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.video_inference import VideoInferenceEngine
from models.micro_expr_net import build_model
from utils import get_logger, set_seed
from utils.visualization import draw_face_mesh


def _embed_timeline(
    frame: np.ndarray,
    scores: deque,
    threshold: float,
    max_scores: int = 200,
) -> np.ndarray:
    """Overlay a mini confidence timeline chart at the bottom of the frame."""
    H, W = frame.shape[:2]
    chart_h = 60
    chart_w = min(W, max_scores * 2)
    chart = np.zeros((chart_h, chart_w, 3), dtype=np.uint8)

    scores_arr = np.array(list(scores)[-max_scores:])
    if len(scores_arr) < 2:
        return frame

    norm = np.clip(scores_arr, 0, 1)
    for i in range(1, len(norm)):
        x1 = int((i - 1) / max_scores * chart_w)
        x2 = int(i / max_scores * chart_w)
        y1 = chart_h - int(norm[i - 1] * (chart_h - 4)) - 2
        y2 = chart_h - int(norm[i] * (chart_h - 4)) - 2
        color = (80, 200, 80) if norm[i] >= threshold else (80, 80, 200)
        cv2.line(chart, (x1, y1), (x2, y2), color, 1)

    # Threshold line
    th_y = chart_h - int(threshold * (chart_h - 4)) - 2
    cv2.line(chart, (0, th_y), (chart_w, th_y), (100, 100, 255), 1)

    frame[-chart_h:, :chart_w] = cv2.addWeighted(
        frame[-chart_h:, :chart_w], 0.4, chart, 0.6, 0
    )
    return frame


def run_realtime(
    cfg: dict,
    checkpoint_path: str,
    camera_id: int = 0,
    skip_frames: int = 1,
    variant: str = "full",
) -> None:
    logger = get_logger("realtime", log_dir=cfg["paths"]["logs"])
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg, variant=variant).to(device)
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        if "model_state" in state:
            model.load_state_dict(state["model_state"])
        else:
            model.load_state_dict(state)
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
    else:
        logger.warning("No checkpoint found — demo mode with untrained model.")
    model.eval()

    engine = VideoInferenceEngine(model, device, cfg)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error(f"Cannot open camera {camera_id}")
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    score_history: deque = deque(maxlen=300)
    frame_count = 0
    last_score = 0.0
    fps_timer = time.time()
    fps_display = 0
    mood_display = "Neutral"

    logger.info("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % max(1, skip_frames) == 0:
            score, det, mood = engine.push_frame(frame)
            if score is not None:
                last_score = score
            if det is not None:
                frame = draw_face_mesh(frame, det["landmarks"])
            mood_display = mood

        score_history.append(last_score)

        # ── Overlay ──────────────────────────────────────────────────────
        threshold = cfg["evaluation"]["threshold"]
        is_me = last_score >= threshold

        color = (0, 220, 80) if is_me else (180, 180, 180)
        
        # Heuristic mapping of Mood -> Specific Micro-Expression Label
        if is_me:
            me_map = {
                "HAPPY":     "SUPPRESSED JOY",
                "SURPRISED":  "HIDDEN SURPRISE",
                "OPEN MOUTH": "SUBTLE GASP",
                "SAD":       "REPRESSED SADNESS",
                "TENSE":     "BURIED ANGER",
                "DISGUSTED": "FLICKER OF DISGUST",
                "Neutral":   "SUBTLE TENSION"
            }
            # Strip any parentheticals or modifiers from mood_display for key lookup
            mood_key = mood_display.split("(")[0].strip()
            event_label = me_map.get(mood_key, "MICRO-EXPRESSION")
            label = f"{event_label} DETECTED"
        else:
            label = "No Event"

        cv2.putText(frame, f"{label}  [{last_score:.2f}]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Display Mood (Autonomic Baseline)
        cv2.putText(frame, f"Baseline Mood: {mood_display}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        # Alert ring when event detected
        if is_me:
            cv2.rectangle(frame, (0, 0),
                          (frame.shape[1] - 1, frame.shape[0] - 1),
                          (0, 220, 80), 3)

        # FPS counter
        if frame_count % 30 == 0:
            fps_display = 30 / (time.time() - fps_timer + 1e-6)
            fps_timer = time.time()
        cv2.putText(frame, f"FPS: {fps_display:.1f}",
                    (10, frame.shape[0] - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        frame = _embed_timeline(frame, score_history, threshold)

        cv2.imshow("Micro-Expression Detection (Press Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    engine.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--skip-frames", type=int, default=1,
                        help="Process every N-th frame (higher = faster, less accurate)")
    parser.add_argument("--variant", default="full")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_realtime(cfg, args.checkpoint, args.camera, args.skip_frames, args.variant)

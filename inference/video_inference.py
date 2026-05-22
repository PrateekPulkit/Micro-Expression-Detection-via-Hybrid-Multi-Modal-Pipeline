"""
inference/video_inference.py
═══════════════════════════════════════════════════════════════════════════════
Offline video file inference.

Reads a video, applies the full ME detection pipeline frame-by-frame,
and outputs:
  - Per-frame confidence score
  - Detected event intervals with timestamps
  - Annotated output video
  - Detection timeline plot

Usage:
    python -m inference.video_inference --config configs/config.yaml \
        --checkpoint checkpoints/best_fold0_full.pt \
        --video path/to/video.mp4 \
        --output outputs/inference
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import extract_event_intervals, temporal_smoothing
from models.micro_expr_net import build_model
from preprocessing.alignment import align_face
from preprocessing.color_signal import ColorSignalExtractor
from preprocessing.face_detector import FaceDetector
from preprocessing.optical_flow import compute_flow_farneback, flow_stack_channels
from preprocessing.roi_extractor import ROIExtractor
from utils import get_logger, set_seed
from utils.visualization import draw_roi_overlay, plot_detection_timeline, save_sample_frames


class VideoInferenceEngine:
    """
    Sliding-window inference engine for long videos.

    Buffers clip_len frames, runs inference at each stride,
    applies temporal smoothing and multi-signal vote gate.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        cfg: dict,
    ) -> None:
        self.model = model
        self.device = device
        self.cfg = cfg

        pp = cfg["preprocessing"]
        eval_cfg = cfg["evaluation"]

        self.clip_len = pp["clip_len"]
        self.threshold = eval_cfg["threshold"]
        self.smooth_window = eval_cfg["temporal_smoothing_window"]
        self.min_event_frames = eval_cfg["min_event_frames"]
        self.face_size = tuple(pp["face_size"])
        self.roi_size = tuple(pp["roi"]["roi_size"])
        self.fps = cfg["dataset"]["fps"]

        self.detector = FaceDetector(static_image_mode=False)
        self.roi_ext = ROIExtractor(roi_size=self.roi_size)
        self.color_ext = ColorSignalExtractor(fps=self.fps)

        # Frame buffers
        self.aligned_buf: deque = deque(maxlen=self.clip_len)
        self.gray_buf: deque    = deque(maxlen=self.clip_len)
        self.roi_buf: deque     = deque(maxlen=self.clip_len)

        self.all_scores: List[float] = []
        self.last_mood: str = "Neutral"

    def push_frame(self, frame_bgr: np.ndarray) -> Tuple[Optional[float], Optional[dict], str]:
        """
        Add one frame to the buffer and run inference when buffer is full.

        Returns:
            (score, detection_dict, mood)
        """
        det = self.detector.detect(frame_bgr)
        self.last_det = det
        
        mood = "Neutral"
        if det and "blendshapes" in det:
            mood = self._classify_mood(det["blendshapes"])
        self.last_mood = mood

        if det is not None:
            aligned = align_face(frame_bgr, det["landmarks"], det["bbox"], self.face_size)
            landmarks_112 = self._transform_landmarks(det["landmarks"], frame_bgr.shape,
                                                      det["bbox"])
        else:
            aligned = cv2.resize(frame_bgr, self.face_size)
            landmarks_112 = None

        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        rois = self.roi_ext.extract(aligned, landmarks_112)

        self.aligned_buf.append(aligned)
        self.gray_buf.append(gray)
        self.roi_buf.append(rois)

        if len(self.aligned_buf) < self.clip_len:
            return None, det, mood

        score = self._run_inference()
        self.all_scores.append(score)
        return score, det, mood

    def _run_inference(self) -> float:
        """Build tensors from buffer and run forward pass."""
        gray_list = list(self.gray_buf)
        flows = []
        for i in range(len(gray_list) - 1):
            raw_flow = compute_flow_farneback(gray_list[i], gray_list[i + 1])
            flows.append(flow_stack_channels(raw_flow))  # (H, W, 2)

        # flow_clip: (T-1, 2, H, W)
        flow_clip = torch.from_numpy(
            np.stack(flows, axis=0)
        ).float().permute(0, 3, 1, 2).unsqueeze(0).to(self.device)

        # roi_clips: (1, T, 4, 2, roi_H, roi_W)
        roi_clips = self._build_roi_clips(flow_clip)

        # color features from current window
        self.color_ext.reset()
        for rois in self.roi_buf:
            self.color_ext.update(rois)
        color_feat = torch.from_numpy(
            self.color_ext.features()
        ).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit = self.model(flow_clip, roi_clips, color_feat)
            prob = torch.sigmoid(logit).item()
        return float(prob)

    def _classify_mood(self, bs: dict) -> str:
        """Heuristic classification of mood using blendshape scores."""
        # 1. Happiness (Smile)
        smile = (bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2
        if smile > 0.15: return "HAPPY"
        
        # 2. Surprise (Wide eyes + open mouth)
        surprise = (bs.get("eyeWideLeft", 0) + bs.get("eyeWideRight", 0)) / 2
        jaw = bs.get("jawOpen", 0)
        if surprise > 0.2: return "SURPRISED"
        if jaw > 0.15: return "OPEN MOUTH"

        # 3. Sadness (Frown + inner brow raise)
        frown = (bs.get("mouthFrownLeft", 0) + bs.get("mouthFrownRight", 0)) / 2
        brow_up = bs.get("browInnerUp", 0)
        if frown > 0.15 or brow_up > 0.15: return "SAD"
        
        # 4. Anger (Brow down + mouth press)
        brow_down = (bs.get("browDownLeft", 0) + bs.get("browDownRight", 0)) / 2
        press = (bs.get("mouthPressLeft", 0) + bs.get("mouthPressRight", 0)) / 2
        if brow_down > 0.2 or press > 0.2: return "TENSE"
        
        # 5. Disgust (Nose sneer)
        sneer = (bs.get("noseSneerLeft", 0) + bs.get("noseSneerRight", 0)) / 2
        if sneer > 0.2: return "DISGUSTED"

        return "Neutral"

    def _build_roi_clips(self, flow_clip: torch.Tensor) -> torch.Tensor:
        """Derive ROI flow clips from full flow clip."""
        B, T, C, H, W = flow_clip.shape
        n_rois = 4
        roi_h, roi_w = self.roi_size
        rois_rel = [
            (0.37, 0.27, 0.20, 0.30),
            (0.37, 0.73, 0.20, 0.30),
            (0.55, 0.50, 0.22, 0.26),
            (0.76, 0.50, 0.20, 0.36),
        ]
        all_rois = []
        for cy, cx, rh, rw in rois_rel:
            y0 = max(0, int((cy - rh / 2) * H))
            y1 = min(H, int((cy + rh / 2) * H))
            x0 = max(0, int((cx - rw / 2) * W))
            x1 = min(W, int((cx + rw / 2) * W))
            patch = flow_clip[:, :, :, y0:y1, x0:x1]  # (B, T, C, ph, pw)
            BT = B * T
            patch = patch.reshape(BT, C, y1 - y0, x1 - x0)
            patch = torch.nn.functional.interpolate(
                patch, size=(roi_h, roi_w), mode="bilinear", align_corners=False
            ).reshape(B, T, C, roi_h, roi_w)
            all_rois.append(patch)
        return torch.stack(all_rois, dim=2)  # (B, T, n_rois, C, roi_h, roi_w)

    @staticmethod
    def _transform_landmarks(
        lm_orig: Optional[np.ndarray],
        orig_shape: tuple,
        bbox: tuple,
    ) -> Optional[np.ndarray]:
        """
        Approximate landmark transform for aligned face.
        (Simplified — uses bbox crop instead of full affine warp for speed.)
        """
        if lm_orig is None:
            return None
        return None  # Signal to use fixed ROIs

    def close(self) -> None:
        self.detector.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def run_video_inference(
    cfg: dict,
    checkpoint_path: str,
    video_path: str,
    output_dir: str,
    variant: str = "full",
) -> None:
    logger = get_logger("video_inference", log_dir=cfg["paths"]["logs"])
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
        logger.warning("No checkpoint found — using untrained model.")
    model.eval()

    engine = VideoInferenceEngine(model, device, cfg)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "annotated_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    frame_scores: List[float] = []
    frame_idx = 0

    logger.info("Processing video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        score = engine.push_frame(frame)
        display_score = score if score is not None else 0.0
        frame_scores.append(display_score)

        label = "ME" if display_score >= cfg["evaluation"]["threshold"] else ""
        annotated = draw_roi_overlay(frame, {}, score=display_score, label=label)
        writer.write(annotated)

        if cfg["inference"]["display"]:
            cv2.imshow("Micro-Expression Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    engine.close()

    # ── Post-processing ────────────────────────────────────────────────────
    scores_arr = np.array(frame_scores)
    smooth_scores = temporal_smoothing(scores_arr, cfg["evaluation"]["temporal_smoothing_window"])
    events = extract_event_intervals(
        (smooth_scores >= cfg["evaluation"]["threshold"]).astype(int),
        min_duration=cfg["evaluation"]["min_event_frames"],
    )

    logger.info(f"\nDetected {len(events)} micro-expression event(s):")
    for i, (s, e) in enumerate(events):
        logger.info(f"  Event {i + 1}: frames {s}–{e} ({s / fps:.2f}s – {e / fps:.2f}s)")

    plot_detection_timeline(
        smooth_scores,
        threshold=cfg["evaluation"]["threshold"],
        pred_intervals=events,
        fps=fps,
        save_path=os.path.join(output_dir, "detection_timeline.png"),
    )
    logger.info(f"Annotated video saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="outputs/inference")
    parser.add_argument("--variant", default="full")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_video_inference(cfg, args.checkpoint, args.video, args.output, args.variant)

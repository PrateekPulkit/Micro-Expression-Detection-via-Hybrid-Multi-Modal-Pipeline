"""
preprocessing/alignment.py
Similarity-transform-based face alignment to a canonical 112x112 space.
Uses five reference landmarks (eye centres + nose tip + mouth corners).
"""
from __future__ import annotations

import cv2
import numpy as np

# Canonical 5-point reference (ArcFace-style, scaled to 112x112)
_REF_112 = np.array(
    [
        [38.29, 51.69],
        [73.53, 51.69],
        [56.02, 71.73],
        [41.55, 92.37],
        [70.73, 92.37],
    ],
    dtype=np.float32,
)

# MediaPipe landmark indices for the 5 reference points
_REF_LM_IDX = {
    "left_eye_centre":   [133, 33],   # average these
    "right_eye_centre":  [362, 263],
    "nose_tip":          [1],
    "mouth_left":        [61],
    "mouth_right":       [291],
}


def _get_5pts_from_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Extract 5 reference points from 468 MediaPipe landmarks."""
    left_eye  = landmarks[[133, 33],  :].mean(axis=0)
    right_eye = landmarks[[362, 263], :].mean(axis=0)
    nose_tip  = landmarks[1]
    m_left    = landmarks[61]
    m_right   = landmarks[291]
    return np.array([left_eye, right_eye, nose_tip, m_left, m_right], dtype=np.float32)


def align_face(
    frame_bgr: np.ndarray,
    landmarks: np.ndarray | None,
    bbox: tuple | None = None,
    out_size: tuple = (112, 112),
) -> np.ndarray:
    """
    Align and crop face using similarity transform.

    Args:
        frame_bgr: Input BGR frame.
        landmarks: (468, 2) MediaPipe landmarks, or None → use simple crop.
        bbox: (x, y, w, h) fallback bounding box when landmarks is None.
        out_size: Output face size (W, H).

    Returns:
        Aligned BGR face crop of shape (out_size[1], out_size[0], 3).
    """
    if landmarks is not None:
        src_pts = _get_5pts_from_landmarks(landmarks)
        scale_x = out_size[0] / 112.0
        scale_y = out_size[1] / 112.0
        dst_pts = _REF_112.copy()
        dst_pts[:, 0] *= scale_x
        dst_pts[:, 1] *= scale_y

        # Estimate similarity transform (rotation + uniform scale + translation)
        M, _ = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, method=cv2.LMEDS
        )
        if M is None:
            # Fallback to bbox crop
            return _bbox_crop(frame_bgr, bbox, out_size)
        aligned = cv2.warpAffine(frame_bgr, M, out_size, flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
        return aligned
    else:
        return _bbox_crop(frame_bgr, bbox, out_size)


def _bbox_crop(
    frame_bgr: np.ndarray,
    bbox: tuple | None,
    out_size: tuple,
) -> np.ndarray:
    """Simple bounding-box crop and resize."""
    h, w = frame_bgr.shape[:2]
    if bbox is None:
        return cv2.resize(frame_bgr, out_size)
    x, y, bw, bh = bbox
    x, y = max(0, x), max(0, y)
    bw = min(bw, w - x)
    bh = min(bh, h - y)
    if bw <= 0 or bh <= 0:
        return cv2.resize(frame_bgr, out_size)
    crop = frame_bgr[y: y + bh, x: x + bw]
    return cv2.resize(crop, out_size)

"""
preprocessing/optical_flow.py
Optical flow computation utilities.

Provides:
  - Farneback dense optical flow (primary method)
  - TV-L1 optical flow (optional, requires OpenCV contrib)
  - Magnitude and angle map extraction
  - HOOF (Histogram of Oriented Optical Flow) — baseline reproduction
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


# ── Farneback parameters (sensible defaults for ME) ─────────────────────────
_DEFAULT_FB = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)


def compute_flow_farneback(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """
    Compute dense optical flow using Farneback algorithm.

    Returns:
        flow: (H, W, 2) float32 — (dx, dy) per pixel.
    """
    params = {**_DEFAULT_FB, **kwargs}
    return cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        params["pyr_scale"], params["levels"], params["winsize"],
        params["iterations"], params["poly_n"], params["poly_sigma"],
        params["flags"],
    )


def compute_flow_tvl1(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
) -> np.ndarray:
    """
    TV-L1 optical flow (requires opencv-contrib-python).
    Falls back to Farneback if unavailable.
    """
    try:
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
        return tvl1.calc(prev_gray, curr_gray, None)
    except AttributeError:
        return compute_flow_farneback(prev_gray, curr_gray)


def flow_to_mag_angle(flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert (dx, dy) flow to magnitude and angle maps."""
    mag, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return mag.astype(np.float32), angle.astype(np.float32)


def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """Convert optical flow to HSV-encoded RGB image (for visualization)."""
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def compute_hoof(
    flow: np.ndarray,
    n_bins: int = 8,
    magnitude_threshold: float = 1.0,
) -> np.ndarray:
    """
    Histogram of Oriented Optical Flow — reproduces the paper's baseline feature.

    Args:
        flow: (H, W, 2) optical flow.
        n_bins: Number of orientation bins (0–360°).
        magnitude_threshold: Minimum magnitude to include.

    Returns:
        hoof: Normalized histogram vector of length n_bins.
    """
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mask = mag > magnitude_threshold
    angles = ang[mask] * 180.0 / np.pi  # 0–360

    hist, _ = np.histogram(angles, bins=n_bins, range=(0, 360),
                           weights=mag[mask])
    norm = hist.sum()
    if norm > 0:
        hist = hist / norm
    return hist.astype(np.float32)


def compute_flow_sequence(
    frames_gray: list[np.ndarray],
    method: str = "farneback",
    **kwargs,
) -> list[np.ndarray]:
    """
    Compute optical flow for a sequence of grayscale frames.

    Returns:
        List of (H, W, 2) flow arrays, length = len(frames_gray) - 1.
    """
    flows = []
    fn = compute_flow_farneback if method == "farneback" else compute_flow_tvl1
    for i in range(len(frames_gray) - 1):
        flows.append(fn(frames_gray[i], frames_gray[i + 1], **kwargs))
    return flows


def flow_stack_channels(
    flow: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """
    Stack magnitude and angle into a 2-channel image normalized to [0, 1].
    This is the representation fed into the CNN.

    Returns:
        stacked: (H, W, 2) float32
    """
    mag, ang = flow_to_mag_angle(flow)
    if normalize:
        mag = cv2.normalize(mag, None, 0.0, 1.0, cv2.NORM_MINMAX)
        ang = ang / (2 * np.pi)         # normalize angle to [0, 1]
    return np.stack([mag, ang], axis=-1)

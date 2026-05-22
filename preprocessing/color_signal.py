"""
preprocessing/color_signal.py
═══════════════════════════════════════════════════════════════════════════════
Novel Contribution — Chromo-Temporal Signal Stream
═══════════════════════════════════════════════════════════════════════════════

Extracts subtle skin-tone variation signals from stable facial ROIs (forehead,
left/right cheek) by analysing temporal fluctuations in YCbCr Cb/Cr channels
and HSV H/S channels.

Physiological basis:
  - Blood flow and micro-emotional responses cause sub-perceptual skin
    colour changes (physiologically linked to vascular response).
  - These changes manifest as slow periodic waves in chrominance channels.
  - Suppressed expressions (micro-expressions) are correlated with these
    vascular responses due to autonomic nervous system activity.

This stream adds a COMPLEMENTARY dimension to motion-based analysis,
reducing false positives when motion is ambiguous.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.signal import butter, filtfilt


# ── Butterworth bandpass filter ──────────────────────────────────────────────
def _bandpass_filter(signal: np.ndarray, fps: float,
                     low: float = 0.5, high: float = 4.0,
                     order: int = 3) -> np.ndarray:
    """
    Bandpass filter for physiological signals.
    Default: 0.5–4.0 Hz — covers resting heart rate range (30–240 bpm).
    For ME (microseconds–seconds), a wider band may be used.
    """
    nyq = fps / 2.0
    low_n = max(low / nyq, 0.001)
    high_n = min(high / nyq, 0.999)
    if low_n >= high_n:
        return signal
    b, a = butter(order, [low_n, high_n], btype="band")
    if len(signal) < max(len(b), len(a)) * 3:
        return signal           # too short to filter
    return filtfilt(b, a, signal)


# ── Per-ROI color extractor ──────────────────────────────────────────────────
def extract_mean_color(
    roi_bgr: np.ndarray,
    channels: Tuple[str, ...] = ("Cb", "Cr", "S"),
) -> Dict[str, float]:
    """
    Extract mean channel values from a single BGR ROI patch.

    Returns:
        dict, e.g. {"Cb": 120.3, "Cr": 135.7, "S": 89.2}
    """
    results: Dict[str, float] = {}
    for ch in channels:
        if ch in ("Y", "Cb", "Cr"):
            ycbcr = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)  # OpenCV: Y Cr Cb
            c_map = {"Y": 0, "Cr": 1, "Cb": 2}
            results[ch] = float(ycbcr[:, :, c_map[ch]].mean())
        elif ch in ("H", "S", "V"):
            hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            c_map = {"H": 0, "S": 1, "V": 2}
            results[ch] = float(hsv[:, :, c_map[ch]].mean())
        elif ch in ("R", "G", "B"):
            c_map = {"B": 0, "G": 1, "R": 2}
            results[ch] = float(roi_bgr[:, :, c_map[ch]].mean())
    return results


# ── Temporal feature extraction from a sequence of ROI patches ──────────────
class ColorSignalExtractor:
    """
    Accumulates color channel values over a clip and extracts temporal features.

    Usage pattern:
        extractor = ColorSignalExtractor()
        for roi_dict in clip_rois:          # list of per-frame ROI dicts
            extractor.update(roi_dict)
        feature_vec = extractor.features()
        extractor.reset()

    clip_rois: list of {region_name: np.ndarray (H,W,3)} per frame.
    """

    _TARGET_REGIONS = ["forehead", "left_cheek", "right_cheek"]
    _CHANNELS = ("Cb", "Cr", "S")

    def __init__(self, fps: float = 200.0, smooth_window: int = 5) -> None:
        self.fps = fps
        self.smooth_window = smooth_window
        self._buffer: Dict[str, Dict[str, List[float]]] = {
            r: {c: [] for c in self._CHANNELS} for r in self._TARGET_REGIONS
        }

    def reset(self) -> None:
        for r in self._TARGET_REGIONS:
            for c in self._CHANNELS:
                self._buffer[r][c] = []

    def update(self, rois: Dict[str, np.ndarray]) -> None:
        """Add one frame's worth of ROIs to the buffer."""
        for region in self._TARGET_REGIONS:
            if region not in rois:
                continue
            vals = extract_mean_color(rois[region], self._CHANNELS)
            for ch, v in vals.items():
                self._buffer[region][ch].append(v)

    def features(self) -> np.ndarray:
        """
        Extract a fixed-length feature vector from accumulated signals.

        Per (region × channel) pair, compute:
          - mean, std, temporal max-min range → 3 features each
          - Total: 3 regions × 3 channels × 3 stats = 27 features

        Also adds:
          - Standard deviation of first-order temporal differences (3×3 = 9)
          - Z-score peak count (3×3 = 9)
          → Grand total: 27 + 9 + 9 = 45, but we cap at 12 via PCA-like selection
            to keep the downstream MLP lightweight.

        Returns:
            feature_vec: np.ndarray, shape (n_features,), float32.
        """
        features = []
        for region in self._TARGET_REGIONS:
            for ch in self._CHANNELS:
                sig = np.array(self._buffer[region][ch], dtype=np.float32)
                if len(sig) < 4:
                    features.extend([0.0, 0.0, 0.0, 0.0])
                    continue

                # Smooth (running mean)
                kernel = np.ones(self.smooth_window) / self.smooth_window
                sig_sm = np.convolve(sig, kernel, mode="same")

                # Temporal difference
                diff = np.diff(sig_sm)

                features.append(float(sig_sm.mean()))
                features.append(float(sig_sm.std()))
                features.append(float(sig_sm.max() - sig_sm.min()))
                features.append(float(diff.std()))

        return np.array(features, dtype=np.float32)  # 3×3×4 = 36 features

    @staticmethod
    def expected_feature_dim() -> int:
        """Return expected feature vector length."""
        return len(ColorSignalExtractor._TARGET_REGIONS) * len(
            ColorSignalExtractor._CHANNELS) * 4   # 36


# ── Clip-level batch extraction ──────────────────────────────────────────────
def extract_color_features_clip(
    roi_sequence: List[Dict[str, np.ndarray]],
    fps: float = 200.0,
) -> np.ndarray:
    """
    Convenience function: extract color signal features from a clip.

    Args:
        roi_sequence: List of per-frame ROI dicts (each: {name: patch}).
        fps: Video frame rate.

    Returns:
        feature_vec: (36,) float32
    """
    extractor = ColorSignalExtractor(fps=fps)
    for rois in roi_sequence:
        extractor.update(rois)
    return extractor.features()

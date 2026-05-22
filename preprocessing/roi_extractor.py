"""
preprocessing/roi_extractor.py
Extracts region-of-interest (ROI) patches from an aligned face.

Regions: left_eye, right_eye, nose, mouth, forehead (color signal only).
Each ROI is resized to roi_size for consistent downstream processing.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

from .face_detector import FaceDetector

# Fixed proportional bounding boxes for aligned 112x112 faces.
# Format: (cx, cy, w, h) as fractions of face width/height.
# Used when landmark-based extraction is unavailable.
_FIXED_ROIS_REL = {
    "left_eye":   (0.27, 0.37, 0.30, 0.18),
    "right_eye":  (0.73, 0.37, 0.30, 0.18),
    "nose":       (0.50, 0.55, 0.26, 0.22),
    "mouth":      (0.50, 0.76, 0.36, 0.20),
    "forehead":   (0.50, 0.15, 0.60, 0.18),
    "left_cheek": (0.22, 0.60, 0.24, 0.22),
    "right_cheek":(0.78, 0.60, 0.24, 0.22),
}


def _rel_to_abs(rel: Tuple, face_w: int, face_h: int) -> Tuple[int, int, int, int]:
    """Convert proportional (cx, cy, w, h) → absolute (x, y, w, h)."""
    cx, cy, rw, rh = rel
    w = int(rw * face_w)
    h = int(rh * face_h)
    x = max(0, int(cx * face_w - w / 2))
    y = max(0, int(cy * face_h - h / 2))
    w = min(w, face_w - x)
    h = min(h, face_h - y)
    return (x, y, w, h)


class ROIExtractor:
    """
    Extracts named facial ROI patches from an aligned face crop.

    Usage:
        extractor = ROIExtractor(roi_size=(32, 32))
        rois = extractor.extract(aligned_face, landmarks_112)
        # rois: {region_name: np.ndarray shape (32, 32, C)}
    """

    MOTION_ROIS = ["left_eye", "right_eye", "nose", "mouth"]
    COLOR_ROIS  = ["forehead", "left_cheek", "right_cheek"]

    # MediaPipe landmark index groups mapped to aligned 112x112 face
    _LM_GROUPS = {
        "left_eye":    [33, 133, 160, 159, 158, 153, 145, 144],
        "right_eye":   [362, 263, 387, 386, 385, 380, 374, 373],
        "nose":        [1, 4, 49, 240, 97, 99, 2, 164],
        "mouth":       [61, 291, 0, 17, 269, 39, 405, 181],
        "forehead":    [10, 338, 297, 332, 284, 251, 389, 109, 67, 103, 54, 21],
        "left_cheek":  [116, 123, 147, 192, 213, 214, 212],
        "right_cheek": [345, 352, 376, 416, 433, 434, 432],
    }

    def __init__(self, roi_size: Tuple[int, int] = (32, 32)) -> None:
        self.roi_size = roi_size

    def extract(
        self,
        aligned_face: np.ndarray,
        landmarks_112: np.ndarray | None = None,
        pad: float = 0.12,
    ) -> Dict[str, np.ndarray]:
        """
        Extract ROI patches from an aligned 112x112 face.

        Args:
            aligned_face: BGR image (H, W, 3) — already aligned.
            landmarks_112: (468, 2) landmarks in aligned face coordinates, or None.
            pad: Padding fraction around landmark bounding box.

        Returns:
            Dict[str, np.ndarray]: {roi_name: (roi_h, roi_w, 3)}
        """
        fh, fw = aligned_face.shape[:2]
        rois: Dict[str, np.ndarray] = {}

        all_regions = self.MOTION_ROIS + self.COLOR_ROIS

        for name in all_regions:
            if landmarks_112 is not None and name in self._LM_GROUPS:
                bbox = FaceDetector.get_roi_bbox(
                    landmarks_112, self._LM_GROUPS[name], (fh, fw), pad=pad
                )
            else:
                bbox = _rel_to_abs(_FIXED_ROIS_REL[name], fw, fh)

            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                rois[name] = np.zeros((*self.roi_size[::-1], 3), dtype=np.uint8)
                continue
            patch = aligned_face[y: y + h, x: x + w]
            rois[name] = cv2.resize(patch, self.roi_size, interpolation=cv2.INTER_LINEAR)

        return rois

    def get_bbox_map(
        self,
        aligned_face: np.ndarray,
        landmarks_112: np.ndarray | None = None,
        pad: float = 0.12,
    ) -> Dict[str, Tuple[int, int, int, int]]:
        """Return raw (x,y,w,h) for each ROI (for visualization)."""
        fh, fw = aligned_face.shape[:2]
        bboxes: Dict[str, Tuple] = {}
        for name in self.MOTION_ROIS + self.COLOR_ROIS:
            if landmarks_112 is not None and name in self._LM_GROUPS:
                bboxes[name] = FaceDetector.get_roi_bbox(
                    landmarks_112, self._LM_GROUPS[name], (fh, fw), pad=pad
                )
            else:
                bboxes[name] = _rel_to_abs(_FIXED_ROIS_REL[name], fw, fh)
        return bboxes

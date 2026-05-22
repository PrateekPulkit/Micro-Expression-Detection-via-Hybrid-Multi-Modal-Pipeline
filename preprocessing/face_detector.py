"""
preprocessing/face_detector.py
MediaPipe FaceMesh-based face detector and landmark extractor.
Provides a stable, lightweight alternative to dlib.
"""
from __future__ import annotations

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    _MP_AVAILABLE = True
except (ImportError, AttributeError):
    _MP_AVAILABLE = False

# Fallback: OpenCV Haar cascade if MediaPipe not installed
_HAAR_XML = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


class FaceDetector:
    """
    Detects faces and returns 468 MediaPipe landmarks (or bbox fallback).

    Usage:
        detector = FaceDetector()
        result = detector.detect(frame_bgr)
        # result: {"bbox": (x, y, w, h), "landmarks": np.ndarray (468, 2)} or None
    """

    # Key MediaPipe landmark indices for facial subregions
    _LANDMARK_IDX = {
        "left_eye":   [33, 133, 160, 159, 158, 144, 153, 145, 144],
        "right_eye":  [362, 263, 387, 386, 385, 373, 380, 374, 373],
        "nose":       [1, 2, 98, 327, 195, 197],
        "mouth":      [61, 291, 39, 269, 0, 17, 405, 181],
        "left_brow":  [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
        "right_brow": [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
        "forehead":   [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                        361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                        176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
                        162, 21, 54, 103, 67, 109],
        "left_cheek": [116, 123, 147, 213, 192, 214, 212, 216, 206, 203],
        "right_cheek": [345, 352, 376, 433, 416, 434, 432, 436, 426, 423],
    }

    def __init__(self, static_image_mode: bool = False,
                 max_num_faces: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5) -> None:
        self._use_mediapipe = _MP_AVAILABLE
        self._model_path = "models/checkpoints/face_landmarker.task"

        if self._use_mediapipe:
            try:
                base_options = python.BaseOptions(model_asset_path=self._model_path)
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO,
                    num_faces=max_num_faces,
                    min_face_detection_confidence=min_detection_confidence,
                    min_face_presence_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=False,
                )
                self._landmarker = vision.FaceLandmarker.create_from_options(options)
                self._static_mode = static_image_mode
                self._frame_timestamp = 0
            except Exception as e:
                print(f"[FaceDetector] MediaPipe Task initialization failed: {e}")
                self._use_mediapipe = False

        if not self._use_mediapipe:
            print("[FaceDetector] Using Haar cascade fallback.")
            self._cascade = cv2.CascadeClassifier(_HAAR_XML)

    # ------------------------------------------------------------------ #
    def detect(self, frame_bgr: np.ndarray) -> dict | None:
        """
        Run detection on a BGR frame.

        Returns:
            dict with keys ["bbox", "landmarks", "roi_indices"] or None.
            landmarks: (468, 2) pixel coords (or None for Haar fallback).
        """
        h, w = frame_bgr.shape[:2]

        if self._use_mediapipe:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            if self._static_mode:
                result = self._landmarker.detect(mp_image)
            else:
                result = self._landmarker.detect_for_video(mp_image, self._frame_timestamp)
                self._frame_timestamp += 33 # assume ~30fps for timestamp inc
            
            if not result or not result.face_landmarks:
                return None
            
            lm = result.face_landmarks[0]
            pts = np.array(
                [(lm_pt.x * w, lm_pt.y * h) for lm_pt in lm],
                dtype=np.float32,
            )  # (478, 2) - Tasks API has 478 points (FaceMesh + mesh refinement)
            
            # Use only first 468 for legacy compatibility if needed, 
            # but our indices map to 468.
            # Actually, MediaPipe Task landmarks are indexed same for first 468.
            
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            bbox = (
                max(0, int(x_min)),
                max(0, int(y_min)),
                min(w, int(x_max - x_min)),
                min(h, int(y_max - y_min)),
            )
            res_dict = {
                "bbox": bbox, 
                "landmarks": pts[:468], 
                "roi_indices": self._LANDMARK_IDX
            }
            
            if hasattr(result, "face_blendshapes") and result.face_blendshapes:
                res_dict["blendshapes"] = {
                    cat.category_name: cat.score for cat in result.face_blendshapes[0]
                }
            return res_dict
        else:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self._cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            if len(faces) == 0:
                return None
            x, y, fw, fh = faces[0]
            return {"bbox": (x, y, fw, fh), "landmarks": None, "roi_indices": {}}

    def close(self) -> None:
        if self._use_mediapipe and hasattr(self, "_landmarker"):
            self._landmarker.close()

    # ------------------------------------------------------------------ #
    @staticmethod
    def get_roi_bbox(
        landmarks: np.ndarray,
        indices: list,
        frame_shape: tuple,
        pad: float = 0.15,
    ) -> tuple:
        """
        Compute padded bounding box around a set of landmark indices.

        Returns:
            (x, y, w, h) in pixel coordinates, clipped to frame.
        """
        h, w = frame_shape[:2]
        pts = landmarks[indices]
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        rw = x_max - x_min
        rh = y_max - y_min
        x_min = max(0, int(x_min - pad * rw))
        y_min = max(0, int(y_min - pad * rh))
        x_max = min(w, int(x_max + pad * rw))
        y_max = min(h, int(y_max + pad * rh))
        return (x_min, y_min, x_max - x_min, y_max - y_min)

"""
preprocessing/dataset_builder.py
═══════════════════════════════════════════════════════════════════════════════
Builds preprocessed clip datasets from raw ME video sequences.

Supports: CASME II, SAMM, SMIC
Outputs:  .npz files per clip containing:
            - flow_clip:   (T-1, H, W, 2) float32  optical flow stacks
            - label:       int (0/1 binary)
            - subject_id:  str
            - clip_id:     str
            - color_feat:  (36,) float32  color signal features
            - roi_keys:    list of ROI names
            - T:           int clip length

Run as a script or import as module.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .alignment import align_face
from .color_signal import extract_color_features_clip
from .face_detector import FaceDetector
from .optical_flow import compute_flow_farneback, flow_stack_channels
from .roi_extractor import ROIExtractor


# ── Dataset reader stubs ─────────────────────────────────────────────────────

def _load_casme2_annotations(data_root: str) -> pd.DataFrame:
    """
    Load CASME II annotation file.
    Expects: data/raw/CASME2/CASME2-coding-20140508.xlsx

    Returns DataFrame with columns:
      subject, filename, onset, apex, offset, action_units, emotion
    """
    ann_path = os.path.join(data_root, "CASME2-coding-20140508.xlsx")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(
            f"CASME II annotation file not found at: {ann_path}\n"
            "Download from: http://casme.psych.ac.cn/casme/e"
        )
    df = pd.read_excel(ann_path)
    # Normalise column names (sheet varies by version)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    rename_map = {
        "subject": "subject", "filename": "filename",
        "onset": "onset", "apex": "apex", "offset": "offset",
        "estimated_emotion": "emotion", "action_units": "action_units",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    return df


def _load_samm_annotations(data_root: str) -> pd.DataFrame:
    """
    Load SAMM annotation file.
    Expects: data/raw/SAMM/SAMM_Micro_FACS_Codes_v2.xlsx
    """
    ann_path = os.path.join(data_root, "SAMM_Micro_FACS_Codes_v2.xlsx")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(
            f"SAMM annotation file not found at: {ann_path}\n"
            "Download from: http://www2.docm.mmu.ac.uk/STAFF/m.yap/dataset.php"
        )
    df = pd.read_excel(ann_path, header=1)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def _load_smic_annotations(data_root: str) -> pd.DataFrame:
    """
    Load SMIC annotation file.
    Expects: data/raw/SMIC/SMIC-HS-E.xls
    """
    ann_path = os.path.join(data_root, "SMIC-HS-E.xls")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(
            f"SMIC annotation file not found at: {ann_path}\n"
            "Download from: http://www.cse.oulu.fi/SMICDatabase"
        )
    df = pd.read_excel(ann_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


_ANN_LOADERS = {
    "CASME2": _load_casme2_annotations,
    "SAMM": _load_samm_annotations,
    "SMIC": _load_smic_annotations,
}


# ── Frame reader ─────────────────────────────────────────────────────────────

def read_sequence_frames(
    seq_dir: str,
    image_ext: str = ".jpg",
    sort: bool = True,
) -> List[np.ndarray]:
    """Load all image frames from a directory in sorted order."""
    pattern = re.compile(r"\." + image_ext.lstrip(".") + "$", re.IGNORECASE)
    paths = sorted(
        [os.path.join(seq_dir, f) for f in os.listdir(seq_dir) if pattern.search(f)]
    ) if sort else [
        os.path.join(seq_dir, f) for f in os.listdir(seq_dir) if pattern.search(f)
    ]
    frames = []
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            frames.append(img)
    return frames


# ── Core preprocessing function ──────────────────────────────────────────────

def preprocess_sequence(
    frames: List[np.ndarray],
    detector: FaceDetector,
    extractor: ROIExtractor,
    face_size: Tuple[int, int] = (112, 112),
    fps: float = 200.0,
    clip_len: int = 16,
    clip_stride: int = 4,
) -> List[Dict]:
    """
    Preprocess a raw ME video sequence into clips.

    Each clip contains:
        - flow_clip: (T-1, H, W, 2)
        - roi_clips: {name: (T, roi_h, roi_w, 3)}
        - color_feat: (36,)
        - aligned_frames: (T, fh, fw, 3)

    Returns: list of clip dicts.
    """
    if len(frames) < 2:
        return []

    # ── Step 1: Detect & align every frame ──────────────────────────────────
    aligned_frames = []
    landmarks_seq = []

    for frame in frames:
        det = detector.detect(frame)
        if det is None:
            # Use previous aligned frame or raw resize if first
            af = (aligned_frames[-1].copy() if aligned_frames
                  else cv2.resize(frame, face_size))
            lm = landmarks_seq[-1] if landmarks_seq else None
        else:
            af = align_face(frame, det["landmarks"], det["bbox"], face_size)
            lm = det["landmarks"]
        aligned_frames.append(af)
        landmarks_seq.append(lm)

    # ── Step 2: Compute optical flow sequence ────────────────────────────────
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in aligned_frames]
    flows = []
    for i in range(len(grays) - 1):
        raw_flow = compute_flow_farneback(grays[i], grays[i + 1])
        flows.append(flow_stack_channels(raw_flow))  # (H, W, 2)

    # ── Step 3: Extract ROIs per frame ───────────────────────────────────────
    rois_seq = []
    for af, lm in zip(aligned_frames, landmarks_seq):
        rois_seq.append(extractor.extract(af, lm))

    # ── Step 4: Sliding-window clipping ─────────────────────────────────────
    clips = []
    n = len(aligned_frames)
    for start in range(0, n - clip_len + 1, clip_stride):
        end = start + clip_len
        clip = {
            "flow_clip": np.stack(flows[start: end - 1], axis=0),  # (T-1, H, W, 2)
            "aligned_frames": np.stack(aligned_frames[start:end], axis=0),
            "color_feat": extract_color_features_clip(rois_seq[start:end], fps),
            "rois": rois_seq[start:end],  # list of dicts
        }
        clips.append(clip)
    return clips


# ── Dataset builder entry point ──────────────────────────────────────────────

def build_dataset(
    dataset_name: str,
    data_root: str,
    save_dir: str,
    face_size: Tuple[int, int] = (112, 112),
    roi_size: Tuple[int, int] = (32, 32),
    clip_len: int = 16,
    clip_stride: int = 4,
    fps: float = 200.0,
    image_ext: str = ".jpg",
    max_sequences: Optional[int] = None,
    binary: bool = True,
) -> None:
    """
    Full dataset build pipeline.

    Iterates over all ME sequences, preprocesses them, and saves
    .npz clips to save_dir/{subject_id}/{clip_id}.npz

    Args:
        dataset_name: "CASME2" | "SAMM" | "SMIC"
        data_root: Root of raw dataset.
        save_dir: Output directory.
        max_sequences: Limit for debugging (None = process all).
        binary: If True, label = 0 (no ME) or 1 (ME). Else multi-class.
    """
    os.makedirs(save_dir, exist_ok=True)

    loader = _ANN_LOADERS.get(dataset_name)
    if loader is None:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose CASME2, SAMM, or SMIC.")
    annotations = loader(data_root)

    detector = FaceDetector(static_image_mode=False)
    roi_ext = ROIExtractor(roi_size=roi_size)

    # ── CASME II sequence layout ─────────────────────────────────────────────
    if dataset_name == "CASME2":
        cropped_root = os.path.join(data_root, "Cropped")
        _build_casme2(annotations, cropped_root, save_dir,
                      detector, roi_ext, face_size, fps, clip_len,
                      clip_stride, image_ext, max_sequences, binary)
    elif dataset_name == "SAMM":
        _build_samm(annotations, data_root, save_dir,
                    detector, roi_ext, face_size, fps, clip_len,
                    clip_stride, image_ext, max_sequences, binary)
    elif dataset_name == "SMIC":
        _build_smic(annotations, data_root, save_dir,
                    detector, roi_ext, face_size, fps, clip_len,
                    clip_stride, image_ext, max_sequences, binary)

    detector.close()
    print(f"[build_dataset] Done. Clips saved to: {save_dir}")


# ── CASME II ─────────────────────────────────────────────────────────────────

def _build_casme2(ann, cropped_root, save_dir, detector, roi_ext,
                  face_size, fps, clip_len, clip_stride, image_ext,
                  max_sequences, binary):
    processed = 0
    for _, row in tqdm(ann.iterrows(), total=len(ann), desc="CASME II"):
        if max_sequences and processed >= max_sequences:
            break
        subject = str(row.get("subject", "")).strip()
        filename = str(row.get("filename", "")).strip()
        onset = int(row.get("onset", 0))
        offset = int(row.get("offset", 0))
        emotion = str(row.get("emotion", "")).strip().lower()
        label = 1 if emotion not in ("", "nan", "repression") else 0
        if not binary:
            label = _casme2_emotion_to_idx(emotion)

        seq_dir = os.path.join(cropped_root, subject, filename)
        if not os.path.isdir(seq_dir):
            continue

        frames = read_sequence_frames(seq_dir, image_ext)
        if len(frames) < clip_len:
            continue

        clips = preprocess_sequence(frames, detector, roi_ext,
                                    face_size, fps, clip_len, clip_stride)
        subj_dir = os.path.join(save_dir, subject)
        os.makedirs(subj_dir, exist_ok=True)
        for ci, clip in enumerate(clips):
            _save_clip(clip, label, subject, f"{filename}_c{ci:03d}",
                       onset, offset, subj_dir)
        processed += 1


def _casme2_emotion_to_idx(emotion: str) -> int:
    mapping = {"happiness": 0, "disgust": 1, "repression": 2,
               "surprise": 3, "fear": 4, "sadness": 5, "others": 6}
    return mapping.get(emotion, 6)


# ── SAMM ─────────────────────────────────────────────────────────────────────

def _build_samm(ann, data_root, save_dir, detector, roi_ext,
                face_size, fps, clip_len, clip_stride, image_ext,
                max_sequences, binary):
    processed = 0
    for _, row in tqdm(ann.iterrows(), total=len(ann), desc="SAMM"):
        if max_sequences and processed >= max_sequences:
            break
        subject = str(int(row.get("subject", 0))).zfill(3)
        seq_name = str(row.get("filename", row.get("id_", ""))).strip()
        label = 1 if str(row.get("action_units", "")).strip() not in ("", "nan") else 0
        seq_dir = os.path.join(data_root, subject, seq_name)
        if not os.path.isdir(seq_dir):
            continue
        frames = read_sequence_frames(seq_dir, ".jpg")
        if len(frames) < clip_len:
            continue
        clips = preprocess_sequence(frames, detector, roi_ext,
                                    face_size, fps, clip_len, clip_stride)
        subj_dir = os.path.join(save_dir, subject)
        os.makedirs(subj_dir, exist_ok=True)
        for ci, clip in enumerate(clips):
            _save_clip(clip, label, subject, f"{seq_name}_c{ci:03d}",
                       0, 0, subj_dir)
        processed += 1


# ── SMIC ─────────────────────────────────────────────────────────────────────

def _build_smic(ann, data_root, save_dir, detector, roi_ext,
                face_size, fps, clip_len, clip_stride, image_ext,
                max_sequences, binary):
    """SMIC uses a different folder layout — subj/class/sequence."""
    processed = 0
    hs_root = os.path.join(data_root, "HS")
    for subj in sorted(os.listdir(hs_root)):
        subj_path = os.path.join(hs_root, subj)
        if not os.path.isdir(subj_path):
            continue
        for cls in sorted(os.listdir(subj_path)):
            cls_path = os.path.join(subj_path, cls)
            label = 1 if cls.lower() in ("positive", "negative", "surprise") else 0
            for seq in sorted(os.listdir(cls_path)):
                if max_sequences and processed >= max_sequences:
                    return
                seq_dir = os.path.join(cls_path, seq)
                if not os.path.isdir(seq_dir):
                    continue
                frames = read_sequence_frames(seq_dir, ".bmp")
                if len(frames) < clip_len:
                    continue
                clips = preprocess_sequence(frames, detector, roi_ext,
                                            face_size, fps, clip_len, clip_stride)
                out_dir = os.path.join(save_dir, subj)
                os.makedirs(out_dir, exist_ok=True)
                for ci, clip in enumerate(clips):
                    _save_clip(clip, label, subj, f"{cls}_{seq}_c{ci:03d}",
                               0, 0, out_dir)
                processed += 1


# ── Save clip ─────────────────────────────────────────────────────────────────

def _save_clip(clip, label, subject, clip_id, onset, offset, out_dir):
    """Save a single preprocessed clip as .npz."""
    out_path = os.path.join(out_dir, f"{clip_id}.npz")
    np.savez_compressed(
        out_path,
        flow_clip=clip["flow_clip"].astype(np.float32),
        aligned_frames=clip["aligned_frames"].astype(np.uint8),
        color_feat=clip["color_feat"].astype(np.float32),
        label=np.array(label, dtype=np.int64),
        subject=np.array(subject),
        clip_id=np.array(clip_id),
        onset=np.array(onset, dtype=np.int32),
        offset=np.array(offset, dtype=np.int32),
    )


# ── Loader (for training) ────────────────────────────────────────────────────

def load_clip_npz(path: str) -> Optional[Dict]:
    """Load a single .npz clip. Returns dict or None on error."""
    try:
        data = np.load(path, allow_pickle=True)
        return {
            "flow_clip": data["flow_clip"],              # (T-1, H, W, 2)
            "aligned_frames": data["aligned_frames"],    # (T, H, W, 3)
            "color_feat": data["color_feat"],            # (36,)
            "label": int(data["label"]),
            "subject": str(data["subject"]),
            "clip_id": str(data["clip_id"]),
        }
    except Exception as e:
        print(f"[load_clip_npz] Failed to load {path}: {e}")
        return None


def collect_all_clips(processed_dir: str) -> List[str]:
    """Recursively find all .npz clip files under processed_dir."""
    clips = []
    for root, _, files in os.walk(processed_dir):
        for f in files:
            if f.endswith(".npz"):
                clips.append(os.path.join(root, f))
    return sorted(clips)


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, yaml

    parser = argparse.ArgumentParser(description="Build preprocessed ME dataset")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--max_sequences", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ds = cfg["dataset"]
    pp = cfg["preprocessing"]

    build_dataset(
        dataset_name=ds["name"],
        data_root=os.path.join(cfg["paths"]["data_raw"], ds["name"]),
        save_dir=cfg["paths"]["data_processed"],
        face_size=tuple(pp["face_size"]),
        roi_size=tuple(pp["roi"]["roi_size"]),
        clip_len=pp["clip_len"],
        clip_stride=pp["clip_stride"],
        fps=ds["fps"],
        image_ext=ds["image_ext"],
        max_sequences=args.max_sequences,
        binary=ds["binary"],
    )

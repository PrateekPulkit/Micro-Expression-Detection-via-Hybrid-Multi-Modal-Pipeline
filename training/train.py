"""
training/train.py
═══════════════════════════════════════════════════════════════════════════════
Main training script.

Supports:
  - LOSO (Leave-One-Subject-Out) cross validation
  - Single train/val split
  - Cosine annealing LR schedule with linear warmup
  - Gradient clipping
  - Focal loss
  - Backbone unfreezing after warmup epochs
  - WandB logging (optional)

Usage:
    python -m training.train --config configs/config.yaml
    python -m training.train --config configs/config.yaml --variant full
    python -m training.train --config configs/config.yaml --variant baseline
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.micro_expr_net import build_model
from preprocessing.dataset_builder import collect_all_clips, load_clip_npz
from training.callbacks import CheckpointManager, EarlyStopping
from training.losses import build_loss
from utils import get_logger, set_seed


# ── Dataset ───────────────────────────────────────────────────────────────────

class MEClipDataset(Dataset):
    """
    PyTorch Dataset for preprocessed .npz clips.

    Returns:
        flow_clip:  (T, 2, H, W)  float32
        roi_clips:  (T, 4, 2, roi_H, roi_W) float32
        color_feat: (36,) float32
        label:      int64 scalar
    """

    MOTION_ROIS = ["left_eye", "right_eye", "nose", "mouth"]
    ROI_SIZE = (32, 32)

    def __init__(
        self,
        clip_paths: List[str],
        augment: bool = False,
        clip_len: int = 16,
    ) -> None:
        self.paths = clip_paths
        self.augment = augment
        self.clip_len = clip_len

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple:
        data = load_clip_npz(self.paths[idx])
        if data is None:
            # Return zero tensors on corrupt file
            return self._zeros()

        flow_clip = torch.from_numpy(data["flow_clip"]).float()   # (T-1, H, W, 2)
        T_minus1, H, W, _ = flow_clip.shape
        flow_clip = flow_clip.permute(0, 3, 1, 2)                 # (T-1, 2, H, W)

        # Pad/trim to self.clip_len - 1 frames
        target_t = self.clip_len - 1
        if T_minus1 < target_t:
            pad = torch.zeros(target_t - T_minus1, 2, H, W)
            flow_clip = torch.cat([flow_clip, pad], dim=0)
        else:
            flow_clip = flow_clip[:target_t]

        # Build dummy ROI clips (if ROI data not stored, compute from aligned frames)
        # Shape: (T, 4, 2, roi_H, roi_W)
        # Note: In a full pipeline, per-roi flow clips would be extracted during
        # preprocessing. Here we use a slice of global flow as a placeholder
        # when per-ROI flow isn't stored separately.
        roi_clips = self._extract_roi_flow(flow_clip)  # (T, 4, 2, 32, 32)

        color_feat = torch.from_numpy(data["color_feat"]).float()
        label = torch.tensor(data["label"], dtype=torch.long)

        if self.augment:
            flow_clip, roi_clips = self._augment(flow_clip, roi_clips)

        return flow_clip, roi_clips, color_feat, label

    def _extract_roi_flow(self, flow: torch.Tensor) -> torch.Tensor:
        """
        Extract 4 fixed spatial crops from full flow map as ROI proxies.
        flow: (T, 2, H, W)
        Returns: (T, 4, 2, 32, 32)
        """
        T, C, H, W = flow.shape
        # Proportional crop coords (cy, cx, h, w) relative to H, W
        rois_rel = [
            (0.37, 0.27, 0.20, 0.30),  # left_eye
            (0.37, 0.73, 0.20, 0.30),  # right_eye
            (0.55, 0.50, 0.22, 0.26),  # nose
            (0.76, 0.50, 0.20, 0.36),  # mouth
        ]
        crops = []
        for cy, cx, rh, rw in rois_rel:
            y0 = max(0, int((cy - rh / 2) * H))
            y1 = min(H, int((cy + rh / 2) * H))
            x0 = max(0, int((cx - rw / 2) * W))
            x1 = min(W, int((cx + rw / 2) * W))
            patch = flow[:, :, y0:y1, x0:x1]  # (T, 2, ph, pw)
            patch = nn.functional.interpolate(
                patch.reshape(T * C, 1, y1 - y0, x1 - x0),
                size=(32, 32), mode="bilinear", align_corners=False
            ).reshape(T, C, 32, 32)
            crops.append(patch)
        return torch.stack(crops, dim=1)  # (T, 4, 2, 32, 32)

    def _augment(self, flow, roi_clips):
        """Horizontal flip augmentation (reversed sign of x-component)."""
        if torch.rand(1).item() > 0.5:
            flow = torch.flip(flow, dims=[-1])
            flow[:, 0, :, :] = -flow[:, 0, :, :]  # negate x-flow
            roi_clips = torch.flip(roi_clips, dims=[-1])
            roi_clips[:, :, 0, :, :] = -roi_clips[:, :, 0, :, :]
        return flow, roi_clips

    def _zeros(self) -> Tuple:
        T = self.clip_len - 1
        return (
            torch.zeros(T, 2, 112, 112),
            torch.zeros(T, 4, 2, 32, 32),
            torch.zeros(36),
            torch.tensor(0, dtype=torch.long),
        )


# ── LOSO split ────────────────────────────────────────────────────────────────

def get_subject_from_path(path: str) -> str:
    """Extract subject ID from path (parent folder name)."""
    return Path(path).parent.name


def loso_splits(clip_paths: List[str]) -> List[Tuple[List[str], List[str]]]:
    """Generate (train, val) splits for each subject."""
    subjects = sorted(set(get_subject_from_path(p) for p in clip_paths))
    splits = []
    for subj in subjects:
        val = [p for p in clip_paths if get_subject_from_path(p) == subj]
        train = [p for p in clip_paths if get_subject_from_path(p) != subj]
        splits.append((train, val))
    return splits


def stratified_kfold_splits(clip_paths: List[str], k: int = 5, seed: int = 42) -> List[Tuple[List[str], List[str]]]:
    """Generate Stratified K-Fold splits based on emotion labels."""
    from sklearn.model_selection import StratifiedKFold
    
    # Extract labels for stratification
    labels = []
    for p in clip_paths:
        d = load_clip_npz(p)
        labels.append(d["label"] if d else 0)
    labels = np.array(labels)
    
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    splits = []
    paths_arr = np.array(clip_paths)
    
    for train_idx, val_idx in skf.split(paths_arr, labels):
        splits.append((paths_arr[train_idx].tolist(), paths_arr[val_idx].tolist()))
    return splits


def random_split(clip_paths: List[str], val_ratio: float = 0.15,
                 seed: int = 42) -> Tuple[List[str], List[str]]:
    rng = np.random.default_rng(seed)
    idxs = rng.permutation(len(clip_paths))
    val_n = int(len(clip_paths) * val_ratio)
    val_idxs = idxs[:val_n]
    train_idxs = idxs[val_n:]
    return [clip_paths[i] for i in train_idxs], [clip_paths[i] for i in val_idxs]


# ── Balanced sampler ──────────────────────────────────────────────────────────

def make_balanced_sampler(paths: List[str]) -> WeightedRandomSampler:
    """Up-sample minority positive class."""
    labels = []
    for p in paths:
        d = load_clip_npz(p)
        labels.append(d["label"] if d else 0)
    labels = np.array(labels)
    n_neg = max(1, (labels == 0).sum())
    n_pos = max(1, (labels == 1).sum())
    weights = np.where(labels == 1, n_neg / n_pos, 1.0)
    return WeightedRandomSampler(
        torch.from_numpy(weights).float(), num_samples=len(weights), replacement=True
    )


# ── Train / Eval loops ────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    for flow_clip, roi_clips, color_feat, labels in loader:
        flow_clip  = flow_clip.to(device)
        roi_clips  = roi_clips.to(device)
        color_feat = color_feat.to(device)
        labels     = labels.to(device).float()

        optimizer.zero_grad()
        logits = model(flow_clip, roi_clips, color_feat)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict:
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []

    for flow_clip, roi_clips, color_feat, labels in loader:
        flow_clip  = flow_clip.to(device)
        roi_clips  = roi_clips.to(device)
        color_feat = color_feat.to(device)
        labels     = labels.to(device).float()

        logits = model(flow_clip, roi_clips, color_feat)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        preds = (probs >= threshold).astype(int)
        y_prob.extend(probs.tolist())
        y_pred.extend(preds.tolist())
        y_true.extend(labels.cpu().numpy().astype(int).tolist())

    from evaluation.metrics import compute_metrics
    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["loss"] = total_loss / max(1, len(loader))
    return metrics


# ── Scheduler factory ─────────────────────────────────────────────────────────

def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    t_cfg = cfg.get("training", {})
    sched = t_cfg.get("scheduler", "cosine")
    epochs = t_cfg.get("epochs", 60)
    warmup = t_cfg.get("warmup_epochs", 5)

    if sched == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup, eta_min=1e-6
        )
    elif sched == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    return None


# ── Main training loop ────────────────────────────────────────────────────────

def train(cfg: dict, variant: str = "full") -> None:
    logger = get_logger("train", log_dir=cfg["paths"]["logs"])
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Collect clips
    processed_dir = cfg["paths"]["data_processed"]
    if not os.path.isdir(processed_dir):
        logger.error(f"Processed data not found: {processed_dir}. Run preprocessing first.")
        return
    all_clips = collect_all_clips(processed_dir)
    if len(all_clips) == 0:
        logger.error("No .npz clips found. Run dataset_builder.py first.")
        return
    logger.info(f"Total clips: {len(all_clips)}")

    t_cfg = cfg["training"]
    use_loso = t_cfg.get("loso", True)
    ckpt_dir = cfg["paths"]["checkpoints"]
    os.makedirs(ckpt_dir, exist_ok=True)

    if use_loso:
        splits = loso_splits(all_clips)
        logger.info(f"Using LOSO with {len(splits)} folds.")
    elif t_cfg.get("kfold", 0) > 1:
        k = t_cfg["kfold"]
        splits = stratified_kfold_splits(all_clips, k=k, seed=cfg.get("seed", 42))
        logger.info(f"Using Stratified {k}-Fold Cross Validation.")
    else:
        train_p, val_p = random_split(all_clips, t_cfg.get("val_split", 0.15), cfg.get("seed", 42))
        splits = [(train_p, val_p)]
        logger.info("Using single train/val split.")

    all_fold_metrics = []
    ckpt_dir = cfg["paths"]["checkpoints"]
    os.makedirs(ckpt_dir, exist_ok=True)

    for fold_i, (train_paths, val_paths) in enumerate(splits):
        if use_loso:
            subject = get_subject_from_path(val_paths[0])
            logger.info(f"\n{'='*20} Fold {fold_i + 1}/{len(splits)}  val_subject={subject} {'='*20}")
        else:
            logger.info(f"\n{'='*20} Fold {fold_i + 1}/{len(splits)} {'='*20}")

        metrics = _run_fold(
            train_paths, val_paths, cfg, variant, device,
            logger, ckpt_dir, fold=fold_i
        )
        all_fold_metrics.append(metrics)
        logger.info(f"  Fold metrics: {metrics}")

    # Average across folds
    avg = {k: np.mean([m[k] for m in all_fold_metrics]) for k in all_fold_metrics[0] if isinstance(all_fold_metrics[0][k], (int, float))}
    prefix = "loso" if use_loso else "kfold"
    logger.info(f"\n══ Final Cross-Validation Metrics ══\n{avg}")
    _save_results(avg, cfg, variant, suffix=f"{prefix}_avg")


def _run_fold(train_paths, val_paths, cfg, variant, device, logger, ckpt_dir, fold):
    t_cfg = cfg["training"]
    threshold = cfg["evaluation"]["threshold"]

    train_ds = MEClipDataset(train_paths, augment=True, clip_len=cfg["preprocessing"]["clip_len"])
    val_ds   = MEClipDataset(val_paths,   augment=False, clip_len=cfg["preprocessing"]["clip_len"])

    sampler = make_balanced_sampler(train_paths)
    train_loader = DataLoader(train_ds, batch_size=t_cfg["batch_size"],
                              sampler=sampler, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=t_cfg["batch_size"],
                              shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(cfg, variant=variant).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=t_cfg["lr"], weight_decay=t_cfg["weight_decay"]
    )
    criterion = build_loss(cfg).to(device)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    warmup_epochs = t_cfg.get("warmup_epochs", 5)
    best_path = os.path.join(ckpt_dir, f"best_fold{fold}_{variant}.pt")
    ckpt_mgr = CheckpointManager(ckpt_dir, keep_last=2, prefix=f"fold{fold}_{variant}")
    early_stop = EarlyStopping(patience=t_cfg["early_stopping_patience"],
                               mode="max", save_path=best_path)

    for epoch in range(1, t_cfg["epochs"] + 1):
        # Unfreeze backbone after warmup epochs
        if epoch == t_cfg.get("warmup_epochs", 5) and hasattr(model, "cnn"):
            model.unfreeze_cnn()
            logger.info("  Unfreezing CNN backbone...")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                     device, t_cfg["gradient_clip"])
        metrics = evaluate(model, val_loader, criterion, device, threshold)

        if scheduler and epoch > warmup_epochs:
            scheduler.step()

        logger.info(
            f"Epoch {epoch:03d} | loss={train_loss:.4f} | "
            f"F1={metrics['f1']:.4f} | Rec={metrics['recall']:.4f} | "
            f"Prec={metrics['precision']:.4f} | FPR={metrics['fpr']:.4f}"
        )

        if epoch % 5 == 0:
            ckpt_mgr.save(epoch, model, optimizer, metrics)

        if early_stop(metrics["f1"], model):
            logger.info(f"Early stopping at epoch {epoch}.")
            break

    # Load best model for final eval
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    final = evaluate(model, val_loader, criterion, device, threshold)
    return final


def _save_results(metrics: dict, cfg: dict, variant: str, suffix: str) -> None:
    out_dir = cfg["paths"]["outputs"]
    os.makedirs(out_dir, exist_ok=True)
    import json
    path = os.path.join(out_dir, f"results_{variant}_{suffix}.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Results saved to {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--variant", default="full",
                        choices=["full", "baseline", "flow_only", "flow_cnn", "flow_cnn_tcn"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, variant=args.variant)

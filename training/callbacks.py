"""
training/callbacks.py
Training callbacks: early stopping, model checkpointing, LR scheduling.
"""
from __future__ import annotations

import os
import shutil

import torch


class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.

    Args:
        patience:   Number of epochs to wait for improvement.
        min_delta:  Minimum change to qualify as improvement.
        mode:       "max" (e.g. F1) or "min" (e.g. loss).
        save_path:  If provided, saves best model weights to this path.
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 0.001,
        mode: str = "max",
        save_path: str | None = None,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.save_path = save_path
        self.best = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, metric: float, model: torch.nn.Module) -> bool:
        """
        Returns True if training should stop.
        Saves model if it is the best so far.
        """
        improved = (
            (metric > self.best + self.min_delta)
            if self.mode == "max"
            else (metric < self.best - self.min_delta)
        )
        if improved:
            self.best = metric
            self.counter = 0
            if self.save_path:
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class CheckpointManager:
    """Saves a checkpoint every N epochs and maintains last K checkpoints."""

    def __init__(
        self,
        save_dir: str,
        keep_last: int = 3,
        prefix: str = "ckpt",
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.keep_last = keep_last
        self.prefix = prefix
        self._saved: list[str] = []

    def save(self, epoch: int, model: torch.nn.Module,
             optimizer: torch.optim.Optimizer, metrics: dict) -> str:
        path = os.path.join(self.save_dir, f"{self.prefix}_epoch{epoch:03d}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
        }, path)
        self._saved.append(path)

        # Remove oldest if over limit
        while len(self._saved) > self.keep_last:
            old = self._saved.pop(0)
            if os.path.exists(old):
                os.remove(old)
        return path

    @staticmethod
    def load(path: str, model: torch.nn.Module,
             optimizer: torch.optim.Optimizer | None = None) -> dict:
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        if optimizer and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        return ckpt

"""
utils/seed.py
Reproducibility helpers — fix seeds across Python, NumPy, and PyTorch.
"""
import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

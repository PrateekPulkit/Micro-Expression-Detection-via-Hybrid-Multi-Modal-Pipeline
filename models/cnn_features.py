"""
models/cnn_features.py
═══════════════════════════════════════════════════════════════════════════════
Spatial CNN feature extractor built on a MobileNetV2 backbone.

Replaces handcrafted HOOF features with learned spatial representations.
Operates on optical flow magnitude/angle maps (2-channel input).

Design choices:
  - MobileNetV2: 3.4M params, runs at 30+ fps on CPU.
  - Flow maps normalised to [0,1] — mapped to 2→3 channels via 1×1 conv
    so pretrained weights are maximally reused.
  - Output: 128-dim per-frame feature vector.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


class FlowChannelAdapter(nn.Module):
    """Expand 2-channel flow map to 3 channels for MobileNetV2 input."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 3, kernel_size=1, bias=False)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class CNNFeatureExtractor(nn.Module):
    """
    MobileNetV2-based per-frame spatial feature extractor.

    Input:  (B, 2, H, W)  — 2-channel flow stack (mag + angle)
    Output: (B, out_dim)  — feature vector

    Args:
        out_dim:            Output feature dimensionality (default 128).
        pretrained:         Load ImageNet weights (recommended).
        freeze_until_layer: Freeze backbone layers up to this index
                            (0 = freeze none, -1 = freeze all).
    """

    def __init__(
        self,
        out_dim: int = 128,
        pretrained: bool = True,
        freeze_until_layer: int = 14,  # freeze first 14 MobileNetV2 blocks
    ) -> None:
        super().__init__()
        self.adapter = FlowChannelAdapter()

        # Load MobileNetV2 backbone
        weights = tvm.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tvm.mobilenet_v2(weights=weights)

        # Remove final classifier; keep feature extraction layers
        self.features = backbone.features  # (B, 1280, H//32, W//32)

        # Optionally freeze early layers
        if freeze_until_layer > 0:
            for i, layer in enumerate(self.features):
                if i < freeze_until_layer:
                    for p in layer.parameters():
                        p.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2, H, W)
        Returns:
            (B, out_dim)
        """
        x = self.adapter(x)                # (B, 3, H, W)
        x = self.features(x)               # (B, 1280, h, w)
        x = self.pool(x)                   # (B, 1280, 1, 1)
        return self.head(x)                # (B, out_dim)

    def unfreeze_all(self) -> None:
        """Unfreeze all backbone parameters (called after warmup epochs)."""
        for p in self.features.parameters():
            p.requires_grad = True


class ROIFeatureExtractor(nn.Module):
    """
    Extract features from multiple facial ROI patches independently,
    then fuse with learnable attention weights.

    Each ROI shares the same CNN weights (weight tying reduces overfitting).

    Input:  List of (B, 2, roi_H, roi_W) tensors, one per ROI.
    Output: (B, roi_out_dim)
    """

    # Regions for motion-based analysis
    MOTION_ROIS = ["left_eye", "right_eye", "nose", "mouth"]

    def __init__(
        self,
        n_rois: int = 4,
        roi_out_dim: int = 64,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.n_rois = n_rois

        # Shared CNN for all ROIs (parameter efficient)
        self.shared_cnn = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(64 * 4, roi_out_dim),
            nn.ReLU(),
        )

        # Learnable per-ROI attention weights
        self.roi_attn = nn.Parameter(torch.ones(n_rois) / n_rois)

    def forward(self, roi_flows: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            roi_flows: List[Tensor(B, 2, H, W)] length = n_rois
        Returns:
            (B, roi_out_dim)
        """
        feats = [self.shared_cnn(r) for r in roi_flows]  # each (B, roi_out_dim)
        feats = torch.stack(feats, dim=1)                 # (B, n_rois, roi_out_dim)
        weights = torch.softmax(self.roi_attn, dim=0)     # (n_rois,)
        fused = (feats * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)  # (B, roi_out_dim)
        return fused

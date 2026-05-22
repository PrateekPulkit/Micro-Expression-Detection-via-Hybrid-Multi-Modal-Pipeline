"""
models/micro_expr_net.py
═══════════════════════════════════════════════════════════════════════════════
MicroExprNet — Full Hybrid Pipeline

Integrates:
  1. Global flow stream  → MobileNetV2 CNN → Temporal CNN
  2. ROI flow stream     → Shared compact CNN → Attention pool
  3. Color signal stream → MLP (Chromo-Temporal)
  4. Attention-gated fusion
  5. Binary classifier head

Also includes:
  - BaselineModel: HOOF + LSTM (paper reproduction)
  - AblationModel: configurable ablation variants
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .cnn_features import CNNFeatureExtractor, ROIFeatureExtractor
from .fusion import ColorStreamMLP, build_fusion
from .temporal_model import BaselineLSTM, LightTransformer, TemporalCNN


# ── Utility ───────────────────────────────────────────────────────────────────

def _classifier_head(in_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),   # single logit → BCEWithLogitsLoss
    )


def compute_hoof_batch(flow_clip: torch.Tensor, n_bins: int = 8) -> torch.Tensor:
    """
    Compute Histogram of Oriented Optical Flow (HOOF) on a batch of clips.
    Input: flow_clip (B, T, 2, H, W)
    Output: hoof_seq (B, T, n_bins)
    """
    B, T, C, H, W = flow_clip.shape
    u = flow_clip[:, :, 0, :, :]
    v = flow_clip[:, :, 1, :, :]
    
    mag = torch.sqrt(u**2 + v**2 + 1e-6)
    angle = torch.atan2(v, u)  # [-pi, pi]
    
    # Map angle to [0, 2*pi]
    angle = angle + torch.where(angle < 0, 2 * torch.pi, 0.0)
    
    bin_width = 2 * torch.pi / n_bins
    bin_idx = (angle / bin_width).floor().long()
    bin_idx = torch.clamp(bin_idx, 0, n_bins - 1)
    
    # Flatten spatial dims to (B, T, H*W)
    bin_idx_flat = bin_idx.view(B, T, -1)
    mag_flat = mag.view(B, T, -1)
    
    # Hist: sum magnitudes per bin
    # We use scatter_add_ for batching
    hoof = torch.zeros(B, T, n_bins, device=flow_clip.device)
    hoof.scatter_add_(2, bin_idx_flat, mag_flat)
    
    # L1 Normalize
    hoof = hoof / (hoof.sum(dim=2, keepdim=True) + 1e-6)
    return hoof


# ── Main Model ────────────────────────────────────────────────────────────────

class MicroExprNet(nn.Module):
    """
    Full hybrid micro-expression detection network.

    Input tensors (all on same device):
        flow_clip:   (B, T, 2, H, W)   — per-frame 2-channel flow stacks
        roi_clips:   (B, T, n_rois, 2, roi_H, roi_W)  — per-ROI flow
        color_feat:  (B, color_dim)    — Chromo-Temporal features

    Output:
        logit: (B, 1)   — raw logit (pass through sigmoid for probability)
    """

    N_MOTION_ROIS = 4

    def __init__(
        self,
        # CNN
        cnn_out_dim: int = 128,
        pretrained_cnn: bool = True,
        # ROI
        roi_out_dim: int = 64,
        # Temporal
        temporal_type: str = "tcn",       # "tcn" | "transformer"
        tcn_channels: list[int] = None,
        tcn_kernel: int = 3,
        temporal_out_dim: int = 128,
        # Color
        color_in_dim: int = 36,
        color_out_dim: int = 32,
        # Fusion
        fusion_type: str = "attention",
        fusion_gate_dim: int = 64,
        fusion_out_dim: int = 128,
        # Classifier
        cls_hidden: int = 64,
        cls_dropout: float = 0.3,
        # Misc
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # ── Stream 1: Global flow ────────────────────────────────────────────
        self.cnn = CNNFeatureExtractor(
            out_dim=cnn_out_dim, pretrained=pretrained_cnn
        )
        if temporal_type == "tcn":
            self.temporal = TemporalCNN(
                in_channels=cnn_out_dim,
                channel_list=tcn_channels or [64, 64, 128],
                kernel_size=tcn_kernel,
                dropout=dropout,
                out_dim=temporal_out_dim,
            )
        else:
            self.temporal = LightTransformer(
                in_dim=cnn_out_dim, out_dim=temporal_out_dim, dropout=dropout
            )
        self.temporal_type = temporal_type

        # ── Stream 2: ROI flow ───────────────────────────────────────────────
        self.roi_cnn = ROIFeatureExtractor(
            n_rois=self.N_MOTION_ROIS,
            roi_out_dim=roi_out_dim,
            pretrained=False,
        )

        # ── Stream 3: Color signal ───────────────────────────────────────────
        self.color_mlp = ColorStreamMLP(
            in_dim=color_in_dim, out_dim=color_out_dim, dropout=dropout
        )

        # ── Fusion ───────────────────────────────────────────────────────────
        self.fusion = build_fusion(
            fusion_type=fusion_type,
            stream_dims=[temporal_out_dim, roi_out_dim, color_out_dim],
            gate_dim=fusion_gate_dim,
            out_dim=fusion_out_dim,
        )

        # ── Classifier ───────────────────────────────────────────────────────
        self.classifier = _classifier_head(fusion_out_dim, cls_hidden, cls_dropout)

    # ──────────────────────────────────────────────────────────────────────── #

    def forward(
        self,
        flow_clip: torch.Tensor,          # (B, T, 2, H, W)
        roi_clips: torch.Tensor,           # (B, T, n_rois, 2, roi_H, roi_W)
        color_feat: torch.Tensor,          # (B, color_dim)
    ) -> torch.Tensor:
        B, T, C, H, W = flow_clip.shape

        # ── Stream 1: Per-frame CNN → (B, T, cnn_out_dim) ───────────────────
        flow_flat = flow_clip.view(B * T, C, H, W)
        cnn_feats = self.cnn(flow_flat).view(B, T, -1)   # (B, T, cnn_out_dim)

        if self.temporal_type == "tcn":
            # TCN expects (B, channels, T)
            temporal_out = self.temporal(cnn_feats.permute(0, 2, 1))
        else:
            # Transformer expects (B, T, channels)
            temporal_out = self.temporal(cnn_feats)
        # temporal_out: (B, temporal_out_dim)

        # ── Stream 2: Per-frame ROI CNN → pool over time ──────────────────
        B, T, n_rois, rc, rH, rW = roi_clips.shape
        # Flatten (B, T) to process all frames at once
        roi_clips_flat = roi_clips.view(B * T, n_rois, rc, rH, rW)
        roi_flows_batched = [roi_clips_flat[:, i, :, :, :] for i in range(n_rois)]
        roi_feats_flat = self.roi_cnn(roi_flows_batched)      # (B*T, roi_out_dim)
        roi_seq = roi_feats_flat.view(B, T, -1)               # (B, T, roi_out_dim)
        roi_out = roi_seq.mean(dim=1)                        # (B, roi_out_dim)

        # ── Stream 3: Color signal MLP ────────────────────────────────────
        color_out = self.color_mlp(color_feat)               # (B, color_out_dim)

        # ── Fusion + classification ───────────────────────────────────────
        fused = self.fusion([temporal_out, roi_out, color_out])
        return self.classifier(fused)                        # (B, 1)

    @torch.no_grad()
    def predict_proba(
        self,
        flow_clip: torch.Tensor,
        roi_clips: torch.Tensor,
        color_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Return sigmoid probabilities."""
        self.eval()
        return torch.sigmoid(self(flow_clip, roi_clips, color_feat))

    def unfreeze_cnn(self) -> None:
        """Unfreeze backbone after warmup."""
        self.cnn.unfreeze_all()


# ── Baseline Model (HOOF + LSTM — paper reproduction) ────────────────────────

class BaselineModel(nn.Module):
    """
    Reproduces the 2019 paper's approach:
      Optical flow → HOOF (8-D) → LSTM → Binary classifier.

    HOOF features are computed during preprocessing and loaded as input.

    Input:
        hoof_seq: (B, T, 8) — HOOF feature sequence
    Output:
        logit: (B, 1)
    """

    def __init__(self, hoof_dim: int = 8, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.lstm = BaselineLSTM(hoof_dim, hidden_dim, n_layers=2,
                                 dropout=dropout, out_dim=hidden_dim)
        self.classifier = _classifier_head(hidden_dim, 32, dropout)

    def forward(
        self,
        flow_clip: torch.Tensor,
        roi_clips: Optional[torch.Tensor] = None,
        color_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # On-the-fly HOOF conversion to match 2019 baseline paper input
        hoof_seq = compute_hoof_batch(flow_clip)
        feat = self.lstm(hoof_seq)
        return self.classifier(feat)

    @torch.no_grad()
    def predict_proba(self, flow_clip: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self.forward(flow_clip))


# ── Ablation variants ─────────────────────────────────────────────────────────

class AblationModel(nn.Module):
    """
    Configurable model for ablation study.
    Disable streams by setting use_roi=False, use_color=False.

    flow_only:       use_roi=False, use_color=False, temporal="lstm"
    flow+cnn:        use_roi=False, use_color=False, temporal="tcn"
    flow+cnn+tcn:    use_roi=True,  use_color=False
    full:            use_roi=True,  use_color=True
    """

    def __init__(
        self,
        cnn_out_dim: int = 128,
        roi_out_dim: int = 64,
        color_in_dim: int = 36,
        color_out_dim: int = 32,
        temporal_out_dim: int = 128,
        use_roi: bool = True,
        use_color: bool = True,
        temporal_type: str = "tcn",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.use_roi = use_roi
        self.use_color = use_color

        self.cnn = CNNFeatureExtractor(out_dim=cnn_out_dim, pretrained=True)

        if temporal_type == "tcn":
            self.temporal = TemporalCNN(cnn_out_dim, out_dim=temporal_out_dim,
                                        dropout=dropout)
        else:
            self.temporal = BaselineLSTM(cnn_out_dim, out_dim=temporal_out_dim,
                                         dropout=dropout)
        self.temporal_type = temporal_type

        stream_dims = [temporal_out_dim]
        if use_roi:
            self.roi_cnn = ROIFeatureExtractor(n_rois=4, roi_out_dim=roi_out_dim)
            stream_dims.append(roi_out_dim)
        if use_color:
            self.color_mlp = ColorStreamMLP(color_in_dim, out_dim=color_out_dim)
            stream_dims.append(color_out_dim)

        fusion_out = 64
        self.fusion = build_fusion("concat", stream_dims, out_dim=fusion_out)
        self.classifier = _classifier_head(fusion_out, 32, dropout)

    def forward(
        self,
        flow_clip: torch.Tensor,
        roi_clips: Optional[torch.Tensor] = None,
        color_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C, H, W = flow_clip.shape
        flow_flat = flow_clip.view(B * T, C, H, W)
        cnn_feats = self.cnn(flow_flat).view(B, T, -1)

        if self.temporal_type == "tcn":
            temporal_out = self.temporal(cnn_feats.permute(0, 2, 1))
        else:
            temporal_out = self.temporal(cnn_feats)

        streams = [temporal_out]

        if self.use_roi and roi_clips is not None:
            _, _, n_rois, rc, rH, rW = roi_clips.shape
            roi_f = [
                self.roi_cnn([roi_clips[:, t, i] for i in range(n_rois)])
                for t in range(T)
            ]
            roi_out = torch.stack(roi_f, dim=1).mean(dim=1)
            streams.append(roi_out)

        if self.use_color and color_feat is not None:
            streams.append(self.color_mlp(color_feat))

        fused = self.fusion(streams)
        return self.classifier(fused)

    def unfreeze_cnn(self) -> None:
        """Unfreeze backbone after warmup."""
        self.cnn.unfreeze_all()


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(cfg: dict, variant: str = "full") -> nn.Module:
    """
    Factory function from config dict.

    variant: "full" | "baseline" | "flow_only" | "flow_cnn" | "flow_cnn_tcn"
    """
    m_cfg = cfg.get("model", {})
    dropout = m_cfg.get("temporal", {}).get("dropout", 0.2)

    if variant == "full":
        return MicroExprNet(
            cnn_out_dim=m_cfg.get("cnn_out_dim", 128),
            temporal_type=m_cfg.get("temporal", {}).get("type", "tcn"),
            tcn_channels=m_cfg.get("temporal", {}).get("n_channels", [64, 64, 128]),
            tcn_kernel=m_cfg.get("temporal", {}).get("kernel_size", 3),
            temporal_out_dim=128,
            color_in_dim=36,
            color_out_dim=m_cfg.get("color_stream", {}).get("out_dim", 16),
            fusion_type=m_cfg.get("fusion", {}).get("type", "attention"),
            fusion_gate_dim=m_cfg.get("fusion", {}).get("gate_dim", 64),
            cls_hidden=m_cfg.get("classifier", {}).get("hidden_dim", 64),
            cls_dropout=m_cfg.get("classifier", {}).get("dropout", 0.3),
            dropout=dropout,
        )
    elif variant == "baseline":
        return BaselineModel()
    elif variant == "flow_only":
        return AblationModel(use_roi=False, use_color=False, temporal_type="lstm", dropout=dropout)
    elif variant == "flow_cnn":
        return AblationModel(use_roi=False, use_color=False, temporal_type="tcn", dropout=dropout)
    elif variant == "flow_cnn_tcn":
        return AblationModel(use_roi=True, use_color=False, temporal_type="tcn", dropout=dropout)
    else:
        raise ValueError(f"Unknown model variant: {variant}")

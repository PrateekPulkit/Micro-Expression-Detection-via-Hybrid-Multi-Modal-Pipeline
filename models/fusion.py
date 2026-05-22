"""
models/fusion.py
═══════════════════════════════════════════════════════════════════════════════
Feature fusion module — adaptively combines three signal streams:
  1. Motion stream  (optical flow → CNN → Temporal CNN)
  2. ROI stream     (per-ROI flow → shared CNN → attention pool)
  3. Color stream   (Chromo-Temporal features → MLP)

Uses a gated attention mechanism to assign importance weights to streams
dynamically per sample, rather than fixed concatenation or weighted sum.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorStreamMLP(nn.Module):
    """
    Lightweight MLP for the Chromo-Temporal color signal.

    Input:  (B, color_dim)   — 36-dim feature vector from ColorSignalExtractor
    Output: (B, out_dim)
    """

    def __init__(
        self,
        in_dim: int = 36,
        hidden_dims: list[int] = None,
        out_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionFusion(nn.Module):
    """
    Attention-gated multi-stream fusion.

    Given K feature streams of dimensions [d1, d2, ..., dK],
    projects each to a common gate_dim, computes softmax attention weights,
    and returns a weighted sum projected to out_dim.

    Args:
        stream_dims: list of input dimensions for each stream.
        gate_dim:    intermediate projection size for gating network.
        out_dim:     output feature dimension.
    """

    def __init__(
        self,
        stream_dims: list[int],
        gate_dim: int = 64,
        out_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_streams = len(stream_dims)

        # Project each stream to gate_dim
        self.proj = nn.ModuleList(
            [nn.Linear(d, gate_dim) for d in stream_dims]
        )
        # Gating network: concat all projections → softmax weights
        self.gate_net = nn.Sequential(
            nn.Linear(gate_dim * self.n_streams, gate_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gate_dim, self.n_streams),
        )
        # Final projection
        self.out_proj = nn.Linear(gate_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, streams: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            streams: List[Tensor(B, di)] — one tensor per stream.
        Returns:
            (B, out_dim)
        """
        projected = [F.relu(p(s)) for p, s in zip(self.proj, streams)]  # each (B, gate_dim)
        gate_input = torch.cat(projected, dim=-1)        # (B, gate_dim * K)
        weights = torch.softmax(self.gate_net(gate_input), dim=-1)  # (B, K)

        fused = sum(w.unsqueeze(-1) * p
                    for w, p in zip(weights.unbind(dim=-1), projected))
        fused = self.dropout(fused)
        return self.out_proj(fused)           # (B, out_dim)


class ConcatFusion(nn.Module):
    """Simple concatenation baseline fusion."""

    def __init__(self, stream_dims: list[int], out_dim: int = 128) -> None:
        super().__init__()
        self.proj = nn.Linear(sum(stream_dims), out_dim)

    def forward(self, streams: list[torch.Tensor]) -> torch.Tensor:
        return self.proj(torch.cat(streams, dim=-1))


class WeightedFusion(nn.Module):
    """Fixed learnable scalar weighted sum of streams."""

    def __init__(self, stream_dims: list[int], out_dim: int = 128) -> None:
        super().__init__()
        # All must be same dim after projection
        self.projs = nn.ModuleList([nn.Linear(d, out_dim) for d in stream_dims])
        self.weights = nn.Parameter(torch.ones(len(stream_dims)))

    def forward(self, streams: list[torch.Tensor]) -> torch.Tensor:
        projected = [p(s) for p, s in zip(self.projs, streams)]
        w = torch.softmax(self.weights, dim=0)
        return sum(wi * pi for wi, pi in zip(w, projected))


def build_fusion(
    fusion_type: str,
    stream_dims: list[int],
    gate_dim: int = 64,
    out_dim: int = 128,
) -> nn.Module:
    """Factory for fusion modules."""
    fusion_type = fusion_type.lower()
    if fusion_type == "attention":
        return AttentionFusion(stream_dims, gate_dim, out_dim)
    elif fusion_type == "concat":
        return ConcatFusion(stream_dims, out_dim)
    elif fusion_type == "weighted":
        return WeightedFusion(stream_dims, out_dim)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")

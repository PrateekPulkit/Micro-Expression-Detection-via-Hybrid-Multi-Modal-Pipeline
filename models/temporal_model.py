"""
models/temporal_model.py
═══════════════════════════════════════════════════════════════════════════════
Temporal sequence modelling — replaces the RNN/LSTM from the original paper.

Two implementations provided:
  1. TemporalCNN (TCN) — dilated 1D convolutions, preferred.
  2. LightTransformer — lightweight 2-head transformer, optional.

TCN advantages over RNN/LSTM:
  - Parallelisable (no sequential bottleneck).
  - Stable gradients (no vanishing gradient over time).
  - Faster convergence on short ME sequences.
  - Lower latency in real-time inference.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── TCN Building Block ────────────────────────────────────────────────────────

class _DilatedResBlock(nn.Module):
    """
    Residual dilated 1D convolution block for TCN.

    Architecture:
        Input → Conv1d (dilation) → BN → ReLU → Dropout
              → Conv1d (dilation) → BN → (+residual) → ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation  # causal padding

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=pad, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      padding=pad, dilation=dilation),
            nn.BatchNorm1d(out_channels),
        )

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)
        self._pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)
        out = self.net(x)
        # Remove causal padding artefact (right side)
        out = out[:, :, : x.size(2)]
        return self.relu(out + residual)


class TemporalCNN(nn.Module):
    """
    Temporal CNN with progressive dilation.

    Input:  (B, in_channels, T)   — feature sequence over time
    Output: (B, out_dim)          — temporal summary vector

    Args:
        in_channels:    Input feature dimension (matched to CNN output).
        channel_list:   Hidden channel sizes per TCN block.
        kernel_size:    Conv kernel size (odd integer).
        dropout:        Dropout rate between layers.
        out_dim:        Final projection dimension.
    """

    def __init__(
        self,
        in_channels: int = 128,
        channel_list: list[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        if channel_list is None:
            channel_list = [64, 64, 128]

        blocks = []
        ch_in = in_channels
        for i, ch_out in enumerate(channel_list):
            dilation = 2 ** i           # exponentially growing dilation
            blocks.append(
                _DilatedResBlock(ch_in, ch_out, kernel_size, dilation, dropout)
            )
            ch_in = ch_out

        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(channel_list[-1], out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, T)
        Returns:
            (B, out_dim)
        """
        x = self.blocks(x)          # (B, last_ch, T)
        x = self.pool(x).squeeze(-1)  # (B, last_ch)
        return self.proj(x)           # (B, out_dim)


# ── Lightweight Transformer (optional) ───────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences up to max_len."""

    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LightTransformer(nn.Module):
    """
    Lightweight self-attention temporal model.

    Input:  (B, T, in_dim)
    Output: (B, out_dim)
    """

    def __init__(
        self,
        in_dim: int = 128,
        d_model: int = 128,
        n_heads: int = 2,
        n_layers: int = 2,
        dropout: float = 0.2,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, in_dim)  — batch-first
        Returns:
            (B, out_dim)
        """
        x = self.input_proj(x)        # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)       # (B, T, d_model)
        x = x.permute(0, 2, 1)       # (B, d_model, T)
        x = self.pool(x).squeeze(-1) # (B, d_model)
        return self.proj(x)           # (B, out_dim)


# ── Baseline LSTM (for ablation comparison) ──────────────────────────────────

class BaselineLSTM(nn.Module):
    """LSTM-based baseline matching the original paper's approach."""

    def __init__(
        self,
        in_dim: int = 8,           # HOOF feature size
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
        out_dim: int = 64,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layers,
                            batch_first=True, dropout=dropout)
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, in_dim)
        Returns:
            (B, out_dim)
        """
        _, (h_n, _) = self.lstm(x)  # h_n: (n_layers, B, hidden)
        return self.proj(h_n[-1])   # (B, out_dim)

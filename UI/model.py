import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv1d → BatchNorm1d → ReLU  (matches training notebook)."""
    def __init__(self, in_ch, out_ch, kernel=3, pad=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CNNLSTM(nn.Module):
    """
    CNN-LSTM architecture — exactly as trained in HAR_3Models_Final.ipynb.

    Input:  (B, 18, 128)  — 18-channel fused tensor (9 time + 9 FFT),
                            channels-first for Conv1d.
    Output: (B, 6)        — logits for 6 activity classes.
    """
    def __init__(self, in_ch: int = 18, n_cls: int = 6, lstm_h: int = 512):
        super().__init__()

        # ── CNN backbone: 5 ConvBlocks (64 → 128 → 256 → 256 → 512) ──────
        self.cnn = nn.Sequential(
            ConvBlock(in_ch, 64),
            ConvBlock(64, 128),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            ConvBlock(256, 512),
            nn.AdaptiveAvgPool1d(32),      # → (B, 512, 32)
        )

        # ── 2-layer LSTM ──────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_h,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )

        # ── Classifier ────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_h),
            nn.Linear(lstm_h, 256), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(256, 128),   nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, n_cls),
        )

    def forward(self, x):
        # x: (B, 18, 128)  — channels-first
        x = self.cnn(x)                    # (B, 512, 32)
        x = x.permute(0, 2, 1)            # (B, 32, 512)
        _, (h_n, _) = self.lstm(x)        # h_n: (2, B, 512)
        return self.classifier(h_n[-1])   # (B, 6)

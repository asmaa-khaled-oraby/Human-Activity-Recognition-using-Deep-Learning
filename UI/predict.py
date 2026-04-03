import torch
import numpy as np
import joblib

from model import CNNLSTM
from utils import preprocess_window

# ── Labels (must match training order) ───────────────────────────────────
LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]

# ── Load model ────────────────────────────────────────────────────────────
model = CNNLSTM(in_ch=18, n_cls=6, lstm_h=512)
model.load_state_dict(torch.load("best_CNN-LSTM.pt", map_location="cpu"))
model.eval()

# ── Load scaler params ────────────────────────────────────────────────────
# The scaler was fit on the 18-channel fused data.
# We extract per-channel stats for time and FFT separately.
_sc = joblib.load("scaler.pkl")

scaler_params = {
    "ch_mean":  _sc.mean_[:9].astype(np.float32),    # first 9  → time-domain
    "ch_std":   _sc.scale_[:9].astype(np.float32),
    "fft_mean": _sc.mean_[9:].astype(np.float32),    # last 9   → FFT
    "fft_std":  _sc.scale_[9:].astype(np.float32),
}


def predict(data: np.ndarray) -> tuple[str, float]:
    """
    Classify a single sensor window.

    Parameters
    ----------
    data : np.ndarray, shape (N, 9)
        Raw un-normalised sensor data. N should be 128; shorter windows
        are zero-padded, longer ones are truncated.

    Returns
    -------
    label : str   — predicted activity name
    conf  : float — softmax confidence (0–1)
    """
    # 1. Preprocess: validate → z-score → FFT → fuse → (128, 18)
    fused = preprocess_window(data, scaler_params)    # (128, 18)

    # 2. To tensor: (128, 18) → (1, 18, 128)  channels-first for Conv1d
    x = torch.tensor(fused.T, dtype=torch.float32).unsqueeze(0)

    # 3. Inference
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0]
        pred   = probs.argmax().item()
        conf   = probs.max().item()

    return LABELS[pred], conf


def predict_top_k(data: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
    """Return top-k (label, confidence) pairs."""
    fused = preprocess_window(data, scaler_params)
    x     = torch.tensor(fused.T, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    top   = probs.topk(k)
    return [(LABELS[i.item()], round(v.item(), 4)) for i, v in zip(top.indices, top.values)]

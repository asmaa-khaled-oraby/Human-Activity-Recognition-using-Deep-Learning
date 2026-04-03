import numpy as np

# ── Constants (must match training notebook exactly) ──────────────────────
WIN   = 128
N_CH  = 9


def fix_length(data: np.ndarray, target: int = WIN) -> np.ndarray:
    """Pad or truncate to exactly `target` rows."""
    if len(data) > target:
        return data[:target]
    elif len(data) < target:
        pad = np.zeros((target - len(data), data.shape[1]), dtype=np.float32)
        return np.vstack([data, pad])
    return data


def validate_input(data: np.ndarray) -> np.ndarray:
    """Check shape and fix length. Returns (128, 9) float32 array."""
    if data.ndim != 2 or data.shape[1] != N_CH:
        raise ValueError(
            f"Input must be shape (N, 9) — got {data.shape}. "
            "Expected 9 sensor channels: body_acc_xyz, body_gyro_xyz, total_acc_xyz."
        )
    data = fix_length(data.astype(np.float32))
    return data


def z_score(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Per-channel z-score normalisation."""
    return ((data - mean) / std).astype(np.float32)


def fft_features(raw: np.ndarray, fft_mean: np.ndarray, fft_std: np.ndarray) -> np.ndarray:
    """
    Compute FFT magnitude features and normalise.
    raw : (128, 9)  → returns (64, 9) float32
    """
    mag = np.abs(np.fft.rfft(raw, axis=0))[1:WIN // 2 + 1, :]   # (64, 9)
    return ((mag - fft_mean) / fft_std).astype(np.float32)


def fuse(x_t: np.ndarray, x_f: np.ndarray) -> np.ndarray:
    """
    Fuse time-domain (128,9) + FFT (64,9) into (128,18).
    FFT is zero-padded from 64 → 128 before concat.
    """
    pad = np.zeros((WIN - x_f.shape[0], N_CH), dtype=np.float32)
    x_f_padded = np.concatenate([x_f, pad], axis=0)           # (128, 9)
    return np.concatenate([x_t, x_f_padded], axis=1)          # (128, 18)


def preprocess_window(raw: np.ndarray, scaler_params: dict) -> np.ndarray:
    """
    Full preprocessing pipeline matching the training notebook:
      raw (128,9)  →  validate  →  z-score  →  FFT  →  fuse  →  (128,18)

    scaler_params: dict with keys
        'ch_mean'  (9,)   — per-channel time-domain mean
        'ch_std'   (9,)   — per-channel time-domain std
        'fft_mean' (9,)   — per-channel FFT mean
        'fft_std'  (9,)   — per-channel FFT std
    """
    raw    = validate_input(raw)                                        # (128,9)
    x_t    = z_score(raw, scaler_params['ch_mean'], scaler_params['ch_std'])
    x_f    = fft_features(raw, scaler_params['fft_mean'], scaler_params['fft_std'])
    fused  = fuse(x_t, x_f)                                            # (128,18)
    return fused

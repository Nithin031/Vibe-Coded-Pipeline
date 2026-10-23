"""
Shared per-window motion features for 80 MHz Nexmon CSI (242 usable tones).

Input: amplitude matrix A (T packets x 242 tones) for ONE receive chain / receiver.
Every packet is first divided by its own median level, which removes automatic gain
changes, so the features describe how the channel *shape* moves, not its loudness.
"""

import numpy as np

# Natural FFT order, VHT80: DC (0, 1, 255) and guard band (123-133) carry no channel data
DROP_BINS = np.r_[0, 1, 123:134, 255]
KEEP_BINS = np.setdiff1d(np.arange(256), DROP_BINS)          # 242 tones

FEATURE_NAMES = [
    "cv_median",        # median over tones of (std over time / mean over time)
    "cv_p90",           # 90th percentile of the same
    "diff_energy",      # mean |A(t) - A(t-1)| relative to mean level
    "pc1_frac",         # share of variance in the first principal component
    "n_pc_90",          # number of components needed for 90 % of the variance
    "spec_flat_std",    # std over tones of the time-averaged normalised profile
    "low_freq_frac",    # share of temporal power below 2 Hz (slow body motion)
]


def window_features(A, rate_hz):
    """A: (T, 242) float amplitude. Returns 7 features (see FEATURE_NAMES)."""
    A = A / (np.median(A, axis=1, keepdims=True) + 1e-6)       # per-packet gain normalisation
    mu = A.mean(0)
    sd = A.std(0)
    cv = sd / (mu + 1e-6)

    d = np.abs(np.diff(A, axis=0)).mean() / (mu.mean() + 1e-6)

    X = A - mu
    s = np.linalg.svd(X, compute_uv=False) ** 2
    frac = s / (s.sum() + 1e-12)
    n90 = int(np.searchsorted(np.cumsum(frac), 0.90) + 1)

    # temporal spectrum of the tone-averaged deviation
    x = X.mean(1)
    P = np.abs(np.fft.rfft(x - x.mean())) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / rate_hz)
    low = P[(f > 0) & (f < 2.0)].sum() / (P[f > 0].sum() + 1e-12)

    return np.array([np.median(cv), np.percentile(cv, 90), d, frac[0], n90, mu.std(), low],
                    dtype=np.float32)

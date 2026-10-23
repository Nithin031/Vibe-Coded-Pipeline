"""Data: read ESP32 CSI recordings, preprocess them causally, and compute the 10 window cues.

    raw CSI amplitude (52 tones)
      -> per-packet gain normalisation (divide by the median over tones)
      -> 50 Hz bins (packets averaged per 20 ms), gaps up to 60 ms interpolated
      -> causal baseline per tone (EMA, 30 s) -> relative fluctuation z = x / baseline - 1
      -> 5 s windows every 2.5 s (first 10 s skipped while the baseline settles)
      -> 10 count features (cues)

Same code as V5/esp32_counter/esp32_counter.py (the training reference); deploy/tests checks they agree.
Nothing here looks at future packets.
"""

import json

import numpy as np
import pandas as pd
from scipy.signal import lfilter


# ===============================================================
# Settings (identical on the ESP32 - counter_model.h has the same values)
# ===============================================================
FS = 50                  # sample rate of the shared grid (Hz)
WIN_S = 5.0              # window length (s)
HOP_S = 2.5              # window hop (s)
WIN = int(WIN_S * FS)    # 250 bins
HOP = int(HOP_S * FS)    # 125 bins
N_TONES = 52

MAX_GAP_BINS = 3         # gaps up to 3 bins (60 ms) are interpolated
MAX_MISSING = 0.20       # windows with more missing bins than this are skipped

TAU_S = 30.0             # time constant of the causal per-tone baseline
ALPHA = 1.0 / (TAU_S * FS)
WARMUP_S = 10.0          # no output for the first 10 s (baseline settling)
WARMUP_BINS = int(WARMUP_S * FS)

CAL_S = 20.0             # empty-room calibration after the warm-up
N_CAL_WINDOWS = int((CAL_S - WIN_S) / HOP_S) + 1     # 7 windows

SMOOTH_S = 30.0          # output = median of the last 30 s of window estimates
SMOOTH_N = int(SMOOTH_S / HOP_S)                    # 12 windows

EPS = 1e-6

FEATURE_NAMES = [
    "cv_median",       # median over tones of the fluctuation std
    "cv_p90",          # 90th percentile of the same
    "cv_p10",          # 10th percentile of the same
    "diff_energy",     # mean |x(t) - x(t-1)|  (motion energy)
    "pc1_frac",        # share of variance in the first principal component
    "n_pc_90",         # components needed for 90 % of the variance
    "tone_corr",       # mean correlation between tones
    "low_freq_frac",   # temporal power below 2 Hz (slow body motion)
    "high_freq_frac",  # temporal power above 5 Hz
    "mean_std",        # std of the tone-averaged series
]

# magnitude features go through log() so the empty-room offset acts as a ratio
LOG_FEATURES = ["cv_median", "cv_p90", "cv_p10", "diff_energy", "mean_std"]
LOG_IDX = [FEATURE_NAMES.index(name) for name in LOG_FEATURES]


def settings_dict():
    return {"fs_hz": FS, "win_s": WIN_S, "hop_s": HOP_S, "n_tones": N_TONES,
            "max_gap_bins": MAX_GAP_BINS, "max_missing": MAX_MISSING,
            "tau_s": TAU_S, "warmup_s": WARMUP_S, "cal_s": CAL_S,
            "n_cal_windows": N_CAL_WINDOWS, "smooth_s": SMOOTH_S, "smooth_n": SMOOTH_N,
            "feature_names": FEATURE_NAMES, "log_features": LOG_FEATURES}


# ===============================================================
# Reading ESP32 recordings
# ===============================================================
WRAP = 2 ** 32           # ESP32 local_timestamp: 32-bit microsecond counter


def to_52_tones(vals):
    """Raw CSI numbers (I/Q pairs) -> the 52 used tones of a 20 MHz channel as 104 numbers, or None.

    104 values: already the 52 tones (WiCount, recording_kit).
    128+ values: 64 subcarriers in esp-csi order 0..31, -32..-1 (L-LTF first) -> keep positions 1-26 and 38-63.
    """
    if len(vals) == 2 * N_TONES:
        return vals
    if len(vals) >= 128:
        keep = list(range(1, 27)) + list(range(38, 64))
        return [v for k in keep for v in vals[2 * k: 2 * k + 2]]
    return None


def read_esp_csv(path):
    """ESP32 CSI CSV (esp-csi csi_recv format, columns local_timestamp + data) -> (t seconds, amplitude (packets, 52)).

    Works for WiCount files and the files written by V5/recording_kit/record_csi.py.
    Times start at 0 and are unwrapped (the 32-bit counter wraps every ~71 min)."""
    df = pd.read_csv(path)
    ts, rows = [], []
    for stamp, txt in zip(df["local_timestamp"], df["data"]):
        try:
            vals = to_52_tones(json.loads(txt))
        except (TypeError, ValueError):
            continue
        if vals is not None:
            ts.append(int(stamp))
            rows.append(vals)
    if not rows:
        raise ValueError(f"{path}: no usable CSI_DATA rows (need 104 or >= 128 values per row)")

    iq = np.array(rows, dtype=np.float32)
    amp = np.hypot(iq[:, 0::2], iq[:, 1::2])             # I/Q order does not matter for the amplitude

    dt = np.diff(np.array(ts, dtype=np.int64))
    dt[dt < 0] += WRAP                                   # counter wrap-around
    t = np.concatenate([[0], np.cumsum(dt)]) * 1e-6
    order = np.argsort(t, kind="stable")
    return t[order], amp[order]


# ===============================================================
# Batch preprocessing (training / recorded files)
# ===============================================================
def gain_normalise(A):
    """Per packet: divide by the median over tones (removes AGC jumps)."""
    return A / (np.median(A, axis=1, keepdims=True) + EPS)


def resample_to_grid(t, A):
    """t: packet times (s, sorted), A: (packets, tones).
    Returns X (bins, tones) with NaN rows for empty bins, and the valid mask."""
    idx = np.floor((t - t[0]) * FS).astype(np.int64)
    n_bins = int(idx[-1]) + 1

    uniq, first = np.unique(idx, return_index=True)        # idx is sorted
    sums = np.add.reduceat(A.astype(np.float64), first, axis=0)
    counts = np.diff(np.append(first, len(idx)))

    X = np.full((n_bins, A.shape[1]), np.nan, dtype=np.float32)
    X[uniq] = sums / counts[:, None]
    valid = np.zeros(n_bins, dtype=bool)
    valid[uniq] = True
    return X, valid


def fill_short_gaps(X, valid):
    """Linear interpolation over gaps of at most MAX_GAP_BINS; longer gaps stay NaN."""
    X, valid = X.copy(), valid.copy()
    missing = np.flatnonzero(~valid)
    if len(missing) == 0:
        return X, valid

    runs = np.split(missing, np.flatnonzero(np.diff(missing) > 1) + 1)
    for run in runs:
        left, right = run[0] - 1, run[-1] + 1
        if len(run) <= MAX_GAP_BINS and left >= 0 and right < len(X):
            w = ((run - left) / (right - left))[:, None]
            X[run] = X[left] * (1 - w) + X[right] * w
            valid[run] = True
    return X, valid


def causal_normalise(X, valid):
    """Each tone relative to its own past (EMA baseline): z = x / baseline - 1.

    The baseline only moves on valid bins and starts at the first valid value."""
    Z = np.full(X.shape, np.nan, dtype=np.float32)
    rows = np.flatnonzero(valid)
    if len(rows) == 0:
        return Z

    xs = X[rows].astype(np.float64)
    zi = (1.0 - ALPHA) * xs[:1]                             # first output = first value
    baseline, _ = lfilter([ALPHA], [1.0, -(1.0 - ALPHA)], xs, axis=0, zi=zi)

    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(baseline > EPS, xs / baseline - 1.0, 0.0)
    Z[rows] = z
    return Z


def window_starts(valid):
    """Window start bins: every HOP, after the warm-up, with enough valid bins."""
    n = len(valid)
    return [a for a in range(0, n - WIN + 1, HOP)
            if a >= WARMUP_BINS and valid[a:a + WIN].mean() >= 1 - MAX_MISSING]


# ===============================================================
# Count features (cues)
# ===============================================================
def window_features(W):
    """W: (bins, 52) normalised window. Missing bins count as 'no change' (0)."""
    W = np.nan_to_num(W.astype(np.float32))
    sd = W.std(axis=0)
    diff_energy = np.abs(np.diff(W, axis=0)).mean()

    Xc = W - W.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False) ** 2
    frac = s / (s.sum() + 1e-12)
    n90 = np.searchsorted(np.cumsum(frac), 0.90) + 1

    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(Xc.T)
    tone_corr = np.nanmean(C[np.triu_indices_from(C, k=1)])

    x = Xc.mean(axis=1)
    P = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / FS)
    p_total = P[f > 0].sum() + 1e-12
    low = P[(f > 0) & (f < 2.0)].sum() / p_total
    high = P[f > 5.0].sum() / p_total

    return np.array([np.median(sd), np.percentile(sd, 90), np.percentile(sd, 10),
                     diff_energy, frac[0], n90, tone_corr, low, high, x.std()],
                    dtype=np.float32)


def features_from_series(X, valid):
    """X: (bins, 52) amplitude on the 50 Hz grid (any per-tone scale).
    Returns window start bins and features (windows, 10)."""
    Z = causal_normalise(X, valid)
    starts = window_starts(valid)
    F = np.zeros((len(starts), len(FEATURE_NAMES)), dtype=np.float32)
    for j, a in enumerate(starts):
        F[j] = window_features(Z[a:a + WIN])
    return np.array(starts, dtype=np.int32), F


def features_from_raw(t, A):
    """Raw packets (times s, amplitude (packets, 52)) -> window starts + features."""
    order = np.argsort(t, kind="stable")
    t, A = np.asarray(t, dtype=np.float64)[order], np.asarray(A)[order]
    X, valid = resample_to_grid(t, gain_normalise(A.astype(np.float32)))
    X, valid = fill_short_gaps(X, valid)
    return features_from_series(X, valid)

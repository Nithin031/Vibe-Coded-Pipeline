"""Hardware-agnostic window features for people counting.

Every window [W, L, S] (time x links x subcarriers) is first turned into a
*relative fluctuation*: each subcarrier is divided by its own median over the
window. That removes absolute gain, the AGC level and the static room
response, so what is left is how much and how fast the channel moves.

The features never depend on how many links or subcarriers a NIC has:

    motion      temporal std / robust std / first-difference energy / range,
                summarised by quantiles over subcarriers
    pca         eigen-spectrum of the window (more independent movers ->
                flatter spectrum)
    spectrum    Doppler-like band powers of the leading components, spectral
                centroid and entropy, autocorrelation at fixed lags
    corr        adjacent-subcarrier, overall and inter-link correlation
    periodicity (V2) spectral peak share, flatness, autocorrelation peak:
                separates periodic machine / fan motion from human motion
    rssi        (V2) RSSI dynamics, computed separately by rssi_features()
"""
from __future__ import annotations

import numpy as np


# Doppler-like bands (Hz). Bands above Nyquist simply come out as zero.
BANDS = [(0.0, 0.3), (0.3, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, np.inf)]
ACF_LAGS_S = [0.2, 0.5, 1.0]
CLIP_MAD = 6.0


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def relative_fluctuation(win: np.ndarray) -> np.ndarray:
    """[W, L, S] amplitude -> [W, L, S] relative fluctuation around 0.

    Each subcarrier is divided by its median over the window, then values
    beyond CLIP_MAD robust deviations are clipped to suppress impulses.
    """
    med = np.median(win, axis=0, keepdims=True)
    rel = win / np.maximum(med, 1e-6) - 1.0

    mad = np.median(np.abs(rel), axis=0, keepdims=True) * 1.4826 + 1e-6
    return np.clip(rel, -CLIP_MAD * mad, CLIP_MAD * mad).astype(np.float32)


# ---------------------------------------------------------------------------
# Feature groups (each takes one link's relative window [W, S])
# ---------------------------------------------------------------------------

def _motion(r: np.ndarray) -> dict:
    std = r.std(axis=0)
    rstd = np.median(np.abs(r - np.median(r, axis=0)), axis=0) * 1.4826
    d = np.diff(r, axis=0)
    diff_energy = (d ** 2).mean(axis=0)
    rng = np.percentile(r, 95, axis=0) - np.percentile(r, 5, axis=0)

    return {
        "std_q10": np.percentile(std, 10),
        "std_q50": np.percentile(std, 50),
        "std_q90": np.percentile(std, 90),
        "rstd_q50": np.median(rstd),
        "rstd_q90": np.percentile(rstd, 90),
        "diff_q50": np.median(diff_energy),
        "diff_q90": np.percentile(diff_energy, 90),
        "range_q50": np.median(rng),
        "range_q90": np.percentile(rng, 90),
        # how unevenly motion is spread over subcarriers
        "std_cv": std.std() / (std.mean() + 1e-9),
    }


def _pca(r: np.ndarray):
    """Eigen-spectrum features + the top-3 component time series."""
    x = r - r.mean(axis=0)
    u, sv, _ = np.linalg.svd(x, full_matrices=False)
    lam = sv ** 2
    p = lam / (lam.sum() + 1e-12)

    ent = -(p[p > 0] * np.log(p[p > 0])).sum()
    feats = {
        "pca_top1": p[0],
        "pca_top3": p[:3].sum(),
        "pca_eff_rank": np.exp(ent),
        "pca_participation": 1.0 / ((p ** 2).sum() + 1e-12),
    }
    comps = u[:, :3] * sv[:3]                                 # [W, 3]
    return feats, comps


def _spectrum(comps: np.ndarray, fs: float) -> dict:
    """Band powers, centroid, entropy and ACF of the leading components."""
    w = len(comps)
    c = comps - comps.mean(axis=0)
    spec = np.abs(np.fft.rfft(c * np.hanning(w)[:, None], axis=0)) ** 2
    psd = spec.sum(axis=1)[1:]                                # drop DC
    f = np.fft.rfftfreq(w, 1.0 / fs)[1:]
    total = psd.sum() + 1e-12

    feats = {}
    for lo, hi in BANDS:
        name = f"band_{lo:g}_{hi:g}" if np.isfinite(hi) else f"band_{lo:g}_up"
        feats[name] = psd[(f >= lo) & (f < hi)].sum() / total

    pn = psd / total
    feats["spec_centroid"] = (f * pn).sum()
    feats["spec_entropy"] = -(pn[pn > 0] * np.log(pn[pn > 0])).sum() / np.log(len(pn))

    # autocorrelation of the first component at fixed physical lags
    pc1 = c[:, 0]
    var = (pc1 ** 2).sum() + 1e-12
    for lag_s in ACF_LAGS_S:
        k = int(round(lag_s * fs))
        feats[f"acf_{lag_s:g}s"] = (pc1[:-k] * pc1[k:]).sum() / var if 0 < k < w else np.nan
    return feats


def _periodicity(comps: np.ndarray, fs: float) -> dict:
    """How 'machine-like' the motion is.

    Fans, AC louvres and machines move the channel *periodically*: one sharp
    spectral line and a strong repeating autocorrelation. People move
    irregularly: a broad spectrum and a fast-decaying autocorrelation.
    (Idea from OpenCSI's spectral-entropy gate and CSI-Bench's non-human class.)
    """
    pc1 = comps[:, 0] - comps[:, 0].mean()
    w = len(pc1)

    spec = np.abs(np.fft.rfft(pc1 * np.hanning(w))) ** 2
    spec = spec[1:]                                            # drop DC
    total = spec.sum() + 1e-12

    # share of power in the strongest line (+/- one bin)
    k = int(spec.argmax())
    peak = spec[max(k - 1, 0): k + 2].sum() / total

    # spectral flatness: geometric / arithmetic mean (1 = noise-like, 0 = one tone)
    flat = np.exp(np.mean(np.log(spec + 1e-12))) / (spec.mean() + 1e-12)

    # strongest autocorrelation peak at lags 0.15-1.5 s (repeating motion)
    ac = np.correlate(pc1, pc1, mode="full")[w - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo, hi = max(int(0.15 * fs), 1), min(int(1.5 * fs), w - 1)
    ac_peak = float(ac[lo:hi].max()) if hi > lo else np.nan

    return {
        "period_peak_share": peak,
        "period_flatness": flat,
        "period_acf_peak": ac_peak,
        "period_peak_freq": (k + 1) * fs / w,
    }


def _corr(r: np.ndarray) -> dict:
    z = (r - r.mean(axis=0)) / (r.std(axis=0) + 1e-9)
    adj = (z[:, :-1] * z[:, 1:]).mean(axis=0)
    cm = (z.T @ z) / len(z)
    off = np.abs(cm[~np.eye(len(cm), dtype=bool)])
    return {
        "corr_adjacent": adj.mean(),
        "corr_abs_mean": off.mean(),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def link_features(r: np.ndarray, fs: float) -> dict:
    """All per-link features for one relative window [W, S]."""
    feats = _motion(r)
    pca_feats, comps = _pca(r)
    feats.update(pca_feats)
    feats.update(_spectrum(comps, fs))
    feats.update(_periodicity(comps, fs))
    feats.update(_corr(r))
    return feats


def rssi_features(rssi: np.ndarray) -> dict:
    """RSSI dynamics in the window (the OTH thesis found RSSI nearly as
    informative as CSI on one link). Absolute RSSI is NOT used: it depends on
    distance and transmit power, not on people."""
    d = np.diff(rssi)
    return {
        "rssi_std": float(rssi.std()),
        "rssi_range": float(np.percentile(rssi, 95) - np.percentile(rssi, 5)),
        "rssi_diff_abs": float(np.abs(d).mean()) if len(d) else 0.0,
    }


def window_features(win: np.ndarray, fs: float) -> dict:
    """Features for one amplitude window [W, L, S].

    Per-link features are averaged over links, so a 1-link ESP32 and a
    6-link Intel 5300 produce the same feature vector. Inter-link correlation
    is added when there is more than one link (NaN otherwise; LightGBM
    handles missing values natively).
    """
    rel = relative_fluctuation(win)
    per_link = [link_features(rel[:, l], fs) for l in range(rel.shape[1])]
    feats = {k: float(np.nanmean([p[k] for p in per_link])) for k in per_link[0]}

    if rel.shape[1] > 1:
        mean_sig = rel.mean(axis=2)                           # [W, L]
        cm = np.corrcoef(mean_sig.T)
        feats["corr_links"] = float(np.nanmean(np.abs(cm[~np.eye(len(cm), dtype=bool)])))
    else:
        feats["corr_links"] = np.nan
    return feats


def cnn_input(win: np.ndarray, n_bins: int = 32) -> np.ndarray:
    """[W, L, S] amplitude -> [L, n_bins, W] relative fluctuation for the CNN.

    Subcarriers are linearly resampled to a fixed n_bins so every NIC gives
    the same channel count. The fluctuation is already relative (unit-free),
    so its level is kept: how much the channel moves is part of the signal.
    """
    rel = relative_fluctuation(win)                           # [W, L, S]
    s = rel.shape[2]

    # linear interpolation as a fixed [S, n_bins] matrix
    pos = np.linspace(0, s - 1, n_bins)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, s - 1)
    frac = pos - lo
    m = np.zeros((s, n_bins), dtype=np.float32)
    m[lo, np.arange(n_bins)] += 1 - frac
    m[hi, np.arange(n_bins)] += frac

    out = rel @ m                                             # [W, L, n_bins]
    return np.ascontiguousarray(out.transpose(1, 2, 0))

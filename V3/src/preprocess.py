"""Turn raw, irregular, hardware-specific CSI into clean uniform windows.

Real CSI is messy: packets arrive at irregular intervals, some are lost,
guard / pilot / DC subcarriers carry junk, and the AGC rescales everything
whenever it feels like it. The steps here deal with that in a way that does
not depend on the NIC:

    1. valid_subcarriers  drop dead and spiky subcarriers (data-driven)
    2. to_uniform_grid    bin-average onto a fixed sample rate (anti-alias),
                          interpolate the gaps, remember where they were
    3. windows            fixed-length overlapping windows + a "missing" ratio
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 1. Subcarrier selection
# ---------------------------------------------------------------------------

def valid_subcarriers(amp: np.ndarray, low: float = 0.15, high: float = 4.0) -> np.ndarray:
    """Boolean mask [S] of subcarriers that look like real data on every link.

    A subcarrier is dropped if its median amplitude is far below (guard / null
    tones) or far above (pilot / leakage artefacts, e.g. Nexmon tones 60-61)
    the typical subcarrier of the same link.
    """
    med = np.median(amp, axis=0)                     # [L, S]
    typical = np.median(med, axis=1, keepdims=True)  # [L, 1]
    ok = (med > low * typical) & (med < high * typical)
    return ok.all(axis=0)


# ---------------------------------------------------------------------------
# 2. Uniform time grid
# ---------------------------------------------------------------------------

def to_uniform_grid(amp: np.ndarray, t: np.ndarray, fs: float):
    """Resample [T, L, S] onto a uniform grid at `fs` Hz.

    Packets falling into the same 1/fs bin are averaged (a box anti-alias
    filter when the native rate is higher than fs). Empty bins (packet loss,
    jitter) are filled by linear interpolation and flagged in `missing`.
    """
    n = int(np.floor(t[-1] * fs)) + 1
    idx = np.minimum((t * fs).astype(np.int64), n - 1)

    flat = amp.reshape(len(t), -1).astype(np.float64)
    sums = np.zeros((n, flat.shape[1]))
    np.add.at(sums, idx, flat)
    hits = np.bincount(idx, minlength=n).astype(np.float64)

    missing = hits == 0
    out = np.empty_like(sums)
    out[~missing] = sums[~missing] / hits[~missing, None]

    if missing.any():
        grid = np.arange(n)
        good = np.flatnonzero(~missing)
        for j in range(flat.shape[1]):
            out[missing, j] = np.interp(grid[missing], good, out[good, j])

    return out.reshape(n, *amp.shape[1:]).astype(np.float32), missing


# ---------------------------------------------------------------------------
# 3. Windows
# ---------------------------------------------------------------------------

def windows(x: np.ndarray, missing: np.ndarray, fs: float, win_s: float, hop_s: float):
    """Yield (start_index, window [W, L, S], missing_ratio)."""
    w = int(round(win_s * fs))
    h = int(round(hop_s * fs))
    for s in range(0, len(x) - w + 1, h):
        yield s, x[s: s + w], float(missing[s: s + w].mean())

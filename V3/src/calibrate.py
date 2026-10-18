"""Empty-room calibration (the OpenCSI / OTH-thesis idea).

At installation the room is left empty for a short time (here 20 s). The
motion features of every later window are expressed *relative to that quiet
reference*, as a log-ratio:

    f_cal = log( (f + eps) / (median_ref(f) + eps) )

A value of 0 means "as quiet as the empty room". The background level of the
room (fans, fridges, crowded WiFi) is divided out, which is what makes a model
transferable between rooms.

Only energy-like features are calibrated. Shape features (PCA spectrum,
periodicity, correlations) are already relative and are kept as they are.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


CAL_PREFIXES = ("std_", "rstd_", "diff_", "range_", "rssi_std", "rssi_range", "rssi_diff")
REF_SECONDS = 20.0

# Floor of the reference level. RSSI is reported in whole dB, so in a quiet
# room its std is exactly 0 and a plain ratio would explode: floor it at 0.5 dB.
EPS_CSI = 1e-3
EPS_RSSI = 0.5


def energy_cols(cols) -> list[str]:
    return [c for c in cols if c.startswith(CAL_PREFIXES)]


def _eps(col: str) -> float:
    return EPS_RSSI if col.startswith("rssi") else EPS_CSI


def reference_mask(df: pd.DataFrame, site_col: str = "site") -> pd.Series:
    """The first REF_SECONDS of the empty recording(s) of each site."""
    return (df["count"] == 0) & (df["t0"] + 3.0 <= REF_SECONDS)


def calibrate(df: pd.DataFrame, cols, site_col: str = "site") -> pd.DataFrame:
    """Return a copy where energy features become log-ratios to the site's
    empty-room reference. Sites without a reference get NaN (not usable)."""
    out = df.copy()
    ecols = energy_cols(cols)
    ref = reference_mask(df, site_col)

    for site, idx in df.groupby(site_col).groups.items():
        r = ref.loc[idx]
        if not r.any():
            out.loc[idx, ecols] = np.nan
            continue
        med = df.loc[r[r].index, ecols].median()
        eps = pd.Series({c: _eps(c) for c in ecols})
        out.loc[idx, ecols] = np.log((df.loc[idx, ecols] + eps) / (med + eps))
    return out

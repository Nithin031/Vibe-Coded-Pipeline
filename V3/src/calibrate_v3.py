"""V3 empty-room calibration for features AND sequences (OpenCSI idea).

A *site* is one installation: dataset + room + device/chip + channel. At install
time the room is empty for a short while; its first REF_SECONDS of empty windows
become the site's reference and are NEVER scored.

    features   energy-like features -> log( f / median_ref(f) )      (0 = as quiet as empty)
    sequences  each subcarrier bin divided by the reference's robust level of that bin,
               times a global constant so values stay at the pre-training scale

Sites without empty data cannot be calibrated and are dropped from '_cal' runs
(the share dropped is reported).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from calibrate import EPS_CSI, EPS_RSSI, energy_cols

REF_SECONDS = 30.0
SEQ_SCALE = 0.03          # typical empty-room level: keeps calibrated inputs at the raw scale


def site_key(df: pd.DataFrame) -> pd.Series:
    return (df["dataset"] + "|" + df["room"].astype(str) + "|" + df["chip"].astype(str) + "|"
            + df["channel"].astype(str))


def reference_mask(df: pd.DataFrame) -> pd.Series:
    """First REF_SECONDS of empty windows of every site (ordered by recording, then time)."""
    empty = df[df["presence"] == 0].sort_values(["recording", "t0"])
    first_rec = empty.groupby("site_cal")["recording"].transform("first")
    ref_idx = empty.index[(empty["recording"] == first_rec) & (empty["t0"] + 3.0 <= REF_SECONDS)]
    return pd.Series(df.index.isin(ref_idx), index=df.index)


def calibrate_all(df: pd.DataFrame, seq: np.ndarray, cols):
    """Return (df_cal, seq_cal, ref_mask, has_ref). seq_cal is float32 [N, 32, T]."""
    df = df.copy()
    df["site_cal"] = site_key(df)
    ref = reference_mask(df)

    ecols = energy_cols(cols)
    eps = pd.Series({c: EPS_RSSI if c.startswith("rssi") else EPS_CSI for c in ecols})
    seq_cal = np.full(seq.shape, np.nan, dtype=np.float32)
    has_ref = pd.Series(False, index=df.index)

    for site, idx in df.groupby("site_cal").groups.items():
        r = idx[ref.loc[idx].to_numpy()]
        if len(r) == 0:
            continue
        has_ref.loc[idx] = True

        med = df.loc[r, ecols].median()
        df.loc[idx, ecols] = np.log((df.loc[idx, ecols] + eps) / (med + eps))

        # per-bin robust fluctuation level of the empty reference
        level = np.median(np.abs(seq[r].astype(np.float32)), axis=(0, 2))          # [32]
        level = np.maximum(level, 1e-4)
        seq_cal[idx] = seq[idx].astype(np.float32) / level[None, :, None] * SEQ_SCALE

    return df, seq_cal, ref, has_ref

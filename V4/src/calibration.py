"""Empty-room calibration (V3 method, V4 bookkeeping).

A site is one installation (see configs/v4_protocol.json). Its reference is the first
30 s of VALID windows of the first empty recording of that site INSIDE THE SAME SPLIT,
so a test site is only ever calibrated with test-side empty data, as at a real install.

    features   energy features -> log( f / median_ref(f) )
    sequences  each subcarrier bin / reference level of that bin * SEQ_SCALE

Reference windows are returned in `ref` and must never be scored. Windows of sites
without an empty reference get has_ref = False and are left out of calibrated scoring.
No label of a scored window is used: only the fact that the reference recording is empty.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import PROTOCOL
from data import feature_cols

from calibrate import EPS_CSI, EPS_RSSI, energy_cols      # V3, read-only
from calibrate_v3 import SEQ_SCALE                         # V3, read-only

REF_S = float(PROTOCOL["calibration"]["reference_s"])


def site_split(table: pd.DataFrame) -> pd.Series:
    return table["site"].astype(str) + "#" + table["split"].astype(str)


def reference_mask(table: pd.DataFrame) -> pd.Series:
    """True for the reference windows of every site-split."""
    key = site_split(table)
    ref = pd.Series(False, index=table.index)
    empty = table[(table["presence"] == 0) & table["valid"]]
    for k, idx in empty.groupby(key[empty.index]).groups.items():
        first = sorted(table.loc[idx, "recording"].unique())[0]
        sel = table.loc[idx]
        sel = sel[sel["recording"] == first]
        ref.loc[sel.index[(sel["t0"] + 3.0) <= REF_S]] = True
    return ref


def calibrate(table: pd.DataFrame, seq: np.ndarray):
    """Return (table_cal, seq_cal, ref, has_ref). Rows stay aligned with the input."""
    assert len(table) == len(seq)
    assert (table.index == np.arange(len(table))).all(), "table index must be 0..N-1 (row <-> seq)"

    table = table.copy()
    ref = reference_mask(table)
    key = site_split(table)
    ecols = energy_cols(feature_cols(table))
    eps = pd.Series({c: EPS_RSSI if c.startswith("rssi") else EPS_CSI for c in ecols})

    seq_cal = np.zeros(seq.shape, dtype=np.float16)
    has_ref = pd.Series(False, index=table.index)
    for k, idx in table.groupby(key).groups.items():
        r = idx[ref.loc[idx].to_numpy()]
        if len(r) == 0:
            continue
        has_ref.loc[idx] = True
        med = table.loc[r, ecols].median()
        table.loc[idx, ecols] = np.log((table.loc[idx, ecols] + eps) / (med + eps))
        level = np.maximum(np.median(np.abs(seq[r].astype(np.float32)), axis=(0, 2)), 1e-4)
        seq_cal[idx] = (seq[idx].astype(np.float32) / level[None, :, None] * SEQ_SCALE).astype(np.float16)
    return table, seq_cal, ref.to_numpy(), has_ref.to_numpy()

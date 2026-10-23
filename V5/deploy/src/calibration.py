"""Calibration: the empty-room offset every installation needs.

At install the room is empty for ~30 s: 10 s warm-up + 20 s = 7 windows. The offset is the mean model input
over those windows; every later window is measured relative to it (log features -> ratios to the empty room).
The offset belongs to one link in one room: recalibrate when a board, cable, antenna or big furniture moves.
"""

import json

import numpy as np

from data import EPS, FEATURE_NAMES, LOG_IDX, N_CAL_WINDOWS


def prepare_features(F, offset=None):
    """Raw features -> model input: NaN -> 0, log of magnitude features, minus offset."""
    T = np.nan_to_num(np.asarray(F, dtype=np.float64), nan=0.0)
    T = np.atleast_2d(T).copy()
    T[:, LOG_IDX] = np.log(np.maximum(T[:, LOG_IDX], 0.0) + EPS)
    if offset is not None:
        T = T - np.asarray(offset, dtype=np.float64)
    return T.astype(np.float32)


def calibration_offset(F_cal):
    """Empty-room offset = mean model input over the calibration windows."""
    return prepare_features(F_cal).astype(np.float64).mean(axis=0)


def offset_from_recording(F, n_windows=N_CAL_WINDOWS):
    """Offset from the first n_windows windows of a recording whose start is an empty room
    (features_from_raw already skips the 10 s warm-up). Returns (offset, number of windows used)."""
    F = np.asarray(F)
    if len(F) < n_windows:
        raise ValueError(f"need {n_windows} windows of empty room, got {len(F)} - record at least 30 s")
    return calibration_offset(F[:n_windows]), n_windows


def save_offset(path, offset, note=""):
    """Offset file: one JSON object, so the device / host can restart without recalibrating."""
    with open(path, "w") as f:
        json.dump({"feature_names": FEATURE_NAMES, "offset": [float(v) for v in offset], "note": note}, f, indent=1)


def load_offset(path):
    with open(path) as f:
        d = json.load(f)
    if d["feature_names"] != FEATURE_NAMES:
        raise ValueError(f"{path}: offset was made for other features")
    return np.asarray(d["offset"], dtype=np.float64)

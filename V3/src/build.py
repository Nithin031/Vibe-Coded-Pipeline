"""ESP32 stream -> window table (features) + window sequences (deep models), cached.

Every source is put on the SAME footing so that a model trained on one ESP32
setup can be applied to another:

    common rate     20 Hz   (the slowest real setup, OpenCSI; bin-averaged)
    window          3 s     (WiFall clips are 3 s long), hop 1 s
    subcarriers     L-LTF only, dead / spiky tones dropped per stream

Outputs, row-aligned:

    table   one row per window: ~45 features + meta columns
            (dataset, room, day, setup, recording, subject, activity, count, win, t0, missing)
    seq     float16 [N, 32, 60]  relative fluctuation, 32 subcarrier bins x 60 time steps
    gap     bool    [N, 60]      True where the time step had no real packet (lost / late)
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from esp32_io import LOADERS
from features import cnn_input, rssi_features, window_features
from preprocess import to_uniform_grid, valid_subcarriers


CACHE = Path(__file__).resolve().parents[1] / "cache"

FS = 20.0
WIN_S = 3.0
HOP_S = 1.0
W = int(round(WIN_S * FS))
H = int(round(HOP_S * FS))

META = ["dataset", "room", "day", "setup", "recording", "subject", "activity", "count",
        "win", "t0", "missing", "site",
        # V3: presence-level labels and recording conditions
        "presence", "motion", "chip", "role", "channel"]


# ---------------------------------------------------------------------------
# One stream
# ---------------------------------------------------------------------------

def stream_windows(s):
    """Return (rows, seqs [n, 32, W], gaps [n, W]) for one stream."""
    mask = valid_subcarriers(s.amp)
    x, missing = to_uniform_grid(s.amp[:, :, mask], s.t, FS)

    # RSSI on the same grid (one 'subcarrier')
    r, _ = to_uniform_grid(s.rssi[:, None, None].astype(np.float32), s.t, FS)
    r = r[:, 0, 0]

    rows, seqs, gaps = [], [], []
    for k, start in enumerate(range(0, len(x) - W + 1, H)):
        win = x[start: start + W]
        miss = missing[start: start + W]

        row = window_features(win, FS)
        row.update(rssi_features(r[start: start + W]))
        row.update(
            dataset=s.meta["dataset"], room=s.meta["room"], day=s.meta["day"],
            setup=s.meta["setup"], recording=s.meta["recording"],
            subject=s.meta.get("subject", ""), activity=s.meta.get("activity", ""),
            count=s.count, win=k, t0=start / FS, missing=float(miss.mean()),
            presence=int(s.meta.get("presence", int(s.count > 0))),
            motion=s.meta.get("motion", ""), chip=s.meta.get("chip", ""),
            role=s.meta.get("role", ""), channel=s.meta.get("channel", ""),
        )
        rows.append(row)
        seqs.append(cnn_input(win)[0].astype(np.float16))        # [32, W] (one link)
        gaps.append(miss)
    return rows, seqs, gaps


# ---------------------------------------------------------------------------
# Whole dataset, cached
# ---------------------------------------------------------------------------

def _paths(dataset):
    stem = f"{dataset}_fs{FS:g}_w{WIN_S:g}_h{HOP_S:g}"
    return CACHE / f"{stem}.pkl", CACHE / f"{stem}_seq.npz"


def _build(dataset):
    rows, seqs, gaps = [], [], []
    for s in LOADERS[dataset]():
        if s.duration < WIN_S - 2.0 / FS:              # WiFall clips are ~2.97 s
            continue
        r, q, g = stream_windows(s)
        rows += r
        seqs += q
        gaps += g

    df = pd.DataFrame(rows)
    tab_path, seq_path = _paths(dataset)
    CACHE.mkdir(exist_ok=True)
    tab_path.write_bytes(pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL))
    np.savez_compressed(seq_path, seq=np.stack(seqs), gap=np.stack(gaps))


def build(dataset: str) -> pd.DataFrame:
    """Feature table of one dataset."""
    tab_path, seq_path = _paths(dataset)
    if not (tab_path.exists() and seq_path.exists()):
        _build(dataset)
    return pickle.loads(tab_path.read_bytes())


def build_seq(dataset: str):
    """(seq float16 [N, 32, W], gap bool [N, W]) row-aligned with build(dataset)."""
    tab_path, seq_path = _paths(dataset)
    if not (tab_path.exists() and seq_path.exists()):
        _build(dataset)
    z = np.load(seq_path)
    return z["seq"], z["gap"]


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META]

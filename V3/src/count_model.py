"""CountModel: the inference API for the demo.

Feed it raw ESP32 serial lines (esp-csi 'CSI_DATA,...' format, any chip, router or
ESP-NOW setup). It needs 20 s of EMPTY room once at start-up (calibration), then
returns a people-count estimate every second.

    from count_model import CountModel

    cm = CountModel("D:/WIFI Based CSI Project/V2/models")
    cm.calibrate(empty_lines)                 # >= 20 s of serial lines, room empty
    for line in serial_port:                  # live loop
        out = cm.push(line)                   # None until a new estimate is ready
        if out:
            print(out["count"], out["probs"], out["presence"], out["confidence"], out["quality"])

Output (every 1 s once 3 s of data are buffered):
    count        smoothed people count 0 / 1 / 2 / 3 (3 means "3 or more")
    probs        class probabilities {0: .., 1: .., 2: .., 3: ..} for the last window
    presence     P(someone in the room) = 1 - P(0)
    confidence   probability of the reported count
    quality      share of the window covered by real packets (1.0 = no packet loss)
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

import deep
from calibrate import EPS_CSI, EPS_RSSI, energy_cols
from esp32_io import lltf_amplitude, parse_data_field, parse_raw_lines
from features import cnn_input, rssi_features, window_features
from preprocess import to_uniform_grid, valid_subcarriers

FS, WIN_S, HOP_S = 20.0, 3.0, 1.0
W = int(FS * WIN_S)


def lines_to_arrays(lines):
    """Raw serial lines -> (amp [T, 1, S], t [T] seconds, rssi [T]) from the ESP32 clock."""
    df = parse_raw_lines(lines)
    df = df[df["len"] == df["len"].mode()[0]]
    vals = parse_data_field(df["data"])
    ts = np.unwrap(df["local_timestamp"].to_numpy(np.float64), period=2 ** 32) / 1e6
    order = np.argsort(ts, kind="stable")
    amp = lltf_amplitude(vals)[order][:, None, :]
    return amp, ts[order], df["rssi"].to_numpy(np.float32)[order]


class CountModel:
    def __init__(self, model_dir, use=None, smooth=5, buffer_s=6.0):
        self.dir = Path(model_dir)
        self.card = json.loads((self.dir / "model_card.json").read_text())
        self.use = use or self.card["default_models"]
        self.cols = self.card["feature_cols"]
        self.ref_norm = self.card["ref_norm"]
        self.calibrated = self.card.get("calibrated", True)

        self.lgbm = [lgb.Booster(model_file=str(self.dir / f)) for f in self.card["files"]["lgbm"]] \
            if "lgbm" in self.use else []
        self.nets = []
        for kind in ("cnn", "tf"):
            if kind in self.use:
                for f in self.card["files"][kind]:
                    net = deep.make_model(kind, torch.load(self.dir / f, map_location="cpu"))
                    net.eval()
                    self.nets.append(net)

        self.lines = deque()
        self.buffer_s = buffer_s
        self.history = deque(maxlen=smooth)
        self.last_t = None
        self.ref = None

    # -- calibration ---------------------------------------------------------

    def calibrate(self, empty_lines):
        """Learn the empty-room reference (>= 20 s of lines with nobody in the room)."""
        amp, t, rssi = lines_to_arrays(empty_lines)
        t = t - t[0]
        self.mask = valid_subcarriers(amp)
        feats, seqs = self._windows(amp, t, rssi)
        if not feats:
            raise ValueError("need at least 3 s of empty-room data (20 s recommended)")

        f = pd.DataFrame(feats)
        ecols = energy_cols(self.cols)
        self.ref = dict(
            energy=f[ecols].median().to_dict(),
            level=float(np.median([np.median(np.abs(s.astype(np.float32))) for s in seqs])),
        )
        return self.ref

    # -- live ------------------------------------------------------------------

    def push(self, line):
        """Add one serial line; returns an estimate dict about once per second."""
        if not line.startswith("CSI_DATA"):
            return None
        self.lines.append(line)
        try:
            ts = int(line.split(",")[parse_ts_index(line)]) / 1e6
        except (ValueError, IndexError):
            return None

        # drop lines older than the buffer
        while len(self.lines) > 10 and self._age(self.lines[0], ts) > self.buffer_s:
            self.lines.popleft()

        if self.last_t is not None and ts - self.last_t < HOP_S:
            return None
        if self._age(self.lines[0], ts) < WIN_S:
            return None
        self.last_t = ts
        return self.predict_lines(list(self.lines))

    def predict_lines(self, lines):
        """Estimate from the LAST 3 s contained in `lines`."""
        if self.ref is None:
            raise RuntimeError("call calibrate() with empty-room data first")
        amp, t, rssi = lines_to_arrays(lines)
        t = t - t[0]
        keep = t >= t[-1] - WIN_S - 0.05
        feats, seqs, quality = self._windows(amp[keep], t[keep] - t[keep][0], rssi[keep], last_only=True)
        if not feats:
            return None

        p = self._predict(feats[0], seqs[0], quality)
        self.history.append(int(np.argmax(p)))
        count = int(np.round(np.median(self.history)))
        return dict(count=count, probs={c: float(v) for c, v in zip(deep.CLASSES, p)},
                    presence=float(1 - p[0]), confidence=float(p[count]), quality=quality)

    # -- internals -------------------------------------------------------------

    @staticmethod
    def _age(line, now):
        return now - int(line.split(",")[parse_ts_index(line)]) / 1e6

    def _windows(self, amp, t, rssi, last_only=False):
        x, missing = to_uniform_grid(amp[:, :, self.mask], t, FS)
        r, _ = to_uniform_grid(rssi[:, None, None].astype(np.float32), t, FS)
        starts = list(range(0, len(x) - W + 1, int(FS * HOP_S)))
        if last_only and starts:
            starts = [len(x) - W]
        feats, seqs = [], []
        for s in starts:
            f = window_features(x[s: s + W], FS)
            f.update(rssi_features(r[s: s + W, 0, 0]))
            feats.append(f)
            seqs.append(cnn_input(x[s: s + W])[0].astype(np.float16))
        if last_only:
            return feats, seqs, float(1 - missing[-W:].mean()) if len(missing) >= W else 0.0
        return feats, seqs

    def _predict(self, feat, seq, quality):
        probs = []

        if self.lgbm:
            f = pd.DataFrame([feat])[self.cols]
            if self.calibrated:
                for c in energy_cols(self.cols):
                    eps = EPS_RSSI if c.startswith("rssi") else EPS_CSI
                    f[c] = np.log((f[c] + eps) / (self.ref["energy"][c] + eps))
            probs += [b.predict(f.to_numpy())[0] for b in self.lgbm]

        if self.nets:
            gap = np.zeros((1, W), dtype=bool)
            ref = np.array([self.ref["level"] / self.ref_norm]) if self.calibrated else None
            for net in self.nets:
                probs.append(deep.predict_proba(net, seq[None], gap, ref=ref)[0])

        return np.mean(probs, axis=0)


def parse_ts_index(line):
    """Index of local_timestamp in a CSI_DATA line (older vs newer chip header)."""
    n = line.partition('"[')[0].rstrip(",").count(",") + 1
    return 18 if n >= 24 else 9

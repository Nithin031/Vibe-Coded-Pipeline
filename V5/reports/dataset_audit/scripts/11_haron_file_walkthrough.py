"""
Haron98 - detailed walk-through of individual .data files (one trial).

For each file given:
  - line-by-line validity (24 CSV fields + 128-int CSI list)
  - every metadata field: value set over the file
  - timing from host timestamp and ESP32 local_timestamp
  - CSI decoded per ESP-IDF LLTF layout: 64 subcarriers, order 0..31, -32..-1,
    each pair stored (imaginary, real); zero / guard tones, amplitude profile,
    temporal variability per subcarrier
Then compares files: receivers of the same trial, and other labels.

Usage:  py -P -E 11_haron_file_walkthrough.py <file1.data> [<file2.data> ...]
"""

import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime

import numpy as np

TS_FMT = "%d.%m.%Y %H:%M:%S.%f %z"
KEYS = ["seq", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth", "smoothing", "not_sounding",
        "aggregation", "stbc", "fec_coding", "sgi", "noise_floor", "ampdu_cnt", "channel",
        "secondary_channel", "local_timestamp", "ant", "sig_len", "rx_state", "len", "first_word"]

# ESP-IDF LLTF order: buffer index k -> subcarrier index
SC_INDEX = np.array(list(range(0, 32)) + list(range(-32, 0)))


def parse(path):
    good, bad = [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for n, line in enumerate(f.read().splitlines(), 1):
            try:
                ts_str, rest = line.split(" CSI_DATA,", 1)
                ts = datetime.strptime(ts_str.strip(), TS_FMT)
                row = next(csv.reader([rest.strip()], quotechar='"'))
                csi = json.loads(row[-1])
                if len(row) != 24 or len(csi) != 128:
                    raise ValueError(f"fields={len(row)} csi_len={len(csi)}")
                good.append((ts, dict(zip(KEYS, row[:-1])), csi))
            except Exception as e:
                bad.append((n, repr(e)[:80], line[:120]))
    return good, bad


def describe(path):
    good, bad = parse(path)
    print("=" * 100)
    print(os.path.basename(path))
    print(f"lines valid {len(good)} / {len(good) + len(bad)}")
    for b in bad:
        print("  malformed line", b[0], b[1], "|", b[2])

    # ---- metadata fields
    print("metadata value sets:")
    for k in KEYS:
        vals = Counter(fr[1][k] for fr in good)
        if k in ("seq", "local_timestamp"):
            v = [int(fr[1][k]) for fr in good]
            print(f"  {k:18s} first {v[0]}  last {v[-1]}  steps {Counter(np.diff(v).tolist()).most_common(3)}")
        else:
            print(f"  {k:18s} {dict(vals.most_common(5))}")

    # ---- timing
    t = np.array([fr[0].timestamp() for fr in good])
    dt = np.diff(t) * 1e3
    lt = np.diff([int(fr[1]["local_timestamp"]) for fr in good]) / 1e3
    print(f"host time {good[0][0]} -> {good[-1][0]}  ({t[-1] - t[0]:.2f} s)")
    print(f"host interval ms: median {np.median(dt):.1f}  min {dt.min():.1f}  max {dt.max():.1f}"
          f"  -> {len(good) / (t[-1] - t[0]):.1f} frames/s")
    print(f"ESP32 interval ms: median {np.median(lt):.1f}  min {lt.min():.1f}  max {lt.max():.1f}")

    # ---- CSI
    raw = np.array([fr[2] for fr in good], dtype=np.float32)          # (frames, 128)
    imag, real = raw[:, 0::2], raw[:, 1::2]                           # ESP-IDF: imaginary first
    H = real + 1j * imag
    A = np.abs(H)
    zero_sc = SC_INDEX[(A == 0).all(axis=0)]
    used = ~(A == 0).all(axis=0)
    print(f"CSI raw range [{raw.min():.0f}, {raw.max():.0f}]")
    print(f"always-zero subcarriers ({len(zero_sc)}): {sorted(zero_sc.tolist())}")
    print(f"usable subcarriers: {used.sum()}")

    order = np.argsort(SC_INDEX)                                       # -32..31 for display
    prof = A.mean(axis=0)[order]
    var = (A.std(axis=0) / (A.mean(axis=0) + 1e-9))[order]
    print("mean amplitude by subcarrier (-32..31):")
    print("  " + " ".join(f"{v:4.0f}" for v in prof))
    print("relative temporal std by subcarrier (-32..31):")
    print("  " + " ".join(f"{v:4.2f}" for v in var))
    print(f"summary: mean amp {A[:, used].mean():.2f}  mean rel. temporal std {var[used[order]].mean():.3f}"
          f"  RSSI mean {np.mean([int(fr[1]['rssi']) for fr in good]):.1f}")
    print("example frame 1, subcarriers -28..-24 (real, imag):",
          [(int(H[0, k].real), int(H[0, k].imag)) for k in np.where(np.isin(SC_INDEX, range(-28, -23)))[0]])
    return {"file": os.path.basename(path), "t0": good[0][0], "A": A, "used": used}


def main():
    infos = [describe(p) for p in sys.argv[1:]]
    print("=" * 100)
    print("cross-file comparison (amplitude profile correlation over usable subcarriers)")
    for i in range(len(infos)):
        for j in range(i + 1, len(infos)):
            u = infos[i]["used"] & infos[j]["used"]
            a, b = infos[i]["A"][:, u].mean(0), infos[j]["A"][:, u].mean(0)
            dt = (infos[j]["t0"] - infos[i]["t0"]).total_seconds()
            print(f"  {infos[i]['file'][:45]:45s} vs {infos[j]['file'][:45]:45s}"
                  f" corr {np.corrcoef(a, b)[0, 1]:+.2f}  start offset {dt:+.1f} s")


if __name__ == "__main__":
    main()

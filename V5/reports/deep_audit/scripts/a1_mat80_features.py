"""
80 MHz Human Sensing (Dataset/PC-<k>a/PC<k>a_nNN.mat) - per-window features, all files.

Memory-light: the compressed MAT stream is read column by column. Real parts are held
as int16, then each imaginary column is combined into an amplitude column (float16).
Peak RAM is about 2 x rows x 256 x 2 bytes (~200 MB for the largest file).

Rows cycle over 4 receive chains (verified in the first audit). This script also checks
that the cycle never slips: in every 4,000-row chunk the 4 row-phases must still match
the 4 chain profiles of the first chunk.

Per file it saves (to <out_dir>/<setup>_nNN.npz):
  X       (windows, 4 chains, 7 features)   1-s windows at the published 173 pkt/s per chain
  profile (windows, 4 chains, 242) float16  mean normalised amplitude per window
and prints a one-line summary.

Usage:  py -P -E a1_mat80_features.py <dataset_root> <out_dir>
"""

import glob
import json
import os
import struct
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features_common import KEEP_BINS, window_features  # noqa: E402

MI_TYPES = {1: "i1", 2: "u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4", 7: "<f4", 9: "<f8"}
RATE = 173                    # packets per second per chain (published)
CHUNK = 4000                  # rows per chain-slip check


# ---------------------------------------------------------------
# Streaming MAT-v5 reader (same approach as dataset_audit/scripts/07_mat80.py)
# ---------------------------------------------------------------
class ZStream:
    def __init__(self, f, comp_len):
        self.f, self.left, self.d, self.buf = f, comp_len, zlib.decompressobj(), bytearray()

    def read(self, n):
        while len(self.buf) < n:
            if self.left <= 0:
                self.buf += self.d.flush()
                break
            chunk = self.f.read(min(4 << 20, self.left))
            self.left -= len(chunk)
            self.buf += self.d.decompress(chunk)
        out = bytes(self.buf[:n])
        del self.buf[:n]
        return out


def read_tag(src):
    raw = src.read(8)
    t, n = struct.unpack("<II", raw)
    if t >> 16:
        return t & 0xFFFF, t >> 16, raw[4:4 + (t >> 16)], True
    return t, n, None, False


def read_element(src):
    t, n, small, is_small = read_tag(src)
    if is_small:
        return t, small
    data = src.read(n)
    if n % 8:
        src.read(8 - n % 8)
    return t, data


def load_amplitude(path):
    """Return amplitude (rows, 256) float16, or None if the file holds no data."""
    with open(path, "rb") as f:
        f.read(128)
        t, n = struct.unpack("<II", f.read(8))
        src = ZStream(f, n)
        read_tag(src)                                   # miMATRIX
        read_element(src)                               # flags
        _, dims = read_element(src)
        nrows, ncols = struct.unpack("<ii", dims[:8])
        read_element(src)                               # name
        if nrows == 0:
            return None

        # real part -> int16 matrix
        t, nbytes, _, _ = read_tag(src)
        dt = np.dtype(MI_TYPES[t])
        R = np.empty((nrows, ncols), dtype=np.int16)
        for c in range(ncols):
            R[:, c] = np.frombuffer(src.read(nrows * dt.itemsize), dtype=dt)
        if nbytes % 8:
            src.read(8 - nbytes % 8)

        # imaginary part -> amplitude column by column
        t, nbytes, _, _ = read_tag(src)
        dt = np.dtype(MI_TYPES[t])
        A = np.empty((nrows, ncols), dtype=np.float16)
        for c in range(ncols):
            im = np.frombuffer(src.read(nrows * dt.itemsize), dtype=dt).astype(np.float32)
            A[:, c] = np.hypot(R[:, c].astype(np.float32), im)
        del R
    return A


# ---------------------------------------------------------------
# Checks and features
# ---------------------------------------------------------------
def norm_profile(rows):
    p = rows[:, KEEP_BINS].astype(np.float32).mean(0)
    return (p - p.mean()) / (p.std() + 1e-6)


def chain_slips(A):
    """Number of 4,000-row chunks whose row-phases no longer match the first chunk's chains."""
    n = (len(A) // 4) * 4
    ref = [norm_profile(A[k:CHUNK:4]) for k in range(4)]
    bad, total = 0, 0
    for s in range(0, n - CHUNK + 1, CHUNK):
        prof = [norm_profile(A[s + k:s + CHUNK:4]) for k in range(4)]
        C = np.array([[np.dot(prof[j], ref[k]) / len(ref[k]) for k in range(4)] for j in range(4)])
        total += 1
        if not np.array_equal(C.argmax(1), np.arange(4)):
            bad += 1
    return bad, total


def main():
    root, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    summary = {}

    for path in sorted(glob.glob(os.path.join(root, "PC-*", "*.mat"))):
        name = os.path.basename(path)[:-4]                     # PC1a_n02
        setup, count = name.split("_")[0], int(name.split("_n")[1])
        out_file = os.path.join(out_dir, f"{name}.npz")
        if os.path.exists(out_file):                           # resume: already done
            continue
        try:
            A = load_amplitude(path)
        except struct.error:                                   # header but no data stream
            A = None
        if A is None:
            print(name, "EMPTY", flush=True)
            summary[name] = {"empty": True}
            continue

        n = (len(A) // 4) * 4
        slips, chunks = chain_slips(A[:n])

        per_chain = n // 4
        W = per_chain // RATE
        X = np.zeros((W, 4, 7), dtype=np.float32)
        P = np.zeros((W, 4, len(KEEP_BINS)), dtype=np.float16)
        for k in range(4):
            Ak = A[k:n:4]
            for w in range(W):
                seg = Ak[w * RATE:(w + 1) * RATE, KEEP_BINS].astype(np.float32)
                X[w, k] = window_features(seg, RATE)
                P[w, k] = (seg / (np.median(seg, axis=1, keepdims=True) + 1e-6)).mean(0)

        np.savez_compressed(os.path.join(out_dir, f"{name}.npz"), X=X, profile=P,
                            setup=setup, count=count)

        thirds = np.array_split(X[:, :, 0].mean(1), 3)
        summary[name] = {"setup": setup, "count": count, "rows": int(len(A)),
                         "seconds": round(per_chain / RATE, 1), "windows": int(W),
                         "chain_slip_chunks": f"{slips}/{chunks}",
                         "cv_median_first_third": round(float(thirds[0].mean()), 4),
                         "cv_median_last_third": round(float(thirds[2].mean()), 4),
                         "cv_median_all": round(float(X[:, :, 0].mean()), 4)}
        print(name, summary[name], flush=True)
        del A

    with open(os.path.join(out_dir, "_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()

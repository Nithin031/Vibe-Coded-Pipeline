"""
80 MHz Nexmon dataset ("Dataset/PC-<k>a/PC<k>a_nNN.mat") - streaming MAT-v5 reader.

Every file holds one variable, csi_buff (N x 256), stored compressed. Loading it
as complex double would need ~800 MB, so this script decompresses the zlib
stream incrementally and walks the column-major data one column at a time:

  - exact per-column (subcarrier-bin) statistics over ALL rows:
      zero fraction, mean |real|, mean |imag|, NaN count
  - stored element type (MATLAB may store integer-valued doubles as int16)
  - a row sample (first 4,000 rows contiguous + every 97th row) of the complex
    values, used to test whether consecutive rows cycle over 4 receive cores
    (row i vs i+1 vs i+4 similarity) and to describe amplitude / phase.

No data are written back; only summary statistics leave this script.

Usage:  py -I 07_mat80.py <dataset_root> <out_json>
"""

import glob
import json
import os
import struct
import sys
import zlib

import numpy as np

MI_TYPES = {1: "i1", 2: "u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4",
            7: "<f4", 9: "<f8", 12: "<i8", 13: "<u8"}
HEAD_ROWS = 4000
STRIDE = 97


# ----------------------------------------------------------------------
# Incremental byte source over a zlib stream
# ----------------------------------------------------------------------

class ZStream:
    def __init__(self, f, comp_len):
        self.f = f
        self.left = comp_len
        self.d = zlib.decompressobj()
        self.buf = bytearray()

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
    if t >> 16:                                   # small data element
        return t & 0xFFFF, t >> 16, raw[4:4 + (t >> 16)], True
    return t, n, None, False


def read_element(src):
    t, n, small, is_small = read_tag(src)
    if is_small:
        return t, small
    data = src.read(n)
    pad = (8 - n % 8) % 8
    if pad:
        src.read(pad)
    return t, data


# ----------------------------------------------------------------------
# Column-wise pass over one numeric part (real or imaginary)
# ----------------------------------------------------------------------

def stream_part(src, nrows, ncols, sample_idx):
    t, nbytes, small, is_small = read_tag(src)
    dt = np.dtype(MI_TYPES[t])
    per_col = nrows * dt.itemsize
    zero = np.zeros(ncols)
    absmean = np.zeros(ncols)
    nan = 0
    sat = 0
    vmin, vmax = np.inf, -np.inf
    sample = np.zeros((len(sample_idx), ncols), dtype=np.float64)
    for c in range(ncols):
        col = np.frombuffer(src.read(per_col), dtype=dt).astype(np.float64)
        zero[c] = (col == 0).mean()
        absmean[c] = np.abs(col).mean()
        nan += int(np.isnan(col).sum())
        sat += int((np.abs(col) >= 2047).sum())
        vmin, vmax = min(vmin, col.min()), max(vmax, col.max())
        sample[:, c] = col[sample_idx]
    pad = (8 - nbytes % 8) % 8
    if pad:
        src.read(pad)
    return {"stored_type": MI_TYPES[t], "zero": zero, "absmean": absmean,
            "nan": nan, "saturated": sat, "min": float(vmin), "max": float(vmax)}, sample


def inspect_file(path):
    with open(path, "rb") as f:
        f.read(128)                                   # text header + version
        t, n = struct.unpack("<II", f.read(8))
        if t != 15:                                   # 15 = miCOMPRESSED
            return {"error": f"top element type {t}, not compressed"}
        src = ZStream(f, n)

        mt, mn, _, _ = read_tag(src)                  # miMATRIX
        _, flags = read_element(src)
        flag_word = struct.unpack("<I", flags[:4])[0]
        is_complex = bool(flag_word & 0x0800)
        mclass = flag_word & 0xFF
        _, dims = read_element(src)
        nrows, ncols = struct.unpack("<ii", dims[:8])
        _, name = read_element(src)

        sample_idx = np.unique(np.concatenate([np.arange(min(HEAD_ROWS, nrows)),
                                               np.arange(0, nrows, STRIDE)]))
        re_stats, re_s = stream_part(src, nrows, ncols, sample_idx)
        im_stats, im_s = (stream_part(src, nrows, ncols, sample_idx)
                          if is_complex else (None, np.zeros_like(re_s)))

    z = re_s + 1j * im_s
    head = z[:min(HEAD_ROWS, nrows)]
    amp = np.abs(head)
    active = np.where(re_stats["zero"] < 0.5)[0]

    # does row i resemble row i+4 more than row i+1? (cyclic RX cores)
    def mean_corr(lag):
        a, b = amp[:-lag, active], amp[lag:, active]
        a = a - a.mean(1, keepdims=True)
        b = b - b.mean(1, keepdims=True)
        num = (a * b).sum(1)
        den = np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1)) + 1e-12
        return float(np.mean(num / den))

    energy = re_stats["absmean"] + (im_stats["absmean"] if im_stats else 0)
    low_bins = np.where(energy < 0.05 * np.median(energy))[0].tolist()

    lag_corr = {str(k): round(mean_corr(k), 4) for k in (1, 2, 3, 4, 8)} if nrows > 10 else {}

    # per-phase-of-4 mean power -> distinct cores have distinct gains
    core_power = [float((amp[k::4, active] ** 2).mean()) for k in range(4)] if nrows > 8 else []

    return {
        "variable": name.decode("latin1").strip("\x00"),
        "mx_class": mclass, "complex": is_complex,
        "shape": [nrows, ncols], "rows_divisible_by_4": nrows % 4 == 0,
        "real_stored_type": re_stats["stored_type"],
        "imag_stored_type": im_stats["stored_type"] if im_stats else None,
        "real_range": [re_stats["min"], re_stats["max"]],
        "imag_range": [im_stats["min"], im_stats["max"]] if im_stats else None,
        "nan": re_stats["nan"] + (im_stats["nan"] if im_stats else 0),
        "n_bins_mostly_zero": int((re_stats["zero"] >= 0.5).sum()),
        "mostly_zero_bins": np.where(re_stats["zero"] >= 0.5)[0].tolist(),
        "n_active_bins": int(len(active)),
        "low_energy_bins": low_bins,
        "n_low_energy_bins": len(low_bins),
        "saturated_values": re_stats["saturated"] + (im_stats["saturated"] if im_stats else 0),
        "saturated_fraction": (re_stats["saturated"] + (im_stats["saturated"] if im_stats else 0))
                              / (2.0 * nrows * ncols),
        "all_zero_rows_in_sample": int((np.abs(z).sum(1) == 0).sum()),
        "amp_head_mean": float(amp[:, active].mean()) if len(active) else None,
        "amp_head_p99": float(np.percentile(amp[:, active], 99)) if len(active) else None,
        "lag_corr_amplitude": lag_corr,
        "mean_power_by_row_mod4": core_power,
        "example_row0_bins_0_8": [[round(v.real, 1), round(v.imag, 1)] for v in z[0, :8]],
        "example_row0_bins_120_128": [[round(v.real, 1), round(v.imag, 1)] for v in z[0, 120:128]],
    }


def main():
    root, out_path = sys.argv[1], sys.argv[2]
    files = sorted(glob.glob(os.path.join(root, "PC-*", "*.mat")))
    results = {}
    for p in files:
        key = os.path.relpath(p, root).replace("\\", "/")
        try:
            results[key] = inspect_file(p)
        except Exception as e:
            results[key] = {"error": repr(e)}
        r = results[key]
        print(key, {k: r.get(k) for k in ("shape", "complex", "real_stored_type", "real_range",
                                          "n_low_energy_bins", "saturated_fraction", "rows_divisible_by_4",
                                          "lag_corr_amplitude", "mean_power_by_row_mod4", "error")},
              flush=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()

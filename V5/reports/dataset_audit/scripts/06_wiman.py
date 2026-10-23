"""
Wiman folder (amp/act_<a>_<b>.npy) - header scan of every file + memory-mapped
signal statistics on a stratified sample.

  - reads only the .npy header (shape, dtype, fortran order) for all files
  - parses filename grammar act_<a>_<b>
  - for a sample (every Nth file) opens with mmap_mode='r' and computes
    NaN/inf/zero counts, ranges, per-axis variance profile (to help decide
    which axis is time), and lag-1 autocorrelation along each axis
Never loads more than one sampled file at a time.

Usage:  py -I 06_wiman.py <wiman_root> <out_json> [sample_every]
"""

import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

NAME_RE = re.compile(r"^act_(\d+)_(\d+)\.npy$")


def read_header(path):
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
    return shape, fortran, str(dtype)


def lag1_autocorr(x, axis):
    """Mean lag-1 correlation along one axis (smooth axis -> close to 1)."""
    x = np.moveaxis(x, axis, 0).reshape(x.shape[axis], -1).astype(np.float64)
    a, b = x[:-1], x[1:]
    a = a - a.mean(0)
    b = b - b.mean(0)
    den = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    ok = den > 0
    return float(np.mean((a * b).sum(0)[ok] / den[ok])) if ok.any() else None


def main():
    root, out_path = sys.argv[1], sys.argv[2]
    every = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    files = sorted(glob.glob(os.path.join(root, "amp", "*.npy")))

    shapes, dtypes, fortran = Counter(), Counter(), Counter()
    first_ids, second_ids = Counter(), Counter()
    by_first = defaultdict(list)
    header_errors = []
    sizes = Counter()

    for p in files:
        fn = os.path.basename(p)
        m = NAME_RE.match(fn)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            first_ids[a] += 1
            second_ids[b] += 1
            by_first[a].append(b)
        try:
            shape, fo, dt = read_header(p)
            shapes[str(shape)] += 1
            dtypes[dt] += 1
            fortran[fo] += 1
            sizes[os.path.getsize(p)] += 1
        except Exception as e:
            header_errors.append(f"{fn}: {e}")

    # ---------------- sampled signal statistics ----------------
    sample = files[::every]
    for extra in ("act_24_89.npy",):                       # file flagged earlier
        ep = os.path.join(root, "amp", extra)
        if os.path.exists(ep) and ep not in sample:
            sample.append(ep)

    stats = []
    for p in sample:
        fn = os.path.basename(p)
        try:
            x = np.load(p, mmap_mode="r")
            arr = np.asarray(x)                           # ~3 MB, one file only
            s = {
                "file": fn, "shape": list(arr.shape), "dtype": str(arr.dtype),
                "nan": int(np.isnan(arr).sum()), "inf": int(np.isinf(arr).sum()),
                "zeros": int((arr == 0).sum()),
                "min": float(np.nanmin(arr)), "max": float(np.nanmax(arr)),
                "mean": float(np.nanmean(arr)),
                "var_by_axis_index": [
                    [round(float(v), 3) for v in np.nanvar(arr, axis=tuple(i for i in range(arr.ndim) if i != ax)).ravel()[:8]]
                    for ax in range(arr.ndim)],
                "lag1_autocorr_by_axis": [lag1_autocorr(arr, ax) if arr.shape[ax] > 2 else None
                                          for ax in range(arr.ndim)],
                "frozen_rows_axis0": int((np.diff(arr.reshape(arr.shape[0], -1), axis=0) == 0).all(axis=1).sum()),
            }
            del arr, x
        except Exception as e:
            s = {"file": fn, "error": repr(e)}
        stats.append(s)

    result = {
        "n_files": len(files),
        "header_shapes": dict(shapes),
        "header_dtypes": dict(dtypes),
        "fortran_order": {str(k): v for k, v in fortran.items()},
        "file_sizes_bytes": dict(sizes.most_common(5)),
        "header_errors": header_errors,
        "first_id_range": [min(first_ids), max(first_ids)],
        "n_first_ids": len(first_ids),
        "files_per_first_id": dict(Counter(first_ids.values())),
        "second_id_range": [min(second_ids), max(second_ids)],
        "n_second_ids": len(second_ids),
        "first_ids_missing": [i for i in range(min(first_ids), max(first_ids) + 1) if i not in first_ids],
        "sample_stats": stats,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    for k, v in result.items():
        if k != "sample_stats":
            print(k, ":", json.dumps(v)[:800])
    for s in stats[:6] + stats[-2:]:
        print(json.dumps(s)[:900])
    agg = [s for s in stats if "error" not in s]
    print("sampled", len(stats), "ok", len(agg),
          "nan", sum(s["nan"] for s in agg), "inf", sum(s["inf"] for s in agg),
          "min", min(s["min"] for s in agg), "max", max(s["max"] for s in agg),
          "frozen", sum(s["frozen_rows_axis0"] for s in agg))


if __name__ == "__main__":
    main()

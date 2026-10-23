"""
WiCount (ESP32-S3, CSV per 60 s recording) - full column-level inspection.

Reads every CSV (dataset is ~41 MB) and reports:
  - every column: dtype, missing rate, unique values / numeric range
  - CSI vector length, value range, always-zero positions
  - packet timing from host timestamp and ESP32 local_timestamp
  - seq gaps (packet loss), per-folder (people count) totals
Read-only; no transformation is written back.

Usage:  py -I 02_wicount.py <wicount_root> <out_json>
"""

import ast
import glob
import json
import os
import sys

import numpy as np
import pandas as pd


def column_profile(s):
    """Small, JSON-friendly summary of one column."""
    out = {"dtype": str(s.dtype), "missing_rate": float(s.isna().mean())}
    if pd.api.types.is_numeric_dtype(s):
        out.update(min=float(s.min()), max=float(s.max()),
                   n_unique=int(s.nunique()))
        if s.nunique() <= 12:
            out["values"] = {str(k): int(v) for k, v in s.value_counts().items()}
    else:
        out["n_unique"] = int(s.nunique())
        out["examples"] = [str(v)[:60] for v in s.dropna().unique()[:3]]
    return out


def main():
    root, out_path = sys.argv[1], sys.argv[2]
    files = sorted(glob.glob(os.path.join(root, "data", "*", "*.csv")))

    per_file = []
    all_cols = {}
    csi_min, csi_max = 0, 0
    zero_counts = None
    n_csi_rows = 0

    for path in files:
        label = os.path.basename(os.path.dirname(path))
        df = pd.read_csv(path)

        # ---- column profiles (merged over files: keep first-seen + ranges)
        for c in df.columns:
            if c == "data":
                continue
            prof = column_profile(df[c])
            all_cols.setdefault(c, []).append(prof)

        # ---- CSI vector
        lens = []
        rows = []
        for v in df["data"].dropna():
            arr = ast.literal_eval(v)
            lens.append(len(arr))
            if len(arr) == 104:
                rows.append(arr)
        csi = np.asarray(rows, dtype=np.int16)
        n_csi_rows += len(csi)
        if len(csi):
            csi_min = min(csi_min, int(csi.min()))
            csi_max = max(csi_max, int(csi.max()))
            z = (csi == 0).sum(axis=0)
            zero_counts = z if zero_counts is None else zero_counts + z

        # ---- timing
        host_t = pd.to_datetime(df["timestamp"], errors="coerce")
        host_dt = host_t.diff().dt.total_seconds().dropna()
        local = df["local_timestamp"].astype("int64")
        local_dt = local.diff().dropna() / 1e6          # microseconds -> s (ESP-IDF)
        seq_gap = df["seq"].diff().dropna()

        duration = (host_t.iloc[-1] - host_t.iloc[0]).total_seconds()
        per_file.append({
            "file": os.path.relpath(path, root).replace("\\", "/"),
            "people_folder_label": int(label),
            "rows": len(df),
            "rows_with_nan": int(df.isna().any(axis=1).sum()),
            "csi_len_values": sorted(set(lens)),
            "duration_s_host": round(duration, 2),
            "mean_rate_hz_host": round(len(df) / duration, 2) if duration else None,
            "host_dt_median_ms": round(host_dt.median() * 1e3, 2),
            "host_dt_p99_ms": round(host_dt.quantile(.99) * 1e3, 2),
            "host_dt_max_ms": round(host_dt.max() * 1e3, 2),
            "host_nonmonotonic": int((host_dt < 0).sum()),
            "local_dt_median_ms": round(local_dt.median() * 1e3, 3),
            "local_dt_max_ms": round(local_dt.max() * 1e3, 2),
            "local_nonmonotonic": int((local_dt < 0).sum()),
            "local_gaps_gt_20ms": int((local_dt > 0.020).sum()),
            "seq_gap_values": {str(k): int(v) for k, v in seq_gap.value_counts().head(6).items()},
            "sig_mode_values": sorted(df["sig_mode"].unique().tolist()),
            "mac_values": sorted(df["mac"].unique().tolist()),
            "rssi_mean": round(float(df["rssi"].mean()), 2),
            "rssi_range": [int(df["rssi"].min()), int(df["rssi"].max())],
        })

    # ---- merge column profiles across files
    merged = {}
    for c, profs in all_cols.items():
        p0 = dict(profs[0])
        if "min" in p0:
            p0["min"] = min(p["min"] for p in profs)
            p0["max"] = max(p["max"] for p in profs)
        merged[c] = p0

    by_label = {}
    for f in per_file:
        b = by_label.setdefault(f["people_folder_label"], {"recordings": 0, "rows": 0, "seconds": 0})
        b["recordings"] += 1
        b["rows"] += f["rows"]
        b["seconds"] += f["duration_s_host"]

    result = {
        "files": per_file,
        "columns": merged,
        "csi": {
            "n_rows_parsed": n_csi_rows,
            "value_range": [csi_min, csi_max],
            "always_zero_positions": [int(i) for i in np.where(zero_counts == n_csi_rows)[0]],
            "zero_fraction_per_position": [round(float(z) / n_csi_rows, 3) for z in zero_counts],
        },
        "by_label": by_label,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps({k: result[k] for k in ("by_label",)}, indent=1))
    print(json.dumps(result["columns"], indent=1)[:4000])
    print("CSI", result["csi"]["value_range"], "always zero:", result["csi"]["always_zero_positions"])
    for f in per_file:
        print({k: f[k] for k in ("file", "rows", "duration_s_host", "mean_rate_hz_host",
                                 "host_dt_median_ms", "host_dt_max_ms", "local_dt_median_ms",
                                 "local_gaps_gt_20ms", "local_nonmonotonic", "seq_gap_values",
                                 "sig_mode_values", "mac_values", "rssi_mean", "csi_len_values")})


if __name__ == "__main__":
    main()

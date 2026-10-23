"""
IIT-BHU multi-human CSV (ESP32-CSI-Tool style CSI_DATA lines) - full streamed parse.

Line layout observed:
  CSI_DATA,<role>,<mac>,<rssi>,...,<len>,[<len ints space-separated>],<unix time>
The bracketed CSI list has no commas, so a line splits into
  [head fields] + [CSI list] + [trailing unix time].

Reports:
  - field count / value sets for every header field (by position)
  - rows per file, duration (trailing unix time), packet rate
  - CSI length, value range, positions that are always zero
  - filename grammar: users, activities, repetition index; overlaps of user IDs
Read-only, one file at a time.

Usage:  py -I 04_iitbhu.py <iit_root> <out_json>
"""

import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

NAME_RE = re.compile(r"^User_((?:\d+_)+)([A-Za-z_]+?)\s*\((\d+)\)\.csv$")


def parse_line(line):
    a = line.find("[")
    b = line.find("]")
    if a < 0 or b < 0:
        return None
    head = line[:a].rstrip(",").split(",")
    tail = line[b + 1:].strip().strip(",")
    try:
        csi = [int(x) for x in line[a + 1:b].split()]
    except ValueError:                      # truncated / garbled token
        return None
    if not csi:
        return None
    return head, csi, tail


def main():
    root, out_path = sys.argv[1], sys.argv[2]
    files = sorted(glob.glob(os.path.join(root, "wifi_dataset_Collected", "*", "*.csv")))

    head_values = defaultdict(Counter)        # position -> value counter (sampled)
    head_len = Counter()
    csi_len = Counter()
    csi_min, csi_max = 0, 0
    zero_pos = defaultdict(lambda: np.zeros(1, dtype=np.int64))
    rows_csi = Counter()
    per_file = []
    name_fail = []
    users_by_group = defaultdict(set)
    activities_by_group = defaultdict(Counter)
    bad_lines = 0

    for path in files:
        group = os.path.basename(os.path.dirname(path))
        fn = os.path.basename(path)
        m = NAME_RE.match(fn)
        if m:
            users = [int(u) for u in m.group(1).strip("_").split("_")]
            acts = m.group(2)
            rep = int(m.group(3))
            users_by_group[group].update(users)
            activities_by_group[group][acts] += 1
        else:
            users, acts, rep = None, None, None
            name_fail.append(fn)

        n_rows, unix_t, rssi = 0, [], []
        macs = Counter()
        local_ts = []
        if os.path.getsize(path) == 0:
            per_file.append({"file": f"{group}/{fn}", "rows": 0, "empty": True})
            continue

        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                p = parse_line(line)
                if p is None:
                    bad_lines += 1
                    continue
                head, csi, tail = p
                n_rows += 1
                head_len[len(head)] += 1
                L = len(csi)
                csi_len[L] += 1
                if n_rows % 20 == 1:                      # sample header values
                    for i, v in enumerate(head):
                        if i not in (2, 18, 23):           # skip mac / ts columns
                            head_values[i][v] += 1
                macs[head[2]] += 1
                rssi.append(int(head[3]))
                try:
                    local_ts.append(int(head[18]))
                except (ValueError, IndexError):
                    pass
                if tail:
                    try:
                        unix_t.append(float(tail))
                    except ValueError:
                        pass
                arr = np.asarray(csi, dtype=np.int16)
                csi_min = min(csi_min, int(arr.min()))
                csi_max = max(csi_max, int(arr.max()))
                if zero_pos[L].shape[0] != L:
                    zero_pos[L] = np.zeros(L, dtype=np.int64)
                zero_pos[L] += (arr == 0)
                rows_csi[L] += 1

        t = np.asarray(unix_t)
        dt = np.diff(t) if len(t) > 1 else np.array([np.nan])
        lt = np.diff(np.asarray(local_ts, dtype=np.int64)) if len(local_ts) > 1 else np.array([0])
        dur = float(t[-1] - t[0]) if len(t) > 1 else None
        per_file.append({
            "file": f"{group}/{fn}", "group": group, "users": users, "activities": acts,
            "rep": rep, "rows": n_rows, "duration_s": dur,
            "rate_hz": round(n_rows / dur, 1) if dur else None,
            "dt_median_ms": round(float(np.nanmedian(dt)) * 1e3, 2),
            "dt_max_ms": round(float(np.nanmax(dt)) * 1e3, 1),
            "unix_nonmonotonic": int((dt < 0).sum()),
            "local_dt_median_us": float(np.median(lt)),
            "local_wraps_or_resets": int((lt < 0).sum()),
            "tx_macs": dict(macs.most_common(3)),
            "rssi_mean": round(float(np.mean(rssi)), 1),
            "start_unix": float(t[0]) if len(t) else None,
        })

    rates = [f["rate_hz"] for f in per_file if f.get("rate_hz")]
    durs = [f["duration_s"] for f in per_file if f.get("duration_s")]
    rows = [f["rows"] for f in per_file if f.get("rows")]

    # user id overlap between groups (same person in several groups?)
    groups = sorted(users_by_group)
    overlap = {f"{a}&{b}": sorted(users_by_group[a] & users_by_group[b])
               for i, a in enumerate(groups) for b in groups[i + 1:]}

    result = {
        "n_files": len(files),
        "bad_lines": bad_lines,
        "name_parse_failures": name_fail,
        "head_field_count": dict(head_len),
        "head_value_sets_sampled": {str(k): dict(v.most_common(8)) for k, v in sorted(head_values.items())},
        "csi_len": dict(csi_len),
        "csi_value_range": [csi_min, csi_max],
        "csi_zero_fraction": {str(L): [round(float(z) / rows_csi[L], 3) for z in zero_pos[L]]
                              for L in rows_csi},
        "rows_per_file": {"min": min(rows), "median": float(np.median(rows)), "max": max(rows)},
        "duration_s": {"min": min(durs), "median": float(np.median(durs)), "max": max(durs)},
        "rate_hz": {"min": min(rates), "median": float(np.median(rates)), "max": max(rates)},
        "users_by_group": {g: sorted(u) for g, u in users_by_group.items()},
        "n_users_by_group": {g: len(u) for g, u in users_by_group.items()},
        "user_overlap_between_groups": overlap,
        "activity_strings_by_group": {g: dict(c) for g, c in activities_by_group.items()},
        "files": per_file,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    for k, v in result.items():
        if k not in ("files", "csi_zero_fraction"):
            print(k, ":", json.dumps(v)[:1500])
    for L, zf in result["csi_zero_fraction"].items():
        zf = np.asarray(zf)
        print("len", L, "positions zero >99%:", np.where(zf > .99)[0].tolist())


if __name__ == "__main__":
    main()

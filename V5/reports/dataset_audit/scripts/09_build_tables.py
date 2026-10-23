"""
Build the tabular deliverables from the measured JSON files in _work/:

  dataset_inventory.csv   - one row per dataset (location, format, size, counts, status)
  label_distribution.csv  - dataset x class: recordings / files / rows / seconds
  audit_errors.log        - every failed / skipped / incomplete check

Only measured values are written by this script; interpretive columns
(identity, hardware, suitability) live in compatibility_matrix.csv and the
Markdown reports, where each claim carries its evidence class.

Usage:  py -P -E 09_build_tables.py <audit_dir>
"""

import csv
import json
import os
import sys
from collections import defaultdict


def load(work, name):
    with open(os.path.join(work, name), encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------
# Inventory
# ----------------------------------------------------------------------

IDENTITY = {
    "CountFi": ("CountFi (Apr-2025 'mixed' subset, route folder '48')", "pcap (Nexmon-CSI UDP)", "Raw complex CSI (int16 I/Q, 256 bins) + RSSI + frame metadata", "Fully audited (all packets decoded); labels unverified"),
    "Dataset": ("80 MHz Human Sensing (Meneghello et al.) - people-counting folders PC-1a..PC-4a", "MAT v5 (csi_buff)", "Raw complex CFR (int16, 256 bins x 4 RX cores interleaved)", "Fully audited (structure); setup semantics unresolved"),
    "Wiman": ("WiMANS (amplitude subset only)", "NPY float32", "Processed CSI amplitude (time x 3 TX x 3 RX x 30 sc)", "Partially audited - labels absent; REMOVED from local disk 2026-10-19 (user keeps it in Colab)"),
    "WiCount": ("WiCount (CSI-BERT2, Zhao et al.)", "CSV (ESP-IDF CSI_DATA)", "Raw complex CSI (52 sc, int) + RSSI + timestamps", "Fully audited"),
    "IIT-BHU": ("CSI Dataset for Multi-Human Activity Recognition (Gupta et al., IIT BHU)", "CSV (ESP32-CSI-Tool CSI_DATA)", "Raw complex CSI (128/384 int8 values) + RSSI + timestamps", "Fully audited (files); no local docs"),
    "CSI HAR": ("Dataset for HAR using Wi-Fi CSI (Zhuravchak & Kapshii, figshare 14386892)", "CSV (headerless)", "Amplitude + phase of 2x2x114 CSI (polar form of integer I/Q)", "Fully audited (files); no timestamps"),
    "Haron98_WIFI-CSI-Dataset": ("WiFi CSI Router Reception Dataset (HF Haron98 copy)", "Text log .data (ESP-IDF CSI_DATA)", "Raw complex CSI (64 sc, int8 values) + RSSI + metadata", "Fully audited (files); label semantics unknown; REMOVED from local disk 2026-10-19 (user keeps it in Colab)"),
}


def write_inventory(audit, inv):
    path = os.path.join(audit, "dataset_inventory.csv")
    cols = ["dataset_dir", "identity", "path", "primary_format", "content", "n_files", "n_dirs",
            "total_bytes", "total_gib", "extensions", "n_archives", "n_docs", "n_scripts",
            "n_label_like_files", "n_small_files_lt_1KiB", "n_exact_duplicate_groups",
            "extraction_state", "audit_status"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in inv["datasets"]:
            ident, fmt, content, status = IDENTITY[r["dataset_dir"]]
            w.writerow([r["dataset_dir"], ident, r["path"], fmt, content, r["n_files"], r["n_dirs"],
                        r["total_bytes"], r["total_gib"],
                        "; ".join(f"{k}:{v}" for k, v in r["extensions"].items()),
                        len(r["archives"]), len(r["docs"]), len(r["scripts"]),
                        r["n_label_like_files"], len(r["small_files"]),
                        len(r["exact_duplicate_groups"]),
                        "fully extracted (no archives present)", status])
    return path


# ----------------------------------------------------------------------
# Label distribution
# ----------------------------------------------------------------------

def label_rows(work):
    rows = []

    # WiCount - folder = people count (documented)
    wc = load(work, "wicount.json")
    for lab, b in sorted(wc["by_label"].items(), key=lambda x: int(x[0])):
        rows.append(["WiCount", "people_count", lab, "documented (README: folder = number of people)",
                     b["recordings"], b["recordings"], b["rows"], round(b["seconds"], 1), "packets", ""])

    # 80 MHz - filename nNN = number of concurrent people (DataPort page)
    m80 = load(work, "mat80.json")
    per = defaultdict(lambda: [0, 0, 0])
    for k, v in m80.items():
        n = int(k.split("_n")[1][:2])
        per[n][0] += 1
        rows_valid = v["shape"][0] if v.get("complex") else 0
        per[n][1] += rows_valid
        per[n][2] += 1 if rows_valid else 0
    for n in range(0, 11):
        if n not in per:
            rows.append(["80MHz (Dataset/)", "people_count", n, "documented (DataPort: PCxy_nN, N = concurrent people)",
                         0, 0, 0, "", "rows (4 per packet)", "MISSING in all 4 setups (n01 not in PC folders)"])
            continue
        files, r, usable = per[n]
        note = "PC2a_n00 holds 1 all-zero row -> only 3 usable n00 recordings" if n == 0 else ""
        rows.append(["80MHz (Dataset/)", "people_count", n, "documented (DataPort: PCxy_nN, N = concurrent people)",
                     usable, files, r, round(r / 4 / 173.0, 1),
                     "rows (4 per packet); seconds estimated at published 173 pkt/s", note])

    # CountFi - numbered subfolder (meaning NOT documented)
    cf = load(work, "countfi.json")["files"]
    agg = defaultdict(lambda: [0, 0, 0.0, set()])
    for k, v in cf.items():
        if v.get("status") != "ok":
            continue
        parts = k.split("/")
        cond, sub = parts[1], parts[2]
        if "_end_" not in cond:                     # head/end are simultaneous duplicates of time
            continue
        a = agg[int(sub)]
        a[0] += 1
        a[1] += v["packets"]
        a[2] += v["duration_s"]
        a[3].add(cond.replace("48_end_", ""))
    for sub in sorted(agg):
        f, p, s, conds = agg[sub]
        rows.append(["CountFi (local subset)", "subfolder_number (very likely passenger count)", sub,
                     "inferred (medium-high): page says ground-truth counts are included; the Apr-2025 download 48.zip holds only numbered folders",
                     len(conds), f, p, round(s, 1), "packets (rear 'end' receiver only)",
                     "conditions: " + ",".join(sorted(conds)) + "; 'head' receiver has the same seconds"])
    for missing in [0] + [n for n in range(1, 21) if n not in agg]:
        rows.append(["CountFi (local subset)", "subfolder_number (very likely passenger count)", missing,
                     "absent locally", 0, 0, 0, "", "", "paper reports classes up to 20"])

    # IIT-BHU - user-group folder (scenario size, not an occupancy annotation)
    ib = load(work, "iitbhu.json")
    g = defaultdict(lambda: [0, 0, 0.0])
    for f in ib["files"]:
        if not f.get("rows"):
            continue
        n = {"single_user": 1, "2_users": 2, "3_users": 3}[f["group"]]
        g[n][0] += 1
        g[n][1] += f["rows"]
        g[n][2] += f["duration_s"] or 0
    for n in sorted(g):
        rows.append(["IIT-BHU", "users_in_scenario (folder)", n,
                     "documented on DataPort abstract (1/2/3-user scenarios); per-file count from folder name",
                     g[n][0], g[n][0], g[n][1], round(g[n][2], 1), "packets",
                     "no 0-person class; 11 duplicate file pairs; 3 empty files excluded"])
    acts = defaultdict(int)
    for f in ib["files"]:
        if f.get("rows"):
            acts[(f["group"], f["activities"].lower())] += 1
    for (grp, a), n in sorted(acts.items()):
        rows.append(["IIT-BHU", "activity_string (filename)", f"{grp}:{a}", "filename", n, n, "", "", "", ""])

    # CSI HAR - per-row activity incl. no_person
    ch = load(work, "csihar.json")
    tot = defaultdict(int)
    for room, dist in ch["label_distribution_by_room"].items():
        for lab, n in dist.items():
            tot[lab] += n
            rows.append(["CSI HAR", "activity_per_row", f"{room}:{lab}", "label.csv (camera-assisted per packet)",
                         "", "", n, "", "rows (packets)", "sampling rate not documented -> seconds unknown"])
    for lab, n in sorted(tot.items()):
        rows.append(["CSI HAR", "activity_per_row (all rooms)", lab, "label.csv", "", "", n, "", "rows", ""])
    rows.append(["CSI HAR", "derived presence (no_person vs any activity)", "0 people", "inferred from 'no_person'",
                 "", "", tot["no_person"], "", "rows", "single-subject dataset: occupancy only 0 or 1"])
    rows.append(["CSI HAR", "derived presence (no_person vs any activity)", "1 person", "inferred",
                 "", "", sum(v for k, v in tot.items() if k != "no_person"), "", "rows", ""])

    # Haron98 - label_00..03 undefined
    hr = load(work, "haron.json")
    for lab in ["label_00", "label_01", "label_02", "label_03"]:
        rows.append(["Haron98", "label_dir (semantics UNDEFINED)", lab, "README: 'TODO'", 400, 1200,
                     "", "", "trials x 3 receivers", "label_00 recorded last for all 4 subjects (22:44-23:21)"
                     if lab == "label_00" else ""])

    # Wiman - labels not available locally
    wm = load(work, "wiman.json")
    rows.append(["WiMANS (Wiman/)", "number_of_users", "0..5 (published)", "annotation.csv NOT PRESENT locally",
                 wm["n_files"], wm["n_files"], "", round(wm["n_files"] * 3.0, 0), "3-s samples",
                 "published totals (all 11,286 samples): 0:594, 1:3564, 2-5:1782 each - cannot be mapped to local files"])
    return rows


def write_labels(audit, rows):
    path = os.path.join(audit, "label_distribution.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "label_type", "class", "label_source", "recordings_or_sessions",
                    "files", "rows_or_packets", "seconds", "row_unit", "notes"])
        w.writerows(rows)
    return path


# ----------------------------------------------------------------------
# Errors log
# ----------------------------------------------------------------------

def write_errors(audit, inv, work):
    lines = []
    for r in inv["datasets"]:
        for s in r["small_files"]:
            lines.append(f"[EMPTY/TINY] {s['path']} ({s['bytes']} bytes)")
        for e in r["errors"]:
            lines.append(f"[IO] {e}")
    cf = load(work, "countfi.json")["files"]
    for k, v in cf.items():
        if v.get("status") != "ok":
            lines.append(f"[PCAP] CountFi/{k}: status={v.get('status')} (no packets)")
    hr = load(work, "haron.json")
    lines.append(f"[PARSE] Haron98: {hr['lines_total'] - hr['lines_valid']} malformed lines "
                 f"(always line 1 of a file) - skipped by parser")
    ib = load(work, "iitbhu.json")
    lines.append(f"[PARSE] IIT-BHU: {ib['bad_lines']} lines with garbled/truncated CSI tokens skipped; "
                 f"{sum(v for k, v in ib['csi_len'].items() if k not in ('128', '384'))} lines with non-standard CSI length")
    lines.append("[MISSING] Wiman/amp/act_24_89.npy: listed as failed (WinError 8, out of memory) in "
                 "Wiman/validation_summary.txt and no longer present on disk; act_24 has 87/99 files")
    lines.append("[MISSING] Wiman: annotation.csv (labels), wifi_csi/mat (raw CSI) and video/ not present locally")
    lines.append("[MISSING] Dataset/: set_names.pdf (setup descriptions for PC-1a..PC-4a) not present; n01 absent in all PC folders")
    lines.append("[CONTENT] Dataset/PC-2a/PC2a_n00.mat: valid MAT, csi_buff is a single all-zero 1x256 uint8 row (no data)")
    lines.append("[DOC] Haron98: label_00..03 meanings are 'TODO' in README; license contradictory (YAML MIT vs body 'no license')")
    lines.append("[DOC] CountFi, IIT-BHU, CSI HAR, Wiman, Dataset/: no README/licence/docs present locally (documented from web)")
    lines.append("[UNRESOLVED] CountFi: meaning of numbered subfolders (1..10) and condition codes fm/mm/mm_v2/mm_v3/ms/rm not documented")
    lines.append("[UNRESOLVED] CSI HAR: packet rate / timestamps absent; label.csv has 1 row fewer than data.csv in every session; "
                 "label_boxes.csv row count exceeds data.csv by 0-8 rows")
    lines.append("[REMOVED] Wiman/ and Haron98_WIFI-CSI-Dataset/ permanently deleted from D:/wifi/Datasets on 2026-10-19 at the user's request (datasets kept in Google Colab); findings retained as a record")
    lines.append("[ENV] Python -I hides user site-packages; dpkt/h5py scripts run with -P -E instead")
    path = os.path.join(audit, "audit_errors.log")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main():
    audit = sys.argv[1]
    work = os.path.join(audit, "_work")
    inv = load(work, "inventory.json")
    print(write_inventory(audit, inv))
    print(write_labels(audit, label_rows(work)))
    print(write_errors(audit, inv, work))


if __name__ == "__main__":
    main()

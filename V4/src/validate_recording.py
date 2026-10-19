"""Check a NEW ESP32 counting recording (and its sidecar manifest) before it is used.

Accepts the esp-csi CSV format (header row with type, local_timestamp, rssi, len, data).
Every recording needs <file>.json next to it, following configs/recording_manifest_schema.json.

Checks
    manifest   all required fields present, count is 0-5, reference link given
    packets    rows kept / dropped and why, timestamp wraps, spikes, duplicates
    timing     duration >= 120 s (30 s for an empty reference), packet rate >= 50 Hz
    quality    share of valid 3 s windows >= 95 %, longest packet gap

Usage:  python V4/src/validate_recording.py path/to/rec.csv [more.csv ...]
Writes  <file>.validation.json next to each recording and prints PASS / FAIL.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import V4
from data import Recording, make_windows, read_esp_csv

SCHEMA = json.loads((V4 / "configs" / "recording_manifest_schema.json").read_text())
MIN_DURATION_S = 120.0
MIN_REFERENCE_S = 30.0
MIN_RATE_HZ = 50.0
MIN_VALID_SHARE = 0.95


def check_manifest(manifest: dict) -> list[str]:
    problems = [f"missing field: {k}" for k in SCHEMA["required"] if k not in manifest]
    count = manifest.get("count")
    if not isinstance(count, int) or not 0 <= count <= 5:
        problems.append(f"count must be an integer 0-5, got {count!r}")
    if count and not manifest.get("empty_reference_file"):
        problems.append("occupied recording without empty_reference_file")
    if isinstance(manifest.get("people_activity"), list) and count is not None \
            and len(manifest["people_activity"]) != count:
        problems.append("people_activity must list one entry per person")
    return problems


def validate(path: Path) -> dict:
    manifest_path = path.with_suffix(".json")
    report = {"file": str(path), "problems": []}
    if not manifest_path.exists():
        report["problems"].append(f"no manifest {manifest_path.name}")
        manifest = {}
    else:
        manifest = json.loads(manifest_path.read_text())
        report["problems"] += check_manifest(manifest)

    head = pd.read_csv(path, nrows=200, dtype=str, on_bad_lines="skip")
    n_values = int(pd.to_numeric(head["len"], errors="coerce").mode()[0])
    amp, t, rssi, stats = read_esp_csv(path, n_values)
    report["packets"] = stats

    is_reference = manifest.get("empty_reference_file") in (None, manifest.get("file_id"))
    need = MIN_REFERENCE_S if (manifest.get("count") == 0 and is_reference) else MIN_DURATION_S
    if stats["duration_s"] < need:
        report["problems"].append(f"duration {stats['duration_s']:.0f} s < {need:.0f} s")
    if stats["rate_hz"] < MIN_RATE_HZ:
        report["problems"].append(f"packet rate {stats['rate_hz']:.0f} Hz < {MIN_RATE_HZ:.0f} Hz")

    meta = {"source": "new", "recording": path.stem, "split": "new", "site": manifest.get("room", ""),
            "label_raw": str(manifest.get("count", "")), "presence": int((manifest.get("count") or 0) > 0),
            "motion": "unknown", "activity": "", "count": manifest.get("count", -1),
            "chip": manifest.get("chip", ""), "room": manifest.get("room", "")}
    rows, _, gap = make_windows(Recording(meta, amp, t, rssi), features=False)
    valid_share = float(np.mean([r["valid"] for r in rows])) if rows else 0.0
    gaps = np.diff(t)
    report["quality"] = {"windows": len(rows), "valid_share": valid_share,
                         "longest_gap_s": float(gaps.max()) if len(gaps) else None}
    if valid_share < MIN_VALID_SHARE:
        report["problems"].append(f"only {valid_share:.0%} valid windows (< {MIN_VALID_SHARE:.0%})")

    report["status"] = "PASS" if not report["problems"] else "FAIL"
    path.with_suffix(".validation.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    failed = 0
    for f in args.files:
        r = validate(f)
        print(f"{r['status']}  {f.name}" + "".join(f"\n      - {p}" for p in r["problems"]))
        failed += r["status"] == "FAIL"
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

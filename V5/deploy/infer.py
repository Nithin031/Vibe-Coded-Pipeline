"""Run the V5 people counter on a recorded ESP32 CSI file.

The model needs an empty-room calibration of the same link in the same room. Give it one of:
    --calibrate-start      the recording starts with >= 30 s of empty room (its first 7 windows are used, not scored)
    --empty EMPTY.csv      a separate empty-room recording (same boards, same placement)
    --offset OFFSET.json   an offset saved earlier (--save-offset)

Examples (from the deploy folder):
    python infer.py session.csv --calibrate-start --save-offset offset_linkA.json
    python infer.py later.csv --offset offset_linkA.json --out later_counts.csv
    python infer.py two_people.csv --empty empty.csv --count 2          (prints scores against the true count)
    python infer.py session.csv --calibrate-start --stream              (packet-by-packet, exactly as the device)

Input: CSV with the esp-csi csi_recv columns (local_timestamp, data) - WiCount files and
V5/recording_kit/record_csi.py output both work. Output: one row per 2.5 s window.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

import calibration  # noqa: E402
import data  # noqa: E402
import metrics  # noqa: E402
import models  # noqa: E402


# ===============================================================
# Batch (fast) and streaming (device path) inference
# ===============================================================
def run_batch(model, t, A, offset, skip_windows):
    starts, F = data.features_from_raw(t, A)
    starts, F = starts[skip_windows:], F[skip_windows:]
    raw = np.clip(model.predict(calibration.prepare_features(F, offset)), 0, model.max_count)
    smoothed = metrics.smooth_predictions(raw)
    return pd.DataFrame({"t_start_s": starts / data.FS, "t_end_s": (starts + data.WIN) / data.FS,
                         "raw": raw, "smoothed": smoothed, "count": metrics.to_count(smoothed, model.max_count)})


def run_stream(model, t, A, offset, calibrate):
    counter = models.StreamingCounter(model, offset=offset)
    if calibrate:
        counter.begin_calibration()
    out = []
    for ti, ai in zip(t, A):
        out += counter.push(ti, ai)
    out += counter.flush()
    ok = [o for o in out if o.get("status") == "ok"]
    table = pd.DataFrame({"t_start_s": [o["t_start_s"] for o in ok], "t_end_s": [o["t_end_s"] for o in ok],
                          "raw": [o["raw"] for o in ok], "smoothed": [o["smoothed"] for o in ok],
                          "count": [o["count"] for o in ok]})
    return table, counter.offset


# ===============================================================
# Main
# ===============================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", help="ESP32 CSI CSV")
    cal = ap.add_mutually_exclusive_group()
    cal.add_argument("--calibrate-start", action="store_true", help="first 30 s of the recording are an empty room")
    cal.add_argument("--empty", help="separate empty-room recording")
    cal.add_argument("--offset", help="saved offset JSON")
    ap.add_argument("--save-offset", help="write the offset used to this JSON file")
    ap.add_argument("--count", type=int, help="true number of people (prints scores)")
    ap.add_argument("--out", help="write per-window results to this CSV")
    ap.add_argument("--stream", action="store_true", help="packet-by-packet (device path) instead of batch")
    ap.add_argument("--model-dir", default=str(HERE), help="folder with trees.json + model_card.json")
    args = ap.parse_args()

    model, card = models.load_bundle(args.model_dir)
    t, A = data.read_esp_csv(args.recording)
    rate = len(t) / max(t[-1] - t[0], 1e-9)
    print(f"{args.recording}: {len(t)} packets, {t[-1] - t[0]:.1f} s, {rate:.0f} packets/s")
    if rate < 50:
        print("WARNING: below 50 packets/s - most windows will be skipped (aim for >= 100)")

    # ---- calibration ----
    offset, skip = None, 0
    if args.offset:
        offset = calibration.load_offset(args.offset)
        source = f"offset file {args.offset}"
    elif args.empty:
        te, Ae = data.read_esp_csv(args.empty)
        _, Fe = data.features_from_raw(te, Ae)
        offset, _ = calibration.offset_from_recording(Fe)
        source = f"empty recording {args.empty}"
    elif args.calibrate_start:
        _, F = data.features_from_raw(t, A)
        offset, skip = calibration.offset_from_recording(F)
        source = f"first {skip} windows of this recording (not scored)"
    else:
        source = "NONE"
        if "empty_cal" in str(card.get("calibration", "")):
            print("WARNING: this model expects an empty-room calibration; counts without one are unreliable")
    print("calibration:", source)

    # ---- inference ----
    if args.stream:
        table, offset_used = run_stream(model, t, A, offset, calibrate=args.calibrate_start)
        offset = offset_used if offset is None else offset
    else:
        table = run_batch(model, t, A, offset, skip)
    table["label"] = [metrics.honest_label(c) for c in table["count"]]

    if args.save_offset and offset is not None:
        calibration.save_offset(args.save_offset, offset, note=f"from {source}")
        print("offset saved:", args.save_offset)

    # ---- report ----
    if table.empty:
        print("no windows answered (recording too short, too many gaps, or only calibration)")
        return
    print(f"\n{len(table)} windows ({data.HOP_S} s apart). Shown answer = 30 s median, honest output empty / 1 / 2 / 3+")
    print(table.groupby("label").size().rename("windows").to_string())
    print(f"most common answer: {table['label'].mode().iloc[0]}  |  median estimate {table['smoothed'].median():.2f}")

    if args.count is not None:
        s = metrics.score(args.count, table["count"].values)
        print(f"\nvs true count {args.count}: " + ", ".join(
            f"{k} {v:.3f}" for k, v in s.items() if k != "windows" and not pd.isna(v)))

    if args.out:
        table.to_csv(args.out, index=False)
        print("per-window results:", args.out)


if __name__ == "__main__":
    main()

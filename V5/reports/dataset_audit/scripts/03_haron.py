"""
Haron98 WIFI-CSI-Dataset (ESP32 CSI_DATA text logs, .data) - full parse.

Every line: "<dd.mm.yyyy HH:MM:SS.fff +TZ> CSI_DATA,<23 fields>,"[128 ints]""
Field names follow the README schema (standard ESP-IDF csi_recv layout).

Reports per (person, label, device):
  - valid / malformed line counts, field value sets (rate, sig_mode, mcs, ...)
  - frame interval from host timestamp, trial duration
  - CSI length, value range, positions that are always zero
  - simple signal statistics per label (amplitude mean / temporal std) -
    used only as a hint about label semantics, never as proof

Usage:  py -I 03_haron.py <haron_root> <out_json>
"""

import csv
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

TS_FMT = "%d.%m.%Y %H:%M:%S.%f %z"
KEYS = ["seq", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth", "smoothing",
        "not_sounding", "aggregation", "stbc", "fec_coding", "sgi", "noise_floor",
        "ampdu_cnt", "channel", "secondary_channel", "local_timestamp", "ant",
        "sig_len", "rx_state", "len", "first_word"]
TRACK = ["mac", "rate", "sig_mode", "mcs", "bandwidth", "smoothing", "not_sounding",
         "aggregation", "stbc", "fec_coding", "sgi", "noise_floor", "ampdu_cnt",
         "channel", "secondary_channel", "ant", "sig_len", "rx_state", "len", "first_word"]


def parse_line(line):
    if " CSI_DATA," not in line:
        return None
    ts_str, rest = line.split(" CSI_DATA,", 1)
    try:
        ts = datetime.strptime(ts_str.strip(), TS_FMT)
        row = next(csv.reader([rest.strip()], delimiter=",", quotechar='"'))
        if len(row) != 24:
            return None
        vals = json.loads(row[-1])
        if len(vals) != 128:
            return None
    except Exception:
        return None
    meta = dict(zip(KEYS, row[:-1]))
    return ts, meta, vals


def main():
    root, out_path = sys.argv[1], sys.argv[2]
    files = sorted(glob.glob(os.path.join(root, "id_person_*", "label_*", "test_*", "*.data")))

    field_values = defaultdict(Counter)
    lines_total = lines_valid = 0
    malformed_line_no = Counter()
    lines_per_file = Counter()
    csi_min, csi_max = 0, 0
    zero_pos = np.zeros(128, dtype=np.int64)
    rssi_by_label = defaultdict(list)
    dt_all = []
    dur_all = []
    t_first, t_last = None, None
    label_stats = defaultdict(lambda: {"amp_mean": [], "amp_tstd": [], "trials": 0})
    per_person_label_time = defaultdict(list)
    seq_steps = Counter()
    dev_macs = Counter()

    for path in files:
        parts = path.replace("\\", "/").split("/")
        person, label = parts[-4], parts[-3]
        dev = os.path.basename(path).split("__")[1].split("_")[0]
        dev_macs[os.path.basename(path).split("__")[1][5:].replace(".data", "")] += 1

        frames = []
        with open(path, encoding="utf-8", errors="replace") as f:
            raw = f.read().splitlines()
        lines_per_file[len(raw)] += 1
        for i, line in enumerate(raw):
            lines_total += 1
            p = parse_line(line)
            if p is None:
                malformed_line_no[i + 1] += 1
                continue
            lines_valid += 1
            frames.append(p)

        if not frames:
            continue
        ts = [fr[0] for fr in frames]
        if t_first is None or ts[0] < t_first:
            t_first = ts[0]
        if t_last is None or ts[-1] > t_last:
            t_last = ts[-1]
        per_person_label_time[(person, label)].append(ts[0])

        dts = np.diff([t.timestamp() for t in ts])
        dt_all.extend(dts.tolist())
        dur_all.append(ts[-1].timestamp() - ts[0].timestamp())

        seqs = [int(fr[1]["seq"]) for fr in frames]
        seq_steps.update(np.diff(seqs).tolist())

        for fr in frames:
            for k in TRACK:
                field_values[k][fr[1][k]] += 1
            rssi_by_label[label].append(int(fr[1]["rssi"]))

        csi = np.asarray([fr[2] for fr in frames], dtype=np.int16)
        csi_min = min(csi_min, int(csi.min()))
        csi_max = max(csi_max, int(csi.max()))
        zero_pos += (csi == 0).sum(axis=0)

        # amplitude per subcarrier: (I, Q) interleaving per README -> evidence checked below
        cplx = csi[:, 0::2].astype(np.float32) + 1j * csi[:, 1::2].astype(np.float32)
        amp = np.abs(cplx)
        if dev == "dev1":
            label_stats[label]["amp_mean"].append(float(amp.mean()))
            label_stats[label]["amp_tstd"].append(float(amp.std(axis=0).mean()))
            label_stats[label]["trials"] += 1

    # time spans per (person, label) - shows recording order / sessions
    order = sorted(((min(v), max(v), k) for k, v in per_person_label_time.items()))
    sessions = [{"person": k[0], "label": k[1], "start": str(a), "end": str(b)} for a, b, k in order]

    dt = np.asarray(dt_all)
    result = {
        "n_files": len(files),
        "lines_per_file": dict(lines_per_file),
        "lines_total": lines_total,
        "lines_valid": lines_valid,
        "malformed_by_line_number": dict(malformed_line_no.most_common(10)),
        "field_value_sets": {k: dict(v.most_common(12)) for k, v in field_values.items()},
        "device_mac_suffixes": dict(dev_macs),
        "seq_step_hist": {str(k): v for k, v in seq_steps.most_common(10)},
        "frame_interval_s": {
            "median": float(np.median(dt)), "p5": float(np.percentile(dt, 5)),
            "p95": float(np.percentile(dt, 95)), "max": float(dt.max()),
            "n_nonpositive": int((dt <= 0).sum()),
        },
        "trial_duration_s": {"median": float(np.median(dur_all)), "min": float(np.min(dur_all)),
                             "max": float(np.max(dur_all))},
        "time_range": [str(t_first), str(t_last)],
        "csi_value_range": [csi_min, csi_max],
        "csi_zero_fraction_per_position": [round(float(z) / lines_valid, 3) for z in zero_pos],
        "rssi_by_label": {k: {"mean": round(float(np.mean(v)), 2), "std": round(float(np.std(v)), 2)}
                          for k, v in sorted(rssi_by_label.items())},
        "dev1_amplitude_by_label": {
            k: {"trials": v["trials"],
                "amp_mean": round(float(np.mean(v["amp_mean"])), 2),
                "temporal_std_mean": round(float(np.mean(v["amp_tstd"])), 3)}
            for k, v in sorted(label_stats.items())},
        "recording_order": sessions,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    for k, v in result.items():
        if k != "csi_zero_fraction_per_position":
            print(k, ":", json.dumps(v)[:1500])
    zf = np.asarray(result["csi_zero_fraction_per_position"])
    print("positions zero >99%:", np.where(zf > .99)[0].tolist())


if __name__ == "__main__":
    main()

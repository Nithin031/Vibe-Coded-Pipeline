"""
CountFi - label / time structure from the saved full-scan JSONs (no pcap reading).

For every (scenario, receiver, count) folder:
  files, seconds, recording date + clock time, contiguous blocks (gap > 5 s),
  RSSI, QoS / Block-Ack share, QoS guard-band ratio (is the VHT80 guard empty?).
Also: head/end pairing (same seconds?) and the time order of counts per scenario.

Usage:  py -P -E c1_countfi_structure.py <countfi.json> <countfi_full.json> <out_json>
"""

import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np

BLOCK_GAP_S = 5.0
HK = timezone(timedelta(hours=8))          # PolyU, Hong Kong local time


def main():
    scan = json.load(open(sys.argv[1], encoding="utf-8"))["files"]
    full = json.load(open(sys.argv[2], encoding="utf-8"))["files"]

    # ---- One row per usable file ----
    rows = []
    for rel, s in scan.items():
        if s.get("status") != "ok":
            continue
        _, scen_full, count, name = rel.split("/")
        recv = scen_full.split("_")[1]                     # head / end
        scen = "_".join(scen_full.split("_")[2:])          # fm, mm, mm_v2 ...
        ft = full[rel]["frame_types"]
        q, b = ft.get("qos", {}), ft.get("blockack", {})
        rows.append({
            "rel": rel, "scen": scen, "recv": recv, "count": int(count), "idx": int(name.split(".")[0]),
            "start": s["start"], "dur": s["duration_s"], "rssi": s["rssi_mean"],
            "n_qos": q.get("packets", 0), "n_ba": b.get("packets", 0),
            "qos_guard": q.get("guard_ratio_median", np.nan),
            "qos_level": q.get("median_level", np.nan),
            "qos_step": q.get("median_rel_adjacent_step", np.nan),
        })

    # ---- Per folder summary ----
    groups = defaultdict(list)
    for r in rows:
        groups[(r["scen"], r["recv"], r["count"])].append(r)

    folders = []
    for (scen, recv, count), g in sorted(groups.items()):
        g.sort(key=lambda r: r["start"])
        starts = np.array([r["start"] for r in g])
        gaps = np.diff(starts)
        n_blocks = 1 + int((gaps > BLOCK_GAP_S).sum())
        t0 = datetime.fromtimestamp(starts[0], HK)
        t1 = datetime.fromtimestamp(starts[-1] + g[-1]["dur"], HK)
        # does file index order follow time order?
        idx_sorted = [r["idx"] for r in g]
        folders.append({
            "scen": scen, "recv": recv, "count": count, "files": len(g),
            "seconds": round(float(sum(r["dur"] for r in g)), 1),
            "date": t0.strftime("%Y-%m-%d"), "from": t0.strftime("%H:%M:%S"), "to": t1.strftime("%H:%M:%S"),
            "span_min": round((starts[-1] - starts[0]) / 60, 1),
            "blocks": n_blocks, "max_gap_s": round(float(gaps.max()), 1) if len(gaps) else 0,
            "index_follows_time": idx_sorted == sorted(idx_sorted),
            "rssi": round(float(np.mean([r["rssi"] for r in g])), 1),
            "qos_share": round(float(np.sum([r["n_qos"] for r in g]) / (1000 * len(g))), 3),
            "qos_guard_median": round(float(np.nanmedian([r["qos_guard"] for r in g])), 3),
            "qos_guard_frac_files_open": round(float(np.mean([r["qos_guard"] > 0.2 for r in g])), 3),
            "qos_level": round(float(np.nanmedian([r["qos_level"] for r in g])), 1),
        })

    # ---- head / end pairing: nearest start in the other receiver ----
    pairing = {}
    for scen in sorted({r["scen"] for r in rows}):
        for count in sorted({r["count"] for r in rows if r["scen"] == scen}):
            h = np.array(sorted(r["start"] for r in rows if r["scen"] == scen and r["count"] == count and r["recv"] == "head"))
            e = np.array(sorted(r["start"] for r in rows if r["scen"] == scen and r["count"] == count and r["recv"] == "end"))
            if len(h) and len(e):
                d = np.abs(h[:, None] - e[None, :]).min(1)
                pairing[f"{scen}/{count}"] = {"head": len(h), "end": len(e),
                                              "median_offset_s": round(float(np.median(d)), 3),
                                              "frac_within_0.5s": round(float((d < 0.5).mean()), 3)}

    # ---- order of counts in time, per scenario (head receiver) ----
    order = {}
    for scen in sorted({r["scen"] for r in rows}):
        f = [x for x in folders if x["scen"] == scen and x["recv"] == "head"]
        f.sort(key=lambda x: (x["date"], x["from"]))
        order[scen] = [f"{x['count']}@{x['date'][5:]} {x['from'][:5]}-{x['to'][:5]}" for x in f]

    out = {"folders": folders, "head_end_pairing": pairing, "count_time_order": order}
    json.dump(out, open(sys.argv[3], "w", encoding="utf-8"), indent=1)

    # ---- Print ----
    print(f"{'scen':6s} {'recv':4s} {'n':>2s} {'files':>5s} {'sec':>6s} {'date':10s} {'from':8s} {'to':8s} "
          f"{'blk':>3s} {'maxgap':>7s} {'rssi':>6s} {'qos%':>5s} {'guardMed':>8s} {'guardOpen':>9s} {'lvl':>6s}")
    for x in folders:
        print(f"{x['scen']:6s} {x['recv']:4s} {x['count']:2d} {x['files']:5d} {x['seconds']:6.0f} {x['date']} "
              f"{x['from']} {x['to']} {x['blocks']:3d} {x['max_gap_s']:7.1f} {x['rssi']:6.1f} "
              f"{x['qos_share']:5.2f} {x['qos_guard_median']:8.3f} {x['qos_guard_frac_files_open']:9.2f} {x['qos_level']:6.0f}")
    print()
    for k, v in pairing.items():
        print("pair", k, v)
    print()
    for k, v in order.items():
        print("order", k, v)


if __name__ == "__main__":
    main()

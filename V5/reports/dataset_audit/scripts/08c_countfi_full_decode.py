"""
CountFi - full decode of EVERY packet in every pcap (follow-up to 08/08b).

Per file, checks:
  pcap integrity    - global magic, link type, truncated records (incl_len < orig_len),
                      IPv4 header length, UDP ports / length field
  Nexmon header     - magic 0x1111, chanspec, core/ss
  CSI per frame type (QoS-data 0x88, Block-Ack 0x94, other):
      packets, all-zero CSI packets, int16 extreme values (|v| >= 32767 / -32768),
      VHT80 guard check (bins 123-133 low vs active median),
      within-packet spectral smoothness (median relative adjacent-bin step),
      consecutive identical CSI payloads
  timing            - packets, duration, gaps

CSI decode = little-endian int16 (real, imag) pairs, established in 08b.

Usage:  py -P -E 08c_countfi_full_decode.py <countfi_root> <out_json>
"""

import glob
import json
import os
import struct
import sys
from collections import Counter, defaultdict

import numpy as np

ETH, IP, UDP, NEX = 14, 20, 8, 18
GUARD = np.arange(123, 134)                       # VHT80 guard band (natural FFT order)
DC = np.array([0, 1, 255])
FRAME_NAMES = {0x88: "qos", 0x94: "blockack"}


def scan(path):
    out = {"bytes": os.path.getsize(path)}
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 24:
        out["status"] = "empty"
        return out, None
    magic, _, _, _, _, snap, link = struct.unpack_from("<IHHiIII", data, 0)
    out["link"] = link
    if magic not in (0xA1B2C3D4, 0xA1B23C4D):
        out["status"] = "bad_magic"
        return out, None

    off = 24
    ts, fctl, rssi, payloads = [], [], [], []
    trunc = bad_ip = bad_udp = bad_nex = 0
    chans, cores = Counter(), Counter()
    while off + 16 <= len(data):
        sec, frac, incl, orig = struct.unpack_from("<IIII", data, off)
        off += 16
        pkt = data[off:off + incl]
        off += incl
        if incl < orig or len(pkt) < incl:
            trunc += 1
            continue
        ihl = (pkt[ETH] & 0x0F) * 4
        if pkt[ETH] >> 4 != 4 or ihl != IP or pkt[ETH + 9] != 17:
            bad_ip += 1
            continue
        sport, dport, ulen = struct.unpack_from(">HHH", pkt, ETH + IP)
        if sport != 5500 or dport != 5500 or ulen != len(pkt) - ETH - IP:
            bad_udp += 1
            continue
        p = pkt[ETH + IP + UDP:]
        if p[:2] != b"\x11\x11" or len(p) != NEX + 1024:
            bad_nex += 1
            continue
        ts.append(sec + frac * 1e-6)
        rssi.append(struct.unpack_from("<b", p, 2)[0])
        fctl.append(p[3])
        cores[struct.unpack_from("<H", p, 12)[0]] += 1
        chans[struct.unpack_from("<H", p, 14)[0]] += 1
        payloads.append(p[NEX:])

    n = len(payloads)
    out.update({"status": "ok" if n else "header_only", "packets": n, "truncated_records": trunc,
                "bad_ip": bad_ip, "bad_udp": bad_udp, "bad_nexmon": bad_nex,
                "chanspec": {f"0x{k:04x}": v for k, v in chans.items()},
                "core_ss": {str(k): v for k, v in cores.items()}})
    if not n:
        return out, None

    raw = np.frombuffer(b"".join(payloads), dtype="<i2").reshape(n, 512)
    Z = raw[:, 0::2].astype(np.float32) + 1j * raw[:, 1::2].astype(np.float32)
    A = np.abs(Z)
    fc = np.asarray(fctl)
    t = np.asarray(ts)
    dt = np.diff(t)
    out.update({"duration_s": float(t[-1] - t[0]),
                "dt_max_ms": float(dt.max() * 1e3) if n > 1 else 0.0,
                "ts_nonmonotonic": int((dt < 0).sum()),
                "rssi_mean": float(np.mean(rssi))})

    groups = {}
    for code in np.unique(fc):
        m = fc == code
        name = FRAME_NAMES.get(int(code), f"0x{int(code):02x}")
        a = A[m]
        r = raw[m]
        active = np.setdiff1d(np.arange(256), np.concatenate([GUARD, DC]))
        level = np.median(a[:, active], axis=1) + 1e-9                     # per packet
        guard_ratio = np.median(a[:, GUARD], axis=1) / level
        step = np.abs(np.diff(a[:, active], axis=1)) / (0.5 * (a[:, active][:, 1:] + a[:, active][:, :-1]) + 1e-9)
        same = (r[1:] == r[:-1]).all(axis=1).sum() if len(r) > 1 else 0
        groups[name] = {
            "packets": int(m.sum()),
            "all_zero_csi": int((a.sum(axis=1) == 0).sum()),
            "extreme_int16_values": int(((r >= 32767) | (r <= -32768)).sum()),
            "values_abs_gt_8192": int((np.abs(r.astype(np.int32)) > 8192).sum()),
            "median_level": float(np.median(level)),
            "guard_ratio_median": float(np.median(guard_ratio)),
            "packets_guard_not_null": int((guard_ratio > 0.2).sum()),
            "median_rel_adjacent_step": float(np.median(step)),
            "consecutive_identical_payloads": int(same),
        }
    out["frame_types"] = groups
    return out, None


def main():
    root, out_path = sys.argv[1], sys.argv[2]
    files = sorted(glob.glob(os.path.join(root, "**", "*.pcap"), recursive=True))
    per = {}
    tot = defaultdict(Counter)
    status = Counter()
    for i, p in enumerate(files):
        rel = os.path.relpath(p, root).replace("\\", "/")
        info, _ = scan(p)
        per[rel] = info
        status[info["status"]] += 1
        for k in ("truncated_records", "bad_ip", "bad_udp", "bad_nexmon", "ts_nonmonotonic"):
            tot["integrity"][k] += info.get(k, 0)
        for name, g in info.get("frame_types", {}).items():
            for k in ("packets", "all_zero_csi", "extreme_int16_values", "values_abs_gt_8192",
                      "packets_guard_not_null", "consecutive_identical_payloads"):
                tot[name][k] += g[k]
        if i % 500 == 0:
            print(i, rel, flush=True)

    # distribution of per-file quality metrics for the QoS frames
    q = [v["frame_types"]["qos"] for v in per.values() if "qos" in v.get("frame_types", {})]
    b = [v["frame_types"]["blockack"] for v in per.values() if "blockack" in v.get("frame_types", {})]
    summ = {
        "files_by_status": dict(status),
        "totals": {k: dict(v) for k, v in tot.items()},
        "qos_per_file": {
            "files": len(q),
            "guard_ratio_median_q": np.quantile([g["guard_ratio_median"] for g in q], [0, .5, 1]).tolist(),
            "rel_step_median_q": np.quantile([g["median_rel_adjacent_step"] for g in q], [0, .5, 1]).tolist(),
            "level_median_q": np.quantile([g["median_level"] for g in q], [0, .5, 1]).tolist(),
        },
        "blockack_per_file": {
            "files": len(b),
            "guard_ratio_median_q": np.quantile([g["guard_ratio_median"] for g in b], [0, .5, 1]).tolist(),
            "rel_step_median_q": np.quantile([g["median_rel_adjacent_step"] for g in b], [0, .5, 1]).tolist(),
        },
        "files_without_qos_frames": [k for k, v in per.items() if v.get("status") == "ok" and "qos" not in v["frame_types"]],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summ, "files": per}, f, indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()

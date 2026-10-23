"""
CountFi pcaps - header-level scan of every file + CSI decode on a sample.

Each pcap record observed: Ethernet / IPv4 / UDP 5500 -> Nexmon-CSI payload
  [0:2]  magic 0x1111
  [2]    rssi (int8, dBm)
  [3]    frame-control byte of the sniffed frame
  [4:10] source MAC of the sniffed frame
  [10:12] sequence number (uint16 LE)
  [12:14] core / spatial-stream word (uint16 LE)
  [14:16] chanspec (uint16 LE)
  [16:18] chip version (uint16 LE)
  [18:]  CSI, nfft x 4 bytes
(Layout from the nexmon_csi README; verified field-by-field below.)

A minimal pcap reader is used (no per-packet object allocation) so all
~4.7 M packets can be scanned in a few minutes. CSI values are decoded as
little-endian int16 (real, imag) pairs only for a few packets per file,
which is the bcm43455c0 export format - this assumption is tested by checking
that the decoded spectrum has a clean null/guard structure.

Usage:  py -P -E 08_countfi.py <countfi_root> <out_json>
"""

import glob
import json
import os
import struct
import sys
from collections import Counter, defaultdict

import numpy as np

PCAP_HDR = 24
REC_HDR = 16
UDP_PAYLOAD_OFF = 14 + 20 + 8          # Ethernet + IPv4 (no options) + UDP


def decode_chanspec(cs):
    bw = {0x0800: 10, 0x1000: 20, 0x1800: 40, 0x2000: 80, 0x2800: 160}.get(cs & 0x3800, f"0x{cs & 0x3800:04x}")
    band = "5G" if (cs & 0xC000) == 0xC000 else ("2G" if (cs & 0xC000) == 0 else f"0x{cs & 0xC000:04x}")
    return {"chanspec": f"0x{cs:04x}", "channel": cs & 0xFF, "bw_mhz": bw, "band": band}


def scan_file(path, csi_samples=3):
    size = os.path.getsize(path)
    info = {"bytes": size}
    if size < PCAP_HDR:
        info["status"] = "empty"
        return info, []
    with open(path, "rb") as f:
        gh = f.read(PCAP_HDR)
        magic = struct.unpack("<I", gh[:4])[0]
        if magic not in (0xA1B2C3D4, 0xA1B23C4D):
            info["status"] = f"bad magic 0x{magic:08x}"
            return info, []
        nsec = magic == 0xA1B23C4D
        linktype = struct.unpack("<I", gh[20:24])[0]
        data = f.read()

    ts, rssi, plen, fctl, macs, seqs, cores, chans, chips, bad = [], [], Counter(), Counter(), Counter(), [], Counter(), Counter(), Counter(), 0
    csi = []
    off, n = 0, 0
    while off + REC_HDR <= len(data):
        sec, frac, incl, orig = struct.unpack_from("<IIII", data, off)
        off += REC_HDR
        pkt = data[off:off + incl]
        off += incl
        if len(pkt) < incl:
            info["truncated_last_record"] = True
            break
        n += 1
        ts.append(sec + frac * (1e-9 if nsec else 1e-6))
        p = pkt[UDP_PAYLOAD_OFF:]
        plen[len(p)] += 1
        if len(p) < 18 or p[0:2] != b"\x11\x11":
            bad += 1
            continue
        rssi.append(struct.unpack_from("<b", p, 2)[0])
        fctl[p[3]] += 1
        macs[p[4:10].hex(":")] += 1
        seqs.append(struct.unpack_from("<H", p, 10)[0])
        cores[struct.unpack_from("<H", p, 12)[0]] += 1
        chans[struct.unpack_from("<H", p, 14)[0]] += 1
        chips[struct.unpack_from("<H", p, 16)[0]] += 1
        if len(csi) < csi_samples and n % 300 == 1:
            csi.append(np.frombuffer(p[18:], dtype="<i2").astype(np.float64))

    t = np.asarray(ts)
    dt = np.diff(t) if len(t) > 1 else np.array([np.nan])
    seq = np.asarray(seqs, dtype=np.int64)
    info.update({
        "status": "ok" if n else "header_only",
        "linktype": linktype,
        "packets": n,
        "non_nexmon_packets": bad,
        "start": float(t[0]) if n else None,
        "duration_s": float(t[-1] - t[0]) if n > 1 else 0.0,
        "dt_median_ms": float(np.nanmedian(dt) * 1e3) if n > 1 else None,
        "dt_p95_ms": float(np.nanpercentile(dt, 95) * 1e3) if n > 1 else None,
        "dt_max_ms": float(np.nanmax(dt) * 1e3) if n > 1 else None,
        "ts_nonmonotonic": int((dt < 0).sum()) if n > 1 else 0,
        "payload_lengths": {str(k): v for k, v in plen.items()},
        "rssi_mean": float(np.mean(rssi)) if rssi else None,
        "rssi_range": [int(min(rssi)), int(max(rssi))] if rssi else None,
        "fctl": {f"0x{k:02x}": v for k, v in fctl.items()},
        "src_macs": dict(macs.most_common(4)),
        "n_src_macs": len(macs),
        "seq_unique_frac": float(len(np.unique(seq)) / len(seq)) if len(seq) else None,
        "core_ss": {str(k): v for k, v in cores.items()},
        "chanspec": {f"0x{k:04x}": v for k, v in chans.items()},
        "chip": {f"0x{k:04x}": v for k, v in chips.items()},
    })
    return info, csi


def main():
    root, out_path = sys.argv[1], sys.argv[2]
    files = sorted(glob.glob(os.path.join(root, "**", "*.pcap"), recursive=True))
    per_file = {}
    csi_pool = []
    agg = defaultdict(Counter)

    for i, p in enumerate(files):
        rel = os.path.relpath(p, root).replace("\\", "/")
        info, csi = scan_file(p)
        per_file[rel] = info
        if len(csi_pool) < 400:
            csi_pool.extend(csi)
        for k in ("chanspec", "chip", "core_ss", "fctl", "payload_lengths", "src_macs"):
            for kk, v in info.get(k, {}).items():
                agg[k][kk] += v
        agg["status"][info["status"]] += 1
        if i % 500 == 0:
            print(i, rel, info.get("packets"), flush=True)

    # ---------------- CSI spectrum structure from sampled packets ------------
    c = np.vstack(csi_pool)
    z = c[:, 0::2] + 1j * c[:, 1::2]
    amp = np.abs(z)
    med = np.median(amp, axis=0)
    low = np.where(med < 0.05 * np.median(med))[0].tolist()
    csi_summary = {
        "n_sampled_packets": int(len(z)),
        "n_bins": int(z.shape[1]),
        "int16_range": [float(c.min()), float(c.max())],
        "low_energy_bins": low,
        "n_low_energy_bins": len(low),
        "median_amp_active": float(np.median(med[med >= 0.05 * np.median(med)])),
        "example_packet_bins_0_6": [[float(v.real), float(v.imag)] for v in z[0, :6]],
    }

    for k in ("chanspec",):
        agg["chanspec_decoded"] = {cs: decode_chanspec(int(cs, 16)) for cs in agg[k]}

    result = {"aggregate": {k: dict(v) if isinstance(v, Counter) else v for k, v in agg.items()},
              "csi_sample": csi_summary, "files": per_file}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result["aggregate"], indent=1)[:3000])
    print(json.dumps(csi_summary, indent=1))


if __name__ == "__main__":
    main()

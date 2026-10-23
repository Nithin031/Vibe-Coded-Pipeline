"""
CountFi - per-file features from QoS-data frames only (Block-Ack frames are a
different CSI type and are skipped). One 1-s pcap = one window.

Saves: _work/countfi_features.npz with
  X (files x 7 features), profile (files x 242 mean normalised amplitude, float16),
  rssi_mean, rssi_std, n_qos, count, scen, recv, start, rel

Usage:  py -P -E c3_countfi_features.py <countfi_root> <countfi.json> <out_npz>
"""

import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features_common import KEEP_BINS, window_features  # noqa: E402

ETH, IP, UDP, NEX = 14, 20, 8, 18


def read_qos(path):
    data = open(path, "rb").read()
    off, ts, rssi, pl = 24, [], [], []
    while off + 16 <= len(data):
        sec, frac, incl, orig = struct.unpack_from("<IIII", data, off)
        off += 16
        pkt = data[off:off + incl]
        off += incl
        p = pkt[ETH + IP + UDP:]
        if p[:2] != b"\x11\x11" or len(p) != NEX + 1024 or p[3] != 0x88:
            continue
        ts.append(sec + frac * 1e-6)
        rssi.append(struct.unpack_from("<b", p, 2)[0])
        pl.append(p[NEX:])
    raw = np.frombuffer(b"".join(pl), dtype="<i2").reshape(len(pl), 512)
    A = np.hypot(raw[:, 0::2].astype(np.float32), raw[:, 1::2].astype(np.float32))[:, KEEP_BINS]
    return np.array(ts), np.array(rssi, dtype=np.float32), A


def main():
    root, scan_json, out = sys.argv[1], sys.argv[2], sys.argv[3]
    scan = json.load(open(scan_json, encoding="utf-8"))["files"]
    rels = sorted(k for k, v in scan.items() if v.get("status") == "ok")

    X, prof, rm, rs, nq, cnt, scen, recv, start = [], [], [], [], [], [], [], [], []
    for i, rel in enumerate(rels):
        t, r, A = read_qos(os.path.join(root, rel))
        rate = (len(t) - 1) / (t[-1] - t[0])
        X.append(window_features(A, rate))
        An = A / (np.median(A, axis=1, keepdims=True) + 1e-6)
        prof.append(An.mean(0).astype(np.float16))
        rm.append(r.mean()); rs.append(r.std()); nq.append(len(t))
        _, sc, c, _ = rel.split("/")
        cnt.append(int(c)); recv.append(sc.split("_")[1]); scen.append("_".join(sc.split("_")[2:]))
        start.append(t[0])
        if i % 500 == 0:
            print(i, "/", len(rels), rel, flush=True)

    np.savez_compressed(out, X=np.array(X), profile=np.array(prof), rssi_mean=np.array(rm),
                        rssi_std=np.array(rs), n_qos=np.array(nq), count=np.array(cnt),
                        scen=np.array(scen), recv=np.array(recv), start=np.array(start), rel=np.array(rels))
    print("saved", out, len(rels), "files")


if __name__ == "__main__":
    main()

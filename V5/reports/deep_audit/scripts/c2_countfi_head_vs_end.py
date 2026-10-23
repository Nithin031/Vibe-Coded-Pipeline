"""
CountFi - why do 'head' QoS frames have energy in the VHT80 guard bins (123-133)?
Decodes the same second from head and end (scenario fm, count 10, file 1) and prints
the mean amplitude profile per frame type, plus the low-energy bin lists.

Usage:  py -P -E c2_countfi_head_vs_end.py <countfi_root> <out_json>
"""

import json
import os
import struct
import sys

import numpy as np

ETH, IP, UDP, NEX = 14, 20, 8, 18


def read_pcap(path):
    data = open(path, "rb").read()
    off, ts, fc, pl = 24, [], [], []
    while off + 16 <= len(data):
        sec, frac, incl, orig = struct.unpack_from("<IIII", data, off)
        off += 16
        pkt = data[off:off + incl]
        off += incl
        p = pkt[ETH + IP + UDP:]
        if p[:2] != b"\x11\x11" or len(p) != NEX + 1024:
            continue
        ts.append(sec + frac * 1e-6)
        fc.append(p[3])
        pl.append(p[NEX:])
    raw = np.frombuffer(b"".join(pl), dtype="<i2").reshape(len(pl), 512)
    Z = raw[:, 0::2].astype(np.float32) + 1j * raw[:, 1::2].astype(np.float32)
    return np.array(ts), np.array(fc), Z


def profile(A):
    m = np.median(A, axis=0)
    return m / (np.median(m) + 1e-9)


def low_bins(p, thr=0.1):
    return np.where(p < thr)[0].tolist()


def main():
    root, out = sys.argv[1], sys.argv[2]
    res = {}
    for scen in ["fm", "mm", "rm"]:
        for recv in ["head", "end"]:
            path = os.path.join(root, "48", f"48_{recv}_{scen}", "10" if scen != "mm" else "6", "1.pcap")
            t, fc, Z = read_pcap(path)
            A = np.abs(Z)
            for code, name in [(0x88, "qos"), (0x94, "blockack")]:
                p = profile(A[fc == code])
                key = f"{scen}/{recv}/{name}"
                res[key] = {"n": int((fc == code).sum()), "low_bins": low_bins(p),
                            "profile": np.round(p, 3).tolist()}
                lb = res[key]["low_bins"]
                print(f"{key:20s} n={res[key]['n']:4d}  low bins ({len(lb)}): {lb}")
                # coarse profile: mean per 16-bin block
                print("    16-bin blocks:", np.round(p.reshape(16, 16).mean(1), 2).tolist())
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()

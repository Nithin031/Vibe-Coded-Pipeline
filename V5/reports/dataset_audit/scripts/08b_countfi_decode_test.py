"""
CountFi - decide the CSI word format from evidence.

Two candidate Nexmon exports for a 256-bin (80 MHz) payload of 1024 bytes:
  A) int16 LE (real, imag) pairs           - bcm43455c0 style
  B) packed 32-bit float words             - bcm4358 / bcm4366c0 style
     [sign_r | 11-bit mantissa_r | sign_i | 11-bit mantissa_i | 6-bit exponent]
     (same bit layout as nexmon_csi unpack_float.c, nman=12, nexp=6)

A decode is physically plausible if
  - the low-energy bins form the VHT80 null layout (guards + DC, ~14 bins), and
  - amplitude is smooth across neighbouring active subcarriers
    (high correlation between |H[k]| and |H[k+1]|).

Usage:  py -P -E 08b_countfi_decode_test.py <pcap> [<pcap> ...]
"""

import json
import struct
import sys

import numpy as np

OFF = 14 + 20 + 8 + 18          # Ethernet + IP + UDP + Nexmon header


def read_payloads(path, limit=1000):
    """Return {frame_control_byte: [csi_bytes, ...]} for one pcap."""
    with open(path, "rb") as f:
        data = f.read()
    off, out, n = 24, {}, 0
    while off + 16 <= len(data) and n < limit:
        _, _, incl, _ = struct.unpack_from("<IIII", data, off)
        off += 16
        fctl = data[off + OFF - 15]                 # Nexmon header byte 3
        out.setdefault(f"0x{fctl:02x}", []).append(data[off + OFF: off + incl])
        off += incl
        n += 1
    return out


def decode_int16(p):
    v = np.frombuffer(p, dtype="<i2").astype(np.float64)
    return v[0::2] + 1j * v[1::2]


def decode_float(p):
    w = np.frombuffer(p, dtype="<u4").astype(np.int64)
    e = w & 0x3F
    e = np.where(e >= 32, e - 64, e)
    vi = (w >> 18) & 0x7FF
    vq = (w >> 6) & 0x7FF
    si = np.where(w & (1 << 29), -1.0, 1.0)
    sq = np.where(w & (1 << 17), -1.0, 1.0)
    scale = np.power(2.0, e)
    return si * vi * scale + 1j * sq * vq * scale


def assess(Z):
    amp = np.abs(Z)
    med = np.median(amp, axis=0)
    thr = 0.05 * np.median(med)
    low = np.where(med < thr)[0]
    act = np.where(med >= thr)[0]
    # smoothness ALONG FREQUENCY inside each packet: relative step between
    # adjacent active bins (small for a physical channel response)
    aset = set(act.tolist())
    pairs = np.array([(k, k + 1) for k in act if k + 1 in aset])
    step = np.abs(amp[:, pairs[:, 1]] - amp[:, pairs[:, 0]])
    level = 0.5 * (amp[:, pairs[:, 1]] + amp[:, pairs[:, 0]]) + 1e-9
    rel = float(np.median(step / level))
    return {"packets": int(len(Z)), "n_low_bins": int(len(low)), "low_bins": low.tolist(),
            "median_relative_adjacent_step": round(rel, 4)}


def main():
    res = {}
    for path in sys.argv[1:]:
        groups = read_payloads(path)
        res[path] = {}
        for fc, P in groups.items():
            if len(P) < 20:
                continue
            res[path][fc] = {
                "int16_pairs": assess(np.vstack([decode_int16(p) for p in P])),
                "packed_float": assess(np.vstack([decode_float(p) for p in P])),
            }
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

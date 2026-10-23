"""
IIT-BHU: per-file transmitter-MAC mix and PHY-mode mix.

The receiver logs CSI for any frame it hears, so a file can mix several
transmitters and PHY modes. This pass measures, per file:
  - share of rows from the dominant MAC, number of distinct MACs
  - share of HT (sig_mode=1, 384-value CSI) vs legacy (128-value) rows
  - distinct channel values

Usage:  py -I 04b_iitbhu_macs.py <iit_root> <out_json>
"""

import glob
import json
import os
import sys
from collections import Counter

import numpy as np


def main():
    root, out_path = sys.argv[1], sys.argv[2]
    files = sorted(glob.glob(os.path.join(root, "wifi_dataset_Collected", "*", "*.csv")))
    rows = []
    global_mac = Counter()

    for path in files:
        if os.path.getsize(path) == 0:
            continue
        macs, modes, chans, lens = Counter(), Counter(), Counter(), Counter()
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                head = line.split("[", 1)[0].split(",")
                if len(head) < 25:
                    continue
                macs[head[2]] += 1
                modes[head[5]] += 1
                chans[head[16]] += 1
                lens[head[24]] += 1
        n = sum(macs.values())
        top_mac, top_n = macs.most_common(1)[0]
        global_mac.update({top_mac: 1})
        rows.append({
            "file": os.path.relpath(path, root).replace("\\", "/"),
            "n": n,
            "n_macs": len(macs),
            "dominant_mac": top_mac,
            "dominant_share": round(top_n / n, 4),
            "ht_share": round(modes.get("1", 0) / n, 4),
            "channels": dict(chans),
            "len_field": dict(lens.most_common(3)),
        })

    share = np.array([r["dominant_share"] for r in rows])
    ht = np.array([r["ht_share"] for r in rows])
    summary = {
        "n_files": len(rows),
        "dominant_share_quantiles": {q: float(np.quantile(share, q)) for q in (0, .05, .25, .5, .75, 1)},
        "files_dominant_share_lt_0.9": int((share < .9).sum()),
        "files_dominant_share_lt_0.5": int((share < .5).sum()),
        "ht_share_quantiles": {q: float(np.quantile(ht, q)) for q in (0, .05, .25, .5, .75, 1)},
        "files_mixed_mode_(0.02<ht<0.98)": int(((ht > .02) & (ht < .98)).sum()),
        "files_mostly_legacy": int((ht < .5).sum()),
        "dominant_mac_by_files": dict(global_mac.most_common()),
        "files_with_multiple_channels": int(sum(len(r["channels"]) > 1 for r in rows)),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "files": rows}, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

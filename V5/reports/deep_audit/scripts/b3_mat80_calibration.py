"""
80 MHz set - leave-one-setup-out with two fixes for the per-setup offset:
  raw     : features as they are
  zsetup  : features z-scored per setup using that setup's own windows, labels unused
            (= an unlabelled calibration recording in a new room)
and two window lengths: 1 s (single window) and 10 s (mean of 10 consecutive 1-s windows).

Usage:  py -P -E b3_mat80_calibration.py <mat80_feature_dir> <out_json>
"""

import glob
import json
import os
import sys
import warnings

import lightgbm as lgb
import numpy as np

warnings.filterwarnings("ignore")
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20, verbose=-1, n_jobs=4)


def load(feat_dir, win):
    X, y, s = [], [], []
    for f in sorted(glob.glob(os.path.join(feat_dir, "PC*.npz"))):
        d = np.load(f)
        x = d["X"].mean(1)                              # average over the 4 chains (robust to chain slips)
        n = len(x) // win
        x = x[:n * win].reshape(n, win, -1).mean(1)
        X.append(x); y.append(np.full(n, int(d["count"]))); s.append(np.full(n, str(d["setup"])))
    return np.concatenate(X), np.concatenate(y), np.concatenate(s)


def sc(y, p):
    r = np.clip(np.rint(p), 0, 10)
    return {"exact": round(float((r == y).mean()), 3), "within_1": round(float((abs(r - y) <= 1).mean()), 3),
            "mae": round(float(abs(p - y).mean()), 3)}


def main():
    out = {}
    for win in [1, 10]:
        X, y, s = load(sys.argv[1], win)
        for mode in ["raw", "zsetup"]:
            Z = X.copy()
            if mode == "zsetup":
                for u in np.unique(s):
                    m = s == u
                    Z[m] = (Z[m] - Z[m].mean(0)) / (Z[m].std(0) + 1e-9)
            per, P, Y = {}, [], []
            for u in np.unique(s):
                tr, te = s != u, s == u
                p = lgb.LGBMRegressor(**PARAMS).fit(Z[tr], y[tr]).predict(Z[te])
                per[u] = sc(y[te], p)
                P.append(p); Y.append(y[te])
            key = f"win{win}s_{mode}"
            out[key] = {"pooled": sc(np.concatenate(Y), np.concatenate(P)), "per_setup": per}
            print(key, out[key]["pooled"], {k: v["mae"] for k, v in per.items()})
    json.dump(out, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()

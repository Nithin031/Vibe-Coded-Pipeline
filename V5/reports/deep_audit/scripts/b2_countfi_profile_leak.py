"""
CountFi - does the static channel shape (mean normalised amplitude over 242 tones)
identify the count? Random split vs leave-one-scenario-out, per receiver.
A high random-split score with a collapse on unseen scenarios = the model is
recognising the recording block, not counting people.

Usage:  py -P -E b2_countfi_profile_leak.py <countfi_features.npz> <out_json>
"""

import json
import sys
import warnings

import lightgbm as lgb
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, verbose=-1, n_jobs=4)


def sc(y, p):
    r = np.clip(np.rint(p), y.min(), y.max())
    return {"exact": round(float((r == y).mean()), 3), "within_1": round(float((abs(r - y) <= 1).mean()), 3),
            "mae": round(float(abs(p - y).mean()), 3)}


def main():
    d = np.load(sys.argv[1])
    P = d["profile"].astype(np.float32)
    y, scen, recv = d["count"], d["scen"], d["recv"]
    out = {}
    for r in ["head", "end"]:
        m = recv == r
        X = PCA(20, random_state=0).fit_transform(P[m])
        yr, sr = y[m], scen[m]
        pr = np.zeros(len(yr))
        for a, b in KFold(5, shuffle=True, random_state=0).split(X):
            pr[b] = lgb.LGBMRegressor(**PARAMS).fit(X[a], yr[a]).predict(X[b])
        pc, yc = [], []
        for s in np.unique(sr):
            a, b = sr != s, sr == s
            pc.append(lgb.LGBMRegressor(**PARAMS).fit(X[a], yr[a]).predict(X[b]))
            yc.append(yr[b])
        out[r] = {"random_split": sc(yr, pr), "leave_one_scenario_out": sc(np.concatenate(yc), np.concatenate(pc))}
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

"""
Is people count learnable from the signal, and how much do leaky splits inflate it?

80 MHz set: 1-s windows x 4 chains x 7 features (28 per window)
CountFi   : 1-s files (QoS frames) x 7 features + RSSI mean/std, head and end separately

Experiments (LightGBM, small and fixed; this is an audit check, not a tuned model):
  80 MHz   R  random 5-fold over windows           (leaky: neighbours in train and test)
           T  first 70 % / last 30 % of every file  (same session, later in time)
           S  leave-one-setup-out                    (honest: unseen setup)
  CountFi  R  random 5-fold over files              (leaky)
           C  leave-one-scenario-out                 (honest)
  plus Spearman rank correlation between count and each feature, per setup / scenario,
  and a time-confound test for CountFi (does clock time alone predict count?).

Usage:  py -P -E b1_learnability.py <mat80_feature_dir> <countfi_features.npz> <out_json> <fig_dir>
"""

import glob
import json
import os
import sys
import warnings

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features_common import FEATURE_NAMES  # noqa: E402

warnings.filterwarnings("ignore")
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=4)


# ---------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------
def scores(y, pred_reg):
    p = np.clip(np.rint(pred_reg), y.min(), y.max())
    return {"exact": round(float((p == y).mean()), 3),
            "within_1": round(float((np.abs(p - y) <= 1).mean()), 3),
            "mae": round(float(np.abs(pred_reg - y).mean()), 3),
            "n_test": int(len(y))}


def fit_predict(Xtr, ytr, Xte):
    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def baseline(ytr, yte):
    """Always predict the training mean (what a model with no signal does)."""
    return scores(yte, np.full(len(yte), ytr.mean()))


# ---------------------------------------------------------------
# 80 MHz
# ---------------------------------------------------------------
def run_mat80(feat_dir, fig_dir):
    X, y, setup, pos = [], [], [], []
    for f in sorted(glob.glob(os.path.join(feat_dir, "PC*.npz"))):
        d = np.load(f)
        x = d["X"].reshape(len(d["X"]), -1)                   # 4 chains x 7
        X.append(x)
        y.append(np.full(len(x), int(d["count"])))
        setup.append(np.full(len(x), str(d["setup"])))
        pos.append(np.linspace(0, 1, len(x), endpoint=False))  # position inside the file
    X, y, setup, pos = map(np.concatenate, (X, y, setup, pos))
    out = {"windows": int(len(y)), "per_count": {int(c): int((y == c).sum()) for c in np.unique(y)}}

    # ---- Spearman per setup (file means of chain-averaged features) ----
    rho = {}
    for s in np.unique(setup):
        m = setup == s
        counts = np.unique(y[m])
        fm = np.array([X[m & (y == c)].reshape(-1, 4, 7).mean((0, 1)) for c in counts])
        rho[s] = {FEATURE_NAMES[j]: round(float(spearmanr(counts, fm[:, j])[0]), 2) for j in range(7)}
    out["spearman_per_setup"] = rho

    # ---- R: random 5-fold over windows ----
    pr = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        pr[te] = fit_predict(X[tr], y[tr], X[te])
    out["R_random_windows"] = scores(y, pr)

    # ---- T: first 70 % -> last 30 % of each file ----
    tr, te = pos < 0.7, pos >= 0.7
    out["T_time_split"] = scores(y[te], fit_predict(X[tr], y[tr], X[te]))

    # ---- S: leave one setup out ----
    per, allp, ally = {}, [], []
    for s in np.unique(setup):
        tr, te = setup != s, setup == s
        p = fit_predict(X[tr], y[tr], X[te])
        per[s] = scores(y[te], p)
        per[s]["baseline"] = baseline(y[tr], y[te])
        # whole-recording estimate: median of the window predictions
        per[s]["file_level"] = {int(c): round(float(np.median(p[y[te] == c])), 2) for c in np.unique(y[te])}
        allp.append(p)
        ally.append(y[te])
    out["S_leave_one_setup_out"] = per
    out["S_pooled"] = scores(np.concatenate(ally), np.concatenate(allp))

    # ---- Figure: features vs count per setup ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, j in zip(axes, [0, 2, 4]):
        for s in np.unique(setup):
            m = setup == s
            counts = np.unique(y[m])
            ax.plot(counts, [X[m & (y == c)].reshape(-1, 4, 7)[:, :, j].mean() for c in counts], "o-", label=s)
        ax.set_xlabel("people")
        ax.set_title(FEATURE_NAMES[j])
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("80 MHz set: motion features vs number of people (mean over 1-s windows, 4 chains)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "mat80_features_vs_count.png"), dpi=130)
    plt.close(fig)
    return out


# ---------------------------------------------------------------
# CountFi
# ---------------------------------------------------------------
def run_countfi(npz, fig_dir):
    d = np.load(npz)
    X = np.column_stack([d["X"], d["rssi_mean"], d["rssi_std"]])
    names = FEATURE_NAMES + ["rssi_mean", "rssi_std"]
    y, scen, recv, start = d["count"], d["scen"], d["recv"], d["start"]
    out = {"files": int(len(y))}

    # ---- Spearman per scenario x receiver (folder means) ----
    rho = {}
    for s in np.unique(scen):
        for r in ["head", "end"]:
            m = (scen == s) & (recv == r)
            counts = np.unique(y[m])
            if len(counts) < 4:
                continue
            fm = np.array([X[m & (y == c)].mean(0) for c in counts])
            rho[f"{s}/{r}"] = {names[j]: round(float(spearmanr(counts, fm[:, j])[0]), 2) for j in range(len(names))}
    out["spearman_per_scenario"] = rho

    for r in ["head", "end"]:
        m = recv == r
        Xr, yr, sr, tr_ = X[m], y[m], scen[m], start[m]
        res = {}

        # R: random 5-fold over files
        pr = np.zeros(len(yr))
        for a, b in KFold(5, shuffle=True, random_state=0).split(Xr):
            pr[b] = fit_predict(Xr[a], yr[a], Xr[b])
        res["R_random_files"] = scores(yr, pr)

        # C: leave one scenario out
        per, allp, ally = {}, [], []
        for s in np.unique(sr):
            a, b = sr != s, sr == s
            p = fit_predict(Xr[a], yr[a], Xr[b])
            per[s] = scores(yr[b], p)
            per[s]["baseline"] = baseline(yr[a], yr[b])
            allp.append(p)
            ally.append(yr[b])
        res["C_leave_one_scenario_out"] = per
        res["C_pooled"] = scores(np.concatenate(ally), np.concatenate(allp))

        # Time confound: clock time since the scenario started + scenario id, random split
        t_rel = np.zeros(len(yr))
        for s in np.unique(sr):
            mm = sr == s
            t_rel[mm] = tr_[mm] - tr_[mm].min()
        onehot = (sr[:, None] == np.unique(sr)).astype(float)
        T = np.column_stack([t_rel, onehot])
        pt = np.zeros(len(yr))
        for a, b in KFold(5, shuffle=True, random_state=0).split(T):
            pt[b] = fit_predict(T[a], yr[a], T[b])
        res["time_only_random_files"] = scores(yr, pt)
        out[r] = res

    # ---- Figure ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, j in zip(axes, [0, 2, 7]):
        for s in np.unique(scen):
            m = (scen == s) & (recv == "end")
            counts = np.unique(y[m])
            ax.plot(counts, [X[m & (y == c), j].mean() for c in counts], "o-", label=s)
        ax.set_xlabel("passengers")
        ax.set_title(names[j])
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("CountFi (end receiver, QoS frames): features vs passenger count, per scenario")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "countfi_features_vs_count.png"), dpi=130)
    plt.close(fig)
    return out


def main():
    feat_dir, cf_npz, out_json, fig_dir = sys.argv[1:5]
    os.makedirs(fig_dir, exist_ok=True)
    res = {}
    if glob.glob(os.path.join(feat_dir, "PC*.npz")):
        res["mat80"] = run_mat80(feat_dir, fig_dir)
    if os.path.exists(cf_npz):
        res["countfi"] = run_countfi(cf_npz, fig_dir)
    json.dump(res, open(out_json, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

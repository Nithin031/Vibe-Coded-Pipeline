"""Generate the V4 vs V5 benchmark notebook (Kaggle).

  kaggle_v4_vs_v5_benchmark.ipynb -> kernel wifi-v4-vs-v5-benchmark (CPU, internet on)
        clones V4 (GitHub Nithin031/Vibe-Coded-Pipeline, branch v4) and runs ITS OWN code next to V5
        on the same WiCount packets, the same splits and the same noisy copies.

Inputs: Kaggle dataset wizard1803/wifi-csi (raw WiCount CSVs) and the output of the
wifi-phase5-features kernel (V5 cues of mat80 + WiMANS, needed to train the V5 final model).
Run:  py -3 colab_tools/build_benchmark_nb.py
"""

import ast
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from build_phase12_nbs import to_cell  # noqa: E402
from build_kaggle_nbs import FINDER  # noqa: E402

with open(os.path.join(ROOT, "esp32_counter", "esp32_counter.py"), encoding="utf-8") as f:
    WRITE_COUNTER = "%%writefile /kaggle/working/esp32_counter.py\n" + f.read()


INTRO = r'''# V4 vs V5 - people-counting benchmark on the ESP32-S3 data (Kaggle)

V4 = `Nithin031/Vibe-Coded-Pipeline`, branch `v4` (LightGBM + gap-aware CNN, 3-s windows at 20 Hz, 5-window median,
trained on ESP32 data only). V5 = `esp32_counter.py` (causal 30 s baseline, 5-s windows at 50 Hz, 30 s median;
final model also trained on mat80 + WiMANS, 20 s empty-room calibration).
**V4 runs from its own GitHub code**, so its numbers are not re-implemented.

**Same input for both:** the WiCount packets after V4's own CSV reader + timestamp cleaning
(15 one-minute files, ESP32-S3, one meeting room, 0-3 people).

| Benchmark | Split | Why |
|---|---|---|
| **B1 V4 protocol** | V4's 3 folds: counts 1-2 whole files held out, counts 0 and 3 (one file each) cut into 3 time blocks | V4's published setting (exact 0.84 for its LightGBM) - also checks the harness |
| **B2 new session** | leave one whole file out (15 folds) | closer to deployment: the test session was never seen; with only one 0 and one 3 file, an ESP32-only model has never seen those counts |
| **B3 noise** | B1 folds, test files replaced by V4's synthetic noise (7 kinds x 3 levels + "crazy") | robustness; same noisy packets for both |
| **B4 device** | - | size, time per window, time to first answer, what runs on the chip |

**Systems:** V4 LightGBM, V4 gap-CNN, V5 WiCount-only (same data as V4), V5 final (Phase 4 recipe).

**Fair scoring:** each system with its own shipped smoothing; only windows that both can answer in time are scored
(window end >= 15 s, V5's first output), and the first 30 s of the empty file are left out (V5's install calibration).
Metrics: exact, MAE, within 1, recall per count, empty false alarms, 0/1/2/3+ macro F1, whole-file exact,
and under noise the share of windows still answered.

Output: `/kaggle/working/benchmark/` (`BENCHMARK.md`, CSVs, figures).'''


SETUP = r'''# 1. Setup: V4 from GitHub, WiCount raw data, V5 code
import glob
import json
import os
import subprocess
import sys
import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

''' + FINDER + r'''

REPO = "/kaggle/working/vcp"
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "--depth", "1", "--branch", "v4",
                    "https://github.com/Nithin031/Vibe-Coded-Pipeline.git", REPO], check=True)
V4_COMMIT = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
print("V4 commit:", V4_COMMIT)

hits = find_dirs(lambda d, f: os.path.basename(d) == "data" and "WiCount" in d
                 and os.path.isdir(os.path.join(d, "0")), max_depth=8)
assert hits, "WiCount/data not found - attach wizard1803/wifi-csi"
WICOUNT_DATA = hits[0]

# V4 reads <CSI_DATASETS>/wicount/extracted/data/<count>/*.csv
V4DATA = "/kaggle/working/v4data"
os.makedirs(os.path.join(V4DATA, "wicount", "extracted"), exist_ok=True)
link = os.path.join(V4DATA, "wicount", "extracted", "data")
if not os.path.exists(link):
    os.symlink(WICOUNT_DATA, link)
os.environ["CSI_DATASETS"] = V4DATA

sys.path.insert(0, os.path.join(REPO, "V4", "src"))
import data as v4data
import metrics as v4metrics
import models as v4models
import noise as v4noise
import run_counting as v4count

sys.path.insert(0, "/kaggle/working")
import esp32_counter as ec

OUT = "/kaggle/working/benchmark"
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
SEED = 0
V5_PARAMS = dict(objective="regression", n_estimators=300, learning_rate=0.05, num_leaves=15,
                 min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                 random_state=SEED, verbose=-1)
FIRST_END_S = ec.WARMUP_S + ec.WIN_S           # 15 s: V5's first answer
CAL_END_S = ec.WARMUP_S + ec.CAL_S             # 30 s: end of V5's install calibration
print("WiCount:", WICOUNT_DATA)'''


LOAD = r'''# 2. The same WiCount packets for both: V4 windows (its code) and V5 windows (esp32_counter.py)
wc, wseq, wgap = v4data.load(["wicount"], "standard")
RECS = list(v4data.recordings("wicount"))                    # V4 order (noise seeds follow it)
V4_COLS = v4data.feature_cols(wc)


def v5_windows(rec):
    """V5 device pipeline on one V4 Recording (52 amplitudes per packet)."""
    starts, F = ec.features_from_raw(rec.t, rec.amp)
    d = pd.DataFrame(F, columns=ec.FEATURE_NAMES)
    d["recording"] = rec.meta["recording"]
    d["count"] = int(rec.meta["count"])
    d["window"] = np.arange(len(d))
    d["t_start_s"] = starts / ec.FS
    return d


v5 = pd.concat([v5_windows(r) for r in RECS], ignore_index=True)
print("V4 windows:", len(wc), f"(valid {wc['valid'].mean():.1%}) | V5 windows:", len(v5))
display(pd.DataFrame({"V4": wc.groupby("count")["recording"].nunique(),
                      "V5": v5.groupby("count")["recording"].nunique()}).rename_axis("count"))


# ---- V5 final model also trains on mat80 + WiMANS (Phase 5a cues, base 10 only) ----
def find_input(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert hits, f"{name} not found - attach the wifi-phase5-features output"
    return hits[0]


st = pd.read_parquet(find_input("streams_features.parquet"))
wim = pd.read_parquet(find_input("wimans_clip_features.parquet"))
mat = st[st["dataset"] == "mat80"].copy()
wim["window"], wim["t_start_s"] = 0, 0.0

# install calibration of the extra rooms (as in Phase 4): never trained on
mat["is_cal"] = False
for setup, g in mat[mat["count"] == 0].groupby("setup"):
    rec = sorted(g["recording"].unique())[0]
    mat.loc[(mat["recording"] == rec) & (mat["t_start_s"] + ec.WIN_S <= CAL_END_S + 1e-9), "is_cal"] = True
wim["is_cal"] = False
for setup, g in wim[wim["count"] == 0].groupby("setup"):
    first = sorted(g["session"].unique())[0]
    clips = (g[g["session"] == first].drop_duplicates("recording").sort_values("sample_no")["recording"]
             .head(ec.N_CAL_WINDOWS))
    wim.loc[wim["recording"].isin(clips), "is_cal"] = True

EXTRA_ROWS, EXTRA_OFFSETS = [], {}
for ds, d in [("mat80", mat), ("wimans", wim)]:
    for (setup, view), g in d[d["is_cal"]].groupby(["setup", "view"]):
        EXTRA_OFFSETS[(ds, setup, view)] = ec.calibration_offset(g[ec.FEATURE_NAMES].values)
    rows = d[~d["is_cal"]].assign(dataset=ds)
    EXTRA_ROWS.append(rows[["dataset", "setup", "view", "count"] + ec.FEATURE_NAMES])
EXTRA_ROWS = pd.concat(EXTRA_ROWS, ignore_index=True)
print("V5 extra training rows:", EXTRA_ROWS["dataset"].value_counts().to_dict())'''


SPLITS = r'''# 3. Splits: V4's 3 folds (applied identically to V5's windows) and leave-one-file-out
N_FOLDS = 3
SINGLE = {c for c, g in wc.groupby("count") if g["recording"].nunique() < N_FOLDS}   # counts 0 and 3
BLOCK_END = {rec: g["t0"].max() + 3.0 for rec, g in wc.groupby("recording")}       # V4's block edges


def v4_fold_masks(table, k):
    return v4count.fold_masks(table, k)


def v5_fold_masks(table, k):
    """V4's rule on V5 windows: same held-out files; same time blocks for the single files."""
    train = np.zeros(len(table), bool)
    test = np.zeros(len(table), bool)
    _, v4_test = v4count.fold_masks(wc, k)
    test_recs = set(wc.loc[v4_test & ~wc["count"].isin(SINGLE), "recording"])
    for rec, g in table.groupby("recording"):
        idx = g.index.to_numpy()
        if g["count"].iloc[0] in SINGLE:
            edges = np.linspace(0, BLOCK_END[rec], N_FOLDS + 1)
            t = g["t_start_s"].to_numpy()
            for b in range(N_FOLDS):
                inside = (t >= edges[b]) & (t + ec.WIN_S <= edges[b + 1])
                (test if b == k else train)[idx[inside]] = True
        elif rec in test_recs:
            test[idx] = True
        else:
            train[idx] = True
    return train, test


def lofo_masks(table, rec):
    test = (table["recording"] == rec).to_numpy()
    return ~test, test


PROTOCOLS = {
    "B1_v4_folds": [(f"fold{k}", (lambda t, k=k: v4_fold_masks(t, k)), (lambda t, k=k: v5_fold_masks(t, k)))
                    for k in range(N_FOLDS)],
    "B2_new_session": [(rec, (lambda t, r=rec: lofo_masks(t, r)), (lambda t, r=rec: lofo_masks(t, r)))
                       for rec in sorted(wc["recording"].unique())],
}
for name, folds in PROTOCOLS.items():
    print(name, len(folds), "folds")'''


SYSTEMS = r'''# 4. The four systems: train on a fold, return a predictor (window table -> count estimate per window)
def train_v4(kind, train_mask):
    mask = train_mask & wc["valid"].to_numpy()
    predict, _, _ = v4count.fit_predict(kind, wc, wseq, wgap, mask, f"bench-{kind}")
    return lambda table, seq, gap: predict(table, seq, gap).argmax(1).astype(float)


def v5_offset(table):
    """Install calibration: first 20 s after the warm-up of the empty file."""
    empty = table[(table["count"] == 0) & (table["t_start_s"] + ec.WIN_S <= CAL_END_S + 1e-9)]
    return ec.calibration_offset(empty[ec.FEATURE_NAMES].values) if len(empty) else None


def train_v5(train_table, final):
    """final=False: WiCount only, no calibration. final=True: + mat80 + WiMANS, empty-room calibration."""
    wic = train_table[~((train_table["count"] == 0) & (train_table["t_start_s"] + ec.WIN_S <= CAL_END_S + 1e-9))]
    if not final:
        X, y, w = ec.prepare_features(wic[ec.FEATURE_NAMES].values), wic["count"].values, np.ones(len(wic))
    else:
        off = v5_offset(v5)                                    # the room's install calibration
        parts_X = [ec.prepare_features(wic[ec.FEATURE_NAMES].values, off)]
        parts_y, parts_d = [wic["count"].values], [np.full(len(wic), "wicount")]
        for key, idx in EXTRA_ROWS.groupby(["dataset", "setup", "view"]).indices.items():
            if key in EXTRA_OFFSETS:
                g = EXTRA_ROWS.iloc[idx]
                parts_X.append(ec.prepare_features(g[ec.FEATURE_NAMES].values, EXTRA_OFFSETS[key]))
                parts_y.append(g["count"].values)
                parts_d.append(g["dataset"].values)
        X, y, d = np.concatenate(parts_X), np.concatenate(parts_y), np.concatenate(parts_d)
        w = 1.0 / pd.Series(d).map(pd.Series(d).value_counts()).values     # each dataset the same weight
    reg = lgb.LGBMRegressor(**V5_PARAMS).fit(X, y, sample_weight=w / w.mean())
    max_count = int(y.max())

    def predict(table, offset=None):
        off = (offset if offset is not None else v5_offset(v5)) if final else None
        return np.clip(reg.predict(ec.prepare_features(table[ec.FEATURE_NAMES].values, off)), 0, max_count)
    return predict, reg, max_count


SYSTEM_NAMES = ["V4 LightGBM", "V4 gap-CNN", "V5 WiCount-only", "V5 final"]'''


SCORE = r'''# 5. One scoring for everybody: own smoothing, common time span, same metrics
def smooth_v4(table, pred):
    return v4metrics.smooth(table.reset_index(drop=True), pred)          # 5-window causal median (V4)


def smooth_v5(table, pred):
    d = table[["recording", "window"]].reset_index(drop=True).assign(p=pred)
    return d.groupby("recording")["p"].transform(lambda s: ec.smooth_predictions(s.values)).values


def frame(system, table, pred, t_start, win_s, smoothed):
    return pd.DataFrame({"system": system, "recording": table["recording"].values, "count": table["count"].values,
                         "t_end_s": t_start + win_s, "raw": pred, "smoothed": smoothed})


def common_span(f):
    keep = f["t_end_s"] >= FIRST_END_S - 1e-9
    keep &= ~((f["count"] == 0) & (f["t_end_s"] <= CAL_END_S + 1e-9))
    return f[keep]


def f1_macro(y, r):
    out = []
    for c in sorted(set(y) | set(r)):
        tp = np.sum((y == c) & (r == c))
        out.append(2 * tp / max(2 * tp + np.sum((y != c) & (r == c)) + np.sum((y == c) & (r != c)), 1))
    return float(np.mean(out))


def score(f):
    y = f["count"].values.astype(int)
    r = np.clip(np.rint(f["smoothed"].values), 0, 10).astype(int)
    out = {"windows": len(y), "exact": float(np.mean(r == y)), "mae": float(np.mean(np.abs(r - y))),
           "within1": float(np.mean(np.abs(r - y) <= 1)),
           "f1_4": f1_macro(np.minimum(y, 3), np.minimum(r, 3)),
           "empty_false_alarm": float(np.mean(r[y == 0] > 0)) if (y == 0).any() else np.nan}
    recalls = []
    for c in range(4):
        m = y == c
        out[f"recall_{c}"] = float(np.mean(r[m] == c)) if m.any() else np.nan
        if m.any():
            recalls.append(out[f"recall_{c}"])
    out["balanced_acc"] = float(np.mean(recalls))
    per_file = f.groupby("recording").agg(y=("count", "first"), p=("smoothed", "median"))
    out["file_exact"] = float(np.mean(np.rint(per_file["p"]) == per_file["y"]))
    return out'''


RUN = r'''# 6. B1 + B2: train every system on every fold, predict the held-out windows
FRAMES, FOLD_MODELS = [], {}
t0 = time.time()
for proto, folds in PROTOCOLS.items():
    for fold_name, m4, m5 in folds:
        tr4, te4 = m4(wc)
        tr5, te5 = m5(v5)
        test4 = wc[te4 & wc["valid"].to_numpy()].reset_index(drop=True)
        seq4, gap4 = wseq[te4 & wc["valid"].to_numpy()], wgap[te4 & wc["valid"].to_numpy()]
        test5 = v5[te5].sort_values(["recording", "window"]).reset_index(drop=True)

        models = {"V4 LightGBM": train_v4("lgbm", tr4), "V4 gap-CNN": train_v4("gapcnn", tr4)}
        for name, fn in models.items():
            p = fn(test4, seq4, gap4)
            FRAMES.append(frame(name, test4, p, test4["t0"].values, 3.0, smooth_v4(test4, p))
                          .assign(protocol=proto, fold=fold_name))

        v5_models = {"V5 WiCount-only": train_v5(v5[tr5], final=False),
                     "V5 final": train_v5(v5[tr5], final=True)}
        for name, (fn, _, _) in v5_models.items():
            p = fn(test5)
            FRAMES.append(frame(name, test5, p, test5["t_start_s"].values, ec.WIN_S, smooth_v5(test5, p))
                          .assign(protocol=proto, fold=fold_name))

        if proto == "B1_v4_folds":
            FOLD_MODELS[fold_name] = {**models, **{k: v[0] for k, v in v5_models.items()}}
        print(f"{proto:15s} {fold_name:32s} [{time.time() - t0:.0f} s]", flush=True)

ALL = pd.concat(FRAMES, ignore_index=True)

# ---- harness check: V4 LightGBM on V4's own protocol and windows (published pooled exact 0.84) ----
v4_check = ALL[(ALL["protocol"] == "B1_v4_folds") & (ALL["system"] == "V4 LightGBM")]
print(f"harness check - V4 LightGBM, B1, all V4 windows: exact {score(v4_check)['exact']:.3f} (published 0.84)")

rows = []
for (proto, system), f in ALL.groupby(["protocol", "system"]):
    rows.append({"protocol": proto, "system": system, **score(common_span(f))})
MAIN = pd.DataFrame(rows)
MAIN["system"] = pd.Categorical(MAIN["system"], SYSTEM_NAMES, ordered=True)
MAIN = MAIN.sort_values(["protocol", "system"])
MAIN.round(3)'''


NOISE = r'''# 7. B3: V4's noise suite on the B1 test windows (same noisy packets for V4 and V5)
CONDS = [(k, lv) for k in v4noise.KINDS for lv in ([3] if k == "crazy" else [1, 2, 3])]
clean_answered = ALL[ALL["protocol"] == "B1_v4_folds"].pipe(common_span).groupby("system").size()
noise_rows = []
t0 = time.time()
for kind, level in CONDS:
    noisy = [v4noise.apply(rec, kind, level, seed=2000 + i) for i, rec in enumerate(RECS)]

    # V4 windows of the noisy stream (its code), aligned to the clean windows by window_id
    rows4, sq, gp = [], [], []
    for rec in noisy:
        r, q, g = v4data.make_windows(rec)
        rows4 += r
        sq.append(q)
        gp.append(g)
    nt, ns, ng = pd.DataFrame(rows4), np.concatenate(sq), np.concatenate(gp)
    nv5_parts = [v5_windows(rec) for rec in noisy if len(rec.t) > ec.FS]
    nv5 = pd.concat(nv5_parts, ignore_index=True) if nv5_parts else v5.iloc[:0]
    noisy_offset = v5_offset(nv5)          # calibration in the same conditions (None -> the clean install offset)

    frames = []
    for k in range(N_FOLDS):
        fold = f"fold{k}"
        _, te4 = v4_fold_masks(wc, k)
        pos = v4count.align(wc[te4], nt)
        sub = nt.iloc[pos[pos >= 0]].reset_index(drop=True)
        keep = sub["valid"].to_numpy()                          # V4 answers 'unknown' on bad windows
        sub, s_seq, s_gap = sub[keep].reset_index(drop=True), ns[pos[pos >= 0]][keep], ng[pos[pos >= 0]][keep]
        for name in ["V4 LightGBM", "V4 gap-CNN"]:
            if not len(sub):                                    # every window 'unknown' in this fold
                continue
            p = FOLD_MODELS[fold][name](sub, s_seq, s_gap)
            frames.append(frame(name, sub, p, sub["t0"].values, 3.0, smooth_v4(sub, p)))

        _, te5 = v5_fold_masks(nv5, k)
        sub5 = nv5[te5].sort_values(["recording", "window"]).reset_index(drop=True)
        for name in ["V5 WiCount-only", "V5 final"]:
            if not len(sub5):                                   # every window skipped in this fold
                continue
            fn = FOLD_MODELS[fold][name]
            p = fn(sub5, noisy_offset) if name == "V5 final" else fn(sub5)
            frames.append(frame(name, sub5, p, sub5["t_start_s"].values, ec.WIN_S, smooth_v5(sub5, p)))

    answered = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["system"])
    for system in SYSTEM_NAMES:
        f = common_span(answered[answered["system"] == system])
        row = {"noise": kind, "level": level, "system": system, "answered": len(f) / clean_answered[system]}
        if len(f):                                              # no answer at all -> scores stay empty
            row.update(score(f))
        row["exact_all"] = row.get("exact", 0.0) * min(row["answered"], 1.0)   # 'unknown' / skipped = wrong
        noise_rows.append(row)
    print(f"{kind:17s} level {level}  [{time.time() - t0:.0f} s]", flush=True)

NOISE = pd.DataFrame(noise_rows)
NOISE["system"] = pd.Categorical(NOISE["system"], SYSTEM_NAMES, ordered=True)
noise_summary = NOISE.groupby("system", observed=True)[["exact", "mae", "f1_4", "answered", "exact_all"]].mean()
display(noise_summary.round(3))
NOISE.pivot_table(index=["noise", "level"], columns="system", values="exact", observed=True).round(3)'''


DEVICE = r'''# 8. B4: device side - size, time per window, time to first answer, what runs on the ESP32
# final models on all WiCount (+ extra rooms for V5 final)
all4 = np.ones(len(wc), bool)
v4_lgbm_pred, _, v4_lgbm = v4count.fit_predict("lgbm", wc, wseq, wgap, all4 & wc["valid"].to_numpy(), "all")
_, _, v4_cnn = v4count.fit_predict("gapcnn", wc, wseq, wgap, all4 & wc["valid"].to_numpy(), "all")
_, v5_reg, v5_max = train_v5(v5, final=True)
v5_tree = ec.TreeModel.from_lightgbm(v5_reg.booster_, v5_max)


def lgbm_nodes(booster):
    return sum(2 * t["num_leaves"] - 1 for t in booster.dump_model()["tree_info"])


v4_nodes = sum(lgbm_nodes(m.booster_) for m in v4_lgbm)
v4_cnn_params = sum(p.numel() for p in v4_cnn.parameters())

# time per window on this CPU (preprocessing + features + model), batch size 1
rec = RECS[1]
t0 = time.perf_counter()
rows4, q4, g4 = v4data.make_windows(rec)
v4_feat_ms = (time.perf_counter() - t0) / len(rows4) * 1000
t4 = pd.DataFrame(rows4)
t0 = time.perf_counter()
for i in range(50):
    v4_lgbm_pred(t4.iloc[i % len(t4):i % len(t4) + 1], q4[i % len(t4):i % len(t4) + 1], g4[i % len(t4):i % len(t4) + 1])
v4_lgbm_ms = (time.perf_counter() - t0) / 50 * 1000

t0 = time.perf_counter()
s5, f5 = ec.features_from_raw(rec.t, rec.amp)
v5_feat_ms = (time.perf_counter() - t0) / len(s5) * 1000
x5 = ec.prepare_features(f5)
t0 = time.perf_counter()
for i in range(50):
    v5_tree.predict(x5[i % len(x5):i % len(x5) + 1])
v5_tree_ms = (time.perf_counter() - t0) / 50 * 1000

DEV = pd.DataFrame([
    {"item": "input rate / window / hop", "V4": "20 Hz / 3 s / 1 s", "V5": "50 Hz / 5 s / 2.5 s"},
    {"item": "smoothing (display)", "V4": "median of 5 windows (~5 s)", "V5": "median of 12 windows (30 s)"},
    {"item": "first answer after start", "V4": "~3 s", "V5": f"{FIRST_END_S:.0f} s (10 s warm-up + 5 s window)"},
    {"item": "install step", "V4": "none for counting", "V5": "20 s empty room"},
    {"item": "counting model size", "V4": f"LightGBM 3 seeds x 6 classes: {v4_nodes} nodes "
                                          f"(~{v4_nodes * 11 / 1024:.0f} KB as arrays); gap-CNN {v4_cnn_params} params "
                                          f"(~{v4_cnn_params * 4 / 1024:.0f} KB fp32)",
     "V5": f"{len(v5_tree.roots)} trees, {v5_tree.n_nodes} nodes (~{v5_tree.flash_bytes() / 1024:.0f} KB)"},
    {"item": "CPU time per window (this Kaggle CPU, Python)",
     "V4": f"features {v4_feat_ms:.2f} ms + LightGBM {v4_lgbm_ms:.2f} ms",
     "V5": f"features {v5_feat_ms:.2f} ms + trees {v5_tree_ms:.2f} ms"},
    {"item": "on-chip code", "V4": "none (Python + PyTorch)", "V5": "C header with trees + input prep; cues still Python"},
    {"item": "bad-signal handling", "V4": "window marked 'unknown' (gap > 0.5 s or > 25 % missing)",
     "V5": "window skipped (> 20 % missing); gaps <= 60 ms filled"},
    {"item": "presence model", "V4": "yes, separate (4 presence datasets)", "V5": "no (count 0 = empty)"},
])
DEV'''


REPORT = r'''# 9. Figures + BENCHMARK.md
def md(df):
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```" + chr(10) + df.to_string(index=False) + chr(10) + "```"


MAIN.to_csv(os.path.join(OUT, "main_scores.csv"), index=False)
NOISE.to_csv(os.path.join(OUT, "noise_scores.csv"), index=False)
DEV.to_csv(os.path.join(OUT, "device.csv"), index=False)
ALL.to_csv(os.path.join(OUT, "window_predictions.csv"), index=False)

# ---- main scores ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, metric in zip(axes, ["exact", "mae", "f1_4"]):
    pv = MAIN.pivot_table(index="protocol", columns="system", values=metric, observed=True)
    pv.plot.bar(ax=ax, rot=0)
    ax.set_title(metric)
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figures", "main_scores.png"), dpi=120)
plt.show()

# ---- noise ----
lab = NOISE["noise"] + " " + NOISE["level"].astype(str)
fig, ax = plt.subplots(figsize=(14, 4.5))
for system, g in NOISE.assign(cond=lab).groupby("system", observed=True):
    ax.plot(g["cond"], g["exact"], "o-", label=system)
ax.set_ylabel("exact (common span)")
ax.set_title("B3 - exact count under V4's noise suite")
ax.tick_params(axis="x", rotation=70)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figures", "noise_exact.png"), dpi=120)
plt.show()

# ---- winners per benchmark ----
win_rows = []
for proto, g in MAIN.groupby("protocol"):
    for metric, better in [("exact", max), ("mae", min), ("f1_4", max), ("file_exact", max), ("recall_3", max)]:
        vals = g.set_index("system")[metric].dropna()
        if len(vals):
            best = better(vals)
            win_rows.append({"benchmark": proto, "metric": metric,
                             "best": ", ".join(str(s) for s, v in vals.items() if abs(v - best) < 1e-9),
                             **{str(s): round(float(v), 3) for s, v in vals.items()}})
ns = noise_summary.reset_index()
for metric, better in [("exact", max), ("mae", min), ("answered", max), ("exact_all", max)]:
    vals = ns.set_index("system")[metric].dropna()
    best = better(vals)
    win_rows.append({"benchmark": "B3_noise (mean of 22)", "metric": metric,
                     "best": ", ".join(str(s) for s, v in vals.items() if abs(v - best) < 1e-9),
                     **{str(s): round(float(v), 3) for s, v in vals.items()}})
WIN = pd.DataFrame(win_rows)
WIN.to_csv(os.path.join(OUT, "winners.csv"), index=False)
display(WIN)

show = ["protocol", "system", "windows", "exact", "mae", "within1", "recall_0", "recall_1", "recall_2", "recall_3",
        "empty_false_alarm", "balanced_acc", "f1_4", "file_exact"]
lines = [
    "# V4 vs V5 - people-counting benchmark (ESP32-S3, WiCount)", "",
    f"Generated {time.strftime('%Y-%m-%d %H:%M')} on Kaggle. V4 = GitHub Nithin031/Vibe-Coded-Pipeline branch v4, "
    f"commit `{V4_COMMIT}`, run with its own code. V5 = esp32_counter.py.", "",
    f"Harness check: V4 LightGBM on its own protocol and windows = exact {score(v4_check)['exact']:.3f} "
    "(published 0.84).", "",
    "## Winners", "", md(WIN), "",
    "## B1 (V4 protocol) and B2 (new session, leave one file out)", "",
    md(MAIN[show].round(3)), "",
    "## B3 - noise (mean over 22 conditions)", "",
    md(noise_summary.reset_index().round(3)), "",
    "### Exact count per noise condition", "",
    md(NOISE.pivot_table(index=["noise", "level"], columns="system", values="exact", observed=True)
       .round(3).reset_index()), "",
    "## B4 - device", "", md(DEV), "",
    "## How to read this", "",
    "- Scored windows: window end >= 15 s and outside the first 30 s of the empty file, for every system.",
    "- Each system uses its own display smoothing (V4 ~5 s, V5 30 s); a longer median is steadier but slower.",
    "- B1 tests counts 0 and 3 on time blocks of the SAME recording the other blocks trained on (easy for those counts).",
    "- B2 holds out whole files: an ESP32-only model has no 0- or 3-person training file when that file is tested.",
    "- 'answered' = share of the clean scored windows that still get an answer under noise "
    "(V4 says 'unknown', V5 skips the window). exact / mae are over answered windows only; "
    "exact_all counts every unanswered window as wrong.",
    "- V5 final also trains on mat80 + WiMANS (other rooms, other hardware); V5 WiCount-only uses exactly V4's data.",
    "- Not compared: presence on the V4 presence datasets (V5 has no presence model), V4's WiCount -> OTH cross-room "
    "test (the OTH files are not available).",
]
with open(os.path.join(OUT, "BENCHMARK.md"), "w") as f:
    f.write(chr(10).join(lines))
print("saved", os.path.join(OUT, "BENCHMARK.md"))'''


CLEANUP = r'''# 10. Keep the output small: drop the cloned repo, V4 cache and the symlinked data
import shutil

for path in [REPO, V4DATA]:
    shutil.rmtree(path, ignore_errors=True)
for name in sorted(os.listdir(OUT)):
    print(" ", name)'''


CELLS = [("md", INTRO), ("code", WRITE_COUNTER), ("code", SETUP), ("code", LOAD), ("code", SPLITS),
         ("code", SYSTEMS), ("code", SCORE), ("code", RUN), ("code", NOISE), ("code", DEVICE),
         ("code", REPORT), ("code", CLEANUP)]

FILENAME = "kaggle_v4_vs_v5_benchmark.ipynb"


def main():
    for kind, text in CELLS:
        if kind == "code":
            ast.parse(text.split("\n", 1)[1] if text.startswith("%%writefile") else text)
    nb = {"cells": [to_cell(k, t) for k, t in CELLS],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = os.path.join(ROOT, FILENAME)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out, f"({len(CELLS)} cells)")


if __name__ == "__main__":
    main()

"""Generate the Phase 5 Kaggle notebooks: which upgrades make the ESP32 counter better, on all 5 datasets.

  kaggle_phase5_features.ipynb     -> kernel wifi-phase5-features (CPU)
        mat80 + CountFi + WiCount (Kaggle dataset wizard1803/wifi-csi) -> Phase 1 -> device (causal) cues,
        WiMANS 5 GHz + CSI-Bench Localization -> clip cues; every row has the 10 base + 9 extra cues
  kaggle_phase5_experiments.ipynb  -> kernel wifi-phase5-experiments (CPU, reads the kernel above)
        leave one site out on all 5 datasets: extra cues, 0/1/2/3+ classifier, 30 s memory,
        two-point calibration, 1 / 2 / 3 links averaged

The cells are reused from build_phase12_nbs.py, build_kaggle_nbs.py and build_phase4_nbs.py, and
esp32_counter.py is written into the kernel byte for byte, so every number comes from the same code.
Run:  py -3 colab_tools/build_phase5_nbs.py
"""

import ast
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from build_phase12_nbs import P1_COUNTFI, P1_HELPERS, P1_MAT80, P1_WICOUNT, to_cell  # noqa: E402
from build_kaggle_nbs import (  # noqa: E402
    C_AUDIT_META, C_LOC_CONVERT, C_SETUP, EXTRA_CUES, FINDER,
    W_AUDIT_LABELS, W_CONVERT, W_EXTRA, W_HELPERS, W_SETUP,
)
from build_phase4_nbs import K_CLEANUP, K_LOC_EXPORT, K_LOC_LAYOUT, K_WIMANS_EXPORT  # noqa: E402

with open(os.path.join(ROOT, "esp32_counter", "esp32_counter.py"), encoding="utf-8") as f:
    ESP32_SRC = f.read()

WRITE_COUNTER = "%%writefile /kaggle/working/esp32_counter.py\n" + ESP32_SRC


def patched(cell, old, new):
    """Replace exactly one occurrence, so a changed source cell fails loudly at build time."""
    assert cell.count(old) == 1, f"pattern not found once: {old[:60]}"
    return cell.replace(old, new)


# ===============================================================
# KERNEL A - features
# ===============================================================
A_INTRO = r'''# Phase 5a - all 5 datasets -> device cues (10 base + 9 extra) for the Phase 5 experiments

| Dataset | Source on Kaggle | Cues |
|---|---|---|
| mat80 (80 MHz people counting) | `wizard1803/wifi-csi` `Dataset/` | Phase 1 -> **causal** (device) cues, 5-s windows every 2.5 s, 16 views |
| CountFi | `wizard1803/wifi-csi` `CountFi/` | same, 4 views x 2 receivers |
| WiCount (ESP32-S3) | `wizard1803/wifi-csi` `WiCount/` | same, 1 view |
| WiMANS 5 GHz | `shuokanghuang/wimans` | clip cues (3-s clips, clip-mean normalised), 9 links |
| CSI-Bench Localization | `guozhenjennzhu/csi-bench` | clip cues (5-s clips), 3 devices |

Base cues = the 10 the ESP32 model uses (`esp32_counter.py`, written into this kernel unchanged).
Extra cues = the 9 "how many separate movers" cues from the WiMANS / CSI-Bench notebooks.

Output in `/kaggle/working/`: `streams_features.parquet`, `wimans_clip_features.parquet`,
`csibench_clip_features.parquet`. CPU only, ~1-2 h.'''


A_SETUP = r'''# 1. Setup: find the three local datasets inside wizard1803/wifi-csi
import glob
import json
import os
import shutil
import struct
import sys
import time
import zlib
from multiprocessing import get_context
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

''' + FINDER + r'''


def one_dir(test, what):
    hits = find_dirs(test, max_depth=8)
    if not hits:
        raise FileNotFoundError(f"{what} not found under /kaggle/input - is wizard1803/wifi-csi attached?")
    return hits[0]


MAT80_ROOT = os.path.dirname(one_dir(lambda d, f: os.path.basename(d) == "PC-1a", "Dataset/PC-1a"))
COUNTFI_ROOT = os.path.dirname(one_dir(lambda d, f: os.path.basename(d) == "48"
                                       and "CountFi" in d, "CountFi/.../48"))
WICOUNT_DATA = one_dir(lambda d, f: os.path.basename(d) == "data" and "WiCount" in d
                       and os.path.isdir(os.path.join(d, "0")), "WiCount/data")
print("mat80  :", MAT80_ROOT)
print("CountFi:", COUNTFI_ROOT)
print("WiCount:", WICOUNT_DATA)

OUT_ROOT = "/kaggle/working/shared_format"      # Phase 1 output (temporary)
os.makedirs(OUT_ROOT, exist_ok=True)

# ---- shared-format settings (identical to the laptop Phase 1 and to esp32_counter.py) ----
FS = 50
WIN_S = 5.0
HOP_S = 2.5
MAX_GAP_BINS = 3
MAX_MISSING = 0.20
N_TONES = 52
N_WORKERS = os.cpu_count()
print("cpus:", N_WORKERS)'''


A_MAT80 = patched(P1_MAT80, 'glob.glob(os.path.join(RAW, "Dataset", "PC-*", "*.mat"))',
                  'glob.glob(os.path.join(MAT80_ROOT, "PC-*", "*.mat"))')
A_COUNTFI = patched(P1_COUNTFI, 'countfi_blocks(os.path.join(RAW, "CountFi"))', 'countfi_blocks(COUNTFI_ROOT)')
A_WICOUNT = patched(P1_WICOUNT, 'glob.glob(os.path.join(RAW, "WiCount", "data", "*", "*.csv"))',
                    'glob.glob(os.path.join(WICOUNT_DATA, "*", "*.csv"))')


A_CAUSAL = r'''# 6. Device (causal) cues for mat80 / CountFi / WiCount: 10 base + 9 extra cues per window and view
sys.path.insert(0, "/kaggle/working")
import esp32_counter as ec

''' + EXTRA_CUES + r'''


def causal_rows(path):
    """Same steps as the device: causal 30 s baseline, warm-up, 5 s windows every 2.5 s."""
    with np.load(path) as z:
        S, valid = z["series"], z["valid"]
        views = [str(v) for v in z["view_names"]]
        meta = json.loads(str(z["meta"]))

    rows = []
    for v, view in enumerate(views):
        X = S[v].astype(np.float32) + 1.0          # Phase 1 stored x / recording_mean - 1; the mean cancels
        Z = ec.causal_normalise(X, valid)
        for j, a in enumerate(ec.window_starts(valid)):
            W = Z[a:a + ec.WIN]
            extra, _ = extra_features(W)
            rows.append({"dataset": meta["dataset"], "recording": meta["recording"], "setup": meta["setup"],
                         "receiver": meta["receiver"], "count": int(meta["count"]), "view": view,
                         "window": j, "t_start_s": a / ec.FS,
                         **dict(zip(ec.FEATURE_NAMES, ec.window_features(W))),
                         **dict(zip(EXTRA_NAMES, extra))})
    return rows


npz_paths = sorted(glob.glob(os.path.join(OUT_ROOT, "*", "*.npz")))
print(len(npz_paths), "recordings")
all_rows = []
t0 = time.time()
with get_context("fork").Pool(N_WORKERS) as pool:
    for k, rows in enumerate(pool.imap_unordered(causal_rows, npz_paths)):
        all_rows += rows
        if (k + 1) % 20 == 0:
            print(f"{k + 1:4d} / {len(npz_paths)}  [{time.time() - t0:.0f} s]", flush=True)

streams = pd.DataFrame(all_rows)
streams.to_parquet("/kaggle/working/streams_features.parquet", index=False)
print(len(streams), "rows")
first_view = streams.groupby("dataset")["view"].transform("first")
display(streams[streams["view"] == first_view].groupby(["dataset", "count"])
        .agg(recordings=("recording", "nunique"), windows=("window", "size")))'''


A_WIMANS_EXPORT = patched(
    K_WIMANS_EXPORT,
    'wim[FEATURE_NAMES] = FEATS[idx].reshape(-1, len(FEATURE_NAMES))\n',
    'wim[FEATURE_NAMES] = FEATS[idx].reshape(-1, len(FEATURE_NAMES))\n'
    'wim[EXTRA_NAMES] = EXTRA[idx].reshape(-1, len(EXTRA_NAMES))\n')
A_WIMANS_EXPORT = patched(A_WIMANS_EXPORT, "# 6. WiMANS", "# 11. WiMANS")

A_LOC_EXPORT = patched(
    K_LOC_EXPORT,
    'cb[FEATURE_NAMES] = LOC_FEATS[idx].reshape(-1, len(FEATURE_NAMES))\n',
    'cb[FEATURE_NAMES] = LOC_FEATS[idx].reshape(-1, len(FEATURE_NAMES))\n'
    'cb[EXTRA_NAMES] = LOC_EXTRA[idx].reshape(-1, len(EXTRA_NAMES))\n')

KERNEL_A = [
    ("md", A_INTRO), ("code", A_SETUP), ("code", P1_HELPERS),
    ("code", A_MAT80), ("code", A_COUNTFI), ("code", A_WICOUNT),
    ("code", WRITE_COUNTER), ("code", A_CAUSAL),
    ("code", W_SETUP), ("code", W_AUDIT_LABELS), ("code", W_HELPERS), ("code", W_CONVERT), ("code", W_EXTRA),
    ("code", A_WIMANS_EXPORT),
    ("code", C_SETUP), ("code", C_AUDIT_META), ("code", K_LOC_LAYOUT), ("code", C_LOC_CONVERT),
    ("code", A_LOC_EXPORT), ("code", K_CLEANUP),
]


# ===============================================================
# KERNEL B - experiments
# ===============================================================
B_INTRO = r'''# Phase 5 - which upgrades make the ESP32 counter better? (all 5 datasets, leave one site out)

Starting point = the Phase 4 final model: LightGBM regression on the 10 base cues, trained on
WiCount + mat80 + WiMANS (+ CSI-Bench where usable), `empty_cal` = 20 s empty room at install.

| Test | What changes | Needs on the device |
|---|---|---|
| `extra_reg` | + 9 "separate movers" cues | 9 more cues in the C code |
| `base_clf4` | predicts the 0 / 1 / 2 / 3+ class directly (4-class LightGBM) | multi-class tree export |
| `mem_reg` | + 30 s memory: mean / std of the last 12 windows' motion cues (streams only) | a 12-window buffer |
| two-point calibration | install = 20 s empty room **+ 20 s with 2 people**; estimates are stretched so empty -> 0 and 2 -> 2 | one extra install step |
| 2 / 3 links | 2 or 3 receivers, each with its own counter + calibration, estimates averaged | more ESP32 boards |

**Datasets (all 5):** mat80 (router 80 MHz, 0-10), WiCount (ESP32-S3, 0-3), WiMANS 5 GHz (0-5), CSI-Bench
Localization (1-2), CountFi (1-10, never trained on). Every site is scored by a model that never saw it
(mat80 setup, WiMANS / CSI-Bench room, WiCount file, CountFi = whole dataset).

**Links per dataset** (what "2 receivers" means here): mat80 = different antennas of one router (same sub-band),
WiMANS = different receive antennas, **CSI-Bench = different devices**, **CountFi = the two receivers**,
WiCount = one link only. Antennas of one device sit close together, so mat80 / WiMANS give a lower bound
for the gain of two ESP32 boards on different walls.

Output: `/kaggle/working/phase5/` (`PHASE5_RESULTS.md`, CSV tables, figures).
No new model is exported: every upgrade here needs a change to `esp32_counter.py` / the firmware first,
so this run decides which changes are worth making.'''


B_SETUP = r'''# 1. Setup: load the Phase 5a cue tables
import glob
import itertools
import json
import os
import sys
import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_input(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not hits:
        raise FileNotFoundError(f"{name} not found - attach the output of the wifi-phase5-features kernel")
    return hits[0]


streams = pd.read_parquet(find_input("streams_features.parquet"))
wimans = pd.read_parquet(find_input("wimans_clip_features.parquet"))
csib = pd.read_parquet(find_input("csibench_clip_features.parquet"))

sys.path.insert(0, "/kaggle/working")
import esp32_counter as ec

OUT = "/kaggle/working/phase5"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

BASE = list(ec.FEATURE_NAMES)
EXTRA = ["pc2_frac", "pc3_frac", "eig_entropy", "doppler_centroid", "doppler_spread", "psd_entropy",
         "activity_cv", "tone_energy_cv", "half_band_corr"]
MEM = ["mem_cv_mean", "mem_cv_std", "mem_diff_mean", "mem_diff_std"]
LOG_COLS = list(ec.LOG_FEATURES)                    # log() before the offset, as on the device
MODES = ["zero_shot", "empty_cal"]
CAL_COUNT = 2                                       # second point of the two-point calibration
SEED = 0

GBM_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
                  verbose=-1, n_jobs=os.cpu_count())

for name, d in [("streams", streams), ("wimans", wimans), ("csibench", csib)]:
    print(f"{name:9s} {d.shape}  datasets={sorted(d['dataset'].unique())}")'''


B_DATA = r'''# 2. One table, the 30 s memory cues, links, and the calibration windows
clips = pd.concat([wimans, csib], ignore_index=True)
clips["receiver"] = clips["view"]
clips["window"] = 0
clips["t_start_s"] = 0.0
streams["session"] = streams["recording"]
streams["sample_no"] = 0

cols = ["dataset", "recording", "setup", "receiver", "session", "sample_no", "count", "view", "window",
        "t_start_s"] + BASE + EXTRA
DATA = pd.concat([streams[cols], clips[cols]], ignore_index=True)
DATA["stream"] = DATA["dataset"].isin(["mat80", "wicount", "countfi"])

# ---- 30 s memory: mean / std of the last 12 windows (only the past) per recording + view ----
DATA = DATA.sort_values(["dataset", "recording", "view", "window"]).reset_index(drop=True)
grp = DATA.groupby(["dataset", "recording", "view"])
for src, name in [("cv_median", "cv"), ("diff_energy", "diff")]:
    roll = grp[src].rolling(ec.SMOOTH_N, min_periods=1)
    DATA[f"mem_{name}_mean"] = roll.mean().reset_index(level=[0, 1, 2], drop=True)
    DATA[f"mem_{name}_std"] = roll.std().reset_index(level=[0, 1, 2], drop=True).fillna(0.0)
DATA.loc[~DATA["stream"], MEM] = np.nan             # clips have no past

# ---- links: which rows are 'the same moment seen by different receivers' ----
#   mat80: antennas of one sub-band | WiMANS: receive antennas of one transmit antenna
#   CSI-Bench: the 3 devices | CountFi: the 2 receivers (same scenario, count, sub-band, window index)
v = DATA["view"].astype(str)
DATA["link"] = "0"
DATA["moment"] = DATA["recording"] + "|" + DATA["window"].astype(str)
m = DATA["dataset"] == "mat80"
DATA.loc[m, "link"] = v[m].str.split("_").str[0]
DATA.loc[m, "moment"] = DATA["recording"][m] + "|" + v[m].str.split("_").str[1] + "|" + DATA["window"][m].astype(str)
m = DATA["dataset"] == "wimans"
DATA.loc[m, "link"] = v[m].str.split("_").str[1]
DATA.loc[m, "moment"] = DATA["recording"][m] + "|" + v[m].str.split("_").str[0]
m = DATA["dataset"] == "csibench"
DATA.loc[m, "link"] = v[m]
DATA.loc[m, "moment"] = DATA["recording"][m]
m = DATA["dataset"] == "countfi"
DATA.loc[m, "link"] = DATA["receiver"][m]
DATA.loc[m, "moment"] = (DATA["setup"][m] + "|" + DATA["count"][m].astype(str) + "|" + v[m]
                         + "|" + DATA["window"][m].astype(str))

# ---- calibration windows (never trained on, never scored) ----
#   streams: first 20 s after the warm-up of the first recording with that count in the setup
#   clips:   first 7 clips of the first session with that count in the room
CAL_END = ec.WARMUP_S + ec.CAL_S + 1e-9


def mark_cal(count):
    mask = np.zeros(len(DATA), dtype=bool)
    for (ds, setup), g in DATA[DATA["count"] == count].groupby(["dataset", "setup"]):
        if g["stream"].iloc[0]:
            rec = sorted(g["recording"].unique())[0]
            mask |= ((DATA["dataset"] == ds) & (DATA["recording"] == rec)
                     & (DATA["t_start_s"] + ec.WIN_S <= CAL_END)).values
        else:
            first = sorted(g["session"].unique())[0]
            recs = (g[g["session"] == first].drop_duplicates("recording")
                    .sort_values("sample_no")["recording"].head(ec.N_CAL_WINDOWS))
            mask |= ((DATA["dataset"] == ds) & DATA["recording"].isin(recs)).values
    return mask


DATA["cal"] = ""
DATA.loc[mark_cal(0), "cal"] = "empty"
DATA.loc[mark_cal(CAL_COUNT) & (DATA["cal"] == ""), "cal"] = "two"
DATA["site"] = np.where(DATA["dataset"] == "wicount", DATA["recording"], DATA["setup"])

display(pd.crosstab(DATA["dataset"], DATA["cal"]))
display(DATA.drop_duplicates(["dataset", "recording"]).pivot_table(
    index="dataset", columns="count", values="recording", aggfunc="count", fill_value=0))'''


B_HELPERS = r'''# 3. Model input (log + empty-room offset, as on the device), training, scoring
def prep(F, cols, offset=None):
    T = np.nan_to_num(np.asarray(F, dtype=np.float64), nan=0.0).copy()
    for k, c in enumerate(cols):
        if c in LOG_COLS:
            T[:, k] = np.log(np.maximum(T[:, k], 0.0) + ec.EPS)
    if offset is not None:
        T = T - offset
    return T.astype(np.float32)


def offsets_for(cols):
    """Empty-room offset per (dataset, setup, view) = mean model input over the empty calibration windows."""
    cal = DATA[DATA["cal"] == "empty"]
    return {key: prep(g[cols].values, cols).astype(np.float64).mean(axis=0)
            for key, g in cal.groupby(["dataset", "setup", "view"])}


OFFSETS = {}


def model_inputs(df, cols, mode):
    df = df.reset_index(drop=True)
    if mode == "zero_shot":
        return prep(df[cols].values, cols), np.ones(len(df), dtype=bool)
    key_cols = tuple(cols)
    if key_cols not in OFFSETS:
        OFFSETS[key_cols] = offsets_for(cols)
    off = OFFSETS[key_cols]
    X = np.zeros((len(df), len(cols)), dtype=np.float32)
    ok = np.zeros(len(df), dtype=bool)
    for key, idx in df.groupby(["dataset", "setup", "view"]).indices.items():
        if key in off:
            X[idx] = prep(df[cols].values[idx], cols, off[key])
            ok[idx] = True
    return X, ok


def fit(train_df, cols, mode, task):
    X, ok = model_inputs(train_df, cols, mode)
    d = train_df.reset_index(drop=True)[ok]
    w = 1.0 / d.groupby("dataset")["dataset"].transform("size").values     # each dataset the same weight
    w = w / w.mean()
    if task == "clf4":
        model = lgb.LGBMClassifier(objective="multiclass", **GBM_PARAMS)
        model.fit(X[ok], np.minimum(d["count"].values, 3), sample_weight=w)
        return model, 3
    model = lgb.LGBMRegressor(objective="regression", **GBM_PARAMS)
    model.fit(X[ok], d["count"].values, sample_weight=w)
    return model, int(d["count"].max())


def predict(model, max_count, df, cols, mode):
    X, ok = model_inputs(df, cols, mode)
    p = np.full(len(df), np.nan)
    if ok.any():
        p[ok] = np.clip(model.predict(X[ok]), 0, max_count)
    return p


def f1_macro(y, r):
    labels = sorted(set(y) | set(r))
    out = []
    for c in labels:
        tp = np.sum((y == c) & (r == c))
        out.append(2 * tp / max(2 * tp + np.sum((y != c) & (r == c)) + np.sum((y == c) & (r != c)), 1))
    return float(np.mean(out))


def score(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    r = np.rint(pred)
    y4, r4 = np.minimum(y, 3), np.minimum(r, 3)
    mae_const = float(np.mean(np.abs(y - y.mean())))
    mae = float(np.mean(np.abs(pred - y)))
    return {"n": len(y), "mae": mae, "rel_mae": mae / max(mae_const, 1e-9),
            "exact": float(np.mean(r == y)), "within1": float(np.mean(np.abs(r - y) <= 1)),
            "acc4": float(np.mean(r4 == y4)), "f1_4": f1_macro(y4, r4),
            "empty_recall": float(np.mean(r[y == 0] == 0)) if (y == 0).any() else np.nan}'''


B_RUN = r'''# 4. Leave one site out for every variant and mode (predictions also on each site's calibration windows)
POOL_ALL = ["wicount", "mat80", "wimans", "csibench"]          # = Phase 4 final pool
POOL_STREAMS = ["wicount", "mat80"]                            # the memory cues exist only for streams
VARIANTS = {
    "base_reg":         dict(cols=BASE, task="reg", pool=POOL_ALL),            # = Phase 4 final model
    "extra_reg":        dict(cols=BASE + EXTRA, task="reg", pool=POOL_ALL),
    "base_clf4":        dict(cols=BASE, task="clf4", pool=POOL_ALL),
    "base_reg_streams": dict(cols=BASE, task="reg", pool=POOL_STREAMS),        # reference for mem_reg
    "mem_reg":          dict(cols=BASE + MEM, task="reg", pool=POOL_STREAMS),
}
ALL_DS = ["mat80", "wicount", "wimans", "csibench", "countfi"]

TRAIN = DATA[DATA["cal"] == ""]
CAL = DATA[DATA["cal"] != ""]
KEEP = ["dataset", "recording", "setup", "site", "receiver", "count", "view", "window", "link", "moment", "cal"]
parts = []
t0 = time.time()

for vname, cfg in VARIANTS.items():
    cols, task, pool = cfg["cols"], cfg["task"], cfg["pool"]
    test_ds = ALL_DS if vname not in ("mem_reg", "base_reg_streams") else ["mat80", "wicount", "countfi"]
    pool_df = TRAIN[TRAIN["dataset"].isin(pool)]

    for mode in MODES:
        full = fit(pool_df, cols, mode, task)
        sites = (TRAIN[TRAIN["dataset"].isin(test_ds)][["dataset", "site"]]
                 .drop_duplicates().values.tolist())

        for ds, site in sites:
            if ds in pool:
                held = (pool_df["dataset"] == ds) & (pool_df["site"] == site)
                model, mc = fit(pool_df[~held], cols, mode, task)
            else:
                model, mc = full
            test = TRAIN[(TRAIN["dataset"] == ds) & (TRAIN["site"] == site)]
            setups = test["setup"].unique()
            cal = CAL[(CAL["dataset"] == ds) & CAL["setup"].isin(setups)]
            both = pd.concat([test, cal])
            d = both[KEEP].copy()
            d["raw"] = predict(model, mc, both, cols, mode)
            d["fold"] = site
            d["variant"], d["mode"], d["max_count"] = vname, mode, mc
            parts.append(d[~d["raw"].isna()])

        print(f"{vname:17s} {mode:9s} {len(sites):3d} sites  [{time.time() - t0:.0f} s]", flush=True)

P = pd.concat(parts, ignore_index=True)
P = P.sort_values(["variant", "mode", "fold", "recording", "view", "window"]).reset_index(drop=True)

# ---- the display value: causal median of the last 30 s per stream (clips: one estimate each) ----
P["smoothed"] = P.groupby(["variant", "mode", "fold", "recording", "view"])["raw"].transform(
    lambda s: ec.smooth_predictions(s.values))
print(len(P), "predictions")'''


B_TWO_POINT = r'''# 5. Two-point calibration: empty -> 0 and the 2-people calibration -> 2 (per fold, site and link)
cal_pts = (P[P["cal"] != ""].groupby(["variant", "mode", "fold", "dataset", "setup", "view", "cal"])["smoothed"]
           .median().unstack("cal"))
cal_pts = cal_pts.rename(columns={"empty": "p0", "two": "p2"}).reset_index()
cal_pts = cal_pts.dropna(subset=["p0", "p2"])
cal_pts = cal_pts[cal_pts["p2"] - cal_pts["p0"] > 0.2]          # flat response -> no stretch possible

S = P[P["cal"] == ""].merge(cal_pts, on=["variant", "mode", "fold", "dataset", "setup", "view"], how="left")
S["two_point"] = np.clip((S["smoothed"] - S["p0"]) * CAL_COUNT / (S["p2"] - S["p0"]), 0, S["max_count"])
S.loc[S["variant"] == "base_clf4", "two_point"] = np.nan        # a class cannot be stretched
print("rows with a two-point calibration:", int(S["two_point"].notna().sum()), "/", len(S))
display(cal_pts.groupby(["variant", "mode", "dataset"]).size().rename("calibrated links").unstack(0))'''


B_LINKS = r'''# 6. Score: single link and 2 / 3 links averaged, without / with the two-point calibration
MAX_COMBOS = 12
rows = []


def link_combos(links, k):
    combos = list(itertools.combinations(sorted(links), k))
    return combos[:MAX_COMBOS]


for (vname, mode, ds), g in S.groupby(["variant", "mode", "dataset"]):
    links = g["link"].unique()
    for value in ["smoothed", "two_point"]:
        h0 = g[g[value].notna()]
        if h0.empty:
            continue
        # k links: average over every k-link combination, only moments where all k links have a value
        for k in [1, 2, 3]:
            if k > len(links):
                continue
            ys, ps = [], []
            for combo in link_combos(links, k):
                h = h0[h0["link"].isin(combo)]
                agg = h.groupby("moment").agg(y=("count", "first"), p=(value, "mean"), n=("link", "nunique"))
                agg = agg[agg["n"] == k]
                ys.append(agg["y"].values)
                ps.append(agg["p"].values)
            if not ys or not len(np.concatenate(ys)):
                continue
            rows.append({"variant": vname, "mode": mode, "dataset": ds, "calibration": value,
                         "links": k, **score(np.concatenate(ys), np.concatenate(ps))})

R = pd.DataFrame(rows)
R["calibration"] = R["calibration"].map({"smoothed": "none / empty only", "two_point": "two-point"})
R.to_csv(os.path.join(OUT, "all_scores.csv"), index=False)
show = ["variant", "mode", "dataset", "calibration", "links", "n", "mae", "rel_mae", "exact", "within1",
        "acc4", "f1_4", "empty_recall"]
R[R["links"] == 1][show].round(3)'''


B_COMPARE = r'''# 7. Each upgrade against its baseline, on exactly the same windows
def paired(rows, value_a, value_b, by, cap3=False):
    """Score two prediction columns on the rows where both exist (cap3: on the 0/1/2/3+ scale)."""
    out = []
    for key, g in rows.groupby(by):
        h = g.dropna(subset=[value_a, value_b])
        if h.empty:
            continue
        y, a, b = h["count"], h[value_a], h[value_b]
        if cap3:
            y, a, b = np.minimum(y, 3), np.minimum(a, 3), np.minimum(b, 3)
        sa, sb = score(y, a), score(y, b)
        out.append({**dict(zip(by, key)), "n": len(h),
                    "mae_base": sa["mae"], "mae_new": sb["mae"], "d_mae": sb["mae"] - sa["mae"],
                    "f1_4_base": sa["f1_4"], "f1_4_new": sb["f1_4"], "d_f1_4": sb["f1_4"] - sa["f1_4"]})
    return pd.DataFrame(out)


single = S[S["cal"] == ""]
key = ["mode", "fold", "dataset", "recording", "view", "window"]
wide = single.pivot_table(index=key + ["count"], columns="variant", values="smoothed").reset_index()

comparisons = []
for new, base in [("extra_reg", "base_reg"), ("base_clf4", "base_reg"), ("mem_reg", "base_reg_streams")]:
    if new in wide and base in wide:
        cap3 = new == "base_clf4"                  # the classifier only knows 0 / 1 / 2 / 3+
        c = paired(wide, base, new, ["mode", "dataset"], cap3=cap3)
        c.insert(0, "upgrade", f"{new} vs {base}" + (" (0/1/2/3+ scale)" if cap3 else ""))
        comparisons.append(c)

# two-point vs none, base model
tp = S[(S["variant"] == "base_reg")]
c = paired(tp, "smoothed", "two_point", ["mode", "dataset"])
c.insert(0, "upgrade", "two-point calibration vs none")
comparisons.append(c)

# 2 / 3 links vs 1 link, base model (from the link table)
lk = R[(R["variant"] == "base_reg")].pivot_table(index=["mode", "dataset", "calibration"], columns="links",
                                                  values="mae").reset_index()
for k in [2, 3]:
    if k in lk:
        c = lk[["mode", "dataset", "calibration", 1, k]].dropna().rename(columns={1: "mae_base", k: "mae_new"})
        c["d_mae"] = c["mae_new"] - c["mae_base"]
        c.insert(0, "upgrade", f"{k} links vs 1 link (base_reg)")
        comparisons.append(c)

CMP = pd.concat(comparisons, ignore_index=True)
CMP.to_csv(os.path.join(OUT, "upgrades_vs_baseline.csv"), index=False)
CMP.round(3)'''


B_FIGS = r'''# 8. Figures: error vs number of links; mean estimate per true count with / without two-point
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, cal in zip(axes, ["none / empty only", "two-point"]):
    h = R[(R["variant"] == "base_reg") & (R["mode"] == "empty_cal") & (R["calibration"] == cal)]
    for ds, g in h.groupby("dataset"):
        ax.plot(g["links"], g["mae"], "o-", label=ds)
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("links averaged")
    ax.set_ylabel("MAE (people)")
    ax.set_title(f"base model, empty_cal, calibration: {cal}")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "mae_vs_links.png"), dpi=120)
plt.show()

b = S[(S["variant"] == "base_reg") & (S["mode"] == "empty_cal")]
per_count = (b.groupby(["dataset", "count"])
             .agg(n=("smoothed", "size"), mean_none=("smoothed", "mean"), mean_two_point=("two_point", "mean"))
             .reset_index())
per_count.to_csv(os.path.join(OUT, "per_count_two_point.csv"), index=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for ax, col, title in zip(axes, ["mean_none", "mean_two_point"], ["empty room only", "two-point"]):
    for ds, g in per_count.groupby("dataset"):
        ax.plot(g["count"], g[col], "o-", label=ds)
    top = int(per_count["count"].max())
    ax.plot([0, top], [0, top], "k--", lw=0.8)
    ax.set_xlabel("true people")
    ax.set_title(f"mean estimate, one link - {title}")
    ax.legend(fontsize=8)
axes[0].set_ylabel("mean estimate")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "per_count_two_point.png"), dpi=120)
plt.show()
per_count.round(2)'''


B_REPORT = r'''# 9. Verdicts + PHASE5_RESULTS.md
def md(df):
    try:
        return df.to_markdown(index=False)
    except ImportError:                              # tabulate missing
        return "```" + chr(10) + df.to_string(index=False) + chr(10) + "```"


# An upgrade 'helps' when it lowers the MAE by >= 0.05 people on at least 2 datasets
# and does not make the ESP32 data (WiCount) worse by more than 0.05.
verdicts = []
for up, g in CMP.groupby("upgrade"):
    for mode, h in g.groupby("mode"):
        if "calibration" in h and h["calibration"].notna().any():
            h = h[h["calibration"] == "none / empty only"]
        better = h.loc[h["d_mae"] <= -0.05, "dataset"].tolist()
        worse = h.loc[h["d_mae"] >= 0.05, "dataset"].tolist()
        wic = h.loc[h["dataset"] == "wicount", "d_mae"]
        wic_ok = wic.empty or float(wic.iloc[0]) < 0.05
        helps = len(better) >= 2 and wic_ok
        verdicts.append({"upgrade": up, "mode": mode, "helps": helps, "better_on": ", ".join(better),
                         "worse_on": ", ".join(worse),
                         "wicount_d_mae": None if wic.empty else round(float(wic.iloc[0]), 3)})
V = pd.DataFrame(verdicts)
V.to_csv(os.path.join(OUT, "verdicts.csv"), index=False)
display(V)

base1 = R[(R["variant"] == "base_reg") & (R["links"] == 1)]
lines = [
    "# Phase 5 results - which upgrades make the ESP32 counter better", "",
    f"Generated {time.strftime('%Y-%m-%d %H:%M')} on Kaggle. Leave one site out on all 5 datasets; "
    "stream datasets scored after the 30 s median, clips one estimate each.", "",
    "Rule: an upgrade **helps** when it lowers the MAE by >= 0.05 people on at least 2 datasets and does not make "
    "WiCount (ESP32) worse by more than 0.05.", "",
    "## Verdicts", "", md(V), "",
    "## Baseline = Phase 4 model (base_reg), one link", "",
    md(base1[show].round(3)), "",
    "## Every upgrade against its baseline (same windows)", "",
    md(CMP.round(3)), "",
    "## Links: 1 / 2 / 3 averaged (base_reg)", "",
    md(R[R["variant"] == "base_reg"][show].round(3)), "",
    "## Mean estimate per true count, empty_cal, one link: empty room only vs two-point", "",
    md(per_count.round(2)), "",
    "## All variants, one link", "",
    md(R[R["links"] == 1][show].round(3)), "",
    "## Notes", "",
    "- Links: mat80 = antennas of one router, WiMANS = receive antennas of one card (both close together, so a lower "
    "bound for 2 boards on different walls); CSI-Bench = 3 devices; CountFi = 2 receivers matched by window index.",
    "- Two-point calibration uses 20 s of the empty room and 20 s with 2 people of the same site; those windows are "
    "never trained on or scored. Not possible on CountFi (no empty) or CSI-Bench (no empty).",
    "- CountFi is never trained on; each count there is one time block (stress test only).",
    "- WiMANS / CSI-Bench clips are clip-mean normalised; they have no 30 s memory, so mem_reg is streams only.",
    "- No model is exported: each upgrade needs esp32_counter.py / firmware changes first.",
]
with open(os.path.join(OUT, "PHASE5_RESULTS.md"), "w") as f:
    f.write(chr(10).join(lines))
print("saved", os.path.join(OUT, "PHASE5_RESULTS.md"))
for name in sorted(os.listdir(OUT)):
    print(" ", name)'''


KERNEL_B = [
    ("md", B_INTRO), ("code", WRITE_COUNTER), ("code", B_SETUP),
    ("code", B_DATA), ("code", B_HELPERS), ("code", B_RUN), ("code", B_TWO_POINT), ("code", B_LINKS),
    ("code", B_COMPARE), ("code", B_FIGS), ("code", B_REPORT),
]


# ===============================================================
# Notebook writer
# ===============================================================
NOTEBOOKS = {
    "kaggle_phase5_features.ipynb": KERNEL_A,
    "kaggle_phase5_experiments.ipynb": KERNEL_B,
}


def check(text):
    """ast-check a code cell; a %%writefile cell is checked as the Python file it writes."""
    if text.startswith("%%writefile"):
        text = text.split("\n", 1)[1]
    ast.parse(text)


def main():
    for filename, cells in NOTEBOOKS.items():
        for kind, text in cells:
            if kind == "code":
                check(text)
        nb = {"cells": [to_cell(k, t) for k, t in cells],
              "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                           "language_info": {"name": "python"}},
              "nbformat": 4, "nbformat_minor": 5}
        out = os.path.join(ROOT, filename)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print("wrote", out, f"({len(cells)} cells)")


if __name__ == "__main__":
    main()

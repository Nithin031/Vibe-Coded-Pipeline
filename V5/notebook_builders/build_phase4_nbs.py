"""Generate the Phase 4 notebooks: one final ESP32-S3 model from every labelled source.

  kaggle_phase4_features.ipynb -> Kaggle: WiMANS (5 GHz) + CSI-Bench Localization -> one feature row
                                  per clip x link (the same 10 cues as esp32_counter.py)
  phase4_final_local.ipynb     -> laptop: pools mat80 + WiCount + WiMANS + CSI-Bench, leave one site out
                                  for every training pool, picks the final model, exports the ESP32 bundle

The Kaggle cells are reused from build_kaggle_nbs.py and the laptop cells from build_final_nb.py,
so every dataset goes through exactly the code that produced the earlier results.
Run:  py -3 colab_tools/build_phase4_nbs.py
"""

import ast
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from build_phase12_nbs import to_cell  # noqa: E402
from build_kaggle_nbs import (  # noqa: E402
    C_AUDIT_META, C_LOC_CONVERT, C_LOC_LAYOUT, C_SETUP,
    W_AUDIT_LABELS, W_CONVERT, W_HELPERS, W_SETUP,
    syntax_check,
)
from build_final_nb import (  # noqa: E402
    BUDGET, CHECKS, DONE, FEATURES_CELL, HELPERS, REPLAY, TESTS,
)


# ===============================================================
# KAGGLE - feature export
# ===============================================================
K_INTRO = r'''# Phase 4a - WiMANS + CSI-Bench count features for the final ESP32 model (Kaggle)

Turns the two Kaggle-only datasets into the **same 10 window cues** the ESP32 counter uses
(`esp32_counter.py`), one row per clip x link, so the laptop notebook `phase4_final_local.ipynb`
can pool them with mat80 and WiCount.

| Dataset | Used | Clip | Links | People |
|---|---|---|---|---|
| WiMANS (Intel 5300) | 5 GHz only (2.4 GHz is white noise) | 3 s, 30 subcarriers -> 52 tones | 9 (3 tx x 3 rx) | 0-5, 3 rooms |
| CSI-Bench Localization | 168-subcarrier recordings | 5 s, 3 x 56 subcarriers -> 52 tones | 3 devices | 1-2, several rooms |

The clips are too short for the device's 30 s causal baseline, so each tone is divided by its mean over the
clip. The laptop notebook measures how much that changes the cues and the count on mat80 / WiCount.

**Kaggle settings:** inputs `shuokanghuang/wimans` and `guozhenjennzhu/csi-bench`, CPU is enough.
Output: `/kaggle/working/wimans_clip_features.parquet` and `csibench_clip_features.parquet` (a few MB).
Download them to `D:/wifi/Kaggle results/phase4_features/`.'''


K_WIMANS_EXPORT = r'''# 6. WiMANS -> one feature row per (clip, link); session + clip number kept for the empty-room calibration
import shutil

EXPORT = "/kaggle/working"
LINK_NAMES_W = [f"tx{a}_rx{b}" for a in range(3) for b in range(3)]

keep = meta["valid_frac"].values >= 1 - MAX_MISSING
idx = np.flatnonzero(keep)
n_links = len(LINK_NAMES_W)

wim = pd.DataFrame({"clip": np.repeat(idx, n_links), "view": np.tile(LINK_NAMES_W, len(idx))})
wim["dataset"] = "wimans"
wim["recording"] = "wimans_" + meta["label"].values[wim["clip"]]
wim["setup"] = meta["environment"].values[wim["clip"]]
wim["session"] = meta["session"].values[wim["clip"]]
wim["sample_no"] = pd.to_numeric(meta["label"].str.split("_").str[2], errors="coerce").values[wim["clip"]]
wim["count"] = meta["count"].values[wim["clip"]].astype(int)
wim["window_s"] = WIN_S
wim[FEATURE_NAMES] = FEATS[idx].reshape(-1, len(FEATURE_NAMES))
wim = wim.drop(columns="clip")

wim.to_parquet(os.path.join(EXPORT, "wimans_clip_features.parquet"), index=False)
print(f"WiMANS: {wim['recording'].nunique()} clips, {len(wim)} rows")
display(pd.crosstab(wim.drop_duplicates("recording")["setup"], wim.drop_duplicates("recording")["count"],
                    margins=True))

# ---- free memory + disk before CSI-Bench (the raw series are not needed any more) ----
del SERIES, VALID
shutil.rmtree("/kaggle/working/shared_format", ignore_errors=True)
shutil.rmtree("/kaggle/working/results", ignore_errors=True)'''


K_LOC_LAYOUT = C_LOC_LAYOUT.replace(
    "ESP_FEATS, ESP_META = FEATS.copy(), esp.copy()        # keep the ESP32 cues for the reality check (cell 13)\n",
    "N_BINS = int(WIN_S * FS)                              # one CSI-Bench clip = one 5-s window\n",
)
assert K_LOC_LAYOUT != C_LOC_LAYOUT, "C_LOC_LAYOUT changed - update the replace() above"


K_LOC_EXPORT = r'''# 9. CSI-Bench Localization -> one feature row per (clip, device link)
keep = LOC_OK & (loc_use["valid_frac"].values >= 1 - MAX_MISSING)
idx = np.flatnonzero(keep)
LINK_NAMES_C = [f"dev{v}" for v in range(LOC_DEVICES)]

cb = pd.DataFrame({"clip": np.repeat(idx, LOC_DEVICES), "view": np.tile(LINK_NAMES_C, len(idx))})
cb["dataset"] = "csibench"
cb["recording"] = "csibench_" + pd.Series(idx).astype(str).str.zfill(6).repeat(LOC_DEVICES).values
cb["setup"] = loc_use["environment"].astype(str).values[cb["clip"]]
session_col = "session" if "session" in loc_use.columns else "abs_path"   # parsed from the file path
cb["session"] = loc_use[session_col].astype(str).values[cb["clip"]]
cb["sample_no"] = cb["clip"]
cb["count"] = loc_use["count"].values[cb["clip"]].astype(int)
cb["window_s"] = WIN_S
cb[FEATURE_NAMES] = LOC_FEATS[idx].reshape(-1, len(FEATURE_NAMES))
cb = cb.drop(columns="clip")

cb.to_parquet(os.path.join(EXPORT, "csibench_clip_features.parquet"), index=False)
print(f"CSI-Bench: {cb['recording'].nunique()} clips, {len(cb)} rows")
display(pd.crosstab(cb.drop_duplicates("recording")["setup"], cb.drop_duplicates("recording")["count"],
                    margins=True))'''


K_CLEANUP = r'''# 10. Keep only the two feature tables in the output (small download)
for name in ["shared_format", "csibench_audit", "results"]:
    shutil.rmtree(os.path.join(EXPORT, name), ignore_errors=True)

for name in sorted(os.listdir(EXPORT)):
    path = os.path.join(EXPORT, name)
    if os.path.isfile(path):
        print(f"{name:40s} {os.path.getsize(path) / 1e6:8.2f} MB")'''


KAGGLE_CELLS = [
    ("md", K_INTRO),
    ("code", W_SETUP), ("code", W_AUDIT_LABELS), ("code", W_HELPERS), ("code", W_CONVERT),
    ("code", K_WIMANS_EXPORT),
    ("code", C_SETUP), ("code", C_AUDIT_META), ("code", K_LOC_LAYOUT), ("code", C_LOC_CONVERT),
    ("code", K_LOC_EXPORT), ("code", K_CLEANUP),
]


# ===============================================================
# LAPTOP - final model
# ===============================================================
L_INTRO = r'''# Phase 4 - final ESP32-S3 people counter from every labelled source (laptop)

Combines the two tracks: the device code and checks of Phase 3 (`esp32_counter.py`) with the extra rooms
from Kaggle (WiMANS, CSI-Bench). **Needs:** Phase 1 output in `D:/wifi/shared_format` (incl. the Phase 3
`causal_features.parquet`) and the output of `kaggle_phase4_features.ipynb` in
`D:/wifi/Kaggle results/phase4_features/`.

| Source | Hardware | People | Held-out unit | Features |
|---|---|---|---|---|
| mat80 | router, 80 MHz -> 16 one-link views | 0, 2-10 | setup (room) | causal, 5 s (device pipeline) |
| WiCount | **ESP32-S3** | 0-3 | file (only one room) | causal, 5 s (device pipeline) |
| WiMANS 5 GHz | Intel 5300, 9 links | 0-5 | room | clip mean, **3 s** clips |
| CSI-Bench Localization | 3 devices | 1-2 | room | clip mean, 5 s clips |
| CountFi | router, 80 MHz | 1-10 | - | stress test only, never trained on |

**Steps**
1. Tests + data checks (same as Phase 3).
2. **Clip check:** on mat80 / WiCount windows, compare the device's causal cues with clip-mean cues (5 s and 3 s).
   This shows how far the WiMANS / CSI-Bench rows are from what the ESP32 computes.
3. **Leave one site out** for 5 training pools x 2 modes (`zero_shot`, `empty_cal` = 20 s empty room at install).
   Every site is scored by a model that never saw it.
4. **Pick the final model** on sites every candidate can be scored on (they have an empty recording):
   - keep only candidates within 0.10 MAE of the best on the ESP32 data (WiCount);
   - of those, lowest **relative MAE** (MAE / MAE of always guessing the mean count), averaged over datasets;
   - within 0.02: `zero_shot` first (no install step), then the pool with more data.
5. Retrain on all its data, export (`trees.json`, `counter_model.h`, `model.txt`, `model_card.json`),
   stream the raw WiCount CSVs through the bundle, check the ESP32 budget, write `PHASE4_RESULTS.md`.

**Needs:** `numpy pandas pyarrow scipy matplotlib lightgbm tabulate` (CPU, ~15-30 min).'''


L_SETUP = r'''# 1. Paths, settings, load the counter code
import glob
import importlib
import json
import os
import sys
import time
import unittest

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WIFI_ROOT = "D:/wifi"
CODE_DIR = f"{WIFI_ROOT}/esp32_counter"            # esp32_counter.py + tests
RAW = f"{WIFI_ROOT}/Datasets"                      # raw WiCount CSVs (replay check)
SHARED = f"{WIFI_ROOT}/shared_format"              # Phase 1 output + Phase 3 causal features
EXTERNAL = f"{WIFI_ROOT}/Kaggle results/phase4_features"     # output of kaggle_phase4_features.ipynb
RESULTS = f"{WIFI_ROOT}/results/phase4_final__mat80_wicount_wimans_csibench"
MODEL_DIR = f"{RESULTS}/model_esp32_counter"       # final model bundle
os.makedirs(RESULTS, exist_ok=True)

sys.path.insert(0, CODE_DIR)
import esp32_counter as ec
importlib.reload(ec)

FEATURES = ec.FEATURE_NAMES
DATASETS = ["mat80", "wicount", "countfi"]         # local Phase 1 datasets
MODES = ["zero_shot", "empty_cal"]
SEED = 0

GBM_PARAMS = dict(objective="regression", n_estimators=300, learning_rate=0.05, num_leaves=15,
                  min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  random_state=SEED, verbose=-1)

print(json.dumps(ec.settings_dict(), indent=1))'''


L_EXTERNAL = r'''# 5. Add WiMANS (5 GHz) + CSI-Bench Localization from Kaggle (one row per clip x link)
EXT_FILES = {"wimans": "wimans_clip_features.parquet", "csibench": "csibench_clip_features.parquet"}

parts = []
for ds, name in EXT_FILES.items():
    path = os.path.join(EXTERNAL, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} is missing - run kaggle_phase4_features.ipynb on Kaggle "
                                f"and download its output to {EXTERNAL}")
    parts.append(pd.read_parquet(path))

ext = pd.concat(parts, ignore_index=True)
ext["window"] = 0                                   # one clip = one window
ext["t_start_s"] = 0.0

# ---- empty-room calibration: first empty session of each room, its first N_CAL_WINDOWS clips ----
# (same idea as the 20 s at install; these clips are never trained on or scored)
ext["is_cal"] = False
for (ds, setup), g in ext[ext["count"] == 0].groupby(["dataset", "setup"]):
    first_session = sorted(g["session"].unique())[0]
    clips = (g[g["session"] == first_session].drop_duplicates("recording")
             .sort_values("sample_no")["recording"].head(ec.N_CAL_WINDOWS))
    ext.loc[ext["recording"].isin(clips), "is_cal"] = True

for key, g in ext[ext["is_cal"]].groupby(["dataset", "setup", "view"]):
    OFFSETS[key] = ec.calibration_offset(g[FEATURES].values)
EMPTY_SITES = {(ds, setup) for ds, setup, _ in OFFSETS}

# ---- one table for everything (re-running this cell does not add the rows twice) ----
DATA = DATA[~DATA["dataset"].isin(EXT_FILES)]
shared_cols = [c for c in DATA.columns if c in ext.columns]
DATA = pd.concat([DATA[shared_cols], ext.loc[~ext["is_cal"], shared_cols]], ignore_index=True)

print("sites with an empty-room calibration:", sorted(EMPTY_SITES))
clips = ext.drop_duplicates("recording")
display(pd.crosstab([clips["dataset"], clips["setup"]], clips["count"], margins=True))
DATA.groupby("dataset").agg(recordings=("recording", "nunique"), sites=("setup", "nunique"),
                            views=("view", "nunique"), windows=("window", "size"))'''


L_SCORE = r'''# 6b. Extra scores: 0 / 1 / 2 / 3+ macro F1, and the MAE of always guessing the mean count
def f1_macro(y, r):
    labels = sorted(set(y) | set(r))
    f1 = []
    for c in labels:
        tp = np.sum((y == c) & (r == c))
        fp = np.sum((y != c) & (r == c))
        fn = np.sum((y == c) & (r != c))
        f1.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f1))


def score_full(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    out = score(y, pred)
    out["f1_4"] = f1_macro(np.minimum(y, 3), np.minimum(np.rint(pred), 3))
    out["mae_const"] = float(np.mean(np.abs(y - y.mean())))
    out["rel_mae"] = out["mae"] / max(out["mae_const"], 1e-9)
    return out'''


L_CLIP_CHECK = r'''# 7. Clip check: clip-mean cues (WiMANS / CSI-Bench) vs the device's causal cues, on the same windows
CLIP3 = int(3.0 * ec.FS)                            # WiMANS clip length in bins


def clip_normalise(W):
    """Each tone relative to its mean over the clip (what the Kaggle notebook did)."""
    mu = np.nanmean(W, axis=0, keepdims=True)
    dead = mu < 1e-3 * np.nanmedian(mu)
    Y = W / (mu + 1e-6) - 1.0
    Y[:, dead[0]] = 0.0
    return Y


local = DATA[DATA["dataset"].isin(["mat80", "wicount"])]
probe = pd.concat([g.sample(min(1500, len(g)), random_state=SEED)       # 1500 windows per dataset
                   for _, g in local.groupby("dataset")])
path_of = {(d, r): p for d, r, p in zip(recs["dataset"], recs["recording"], recs["path"])}

clip_rows = []
for (ds, rec), g in probe.groupby(["dataset", "recording"]):
    with np.load(path_of[(ds, rec)]) as z:
        S = z["series"].astype(np.float32) + 1.0    # Phase 1 stored x / recording_mean - 1
        view_pos = {str(v): i for i, v in enumerate(z["view_names"])}

    for idx, view, t_start in zip(g.index, g["view"], g["t_start_s"]):
        a = int(round(t_start * ec.FS))
        W = S[view_pos[view], a:a + ec.WIN]
        clip_rows.append((idx, ec.window_features(clip_normalise(W)),
                          ec.window_features(clip_normalise(W[:CLIP3]))))

idx = [r[0] for r in clip_rows]
inputs = {"causal (device)": ec.prepare_features(DATA.loc[idx, FEATURES].values),
          "clip mean, 5 s": ec.prepare_features(np.array([r[1] for r in clip_rows])),
          "clip mean, 3 s": ec.prepare_features(np.array([r[2] for r in clip_rows]))}

# ---- per-feature agreement ----
ref = inputs["causal (device)"]
agree = pd.DataFrame({
    name: {"corr_5s": np.corrcoef(ref[:, k], inputs["clip mean, 5 s"][:, k])[0, 1],
           "corr_3s": np.corrcoef(ref[:, k], inputs["clip mean, 3 s"][:, k])[0, 1],
           "shift_5s": np.median(inputs["clip mean, 5 s"][:, k] - ref[:, k]),
           "shift_3s": np.median(inputs["clip mean, 3 s"][:, k] - ref[:, k])}
    for k, name in enumerate(FEATURES)}).T
display(agree.round(3))

# ---- model agreement: a causal-trained zero-shot model fed the three inputs ----
reg, mc = fit(local.drop(index=idx), "zero_shot")
y = DATA.loc[idx, "count"].values
p_ref = np.clip(reg.predict(ref), 0, mc)
check = []
for name, X in inputs.items():
    p = np.clip(reg.predict(X), 0, mc)
    check.append({"input": name, **score(y, p),
                  "mean_abs_diff_vs_causal": float(np.mean(np.abs(p - p_ref)))})
clip_check = pd.DataFrame(check)
display(clip_check.round(3))

gap3 = clip_check.loc[clip_check["input"] == "clip mean, 3 s", "mean_abs_diff_vs_causal"].iloc[0]
if gap3 > 0.5:
    print(f"WARNING: 3-s clip cues move the estimate by {gap3:.2f} people on average - "
          "WiMANS rows are far from what the ESP32 computes; trust the pools without WiMANS more.")'''


L_EXPERIMENTS = r'''# 8. Leave one site out, for every training pool and calibration mode
TRAINABLE = ["mat80", "wicount", "wimans", "csibench"]
POOLS = {                                          # preference order on ties: more data first
    "all4": ["wicount", "mat80", "wimans", "csibench"],
    "esp32+mat80+wimans": ["wicount", "mat80", "wimans"],
    "esp32+wimans+csibench": ["wicount", "wimans", "csibench"],
    "esp32+mat80": ["wicount", "mat80"],           # = Phase 3 final (E4)
    "esp32_only": ["wicount"],
}


def site_of(df):
    """Held-out unit: the room / setup. WiCount has one room only, so there it is one file."""
    return np.where(df["dataset"] == "wicount", df["recording"], df["setup"])


DATA["site"] = site_of(DATA)
SITES = (DATA[DATA["dataset"].isin(TRAINABLE)][["dataset", "site"]]
         .drop_duplicates().sort_values(["dataset", "site"]).values.tolist())
print(len(SITES), "held-out sites:", pd.Series([d for d, _ in SITES]).value_counts().to_dict())

PRED_PARTS.clear()
t0 = time.time()
for pool_name, pool in POOLS.items():
    pool_df = DATA[DATA["dataset"].isin(pool)]
    train_desc = " + ".join(pool)

    for mode in MODES:
        full = fit(pool_df, mode)                  # for sites outside the pool and for CountFi

        for ds, site in SITES:
            in_site = (DATA["dataset"] == ds) & (DATA["site"] == site)
            if ds in pool:
                reg, mc = fit(pool_df[~in_site.loc[pool_df.index]], mode)
            else:
                reg, mc = full
            test = DATA[in_site]
            collect(pool_name, mode, train_desc, test, predict(reg, mc, test, mode))

        test = DATA[DATA["dataset"] == "countfi"]
        collect(pool_name, mode, train_desc, test, predict(*full, test, mode))
        print(f"{pool_name:24s} {mode:9s} done  [{time.time() - t0:.0f} s]", flush=True)


# ---- scores: all sites, and only sites with an empty recording (same windows for every candidate) ----
preds = pd.concat(PRED_PARTS, ignore_index=True)
rows = []
for (exp, mode, test_ds), g in preds.groupby(["experiment", "mode", "dataset"]):
    with_empty = g[[(d, s) in EMPTY_SITES for d, s in zip(g["dataset"], g["setup"])]]
    for sites_name, h in {"all": g, "with_empty": with_empty}.items():
        if h.empty:
            continue
        for level in ["raw", "smoothed"]:
            rows.append({"experiment": exp, "mode": mode, "test": test_ds, "sites": sites_name,
                         "level": level, "train": h["train"].iloc[0], **score_full(h["count"], h[level])})
results = pd.DataFrame(rows)

# ---- whole-recording estimate + rank correlation (a WiMANS / CSI-Bench 'recording' is one clip) ----
per_rec = (preds.groupby(["experiment", "mode", "dataset", "recording"])
           .agg(count=("count", "first"), pred=("smoothed", "median")).reset_index())
spearman = (per_rec.groupby(["experiment", "mode", "dataset"])
            .apply(lambda h: h["pred"].corr(h["count"], method="spearman"))
            .rename("spearman_rec").reset_index())

table = (results[(results["level"] == "smoothed") & (results["sites"] == "all")]
         .merge(spearman.rename(columns={"dataset": "test"}), on=["experiment", "mode", "test"], how="left"))
table[["experiment", "mode", "test", "n", "mae", "rel_mae", "exact", "within1", "f1_4", "spearman_rec"]].round(3)'''


L_FINAL = r'''# 9. Pick the final model, retrain on all its data, export
WIC_SLACK = 0.10                                   # allowed MAE loss on the ESP32 data
TIE = 0.02                                         # relative-MAE tie
CANDIDATES = [(p, m) for m in MODES for p in POOLS]          # zero_shot first, then more data first

sel = results[(results["level"] == "smoothed") & (results["sites"] == "with_empty")
              & results["test"].isin(TRAINABLE)]
summary = sel.pivot_table(index=["experiment", "mode"], columns="test", values="mae")
rel = sel.pivot_table(index=["experiment", "mode"], columns="test", values="rel_mae")
summary.columns = [f"mae_{c}" for c in summary.columns]
summary["rel_mae_mean"] = rel.mean(axis=1)
summary["datasets_scored"] = rel.notna().sum(axis=1)

best_wic = summary["mae_wicount"].min()
summary["esp32_ok"] = summary["mae_wicount"] <= best_wic + WIC_SLACK
ok = summary[summary["esp32_ok"]]
best_rel = ok["rel_mae_mean"].min()
CHOSEN = next(c for c in CANDIDATES if c in ok.index and ok.loc[c, "rel_mae_mean"] <= best_rel + TIE)
exp, mode = CHOSEN
display(summary.round(3))
print("scored on sites with an empty recording:",
      sorted({f"{d}/{s}" for d, s in EMPTY_SITES if d != "wicount"}), "+ WiCount files")
print("chosen:", CHOSEN)

# ---- final training on everything the chosen pool uses ----
final_train = DATA[DATA["dataset"].isin(POOLS[exp])]
reg, max_count = fit(final_train, mode)
tree = ec.TreeModel.from_lightgbm(reg.booster_, max_count)

# ---- the exported trees must give the LightGBM numbers on the real training inputs ----
X, ok_rows = model_inputs(final_train, mode)
parity = float(np.max(np.abs(tree.predict(X[ok_rows]) - reg.predict(X[ok_rows]))))
print(f"export parity on {ok_rows.sum()} training windows: max |diff| = {parity:.2e}")
if parity > 1e-3:
    raise RuntimeError("exported trees differ from LightGBM")

wic = DATA[DATA["dataset"] == "wicount"]           # used by the replay cell
held_out = table[(table["experiment"] == exp) & (table["mode"] == mode)]
card = {
    "name": "esp32_people_counter",
    "phase": 4,
    "created": time.strftime("%Y-%m-%d %H:%M"),
    "model": "LightGBM regression on 10 window features, exported as flat trees",
    "input": "ESP32-S3 CSI amplitude, 52 tones (HT20 L-LTF data tones), >= 50 packets/s",
    "output": "people count 0-%d, updated every 2.5 s (median of the last 30 s)" % max_count,
    # datasets that really reach the model (empty_cal drops sites without an empty recording)
    "training_data": sorted(final_train.reset_index(drop=True)[ok_rows]["dataset"].unique().tolist()),
    "n_training_windows": int(ok_rows.sum()),
    "calibration": mode,
    "calibration_howto": ("Leave the room empty, start the counter, call begin_calibration(); "
                          "the next %d windows (%.0f s after the %.0f s warm-up) set the offset."
                          % (ec.N_CAL_WINDOWS, ec.CAL_S, ec.WARMUP_S)) if mode == "empty_cal"
                         else "none (zero_shot)",
    "gbm_params": GBM_PARAMS,
    "selection": {"rule": (f"sites with an empty recording; WiCount MAE within {WIC_SLACK} of the best; "
                           f"then lowest mean relative MAE over datasets; within {TIE} prefer zero_shot, "
                           "then the pool with more data"),
                  "candidates": json.loads(summary.reset_index().to_json(orient="records")),
                  "chosen": list(CHOSEN)},
    "held_out_results": json.loads(held_out.to_json(orient="records")),
    "clip_check": json.loads(clip_check.to_json(orient="records")),
    "export_parity_max_abs": parity,
    "caveats": [
        "WiCount is the only ESP32-S3 data: one room, one evening, counts 0-3 (0 and 3 one file each); "
        "it was also used in the selection rule.",
        "WiMANS / CSI-Bench rows use clip-mean normalisation (3 s / 5 s clips), not the device's 30 s causal "
        "baseline - see clip_check for the size of that difference.",
        "Packet times of WiMANS (1 kHz) and CSI-Bench (100 Hz) are assumed uniform (no timestamps).",
        "Counts above 3 come from router / Intel 5300 data only.",
        "Re-validate on your own ESP32-S3 recordings before trusting the count.",
    ],
}
card = ec.save_bundle(MODEL_DIR, tree, card, reg.booster_)
print(f"saved {MODEL_DIR}: {card['n_trees']} trees, {card['n_nodes']} nodes, "
      f"~{card['flash_bytes_est'] / 1024:.0f} KB flash")'''


L_REPORT = r'''# 12. Save results, figures and the Phase 4 report
results.to_csv(os.path.join(RESULTS, "results.csv"), index=False)
summary.reset_index().to_csv(os.path.join(RESULTS, "selection.csv"), index=False)
per_rec.to_csv(os.path.join(RESULTS, "recording_predictions.csv"), index=False)
clip_check.to_csv(os.path.join(RESULTS, "clip_check.csv"), index=False)
agree.to_csv(os.path.join(RESULTS, "clip_check_features.csv"))
replay.to_csv(os.path.join(RESULTS, "replay_wicount.csv"), index=False)
budget.to_csv(os.path.join(RESULTS, "esp32_budget.csv"), index=False)

fig_dir = os.path.join(RESULTS, "figures")
os.makedirs(fig_dir, exist_ok=True)

# ---- relative MAE per candidate (lower = better, 1 = no better than guessing the mean) ----
s = summary.reset_index()
s["label"] = s["experiment"] + " / " + s["mode"]
fig, ax = plt.subplots(figsize=(8, 0.45 * len(s) + 1))
colors = ["tab:green" if (e, m) == CHOSEN else ("tab:blue" if k else "tab:gray")
          for e, m, k in zip(s["experiment"], s["mode"], s["esp32_ok"])]
ax.barh(s["label"], s["rel_mae_mean"], color=colors)
ax.axvline(1.0, color="k", ls="--", lw=0.8)
ax.set_xlabel("mean relative MAE over datasets (1 = mean guess)")
ax.set_title("Candidates (green = final, gray = too weak on ESP32 data)")
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, "candidates_rel_mae.png"), dpi=120)
plt.show()

# ---- final model: mean estimate per true count, per dataset (shows saturation) ----
fin = preds[(preds["experiment"] == exp) & (preds["mode"] == mode)]
per_count = (fin.groupby(["dataset", "count"])
             .agg(n=("smoothed", "size"), mean_pred=("smoothed", "mean"),
                  exact=("smoothed", lambda p: float(np.mean(np.rint(p) == fin.loc[p.index, "count"]))))
             .reset_index())
fig, ax = plt.subplots(figsize=(6, 5))
for ds, h in per_count.groupby("dataset"):
    ax.plot(h["count"], h["mean_pred"], "o-", label=ds)
top = int(per_count["count"].max())
ax.plot([0, top], [0, top], "k--", lw=0.8)
ax.set_xlabel("true people")
ax.set_ylabel("mean estimate (held-out)")
ax.set_title(f"Final model {exp} / {mode}")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, "final_mean_estimate_per_count.png"), dpi=120)
plt.show()
per_count.to_csv(os.path.join(RESULTS, "final_per_count.csv"), index=False)

# ---- markdown report ----
show = ["experiment", "mode", "test", "n", "mae", "rel_mae", "exact", "within1", "f1_4", "spearman_rec"]
lines = [
    "# Phase 4 results - final ESP32-S3 people counter (all labelled sources)", "",
    f"Generated {time.strftime('%Y-%m-%d %H:%M')}. One link, device preprocessing, "
    f"{ec.WIN_S} s windows every {ec.HOP_S} s, output = median of the last {ec.SMOOTH_S:.0f} s "
    "(WiMANS / CSI-Bench: one clip = one estimate).", "",
    "## Final model", "",
    f"- Chosen: **{exp} / {mode}**",
    f"- Rule: {card['selection']['rule']}",
    f"- Trained on: {', '.join(card['training_data'])} ({card['n_training_windows']} windows), "
    f"counts 0-{max_count}",
    f"- Calibration: {card['calibration_howto']}",
    f"- Size: {card['n_trees']} trees, {card['n_nodes']} nodes, ~{card['flash_bytes_est'] / 1024:.0f} KB flash",
    f"- Files: `{MODEL_DIR}/` (trees.json, counter_model.h, model.txt, model_card.json)", "",
    "## Selection (sites with an empty recording, smoothed MAE per dataset)", "",
    summary.reset_index().round(3).to_markdown(index=False), "",
    "## Held-out results of the final model (all sites)", "",
    held_out[show].round(3).to_markdown(index=False), "",
    "## Final model: mean estimate per true count", "",
    per_count.round(3).to_markdown(index=False), "",
    "## All candidates, all sites", "",
    table[show].round(3).to_markdown(index=False), "",
    "## Clip check (mat80 + WiCount windows; causal = what the ESP32 computes)", "",
    clip_check.round(3).to_markdown(index=False), "",
    "## Replay of the raw ESP32 CSVs through the saved bundle (pipeline check)", "",
    replay.to_markdown(index=False), "",
    "## ESP32-S3 budget (estimates)", "",
    budget.to_markdown(index=False), "",
    "## Caveats", "",
] + [f"- {c}" for c in card["caveats"]]

with open(os.path.join(RESULTS, "PHASE4_RESULTS.md"), "w") as f:
    f.write(chr(10).join(lines))
print("saved", os.path.join(RESULTS, "PHASE4_RESULTS.md"))'''


# Phase 3 cells renumbered for this notebook
L_HELPERS = HELPERS.replace("# 5. Train / predict / score helpers", "# 6. Train / predict / score helpers", 1)
L_REPLAY = REPLAY.replace("# 8. Final run:", "# 10. Final run:", 1)
L_BUDGET = BUDGET.replace("# 9. ESP32-S3 budget check", "# 11. ESP32-S3 budget check", 1)
L_DONE = (DONE.replace("# 11. Done", "# 13. Done", 1)
          .replace('print("Phase 3 finished.")', 'print("Phase 4 finished.")', 1)
          .replace('"PHASE3_RESULTS.md"', '"PHASE4_RESULTS.md"', 1))
for new, old in [(L_HELPERS, HELPERS), (L_REPLAY, REPLAY), (L_BUDGET, BUDGET), (L_DONE, DONE)]:
    assert new != old, "a Phase 3 cell header changed - update the replace() calls"

LOCAL_CELLS = [
    ("md", L_INTRO), ("code", L_SETUP), ("code", TESTS), ("code", CHECKS), ("code", FEATURES_CELL),
    ("code", L_EXTERNAL), ("code", L_HELPERS), ("code", L_SCORE), ("code", L_CLIP_CHECK),
    ("code", L_EXPERIMENTS), ("code", L_FINAL), ("code", L_REPLAY), ("code", L_BUDGET),
    ("code", L_REPORT), ("code", L_DONE),
]


# ===============================================================
# Notebook writer
# ===============================================================
NOTEBOOKS = {
    "kaggle_phase4_features.ipynb": KAGGLE_CELLS,
    "phase4_final_local.ipynb": LOCAL_CELLS,
}


def main():
    for filename, cells in NOTEBOOKS.items():
        for kind, text in cells:
            if kind == "code":
                syntax_check(text)
        if filename.endswith("_local.ipynb"):
            assert all("/kaggle" not in t and "google.colab" not in t for _, t in cells)

        nb = {"cells": [to_cell(k, t) for k, t in cells],
              "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                          "name": "python3"},
                           "language_info": {"name": "python"}},
              "nbformat": 4, "nbformat_minor": 5}

        out = os.path.join(ROOT, filename)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print("wrote", out, f"({len(cells)} cells)")


if __name__ == "__main__":
    main()

"""Generate the Phase 3 (final ESP32-S3 model) notebook - laptop version.

  phase3_esp32_final_local.ipynb -> tests, checks, held-out experiments, final training,
                                    export to D:/wifi/results/phase3_esp32_final__mat80_wicount_countfi/model_esp32_counter, replay on raw ESP32 data

All model / preprocessing code lives in D:/wifi/esp32_counter/esp32_counter.py; the notebook
only imports it, so the notebook, the tests and the device use the same code.
Run:  py -3 colab_tools/build_final_nb.py
"""

import ast
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from build_phase12_nbs import to_cell  # noqa: E402


# ===============================================================
# Cells
# ===============================================================
INTRO = r'''# Phase 3 - final ESP32-S3 people counter (one run, top to bottom)

Reads the Phase 1 output in `D:/wifi/shared_format` and the raw WiCount CSVs. **Run Phase 1 first.**
All the counting code is in `D:/wifi/esp32_counter/esp32_counter.py`; this notebook only drives it.

**What is different from Phase 2 (so the numbers match what the device will do)**

| Phase 2 | Phase 3 (device-ready) |
|---|---|
| each tone divided by its mean over the **whole recording** (uses the future) | **causal** baseline per tone (30 s EMA), first 10 s skipped |
| calibration from **all** windows of the test setup | **20 s empty-room calibration** at install, or none |
| 16 views averaged | **one view** = one ESP32 link |
| per-window estimate | per-window + **30 s rolling median** (what the display shows) |
| GBM + CNN | GBM regression only (best on one link in Phase 2, ~10x smaller than the classifier) |

**Experiments** (every test group is unseen in training; `empty_cal` = 20 s empty-room offset)

| Id | Train | Test | Question |
|---|---|---|---|
| E1 | 80 MHz set, 3 setups | 4th setup | how good is one link on unseen rooms? |
| E2 | WiCount (ESP32-S3), 14 files | 15th file | ESP32-only model |
| E3 | 80 MHz set | WiCount | does a router-trained model work on ESP32 hardware? |
| E4 | 80 MHz set + 14 WiCount files | 15th file | does the 80 MHz data help the ESP32? |
| E5 | 80 MHz set | CountFi | stress test only (time-confounded labels) |

**Final model:** the candidate with the lowest smoothed MAE on WiCount (E2/E3/E4); within 0.05 the one
with more training data and no install step wins. It is retrained on all its data and exported:

```
D:/wifi/results/phase3_esp32_final__mat80_wicount_countfi/model_esp32_counter/
    trees.json        model for Python (no LightGBM needed)
    counter_model.h   same model for the ESP32 firmware (C arrays + cc_prepare + cc_predict)
    model.txt         LightGBM native file
    model_card.json   settings, calibration, held-out results, caveats
```

**Needs:** `numpy pandas pyarrow scipy matplotlib lightgbm tabulate` (no GPU). gcc/clang on PATH is optional
(one test compiles the C header). Results go to `D:/wifi/results/phase3_esp32_final__mat80_wicount_countfi/`.'''


SETUP = r'''# 1. Paths, settings, load the counter code
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
SHARED = f"{WIFI_ROOT}/shared_format"              # Phase 1 output
RESULTS = f"{WIFI_ROOT}/results/phase3_esp32_final__mat80_wicount_countfi"
MODEL_DIR = f"{WIFI_ROOT}/results/phase3_esp32_final__mat80_wicount_countfi/model_esp32_counter"     # final model bundle
os.makedirs(RESULTS, exist_ok=True)

sys.path.insert(0, CODE_DIR)
import esp32_counter as ec
importlib.reload(ec)

FEATURES = ec.FEATURE_NAMES
DATASETS = ["mat80", "wicount", "countfi"]
MODES = ["zero_shot", "empty_cal"]
SEED = 0

GBM_PARAMS = dict(objective="regression", n_estimators=300, learning_rate=0.05, num_leaves=15,
                  min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  random_state=SEED, verbose=-1)

print(json.dumps(ec.settings_dict(), indent=1))'''


TESTS = r'''# 2. Unit tests - the run stops here if anything fails
import test_esp32_counter as tests
importlib.reload(tests)

suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)

if not result.wasSuccessful():
    raise RuntimeError("Unit tests failed - fix esp32_counter.py before training.")
print(f"all tests passed ({len(result.skipped)} skipped)")'''


CHECKS = r'''# 3. Data checks: Phase 1 settings, recordings per site, real ESP32 packet rate

# ---- Phase 1 must use the same grid as the counter ----
fmt = json.load(open(os.path.join(SHARED, "format.json")))
for key in ["fs_hz", "win_s", "hop_s", "n_tones", "max_gap_bins", "max_missing"]:
    if fmt[key] != ec.settings_dict()[key]:
        raise ValueError(f"format.json {key}={fmt[key]} differs from esp32_counter.py")

# ---- recordings per site (metadata only) ----
meta_rows = []
for ds in DATASETS:
    for path in sorted(glob.glob(os.path.join(SHARED, ds, "*.npz"))):
        with np.load(path) as z:
            meta = json.loads(str(z["meta"]))
            n_views = len(z["view_names"])
        meta_rows.append({**meta, "views": n_views, "path": path})

recs = pd.DataFrame(meta_rows)
if recs.empty:
    raise FileNotFoundError(f"No recordings in {SHARED}. Run phase1_shared_format_local.ipynb first.")

sites = (recs.groupby(["dataset", "setup"])
         .agg(recordings=("recording", "size"),
              counts=("count", lambda c: sorted(set(c))),
              has_empty=("count", lambda c: bool((c == 0).any())),
              minutes=("seconds", lambda s: round(s.sum() / 60, 1)),
              missing=("missing_frac", "mean")))
display(sites)


# ---- raw ESP32-S3 packets (WiCount) ----
WRAP = 2 ** 32                                       # local_timestamp is a 32-bit microsecond counter


def read_wicount_csv(path):
    """Raw ESP32-S3 packets: times (s) and amplitude (packets, 52), time-ordered."""
    df = pd.read_csv(path)
    rows, keep_t = [], []
    for ts, txt in zip(df["local_timestamp"], df["data"]):
        try:
            vals = json.loads(txt)
        except (TypeError, ValueError):
            continue
        if len(vals) == 2 * ec.N_TONES:
            rows.append(vals)
            keep_t.append(ts)

    iq = np.array(rows, dtype=np.float32)
    A = np.hypot(iq[:, 0::2], iq[:, 1::2])

    dt = np.diff(np.array(keep_t, dtype=np.int64))
    dt[dt < 0] += WRAP
    t = np.concatenate([[0], np.cumsum(dt)]) * 1e-6
    order = np.argsort(t, kind="stable")
    return t[order], A[order]


WICOUNT_CSVS = sorted(glob.glob(os.path.join(RAW, "WiCount", "data", "*", "*.csv")))
rate_rows = []
for path in WICOUNT_CSVS:
    t, A = read_wicount_csv(path)
    dt_ms = np.diff(t) * 1000
    bins = np.unique(np.floor(t * ec.FS).astype(int))
    rate_rows.append({"file": os.path.basename(path), "count": int(os.path.basename(os.path.dirname(path))),
                      "packets": len(t), "seconds": round(t[-1], 1),
                      "rate_hz": round(len(t) / max(t[-1], 1e-9), 1),
                      "dt_median_ms": round(float(np.median(dt_ms)), 1),
                      "dt_p99_ms": round(float(np.percentile(dt_ms, 99)), 1),
                      "bins_with_data": round(len(bins) / (int(t[-1] * ec.FS) + 1), 3)})
rate = pd.DataFrame(rate_rows)
display(rate)

slow = rate[rate["rate_hz"] < ec.FS]
if len(slow):
    print(f"WARNING: {len(slow)} WiCount files send fewer than {ec.FS} packets/s - "
          "set the ESP32 traffic to >= 100 packets/s for your own recordings.")
print(f"median ESP32 packet rate: {rate['rate_hz'].median():.0f} Hz, "
      f"bins with data: {rate['bins_with_data'].median():.1%}")'''


FEATURES_CELL = r'''# 4. Causal features for every recording and view (cached)
CACHE = os.path.join(SHARED, "causal_features.parquet")
CACHE_KEY = os.path.join(SHARED, "causal_features.json")

cache_ok = (os.path.exists(CACHE) and os.path.exists(CACHE_KEY)
            and json.load(open(CACHE_KEY)) == ec.settings_dict())

if cache_ok:
    cf = pd.read_parquet(CACHE)
    print("loaded cached causal features:", len(cf), "rows")
else:
    rows = []
    t0 = time.time()
    for i, r in enumerate(recs.to_dict("records")):
        with np.load(r["path"]) as z:
            S, valid = z["series"], z["valid"]
            views = [str(v) for v in z["view_names"]]

        for v, view in enumerate(views):
            # Phase 1 stored x / recording_mean - 1; the mean cancels in the causal baseline
            X = S[v].astype(np.float32) + 1.0
            starts, F = ec.features_from_series(X, valid)
            for j, a in enumerate(starts):
                rows.append({"dataset": r["dataset"], "recording": r["recording"], "setup": r["setup"],
                             "count": r["count"], "view": view, "window": j, "t_start_s": a / ec.FS,
                             **dict(zip(FEATURES, F[j]))})

        if (i + 1) % 20 == 0 or i + 1 == len(recs):
            print(f"{i + 1:4d}/{len(recs)} recordings  [{time.time() - t0:.0f} s]")

    cf = pd.DataFrame(rows)
    cf.to_parquet(CACHE, index=False)
    json.dump(ec.settings_dict(), open(CACHE_KEY, "w"))

# ---- empty-room calibration windows: first 20 s after warm-up of each site's empty recording ----
empty_rec = (recs[recs["count"] == 0].sort_values("recording")
             .groupby(["dataset", "setup"])["recording"].first())

cf["is_cal"] = False
for (ds, setup), rec in empty_rec.items():
    m = ((cf["dataset"] == ds) & (cf["recording"] == rec)
         & (cf["t_start_s"] + ec.WIN_S <= ec.WARMUP_S + ec.CAL_S + 1e-9))
    cf.loc[m, "is_cal"] = True

OFFSETS = {key: ec.calibration_offset(g[FEATURES].values)
           for key, g in cf[cf["is_cal"]].groupby(["dataset", "setup", "view"])}
EMPTY_SITES = {(ds, setup) for ds, setup, _ in OFFSETS}

# calibration windows are never used for training or scoring
DATA = cf[~cf["is_cal"]].reset_index(drop=True)

print("empty recording per site:", dict(empty_rec))
print("sites without an empty recording (no empty_cal):",
      sorted(set(zip(recs["dataset"], recs["setup"])) - EMPTY_SITES))
DATA.groupby("dataset").agg(recordings=("recording", "nunique"), views=("view", "nunique"),
                            windows=("window", "size"))'''


HELPERS = r'''# 5. Train / predict / score helpers
PRED_PARTS = []


def model_inputs(df, mode):
    """Model input + mask of usable rows (empty_cal needs an offset for the row's site + view)."""
    df = df.reset_index(drop=True)
    F = df[FEATURES].values
    if mode == "zero_shot":
        return ec.prepare_features(F), np.ones(len(df), dtype=bool)

    X = np.zeros((len(df), len(FEATURES)), dtype=np.float32)
    ok = np.zeros(len(df), dtype=bool)
    for key, idx in df.groupby(["dataset", "setup", "view"]).indices.items():
        if key in OFFSETS:
            X[idx] = ec.prepare_features(F[idx], OFFSETS[key])
            ok[idx] = True
    return X, ok


def fit(train_df, mode):
    """LightGBM regression; every dataset in the training set gets the same total weight."""
    X, ok = model_inputs(train_df, mode)
    d = train_df.reset_index(drop=True)[ok]
    w = 1.0 / d.groupby("dataset")["dataset"].transform("size").values
    reg = lgb.LGBMRegressor(**GBM_PARAMS).fit(X[ok], d["count"].values, sample_weight=w / w.mean())
    return reg, int(d["count"].max())


def predict(reg, max_count, test_df, mode):
    X, ok = model_inputs(test_df, mode)
    p = np.full(len(test_df), np.nan)
    if ok.any():
        p[ok] = np.clip(reg.predict(X[ok]), 0, max_count)
    return p


def collect(experiment, mode, train_desc, test_df, p):
    """Store window predictions + the causal 30 s rolling median per (recording, view) stream."""
    d = test_df[["dataset", "recording", "setup", "count", "view", "window"]].reset_index(drop=True).copy()
    d["raw"] = p
    d = d[~np.isnan(d["raw"])].sort_values(["recording", "view", "window"])
    if d.empty:
        return                                      # e.g. empty_cal on a site without an empty recording
    d["smoothed"] = d.groupby(["recording", "view"])["raw"].transform(
        lambda s: ec.smooth_predictions(s.values))
    d["experiment"], d["mode"], d["train"] = experiment, mode, train_desc
    PRED_PARTS.append(d)


def score(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    rounded = np.rint(pred)
    return {"n": len(y), "mae": float(np.mean(np.abs(pred - y))),
            "exact": float(np.mean(rounded == y)),
            "within1": float(np.mean(np.abs(rounded - y) <= 1))}'''


EXPERIMENTS = r'''# 6. Held-out experiments E1-E5 (one view = one ESP32 link)
mat = DATA[DATA["dataset"] == "mat80"]
wic = DATA[DATA["dataset"] == "wicount"]
cfi = DATA[DATA["dataset"] == "countfi"]
PRED_PARTS.clear()
t0 = time.time()

# ---- E1: 80 MHz set, leave one setup out ----
for mode in MODES:
    for site in sorted(mat["setup"].unique()):
        reg, mc = fit(mat[mat["setup"] != site], mode)
        test = mat[mat["setup"] == site]
        collect("E1_mat80_loso", mode, "mat80 other setups", test, predict(reg, mc, test, mode))
    print(f"E1 {mode:9s} done  [{time.time() - t0:.0f} s]")

# ---- E2: ESP32 only, leave one WiCount recording out ----
# (one site only, so the empty-room offset shifts train and test equally: zero_shot == empty_cal)
for rec in sorted(wic["recording"].unique()):
    reg, mc = fit(wic[wic["recording"] != rec], "zero_shot")
    test = wic[wic["recording"] == rec]
    collect("E2_esp32_only", "zero_shot", "wicount other files", test, predict(reg, mc, test, "zero_shot"))
print(f"E2 done  [{time.time() - t0:.0f} s]")

# ---- E3: router data -> ESP32 hardware;  E5: router data -> CountFi (stress test) ----
for mode in MODES:
    reg, mc = fit(mat, mode)
    collect("E3_mat80_to_esp32", mode, "mat80 all", wic, predict(reg, mc, wic, mode))
    if mode == "zero_shot":
        collect("E5_mat80_to_countfi", mode, "mat80 all", cfi, predict(reg, mc, cfi, mode))
print(f"E3 + E5 done  [{time.time() - t0:.0f} s]")

# ---- E4: router data + other ESP32 files -> left-out ESP32 file ----
for mode in MODES:
    for rec in sorted(wic["recording"].unique()):
        reg, mc = fit(pd.concat([mat, wic[wic["recording"] != rec]]), mode)
        test = wic[wic["recording"] == rec]
        collect("E4_mat80_plus_esp32", mode, "mat80 + wicount other files", test,
                predict(reg, mc, test, mode))
    print(f"E4 {mode:9s} done  [{time.time() - t0:.0f} s]")


# ---- scores: per window ('raw') and after the 30 s median ('smoothed') ----
preds = pd.concat(PRED_PARTS, ignore_index=True)
rows = []
for (exp, mode, test_ds), g in preds.groupby(["experiment", "mode", "dataset"]):
    subsets = {"all": g}
    with_empty = g[[(d, s) in EMPTY_SITES for d, s in zip(g["dataset"], g["setup"])]]
    if 0 < len(with_empty) < len(g):
        subsets["with_empty"] = with_empty                  # same sites as the empty_cal run
    for sites_name, h in subsets.items():
        for level in ["raw", "smoothed"]:
            rows.append({"experiment": exp, "mode": mode, "test": test_ds, "sites": sites_name,
                         "level": level, "train": h["train"].iloc[0], **score(h["count"], h[level])})
results = pd.DataFrame(rows)

# ---- whole-recording estimate (median of the smoothed output) + rank correlation ----
per_rec = (preds.groupby(["experiment", "mode", "dataset", "recording"])
           .agg(count=("count", "first"), pred=("smoothed", "median")).reset_index())
spearman = (per_rec.groupby(["experiment", "mode", "dataset"])
            .apply(lambda h: h["pred"].corr(h["count"], method="spearman"))
            .rename("spearman_rec").reset_index())

table = (results[results["level"] == "smoothed"]
         .merge(spearman.rename(columns={"dataset": "test"}), on=["experiment", "mode", "test"], how="left"))
table[["experiment", "mode", "test", "sites", "train", "n", "mae", "exact", "within1", "spearman_rec"]].round(3)'''


FINAL = r'''# 7. Pick the final model, retrain on all its data, export
CANDIDATES = [                                   # preference order when scores tie
    ("E4_mat80_plus_esp32", "zero_shot"),        # most data, no install step
    ("E4_mat80_plus_esp32", "empty_cal"),
    ("E3_mat80_to_esp32", "zero_shot"),
    ("E3_mat80_to_esp32", "empty_cal"),
    ("E2_esp32_only", "zero_shot"),
]
TIE = 0.05
TRAIN_SETS = {"E4_mat80_plus_esp32": ["mat80", "wicount"],
              "E3_mat80_to_esp32": ["mat80"],
              "E2_esp32_only": ["wicount"]}

esp = (results[(results["test"] == "wicount") & (results["level"] == "smoothed") & (results["sites"] == "all")]
       .set_index(["experiment", "mode"]))
cand = esp.loc[[c for c in CANDIDATES if c in esp.index], ["n", "mae", "exact", "within1"]]
best_mae = cand["mae"].min()
CHOSEN = next(c for c in CANDIDATES if c in cand.index and cand.loc[c, "mae"] <= best_mae + TIE)
exp, mode = CHOSEN
display(cand.round(3))
print("chosen:", CHOSEN)

e3_rho = spearman.query("experiment == 'E3_mat80_to_esp32' and dataset == 'wicount'")["spearman_rec"].max()
if not e3_rho > 0.5:
    print(f"WARNING: router-trained model barely ranks ESP32 recordings (spearman {e3_rho:.2f}) - "
          "the 80 MHz data does not transfer well to ESP32 hardware.")

# ---- final training on everything the chosen candidate uses ----
final_train = DATA[DATA["dataset"].isin(TRAIN_SETS[exp])]
reg, max_count = fit(final_train, mode)
tree = ec.TreeModel.from_lightgbm(reg.booster_, max_count)

# ---- the exported trees must give the LightGBM numbers on the real training inputs ----
X, ok = model_inputs(final_train, mode)
parity = float(np.max(np.abs(tree.predict(X[ok]) - reg.predict(X[ok]))))
print(f"export parity on {ok.sum()} training windows: max |diff| = {parity:.2e}")
if parity > 1e-3:
    raise RuntimeError("exported trees differ from LightGBM")

held_out = results[(results["experiment"] == exp) & (results["mode"] == mode)]
card = {
    "name": "esp32_people_counter",
    "created": time.strftime("%Y-%m-%d %H:%M"),
    "model": "LightGBM regression on 10 window features, exported as flat trees",
    "input": "ESP32-S3 CSI amplitude, 52 tones (HT20 L-LTF data tones), >= 50 packets/s",
    "output": "people count 0-%d, updated every 2.5 s (median of the last 30 s)" % max_count,
    "training_data": TRAIN_SETS[exp],
    "n_training_windows": int(ok.sum()),
    "calibration": mode,
    "calibration_howto": ("Leave the room empty, start the counter, call begin_calibration(); "
                          "the next %d windows (%.0f s after the %.0f s warm-up) set the offset."
                          % (ec.N_CAL_WINDOWS, ec.CAL_S, ec.WARMUP_S)) if mode == "empty_cal"
                         else "none (zero_shot)",
    "gbm_params": GBM_PARAMS,
    "selection": {"rule": f"lowest smoothed MAE on WiCount held-out files; within {TIE} prefer the "
                          "earlier candidate (more data, no install step)",
                  "candidates": json.loads(cand.reset_index().to_json(orient="records")),
                  "chosen": list(CHOSEN)},
    "held_out_results": json.loads(held_out.to_json(orient="records")),
    "export_parity_max_abs": parity,
    "caveats": [
        "The WiCount score of the chosen model is optimistic: the same files were used to choose it.",
        "WiCount = one room, one evening, counts 0-3; counts 0 and 3 have one file each.",
        "No ESP32 data above 3 people; higher counts come from 4-antenna 80 MHz router data only.",
        "80 MHz setup PC2a has no empty recording; PC4a_n00 looked occupied in the audit.",
        "Re-validate on your own ESP32-S3 recordings before trusting the count.",
    ],
}
card = ec.save_bundle(MODEL_DIR, tree, card, reg.booster_)
print(f"saved {MODEL_DIR}: {card['n_trees']} trees, {card['n_nodes']} nodes, "
      f"~{card['flash_bytes_est'] / 1024:.0f} KB flash")'''


REPLAY = r'''# 8. Final run: stream the raw ESP32-S3 CSVs packet by packet through the saved bundle
# Pipeline check (raw CSV -> device code -> count). For E2/E4 models WiCount was in training,
# so this is NOT an accuracy number - it must agree with the batch pipeline.
model, card = ec.load_bundle(MODEL_DIR)
calibrated = card["calibration"] == "empty_cal"
empty_csv = [p for p in WICOUNT_CSVS if os.path.basename(os.path.dirname(p)) == "0"][0]
cal_end_s = ec.WARMUP_S + ec.CAL_S

# ---- install step: 20 s of the empty room ----
offset = None
if calibrated:
    t, A = read_wicount_csv(empty_csv)
    c = ec.StreamingCounter(model)
    c.begin_calibration()
    for ti, ai in zip(t, A):
        c.push(ti, ai)
        if not c.calibrating:
            break
    offset = c.offset
    print("calibrated on", os.path.basename(empty_csv))

# ---- batch predictions of the same model on the Phase 1 series (for comparison) ----
X, ok = model_inputs(wic, mode)
batch = wic.reset_index(drop=True)[ok].copy()
batch["raw"] = np.clip(model.predict(X[ok]), 0, model.max_count)
batch = batch.sort_values(["recording", "window"])
batch["smoothed"] = batch.groupby("recording")["raw"].transform(lambda s: ec.smooth_predictions(s.values))
batch_med = batch.groupby("recording")["smoothed"].median()

# ---- streaming ----
replay_rows = []
for path in WICOUNT_CSVS:
    t, A = read_wicount_csv(path)
    c = ec.StreamingCounter(model, offset)
    out = []
    for ti, ai in zip(t, A):
        out += c.push(ti, ai)
    out += c.flush()

    ok_out = [o for o in out if o.get("status") == "ok"]
    if calibrated and path == empty_csv:
        ok_out = [o for o in ok_out if o["t_end_s"] > cal_end_s + 1e-9]
    name = os.path.splitext(os.path.basename(path))[0]
    true = int(os.path.basename(os.path.dirname(path)))
    counts = np.array([o["count"] for o in ok_out])
    replay_rows.append({"file": name, "true": true, "windows": len(ok_out),
                        "last_count": int(counts[-1]) if len(counts) else None,
                        "exact_share": round(float(np.mean(counts == true)), 2) if len(counts) else None,
                        "stream_median": round(float(np.median([o["smoothed"] for o in ok_out])), 2)
                        if ok_out else None,
                        "batch_median": round(float(batch_med.get(name, np.nan)), 2)})

replay = pd.DataFrame(replay_rows)
replay["stream_vs_batch"] = (replay["stream_median"] - replay["batch_median"]).abs().round(3)
display(replay)

worst = replay["stream_vs_batch"].max()
print(f"largest stream-vs-batch difference: {worst:.3f} people "
      "(float16 storage in Phase 1 explains small differences)")
if worst > 0.25:
    print("WARNING: streaming and batch disagree - check esp32_counter.py before deploying.")'''


BUDGET = r'''# 9. ESP32-S3 budget check (estimates; the S3 has 512 KB SRAM, usually 8 MB flash, FPU at 240 MHz)
n = ec.N_TONES
ram = {
    "window ring buffer (250 x 52 float32)": ec.WIN * n * 4,
    "baseline + bin accumulator + last bin": 3 * n * 4,
    "covariance 52 x 52 (pc1, n_pc_90, tone_corr)": n * n * 4,
    "eigen workspace 52 x 52 (Jacobi)": n * n * 4,
    "FFT 256 + features + offset + 30 s median": 256 * 8 + 3 * 10 * 4 + ec.SMOOTH_N * 4,
}
ops = {
    "std / diff / centring": 4 * ec.WIN * n,
    "covariance": ec.WIN * n * (n + 1) // 2 * 2,
    "eigenvalues (Jacobi, ~8 sweeps)": 8 * 2 * n ** 3,
    "FFT 256": 5 * 256 * 8,
    "trees": card["n_trees"] * 12,
}
budget = pd.DataFrame(
    [{"item": "flash: model arrays", "value": f"{tree.flash_bytes() / 1024:.0f} KB",
      "limit": "512 KB", "ok": tree.flash_bytes() < 512 * 1024},
     {"item": "RAM: counter state", "value": f"{sum(ram.values()) / 1024:.0f} KB",
      "limit": "128 KB", "ok": sum(ram.values()) < 128 * 1024},
     {"item": "compute per 2.5 s window", "value": f"{sum(ops.values()) / 1e6:.1f} M flops",
      "limit": "25 M (~10 % of one core)", "ok": sum(ops.values()) < 25e6},
     {"item": "packet rate (WiCount median)", "value": f"{rate['rate_hz'].median():.0f} Hz",
      "limit": f">= {ec.FS} Hz", "ok": rate["rate_hz"].median() >= ec.FS}])
display(pd.DataFrame({"bytes": ram}))
budget'''


REPORT = r'''# 10. Save results, figures and the Phase 3 report
results.to_csv(os.path.join(RESULTS, "results.csv"), index=False)
per_rec.to_csv(os.path.join(RESULTS, "recording_predictions.csv"), index=False)
replay.to_csv(os.path.join(RESULTS, "replay_wicount.csv"), index=False)
budget.to_csv(os.path.join(RESULTS, "esp32_budget.csv"), index=False)
rate.to_csv(os.path.join(RESULTS, "wicount_packet_rate.csv"), index=False)

fig_dir = os.path.join(RESULTS, "figures")
os.makedirs(fig_dir, exist_ok=True)

# ---- smoothed MAE per experiment ----
s = results[(results["level"] == "smoothed") & (results["sites"] == "all")].copy()
s["label"] = s["experiment"] + " / " + s["mode"] + " -> " + s["test"]
s = s.sort_values("label")
fig, ax = plt.subplots(figsize=(9, 0.45 * len(s) + 1))
colors = ["tab:green" if (e, m) == CHOSEN else "tab:blue" for e, m in zip(s["experiment"], s["mode"])]
ax.barh(s["label"], s["mae"], color=colors)
ax.set_xlabel("MAE (people), one link, 30 s median")
ax.set_title("Held-out error per experiment (green = final model)")
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, "mae_per_experiment.png"), dpi=120)
plt.show()

# ---- WiCount: whole-file estimate per candidate ----
w = per_rec[per_rec["dataset"] == "wicount"]
fig, ax = plt.subplots(figsize=(6, 5))
for (e, m), h in w.groupby(["experiment", "mode"]):
    jitter = np.random.default_rng(0).uniform(-0.1, 0.1, len(h))
    ax.scatter(h["count"] + jitter, h["pred"], s=25, label=f"{e}/{m}", alpha=0.8)
ax.plot([0, 3], [0, 3], "k--", lw=0.8)
ax.set_xlabel("true people")
ax.set_ylabel("predicted (median of smoothed output)")
ax.set_title("ESP32-S3 (WiCount) held-out files")
ax.legend(fontsize=6)
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, "wicount_recordings.png"), dpi=120)
plt.show()

# ---- markdown report ----
show = ["experiment", "mode", "test", "sites", "train", "n", "mae", "exact", "within1", "spearman_rec"]
lines = [
    "# Phase 3 results - final ESP32-S3 people counter", "",
    f"Generated {time.strftime('%Y-%m-%d %H:%M')}. One link (one view), causal preprocessing, "
    f"{ec.WIN_S} s windows every {ec.HOP_S} s, output = median of the last {ec.SMOOTH_S:.0f} s.", "",
    "## Final model", "",
    f"- Chosen: **{CHOSEN[0]} / {CHOSEN[1]}** (rule: {card['selection']['rule']})",
    f"- Trained on: {', '.join(card['training_data'])} ({card['n_training_windows']} windows), "
    f"counts 0-{model.max_count}",
    f"- Calibration: {card['calibration_howto']}",
    f"- Size: {card['n_trees']} trees, {card['n_nodes']} nodes, ~{card['flash_bytes_est'] / 1024:.0f} KB flash",
    f"- Files: `{MODEL_DIR}/` (trees.json, counter_model.h, model.txt, model_card.json)", "",
    "## Held-out results (smoothed output)", "",
    table[show].round(3).to_markdown(index=False), "",
    "## Candidates on the ESP32-S3 data (WiCount)", "",
    cand.reset_index().round(3).to_markdown(index=False), "",
    "## Replay of the raw ESP32 CSVs through the saved bundle (pipeline check)", "",
    replay.to_markdown(index=False), "",
    "## ESP32-S3 budget (estimates)", "",
    budget.to_markdown(index=False), "",
    "## Caveats", "",
] + [f"- {c}" for c in card["caveats"]]

with open(os.path.join(RESULTS, "PHASE3_RESULTS.md"), "w") as f:
    f.write(chr(10).join(lines))
print("saved", os.path.join(RESULTS, "PHASE3_RESULTS.md"))'''


DONE = r'''# 11. Done - how to use the model
print("Phase 3 finished.")
print("Model bundle :", MODEL_DIR)
print("Report       :", os.path.join(RESULTS, "PHASE3_RESULTS.md"))
print()
print("Python (demo / friend's app):")
print("    import esp32_counter as ec")
print("    model, card = ec.load_bundle(MODEL_DIR)")
print("    counter = ec.StreamingCounter(model)")
if card["calibration"] == "empty_cal":
    print("    counter.begin_calibration()        # room empty for the first 30 s")
print("    for t, amp52 in packets:             # amp52 = |CSI| of the 52 data tones")
print("        for r in counter.push(t, amp52):")
print("            if r['status'] == 'ok': print(r['count'])")
print()
print("ESP32 firmware: #include \"counter_model.h\" -> cc_prepare(raw_features, offset, x); cc_predict(x)")'''


# ===============================================================
# Notebook writer
# ===============================================================
CELLS = [("md", INTRO), ("code", SETUP), ("code", TESTS), ("code", CHECKS), ("code", FEATURES_CELL),
         ("code", HELPERS), ("code", EXPERIMENTS), ("code", FINAL), ("code", REPLAY),
         ("code", BUDGET), ("code", REPORT), ("code", DONE)]

FILENAME = "phase3_esp32_final_local.ipynb"


def main():
    for kind, text in CELLS:
        if kind == "code":
            ast.parse(text)                          # syntax check every code cell
            assert "google.colab" not in text and "/content" not in text

    nb = {"cells": [to_cell(k, t) for k, t in CELLS],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}

    out = os.path.join(ROOT, FILENAME)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out, f"({len(CELLS)} cells)")


if __name__ == "__main__":
    main()

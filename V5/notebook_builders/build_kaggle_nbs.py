"""Generate the Kaggle notebooks for WiMANS, CSI-Bench and EHUNAM.

  kaggle_wimans.ipynb          Phase 0 audit + Phase 1 shared format + Phase 2 baselines (0-5 people)
  kaggle_csibench.ipynb        Phase 0 audit + ESP32-S3 subset -> shared format (for Phase 3 pre-training)
  kaggle_ehunam_explore.ipynb  Phase 0 discovery: Summary.xlsx, file-name codes, .mat keys/shapes

The shared-format helpers are copied from build_phase12_nbs.py, so every dataset is
processed exactly like the local ones (52 tones, 50 Hz, gain + recording normalisation).
Run:  py -3 colab_tools/build_kaggle_nbs.py
"""

import ast
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from build_phase12_nbs import P1_HELPERS, to_cell  # noqa: E402


# ===============================================================
# Shared cells
# ===============================================================
FINDER = r'''# Locate dataset folders under /kaggle/input (works for any dataset slug / nesting)
SKIP_DIRS = {"amp", "mat", "video"}            # huge leaf folders: never walk into them


def find_dirs(test, root="/kaggle/input", max_depth=6):
    """Return folders for which test(folder, file_names) is True (shallow walk)."""
    hits = []
    base = root.rstrip("/").count("/")
    for folder, subdirs, files in os.walk(root):
        if test(folder, files):
            hits.append(folder)
        if folder.count("/") - base >= max_depth:
            subdirs[:] = []
        subdirs[:] = [s for s in subdirs if s not in SKIP_DIRS and not s.startswith("sub_")]
    return sorted(hits)


print("attached datasets:", sorted(os.listdir("/kaggle/input")))'''


SHARED_SETTINGS = r'''# ---- shared-format settings (identical to the local-dataset notebooks) ----
FS = 50              # common sample rate (Hz)
MAX_GAP_BINS = 3     # gaps up to 60 ms are interpolated
MAX_MISSING = 0.20   # samples/windows with more missing bins are dropped
N_TONES = 52'''


TONE_INTERP = r'''# Map any number of subcarriers onto the 52-tone grid (linear interpolation over frequency)
def tone_interp_matrix(n_src, n_dst=N_TONES):
    src = np.linspace(0.0, 1.0, n_src)
    dst = np.linspace(0.0, 1.0, n_dst)
    W = np.zeros((n_dst, n_src), dtype=np.float32)
    for i, x in enumerate(dst):
        j = min(np.searchsorted(src, x, side="right") - 1, n_src - 2)
        w = (x - src[j]) / (src[j + 1] - src[j])
        W[i, j] = 1.0 - w
        W[i, j + 1] = w
    return W'''


METRICS = r'''# Metrics and result collection
RESULT_ROWS, PER_COUNT_ROWS = [], []


def to4(c):
    """Map a people count to the 0 / 1 / 2 / 3+ classes."""
    return np.minimum(c, 3)


def score(y, pred, max_count):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    rounded = np.clip(np.rint(pred), 0, max_count)
    y4, r4 = to4(y), to4(rounded)
    labels4 = sorted(set(y4) | set(r4))                    # classes absent from both are not scored
    return {"n": len(y),
            "mae": float(np.mean(np.abs(pred - y))),
            "exact": float(np.mean(rounded == y)),
            "within1": float(np.mean(np.abs(rounded - y) <= 1)),
            "acc4": float(np.mean(r4 == y4)),
            "f1_4": float(f1_score(y4, r4, average="macro", labels=labels4, zero_division=0))}


def evaluate(setting, model, task, calib, rows, pred):
    """rows: one row per (sample, view) with columns sample, count, count4; pred aligned with it.
    Task 'clf4' predicts the 0/1/2/3+ class directly, so it is scored against count4."""
    target = "count4" if task == "clf4" else "count"
    max_count = 3 if task == "clf4" else MAX_COUNT

    d = rows[["sample", target]].rename(columns={target: "y"})
    d["pred"] = pred
    RESULT_ROWS.append({"setting": setting, "model": model, "task": task, "calib": calib,
                        "level": "single_view", **score(d["y"], d["pred"], max_count)})

    ens = d.groupby("sample").agg(y=("y", "first"), pred=("pred", "mean"))
    RESULT_ROWS.append({"setting": setting, "model": model, "task": task, "calib": calib,
                        "level": "view_ensemble", **score(ens["y"], ens["pred"], max_count)})

    rounded = np.clip(np.rint(ens["pred"].values), 0, max_count)
    for c in sorted(ens["y"].unique()):
        m = ens["y"].values == c
        PER_COUNT_ROWS.append({"setting": setting, "model": model, "task": task, "calib": calib,
                               "count": int(c), "n": int(m.sum()),
                               "recall": float(np.mean(rounded[m] == c)),
                               "mean_pred": float(ens["pred"].values[m].mean())})
    return ens'''


TINY_CNN = r'''# Tiny 1D-CNN (~35k parameters, fits the ESP32-S3 budget). Input (batch, 52 tones, time).
class TinyCNN(nn.Module):
    def __init__(self, n_tones=N_TONES, n_classes=MAX_COUNT + 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(n_tones, 32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 48, kernel_size=5, padding=4, dilation=2), nn.BatchNorm1d(48), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(48, 64, kernel_size=5, padding=4, dilation=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.2),
        )
        self.cls_head = nn.Linear(64, n_classes)
        self.reg_head = nn.Linear(64, 1)

    def forward(self, x):
        h = self.body(x)
        return self.cls_head(h), self.reg_head(h).squeeze(1)


class MultiLinkCNN(nn.Module):
    """Several receivers: the same TinyCNN encoder runs on every link, the link features are
    averaged, then the same heads. Same parameter count; input (batch, links, 52 tones, time)."""

    def __init__(self):
        super().__init__()
        self.net = TinyCNN()

    def forward(self, x):
        b, n_links = x.shape[:2]
        h = self.net.body(x.flatten(0, 1)).view(b, n_links, -1).mean(1)
        return self.net.cls_head(h), self.net.reg_head(h).squeeze(1)


print("TinyCNN parameters:", sum(p.numel() for p in TinyCNN().parameters()))'''


# ===============================================================
# WiMANS
# ===============================================================
W_INTRO = r'''# WiMANS on Kaggle - people counting, aimed at accuracy and ESP32 deployment (v3)

**Data:** WiMANS (Huang et al.), Intel 5300, 3 x 3 antennas, 30 subcarriers, ~1 kHz, 3-s samples,
**0-5 people**, 3 environments x 2 bands (2.4 / 5 GHz). Labels in `annotation.csv`
(`label`, `environment`, `wifi_band`, `number_of_users`, `user_<k>_location`, `user_<k>_activity`).
This notebook uses `wifi_csi/amp/<label>.npy` (amplitude, shape T x 3 x 3 x 30).

**What earlier runs showed:** raw 2.4 GHz amplitudes are white noise packet to packet (lag-1 autocorrelation
~0.00-0.07 at every count), so only **5 GHz** is used from Phase 1 on. On 5 GHz, empty rooms are found ~100 %
of the time, but 3 / 4 / 5 people blur together: total "wobble" saturates after ~2 people.

| Part | What it does |
|---|---|
| Phase 0 | labels, files, shapes, and the per-band signal probe (both bands, to keep the 2.4 GHz evidence) |
| Phase 1 | 5 GHz samples -> 9 link **views**: 30 subcarriers -> 52 tones, 50 Hz, normalised, 10 base count cues |
| Phase 2a | 9 new cues per link aimed at **how many separate movers** (PCA spread, Doppler spread, burstiness, tone spread) + 2 cross-link cues |
| Phase 2b | LightGBM on 3 feature sets, tiny CNN on one link and on all 9 links pooled |
| Phase 2c | deployment: int8 weights, parameters, multiply-adds, ONNX export |

**Targets:** exact count 0-5 and the coarser **0 / 1 / 2 / 3+** classes (`acc4`, macro `f1_4`).

**Test settings** (always a room the model was not trained on):

| Setting | What the installer does in the new room |
|---|---|
| `zero_shot` | nothing |
| `cal_empty` | records 30 s of the empty room |
| `cal_all` | records 30 s for each people count 0-5 |

Calibration data comes from one session per (room, count); that whole session is removed from the test set
in **every** setting, so all three settings are scored on exactly the same samples.

WiMANS samples are 3 s long, so the window here is the whole 3-s sample (150 bins), not 5 s.

**Kaggle settings:** attach your WiMANS dataset (Add Input), turn on **GPU** (T4 or P100) and Internet is not needed.
Outputs go to `/kaggle/working/` (`shared_format/wimans/`, `results/wimans/`). After *Save Version*,
the output can be attached to later notebooks as a dataset.'''


W_SETUP = r'''# 1. Setup and paths
import copy
import glob
import json
import os
import time
from multiprocessing import get_context

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from sklearn.metrics import f1_score

''' + FINDER + r'''

roots = find_dirs(lambda folder, files: "annotation.csv" in files)
assert roots, "annotation.csv not found - is the WiMANS dataset attached?"
WIMANS_ROOT = roots[0]
ANNOT = os.path.join(WIMANS_ROOT, "annotation.csv")
AMP_DIR = os.path.join(WIMANS_ROOT, "wifi_csi", "amp")
if not os.path.isdir(AMP_DIR):
    hits = glob.glob(os.path.join(WIMANS_ROOT, "**", "amp"), recursive=True)
    AMP_DIR = hits[0]

OUT_ROOT = "/kaggle/working/shared_format"
SHARED = os.path.join(OUT_ROOT, "wimans")
RESULTS = "/kaggle/working/results/wimans"
os.makedirs(SHARED, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

''' + SHARED_SETTINGS + r'''
WIN_S = 3.0          # one WiMANS sample = one window
HOP_S = 3.0
WIMANS_RATE = 1000   # nominal packets/s (amp files carry no timestamps)
BANDS_USED = ["5"]   # 2.4 GHz is white noise in WiMANS (see the Phase 0 signal probe)
N_BINS = int(WIN_S * FS)
N_WORKERS = os.cpu_count()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
print("WiMANS root:", WIMANS_ROOT)
print("amp dir    :", AMP_DIR, "  files:", len(os.listdir(AMP_DIR)))
print("device     :", DEVICE, "  cpus:", N_WORKERS)'''


W_AUDIT_LABELS = r'''# 2. Phase 0 - labels
ann = pd.read_csv(ANNOT, dtype=str)
print(ann.shape)
print(list(ann.columns))

ann["count"] = ann["number_of_users"].astype(int)
ann["session"] = ann["label"].str.split("_").str[1]          # act_<session>_<sample>
ann["has_file"] = [os.path.exists(os.path.join(AMP_DIR, l + ".npy")) for l in ann["label"]]
print("annotation rows with an amp file:", ann["has_file"].sum(), "/", len(ann))

display(pd.crosstab([ann["environment"], ann["wifi_band"]], ann["count"], margins=True))

# how many sessions per environment/band, and are counts mixed inside a session?
per_session = ann.groupby("session").agg(env=("environment", "first"), band=("wifi_band", "first"),
                                         n=("label", "size"), counts=("count", "nunique"),
                                         envs=("environment", "nunique"), bands=("wifi_band", "nunique"))
print("sessions:", len(per_session), "| sessions spanning >1 env:", (per_session["envs"] > 1).sum(),
      "| >1 band:", (per_session["bands"] > 1).sum(), "| with >1 count:", (per_session["counts"] > 1).sum())
display(per_session.groupby(["env", "band"]).agg(sessions=("n", "size"), samples=("n", "sum")))

act_cols = [c for c in ann.columns if c.endswith("_activity")]
print("activities:", sorted(set(ann[act_cols].stack().dropna()) - {"nan"}))'''


W_AUDIT_SIGNAL = r'''# 3. Phase 0 - array shapes, value ranges, and time correlation per environment/band
rng = np.random.default_rng(SEED)
probe = (ann[ann["has_file"]].sample(frac=1, random_state=SEED)
         .groupby(["environment", "wifi_band", "count"]).head(8))


def probe_file(label):
    A = np.load(os.path.join(AMP_DIR, label + ".npy")).astype(np.float32)
    x = A.reshape(A.shape[0], -1)
    x = x - x.mean(0)
    num = (x[:-1] * x[1:]).sum(0)
    den = np.sqrt((x[:-1] ** 2).sum(0) * (x[1:] ** 2).sum(0)) + 1e-9
    return {"label": label, "shape": str(A.shape), "T": A.shape[0],
            "min": float(A.min()), "max": float(A.max()), "zeros": int((A == 0).sum()),
            "lag1_autocorr": float(np.median(num / den))}


with get_context("fork").Pool(N_WORKERS) as pool:
    sig = pd.DataFrame(pool.map(probe_file, probe["label"].tolist()))
sig = sig.merge(ann[["label", "environment", "wifi_band", "count", "session"]], on="label")

print("shapes seen:", sig["shape"].value_counts().head().to_dict())
print("T range:", sig["T"].min(), "-", sig["T"].max(), "| max value:", sig["max"].max())
display(sig.groupby(["environment", "wifi_band"])["lag1_autocorr"].describe().round(3))
display(sig.groupby("session")["lag1_autocorr"].median().round(2).sort_values().to_frame().T)
sig.to_csv(os.path.join(RESULTS, "phase0_signal_probe.csv"), index=False)'''


W_HELPERS = P1_HELPERS + "\n\n\n" + TONE_INTERP


W_CONVERT = r'''# 4. Phase 1 - convert every 5 GHz sample to the shared format (9 views x 150 bins x 52 tones)
W30 = tone_interp_matrix(30)
labels = ann.loc[ann["has_file"] & ann["wifi_band"].isin(BANDS_USED), "label"].tolist()
print("samples to convert:", len(labels), "bands:", BANDS_USED)


def convert_sample(label):
    A = np.load(os.path.join(AMP_DIR, label + ".npy")).astype(np.float32)   # (T, 3, 3, 30)
    t = np.arange(A.shape[0]) / WIMANS_RATE
    series = np.full((9, N_BINS, N_TONES), np.nan, dtype=np.float32)
    valid = np.zeros(N_BINS, dtype=bool)
    feats = np.zeros((9, len(FEATURE_NAMES)), dtype=np.float32)

    for v in range(9):
        a, b = divmod(v, 3)
        X, ok = to_shared_series(t, A[:, a, b, :] @ W30.T)
        n = min(len(X), N_BINS)
        series[v, :n] = X[:n]
        valid[:n] = ok[:n]
        feats[v] = window_features(series[v])
    return label, series.astype(np.float16), valid, feats


N = len(labels)
SERIES = np.zeros((N, 9, N_BINS, N_TONES), dtype=np.float16)       # ~0.8 GB for 5 GHz
VALID = np.zeros((N, N_BINS), dtype=bool)
FEATS = np.zeros((N, 9, len(FEATURE_NAMES)), dtype=np.float32)
pos = {l: i for i, l in enumerate(labels)}

t0 = time.time()
with get_context("fork").Pool(N_WORKERS) as pool:
    for k, (label, s, ok, f) in enumerate(pool.imap_unordered(convert_sample, labels, chunksize=16)):
        i = pos[label]
        SERIES[i], VALID[i], FEATS[i] = s, ok, f
        if k % 1000 == 0:
            print(f"{k:6d} / {N}   [{time.time() - t0:.0f} s]", flush=True)

meta = ann.set_index("label").loc[labels].reset_index()
meta["valid_frac"] = VALID.mean(1)
np.save(os.path.join(SHARED, "series.npy"), SERIES)
np.save(os.path.join(SHARED, "feats.npy"), FEATS)
meta.to_parquet(os.path.join(SHARED, "meta.parquet"), index=False)
with open(os.path.join(SHARED, "format.json"), "w") as f:
    json.dump({"fs_hz": FS, "win_s": WIN_S, "n_tones": N_TONES, "views": "tx*3+rx (9 links)",
               "tone_mapping": "30 Intel-5300 subcarriers linearly interpolated to 52",
               "feature_names": FEATURE_NAMES}, f, indent=1)
print(f"done in {time.time() - t0:.0f} s; samples with >= 80 % valid bins:",
      int((meta["valid_frac"] >= 1 - MAX_MISSING).sum()), "/", N)'''


W_CHECK = r'''# 5. Sanity check: does the main count cue rise with people in every environment/band?
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
cue = pd.DataFrame({"count": meta["count"], "env": meta["environment"], "band": meta["wifi_band"],
                    "cv_median": FEATS[:, :, 0].mean(1), "diff_energy": FEATS[:, :, 3].mean(1)})
for ax, col in zip(axes, ["cv_median", "diff_energy"]):
    for (env, band), g in cue.groupby(["env", "band"]):
        m = g.groupby("count")[col].median()
        ax.plot(m.index, m.values, marker="o", label=f"{env}/{band}")
    ax.set_xlabel("people")
    ax.set_ylabel(col + " (median, mean over 9 links)")
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "cues_vs_count.png"), dpi=120)
plt.show()

display(cue.groupby(["env", "band"]).apply(
    lambda g: g["cv_median"].corr(g["count"], method="spearman")).rename("spearman").round(3))'''


EXTRA_CUES = r'''# Extra cues for "how many separate movers" (all cheap enough for an ESP32)
EXTRA_NAMES = [
    "pc2_frac",          # share of variance in the 2nd principal component
    "pc3_frac",          # ... and in the 3rd (several movers spread variance over components)
    "eig_entropy",       # entropy of the PCA spectrum = effective number of independent motions
    "doppler_centroid",  # mean temporal frequency of the motion (Hz)
    "doppler_spread",    # spread of the temporal frequencies (Hz)
    "psd_entropy",       # flatness of the temporal spectrum (0..1)
    "activity_cv",       # burstiness of motion over 0.2-s segments (one mover: bursty, many: steady)
    "tone_energy_cv",    # how unevenly motion is spread over the 52 tones
    "half_band_corr",    # correlation of motion in the lower vs upper 26 tones
]
LINK_NAMES = [
    "link_env_corr",     # mean correlation of the 9 links' motion envelopes (needs several receivers)
    "link_energy_cv",    # spread of motion energy over the 9 links
]


def extra_features(W):
    """W: (bins, 52) normalised window -> (extra cues, motion envelope over time)."""
    W = np.nan_to_num(W.astype(np.float32))
    Xc = W - W.mean(axis=0)

    # spread of the motion over principal components
    s = np.linalg.svd(Xc, compute_uv=False) ** 2
    frac = s / (s.sum() + 1e-12)
    eig_entropy = -(frac * np.log(frac + 1e-12)).sum()

    # temporal (Doppler) spectrum summed over tones, DC removed
    P = (np.abs(np.fft.rfft(Xc, axis=0)) ** 2).sum(axis=1)[1:]
    f = np.fft.rfftfreq(len(W), 1.0 / FS)[1:]
    p = P / (P.sum() + 1e-12)
    centroid = (p * f).sum()
    spread = np.sqrt((p * (f - centroid) ** 2).sum())
    psd_entropy = -(p * np.log(p + 1e-12)).sum() / np.log(len(p))

    # motion = absolute change between bins
    d = np.abs(np.diff(W, axis=0))
    env = d.mean(axis=1)
    seg = env[:len(env) // 10 * 10].reshape(-1, 10).mean(axis=1)
    activity_cv = seg.std() / (seg.mean() + 1e-9)
    tone_e = d.mean(axis=0)
    tone_energy_cv = tone_e.std() / (tone_e.mean() + 1e-9)

    half = W.shape[1] // 2
    lo, hi = d[:, :half].mean(axis=1), d[:, half:].mean(axis=1)
    half_band_corr = np.corrcoef(lo, hi)[0, 1] if lo.std() > 0 and hi.std() > 0 else 0.0

    cues = np.array([frac[1], frac[2], eig_entropy, centroid, spread, psd_entropy,
                     activity_cv, tone_energy_cv, half_band_corr], dtype=np.float32)
    return cues, env


def link_cues(envs):
    """Cross-link cues from the motion envelopes of all links of one sample."""
    E = np.array(envs)
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(E)
    energy = E.mean(axis=1)
    return np.array([np.nanmean(C[np.triu_indices(len(E), k=1)]),
                     energy.std() / (energy.mean() + 1e-9)], dtype=np.float32)'''


W_EXTRA = r'''# 6. Phase 2a - extra cues for "how many separate movers" (all cheap enough for an ESP32)
''' + EXTRA_CUES + r'''


def sample_extra(i):
    cues, envs = [], []
    for v in range(9):
        c, e = extra_features(SERIES[i, v])
        cues.append(c)
        envs.append(e)
    return i, np.array(cues), link_cues(envs)


EXTRA = np.zeros((N, 9, len(EXTRA_NAMES)), dtype=np.float32)
LINK = np.zeros((N, len(LINK_NAMES)), dtype=np.float32)
t0 = time.time()
with get_context("fork").Pool(N_WORKERS) as pool:
    for i, cues, link in pool.imap_unordered(sample_extra, range(N), chunksize=64):
        EXTRA[i], LINK[i] = cues, link
print(f"extra cues for {N} samples in {time.time() - t0:.0f} s")
np.save(os.path.join(SHARED, "extra_feats.npy"), EXTRA)
np.save(os.path.join(SHARED, "link_feats.npy"), LINK)

# how each cue moves with the people count, per room (median over samples, mean over links)
cue = pd.DataFrame(EXTRA.mean(axis=1), columns=EXTRA_NAMES)
cue[LINK_NAMES] = LINK
cue["count"], cue["env"] = meta["count"].values, meta["environment"].values
names = EXTRA_NAMES + LINK_NAMES
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
for ax, col in zip(axes.flat, names):
    for env, g in cue.groupby("env"):
        m = g.groupby("count")[col].median()
        ax.plot(m.index, m.values, marker="o", label=env)
    ax.set_title(col, fontsize=9)
    ax.set_xlabel("people")
axes.flat[0].legend(fontsize=7)
for ax in axes.flat[len(names):]:
    ax.axis("off")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "extra_cues_vs_count.png"), dpi=110)
plt.show()

spearman = pd.DataFrame({env: [g[c].corr(g["count"], method="spearman") for c in names]
                         for env, g in cue.groupby("env")}, index=names).round(3)
spearman.to_csv(os.path.join(RESULTS, "extra_cue_spearman.csv"))
display(spearman)'''


W_TABLE = r'''# 7. Phase 2 - feature table (one row per sample x link), calibration sessions, settings
keep = (meta["valid_frac"] >= 1 - MAX_MISSING).values
idx = np.flatnonzero(keep)
MAX_COUNT = int(meta["count"].max())

rows = pd.DataFrame({"sample": np.repeat(idx, 9), "view": np.tile(np.arange(9), len(idx))})
for c in ["count", "environment", "session"]:
    rows[c] = meta[c].values[rows["sample"]]
rows["count4"] = np.minimum(rows["count"], 3)
rows["sample_no"] = meta["label"].str.split("_").str[2].astype(int).values[rows["sample"]]
rows[FEATURE_NAMES] = np.nan_to_num(FEATS[idx].reshape(-1, len(FEATURE_NAMES)))
rows[EXTRA_NAMES] = np.nan_to_num(EXTRA[idx].reshape(-1, len(EXTRA_NAMES)))
rows[LINK_NAMES] = np.nan_to_num(np.repeat(LINK[idx], 9, axis=0))
rows["calib_group"] = rows["environment"]
ENVS = sorted(rows["environment"].unique())

FEATURE_SETS = {
    "base10": FEATURE_NAMES,                                    # the 10 cues of the earlier runs
    "single_link": FEATURE_NAMES + EXTRA_NAMES,                 # everything one ESP32 link can compute
    "multi_link": FEATURE_NAMES + EXTRA_NAMES + LINK_NAMES,     # + cues that need several receivers
}

# ---- install-time calibration: one session per (room, count), first 30 s of it ----
CAL_SAMPLES = 10                     # 10 samples x 3 s = 30 s per people count
sessions = rows.drop_duplicates("session")[["environment", "count", "session"]]
per_group = sessions.groupby(["environment", "count"])["session"].apply(sorted)
print("sessions per room and count:")
display(per_group.apply(len).unstack())

rng = np.random.default_rng(SEED)
CAL_SESSIONS = {s[rng.integers(len(s))] for s in per_group if len(s) >= 2}   # keep >= 1 session to test
rows["cal_session"] = rows["session"].isin(CAL_SESSIONS)
rank = rows.groupby("session")["sample_no"].rank(method="dense")
rows["cal_pick"] = rows["cal_session"] & (rank <= CAL_SAMPLES)

SETTINGS = {"zero_shot": None, "cal_empty": [0], "cal_all": list(range(MAX_COUNT + 1))}
VIEW0 = (rows["view"] == 0).values


def folds(setting):
    """Leave one room out. Calibration sessions are never tested; in the cal_* settings their
    first 30 s (only the listed counts) are added to the training data."""
    cal_counts = SETTINGS[setting]
    for env in ENVS:
        in_env = (rows["environment"] == env).values
        test = in_env & ~rows["cal_session"].values
        train = ~in_env
        if cal_counts is not None:
            train = train | (in_env & rows["cal_pick"].values & rows["count"].isin(cal_counts).values)
        yield env, train, test


print(len(idx), "samples,", len(rows), "rows,", len(CAL_SESSIONS), "calibration sessions,",
      "test samples:", int((~rows["cal_session"] & VIEW0).sum()))'''


GBM_DEFS = r'''GBM_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  random_state=SEED, verbose=-1)


def calibrate(df, feats):
    """z-score each feature within (room, link) using that room's own unlabelled samples."""
    out = df.copy()
    g = out.groupby(["calib_group", "view"])[feats]
    out[feats] = (out[feats] - g.transform("mean")) / (g.transform("std") + 1e-6)
    return out'''


W_GBM = r'''# 9. Phase 2b - LightGBM: 3 feature sets x 3 settings x (6-class, 4-class, regression)
''' + GBM_DEFS + r'''


IMPORTANCE = []
t0 = time.time()
for calib in ["none", "group_z"]:
    for fs_name, feats in FEATURE_SETS.items():
        d = calibrate(rows, feats) if calib == "group_z" else rows
        for setting in SETTINGS:
            preds = {k: np.full(len(d), np.nan) for k in ["clf", "clf4", "reg", "mean"]}
            for env, train, test in folds(setting):
                X_tr, X_te = d.loc[train, feats], d.loc[test, feats]
                y_tr = d.loc[train, "count"]

                preds["clf"][test] = lgb.LGBMClassifier(**GBM_PARAMS).fit(X_tr, y_tr).predict(X_te)
                preds["clf4"][test] = lgb.LGBMClassifier(**GBM_PARAMS).fit(
                    X_tr, d.loc[train, "count4"]).predict(X_te)
                reg = lgb.LGBMRegressor(**GBM_PARAMS).fit(X_tr, y_tr)
                preds["reg"][test] = reg.predict(X_te)
                preds["mean"][test] = y_tr.mean()
                if setting == "zero_shot":
                    IMPORTANCE.append(pd.Series(reg.booster_.feature_importance("gain"), index=feats,
                                                name=f"{fs_name}|{calib}|{env}"))

            tested = ~np.isnan(preds["reg"])
            for task in ["clf", "clf4", "reg"]:
                evaluate(setting, f"gbm_{fs_name}", task, calib, d[tested], preds[task][tested])
            if calib == "none" and fs_name == "base10":
                evaluate(setting, "mean_baseline", "reg", "none", d[tested], preds["mean"][tested])
        print(f"calib={calib:8s} features={fs_name:12s} [{time.time() - t0:.0f} s]", flush=True)

res = pd.DataFrame(RESULT_ROWS)
display(res[res["level"] == "view_ensemble"].sort_values("f1_4", ascending=False).head(15).round(3))'''


CNN_DEFS = r'''EPOCHS = 15
BATCH = 256
SAMPLES_PER_EPOCH = 40000      # link windows per epoch; the multi-link model sees the same total
LR = 2e-3
CAL_SHARE = 0.2                # share of each epoch drawn from the new room's calibration samples


def load_x(samples, views=None, train=False, rng=None):
    """(batch, 52, 150) for one link per row, or (batch, 9, 52, 150) for all links (views=None)."""
    x = SERIES[samples] if views is None else SERIES[samples, views]
    x = np.nan_to_num(x.astype(np.float32)).clip(-3, 3)
    if train:
        # light augmentation against room changes: overall scale, per-tone gain, time shift
        lead = x.shape[:-2]
        x = x * rng.uniform(0.85, 1.15, size=lead + (1, 1))
        x = x * np.exp(rng.normal(0.0, 0.1, size=lead + (1, x.shape[-1])))
        x = np.roll(x, int(rng.integers(x.shape[-2])), axis=-2)
    x = np.ascontiguousarray(np.swapaxes(x, -1, -2), dtype=np.float32)    # time last
    return torch.from_numpy(x).to(DEVICE)


def epoch_weights(train_df, cal_mask):
    """Balanced over people counts; the new room's calibration rows get CAL_SHARE of each epoch."""
    w = 1.0 / train_df["count"].map(train_df["count"].value_counts()).values
    if cal_mask.any():
        w_cal, w_rest = w[cal_mask].sum(), w[~cal_mask].sum()
        w[cal_mask] *= CAL_SHARE * w_rest / ((1 - CAL_SHARE) * w_cal)
    return w / w.sum()


def train_model(kind, train_df, cal_mask):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = (TinyCNN() if kind == "single" else MultiLinkCNN()).to(DEVICE)
    model.seen = torch.zeros(MAX_COUNT + 1, dtype=torch.bool, device=DEVICE)
    model.seen[torch.tensor(sorted(train_df["count"].unique()), device=DEVICE)] = True

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = SAMPLES_PER_EPOCH if kind == "single" else SAMPLES_PER_EPOCH // N_VIEWS
    batch = BATCH if kind == "single" else BATCH // 8
    p = epoch_weights(train_df, cal_mask)

    for epoch in range(EPOCHS):
        model.train()
        ep = train_df.iloc[rng.choice(len(train_df), size=n, replace=True, p=p)]
        for b in range(0, n, batch):
            sub = ep.iloc[b:b + batch]
            if len(sub) < 2:                                   # BatchNorm needs > 1 sample
                continue
            views = sub["view"].values if kind == "single" else None
            y = torch.tensor(sub["count"].values, device=DEVICE)
            logits, reg = model(load_x(sub["sample"].values, views, train=True, rng=rng))
            loss = Fnn.cross_entropy(logits, y) + Fnn.huber_loss(reg, y.float())
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return model


def predict(model, kind, df):
    model.eval()
    step = 2048 if kind == "single" else 256
    p_cls, p_reg = [], []
    with torch.no_grad():
        for b in range(0, len(df), step):
            sub = df.iloc[b:b + step]
            views = sub["view"].values if kind == "single" else None
            logits, reg = model(load_x(sub["sample"].values, views))
            p_cls.append(logits.masked_fill(~model.seen, -1e9).argmax(1).cpu().numpy())
            p_reg.append(reg.cpu().numpy())
    return np.concatenate(p_cls), np.concatenate(p_reg)


def int8_copy(model):
    """Weights rounded to int8 (symmetric, per output channel), as stored on the ESP32."""
    q = copy.deepcopy(model)
    with torch.no_grad():
        for m in q.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                w = m.weight
                s = w.abs().amax(dim=tuple(range(1, w.dim())), keepdim=True) / 127 + 1e-12
                m.weight.copy_(torch.round(w / s) * s)
    return q'''


W_CNN = r'''# 11. Phase 2b - tiny CNNs: one link (one ESP32) and all 9 links pooled (several receivers)
N_VIEWS = 9                    # links per sample
''' + CNN_DEFS + r'''


t0 = time.time()
for kind in ["single", "multi"]:
    for setting in SETTINGS:
        out = {(v, t): np.full(len(rows), np.nan) for v in ["fp32", "int8"] for t in ["clf", "reg"]}
        for env, train, test in folds(setting):
            if kind == "multi":                                # one row per sample
                train, test = train & VIEW0, test & VIEW0
            tr = rows[train]
            model = train_model(kind, tr, (tr["environment"] == env).values)
            for v, m in [("fp32", model), ("int8", int8_copy(model))]:
                out[(v, "clf")][test], out[(v, "reg")][test] = predict(m, kind, rows[test])
            print(f"cnn_{kind:6s} {setting:9s} left out {env:14s} [{time.time() - t0:.0f} s]", flush=True)

        tested = ~np.isnan(out[("fp32", "reg")])
        for (v, task), pred in out.items():
            name = f"cnn_{kind}" + ("_int8" if v == "int8" else "")
            evaluate(setting, name, task, "none", rows[tested], pred[tested])

res = pd.DataFrame(RESULT_ROWS)
display(res[res["model"].str.startswith("cnn") & (res["level"] == "view_ensemble")]
        .sort_values(["setting", "f1_4"], ascending=[True, False]).round(3))'''


W_DEPLOY = r'''# 12. Phase 2c - deployment: final CNNs on all 5 GHz data, size, compute, ONNX export
def count_macs(model, x):
    """Multiply-adds for one inference (x has batch size 1)."""
    macs = []

    def hook(m, inp, out):
        if isinstance(m, nn.Conv1d):
            macs.append(out.numel() * (m.in_channels // m.groups) * m.kernel_size[0])
        else:
            macs.append(out.numel() * m.in_features)

    hooks = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv1d, nn.Linear))]
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return int(sum(macs))


def export_onnx(model, x, path):
    kwargs = dict(input_names=["csi"], output_names=["count_logits", "count_reg"], opset_version=17)
    try:
        torch.onnx.export(model, x, path, dynamo=False, **kwargs)
    except TypeError:                                          # older torch: no 'dynamo' argument
        torch.onnx.export(model, x, path, **kwargs)


DEPLOY = {}
for kind in ["single", "multi"]:
    tr = rows if kind == "single" else rows[VIEW0]
    model = train_model(kind, tr, np.zeros(len(tr), dtype=bool))
    model.eval()
    shape = (1, N_TONES, N_BINS) if kind == "single" else (1, N_VIEWS, N_TONES, N_BINS)
    x = torch.zeros(shape, device=DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    DEPLOY[kind] = {"input": list(shape[1:]), "window_s": WIN_S, "params": n_params,
                    "macs_per_window": count_macs(model, x),
                    "weights_fp32_kb": round(n_params * 4 / 1024, 1),
                    "weights_int8_kb": round(n_params / 1024, 1)}
    torch.save(model.state_dict(), os.path.join(RESULTS, f"tinycnn_{kind}_5ghz.pt"))
    try:
        export_onnx(model.cpu(), x.cpu(), os.path.join(RESULTS, f"tinycnn_{kind}_5ghz.onnx"))
        DEPLOY[kind]["onnx"] = "ok"
    except Exception as e:                                     # report, don't stop
        DEPLOY[kind]["onnx"] = repr(e)[:200]

with open(os.path.join(RESULTS, "deployment.json"), "w") as f:
    json.dump(DEPLOY, f, indent=1)
display(pd.DataFrame(DEPLOY).T)'''


W_SAVE = r'''# 13. Save results, plots and a short report
results = pd.DataFrame(RESULT_ROWS)
per_count = pd.DataFrame(PER_COUNT_ROWS)
importance = pd.concat(IMPORTANCE, axis=1)
results.to_csv(os.path.join(RESULTS, "results.csv"), index=False)
per_count.to_csv(os.path.join(RESULTS, "per_count_recall.csv"), index=False)
importance.to_csv(os.path.join(RESULTS, "gbm_feature_importance.csv"))

cols = ["setting", "model", "task", "calib", "mae", "exact", "within1", "acc4", "f1_4", "n"]
ens = results[results["level"] == "view_ensemble"]
best = (ens.sort_values("f1_4", ascending=False)
        .groupby(["setting", "model"]).head(1)
        .sort_values(["setting", "f1_4"], ascending=[True, False]))

# what ONE ESP32 link sees: models that use a single link, scored per link
single_models = ["gbm_base10", "gbm_single_link", "cnn_single", "cnn_single_int8"]
one_link = results[(results["level"] == "single_view") & results["model"].isin(single_models)]
one_link_best = (one_link.sort_values("f1_4", ascending=False)
                 .groupby(["setting", "model"]).head(1)
                 .sort_values(["setting", "f1_4"], ascending=[True, False]))

# ---- plot 1: 0/1/2/3+ macro F1 per model and setting ----
pivot = best.pivot(index="model", columns="setting", values="f1_4")[list(SETTINGS)]
ax = pivot.sort_values("zero_shot").plot.barh(figsize=(9, 6))
ax.set_xlabel("macro F1, 0 / 1 / 2 / 3+ (view ensemble, best task + calib)")
ax.set_xlim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "f1_4_by_model.png"), dpi=120)
plt.show()

# ---- plot 2: per-count recall of the best 6-class model per setting ----
fig, axes = plt.subplots(1, len(SETTINGS), figsize=(6 * len(SETTINGS), 4))
for ax, setting in zip(axes, SETTINGS):
    top = (ens[(ens["setting"] == setting) & (ens["task"] != "clf4")]
           .sort_values("mae").drop_duplicates("model").head(4))
    for _, r in top.iterrows():
        h = per_count[(per_count["setting"] == setting) & (per_count["model"] == r["model"])
                      & (per_count["task"] == r["task"]) & (per_count["calib"] == r["calib"])]
        ax.plot(h["count"], h["recall"], marker="o", label=f"{r['model']}/{r['task']}/{r['calib']}")
    ax.set_title(f"{setting}: per-count recall (4 lowest-MAE models)")
    ax.set_xlabel("people")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "per_count_recall.png"), dpi=120)
plt.show()

imp = importance.copy()
imp.columns = [c.split("|")[0] for c in imp.columns]
imp = imp.T.groupby(level=0).mean().T
imp = (imp / imp.sum()).round(3)

lines = ["# WiMANS - people counting v3 (5 GHz, unseen room)", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')}; 3-s windows, {FS} Hz, 52 tones, 9 links per sample.", "",
         "## Best per model and setting (all 9 links averaged; best task + calibration by 0/1/2/3+ F1)", "",
         best[cols].round(3).to_markdown(index=False), "",
         "## One ESP32 link (single-link models, scored per link)", "",
         one_link_best[cols].round(3).to_markdown(index=False), "",
         "## Deployment (final models trained on all 5 GHz samples)", "",
         pd.DataFrame(DEPLOY).T.to_markdown(), "",
         "## LightGBM feature importance (share of gain, zero-shot)", "", imp.to_markdown(), "",
         "## Caveats", "",
         "- Intel 5300 30 grouped subcarriers were interpolated to 52 tones; no ESP32 hardware involved.",
         "- group_z calibration uses all unlabelled samples of the test room (slightly optimistic).",
         "- cal_* settings add the first 30 s of one session per count (cal_empty: count 0 only); those sessions are never tested.",
         "- int8 rows quantise weights only; activations stay float. BatchNorm would be folded into the convs on device.",
         "- multi-link models need all 9 links of a sample at once (several receivers).",
         "- Packet times are assumed uniform at 1 kHz (amp files have no timestamps)."]
with open(os.path.join(RESULTS, "WIMANS_RESULTS.md"), "w") as f:
    f.write(chr(10).join(lines))
display(best[cols].round(3))
display(one_link_best[cols].round(3))'''


# ===============================================================
# CSI-Bench
# ===============================================================
C_INTRO = r'''# CSI-Bench on Kaggle - Phase 0 audit + ESP32-S3 subset to the shared format

**Data:** CSI-Bench (Zhu et al., NeurIPS 2025): 26 rooms, 35 users, commodity devices, amplitude only.
Tasks: `FallDetection`, `BreathingDetection`, `Localization`, `MotionSourceRecognition`, and `Multitask`
(`HumanActivityRecognition`, `HumanIdentification`, `ProximityRecognition`, sharing `sub_Human_h5/`).
Each task folder has `metadata/sample_metadata.csv`, `metadata/label_mapping.json`, `splits/*.json`
and `.h5` files (key `CSI_amps`, shape **subcarriers x time x 1**, e.g. ESP32 `(64, 500, 1)`) at
`sub_Human/user_<U>/act_<A>/env_<E>/device_<D>/session_<T>__freq<F>.h5`.
The Multitask tasks share `Multitask/sub_Human_h5/`; `RawContinuousRecording/` holds long recordings
(`U01_E01` ... `U06_E06`, `FloorPlan/`, `Groundtruth.xlsx`, `Readme.txt`).

**Count labels:** only `Localization` has them indirectly - its labels are zone bitmasks
(`0, 1, 10, 100, 11, 101, 110`), so the number of 1s = people (0-2). Everything else is for **Phase 3
self-supervised pre-training**. The ESP32 part turned out to be small (494 FallDetection samples, 64 subcarriers, ~100 Hz).

**Fixes since the first run:** `.h5` arrays are read as (subcarriers, time) - the first run had the axes
swapped, so the ESP32 shared format was wrong; Multitask file paths are now resolved under `Multitask/`.

| Part | What it does |
|---|---|
| Phase 0 | per task: metadata columns, label/device/user/environment counts, split sizes, `.h5` shapes and value ranges per device, Localization people counts, RawContinuousRecording contents + full Groundtruth.xlsx |
| Phase 1 | every ESP32-S3 sample -> 52 tones, 50 Hz, normalised, 5-s window, plus the 10 count cues; saved with its task/user/env/session for later |
| **v3: Localization counting** | layout check (168 subcarriers = 3 devices x 56?), 3 links per sample -> shared format + base and "separate movers" cues; LightGBM and tiny CNNs, leave one room out, counts 0-2 |
| **v3: cross-dataset** | WiMANS (5 GHz, 0-2 people) <-> CSI-Bench Localization, base cues, per-room z-score; needs the WiMANS notebook output attached |
| **v3: ESP32 reality check** | base cues of the 494 real ESP32 clips next to Localization devices and WiMANS |

**Kaggle settings:** attach CSI-Bench and the output of `wifi-wimans-counting`; GPU for the CNNs.
Outputs go to `/kaggle/working/` (`csibench_audit/`, `shared_format/csibench_esp32/`,
`shared_format/csibench_loc/`, `results/csibench/`).'''


C_SETUP = r'''# 1. Setup and task discovery
import copy
import glob
import json
import os
import re
import time
from multiprocessing import get_context

import h5py
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from sklearn.metrics import f1_score

''' + FINDER + r'''

TASK_DIRS = find_dirs(lambda folder, files: os.path.exists(
    os.path.join(folder, "metadata", "sample_metadata.csv")), max_depth=7)
assert TASK_DIRS, "no <task>/metadata/sample_metadata.csv found - is CSI-Bench attached?"
for d in TASK_DIRS:
    print(d)

OUT_ROOT = "/kaggle/working/shared_format"
AUDIT = "/kaggle/working/csibench_audit"
SHARED = os.path.join(OUT_ROOT, "csibench_esp32")
SHARED_LOC = os.path.join(OUT_ROOT, "csibench_loc")
RESULTS = "/kaggle/working/results/csibench"
for folder in [AUDIT, SHARED, SHARED_LOC, RESULTS]:
    os.makedirs(folder, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
print("device:", DEVICE)

''' + SHARED_SETTINGS + r'''
WIN_S = 5.0                   # CSI-Bench samples are 5 s (10 s for breathing)
HOP_S = 5.0
DEVICE_PATTERN = "esp32"      # case-insensitive match on the device name
ASSUMED_RATE_HZ = 100         # checked against the sample lengths in cell 3
N_WORKERS = os.cpu_count()

PATH_FIELDS = re.compile(r"user_(?P<user>[^/]+)/act_(?P<act>[^/]+)/env_(?P<env>[^/]+)/"
                         r"device_(?P<device>[^/]+)/session_(?P<session>[^/]+?)__freq(?P<freq>[^/]+?)\.h5$")


def read_amps(f):
    """CSI-Bench stores CSI_amps as (subcarriers, time, 1). Return (time, subcarriers)."""
    key = "CSI_amps" if "CSI_amps" in f else sorted(f.keys())[0]
    a = np.asarray(f[key], dtype=np.float32)
    if a.ndim == 3:
        a = a[:, :, 0]
    return a.T


def from_sub_folder(p):
    """Cut a metadata file_path down to the part starting at its first 'sub_*' folder."""
    parts = p.replace("\\", "/").split("/")
    for i, part in enumerate(parts):
        if part.startswith("sub_"):
            return "/".join(parts[i:])
    return p.lstrip("./")


def resolve_paths(task_dir, rel_paths, n_test=20):
    """Find which folder (the task folder or one of its parents) the file paths are relative to."""
    tail = [from_sub_folder(p) for p in rel_paths]
    base = task_dir
    while base.count("/") > 2:
        hits = sum(os.path.exists(os.path.join(base, t)) for t in tail[:n_test])
        if hits >= max(1, min(n_test, len(tail)) // 2):
            return [os.path.join(base, t) for t in tail], base
        base = os.path.dirname(base)
    print("  WARNING: could not resolve file paths for", task_dir)
    return [os.path.normpath(os.path.join(task_dir, p)) for p in rel_paths], None'''


C_AUDIT_META = r'''# 2. Phase 0 - metadata, labels, devices and splits per task
def task_name(d):
    parts = d.rstrip("/").split("/")
    return parts[-1] if parts[-2] != "Multitask" else "Multitask/" + parts[-1]


all_meta, audit = [], {}
for d in TASK_DIRS:
    name = task_name(d)
    md = pd.read_csv(os.path.join(d, "metadata", "sample_metadata.csv"))
    print("=" * 100)
    print(name, md.shape)
    print("columns:", list(md.columns))
    display(md.head(3))

    # fields encoded in the file path (user / activity / environment / device / session / freq)
    fp = md["file_path"].astype(str) if "file_path" in md.columns else pd.Series([""] * len(md))
    parsed = fp.str.extract(PATH_FIELDS)
    for c in parsed.columns:
        if c not in md.columns:
            md[c] = parsed[c]
    md["abs_path"], base = resolve_paths(d, fp.tolist())
    print("  files resolved under:", base)
    md["task"] = name

    # Localization labels are zone bitmasks: number of 1s = number of people
    if name == "Localization":
        md["count"] = md["label"].astype(str).str.count("1")
        print("  people count from label bitmask:")
        display(pd.crosstab(md["count"], md.get("is_multi_subject", "?"), margins=True))

    low_card = [c for c in md.columns if 1 < md[c].nunique() <= 40 and c not in ("abs_path", "file_path")]
    for c in low_card:
        print(f"  {c}: {md[c].value_counts().to_dict()}")

    splits = {}
    for sp in sorted(glob.glob(os.path.join(d, "splits", "*.json"))):
        ids = json.load(open(sp))
        splits[os.path.basename(sp)[:-5]] = len(ids)
    print("  splits:", splits)
    mapping = os.path.join(d, "metadata", "label_mapping.json")
    if os.path.exists(mapping):
        print("  label_mapping:", json.load(open(mapping)))

    audit[name] = {"rows": len(md), "columns": list(md.columns), "splits": splits, "files_base": base,
                   "value_counts": {c: md[c].astype(str).value_counts().to_dict() for c in low_card}}
    all_meta.append(md)

meta = pd.concat(all_meta, ignore_index=True)
if "device" not in meta.columns:
    meta["device"] = "?"
meta["is_esp32"] = meta["device"].astype(str).str.contains(DEVICE_PATTERN, case=False, na=False)
print("=" * 100)
display(pd.crosstab(meta["task"], meta["device"].fillna("?"), margins=True))
with open(os.path.join(AUDIT, "metadata_audit.json"), "w") as f:
    json.dump(audit, f, indent=1, default=str)'''


C_AUDIT_H5 = r'''# 3. Phase 0 - .h5 contents per (task, device): keys, shapes, value ranges
def probe_h5(path):
    out = {"abs_path": path}
    try:
        with h5py.File(path, "r") as f:
            out["keys"] = ",".join(sorted(f.keys()))
            out["shape"] = str(f["CSI_amps"].shape if "CSI_amps" in f else "?")
            a = read_amps(f)                                 # (time, subcarriers)
        out.update({"T": a.shape[0], "F": a.shape[1],
                    "min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()),
                    "has_negative": bool((a < 0).any()), "nan": int(np.isnan(a).sum())})
        tone_mean = np.abs(a).mean(0)
        out["dead_tones"] = int((tone_mean < 1e-3 * np.median(tone_mean + 1e-9)).sum())

        # the subcarrier count should match the freq<N> token in the file name
        m = re.search(r"__freq(\d+)\.h5$", path)
        out["F_matches_name"] = bool(m) and int(m.group(1)) == a.shape[1]
    except Exception as e:                                   # report, don't stop
        out["error"] = repr(e)[:200]
    return out


probe = (meta.drop_duplicates("abs_path").sample(frac=1, random_state=0)
         .groupby(["task", "device"]).head(6))
with get_context("fork").Pool(N_WORKERS) as pool:
    h5 = pd.DataFrame(pool.map(probe_h5, probe["abs_path"].tolist()))
h5 = h5.merge(probe[["abs_path", "task", "device"]], on="abs_path")
h5.to_csv(os.path.join(AUDIT, "h5_probe.csv"), index=False)

summary = h5.groupby(["task", "device"]).agg(files=("abs_path", "size"), shapes=("shape", lambda s: sorted(set(s))),
                                             T_min=("T", "min"), T_max=("T", "max"), F=("F", "max"),
                                             vmin=("min", "min"), vmax=("max", "max"),
                                             negative=("has_negative", "any"), dead_tones=("dead_tones", "max"),
                                             F_matches_name=("F_matches_name", "mean"))
display(summary)
if "error" in h5.columns:
    errors = h5[h5["error"].notna()]
    print("unreadable files:", len(errors), "/", len(h5))
    display(errors[["task", "device", "error"]].head(20))'''


C_AUDIT_RAW = r'''# 3b. Phase 0 - RawContinuousRecording: readme, ground truth, file types per folder
CSI_ROOT = os.path.dirname([d for d in TASK_DIRS if "/Multitask/" not in d][0])
RAW = os.path.join(CSI_ROOT, "RawContinuousRecording")
raw_audit = {"exists": os.path.isdir(RAW)}

if os.path.isdir(RAW):
    for name in ["Readme.txt", "README.md"]:
        p = os.path.join(RAW, name)
        if os.path.exists(p):
            print("=" * 100, "\n", name)
            print(open(p, errors="replace").read()[:4000])

    gt = os.path.join(RAW, "Groundtruth.xlsx")
    if os.path.exists(gt):
        sheets = pd.read_excel(gt, sheet_name=None)
        raw_audit["groundtruth"] = {}
        for s, df in sheets.items():
            print("=" * 100, "\n", "Groundtruth.xlsx sheet:", s, df.shape)
            with pd.option_context("display.max_colwidth", 200, "display.width", 250):
                print(df.to_string())                        # every row: the event log is short
            df.to_csv(os.path.join(AUDIT, f"groundtruth_{s}.csv"), index=False)
            raw_audit["groundtruth"][s] = {"shape": list(df.shape), "columns": [str(c) for c in df.columns]}

    raw_audit["folders"] = {}
    for sub in sorted(os.listdir(RAW)):
        p = os.path.join(RAW, sub)
        if not os.path.isdir(p):
            continue
        exts, n, size = {}, 0, 0
        for folder, _, files in os.walk(p):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                exts[ext] = exts.get(ext, 0) + 1
                n += 1
                size += os.path.getsize(os.path.join(folder, fn))
        raw_audit["folders"][sub] = {"files": n, "gb": round(size / 1e9, 2), "extensions": exts}
        print(f"{sub:12s} files={n:6d}  {size / 1e9:6.2f} GB  {exts}")

with open(os.path.join(AUDIT, "raw_continuous_audit.json"), "w") as f:
    json.dump(raw_audit, f, indent=1, default=str)'''


C_HELPERS = P1_HELPERS + "\n\n\n" + TONE_INTERP


C_CONVERT = r'''# 4. Phase 1 - ESP32-S3 samples -> shared format (1 view x 250 bins x 52 tones)
esp = meta[meta["is_esp32"]].drop_duplicates("abs_path").reset_index(drop=True)
print(len(esp), "unique ESP32 samples in", esp["task"].nunique(), "tasks")
N_BINS = int(WIN_S * FS)


def esp32_tone_mask(paths, n_probe=200):
    """Tones whose mean amplitude is ~0 across many samples are null/guard tones: drop them."""
    acc = None
    for p in paths[:n_probe]:
        with h5py.File(p, "r") as f:
            a = read_amps(f)                                         # (time, subcarriers)
        m = np.abs(a).mean(0)
        acc = m if acc is None else acc + m
    return acc > 0.02 * np.median(acc)


KEEP_TONES = esp32_tone_mask(esp["abs_path"].sample(frac=1, random_state=0).tolist())
W_ESP = tone_interp_matrix(int(KEEP_TONES.sum()))
print("ESP32 tones kept:", int(KEEP_TONES.sum()), "of", len(KEEP_TONES), "-> mapped to", N_TONES)
assert len(KEEP_TONES) <= 128, "more than 128 'tones' - the time and subcarrier axes are swapped again"


def convert_esp(path):
    try:
        with h5py.File(path, "r") as f:
            a = read_amps(f)                                         # (time, subcarriers)
        a = a[:, KEEP_TONES] @ W_ESP.T                               # (time, 52)
    except Exception:                                                # unreadable / other tone count
        return path, None, None, None

    series = np.full((N_BINS, N_TONES), np.nan, dtype=np.float32)
    valid = np.zeros(N_BINS, dtype=bool)
    if (a < 0).any():                                                # already normalised upstream
        X = (a - a.mean(0)) / (a.std(0) + 1e-6)
        X = X[::max(1, round(ASSUMED_RATE_HZ / FS))]
        ok = np.ones(len(X), dtype=bool)
    else:
        t = np.arange(len(a)) / ASSUMED_RATE_HZ
        X, ok = to_shared_series(t, a)
    n = min(len(X), N_BINS)
    series[:n], valid[:n] = X[:n], ok[:n]
    return path, series.astype(np.float16), valid, window_features(series)


N = len(esp)
SERIES = np.zeros((N, 1, N_BINS, N_TONES), dtype=np.float16)
VALID = np.zeros((N, N_BINS), dtype=bool)
FEATS = np.zeros((N, 1, len(FEATURE_NAMES)), dtype=np.float32)
OK = np.zeros(N, dtype=bool)
pos = {p: i for i, p in enumerate(esp["abs_path"])}

t0 = time.time()
with get_context("fork").Pool(N_WORKERS) as pool:
    for k, (p, s, v, f) in enumerate(pool.imap_unordered(convert_esp, esp["abs_path"].tolist(), chunksize=32)):
        if s is not None:
            i = pos[p]
            SERIES[i, 0], VALID[i], FEATS[i, 0], OK[i] = s, v, f, True
        if k % 5000 == 0:
            print(f"{k:7d} / {N}   [{time.time() - t0:.0f} s]", flush=True)

esp["converted"] = OK
esp["valid_frac"] = VALID.mean(1)
np.save(os.path.join(SHARED, "series.npy"), SERIES)
np.save(os.path.join(SHARED, "feats.npy"), FEATS)
out_meta = esp.rename(columns={"abs_path": "source_path"})
obj_cols = out_meta.select_dtypes("object").columns
out_meta[obj_cols] = out_meta[obj_cols].astype(str)                  # mixed types break parquet
out_meta.to_parquet(os.path.join(SHARED, "meta.parquet"), index=False)
with open(os.path.join(SHARED, "format.json"), "w") as f:
    json.dump({"fs_hz": FS, "win_s": WIN_S, "n_tones": N_TONES, "assumed_rate_hz": ASSUMED_RATE_HZ,
               "esp32_tones_kept": int(KEEP_TONES.sum()), "feature_names": FEATURE_NAMES}, f, indent=1)
print(f"converted {OK.sum()} / {N} in {time.time() - t0:.0f} s;",
      "with >= 80 % valid bins:", int((esp["valid_frac"] >= 1 - MAX_MISSING).sum()))'''


C_CHECK = r'''# 5. Quick look: count-cue distributions of the ESP32 data per task / activity
esp["cv_median"] = FEATS[:, 0, 0]
esp["diff_energy"] = FEATS[:, 0, 3]
display(esp[esp["converted"]].groupby(["task", "act"])[["cv_median", "diff_energy"]].median().round(4))

fig, ax = plt.subplots(figsize=(10, 4))
for task, g in esp[esp["converted"]].groupby("task"):
    ax.hist(np.log10(g["cv_median"] + 1e-4), bins=60, histtype="step", label=task)
ax.set_xlabel("log10 cv_median")
ax.legend(fontsize=7)
fig.savefig(os.path.join(AUDIT, "esp32_cv_median_hist.png"), dpi=120)
plt.show()'''


C_LOC_LAYOUT = r'''# 6. Localization - layout check: are the 168 subcarriers really 3 devices x 56?
ESP_FEATS, ESP_META = FEATS.copy(), esp.copy()        # keep the ESP32 cues for the reality check (cell 13)
LOC_DEVICES = 3
LOC_SUB = 168

loc = meta[meta["task"] == "Localization"].drop_duplicates("abs_path").reset_index(drop=True)
loc["n_sub"] = loc["num_sub"].astype(int)
loc["count"] = loc["count"].astype(int)
print("Localization samples per room / subcarrier count / people:")
display(pd.crosstab([loc["environment"], loc["n_sub"]], loc["count"], margins=True))
display(pd.crosstab(loc["device"], loc["n_sub"]))


def mean_profile(paths):
    """Mean |amplitude| per subcarrier over several files."""
    acc = None
    for p in paths:
        with h5py.File(p, "r") as f:
            m = np.abs(read_amps(f)).mean(axis=0)
        acc = m if acc is None else acc + m
    return acc / len(paths)


profiles = {n: mean_profile(g["abs_path"].sample(min(60, len(g)), random_state=SEED).tolist())
            for n, g in loc.groupby("n_sub")}

fig, axes = plt.subplots(len(profiles), 1, figsize=(14, 3 * len(profiles)))
for ax, (n_sub, prof) in zip(np.atleast_1d(axes), profiles.items()):
    ax.plot(prof)
    for k in range(1, LOC_DEVICES):
        ax.axvline(k * n_sub / LOC_DEVICES, color="r", ls="--")
    ax.set_title(f"mean amplitude per subcarrier, {n_sub} subcarriers (red = assumed device borders)")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "loc_subcarrier_profile.png"), dpi=110)
plt.show()

# if the 3 x 56 layout is right, each device block has its own null tones at the same places
blocks = profiles[LOC_SUB].reshape(LOC_DEVICES, -1)
print("correlation between the 3 device blocks' profiles:")
print(np.round(np.corrcoef(blocks), 3))'''


C_LOC_CONVERT = r'''# 7. Localization - 168-subcarrier samples -> 3 links x 250 bins x 52 tones + count cues
SUB_PER_DEV = LOC_SUB // LOC_DEVICES
loc_use = loc[loc["n_sub"] == LOC_SUB].reset_index(drop=True)
prof = blocks.mean(axis=0)
KEEP_LOC = prof > 0.02 * np.median(prof)              # drop null tones inside a device block
W_LOC = tone_interp_matrix(int(KEEP_LOC.sum()))
print(len(loc_use), "samples;", int(KEEP_LOC.sum()), "of", SUB_PER_DEV, "tones kept per device")

''' + EXTRA_CUES + r'''


def convert_loc(path):
    try:
        with h5py.File(path, "r") as f:
            a = read_amps(f)                                     # (time, 168)
    except Exception:                                            # unreadable file
        return path, None, None, None, None, None

    t = np.arange(len(a)) / ASSUMED_RATE_HZ
    series = np.full((LOC_DEVICES, N_BINS, N_TONES), np.nan, dtype=np.float32)
    valid = np.zeros(N_BINS, dtype=bool)
    feats = np.zeros((LOC_DEVICES, len(FEATURE_NAMES)), dtype=np.float32)
    extra = np.zeros((LOC_DEVICES, len(EXTRA_NAMES)), dtype=np.float32)
    envs = []
    for v in range(LOC_DEVICES):
        block = a[:, v * SUB_PER_DEV:(v + 1) * SUB_PER_DEV][:, KEEP_LOC] @ W_LOC.T
        X, ok = to_shared_series(t, block)
        n = min(len(X), N_BINS)
        series[v, :n] = X[:n]
        valid[:n] = ok[:n]
        feats[v] = window_features(series[v])
        extra[v], e = extra_features(series[v])
        envs.append(e)
    return path, series.astype(np.float16), valid, feats, extra, link_cues(envs)


NL = len(loc_use)
LOC_SERIES = np.zeros((NL, LOC_DEVICES, N_BINS, N_TONES), dtype=np.float16)
LOC_VALID = np.zeros((NL, N_BINS), dtype=bool)
LOC_FEATS = np.zeros((NL, LOC_DEVICES, len(FEATURE_NAMES)), dtype=np.float32)
LOC_EXTRA = np.zeros((NL, LOC_DEVICES, len(EXTRA_NAMES)), dtype=np.float32)
LOC_LINK = np.zeros((NL, len(LINK_NAMES)), dtype=np.float32)
LOC_OK = np.zeros(NL, dtype=bool)
pos = {p: i for i, p in enumerate(loc_use["abs_path"])}

t0 = time.time()
with get_context("fork").Pool(N_WORKERS) as pool:
    results_iter = pool.imap_unordered(convert_loc, loc_use["abs_path"].tolist(), chunksize=32)
    for k, (p, s, v, f, e, l) in enumerate(results_iter):
        if s is not None:
            i = pos[p]
            LOC_SERIES[i], LOC_VALID[i], LOC_FEATS[i] = s, v, f
            LOC_EXTRA[i], LOC_LINK[i], LOC_OK[i] = e, l, True
        if k % 2000 == 0:
            print(f"{k:6d} / {NL}   [{time.time() - t0:.0f} s]", flush=True)

loc_use["converted"] = LOC_OK
loc_use["valid_frac"] = LOC_VALID.mean(1)
np.save(os.path.join(SHARED_LOC, "series.npy"), LOC_SERIES)
np.save(os.path.join(SHARED_LOC, "feats.npy"), LOC_FEATS)
np.save(os.path.join(SHARED_LOC, "extra_feats.npy"), LOC_EXTRA)
np.save(os.path.join(SHARED_LOC, "link_feats.npy"), LOC_LINK)
out_meta = loc_use.rename(columns={"abs_path": "source_path"})
obj_cols = out_meta.select_dtypes("object").columns
out_meta[obj_cols] = out_meta[obj_cols].astype(str)                  # mixed types break parquet
out_meta.to_parquet(os.path.join(SHARED_LOC, "meta.parquet"), index=False)
with open(os.path.join(SHARED_LOC, "format.json"), "w") as f:
    json.dump({"fs_hz": FS, "win_s": WIN_S, "n_tones": N_TONES, "assumed_rate_hz": ASSUMED_RATE_HZ,
               "views": "3 device links (168 = 3 x 56 subcarriers)", "tones_kept_per_device": int(KEEP_LOC.sum()),
               "feature_names": FEATURE_NAMES, "extra_names": EXTRA_NAMES, "link_names": LINK_NAMES}, f, indent=1)
print(f"converted {LOC_OK.sum()} / {NL} in {time.time() - t0:.0f} s;",
      "with >= 80 % valid bins:", int((loc_use["valid_frac"] >= 1 - MAX_MISSING).sum()))'''


C_LOC_CHECK = r'''# 8. Localization - do the cues rise from 0 to 1 to 2 people in every room?
CUE_NAMES = FEATURE_NAMES + EXTRA_NAMES + LINK_NAMES
cue = pd.DataFrame(np.concatenate([LOC_FEATS.mean(1), LOC_EXTRA.mean(1), LOC_LINK], axis=1), columns=CUE_NAMES)
cue["count"], cue["env"] = loc_use["count"].values, loc_use["environment"].values
cue = cue[LOC_OK]

display(cue.groupby(["env", "count"])[["cv_median", "diff_energy", "eig_entropy", "doppler_spread",
                                        "activity_cv", "link_env_corr"]].median().round(4))

spearman = pd.DataFrame({env: [g[c].corr(g["count"], method="spearman") for c in CUE_NAMES]
                         for env, g in cue.groupby("env")}, index=CUE_NAMES).round(3)
spearman.to_csv(os.path.join(RESULTS, "loc_cue_spearman.csv"))
display(spearman)'''


C_LOC_TABLE = r'''# 9. Localization - feature table (one row per sample x device link), leave one room out
keep = LOC_OK & (loc_use["valid_frac"].values >= 1 - MAX_MISSING)
idx = np.flatnonzero(keep)
MAX_COUNT = 2
N_LINKS = LOC_DEVICES

rows = pd.DataFrame({"sample": np.repeat(idx, N_LINKS), "view": np.tile(np.arange(N_LINKS), len(idx))})
for c in ["count", "environment"]:
    rows[c] = loc_use[c].values[rows["sample"]]
rows["count4"] = np.minimum(rows["count"], 3)
rows[FEATURE_NAMES] = np.nan_to_num(LOC_FEATS[idx].reshape(-1, len(FEATURE_NAMES)))
rows[EXTRA_NAMES] = np.nan_to_num(LOC_EXTRA[idx].reshape(-1, len(EXTRA_NAMES)))
rows[LINK_NAMES] = np.nan_to_num(np.repeat(LOC_LINK[idx], N_LINKS, axis=0))
rows["calib_group"] = rows["environment"]
ENVS = sorted(rows["environment"].unique())
VIEW0 = (rows["view"] == 0).values

FEATURE_SETS = {
    "base10": FEATURE_NAMES,
    "single_link": FEATURE_NAMES + EXTRA_NAMES,
    "multi_link": FEATURE_NAMES + EXTRA_NAMES + LINK_NAMES,
}


def folds():
    """Leave one room out."""
    for env in ENVS:
        test = (rows["environment"] == env).values
        yield env, ~test, test


print(len(idx), "samples,", len(rows), "rows, rooms:", ENVS)
display(pd.crosstab(rows.loc[VIEW0, "environment"], rows.loc[VIEW0, "count"], margins=True))'''


C_LOC_GBM = r'''# 10. Localization - LightGBM on 3 feature sets, with / without per-room z-score
''' + GBM_DEFS + r'''


IMPORTANCE = []
t0 = time.time()
for calib in ["none", "group_z"]:
    for fs_name, feats in FEATURE_SETS.items():
        d = calibrate(rows, feats) if calib == "group_z" else rows
        preds = {k: np.full(len(d), np.nan) for k in ["clf", "reg", "mean"]}
        for env, train, test in folds():
            X_tr, X_te = d.loc[train, feats], d.loc[test, feats]
            y_tr = d.loc[train, "count"]
            preds["clf"][test] = lgb.LGBMClassifier(**GBM_PARAMS).fit(X_tr, y_tr).predict(X_te)
            reg = lgb.LGBMRegressor(**GBM_PARAMS).fit(X_tr, y_tr)
            preds["reg"][test] = reg.predict(X_te)
            preds["mean"][test] = y_tr.mean()
            IMPORTANCE.append(pd.Series(reg.booster_.feature_importance("gain"), index=feats,
                                        name=f"{fs_name}|{calib}|{env}"))

        for task in ["clf", "reg"]:
            evaluate("zero_shot", f"gbm_{fs_name}", task, calib, d, preds[task])
        if calib == "none" and fs_name == "base10":
            evaluate("zero_shot", "mean_baseline", "reg", "none", d, preds["mean"])
        print(f"calib={calib:8s} features={fs_name:12s} [{time.time() - t0:.0f} s]", flush=True)

res = pd.DataFrame(RESULT_ROWS)
display(res[res["level"] == "view_ensemble"].sort_values("mae").round(3))'''


C_LOC_CNN = r'''# 11. Localization - tiny CNNs: one device link, and the 3 links pooled
N_VIEWS = LOC_DEVICES          # links per sample
SERIES = LOC_SERIES            # the CNN helpers read windows from SERIES
''' + CNN_DEFS + r'''


t0 = time.time()
for kind in ["single", "multi"]:
    out = {(v, t): np.full(len(rows), np.nan) for v in ["fp32", "int8"] for t in ["clf", "reg"]}
    for env, train, test in folds():
        if kind == "multi":                                    # one row per sample
            train, test = train & VIEW0, test & VIEW0
        tr = rows[train]
        model = train_model(kind, tr, np.zeros(len(tr), dtype=bool))
        for v, m in [("fp32", model), ("int8", int8_copy(model))]:
            out[(v, "clf")][test], out[(v, "reg")][test] = predict(m, kind, rows[test])
        print(f"cnn_{kind:6s} left out {env:6s} [{time.time() - t0:.0f} s]", flush=True)

    tested = ~np.isnan(out[("fp32", "reg")])
    for (v, task), pred in out.items():
        name = f"cnn_{kind}" + ("_int8" if v == "int8" else "")
        evaluate("zero_shot", name, task, "none", rows[tested], pred[tested])

res = pd.DataFrame(RESULT_ROWS)
display(res[res["model"].str.startswith("cnn") & (res["level"] == "view_ensemble")].sort_values("mae").round(3))'''


C_CROSS = r'''# 12. Cross-dataset: WiMANS (5 GHz) <-> CSI-Bench Localization, 0-2 people, per-room z-scored cues
def find_wimans():
    """Folder of the attached wifi-wimans-counting output (shared_format/wimans), or None."""
    skip = {"csi-bench", "RawContinuousRecording", "amp", "wifi_csi"}
    for folder, subdirs, files in os.walk("/kaggle/input"):
        subdirs[:] = [s for s in subdirs if s not in skip and not s.startswith("sub_")]
        if os.path.basename(folder) == "wimans" and "feats.npy" in files and "meta.parquet" in files:
            return folder
    return None


def zscore_room(df, feats):
    """z-score every cue within each room, using that room's own unlabelled samples."""
    out = df.copy()
    g = out.groupby("calib_group")[feats]
    out[feats] = (out[feats] - g.transform("mean")) / (g.transform("std") + 1e-6)
    return out


WIM = find_wimans()
print("WiMANS shared format:", WIM)
wfeats = None
if WIM is None:
    print("wifi-wimans-counting output not attached - skipping the cross-dataset test")
else:
    wmeta = pd.read_parquet(os.path.join(WIM, "meta.parquet"))
    wfeats = np.load(os.path.join(WIM, "feats.npy"))                      # (samples, 9, 10)
    has_extra = os.path.exists(os.path.join(WIM, "extra_feats.npy"))
    cross_sets = {"base10": FEATURE_NAMES}
    if has_extra:
        wextra = np.load(os.path.join(WIM, "extra_feats.npy"))
        cross_sets["single_link"] = FEATURE_NAMES + EXTRA_NAMES

    wkeep = ((wmeta["wifi_band"].astype(str) == "5") & (wmeta["count"].astype(int) <= MAX_COUNT)
             & (wmeta["valid_frac"] >= 1 - MAX_MISSING))
    wi = np.flatnonzero(wkeep.values)
    wrows = pd.DataFrame({"sample": np.repeat(wi, 9), "view": np.tile(np.arange(9), len(wi))})
    wrows["count"] = wmeta["count"].astype(int).values[wrows["sample"]]
    wrows["count4"] = wrows["count"]
    wrows["calib_group"] = "wimans|" + wmeta["environment"].astype(str).values[wrows["sample"]]
    wrows[FEATURE_NAMES] = np.nan_to_num(wfeats[wi].reshape(-1, len(FEATURE_NAMES)))
    if has_extra:
        wrows[EXTRA_NAMES] = np.nan_to_num(wextra[wi].reshape(-1, len(EXTRA_NAMES)))
    print("WiMANS 5 GHz samples with 0-2 people:", len(wi), "| extra cues:", has_extra)

    crows = rows.copy()
    crows["calib_group"] = "csibench|" + crows["environment"]

    for fs_name, feats in cross_sets.items():
        for calib in ["none", "room_z"]:
            W = zscore_room(wrows, feats) if calib == "room_z" else wrows
            C = zscore_room(crows, feats) if calib == "room_z" else crows
            for src, dst, setting in [(W, C, "wimans->csibench"), (C, W, "csibench->wimans")]:
                clf = lgb.LGBMClassifier(**GBM_PARAMS).fit(src[feats], src["count"])
                reg = lgb.LGBMRegressor(**GBM_PARAMS).fit(src[feats], src["count"])
                evaluate(setting, f"gbm_{fs_name}", "clf", calib, dst, clf.predict(dst[feats]))
                evaluate(setting, f"gbm_{fs_name}", "reg", calib, dst, reg.predict(dst[feats]))
                if fs_name == "base10" and calib == "none":
                    evaluate(setting, "mean_baseline", "reg", "none", dst,
                             np.full(len(dst), src["count"].mean()))

    res = pd.DataFrame(RESULT_ROWS)
    display(res[res["setting"].str.contains("->") & (res["level"] == "view_ensemble")]
            .sort_values(["setting", "mae"]).round(3))'''


C_ESP_REALITY = r'''# 13. ESP32 reality check: base cues of the real ESP32 clips next to the other devices
parts = []
e = pd.DataFrame(ESP_FEATS[:, 0, :], columns=FEATURE_NAMES)
e["source"] = "ESP32 fall clips: " + ESP_META["label"].astype(str).values
parts.append(e[ESP_META["converted"].values])

lf = LOC_FEATS[LOC_OK]
l = pd.DataFrame(lf.reshape(-1, len(FEATURE_NAMES)), columns=FEATURE_NAMES)
l["source"] = "CSI-Bench loc: " + np.repeat(loc_use.loc[LOC_OK, "count"].values, LOC_DEVICES).astype(str) + " people"
parts.append(l)

if wfeats is not None:
    w = pd.DataFrame(wfeats[wi].reshape(-1, len(FEATURE_NAMES)), columns=FEATURE_NAMES)
    w["source"] = "WiMANS 5 GHz: " + np.repeat(wmeta["count"].astype(int).values[wi], 9).astype(str) + " people"
    parts.append(w)

cmp_df = pd.concat(parts, ignore_index=True)
show = ["cv_median", "diff_energy", "tone_corr", "n_pc_90"]
summary = cmp_df.groupby("source")[show].median().round(4)
summary.to_csv(os.path.join(RESULTS, "esp32_vs_others.csv"))
display(summary)

order = sorted(cmp_df["source"].unique())
fig, axes = plt.subplots(1, len(show), figsize=(5 * len(show), 6))
for ax, col in zip(axes, show):
    ax.boxplot([cmp_df.loc[cmp_df["source"] == s, col].values for s in order], vert=False, showfliers=False)
    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels(order if col == show[0] else [], fontsize=7)
    ax.set_title(col)
    if col in ("cv_median", "diff_energy"):
        ax.set_xscale("log")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "esp32_vs_others.png"), dpi=110)
plt.show()'''


C_SAVE = r'''# 14. Save results and a short report
results = pd.DataFrame(RESULT_ROWS)
per_count = pd.DataFrame(PER_COUNT_ROWS)
results.to_csv(os.path.join(RESULTS, "results.csv"), index=False)
per_count.to_csv(os.path.join(RESULTS, "per_count_recall.csv"), index=False)
importance = pd.concat(IMPORTANCE, axis=1)
importance.to_csv(os.path.join(RESULTS, "gbm_feature_importance.csv"))

cols = ["setting", "model", "task", "calib", "mae", "exact", "within1", "f1_4", "n"]
ens = results[results["level"] == "view_ensemble"]
loc_best = (ens[ens["setting"] == "zero_shot"].sort_values("mae")
            .groupby("model").head(1).sort_values("mae"))
single_models = ["gbm_base10", "gbm_single_link", "cnn_single", "cnn_single_int8"]
one_link = results[(results["level"] == "single_view") & (results["setting"] == "zero_shot")
                   & results["model"].isin(single_models)]
one_link_best = one_link.sort_values("mae").groupby("model").head(1).sort_values("mae")
cross = ens[ens["setting"].str.contains("->")].sort_values(["setting", "mae"])

pc = per_count[per_count["setting"] == "zero_shot"]
fig, ax = plt.subplots(figsize=(7, 4))
for _, r in loc_best.head(5).iterrows():
    h = pc[(pc["model"] == r["model"]) & (pc["task"] == r["task"]) & (pc["calib"] == r["calib"])]
    ax.plot(h["count"], h["recall"], marker="o", label=f"{r['model']}/{r['task']}/{r['calib']}")
ax.set_xlabel("people")
ax.set_ylim(0, 1)
ax.set_title("CSI-Bench Localization, unseen room: per-count recall")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "loc_per_count_recall.png"), dpi=120)
plt.show()

lines = ["# CSI-Bench v3 - Localization people counting (0-2) + cross-dataset + ESP32 check", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')}; 5-s windows, {FS} Hz, 52 tones, 3 device links per sample.", "",
         "## Localization, unseen room (best row per model, all 3 links averaged)", "",
         loc_best[cols].round(3).to_markdown(index=False), "",
         "## One device link (single-link models, scored per link)", "",
         one_link_best[cols].round(3).to_markdown(index=False), "",
         "## Cross-dataset (train on one dataset, test on the other; 0-2 people)", "",
         (cross[cols].round(3).to_markdown(index=False) if len(cross) else "skipped: WiMANS output not attached"), "",
         "## ESP32 reality check (median base cues per source)", "", summary.to_markdown(), "",
         "## Caveats", "",
         "- Only the 168-subcarrier Localization recordings (3 devices x 56) are used; the 2496-subcarrier ones are left out.",
         "- Packet times are assumed uniform at 100 Hz (the h5 files have no timestamps).",
         "- f1_4 here covers only the classes present (0-2).",
         "- Cross-dataset: WiMANS windows are 3 s, CSI-Bench 5 s; room_z z-scores each room with its own unlabelled data.",
         "- int8 rows quantise weights only; activations stay float."]
with open(os.path.join(RESULTS, "CSIBENCH_RESULTS.md"), "w") as f:
    f.write(chr(10).join(lines))
display(loc_best[cols].round(3))
display(cross[cols].round(3))'''


# ===============================================================
# EHUNAM discovery
# ===============================================================
E_INTRO = r'''# EHUNAM on Kaggle - Phase 0 discovery

EHUNAM (figshare 28541225, Scientific Data 2025): 23 days, 8 environments, up to 21 people and 9 machines;
tasks People Counting (PC), Human Activity Recognition (HAR), Machine Recognition (MR), Machine Activity
Recognition (MAR). 2,401 `.mat` files + `Summary.xlsx` / `Summary.mat` describing each file.

The paper with the exact file layout is not freely readable, so this notebook **discovers** it:
1. reads `Summary.xlsx` (columns, counts, people-count column);
2. splits the file names into their codes and matches them with the summary;
3. opens one file of each code and prints every variable, shape and dtype;
4. if Kaggle kept the upload as `EHUNAM.zip`, lists it and extracts a few members (Deflate64).

**Save the notebook after running** and share the output, so the Phase 1 converter can be written for EHUNAM.'''


E_SETUP = r'''# 1. Setup and discovery of the uploaded files
import glob
import os
import re
import zipfile
from collections import Counter

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio

''' + FINDER + r'''

summary_dirs = find_dirs(lambda folder, files: "Summary.xlsx" in files or "Summary.mat" in files)
zip_dirs = find_dirs(lambda folder, files: "EHUNAM.zip" in files)
mat_dirs = find_dirs(lambda folder, files: any(f.endswith(".mat") and not f.startswith("Summary") for f in files))
print("Summary in :", summary_dirs)
print("zip in     :", zip_dirs)
print(".mat in    :", mat_dirs[:10], "..." if len(mat_dirs) > 10 else "")

WORK = "/kaggle/working/ehunam_explore"
os.makedirs(WORK, exist_ok=True)'''


E_SUMMARY = r'''# 2. Summary.xlsx: columns, value counts, people-count column
assert summary_dirs, "Summary.xlsx not found - is the EHUNAM upload finished and attached?"
xlsx = os.path.join(summary_dirs[0], "Summary.xlsx")
sheets = pd.read_excel(xlsx, sheet_name=None)
for name, df in sheets.items():
    print("=" * 100)
    print("sheet", name, df.shape)
    print(list(df.columns))
    display(df.head(10))
    for c in df.columns:
        n = df[c].nunique()
        if 1 < n <= 40:
            print(f"  {c!r}: {df[c].value_counts().sort_index().to_dict()}")
        else:
            print(f"  {c!r}: {n} unique, e.g. {df[c].dropna().astype(str).head(3).tolist()}")

summary = next(iter(sheets.values()))
summary.to_csv(os.path.join(WORK, "summary_sheet1.csv"), index=False)

smat = os.path.join(summary_dirs[0], "Summary.mat")
if os.path.exists(smat):
    s = sio.loadmat(smat, squeeze_me=True, struct_as_record=False)
    print("Summary.mat variables:", {k: type(v).__name__ for k, v in s.items() if not k.startswith("__")})'''


E_NAMES = r'''# 3. File names -> codes (MCx_<id>_<n>_<task code>_...), matched against the summary
if mat_dirs:
    names = sorted(os.path.basename(p) for d in mat_dirs for p in glob.glob(os.path.join(d, "*.mat"))
                   if not os.path.basename(p).startswith("Summary"))
    paths = {os.path.basename(p): p for d in mat_dirs for p in glob.glob(os.path.join(d, "*.mat"))}
    SOURCE = "files"
elif zip_dirs:
    zf = zipfile.ZipFile(os.path.join(zip_dirs[0], "EHUNAM.zip"))
    infos = [i for i in zf.infolist() if i.filename.endswith(".mat")]
    names = sorted(os.path.basename(i.filename) for i in infos)
    print("zip entries:", len(infos), "| compression methods:", Counter(i.compress_type for i in infos))
    SOURCE = "zip"
else:
    raise SystemExit("no EHUNAM .mat files or EHUNAM.zip attached yet")

print(len(names), "mat files; first:", names[:5])
tokens = pd.DataFrame([n[:-4].split("_") for n in names])
tokens.insert(0, "name", names)
for c in tokens.columns[1:8]:
    vc = tokens[c].value_counts()
    print(f"token {c}: {len(vc)} unique  {vc.head(15).to_dict()}")

TASK_CODES = ["PCMAR", "PC", "HAR", "MAR", "MR", "E"]
tokens["code"] = [next((t for t in TASK_CODES if f"_{t}_" in f"_{n[:-4]}_"), "?") for n in names]
print("task codes:", tokens["code"].value_counts().to_dict())

# which summary column holds the file names?
name_stems = set(n[:-4] for n in names)
for c in summary.columns:
    hit = summary[c].astype(str).str.replace(".mat", "", regex=False).isin(name_stems).mean()
    if hit > 0.1:
        print(f"summary column {c!r} matches {hit:.0%} of rows to file names")
tokens.to_csv(os.path.join(WORK, "file_tokens.csv"), index=False)'''


E_OPEN = r'''# 4. Open one file per task code: every variable, shape, dtype, small value sample
if SOURCE == "zip":
    !pip -q install zipfile-deflate64
    import zipfile_deflate64 as zipfile64                   # handles Deflate64 (method 9)
    zf64 = zipfile64.ZipFile(os.path.join(zip_dirs[0], "EHUNAM.zip"))
    by_name = {os.path.basename(i.filename): i.filename for i in zf64.infolist()}


def local_path(name):
    if SOURCE == "files":
        return paths[name]
    out = os.path.join(WORK, name)
    if not os.path.exists(out):
        with zf64.open(by_name[name]) as src, open(out, "wb") as dst:
            dst.write(src.read())
    return out


def describe(name, value, depth=0):
    pad = "    " * depth
    if isinstance(value, np.ndarray) and value.dtype.names:
        print(f"{pad}{name}: struct array {value.shape} fields={value.dtype.names}")
        for field in value.dtype.names[:20]:
            describe(field, value[field].flat[0] if value.size else None, depth + 1)
    elif isinstance(value, np.ndarray):
        sample = value.ravel()[:6]
        print(f"{pad}{name}: {value.dtype} {value.shape}  e.g. {sample}")
    elif isinstance(value, h5py.Dataset):
        print(f"{pad}{name}: h5 {value.dtype} {value.shape}")
    elif isinstance(value, h5py.Group):
        print(f"{pad}{name}: h5 group {list(value.keys())[:20]}")
        for k in list(value.keys())[:20]:
            describe(k, value[k], depth + 1)
    else:
        print(f"{pad}{name}: {type(value).__name__} {str(value)[:80]}")


for code, g in tokens.groupby("code"):
    name = g["name"].iloc[0]
    path = local_path(name)
    print("=" * 100)
    print(code, name, f"{os.path.getsize(path) / 1e6:.1f} MB")
    try:
        m = sio.loadmat(path)
        for k, v in m.items():
            if not k.startswith("__"):
                describe(k, v, 1)
    except NotImplementedError:                              # MATLAB v7.3 = HDF5
        with h5py.File(path, "r") as f:
            for k in f.keys():
                describe(k, f[k], 1)'''


# ===============================================================
# Notebook writer
# ===============================================================
NOTEBOOKS = {
    "kaggle_wimans.ipynb": [
        ("md", W_INTRO), ("code", W_SETUP), ("code", W_AUDIT_LABELS), ("code", W_AUDIT_SIGNAL),
        ("code", W_HELPERS), ("code", W_CONVERT), ("code", W_CHECK), ("code", W_EXTRA), ("code", W_TABLE),
        ("code", METRICS), ("code", W_GBM), ("code", TINY_CNN), ("code", W_CNN), ("code", W_DEPLOY),
        ("code", W_SAVE),
    ],
    "kaggle_csibench.ipynb": [
        ("md", C_INTRO), ("code", C_SETUP), ("code", C_AUDIT_META), ("code", C_AUDIT_H5),
        ("code", C_AUDIT_RAW), ("code", C_HELPERS), ("code", C_CONVERT), ("code", C_CHECK),
        ("code", C_LOC_LAYOUT), ("code", C_LOC_CONVERT), ("code", C_LOC_CHECK), ("code", C_LOC_TABLE),
        ("code", METRICS), ("code", C_LOC_GBM), ("code", TINY_CNN), ("code", C_LOC_CNN),
        ("code", C_CROSS), ("code", C_ESP_REALITY), ("code", C_SAVE),
    ],
    "kaggle_ehunam_explore.ipynb": [
        ("md", E_INTRO), ("code", E_SETUP), ("code", E_SUMMARY), ("code", E_NAMES), ("code", E_OPEN),
    ],
}


def syntax_check(text):
    """ast-check a cell, ignoring IPython '!' shell lines."""
    lines = [line[:len(line) - len(line.lstrip())] + "pass" if line.strip().startswith("!") else line
             for line in text.split("\n")]
    ast.parse("\n".join(lines))


def main():
    for filename, cells in NOTEBOOKS.items():
        for kind, text in cells:
            if kind == "code":
                syntax_check(text)

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

"""V2 evaluation on REAL ESP32 recordings only (no synthetic noise in any score).

    R1  WiCount same session      first 40 s -> last 16 s of every recording (4 s guard)
    R2  SRIBD -> OTH room         train WiCount -> test OTH thesis room (other room, firmware, band)
    R3  OTH room -> SRIBD         train OTH thesis room -> test WiCount
    R4  both rooms, same session  train first 40 s of every recording in BOTH rooms -> last 16 s

Variants (each averaged over 3 seeds)
    lgbm / lgbm_cal         LightGBM on features, raw / empty-room calibrated
    cnn / cnn_cal           1D-CNN on window sequences
    tf / tf_cal             CSI-BERT2-style transformer, self-supervised pre-trained on
                            WiGesture + WiFall (never used as test data), then fine-tuned
    ensemble_cal            mean probability of lgbm_cal, cnn_cal, tf_cal

The 20 s empty-room reference of each site is never part of a test set.
Scores: window level and after 5-window smoothing within a recording.

Usage:  python experiments.py R1 R2 R3 R4        (default: all)
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

import deep
from trainlog import log
from build import WIN_S, build, build_seq, feature_cols
from calibrate import REF_SECONDS, calibrate, reference_mask

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
MODELS = ROOT / "models"
CLASSES = deep.CLASSES
SMOOTH = 5
SEEDS = (0, 1, 2)

LGB_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  class_weight="balanced", verbose=-1, n_jobs=8)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load(names):
    """Concatenated feature table + aligned sequences, plus per-row empty-room levels."""
    dfs, seqs, gaps = [], [], []
    for n in names:
        dfs.append(build(n))
        s, g = build_seq(n)
        seqs.append(s)
        gaps.append(g)
    df = pd.concat(dfs, ignore_index=True)
    seq, gap = np.concatenate(seqs), np.concatenate(gaps)

    df["site"] = df["room"] + "|" + df["day"].astype(str)

    # empty-room fluctuation level of every site (for the deep models' calibration)
    level = np.median(np.abs(seq.astype(np.float32)), axis=(1, 2))
    ref = reference_mask(df)
    site_ref = pd.Series(level[ref.to_numpy()], index=df.index[ref]).groupby(df.loc[ref, "site"]).median()
    df["ref_level"] = df["site"].map(site_ref)
    return df, seq, gap


def not_reference(df):
    """Test windows must not overlap the calibration reference (+ a 2 s guard)."""
    return ~((df["count"] == 0) & (df["t0"] < REF_SECONDS + 2.0))


# ---------------------------------------------------------------------------
# Pre-trained transformer (shared by all protocols)
# ---------------------------------------------------------------------------

def pretrained_state():
    path = MODELS / "transformer_pretrained.pt"
    if path.exists():
        return torch.load(path, map_location="cpu")

    log("pre-training transformer on WiGesture + WiFall (unlabelled) ...")
    s1, g1 = build_seq("wigesture")
    s2, g2 = build_seq("wifall")
    model = deep.pretrain(np.concatenate([s1, s2]), np.concatenate([g1, g2]), epochs=40)
    MODELS.mkdir(exist_ok=True)
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(state, path)
    return state


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def smooth(df, pred):
    s = pd.Series(pred, index=df.index, dtype=float)
    out = s.groupby(df["recording"]).transform(lambda x: x.rolling(SMOOTH, min_periods=1).median())
    return np.round(out.to_numpy()).astype(int)


def scores(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    pres_y, pres_p = (y > 0).astype(int), (p > 0).astype(int)
    return dict(
        n=int(len(y)),
        acc=float((y == p).mean()),
        bal_acc=float(balanced_accuracy_score(y, p)),
        macro_f1=float(f1_score(y, p, labels=np.unique(y), average="macro", zero_division=0)),
        mae=float(np.abs(y - p).mean()),
        within1=float((np.abs(y - p) <= 1).mean()),
        presence_bal_acc=float(balanced_accuracy_score(pres_y, pres_p)) if len(np.unique(pres_y)) > 1 else None,
        confusion=confusion_matrix(y, p, labels=CLASSES).tolist(),
    )


def evaluate(test, proba) -> dict:
    pred = np.asarray(CLASSES)[proba.argmax(axis=1)]
    return dict(window=scores(test["count"], pred), smoothed=scores(test["count"], smooth(test, pred)))


# ---------------------------------------------------------------------------
# Models on one split
# ---------------------------------------------------------------------------

def lgbm_proba(train, test, cols):
    proba = np.zeros((len(test), len(CLASSES)))
    for sd in SEEDS:
        m = lgb.LGBMClassifier(random_state=sd, **LGB_PARAMS).fit(train[cols], train["count"])
        p = m.predict_proba(test[cols])
        for j, c in enumerate(m.classes_):
            proba[:, CLASSES.index(c)] += p[:, j] / len(SEEDS)
    return proba


def deep_proba(kind, tr_idx, te_idx, df, seq, gap, cal, init):
    ref_tr = df.loc[tr_idx, "ref_level"].to_numpy() / REF_NORM if cal else None
    ref_te = df.loc[te_idx, "ref_level"].to_numpy() / REF_NORM if cal else None
    proba = np.zeros((len(te_idx), len(CLASSES)))
    for sd in SEEDS:
        model = deep.fit(kind, seq[tr_idx], gap[tr_idx], df.loc[tr_idx, "count"].to_numpy(),
                         ref=ref_tr, seed=sd, init_state=init,
                         tag=f"{CURRENT} {kind}{'_cal' if cal else ''} s{sd}")
        proba += deep.predict_proba(model, seq[te_idx], gap[te_idx], ref=ref_te) / len(SEEDS)
    return proba


CURRENT = ""             # protocol name shown in the live log
REF_NORM = 0.03          # typical empty-room level; keeps calibrated inputs at the pre-training scale


def run_split(df, seq, gap, train_mask, test_mask) -> dict:
    cols = feature_cols(df.drop(columns=["ref_level"]))
    train, test = df[train_mask], df[test_mask]
    tr_idx, te_idx = train.index.to_numpy(), test.index.to_numpy()
    init = pretrained_state()

    # calibrated tables (sites without an empty reference drop out of '_cal' training)
    cal_df = calibrate(df, cols)
    has_ref = df["ref_level"].notna()
    tr_cal_idx = train.index[has_ref.loc[train.index]].to_numpy()

    proba, res = {}, {}
    t = time.time()
    log(f"{CURRENT}: train {len(train)} windows, test {len(test)} windows")

    steps = [
        ("lgbm", lambda: lgbm_proba(train, test, cols)),
        ("lgbm_cal", lambda: lgbm_proba(cal_df.loc[tr_cal_idx], cal_df.loc[te_idx], cols)),
        ("cnn", lambda: deep_proba("cnn", tr_idx, te_idx, df, seq, gap, False, None)),
        ("cnn_cal", lambda: deep_proba("cnn", tr_cal_idx, te_idx, df, seq, gap, True, None)),
        ("tf", lambda: deep_proba("tf", tr_idx, te_idx, df, seq, gap, False, init)),
        ("tf_cal", lambda: deep_proba("tf", tr_cal_idx, te_idx, df, seq, gap, True, init)),
    ]
    for name, fn in steps:
        log(f"{CURRENT} -> {name}")
        proba[name] = fn()
        res[name] = evaluate(test, proba[name])
        m = res[name]["smoothed"]
        log(f"{CURRENT} <- {name:12s} test acc {m['acc']:.3f}  bal {m['bal_acc']:.3f}  "
            f"MAE {m['mae']:.2f}  within1 {m['within1']:.3f}  (smoothed)")

    proba["ensemble_cal"] = (proba["lgbm_cal"] + proba["cnn_cal"] + proba["tf_cal"]) / 3
    res["ensemble_cal"] = evaluate(test, proba["ensemble_cal"])
    maj = train["count"].mode()[0]
    res["majority"] = dict(window=scores(test["count"], np.full(len(test), maj)))
    log(f"{CURRENT} done in {time.time() - t:.0f} s")
    return res


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def show(title, res):
    print(f"\n== {title}")
    print(f"  {'variant':13s} {'acc':>6s} {'bal':>6s} {'MAE':>5s} {'±1':>6s} {'pres':>6s}   (smoothed)")
    for v, r in res.items():
        m = r.get("smoothed", r["window"])
        pb = m["presence_bal_acc"]
        print(f"  {v:13s} {m['acc']:6.3f} {m['bal_acc']:6.3f} {m['mae']:5.2f} {m['within1']:6.3f} "
              f"{'  -  ' if pb is None else f'{pb:6.3f}'}")


def save(name, obj):
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1))


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

def r1():
    global CURRENT
    CURRENT = "R1"
    df, seq, gap = load(["wicount"])
    res = run_split(df, seq, gap, df["t0"] + WIN_S <= 40, (df["t0"] >= 44) & not_reference(df))
    show("R1 WiCount same session (first 40 s -> last 16 s)", res)
    save("R1_wicount_session", res)


def r2():
    global CURRENT
    CURRENT = "R2"
    df, seq, gap = load(["wicount", "thesis"])
    res = run_split(df, seq, gap, df["room"] == "sribd_meeting",
                    (df["room"] == "oth_room2") & not_reference(df))
    show("R2 SRIBD meeting room -> OTH room", res)
    save("R2_sribd_to_oth", res)


def r3():
    global CURRENT
    CURRENT = "R3"
    df, seq, gap = load(["wicount", "thesis"])
    res = run_split(df, seq, gap, df["room"] == "oth_room2",
                    (df["room"] == "sribd_meeting") & not_reference(df))
    show("R3 OTH room -> SRIBD meeting room", res)
    save("R3_oth_to_sribd", res)


def r4():
    global CURRENT
    CURRENT = "R4"
    df, seq, gap = load(["wicount", "thesis"])
    res = run_split(df, seq, gap, df["t0"] + WIN_S <= 40, (df["t0"] >= 44) & not_reference(df))
    show("R4 both rooms, same session (first 40 s -> last 16 s)", res)
    save("R4_both_rooms_session", res)


PROTOCOLS = {"R1": r1, "R2": r2, "R3": r3, "R4": r4}

if __name__ == "__main__":
    for name in (sys.argv[1:] or list(PROTOCOLS)):
        PROTOCOLS[name]()

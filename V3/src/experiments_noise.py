"""V3 real-noise evaluation: presence and motion state on REAL noisy ESP32 recordings.

    N1  ESPectre leave-one-room-out     bedroom / living room / hobby room (real homes, 5 chip types)
    N2  pipeline quiet -> congested     same room, receiver forced onto the most congested WiFi channel
    N3  pipeline room 1 -> room 2       unseen room
    N4  all other ESP32 data -> pipeline   new setup, new hardware, congested channel included

Tasks
    presence    empty (0) vs occupied (1)
    motion      empty (0) / static person (1) / moving person (2)

Models: LightGBM (features), CNN, transformer (fine-tuned from the self-supervised
checkpoint pre-trained on WiGesture + WiFall). 5-window median smoothing.

Key metric for fans / AC: the FALSE-ALARM rate = share of empty-room windows called occupied.

Usage:  python experiments_noise.py N1 N2 N3 N4      (default: all)
"""
from __future__ import annotations

import json
import shutil
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
from build import build, build_seq, feature_cols
from trainlog import log

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
MODELS = ROOT / "models"
SMOOTH = 5
MOTION = {"empty": 0, "static": 1, "moving": 2}

LGB_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  class_weight="balanced", verbose=-1, n_jobs=8)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load(names):
    dfs, seqs, gaps = [], [], []
    for n in names:
        dfs.append(build(n))
        s, g = build_seq(n)
        seqs.append(s)
        gaps.append(g)
    df = pd.concat(dfs, ignore_index=True)
    df["motion_id"] = df["motion"].map(MOTION).fillna(-1).astype(int)
    return df, np.concatenate(seqs), np.concatenate(gaps)


def pretrained_state():
    """Reuse V2's self-supervised checkpoint (copied into V3/models, V2 untouched)."""
    path = MODELS / "transformer_pretrained.pt"
    if not path.exists():
        MODELS.mkdir(exist_ok=True)
        shutil.copy(ROOT.parent / "V2" / "models" / "transformer_pretrained.pt", path)
    return torch.load(path, map_location="cpu")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def smooth(test, pred):
    s = pd.Series(pred, index=test.index, dtype=float)
    out = s.groupby(test["recording"]).transform(lambda x: x.rolling(SMOOTH, min_periods=1).median())
    return np.round(out.to_numpy()).astype(int)


def scores(y, p, n_cls):
    y, p = np.asarray(y), np.asarray(p)
    empty = y == 0
    return dict(
        n=int(len(y)),
        acc=float((y == p).mean()),
        bal_acc=float(balanced_accuracy_score(y, p)),
        macro_f1=float(f1_score(y, p, labels=np.unique(y), average="macro", zero_division=0)),
        false_alarm=float((p[empty] > 0).mean()) if empty.any() else None,      # empty called occupied
        miss=float((p[~empty] == 0).mean()) if (~empty).any() else None,        # occupied called empty
        confusion=confusion_matrix(y, p, labels=list(range(n_cls))).tolist(),
    )


# ---------------------------------------------------------------------------
# One split
# ---------------------------------------------------------------------------

def run_split(name, df, seq, gap, train_mask, test_mask, task, seeds=(0,), epochs=30) -> dict:
    ycol, n_cls = ("presence", 2) if task == "presence" else ("motion_id", 3)
    keep = df[ycol] >= 0
    train, test = df[train_mask & keep], df[test_mask & keep]
    cols = feature_cols(df.drop(columns=["motion_id"]))
    tr, te = train.index.to_numpy(), test.index.to_numpy()
    y_tr, y_te = train[ycol].to_numpy(), test[ycol].to_numpy()
    log(f"{name} [{task}] train {len(train)}  test {len(test)}  "
        f"train classes {np.bincount(y_tr, minlength=n_cls).tolist()}  test {np.bincount(y_te, minlength=n_cls).tolist()}")

    res, t = {}, time.time()

    # LightGBM
    proba = np.zeros((len(te), n_cls))
    for sd in seeds:
        m = lgb.LGBMClassifier(random_state=sd, **LGB_PARAMS).fit(train[cols], y_tr)
        p = m.predict_proba(test[cols])
        for j, c in enumerate(m.classes_):
            proba[:, c] += p[:, j] / len(seeds)
    all_proba = {"lgbm": proba}

    # deep models
    init = pretrained_state()
    for kind, st in [("cnn", None), ("tf", init)]:
        proba = np.zeros((len(te), n_cls))
        for sd in seeds:
            net = deep.fit(kind, seq[tr], gap[tr], y_tr, seed=sd, epochs=epochs, init_state=st,
                           tag=f"{name} {task} {kind} s{sd}")
            proba += deep.predict_proba(net, seq[te], gap[te])[:, :n_cls] / len(seeds)
        all_proba[kind] = proba
    all_proba["ensemble"] = (all_proba["lgbm"] + all_proba["cnn"] + all_proba["tf"]) / 3

    for k, proba in all_proba.items():
        pred = proba.argmax(axis=1)
        res[k] = dict(window=scores(y_te, pred, n_cls), smoothed=scores(y_te, smooth(test, pred), n_cls))
        m = res[k]["smoothed"]
        fa = "-" if m["false_alarm"] is None else f"{m['false_alarm']:.3f}"
        log(f"{name} [{task}] <- {k:9s} bal {m['bal_acc']:.3f}  acc {m['acc']:.3f}  F1 {m['macro_f1']:.3f}  "
            f"false-alarm {fa}  (smoothed)")
    log(f"{name} [{task}] done in {time.time() - t:.0f} s")
    return res


def save(name, obj):
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1))


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

def n1():
    df, seq, gap = load(["espectre"])
    df = df[df["role"] != "exclude"]                      # authors' own 'exclude' recordings
    out = {}
    for room in ["bedroom", "living_room", "hobby_room"]:
        for task in ["presence", "motion"]:
            out[f"{room}|{task}"] = run_split(f"N1 test={room}", df, seq, gap,
                                              df["room"] != room, df["room"] == room, task)
    save("N1_espectre_leave_room_out", out)


def n2():
    df, seq, gap = load(["pipeline"])
    out = {}
    for task in ["presence", "motion"]:
        out[task] = run_split("N2 quiet->congested", df, seq, gap,
                              df["recording"].str.startswith("room1_quiet"),
                              df["recording"].str.startswith("room1_noisy"), task)
    save("N2_pipeline_quiet_to_congested", out)


def n3():
    df, seq, gap = load(["pipeline"])
    out = {}
    for task in ["presence", "motion"]:
        out[task] = run_split("N3 room1->room2", df, seq, gap, df["room"] == "room1", df["room"] == "room2", task)
    save("N3_pipeline_room1_to_room2", out)


def n4():
    df, seq, gap = load(["espectre", "wicount", "thesis", "wigesture", "wifall", "pipeline"])
    df = df[df["role"] != "exclude"]
    out = {"presence": run_split("N4 all->pipeline", df, seq, gap,
                                 df["dataset"] != "pipeline", df["dataset"] == "pipeline", "presence")}
    save("N4_all_to_pipeline", out)


# ---------------------------------------------------------------------------
# Calibrated versions: 30 s empty-room reference per site, never scored
# ---------------------------------------------------------------------------

def calibrated(df, seq):
    """Calibrated table + sequences; returns (dfc, seqc, usable) where usable = has a
    reference and is not itself a reference window."""
    from calibrate_v3 import calibrate_all
    cols = feature_cols(df.drop(columns=["motion_id"]))
    dfc, seqc, ref, has_ref = calibrate_all(df, seq, cols)
    dfc = dfc.drop(columns=["site_cal"])
    log(f"calibration: {has_ref.mean():.0%} of windows have an empty-room reference; "
        f"{int(ref.sum())} reference windows excluded from scoring")
    return dfc, seqc, has_ref & ~ref


def n1c():
    df, seq, gap = load(["espectre"])
    df = df[df["role"] != "exclude"]
    dfc, seqc, ok = calibrated(df, seq)
    out = {}
    for room in ["bedroom", "living_room", "hobby_room"]:
        for task in ["presence", "motion"]:
            out[f"{room}|{task}"] = run_split(f"N1c test={room}", dfc, seqc, gap,
                                              (dfc["room"] != room) & ok, (dfc["room"] == room) & ok, task)
    save("N1c_espectre_leave_room_out_calibrated", out)


def n2c():
    df, seq, gap = load(["pipeline"])
    dfc, seqc, ok = calibrated(df, seq)
    out = {}
    for task in ["presence", "motion"]:
        out[task] = run_split("N2c quiet->congested", dfc, seqc, gap,
                              dfc["recording"].str.startswith("room1_quiet") & ok,
                              dfc["recording"].str.startswith("room1_noisy") & ok, task)
    save("N2c_pipeline_quiet_to_congested_calibrated", out)


def n3c():
    df, seq, gap = load(["pipeline"])
    dfc, seqc, ok = calibrated(df, seq)
    out = {}
    for task in ["presence", "motion"]:
        out[task] = run_split("N3c room1->room2", dfc, seqc, gap,
                              (dfc["room"] == "room1") & ok, (dfc["room"] == "room2") & ok, task)
    save("N3c_pipeline_room1_to_room2_calibrated", out)


PROTOCOLS = {"N1": n1, "N2": n2, "N3": n3, "N4": n4, "N1c": n1c, "N2c": n2c, "N3c": n3c}

if __name__ == "__main__":
    for name in (sys.argv[1:] or list(PROTOCOLS)):
        PROTOCOLS[name]()

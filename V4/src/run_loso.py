"""Leave-one-source-out presence test: the FAIR comparison with frozen V3.

For each target source, V4 models are trained on every OTHER presence source (train + val
splits) and tested on ALL recordings of the target. Nothing from the target is used for
training, epochs or thresholds (threshold fixed at 0.5, deep models 10 epochs).

V3 never saw HomeOccupancy or OfficeLocalization, so those two targets compare V3 and V4
on equal terms. V3 was trained on all of ESPectre and pipeline, so for those targets only
the V4 numbers are meaningful (V3 rows are marked 'seen by V3').

Usage:  python V4/src/run_loso.py
        python V4/src/run_loso.py --prep robust --augment
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import log, new_run, save_result
from calibration import calibrate
from data import PRESENCE_SOURCES, feature_cols, load
import metrics
import models

KINDS = ["lgbm_raw", "lgbm_cal", "lgbm_norssi_cal", "gapcnn_raw", "gapcnn_cal", "cnn_raw"]
EPOCHS = 10
SEEN_BY_V3 = {"espectre", "pipeline"}


def fit_score(name, table, seq, gap, train):
    kind, cal = name.rsplit("_", 1)
    y = table["presence"].to_numpy()
    w = models.balanced_weights(table[train].reset_index(drop=True), "presence")
    if kind.startswith("lgbm"):
        cols = feature_cols(table)
        if kind == "lgbm_norssi":
            cols = [c for c in cols if not c.startswith("rssi")]
        return np.mean([models.train_lgbm(table[train], y[train], w, seed=s, cols=cols)[0]
                        .predict_proba(table[cols])[:, 1] for s in (0, 1, 2)], axis=0)
    net, _ = models.train_net(kind, seq[train], gap[train], y[train], w, 2, EPOCHS, seed=0, tag=name)
    return models.predict_net(net, seq, gap)[:, 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prep", choices=["standard", "robust"], default="standard")
    parser.add_argument("--augment", action="store_true")
    args = parser.parse_args()
    variant = args.prep + ("_aug" if args.augment else "")
    folder = new_run(f"loso_{variant}", vars(args))

    table, seq, gap = load(PRESENCE_SOURCES, args.prep)
    if args.augment:
        from noise import augmented
        at, aseq, agap = augmented(PRESENCE_SOURCES, args.prep)
        table = pd.concat([table, at], ignore_index=True)
        seq, gap = np.concatenate([seq, aseq]), np.concatenate([gap, agap])
    table_cal, seq_cal, ref, has_ref = calibrate(table, seq)
    valid = table["valid"].to_numpy()
    is_aug = table["window_id"].str.contains("#aug").to_numpy()
    usable = valid & (table["split"] != "unused").to_numpy()

    result = {"variant": variant, "threshold": 0.5, "targets": {}}
    for target in PRESENCE_SOURCES:
        in_target = (table["source"] == target).to_numpy()
        train = usable & ~in_target
        test = usable & in_target & ~is_aug & has_ref & ~ref          # same windows for every model
        log(f"LOSO target {target}: train {train.sum()}  test {test.sum()}")
        out = {}

        names = [] if args.augment or args.prep == "robust" else ["v3_cnn_raw", "v3_tf_raw", "v3_cnn_cal"]
        for name in names + KINDS:
            cal = name.endswith("_cal")
            t, s_ = (table_cal, seq_cal) if cal else (table, seq)
            if name.startswith("v3_"):
                fn = models.v3_presence(name.split("_")[1])
                score = np.full(len(table), np.nan)
                score[test] = fn(t[test], s_[test], gap[test])
            else:
                tr = train & has_ref if cal else train
                score = fit_score(name, t, s_, gap, tr)
            m = metrics.presence(table[test], score[test], 0.5)
            m["per_recording"] = metrics.per_recording_presence(table[test], score[test], 0.5)
            m["seen_by_v3"] = name.startswith("v3_") and target in SEEN_BY_V3
            out[name] = m
            log(f"  {target:20s} {name:18s} bal {m['bal_acc']:.3f}  FA {m['false_alarm']:.3f}  miss {m['miss']:.3f}")
        result["targets"][target] = out

    save_result(folder, result)


if __name__ == "__main__":
    main()

"""V4 presence experiment: frozen V3 baselines vs V4 models, raw vs calibrated.

Training data   train split of espectre + pipeline + homeoccupancy + officelocalization
Selection       validation split only (epochs, thresholds, model choice)
Test groups     espectre/test            unseen sessions (ESPectre's own holdout)
                pipeline/congested       room 1 on the most congested Wi-Fi channel
                pipeline/unseen_room     room 2
                homeoccupancy/test_dev   recording 3 (already inspected -> development benchmark)
                officelocalization/test_external   published test files (never used before)
                officelocalization/test_extra_empty  two extra empty recordings (false alarms only)

Models          V3 frozen: lgbm, cnn, tf (raw; deep ones also with calibration)
                V4: lgbm, lgbm_norssi (RSSI ablation), cnn (no gap mask), gapcnn; each raw and calibrated

Usage:  python V4/src/run_presence.py                         standard preprocessing
        python V4/src/run_presence.py --prep robust --augment  V4 robustness variant
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from common import MODELS, log, new_run, save_result
from calibration import calibrate
from data import PRESENCE_SOURCES, feature_cols, load
import metrics
import models

V4_KINDS = ["lgbm", "lgbm_norssi", "cnn", "gapcnn"]
LGBM_SEEDS = [0, 1, 2]
DEEP_EPOCHS = 20


# ---------------------------------------------------------------------------
# Evaluation groups
# ---------------------------------------------------------------------------

def eval_group(table: pd.DataFrame) -> pd.Series:
    g = table["source"] + "/" + table["split"]
    pipe = (table["source"] == "pipeline") & (table["split"] == "test")
    g[pipe & table["recording"].str.startswith("room1_noisy")] = "pipeline/congested"
    g[pipe & table["recording"].str.startswith("room2")] = "pipeline/unseen_room"
    return g


TEST_GROUPS = ["espectre/test", "pipeline/congested", "pipeline/unseen_room", "homeoccupancy/test_dev",
               "officelocalization/test_external", "officelocalization/test_extra_empty"]


def score_groups(table, score, threshold, mask):
    """Metrics per test group on rows in `mask`, plus the pooled validation score."""
    out = {}
    groups = eval_group(table)
    for g in TEST_GROUPS + ["val"]:
        sel = mask & ((table["split"] == "val").to_numpy() if g == "val" else (groups == g).to_numpy())
        if sel.sum() == 0:
            continue
        t = table[sel]
        out[g] = metrics.presence(t, score[sel], threshold)
        out[g]["raw_window"] = metrics.presence(t, score[sel], threshold, smoothed=False)
    return out


def summary(groups: dict) -> dict:
    """Mean over test groups: balanced accuracy (where both classes exist), false alarm, miss."""
    bal = [groups[g]["bal_acc"] for g in TEST_GROUPS if g in groups and groups[g]["bal_acc"] is not None]
    fa = [groups[g]["false_alarm"] for g in TEST_GROUPS if g in groups and groups[g]["false_alarm"] is not None]
    miss = [groups[g]["miss"] for g in TEST_GROUPS if g in groups and groups[g]["miss"] is not None]
    return {"mean_bal_acc": float(np.mean(bal)) if bal else None, "mean_false_alarm": float(np.mean(fa)) if fa else None,
            "mean_miss": float(np.mean(miss)) if miss else None, "worst_false_alarm": float(np.max(fa)) if fa else None,
            "worst_miss": float(np.max(miss)) if miss else None}


# ---------------------------------------------------------------------------
# Training one V4 model
# ---------------------------------------------------------------------------

def fit_v4(kind, table, seq, gap, train, val, out_dir, name):
    """Train on rows `train`, select on rows `val`; return P(occupied) for every row + info."""
    y = table["presence"].to_numpy()
    w = models.balanced_weights(table[train].reset_index(drop=True), "presence")
    info = {"kind": kind, "n_train": int(train.sum()), "n_val": int(val.sum())}

    if kind.startswith("lgbm"):
        cols = feature_cols(table)
        if kind == "lgbm_norssi":
            cols = [c for c in cols if not c.startswith("rssi")]
        score = np.zeros(len(table))
        for seed in LGBM_SEEDS:
            m, cols = models.train_lgbm(table[train], y[train], w, seed=seed, cols=cols)
            score += models.predict_lgbm(m, cols, table)[:, 1] / len(LGBM_SEEDS)
            m.booster_.save_model(str(out_dir / f"{name}_s{seed}.txt"))
        (out_dir / f"{name}.json").write_text(json.dumps({"kind": kind, "features": cols, "seeds": LGBM_SEEDS}, indent=2))
        info["files"] = [f"{name}_s{s}.txt" for s in LGBM_SEEDS]
    else:
        net, hist = models.train_net(kind, seq[train], gap[train], y[train], w, 2, DEEP_EPOCHS, seed=0,
                                     val=(seq[val], gap[val], y[val]), tag=name)
        score = models.predict_net(net, seq, gap)[:, 1]
        models.save_net(net, kind, 2, out_dir / f"{name}.pt", {"history": hist})
        info.update(files=[f"{name}.pt"], history=hist,
                    best_epoch=int(np.argmax([h["val_bal_acc"] for h in hist])))
    return score, info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prep", choices=["standard", "robust"], default="standard")
    parser.add_argument("--augment", action="store_true", help="add one noisy copy of every training recording")
    parser.add_argument("--kinds", nargs="+", default=V4_KINDS, choices=V4_KINDS)
    args = parser.parse_args()

    variant = args.prep + ("_aug" if args.augment else "")
    folder = new_run(f"presence_{variant}", vars(args))
    out_dir = MODELS / "presence" / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ------------------------------------------------------------
    table, seq, gap = load(PRESENCE_SOURCES, args.prep)
    if args.augment:
        from noise import augmented
        at, aseq, agap = augmented(PRESENCE_SOURCES, args.prep)
        table = pd.concat([table, at], ignore_index=True)
        seq, gap = np.concatenate([seq, aseq]), np.concatenate([gap, agap])
    assert table["window_id"].is_unique and len(table) == len(seq) == len(gap)

    table_cal, seq_cal, ref, has_ref = calibrate(table, seq)
    valid = table["valid"].to_numpy()
    split = table["split"].to_numpy()
    train, val = valid & (split == "train"), valid & (split == "val")
    is_eval = ~np.isin(split, ["train", "unused"]) & ~table["window_id"].str.contains("#aug").to_numpy()
    common = valid & has_ref & ~ref                       # same windows for raw and calibrated scoring
    log(f"presence {variant}: {len(table)} windows, train {train.sum()}, val {val.sum()}, "
        f"common-eval coverage {common[is_eval].mean():.2f}")

    scores, infos, result = {}, {}, {"variant": variant, "models": {}}

    # ---- frozen V3 baselines (raw only for evaluation rows) ---------------
    if args.prep == "standard" and not args.augment:
        for kind in ["lgbm", "cnn", "tf"]:
            fn = models.v3_presence(kind)
            s = np.full(len(table), np.nan)
            s[is_eval] = fn(table[is_eval], seq[is_eval], gap[is_eval])
            scores[f"v3_{kind}_raw"] = s
            if kind != "lgbm":                            # V3 LightGBM features are not calibration-compatible
                sc = np.full(len(table), np.nan)
                m = is_eval & has_ref
                sc[m] = fn(table_cal[m], seq_cal[m], gap[m])
                scores[f"v3_{kind}_cal"] = sc

    # ---- V4 models --------------------------------------------------------
    for kind in args.kinds:
        for cal in ["raw", "cal"]:
            name = f"{kind}_{cal}"
            t, s_ = (table_cal, seq_cal) if cal == "cal" else (table, seq)
            tr = train & has_ref if cal == "cal" else train
            va = val & has_ref if cal == "cal" else val
            started = time.time()
            scores[name], infos[name] = fit_v4(kind, t, s_, gap, tr, va, out_dir, name)
            infos[name]["seconds"] = round(time.time() - started, 1)
            log(f"  trained {name} in {infos[name]['seconds']} s")

    # ---- thresholds on validation, then score everything -----------------
    for name, s in scores.items():
        v = val & ~np.isnan(s)
        if name.endswith("_cal"):
            v &= has_ref & ~ref
        thr = metrics.best_threshold(table[v], s[v])
        entry = {"threshold_from_val": thr, "info": infos.get(name, {"frozen": "V3"})}
        ok = ~np.isnan(s)
        if name.endswith("_raw"):
            entry["all_valid"] = score_groups(table, np.nan_to_num(s), thr, valid & ok)
            entry["all_valid_default_threshold"] = score_groups(table, np.nan_to_num(s), 0.5, valid & ok)
        entry["common"] = score_groups(table, np.nan_to_num(s), thr, common & ok)
        entry["summary_common"] = summary(entry["common"])
        if "all_valid" in entry:
            entry["summary_all_valid"] = summary(entry["all_valid"])
        entry["val_bal_acc"] = entry["common"].get("val", {}).get("bal_acc")
        entry["tradeoff_common"] = {g: metrics.tradeoff_curve(table[common & ok & (eval_group(table) == g).to_numpy()],
                                                              s[common & ok & (eval_group(table) == g).to_numpy()])
                                    for g in TEST_GROUPS if (common & ok & (eval_group(table) == g).to_numpy()).any()}
        entry["per_recording"] = metrics.per_recording_presence(table[common & ok & is_eval], s[common & ok & is_eval], thr)
        result["models"][name] = entry
        sm = entry["summary_common"]
        log(f"  {name:20s} val {entry['val_bal_acc']}  test mean bal {sm['mean_bal_acc']}  "
            f"FA {sm['mean_false_alarm']}  miss {sm['mean_miss']}")

    # ---- candidate choice: best validation balanced accuracy among V4 models
    v4 = {k: v for k, v in result["models"].items() if not k.startswith("v3_")}
    best = max(v4, key=lambda k: v4[k]["val_bal_acc"] or 0)
    result["selected_on_validation"] = best
    result["coverage"] = {"common_share_of_eval_windows": float(common[is_eval].mean()),
                          "rejected_share_of_eval_windows": float(1 - valid[is_eval].mean())}

    np.savez_compressed(folder / "scores.npz", window_id=table["window_id"].to_numpy()[is_eval],
                        **{k: v[is_eval] for k, v in scores.items()})
    save_result(folder, result, models=sorted(out_dir.glob("*.pt")) + sorted(out_dir.glob("*.txt")))
    log(f"selected on validation: {best}")


if __name__ == "__main__":
    main()

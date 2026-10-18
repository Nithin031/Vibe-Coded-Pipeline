"""Noise stress test for presence models (synthetic noise on the raw stream).

For every validation / test recording (first 180 s), each noise type and level is applied
to the raw packets; the noisy stream then goes through the full pipeline (grid, gap mask,
quality gate, features, calibration from the SAME noisy stream, model). Validation noise
uses seeds 1000+, test noise 2000+, so improvements chosen on validation are not tuned to
the exact test noise.

Usage:  python V4/src/run_noise.py --variant standard
        python V4/src/run_noise.py --variant robust_aug
"""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import METRICS, MODELS, PROTOCOL, log, new_run, save_result
from calibration import calibrate
from data import PRESENCE_SOURCES, make_windows, recordings
import metrics
import models
import noise
from run_presence import TEST_GROUPS, eval_group, summary

CROP_S = float(PROTOCOL["noise_stress"]["eval_crop_s"])
EVAL_SPLITS = ["val", "test", "test_dev", "test_external", "test_extra_empty"]
CONDITIONS = [("clean", 0)] + [(k, l) for k in noise.KINDS if k != "crazy" for l in (1, 2, 3)] + [("crazy", 3)]


# ---------------------------------------------------------------------------
# Saved models of one variant
# ---------------------------------------------------------------------------

def load_variant(variant: str):
    """{name: (predict(table, seq, gap) -> P(occupied), calibrated?, threshold)}."""
    runs = sorted(METRICS.glob(f"presence_{variant}_*/result.json"))
    result = json.loads(runs[-1].read_text())
    folder = MODELS / "presence" / variant
    out = {}
    for name, entry in result["models"].items():
        thr = entry["threshold_from_val"]
        cal = name.endswith("_cal")
        if name.startswith("v3_"):
            kind = name.split("_")[1]
            out[name] = (models.v3_presence(kind), cal, thr)
            continue
        info = json.loads((folder / f"{name}.json").read_text())
        if name.startswith("lgbm"):
            boosters = [lgb.Booster(model_file=str(folder / f"{name}_s{s}.txt")) for s in info["seeds"]]
            cols = info["features"]
            out[name] = ((lambda b, c: lambda t, s_, g_: np.mean([x.predict(t[c]) for x in b], axis=0))(boosters, cols),
                         cal, thr)
        else:
            net, _ = models.load_net(folder / f"{name}.pt")
            out[name] = ((lambda n: lambda t, s_, g_: models.predict_net(n, s_, g_)[:, 1])(net), cal, thr)
    return out, runs[-1].parent.name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="standard", help="presence model variant folder (standard, robust_aug, ...)")
    parser.add_argument("--sources", nargs="+", default=PRESENCE_SOURCES)
    args = parser.parse_args()

    prep = "robust" if args.variant.startswith("robust") else "standard"
    model_set, source_run = load_variant(args.variant)
    folder = new_run(f"noise_{args.variant}", vars(args) | {"source_run": source_run})
    log(f"noise stress: variant {args.variant} ({prep} prep), {len(model_set)} models, {len(CONDITIONS)} conditions")

    # per condition: list of (table, {model: score}) parts, one per source
    parts = {c: [] for c in CONDITIONS}
    for source in args.sources:
        recs = [noise.crop(r, CROP_S) for r in recordings(source) if r.meta["split"] in EVAL_SPLITS]
        log(f"  {source}: {len(recs)} recordings")
        for cond in CONDITIONS:
            rows, sq, gp = [], [], []
            for i, rec in enumerate(recs):
                seed = (1000 if rec.meta["split"] == "val" else 2000) + i
                r, q, g = make_windows(noise.apply(rec, cond[0], cond[1], seed), robust=(prep == "robust"))
                rows += r
                sq.append(q)
                gp.append(g)
            table = pd.DataFrame(rows)
            seq, gap = np.concatenate(sq), np.concatenate(gp)
            table_cal, seq_cal, ref, has_ref = calibrate(table, seq)
            table["common"] = table["valid"].to_numpy() & has_ref & ~ref
            scores = {}
            for name, (fn, cal, thr) in model_set.items():
                s = np.full(len(table), np.nan)
                m = has_ref if cal else np.ones(len(table), bool)
                if m.any():
                    s[m] = fn(table_cal[m], seq_cal[m], gap[m]) if cal else fn(table, seq, gap)
                scores[name] = s
            parts[cond].append((table, scores))
        log(f"  {source} done")

    # metrics per condition, model and group (common windows: identical for raw and calibrated)
    result = {"variant": args.variant, "source_run": source_run, "conditions": {}}
    for cond, plist in parts.items():
        table = pd.concat([p[0] for p in plist], ignore_index=True)
        key = f"{cond[0]}_{cond[1]}"
        result["conditions"][key] = {"rejected_rate": float(1 - table["valid"].mean())}
        groups = eval_group(table)
        for name, (_, cal, thr) in model_set.items():
            s = np.concatenate([p[1][name] for p in plist])
            per = {}
            for g in TEST_GROUPS + ["val"]:
                sel = table["common"].to_numpy() & ~np.isnan(s)
                sel &= (table["split"] == "val").to_numpy() if g == "val" else (groups == g).to_numpy()
                if sel.any():
                    per[g] = metrics.presence(table[sel], s[sel], thr)
            result["conditions"][key][name] = {"groups": per, "summary": summary(per),
                                               "val_bal_acc": per.get("val", {}).get("bal_acc")}
        log(f"  scored {key}")

    save_result(folder, result)


if __name__ == "__main__":
    main()

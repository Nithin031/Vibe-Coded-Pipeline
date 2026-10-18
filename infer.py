"""Run the V4 models on a recorded ESP32 CSI file (offline; live capture not yet tested).

Inputs
    --reference   CSV of the EMPTY room, >= 30 s, same boards / positions / channel
    --input       CSV to analyse (esp-csi format, see README.md)
    --env         clean | noisy   (which counting model + preprocessing to use)
    --model       lgbm | cnn      (presence model; lgbm is the V4 final choice)

Output: one row per second (CSV printed or written with --out):
    t_end_s     end of the 3 s window, seconds from the first packet
    quality     ok | unknown   (unknown = packet gaps too long; no prediction made)
    p_occupied  presence probability after calibration (raw, before smoothing)
    presence    empty | occupied  (5-window causal median, threshold from the model card)
    count       0-5 people (5-window median; 4 and 5 are untrained, see README.md)

Usage:  python V4/deploy/infer.py --reference empty.csv --input room.csv --out result.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from data import Recording, feature_cols, make_windows, read_esp_csv    # noqa: E402
from calibration import EPS_CSI, EPS_RSSI, REF_S, SEQ_SCALE, energy_cols  # noqa: E402
import metrics                                                          # noqa: E402
import models                                                           # noqa: E402

CARD = json.loads((HERE / "model_card.json").read_text())


# ---------------------------------------------------------------------------
# Reading and windows
# ---------------------------------------------------------------------------

def windows_of(path: Path, robust: bool):
    head = pd.read_csv(path, nrows=200, dtype=str, on_bad_lines="skip")
    n_values = int(pd.to_numeric(head["len"], errors="coerce").mode()[0])
    amp, t, rssi, stats = read_esp_csv(path, n_values)
    meta = {"source": "live", "recording": path.stem, "split": "live", "site": "live", "label_raw": "",
            "presence": -1, "motion": "", "activity": "", "count": -1, "chip": "", "room": ""}
    rows, seq, gap = make_windows(Recording(meta, amp, t, rssi), robust=robust)
    return pd.DataFrame(rows), seq, gap, stats


# ---------------------------------------------------------------------------
# Calibration with the empty reference (same maths as V4/src/calibration.py)
# ---------------------------------------------------------------------------

def reference_levels(ref_table, ref_seq):
    use = (ref_table["valid"] & (ref_table["t0"] + 3.0 <= REF_S)).to_numpy()
    if use.sum() < 10:
        sys.exit(f"reference has only {use.sum()} valid windows in its first {REF_S:.0f} s (need >= 10)")
    ecols = energy_cols(feature_cols(ref_table))
    med = ref_table.loc[use, ecols].median()
    level = np.maximum(np.median(np.abs(ref_seq[use].astype(np.float32)), axis=(0, 2)), 1e-4)
    return ecols, med, level


def apply_calibration(table, seq, ecols, med, level):
    table = table.copy()
    eps = pd.Series({c: EPS_RSSI if c.startswith("rssi") else EPS_CSI for c in ecols})
    table[ecols] = np.log((table[ecols] + eps) / (med + eps))
    seq_cal = (seq.astype(np.float32) / level[None, :, None] * SEQ_SCALE).astype(np.float16)
    return table, seq_cal


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def presence_scores(kind, table, seq, gap):
    info = CARD["presence"][kind]
    if kind == "lgbm":
        cols = json.loads((HERE / "presence" / "lgbm_cal.json").read_text())["features"]
        boosters = [lgb.Booster(model_file=str(HERE / f)) for f in info["files"]]
        return np.mean([b.predict(table[cols]) for b in boosters], axis=0), info["threshold"]
    net, _ = models.load_net(HERE / info["files"][0])
    return models.predict_net(net, seq, gap)[:, 1], info["threshold"]


def count_predictions(env, seq, gap):
    net, _ = models.load_net(HERE / CARD["counting"][env]["files"][0])
    return models.predict_net(net, seq, gap).argmax(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--env", choices=["clean", "noisy"], default="clean")
    parser.add_argument("--model", choices=["lgbm", "cnn"], default="lgbm")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    # presence always uses the standard preprocessing (the robust variant lost on validation)
    ref_t, ref_s, _, _ = windows_of(args.reference, robust=False)
    tab, seq, gap, stats = windows_of(args.input, robust=False)
    ecols, med, level = reference_levels(ref_t, ref_s)
    tab_cal, seq_cal = apply_calibration(tab, seq, ecols, med, level)
    p, threshold = presence_scores(args.model, tab_cal, seq_cal, gap)

    # counting: 'noisy' uses the robust preprocessing it was trained with
    if args.env == "noisy":
        _, cseq, cgap, _ = windows_of(args.input, robust=True)
    else:
        cseq, cgap = seq, gap
    counts = count_predictions(args.env, cseq, cgap)

    valid = tab["valid"].to_numpy()
    out = pd.DataFrame({"t_end_s": tab["t0"] + 3.0, "quality": np.where(valid, "ok", "unknown"),
                        "p_occupied": np.round(p, 4)})
    ok = tab[valid]
    pres = np.full(len(tab), "", dtype=object)
    cnt = np.full(len(tab), "", dtype=object)
    pres[valid] = np.where(metrics.smooth(ok, p[valid]) >= threshold, "occupied", "empty")
    cnt[valid] = np.clip(np.round(metrics.smooth(ok, counts[valid])), 0, 5).astype(int).astype(str)
    out["presence"], out["count"] = pres, cnt
    out.loc[~valid, "p_occupied"] = np.nan

    print(f"# {args.input.name}: {stats['packets_kept']} packets, {stats['rate_hz']:.0f} Hz, "
          f"{len(out)} windows, {(~valid).mean():.0%} unknown", file=sys.stderr)
    if args.out:
        out.to_csv(args.out, index=False)
    else:
        print(out.to_csv(index=False))


if __name__ == "__main__":
    main()

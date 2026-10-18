"""V4 people counting (0-5 head, trained on the 0-3 that real ESP32 data supports).

Data      WiCount (one meeting room, ESP32-S3): count 0 x1, 1 x7, 2 x6, 3 x1 recordings of ~60 s
          OTH room2 samples: one ~60 s recording per count 0-3 (different room and setup)

Folds     3 folds. Counts 1 and 2: held-out RECORDINGS (groups 3/2/2 and 2/2/2).
          Counts 0 and 3 (single recordings): fold k tests time block k of 3 (weaker evidence).
          Windows that cross a block edge are dropped from that fold.

Models    lgbm (3 seeds) and gapcnn (30 epochs), fixed in advance: no validation selection.
Extra     cross-room checks (WiCount -> OTH, OTH -> WiCount), synthetic noise stress test on
          every fold's test windows, inference latency, and a spliced-stream reaction delay.

Usage:  python V4/src/run_counting.py
        python V4/src/run_counting.py --prep robust --augment
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from common import MODELS, log, new_run, save_result
from data import Recording, feature_cols, load, make_windows, recordings
import metrics
import models
import noise

N_CLASSES = 6
LGBM_SEEDS = [0, 1, 2]
CNN_EPOCHS = 30
KINDS = ["lgbm", "gapcnn"]


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------

def fold_masks(table: pd.DataFrame, k: int, n_folds: int = 3):
    """(train mask, test mask) for fold k of the WiCount protocol."""
    train = np.zeros(len(table), bool)
    test = np.zeros(len(table), bool)
    for count, sub in table.groupby("count"):
        recs = sorted(sub["recording"].unique())
        idx = sub.index.to_numpy()
        if len(recs) >= n_folds:
            groups = np.array_split(np.array(recs), n_folds)
            held = set(groups[k])
            is_test = sub["recording"].isin(held).to_numpy()
            test[idx[is_test]] = True
            train[idx[~is_test]] = True
        else:                                                   # single recording: time blocks
            for rec in recs:
                r = sub[sub["recording"] == rec]
                end = r["t0"].max() + 3.0
                edges = np.linspace(0, end, n_folds + 1)
                block = np.full(len(r), -1)
                for b in range(n_folds):
                    inside = (r["t0"] >= edges[b]) & (r["t0"] + 3.0 <= edges[b + 1])
                    block[inside.to_numpy()] = b
                ri = r.index.to_numpy()
                test[ri[block == k]] = True
                train[ri[(block >= 0) & (block != k)]] = True
    assert not (train & test).any()
    return train, test


def evidence(table):
    """Which kind of split each count used (for honest reporting)."""
    return {int(c): ("held-out recordings" if sub["recording"].nunique() >= 3 else "time blocks of one recording")
            for c, sub in table.groupby("count")}


# ---------------------------------------------------------------------------
# Train / predict one model
# ---------------------------------------------------------------------------

def fit_predict(kind, table, seq, gap, train, tag):
    """Train on `train`; return (predict function, info)."""
    y = table["count"].to_numpy()
    w = models.balanced_weights(table[train].reset_index(drop=True), "count")
    if kind == "lgbm":
        cols = feature_cols(table)
        fitted = [models.train_lgbm(table[train], y[train], w, seed=s, cols=cols)[0] for s in LGBM_SEEDS]

        def predict(t, s_, g_):
            p = np.zeros((len(t), N_CLASSES))
            for m in fitted:
                pm = m.predict_proba(t[cols])
                p[:, m.classes_] += pm / len(fitted)
            return p
        return predict, {"seeds": LGBM_SEEDS}, fitted

    net, hist = models.train_net("gapcnn", seq[train], gap[train], y[train], w, N_CLASSES, CNN_EPOCHS, seed=0, tag=tag)
    return (lambda t, s_, g_: models.predict_net(net, s_, g_)), {"history": hist}, net


# ---------------------------------------------------------------------------
# Noisy copies of every counting recording (seeds 2000+)
# ---------------------------------------------------------------------------

def noisy_windows(sources, prep):
    """{(kind, level): (table, seq, gap)} for every recording of `sources`."""
    out = {}
    recs = [r for s in sources for r in recordings(s)]
    for kind in noise.KINDS:
        for level in ([3] if kind == "crazy" else [1, 2, 3]):
            rows, sq, gp = [], [], []
            for i, rec in enumerate(recs):
                r, q, g = make_windows(noise.apply(rec, kind, level, seed=2000 + i), robust=(prep == "robust"))
                rows += r
                sq.append(q)
                gp.append(g)
            out[(kind, level)] = (pd.DataFrame(rows), np.concatenate(sq), np.concatenate(gp))
        log(f"  noisy counting windows: {kind}")
    return out


def align(base: pd.DataFrame, other: pd.DataFrame):
    """Row positions in `other` of each base window_id (-1 if the noisy stream lost it)."""
    pos = pd.Series(np.arange(len(other)), index=other["window_id"])
    return pos.reindex(base["window_id"]).fillna(-1).astype(int).to_numpy()


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------

def latency(predict_fns, table, seq, gap):
    """Per-window feature extraction + inference time (ms), batch size 1."""
    rec = next(recordings("wicount"))
    t0 = time.perf_counter()
    rows, sq, gp = make_windows(rec)
    feat_ms = (time.perf_counter() - t0) / max(len(rows), 1) * 1000
    out = {"preprocess_and_features_ms_per_window": feat_ms}
    for name, fn in predict_fns.items():
        t0 = time.perf_counter()
        for i in range(50):
            fn(table.iloc[i:i + 1], seq[i:i + 1], gap[i:i + 1])
        out[f"{name}_inference_ms_per_window"] = (time.perf_counter() - t0) / 50 * 1000
    return out


def splice_delay(predict_fn, prep):
    """Synthetic splice: empty recording then a 2-person recording. Seconds from the splice
    until the smoothed prediction first equals 2 and until it first says 'occupied'."""
    recs = {r.meta["recording"]: r for r in recordings("wicount")}
    empty = next(r for r in recs.values() if r.meta["count"] == 0)
    two = next(r for r in recs.values() if r.meta["count"] == 2)
    cut = empty.t[-1]
    amp = np.concatenate([empty.amp, two.amp])
    t = np.concatenate([empty.t, two.t + cut + 0.01])
    rssi = np.concatenate([empty.rssi, two.rssi])
    meta = dict(two.meta, recording="splice", count=-1)
    rows, sq, gp = make_windows(Recording(meta, amp, t, rssi), robust=(prep == "robust"))
    tab = pd.DataFrame(rows)
    pred = predict_fn(tab, sq, gp).argmax(1)
    smoothed = metrics.smooth(tab, pred)
    after = tab["t0"].to_numpy() + 3.0 - cut                   # window END relative to the splice
    first = lambda cond: float(after[np.flatnonzero(cond & (after > 0))[0]]) if (cond & (after > 0)).any() else None
    return {"note": "synthetic splice of two real recordings, window end time after the splice",
            "first_occupied_s": first(smoothed >= 1), "first_correct_count_s": first(smoothed == 2)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prep", choices=["standard", "robust"], default="standard")
    parser.add_argument("--augment", action="store_true", help="add one noisy copy of every recording (train side only)")
    parser.add_argument("--no-noise", action="store_true", help="skip the noise stress test")
    args = parser.parse_args()

    variant = args.prep + ("_aug" if args.augment else "")
    folder = new_run(f"counting_{variant}", vars(args))
    out_dir = MODELS / "counting" / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    wc, wseq, wgap = load(["wicount"], args.prep)
    oth, oseq, ogap = load(["oth"], args.prep)
    result = {"variant": variant, "evidence_per_count": evidence(wc),
              "classes": list(range(N_CLASSES)), "classes_with_training_data": [0, 1, 2, 3],
              "folds": {}, "pooled": {}, "cross_room": {}, "noise": {}}

    # augmentation: noisy copy of each recording; inherits fold membership through window_id
    aug = None
    if args.augment:
        rng = np.random.default_rng(7)
        rows, sq, gp = [], [], []
        for i, rec in enumerate(recordings("wicount")):
            kind = noise.KINDS[rng.integers(len(noise.KINDS))]
            r, q, g = make_windows(noise.apply(rec, kind, int(rng.integers(1, 4)), seed=i), robust=(args.prep == "robust"))
            rows += r
            sq.append(q)
            gp.append(g)
        aug = (pd.DataFrame(rows), np.concatenate(sq), np.concatenate(gp))

    noisy = None if args.no_noise else noisy_windows(["wicount"], args.prep)
    pooled_pred = {k: np.full(len(wc), np.nan) for k in KINDS}
    noisy_pred = {k: {} for k in KINDS}

    for k in range(3):
        train, test = fold_masks(wc, k)
        train &= wc["valid"].to_numpy()
        t_table, t_seq, t_gap, t_mask = wc, wseq, wgap, train
        if aug is not None:                                     # same fold rule for the noisy copies
            a_train, _ = fold_masks(aug[0], k)
            a_train &= aug[0]["valid"].to_numpy()
            t_table = pd.concat([wc, aug[0]], ignore_index=True)
            t_seq, t_gap = np.concatenate([wseq, aug[1]]), np.concatenate([wgap, aug[2]])
            t_mask = np.concatenate([train, a_train])

        result["folds"][k] = {"test_recordings": sorted(wc.loc[test, "recording"].unique()),
                              "n_train": int(t_mask.sum()), "n_test": int(test.sum())}
        for kind in KINDS:
            predict, info, _ = fit_predict(kind, t_table, t_seq, t_gap, t_mask, f"fold{k}")
            p = predict(wc[test], wseq[test], wgap[test]).argmax(1)
            pooled_pred[kind][test] = p
            result["folds"][k][kind] = metrics.counting(wc[test], p)
            log(f"  fold {k} {kind}: exact {result['folds'][k][kind]['exact_accuracy']:.3f} "
                f"MAE {result['folds'][k][kind]['mae']:.3f}")

            if noisy is not None:                                # same test windows under noise
                for cond, (nt, ns, ng) in noisy.items():
                    pos = align(wc[test], nt)
                    have = pos >= 0
                    sub = nt.iloc[pos[have]].reset_index(drop=True)
                    pn = predict(sub, ns[pos[have]], ng[pos[have]]).argmax(1)
                    noisy_pred[kind].setdefault(cond, []).append((sub, pn))

    # pooled over folds: every window is tested exactly once
    tested = ~np.isnan(pooled_pred["lgbm"])
    for kind in KINDS:
        p = pooled_pred[kind][tested].astype(int)
        t = wc[tested]
        result["pooled"][kind] = metrics.counting(t, p)
        result["pooled"][kind]["per_recording"] = metrics.per_recording_counting(t, p)
        result["pooled"][kind]["evidence_per_count"] = evidence(wc)       # how each count was split
        if noisy is not None:
            for cond, parts in noisy_pred[kind].items():
                sub = pd.concat([s for s, _ in parts], ignore_index=True)
                pn = np.concatenate([pp for _, pp in parts])
                result["noise"].setdefault(kind, {})[f"{cond[0]}_{cond[1]}"] = metrics.counting(sub, pn)

    # cross-room checks + final models trained on all WiCount
    fns = {}
    all_w = wc["valid"].to_numpy()
    for kind in KINDS:
        predict, info, fitted = fit_predict(kind, wc, wseq, wgap, all_w, "all-wicount")
        fns[kind] = predict
        result["cross_room"][f"{kind}_wicount_to_oth"] = metrics.counting(oth, predict(oth, oseq, ogap).argmax(1))
        if kind == "lgbm":
            for i, m in enumerate(fitted):
                m.booster_.save_model(str(out_dir / f"count_lgbm_s{i}.txt"))
        else:
            models.save_net(fitted, "gapcnn", N_CLASSES, out_dir / "count_gapcnn.pt",
                            {"trained_on": "all WiCount valid windows", "classes": list(range(N_CLASSES))})
        p_back, _, _ = fit_predict(kind, oth, oseq, ogap, oth["valid"].to_numpy(), "all-oth")
        result["cross_room"][f"{kind}_oth_to_wicount"] = metrics.counting(wc, p_back(wc, wseq, wgap).argmax(1))
        log(f"  cross-room {kind}: WiCount->OTH exact "
            f"{result['cross_room'][f'{kind}_wicount_to_oth']['exact_accuracy']:.3f}")

    result["latency"] = latency(fns, wc, wseq, wgap)
    result["splice_delay"] = {k: splice_delay(fn, args.prep) for k, fn in fns.items()}
    save_result(folder, result, models=sorted(out_dir.glob("*.pt")) + sorted(out_dir.glob("*.txt")))


if __name__ == "__main__":
    main()

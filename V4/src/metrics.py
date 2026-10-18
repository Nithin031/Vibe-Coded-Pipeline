"""Metrics for presence (empty / occupied) and people counting (0-5).

All metrics are computed on VALID windows only. Invalid (low-quality) windows are
reported as 'unknown' through `rejected_rate`; they are never counted as empty
or occupied.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support, roc_auc_score)

SMOOTH = 5                      # causal rolling median over 5 windows (V3)
COUNT_CLASSES = [0, 1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def smooth(table: pd.DataFrame, values: np.ndarray, k: int = SMOOTH) -> np.ndarray:
    """Causal rolling median within each recording, in time order."""
    s = pd.Series(np.asarray(values, dtype=float), index=table.index)
    order = table.sort_values(["recording", "t0"]).index
    out = s.loc[order].groupby(table.loc[order, "recording"]).transform(
        lambda x: x.rolling(k, min_periods=1).median())
    return out.reindex(table.index).to_numpy()


# ---------------------------------------------------------------------------
# Presence
# ---------------------------------------------------------------------------

def false_alerts_per_hour(table: pd.DataFrame, pred: np.ndarray) -> float | None:
    """Rising edges of 'occupied' inside EMPTY recordings, per hour of empty time."""
    empty = table["presence"].to_numpy() == 0
    if not empty.any():
        return None
    frame = table.loc[empty, ["recording", "t0"]].assign(p=np.asarray(pred)[empty]).sort_values(["recording", "t0"])
    edges = frame.groupby("recording")["p"].apply(lambda x: int(((x.values[1:] == 1) & (x.values[:-1] == 0)).sum()
                                                               + (x.values[0] == 1))).sum()
    hours = len(frame) / 3600.0                  # one window per second
    return float(edges / hours)


def presence(table: pd.DataFrame, score: np.ndarray, threshold: float = 0.5, smoothed: bool = True) -> dict:
    """Scores on valid windows. `score` is P(occupied) per window (row-aligned with table)."""
    valid = table["valid"].to_numpy()
    out = {"n_windows": int(len(table)), "rejected_rate": float(1 - valid.mean()) if len(table) else None}
    t = table[valid]
    s = np.asarray(score)[valid]
    if smoothed:
        s = smooth(t, s)
    y = t["presence"].to_numpy().astype(int)
    p = (s >= threshold).astype(int)
    empty, occ = y == 0, y == 1

    out.update(
        n=int(len(y)), threshold=float(threshold), smoothed=smoothed,
        bal_acc=float(balanced_accuracy_score(y, p)) if empty.any() and occ.any() else None,
        accuracy=float((y == p).mean()) if len(y) else None,
        false_alarm=float(p[empty].mean()) if empty.any() else None,
        miss=float((p[occ] == 0).mean()) if occ.any() else None,
        sensitivity=float(p[occ].mean()) if occ.any() else None,
        specificity=float((p[empty] == 0).mean()) if empty.any() else None,
        precision=float(y[p == 1].mean()) if (p == 1).any() else None,
        macro_f1=float(f1_score(y, p, average="macro", labels=[0, 1], zero_division=0)) if len(y) else None,
        auroc=float(roc_auc_score(y, s)) if empty.any() and occ.any() else None,
        auprc=float(average_precision_score(y, s)) if empty.any() and occ.any() else None,
        false_alerts_per_hour=false_alerts_per_hour(t, p),
        confusion=confusion_matrix(y, p, labels=[0, 1]).tolist(),
        mean_gap_ratio=float(table["missing"].mean()) if len(table) else None,
    )
    return out


def per_recording_presence(table: pd.DataFrame, score: np.ndarray, threshold: float) -> list[dict]:
    """Accuracy of each recording (valid windows, smoothed): shows session-to-session spread."""
    rows = []
    valid = table["valid"].to_numpy()
    t = table[valid]
    p = (smooth(t, np.asarray(score)[valid]) >= threshold).astype(int)
    for rec, idx in t.groupby("recording").indices.items():
        y = t["presence"].to_numpy()[idx]
        rows.append({"recording": rec, "source": t["source"].iloc[idx[0]], "label": int(y[0]),
                     "n": int(len(idx)), "accuracy": float((p[idx] == y).mean())})
    return rows


def best_threshold(table: pd.DataFrame, score: np.ndarray) -> float:
    """Threshold maximising smoothed balanced accuracy (VALIDATION data only)."""
    grid = np.linspace(0.05, 0.95, 19)
    vals = [presence(table, score, thr)["bal_acc"] or 0 for thr in grid]
    return float(grid[int(np.argmax(vals))])


def tradeoff_curve(table: pd.DataFrame, score: np.ndarray) -> list[dict]:
    """False-alarm rate vs miss rate over thresholds (for the trade-off figure)."""
    curve = []
    for thr in np.linspace(0.02, 0.98, 49):
        m = presence(table, score, thr)
        curve.append({"threshold": float(thr), "false_alarm": m["false_alarm"], "miss": m["miss"]})
    return curve


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------

def counting(table: pd.DataFrame, pred: np.ndarray, smoothed: bool = True) -> dict:
    """Exact accuracy, MAE, per-class precision/recall, over/under counting (valid windows)."""
    valid = table["valid"].to_numpy()
    t = table[valid]
    p = np.asarray(pred, dtype=float)[valid]
    if smoothed:
        p = smooth(t, p)
    p = np.clip(np.round(p), 0, 5).astype(int)
    y = t["count"].to_numpy().astype(int)

    prec, rec, f1, support = precision_recall_fscore_support(y, p, labels=COUNT_CLASSES, zero_division=0)
    present = [c for c in COUNT_CLASSES if (y == c).any()]
    return {
        "n": int(len(y)),
        "rejected_rate": float(1 - valid.mean()) if len(table) else None,
        "smoothed": smoothed,
        "exact_accuracy": float((y == p).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)) if len(present) > 1 else None,
        "mae": float(np.abs(y - p).mean()),
        "within_one": float((np.abs(y - p) <= 1).mean()),
        "overcount_rate": float((p > y).mean()),
        "undercount_rate": float((p < y).mean()),
        "empty_false_alarm": float((p[y == 0] > 0).mean()) if (y == 0).any() else None,
        "per_class": {str(c): {"precision": float(prec[i]), "recall": float(rec[i]), "support": int(support[i]),
                               "has_training_data": c <= 3}
                      for i, c in enumerate(COUNT_CLASSES)},
        "confusion": confusion_matrix(y, p, labels=COUNT_CLASSES).tolist(),
    }


def per_recording_counting(table: pd.DataFrame, pred: np.ndarray) -> list[dict]:
    valid = table["valid"].to_numpy()
    t = table[valid]
    p = np.clip(np.round(smooth(t, np.asarray(pred, dtype=float)[valid])), 0, 5).astype(int)
    rows = []
    for rec, idx in t.groupby("recording").indices.items():
        y = int(t["count"].iloc[idx[0]])
        rows.append({"recording": rec, "count": y, "n": int(len(idx)),
                     "exact_accuracy": float((p[idx] == y).mean()), "mean_predicted": float(p[idx].mean())})
    return rows

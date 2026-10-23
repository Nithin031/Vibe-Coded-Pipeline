"""Metrics: from window estimates to the shown count, and scores against a known count.

Output rule (same as the device): clip each window estimate to the trained range, take the median of the last
30 s (12 windows), round. Show it honestly as empty / 1 / 2 / 3+ (estimates bunch together above 2-3 people).
"""

import numpy as np

from data import SMOOTH_N


def smooth_predictions(pred):
    """Causal rolling median over the last SMOOTH_N window estimates."""
    pred = np.asarray(pred, dtype=np.float64)
    out = np.empty_like(pred)
    for i in range(len(pred)):
        out[i] = np.median(pred[max(0, i - SMOOTH_N + 1):i + 1])
    return out


def to_count(pred, max_count):
    return np.clip(np.rint(pred), 0, max_count).astype(int)


def honest_label(count):
    """The answer to show: 'empty', '1', '2' or '3+'."""
    count = int(count)
    return "empty" if count == 0 else ("3+" if count >= 3 else str(count))


def score(y_true, counts):
    """Scores of shown counts against the true count (one number or one per window)."""
    counts = np.asarray(counts, dtype=int)
    y = np.broadcast_to(np.asarray(y_true, dtype=int), counts.shape)
    out = {"windows": int(len(counts)),
           "mae": float(np.mean(np.abs(counts - y))) if len(counts) else np.nan,
           "exact": float(np.mean(counts == y)) if len(counts) else np.nan,
           "within1": float(np.mean(np.abs(counts - y) <= 1)) if len(counts) else np.nan,
           "exact_honest": float(np.mean(np.minimum(counts, 3) == np.minimum(y, 3))) if len(counts) else np.nan}
    for c in range(4):
        m = (np.minimum(y, 3) == c)
        out[f"recall_{c}"] = float(np.mean(np.minimum(counts[m], 3) == c)) if m.any() else np.nan
    m0 = y == 0
    out["empty_false_alarm"] = float(np.mean(counts[m0] > 0)) if m0.any() else np.nan
    return out

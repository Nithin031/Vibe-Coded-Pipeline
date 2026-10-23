"""ESP32-S3 people counter - reference implementation of the final model.

Everything the firmware has to do, written in Python, packet by packet:

    raw CSI amplitude (52 tones)
      -> per-packet gain normalisation (divide by the median over tones)
      -> 50 Hz bins (packets averaged per 20 ms), gaps up to 60 ms interpolated
      -> causal baseline per tone (EMA, 30 s) -> relative fluctuation z = x / baseline - 1
      -> 5 s windows every 2.5 s (first 10 s skipped while the baseline settles)
      -> 10 count features
      -> log of the magnitude features, minus the empty-room offset (if calibrated)
      -> tree model (LightGBM regression, exported to plain arrays / C header)
      -> clip to the trained count range -> median of the last 30 s -> people count

The batch functions (used for training) and StreamingCounter (what the device runs)
share the same code, and test_esp32_counter.py checks that they give the same numbers.
Nothing here looks at future packets.
"""

import json
import math
import os
import time
from collections import deque

import numpy as np
from scipy.signal import lfilter


# ===============================================================
# Settings (identical on the ESP32)
# ===============================================================
FS = 50                  # sample rate of the shared grid (Hz)
WIN_S = 5.0              # window length (s)
HOP_S = 2.5              # window hop (s)
WIN = int(WIN_S * FS)    # 250 bins
HOP = int(HOP_S * FS)    # 125 bins
N_TONES = 52

MAX_GAP_BINS = 3         # gaps up to 3 bins (60 ms) are interpolated
MAX_MISSING = 0.20       # windows with more missing bins than this are skipped

TAU_S = 30.0             # time constant of the causal per-tone baseline
ALPHA = 1.0 / (TAU_S * FS)
WARMUP_S = 10.0          # no output for the first 10 s (baseline settling)
WARMUP_BINS = int(WARMUP_S * FS)

CAL_S = 20.0             # empty-room calibration after the warm-up
N_CAL_WINDOWS = int((CAL_S - WIN_S) / HOP_S) + 1     # 7 windows

SMOOTH_S = 30.0          # output = median of the last 30 s of window estimates
SMOOTH_N = int(SMOOTH_S / HOP_S)                    # 12 windows

EPS = 1e-6

FEATURE_NAMES = [
    "cv_median",       # median over tones of the fluctuation std
    "cv_p90",          # 90th percentile of the same
    "cv_p10",          # 10th percentile of the same
    "diff_energy",     # mean |x(t) - x(t-1)|  (motion energy)
    "pc1_frac",        # share of variance in the first principal component
    "n_pc_90",         # components needed for 90 % of the variance
    "tone_corr",       # mean correlation between tones
    "low_freq_frac",   # temporal power below 2 Hz (slow body motion)
    "high_freq_frac",  # temporal power above 5 Hz
    "mean_std",        # std of the tone-averaged series
]

# magnitude features go through log() so the empty-room offset acts as a ratio
LOG_FEATURES = ["cv_median", "cv_p90", "cv_p10", "diff_energy", "mean_std"]
LOG_IDX = [FEATURE_NAMES.index(name) for name in LOG_FEATURES]


def settings_dict():
    return {"fs_hz": FS, "win_s": WIN_S, "hop_s": HOP_S, "n_tones": N_TONES,
            "max_gap_bins": MAX_GAP_BINS, "max_missing": MAX_MISSING,
            "tau_s": TAU_S, "warmup_s": WARMUP_S, "cal_s": CAL_S,
            "n_cal_windows": N_CAL_WINDOWS, "smooth_s": SMOOTH_S, "smooth_n": SMOOTH_N,
            "feature_names": FEATURE_NAMES, "log_features": LOG_FEATURES}


# ===============================================================
# Batch preprocessing (training)
# ===============================================================
def gain_normalise(A):
    """Per packet: divide by the median over tones (removes AGC jumps)."""
    return A / (np.median(A, axis=1, keepdims=True) + EPS)


def resample_to_grid(t, A):
    """t: packet times (s, sorted), A: (packets, tones).
    Returns X (bins, tones) with NaN rows for empty bins, and the valid mask."""
    idx = np.floor((t - t[0]) * FS).astype(np.int64)
    n_bins = int(idx[-1]) + 1

    uniq, first = np.unique(idx, return_index=True)        # idx is sorted
    sums = np.add.reduceat(A.astype(np.float64), first, axis=0)
    counts = np.diff(np.append(first, len(idx)))

    X = np.full((n_bins, A.shape[1]), np.nan, dtype=np.float32)
    X[uniq] = sums / counts[:, None]
    valid = np.zeros(n_bins, dtype=bool)
    valid[uniq] = True
    return X, valid


def fill_short_gaps(X, valid):
    """Linear interpolation over gaps of at most MAX_GAP_BINS; longer gaps stay NaN."""
    X, valid = X.copy(), valid.copy()
    missing = np.flatnonzero(~valid)
    if len(missing) == 0:
        return X, valid

    runs = np.split(missing, np.flatnonzero(np.diff(missing) > 1) + 1)
    for run in runs:
        left, right = run[0] - 1, run[-1] + 1
        if len(run) <= MAX_GAP_BINS and left >= 0 and right < len(X):
            w = ((run - left) / (right - left))[:, None]
            X[run] = X[left] * (1 - w) + X[right] * w
            valid[run] = True
    return X, valid


def causal_normalise(X, valid):
    """Each tone relative to its own past (EMA baseline): z = x / baseline - 1.

    The baseline only moves on valid bins and starts at the first valid value.
    Because it is linear, any constant per-tone scale of X cancels out, so the
    Phase 1 series (already divided by the recording mean) give the same z."""
    Z = np.full(X.shape, np.nan, dtype=np.float32)
    rows = np.flatnonzero(valid)
    if len(rows) == 0:
        return Z

    xs = X[rows].astype(np.float64)
    zi = (1.0 - ALPHA) * xs[:1]                             # first output = first value
    baseline, _ = lfilter([ALPHA], [1.0, -(1.0 - ALPHA)], xs, axis=0, zi=zi)

    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(baseline > EPS, xs / baseline - 1.0, 0.0)
    Z[rows] = z
    return Z


def window_starts(valid):
    """Window start bins: every HOP, after the warm-up, with enough valid bins."""
    n = len(valid)
    return [a for a in range(0, n - WIN + 1, HOP)
            if a >= WARMUP_BINS and valid[a:a + WIN].mean() >= 1 - MAX_MISSING]


# ===============================================================
# Count features (same code as Phase 1)
# ===============================================================
def window_features(W):
    """W: (bins, 52) normalised window. Missing bins count as 'no change' (0)."""
    W = np.nan_to_num(W.astype(np.float32))
    sd = W.std(axis=0)
    diff_energy = np.abs(np.diff(W, axis=0)).mean()

    Xc = W - W.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False) ** 2
    frac = s / (s.sum() + 1e-12)
    n90 = np.searchsorted(np.cumsum(frac), 0.90) + 1

    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(Xc.T)
    tone_corr = np.nanmean(C[np.triu_indices_from(C, k=1)])

    x = Xc.mean(axis=1)
    P = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / FS)
    p_total = P[f > 0].sum() + 1e-12
    low = P[(f > 0) & (f < 2.0)].sum() / p_total
    high = P[f > 5.0].sum() / p_total

    return np.array([np.median(sd), np.percentile(sd, 90), np.percentile(sd, 10),
                     diff_energy, frac[0], n90, tone_corr, low, high, x.std()],
                    dtype=np.float32)


def features_from_series(X, valid):
    """X: (bins, 52) amplitude on the 50 Hz grid (any per-tone scale).
    Returns window start bins and features (windows, 10)."""
    Z = causal_normalise(X, valid)
    starts = window_starts(valid)
    F = np.zeros((len(starts), len(FEATURE_NAMES)), dtype=np.float32)
    for j, a in enumerate(starts):
        F[j] = window_features(Z[a:a + WIN])
    return np.array(starts, dtype=np.int32), F


def features_from_raw(t, A):
    """Raw packets (times s, amplitude (packets, 52)) -> window starts + features."""
    order = np.argsort(t, kind="stable")
    t, A = np.asarray(t, dtype=np.float64)[order], np.asarray(A)[order]
    X, valid = resample_to_grid(t, gain_normalise(A.astype(np.float32)))
    X, valid = fill_short_gaps(X, valid)
    return features_from_series(X, valid)


# ===============================================================
# Model input, calibration, smoothing
# ===============================================================
def prepare_features(F, offset=None):
    """Raw features -> model input: NaN -> 0, log of magnitude features, minus offset."""
    T = np.nan_to_num(np.asarray(F, dtype=np.float64), nan=0.0)
    T = np.atleast_2d(T).copy()
    T[:, LOG_IDX] = np.log(np.maximum(T[:, LOG_IDX], 0.0) + EPS)
    if offset is not None:
        T = T - np.asarray(offset, dtype=np.float64)
    return T.astype(np.float32)


def calibration_offset(F_cal):
    """Empty-room offset = mean model input over the calibration windows."""
    return prepare_features(F_cal).astype(np.float64).mean(axis=0)


def smooth_predictions(pred):
    """Causal rolling median over the last SMOOTH_N window estimates."""
    pred = np.asarray(pred, dtype=np.float64)
    out = np.empty_like(pred)
    for i in range(len(pred)):
        out[i] = np.median(pred[max(0, i - SMOOTH_N + 1):i + 1])
    return out


def to_count(pred, max_count):
    return np.clip(np.rint(pred), 0, max_count).astype(int)


# ===============================================================
# Tree model: LightGBM -> flat arrays (Python, JSON, C header)
# ===============================================================
def _float32_at_most(value):
    """Largest float32 <= value, so 'x <= thr' in float32 equals 'x <= value' in double."""
    thr = np.float32(value)
    if float(thr) > value:
        thr = np.nextafter(thr, np.float32(-np.inf))
    return float(thr)


class TreeModel:
    """Sum of regression trees stored as flat node arrays.

    Node i: feature[i] = -1 for a leaf (value[i] = leaf output), otherwise go to
    left[i] if x[feature[i]] <= value[i] (threshold) else right[i]."""

    def __init__(self, roots, feature, value, left, right, max_count, n_features):
        self.roots = np.asarray(roots, dtype=np.int32)
        self.feature = np.asarray(feature, dtype=np.int32)
        self.value = np.asarray(value, dtype=np.float32)
        self.left = np.asarray(left, dtype=np.int32)
        self.right = np.asarray(right, dtype=np.int32)
        self.max_count = int(max_count)
        self.n_features = int(n_features)

    # ---------------------------------------------------------------
    # Build / save / load
    # ---------------------------------------------------------------
    @classmethod
    def from_lightgbm(cls, booster, max_count):
        dump = booster.dump_model()
        feature, value, left, right, roots = [], [], [], [], []

        def add(node):
            i = len(feature)
            feature.append(-1)
            value.append(0.0)
            left.append(-1)
            right.append(-1)

            if "leaf_value" in node:
                value[i] = float(np.float32(node["leaf_value"]))
                return i

            if node["decision_type"] != "<=":
                raise ValueError(f"unsupported split type {node['decision_type']}")
            if node.get("missing_type", "None") not in ("None", "NaN"):
                raise ValueError(f"unsupported missing type {node['missing_type']}")

            feature[i] = int(node["split_feature"])
            value[i] = _float32_at_most(float(node["threshold"]))
            left[i] = add(node["left_child"])
            right[i] = add(node["right_child"])
            return i

        for tree in dump["tree_info"]:
            roots.append(add(tree["tree_structure"]))

        return cls(roots, feature, value, left, right, max_count,
                   n_features=dump["max_feature_idx"] + 1)

    def to_dict(self):
        return {"roots": self.roots.tolist(), "feature": self.feature.tolist(),
                "value": [float(v) for v in self.value], "left": self.left.tolist(),
                "right": self.right.tolist(), "max_count": self.max_count,
                "n_features": self.n_features}

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["roots"], d["feature"], d["value"], d["left"], d["right"],
                   d["max_count"], d["n_features"])

    # ---------------------------------------------------------------
    # Inference (float32, same order of operations as the C code)
    # ---------------------------------------------------------------
    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        if np.isnan(X).any():
            raise ValueError("model input contains NaN - use prepare_features()")

        rows = np.arange(len(X))
        total = np.zeros(len(X), dtype=np.float32)
        for root in self.roots:
            node = np.full(len(X), root, dtype=np.int32)
            while True:
                f = self.feature[node]
                split = f >= 0
                if not split.any():
                    break
                go_left = X[rows, np.maximum(f, 0)] <= self.value[node]
                node = np.where(split, np.where(go_left, self.left[node], self.right[node]), node)
            total += self.value[node]
        return total

    @property
    def n_nodes(self):
        return len(self.feature)

    # ---------------------------------------------------------------
    # C header for the ESP32 firmware
    # ---------------------------------------------------------------
    def write_c_header(self, path, note=""):
        idx_type = "int16_t" if self.n_nodes < 32767 else "int32_t"

        def array(ctype, name, values, fmt):
            body = ", ".join(fmt(v) for v in values)
            lines, line = [], "    "
            for item in body.split(", "):
                if len(line) + len(item) > 96:
                    lines.append(line.rstrip())
                    line = "    "
                line += item + ", "
            lines.append(line.rstrip().rstrip(","))
            return f"static const {ctype} {name}[{len(values)}] = {{\n" + "\n".join(lines) + "\n};\n"

        log_flags = [1 if name in LOG_FEATURES else 0 for name in FEATURE_NAMES]
        float_fmt = lambda v: f"{float(v):.9g}f"
        int_fmt = lambda v: str(int(v))

        text = [
            "/* ESP32-S3 people counter - generated by esp32_counter.py. Do not edit. */",
            f"/* {note} */" if note else "",
            "#ifndef CC_COUNTER_MODEL_H",
            "#define CC_COUNTER_MODEL_H",
            "",
            "#include <math.h>",
            "#include <stdint.h>",
            "",
            "/* ---- preprocessing settings (must match esp32_counter.py) ---- */",
            f"#define CC_FS_HZ          {FS}",
            f"#define CC_N_TONES        {N_TONES}",
            f"#define CC_WIN_BINS       {WIN}",
            f"#define CC_HOP_BINS       {HOP}",
            f"#define CC_MAX_GAP_BINS   {MAX_GAP_BINS}",
            f"#define CC_MAX_MISSING    {MAX_MISSING}f",
            f"#define CC_EMA_ALPHA      {ALPHA:.9g}f",
            f"#define CC_WARMUP_BINS    {WARMUP_BINS}",
            f"#define CC_N_CAL_WINDOWS  {N_CAL_WINDOWS}",
            f"#define CC_SMOOTH_N       {SMOOTH_N}",
            "",
            "/* ---- model ---- */",
            f"#define CC_N_FEATURES     {self.n_features}",
            f"#define CC_N_TREES        {len(self.roots)}",
            f"#define CC_N_NODES        {self.n_nodes}",
            f"#define CC_MAX_COUNT      {self.max_count}",
            "",
            "/* features, in order: " + ", ".join(FEATURE_NAMES) + " */",
            array("uint8_t", "cc_log_feature", log_flags, int_fmt),
            array(idx_type, "cc_roots", self.roots, int_fmt),
            array("int8_t", "cc_feature", self.feature, int_fmt),
            array("float", "cc_value", self.value, float_fmt),
            array(idx_type, "cc_left", self.left, int_fmt),
            array(idx_type, "cc_right", self.right, int_fmt),
            "/* raw window features -> model input. offset = NULL when not calibrated. */",
            "static inline void cc_prepare(const float *raw, const float *offset, float *out)",
            "{",
            "    for (int i = 0; i < CC_N_FEATURES; i++) {",
            "        float v = isnan(raw[i]) ? 0.0f : raw[i];",
            "        if (cc_log_feature[i]) {",
            "            v = logf(fmaxf(v, 0.0f) + 1e-6f);",
            "        }",
            "        if (offset) {",
            "            v -= offset[i];",
            "        }",
            "        out[i] = v;",
            "    }",
            "}",
            "",
            "/* model input -> people estimate (not yet clipped / smoothed) */",
            "static inline float cc_predict(const float *x)",
            "{",
            "    float total = 0.0f;",
            "    for (int t = 0; t < CC_N_TREES; t++) {",
            "        int n = cc_roots[t];",
            "        while (cc_feature[n] >= 0) {",
            "            n = (x[cc_feature[n]] <= cc_value[n]) ? cc_left[n] : cc_right[n];",
            "        }",
            "        total += cc_value[n];",
            "    }",
            "    return total;",
            "}",
            "",
            "#endif /* CC_COUNTER_MODEL_H */",
            "",
        ]
        with open(path, "w") as f:
            f.write("\n".join(text))
        return os.path.getsize(path)

    def flash_bytes(self):
        """Approximate flash used by the node arrays in the C header."""
        idx_bytes = 2 if self.n_nodes < 32767 else 4
        per_node = 1 + 4 + 2 * idx_bytes                    # feature + value + left + right
        return self.n_nodes * per_node + len(self.roots) * idx_bytes


# ===============================================================
# Streaming counter (what the ESP32 runs)
# ===============================================================
class StreamingCounter:
    """Feed packets one by one with push(t_seconds, amplitude_52).

    Every 2.5 s (after the warm-up) push() returns a result dict with the window
    features and, if a model is loaded, the raw / smoothed estimate and the count.
    Packets must arrive in time order; late packets are dropped."""

    def __init__(self, model=None, offset=None):
        self.model = model
        self.offset = None if offset is None else np.asarray(offset, dtype=np.float64)

        # 50 Hz binning
        self.t0 = None
        self.cur_bin = -1
        self.acc_sum = None
        self.acc_n = 0
        self.last_valid_bin = -1
        self.last_valid_x = None

        # causal baseline + window buffer
        self.baseline = None
        self.z_buf = deque(maxlen=WIN)
        self.v_buf = deque(maxlen=WIN)
        self.n_bins = 0

        # output smoothing + calibration
        self.recent = deque(maxlen=SMOOTH_N)
        self.cal_feats = None
        self.cal_target = 0

    # ---------------------------------------------------------------
    # Calibration: leave the room empty, call begin_calibration(), keep pushing
    # ---------------------------------------------------------------
    def begin_calibration(self, n_windows=N_CAL_WINDOWS):
        self.cal_feats = []
        self.cal_target = n_windows

    @property
    def calibrating(self):
        return self.cal_feats is not None

    # ---------------------------------------------------------------
    # Packet input
    # ---------------------------------------------------------------
    def push(self, t, amplitude):
        x = gain_normalise(np.asarray(amplitude, dtype=np.float32)[None])[0]
        if self.t0 is None:
            self.t0 = float(t)
        b = int(math.floor((float(t) - self.t0) * FS))

        results = []
        if b < self.cur_bin:
            return results                                  # late packet

        if b > self.cur_bin:
            if self.acc_n:
                results += self._close_bin(self.cur_bin, self.acc_sum / self.acc_n)
            self.cur_bin = b
            self.acc_sum = np.zeros(N_TONES, dtype=np.float64)
            self.acc_n = 0

        self.acc_sum += x
        self.acc_n += 1
        return results

    def flush(self):
        """Close the last open bin (end of a recording)."""
        results = []
        if self.acc_n:
            results = self._close_bin(self.cur_bin, self.acc_sum / self.acc_n)
            self.acc_n = 0
        return results

    # ---------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------
    def _close_bin(self, k, mean):
        x = mean.astype(np.float32)
        out = []

        if self.last_valid_bin >= 0:
            gap = k - self.last_valid_bin - 1
            if 0 < gap <= MAX_GAP_BINS:
                left = self.last_valid_bin
                for j in range(left + 1, k):
                    w = (j - left) / (k - left)
                    xi = (self.last_valid_x * (1 - w) + x * w).astype(np.float32)
                    out += self._add_bin(xi, True)
            else:
                for _ in range(gap):
                    out += self._add_bin(None, False)

        self.last_valid_bin, self.last_valid_x = k, x
        out += self._add_bin(x, True)
        return out

    def _add_bin(self, x, valid):
        if valid:
            xd = x.astype(np.float64)
            if self.baseline is None:
                self.baseline = xd.copy()
            else:
                self.baseline = self.baseline + ALPHA * (xd - self.baseline)
            with np.errstate(invalid="ignore", divide="ignore"):
                z = np.where(self.baseline > EPS, xd / self.baseline - 1.0, 0.0)
            z = z.astype(np.float32)
        else:
            z = np.full(N_TONES, np.nan, dtype=np.float32)

        self.z_buf.append(z)
        self.v_buf.append(valid)
        self.n_bins += 1

        a = self.n_bins - WIN
        if a < 0 or a % HOP or a < WARMUP_BINS:
            return []
        if np.mean(self.v_buf) < 1 - MAX_MISSING:
            return []

        feats = window_features(np.array(self.z_buf))
        return [self._decide(a, feats)]

    def _decide(self, start_bin, feats):
        res = {"t_start_s": start_bin / FS, "t_end_s": (start_bin + WIN) / FS,
               "features": feats}

        if self.calibrating:
            self.cal_feats.append(feats)
            res["status"] = "calibrating"
            if len(self.cal_feats) >= self.cal_target:
                self.offset = calibration_offset(np.array(self.cal_feats))
                self.cal_feats = None
                self.recent.clear()
                res["status"] = "calibrated"
            return res

        if self.model is None:
            return res

        x = prepare_features(feats[None], self.offset)
        raw = float(np.clip(self.model.predict(x)[0], 0, self.model.max_count))
        self.recent.append(raw)
        smoothed = float(np.median(self.recent))
        res.update(status="ok", raw=raw, smoothed=smoothed,
                   count=int(to_count(smoothed, self.model.max_count)))
        return res


# ===============================================================
# Model bundle (what goes to the device / the friend's demo)
# ===============================================================
def save_bundle(model_dir, tree_model, card, booster=None):
    """Writes trees.json, model_card.json, counter_model.h (+ model.txt if booster)."""
    os.makedirs(model_dir, exist_ok=True)
    tree_model.save(os.path.join(model_dir, "trees.json"))

    note = f"{card.get('name', 'counter')} | calibration={card.get('calibration')} | " \
           f"counts 0-{tree_model.max_count} | {time.strftime('%Y-%m-%d %H:%M')}"
    header_bytes = tree_model.write_c_header(os.path.join(model_dir, "counter_model.h"), note)

    if booster is not None:
        booster.save_model(os.path.join(model_dir, "model.txt"))

    card = {**card, "settings": settings_dict(), "n_trees": len(tree_model.roots),
            "n_nodes": tree_model.n_nodes, "flash_bytes_est": tree_model.flash_bytes(),
            "header_file_bytes": header_bytes}
    with open(os.path.join(model_dir, "model_card.json"), "w") as f:
        json.dump(card, f, indent=1)
    return card


def load_bundle(model_dir):
    """Returns (TreeModel, card). Use with StreamingCounter(model)."""
    model = TreeModel.load(os.path.join(model_dir, "trees.json"))
    with open(os.path.join(model_dir, "model_card.json")) as f:
        card = json.load(f)
    if card["settings"] != settings_dict():
        raise ValueError("model_card settings differ from this esp32_counter.py")
    return model, card

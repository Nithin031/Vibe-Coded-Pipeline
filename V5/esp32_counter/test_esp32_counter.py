"""Tests for esp32_counter.py.

Run:  py -3 -m unittest D:/wifi/esp32_counter/test_esp32_counter.py -v
(the Phase 3 notebook runs them first and stops if any fails)
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import esp32_counter as ec  # noqa: E402

SHARED = os.path.join(os.path.dirname(HERE), "shared_format")


# ===============================================================
# Synthetic ESP32-like packets
# ===============================================================
def synthetic_packets(seconds=60.0, rate=100.0, seed=0, n_people=2):
    """Jittered packet times with one short gap (interpolated) and one long gap
    (missing), slow per-tone fading, motion ripples and AGC gain jumps."""
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    t = np.cumsum(rng.uniform(0.5, 1.5, n) / rate)

    keep = ~(((t > 20.00) & (t < 20.03)) | ((t > 35.0) & (t < 35.6)))
    t = t[keep]

    base = rng.uniform(20, 60, ec.N_TONES)
    fade = 1 + 0.05 * np.sin(2 * np.pi * 0.02 * t)[:, None]
    motion = np.zeros((len(t), ec.N_TONES))
    for _ in range(n_people):
        f = rng.uniform(0.3, 2.0)
        phase = rng.uniform(0, 2 * np.pi, ec.N_TONES)
        motion += 0.08 * np.sin(2 * np.pi * f * t[:, None] + phase)
    gain = np.where((t // 7) % 2 == 0, 1.0, 1.35)[:, None]
    noise = 0.02 * rng.standard_normal((len(t), ec.N_TONES))

    A = np.abs(base * fade * (1 + motion + noise) * gain)
    return t, A.astype(np.float32)


def stream(t, A, model=None, offset=None):
    counter = ec.StreamingCounter(model=model, offset=offset)
    out = []
    for ti, ai in zip(t, A):
        out += counter.push(ti, ai)
    out += counter.flush()
    return out


def small_model(seed=0, n_estimators=40):
    import lightgbm as lgb

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(3000, len(ec.FEATURE_NAMES))).astype(np.float32)
    y = np.clip(2 * X[:, 0] - X[:, 3] + 0.5 * X[:, 5] * X[:, 6] + 4 + 0.3 * rng.normal(size=3000), 0, 8)
    reg = lgb.LGBMRegressor(n_estimators=n_estimators, num_leaves=15, learning_rate=0.1,
                            random_state=seed, verbose=-1).fit(X, y)
    return reg, ec.TreeModel.from_lightgbm(reg.booster_, max_count=8), X


# ===============================================================
# Preprocessing
# ===============================================================
class TestPreprocessing(unittest.TestCase):

    def test_streaming_matches_batch(self):
        """The device path (packet by packet) gives the same windows + features as training."""
        t, A = synthetic_packets()
        starts, F = ec.features_from_raw(t, A)
        out = stream(t, A)

        self.assertGreater(len(starts), 5)
        self.assertEqual([int(round(o["t_start_s"] * ec.FS)) for o in out], starts.tolist())
        F_stream = np.array([o["features"] for o in out])
        np.testing.assert_allclose(F_stream, F, rtol=1e-3, atol=1e-5)

    def test_recording_scale_cancels(self):
        """Phase 1 series (divided by the recording mean) give the same causal z as raw."""
        t, A = synthetic_packets(seed=1)
        X, valid = ec.resample_to_grid(t, ec.gain_normalise(A))
        X, valid = ec.fill_short_gaps(X, valid)
        Y = X / np.nanmean(X, axis=0, keepdims=True) - 1.0           # Phase 1 storage

        Z_raw = ec.causal_normalise(X, valid)
        Z_stored = ec.causal_normalise(Y + 1.0, valid)
        np.testing.assert_allclose(Z_stored, Z_raw, atol=1e-5, equal_nan=True)

    def test_no_future_leakage(self):
        """Changing packets after 40 s does not change any output before 40 s."""
        t, A = synthetic_packets(seed=2)
        A2 = A.copy()
        later = t >= 40.0
        A2[later] *= np.random.default_rng(9).uniform(0.5, 2.0, A2[later].shape).astype(np.float32)

        before = [o for o in stream(t, A) if o["t_end_s"] < 39.9]
        before2 = [o for o in stream(t, A2) if o["t_end_s"] < 39.9]
        self.assertGreater(len(before), 3)
        self.assertEqual(len(before), len(before2))
        for o, o2 in zip(before, before2):
            np.testing.assert_array_equal(o["features"], o2["features"])

    def test_warmup_and_missing_windows(self):
        t, A = synthetic_packets(seed=3)
        starts, _ = ec.features_from_raw(t, A)
        self.assertTrue(all(a >= ec.WARMUP_BINS for a in starts))
        # the 0.6 s gap at 35 s is > 60 ms, so it stays missing, but 30 bins < 20 % of a window
        X, valid = ec.resample_to_grid(t, ec.gain_normalise(A))
        X, valid = ec.fill_short_gaps(X, valid)
        self.assertFalse(valid[int(35.2 * ec.FS)])
        self.assertTrue(valid[int(20.02 * ec.FS)])

    def test_features_match_phase1_on_real_data(self):
        """window_features() here == Phase 1 (stored feats were made from the same series)."""
        files = sorted(glob.glob(os.path.join(SHARED, "wicount", "*.npz")))
        if not files:
            self.skipTest("shared_format/wicount not found")
        z = np.load(files[0])
        S, starts, feats = z["series"], z["starts"], z["feats"]
        mine = np.array([ec.window_features(S[0, a:a + ec.WIN]) for a in starts])
        rel = np.abs(mine - feats[0]) / (np.abs(feats[0]) + 1e-6)
        self.assertLess(float(np.median(rel)), 0.02)      # stored series is float16


# ===============================================================
# Model export
# ===============================================================
class TestModel(unittest.TestCase):

    def test_tree_export_matches_lightgbm(self):
        reg, tm, X = small_model()
        np.testing.assert_allclose(tm.predict(X), reg.predict(X), atol=1e-4)

    def test_json_roundtrip(self):
        _, tm, X = small_model(seed=1)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trees.json")
            tm.save(path)
            tm2 = ec.TreeModel.load(path)
        np.testing.assert_array_equal(tm2.predict(X), tm.predict(X))

    def test_bundle_roundtrip(self):
        reg, tm, X = small_model(seed=2)
        with tempfile.TemporaryDirectory() as d:
            card = ec.save_bundle(d, tm, {"name": "test", "calibration": "zero_shot"}, reg.booster_)
            for name in ["trees.json", "model_card.json", "counter_model.h", "model.txt"]:
                self.assertTrue(os.path.exists(os.path.join(d, name)), name)
            tm2, card2 = ec.load_bundle(d)
        self.assertEqual(card2["n_nodes"], tm.n_nodes)
        np.testing.assert_array_equal(tm2.predict(X), tm.predict(X))

    def test_c_header_matches_python(self):
        """Compile the generated header with gcc/clang (skipped if no C compiler)."""
        cc = shutil.which("gcc") or shutil.which("clang") or shutil.which("cc")
        if cc is None:
            self.skipTest("no C compiler on PATH")

        _, tm, X = small_model(seed=3)
        X = X[:200]
        with tempfile.TemporaryDirectory() as d:
            tm.write_c_header(os.path.join(d, "counter_model.h"))
            rows = ",\n".join("{" + ", ".join(f"{v:.9g}f" for v in row) + "}" for row in X)
            main = (
                '#include <stdio.h>\n#include "counter_model.h"\n'
                f"static const float xs[{len(X)}][CC_N_FEATURES] = {{\n{rows}\n}};\n"
                "int main(void) {\n"
                f"    for (int i = 0; i < {len(X)}; i++) printf(\"%.6f\\n\", cc_predict(xs[i]));\n"
                "    return 0;\n}\n"
            )
            with open(os.path.join(d, "main.c"), "w") as f:
                f.write(main)
            exe = os.path.join(d, "test_model.exe")
            subprocess.run([cc, "-O2", "-o", exe, os.path.join(d, "main.c"), "-lm"],
                           check=True, cwd=d)
            out = subprocess.run([exe], check=True, capture_output=True, text=True).stdout
        c_pred = np.array([float(v) for v in out.split()])
        np.testing.assert_allclose(c_pred, tm.predict(X), atol=1e-4)

    def test_flash_budget(self):
        _, tm, _ = small_model(seed=4, n_estimators=300)
        self.assertLess(tm.flash_bytes(), 512 * 1024)


# ===============================================================
# Calibration + full streaming counter
# ===============================================================
class TestCounter(unittest.TestCase):

    def test_calibration_offset_centres_features(self):
        t, A = synthetic_packets(seed=5)
        _, F = ec.features_from_raw(t, A)
        offset = ec.calibration_offset(F)
        np.testing.assert_allclose(ec.prepare_features(F, offset).mean(axis=0), 0.0, atol=1e-4)

    def test_prepare_handles_nan(self):
        F = np.full((2, len(ec.FEATURE_NAMES)), np.nan, dtype=np.float32)
        self.assertFalse(np.isnan(ec.prepare_features(F)).any())

    def test_streaming_counter_end_to_end(self):
        _, tm, _ = small_model(seed=6)
        t, A = synthetic_packets(seed=6, seconds=90)

        counter = ec.StreamingCounter(model=tm)
        counter.begin_calibration()
        out = []
        for ti, ai in zip(t, A):
            out += counter.push(ti, ai)

        status = [o["status"] for o in out]
        self.assertEqual(status[:ec.N_CAL_WINDOWS].count("calibrating"), ec.N_CAL_WINDOWS - 1)
        self.assertEqual(status[ec.N_CAL_WINDOWS - 1], "calibrated")
        self.assertIsNotNone(counter.offset)

        counts = [o["count"] for o in out if o["status"] == "ok"]
        self.assertGreater(len(counts), 3)
        self.assertTrue(all(0 <= c <= tm.max_count for c in counts))

    def test_smoothing_is_causal_median(self):
        p = np.arange(20, dtype=float)
        s = ec.smooth_predictions(p)
        self.assertEqual(s[0], 0.0)
        self.assertEqual(s[19], np.median(p[19 - ec.SMOOTH_N + 1:20]))


if __name__ == "__main__":
    unittest.main(verbosity=2)

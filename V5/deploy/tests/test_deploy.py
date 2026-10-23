"""Tests for the deploy folder: deploy/src must give exactly what V5/esp32_counter (the training reference) gives,
the shipped model files must agree with each other, and infer.py must run end to end.

Run from the V5 folder:  python -m unittest deploy/tests/test_deploy.py -v
(set ESP32_COUNTER_DIR if esp32_counter/ is not next to deploy/)
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

DEPLOY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_DIR = os.environ.get("ESP32_COUNTER_DIR", os.path.join(os.path.dirname(DEPLOY), "esp32_counter"))
sys.path.insert(0, os.path.join(DEPLOY, "src"))
sys.path.insert(0, REF_DIR)

import calibration  # noqa: E402
import data  # noqa: E402
import esp32_counter as ec  # noqa: E402
import metrics  # noqa: E402
import models  # noqa: E402
from test_esp32_counter import small_model, synthetic_packets  # noqa: E402


def write_esp_csv(path, t, A, n_values=104, seed=0, wrap_at=None):
    """Synthetic esp-csi CSV: integer I/Q with random phase; 104 values (52 tones) or 128 (64 subcarriers)."""
    rng = np.random.default_rng(seed)
    ts = np.round(t * 1e6).astype(np.int64) + 1_000_000
    if wrap_at is not None:
        ts = ts - ts[0] + (2 ** 32 - wrap_at)               # wraps wrap_at microseconds after the start
        ts %= 2 ** 32
    rows = []
    for stamp, amp in zip(ts, A):
        ph = rng.uniform(0, 2 * np.pi, len(amp))
        iq = np.stack([np.round(amp * np.sin(ph)), np.round(amp * np.cos(ph))], axis=1).astype(int)
        if n_values == 128:
            full = np.zeros((64, 2), dtype=int)
            full[1:27], full[38:64] = iq[:26], iq[26:]
            iq = full
        rows.append({"type": "CSI_DATA", "local_timestamp": int(stamp), "len": n_values,
                     "data": json.dumps(iq.reshape(-1).tolist())})
    pd.DataFrame(rows).to_csv(path, index=False)


class TestSameAsReference(unittest.TestCase):

    def test_settings_and_features_identical(self):
        self.assertEqual(data.settings_dict(), ec.settings_dict())
        t, A = synthetic_packets(seed=11)
        s1, F1 = data.features_from_raw(t, A)
        s2, F2 = ec.features_from_raw(t, A)
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(F1, F2)

    def test_calibration_identical(self):
        t, A = synthetic_packets(seed=12)
        _, F = data.features_from_raw(t, A)
        off, n = calibration.offset_from_recording(F)
        np.testing.assert_array_equal(off, ec.calibration_offset(F[:n]))
        np.testing.assert_array_equal(calibration.prepare_features(F, off), ec.prepare_features(F, off))

    def test_streaming_counter_identical(self):
        _, tm, _ = small_model(seed=13)
        t, A = synthetic_packets(seed=13, seconds=90)
        mine = models.StreamingCounter(models.TreeModel(**_tree_args(tm)))
        ref = ec.StreamingCounter(tm)
        mine.begin_calibration()
        ref.begin_calibration()
        out1, out2 = [], []
        for ti, ai in zip(t, A):
            out1 += mine.push(ti, ai)
            out2 += ref.push(ti, ai)
        self.assertEqual(len(out1), len(out2))
        for a, b in zip(out1, out2):
            self.assertEqual(a["status"], b["status"])
            np.testing.assert_array_equal(a["features"], b["features"])
            self.assertEqual(a.get("count"), b.get("count"))
            self.assertEqual(a.get("smoothed"), b.get("smoothed"))

    def test_smoothing_identical(self):
        p = np.random.default_rng(0).uniform(0, 5, 50)
        np.testing.assert_array_equal(metrics.smooth_predictions(p), ec.smooth_predictions(p))


def _tree_args(tm):
    return dict(roots=tm.roots, feature=tm.feature, value=tm.value, left=tm.left, right=tm.right,
                max_count=tm.max_count, n_features=tm.n_features)


class TestShippedModel(unittest.TestCase):

    def setUp(self):
        if not os.path.exists(os.path.join(DEPLOY, "trees.json")):
            self.skipTest("no model files in deploy/")

    def test_bundle_loads_and_matches_lightgbm(self):
        model, card = models.load_bundle(DEPLOY)
        self.assertEqual(model.n_nodes, card["n_nodes"])
        X = np.random.default_rng(1).normal(size=(300, model.n_features)).astype(np.float32)
        np.testing.assert_array_equal(model.predict(X), ec.TreeModel.load(os.path.join(DEPLOY, "trees.json")).predict(X))
        try:
            import lightgbm as lgb
        except ImportError:
            return
        if not os.path.exists(os.path.join(DEPLOY, "model.txt")):
            return
        booster = lgb.Booster(model_file=os.path.join(DEPLOY, "model.txt"))
        np.testing.assert_allclose(model.predict(X), booster.predict(X), atol=1e-4)

    def test_c_header_matches_trees(self):
        model, _ = models.load_bundle(DEPLOY)
        with tempfile.TemporaryDirectory() as d:
            model.write_c_header(os.path.join(d, "h.h"))
            with open(os.path.join(d, "h.h")) as f:
                regenerated = [line for line in f.read().splitlines() if line and not line.startswith("/*")]
        with open(os.path.join(DEPLOY, "counter_model.h")) as f:
            shipped = [line for line in f.read().splitlines() if line and not line.startswith("/*")]
        self.assertEqual(regenerated, shipped)


class TestReadAndInfer(unittest.TestCase):

    def test_read_104_and_128_values_and_wrap(self):
        t, A = synthetic_packets(seed=14, seconds=30)
        with tempfile.TemporaryDirectory() as d:
            p104, p128, pwrap = (os.path.join(d, n) for n in ["a.csv", "b.csv", "c.csv"])
            write_esp_csv(p104, t, A, 104, seed=1)
            write_esp_csv(p128, t, A, 128, seed=1)
            write_esp_csv(pwrap, t, A, 104, seed=1, wrap_at=5_000_000)
            t1, A1 = data.read_esp_csv(p104)
            t2, A2 = data.read_esp_csv(p128)
            t3, A3 = data.read_esp_csv(pwrap)
        self.assertEqual(A1.shape, (len(t), 52))
        np.testing.assert_array_equal(A1, A2)
        np.testing.assert_allclose(t1, t - t[0], atol=1e-6)
        np.testing.assert_allclose(t3, t1, atol=1e-6)
        np.testing.assert_allclose(A1, A, rtol=0.05, atol=1.0)       # integer I/Q rounding

    def test_infer_cli_end_to_end(self):
        reg, tm, _ = small_model(seed=15)
        t, A = synthetic_packets(seed=15, seconds=90)
        with tempfile.TemporaryDirectory() as d:
            ec.save_bundle(d, tm, {"name": "test", "calibration": "empty_cal"}, reg.booster_)
            rec, out, off = (os.path.join(d, n) for n in ["rec.csv", "out.csv", "off.json"])
            write_esp_csv(rec, t, A, 104)
            cmd = [sys.executable, os.path.join(DEPLOY, "infer.py"), rec, "--model-dir", d]
            r1 = subprocess.run(cmd + ["--calibrate-start", "--save-offset", off, "--out", out, "--count", "2"],
                                capture_output=True, text=True)
            self.assertEqual(r1.returncode, 0, r1.stderr)
            self.assertIn("vs true count 2", r1.stdout)
            table = pd.read_csv(out)
            r2 = subprocess.run(cmd + ["--calibrate-start", "--stream"], capture_output=True, text=True)
            self.assertEqual(r2.returncode, 0, r2.stderr)
            r3 = subprocess.run(cmd + ["--offset", off], capture_output=True, text=True)
            self.assertEqual(r3.returncode, 0, r3.stderr)
        self.assertGreater(len(table), 10)
        self.assertTrue(set(table["label"].astype(str)) <= {"empty", "1", "2", "3+"})


if __name__ == "__main__":
    unittest.main(verbosity=2)

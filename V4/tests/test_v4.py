"""V4 tests: parser, timestamps, gaps, labels, splits, alignment, calibration, models, V3 integrity.

Run:  python -m unittest discover -s V4/tests -v
Tests marked 'cache' read V4/cache/ (built by `python V4/src/data.py --build all`).
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "V4" / "src"))

import calibration                                       # noqa: E402
import data                                              # noqa: E402
import metrics                                           # noqa: E402
import models                                            # noqa: E402
import noise                                             # noqa: E402
import snapshot_repo                                     # noqa: E402
from run_counting import fold_masks                      # noqa: E402

CLOCK = 2 ** 32


def fake_recording(seconds=20.0, rate=100.0, gaps=(), n_sub=64, seed=0, **meta):
    """Synthetic stream; `gaps` = list of (start_s, length_s) with no packets."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / rate)
    for start, length in gaps:
        t = t[(t < start) | (t >= start + length)]
    amp = (10 + rng.normal(0, 0.5, (len(t), n_sub))).astype(np.float32)
    base = {"source": "test", "recording": "rec", "split": "train", "site": "s", "label_raw": "empty",
            "presence": 0, "motion": "empty", "activity": "", "count": -1, "chip": "", "room": ""}
    base.update(meta)
    return data.Recording(base, amp, t, np.full(len(t), -50.0, np.float32))


# ---------------------------------------------------------------------------
# Parser and timestamps
# ---------------------------------------------------------------------------

class TimestampTests(unittest.TestCase):
    def test_out_of_range_removed_before_unwrap(self):
        ts = np.arange(1000, dtype=float) * 5000 + 1e6
        ts[[100, 500]] = 9.19e17                        # the corruption seen in HomeOccupancy
        keep, t, st = data.clean_timestamps(ts)
        self.assertEqual(st["dropped_out_of_range"], 2)
        self.assertAlmostEqual(t[-1] - t[0], 999 * 0.005, places=6)

    def test_signed_int32_clock_is_converted(self):
        ts = (np.arange(100) * 5000 + 2 ** 31 - 200000).astype(float)
        signed = np.where(ts >= 2 ** 31, ts - CLOCK, ts)   # same clock printed as int32
        keep, t, st = data.clean_timestamps(signed)
        self.assertGreater(st["signed_int32_converted"], 0)
        self.assertTrue(keep.all())
        self.assertAlmostEqual(t[-1] - t[0], 99 * 0.005, places=6)

    def test_wrap_and_spike(self):
        ts = (np.arange(400) * 5000 + CLOCK - 1e6) % CLOCK   # wraps once
        ts = ts.astype(float)
        ts[200] = ts[200] + 3e8                              # isolated 300 s spike
        keep, t, st = data.clean_timestamps(ts)
        self.assertEqual(st["dropped_spikes"], 1)
        self.assertEqual(st["wraps"], 1)
        self.assertTrue(np.all(np.diff(t) > 0))

    def test_duplicates_counted(self):
        ts = np.array([1e6, 1e6 + 5000, 1e6 + 5000, 1e6 + 10000])
        _, _, st = data.clean_timestamps(ts)
        self.assertEqual(st["duplicates"], 1)


class ParserTests(unittest.TestCase):
    def test_malformed_rows_rejected(self):
        good = "[" + ",".join(["3"] * 128) + "]"
        lines = ["type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,channel,local_timestamp,sig_len,rx_state,len,first_word,data"]
        for i in range(50):
            lines.append(f'CSI_DATA,{i},m,-50,11,-93,5,44,6,{1000000 + i * 5000},47,0,128,0,"{good}"')
        lines.append('CSI_DATA,99,m,-50,11,-93,5,44,6,1300000,47,0,128,0,"[1,2,x3]"')     # garbled I/Q
        lines.append('CSI_DATA,99,m,-60,11,-93,5,44,6,1300000,47,0,-60,0,"[1,2]"')         # bad len
        lines.append(f'CSI_DATA,99,m,999,11,-93,5,44,6,1300000,47,0,128,0,"{good}"')        # impossible RSSI
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.csv"
            path.write_text("\n".join(lines))
            amp, t, rssi, st = data.read_esp_csv(path, 128)
        self.assertEqual(amp.shape, (50, 64))
        self.assertEqual(st["dropped_bad_fields"] + st["dropped_bad_iq"], 3)


# ---------------------------------------------------------------------------
# Windows, shapes and gaps
# ---------------------------------------------------------------------------

class WindowTests(unittest.TestCase):
    def test_shapes(self):
        rows, seq, gap = data.make_windows(fake_recording())
        self.assertEqual(seq.shape[1:], (32, 60))
        self.assertEqual(gap.shape[1:], (60,))
        self.assertEqual(len(rows), len(seq))

    def test_long_gap_invalidates_windows(self):
        rows, seq, gap = data.make_windows(fake_recording(gaps=[(8.0, 2.0)]))
        t0 = np.array([r["t0"] for r in rows])
        valid = np.array([r["valid"] for r in rows])
        overlaps = (t0 < 10.0) & (t0 + 3.0 > 8.0)
        self.assertFalse(valid[overlaps].any())
        self.assertTrue(valid[~overlaps].all())

    def test_short_gap_kept_and_flagged(self):
        rows, seq, gap = data.make_windows(fake_recording(gaps=[(8.0, 0.2)]))
        k = [i for i, r in enumerate(rows) if r["t0"] == 7.0][0]
        self.assertTrue(rows[k]["valid"])
        self.assertTrue(gap[k].any())                    # interpolated steps are marked

    def test_robust_prep_removes_spikes_and_gain_steps(self):
        rec = fake_recording(seed=3)
        spiky = noise.apply(rec, "impulses", 3, seed=1)
        clean = data.robust_prep(spiky.amp)
        ratio = clean / np.median(clean)
        self.assertLess(ratio.max(), 2.0)


# ---------------------------------------------------------------------------
# Labels and splits
# ---------------------------------------------------------------------------

class LabelSplitTests(unittest.TestCase):
    def test_unknown_label_raises(self):
        with self.assertRaises(KeyError):
            data.map_label("homeoccupancy", "walking")
        with self.assertRaises(KeyError):
            data.map_count("4")                         # no 4-person data exists

    def test_sleep_and_zones_are_not_static_or_moving(self):
        self.assertEqual(data.map_label("homeoccupancy", "sleep")[1], "unknown")
        self.assertEqual(data.map_label("officelocalization", "two")[1], "unknown")

    def test_split_rules(self):
        self.assertEqual(data.assign_split("homeoccupancy", "work_3"), "test_dev")
        self.assertEqual(data.assign_split("officelocalization", "one_2"), "test_external")
        self.assertEqual(data.assign_split("officelocalization", "empty_4"), "test_extra_empty")
        self.assertEqual(data.assign_split("espectre", "x", "selection"), "val")
        self.assertEqual(data.assign_split("pipeline", "room1_noisy_empty_trail"), "test")


@unittest.skipUnless((data.CACHE / "homeoccupancy_standard.pkl").exists(), "cache not built")
class CacheTests(unittest.TestCase):
    """Checks on the real built caches."""

    @classmethod
    def setUpClass(cls):
        cls.table, cls.seq, cls.gap = data.load(data.PRESENCE_SOURCES)

    def test_one_split_per_recording_and_no_overlap(self):
        per_rec = self.table.groupby(["source", "recording"])["split"].nunique()
        self.assertTrue((per_rec == 1).all())
        recs = self.table.groupby("split")["recording"].apply(lambda r: set(self.table.loc[r.index, "source"] + "/" + r))
        for a in recs.index:
            for b in recs.index:
                if a < b:
                    self.assertFalse(recs[a] & recs[b], f"{a} and {b} share recordings")

    def test_rows_aligned_after_concat_and_filter(self):
        self.assertTrue(self.table["window_id"].is_unique)
        self.assertEqual(len(self.table), len(self.seq))
        self.assertTrue((self.table.index == np.arange(len(self.table))).all())
        # window_id encodes source/recording/index -> must agree with the columns of the same row
        parts = self.table["window_id"].str.split("/", expand=True)
        self.assertTrue((parts[0] == self.table["source"]).all())
        self.assertTrue((parts[2].astype(int) == self.table["win"]).all())
        # after a filter + reset, sequence row i is still the window of table row i
        sub = self.table[self.table["source"] == "officelocalization"]
        keep = sub.index.to_numpy()
        self.assertTrue(np.array_equal(self.seq[keep][:5], self.seq[sub.index[:5]]))

    def test_rebuilt_window_matches_cache(self):
        rec = next(data.recordings("wicount"))
        rows, seq, gap = data.make_windows(rec)
        wc, wseq, _ = data.build("wicount")
        i = wc.index[wc["window_id"] == rows[3]["window_id"]][0]
        self.assertTrue(np.allclose(wseq[i].astype(np.float32), seq[3].astype(np.float32), atol=1e-3))

    def test_corrupted_home_timestamps_no_longer_create_fake_windows(self):
        ho = self.table[self.table["source"] == "homeoccupancy"]
        self.assertLess((ho["missing"] >= 1).mean(), 0.05)     # was 50-83 % before the fix


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class CalibrationTests(unittest.TestCase):
    def table(self):
        rows = []
        for split in ["train", "test"]:
            for rec, presence in [("empty_a", 0), ("occ_a", 1)]:
                for k in range(40):
                    rows.append({"site": "s", "split": split, "recording": f"{split}_{rec}", "presence": presence,
                                 "valid": True, "t0": float(k), "window_id": f"{split}/{rec}/{k}", "std_q50": 1.0 + presence})
        return pd.DataFrame(rows)

    def test_reference_is_empty_same_split_first_30s(self):
        t = self.table()
        ref = calibration.reference_mask(t)
        r = t[ref]
        self.assertTrue((r["presence"] == 0).all())
        self.assertTrue((r["t0"] + 3 <= 30).all())
        self.assertEqual(set(r["split"]), {"train", "test"})        # each split has its own reference

    def test_reference_windows_never_scored(self):
        t = self.table()
        seq = np.ones((len(t), 32, 60), np.float16)
        _, _, ref, has_ref = calibration.calibrate(t, seq)
        common = t["valid"].to_numpy() & has_ref & ~ref
        self.assertFalse((common & ref).any())
        self.assertEqual(int(ref.sum()), 2 * 28)


# ---------------------------------------------------------------------------
# Counting folds
# ---------------------------------------------------------------------------

class CountingFoldTests(unittest.TestCase):
    def test_folds_isolated(self):
        rows = []
        for count, n_rec in [(0, 1), (1, 7), (2, 6), (3, 1)]:
            for r in range(n_rec):
                for k in range(58):
                    rows.append({"count": count, "recording": f"c{count}_r{r}", "t0": float(k)})
        t = pd.DataFrame(rows)
        tested = np.zeros(len(t), bool)
        for k in range(3):
            train, test = fold_masks(t, k)
            self.assertFalse((train & test).any())
            for c in (1, 2):                                     # no recording on both sides
                tr = set(t[train & (t["count"] == c)]["recording"])
                te = set(t[test & (t["count"] == c)]["recording"])
                self.assertFalse(tr & te)
            tested |= test
        self.assertGreater(tested.mean(), 0.9)                    # nearly every window tested once


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ModelTests(unittest.TestCase):
    def test_gapcnn_ignores_values_at_gap_steps(self):
        net = models.make_net("gapcnn", 6).eval()
        x = torch.randn(4, 32, 60, device=models.DEVICE)
        g = torch.zeros(4, 60, dtype=torch.bool, device=models.DEVICE)
        g[:, 20:30] = True
        x2 = x.clone()
        x2[:, :, 20:30] = 100.0                                   # garbage where there were no packets
        with torch.no_grad():
            a, b = net(x, g), net(x2, g)
        self.assertEqual(tuple(a.shape), (4, 6))
        self.assertTrue(torch.allclose(a, b, atol=1e-5))

    def test_plain_cnn_does_not_use_gaps(self):
        net = models.make_net("cnn", 2).eval()
        x = torch.randn(2, 32, 60, device=models.DEVICE)
        g0 = torch.zeros(2, 60, dtype=torch.bool, device=models.DEVICE)
        with torch.no_grad():
            self.assertTrue(torch.allclose(net(x, g0), net(x, ~g0)))

    def test_save_load_round_trip(self):
        net = models.make_net("gapcnn", 6).eval()
        seq = np.random.rand(3, 32, 60).astype(np.float16)
        gap = np.zeros((3, 60), bool)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.pt"
            models.save_net(net, "gapcnn", 6, path, {})
            net2, info = models.load_net(path)
        p1, p2 = models.predict_net(net, seq, gap), models.predict_net(net2, seq, gap)
        self.assertEqual(p1.shape, (3, 6))
        self.assertTrue(np.allclose(p1, p2, atol=1e-6))
        self.assertTrue(np.allclose(p1.sum(1), 1, atol=1e-5))

    def test_count_metrics_class_order(self):
        t = pd.DataFrame({"count": [0, 1, 2, 3], "valid": True, "recording": list("abcd"), "t0": 0.0})
        m = metrics.counting(t, np.array([0, 1, 2, 3]))
        self.assertEqual(len(m["confusion"]), 6)
        self.assertEqual(m["exact_accuracy"], 1.0)
        self.assertFalse(m["per_class"]["4"]["has_training_data"])


class NoiseTests(unittest.TestCase):
    def test_every_kind_runs_and_is_deterministic(self):
        rec = fake_recording(seconds=10)
        for kind in noise.KINDS:
            a = noise.apply(rec, kind, 3, seed=5)
            b = noise.apply(rec, kind, 3, seed=5)
            self.assertTrue(np.array_equal(a.amp, b.amp), kind)
            self.assertEqual(a.amp.shape[1], 64)

    def test_packet_loss_rate(self):
        rec = fake_recording(seconds=200)
        lost = 1 - len(noise.apply(rec, "packet_loss", 2, seed=1).t) / len(rec.t)
        self.assertAlmostEqual(lost, 0.30, delta=0.08)


# ---------------------------------------------------------------------------
# V3 must be unchanged
# ---------------------------------------------------------------------------

class V3FrozenTests(unittest.TestCase):
    def test_v3_hashes_unchanged(self):
        self.assertEqual(snapshot_repo.check(), [])


if __name__ == "__main__":
    unittest.main()

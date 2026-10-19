"""V4 data layer: raw ESP32 CSI -> validated, labelled, split, quality-gated windows.

Sources (raw files live in <project>/datasets/, or wherever CSI_DATASETS points):

    espectre            real homes, 5 chip types          presence + static/moving
    pipeline            room 1 quiet / congested, room 2  presence + static/moving
    homeoccupancy       one home, ESP32-C6                presence (+ sleep/work activity)
    officelocalization  one office, ESP32-C6              presence (+ zone, not used)
    wicount             one meeting room, ESP32-S3        people count 0-3
    oth                 OTH room 2 samples                people count 0-3 (cross-room check)

What this module guarantees (tested in tests/test_v4.py):

    * corrupted timestamps are removed BEFORE time is reconstructed
    * splits are assigned per RECORDING, before any window is cut
    * every window carries its own window_id, recording, split, labels and quality
    * windows with long packet gaps are flagged invalid, never silently interpolated
    * unknown labels raise an error instead of being guessed

Usage:  python V4/src/data.py --build all          build / refresh every cache
        python V4/src/data.py --build wicount --prep robust
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from common import CACHE, DATA, OTH_DIR, PROTOCOL, log

import esp32_io                                    # V3, read-only

# V3's loaders find their data relative to the V3 folder; point them at V4's DATA instead
# (module attributes only, V3 files are not changed)
esp32_io.ESPECTRE_DIR = DATA / "espectre" / "data"
esp32_io.PIPELINE_DIR = DATA / "esp32_csi_pipeline" / "dataset"
from features import cnn_input, rssi_features, window_features
from preprocess import to_uniform_grid, valid_subcarriers

FS = float(PROTOCOL["signal_contract"]["sample_rate_hz"])
WIN_S = float(PROTOCOL["signal_contract"]["window_s"])
HOP_S = float(PROTOCOL["signal_contract"]["hop_s"])
W = int(round(WIN_S * FS))
H = int(round(HOP_S * FS))

CLOCK = float(PROTOCOL["timestamp_rules"]["esp32_clock_period_us"])
SPIKE_S = float(PROTOCOL["timestamp_rules"]["spike_s"])
MAX_GAP_STEPS = int(round(PROTOCOL["window_quality"]["max_interpolated_gap_s"] * FS))
MAX_MISSING = float(PROTOCOL["window_quality"]["max_missing_ratio"])

SOURCES = ["espectre", "pipeline", "homeoccupancy", "officelocalization", "wicount", "oth"]
PRESENCE_SOURCES = ["espectre", "pipeline", "homeoccupancy", "officelocalization"]
COUNT_SOURCES = ["wicount", "oth"]

# columns that describe a window but must never be used as model features
META = ["window_id", "source", "recording", "split", "site", "label_raw", "presence", "motion",
        "activity", "count", "chip", "room", "win", "t0", "missing", "max_gap_steps", "valid"]


# ---------------------------------------------------------------------------
# Label maps (anything not listed raises)
# ---------------------------------------------------------------------------

# raw label -> (presence, motion, activity)
LABELS = {
    "espectre": {"empty": (0, "empty", ""), "static_presence": (1, "static", ""), "motion": (1, "moving", "")},
    "pipeline": {"empty": (0, "empty", ""), "stationary": (1, "static", ""), "moving": (1, "moving", "")},
    "homeoccupancy": {"empty": (0, "empty", ""), "sleep": (1, "unknown", "sleep"), "work": (1, "unknown", "work")},
    "officelocalization": {"empty": (0, "empty", ""), "one": (1, "unknown", "zone_one"),
                           "two": (1, "unknown", "zone_two"), "five": (1, "unknown", "zone_five")},
}
COUNT_LABELS = {"0": 0, "1": 1, "2": 2, "3": 3}


def map_label(source: str, raw: str):
    """(presence, motion, activity) for a presence source; KeyError on unknown labels."""
    try:
        return LABELS[source][raw]
    except KeyError:
        raise KeyError(f"{source}: label '{raw}' has no documented mapping") from None


def map_count(raw: str) -> int:
    if raw not in COUNT_LABELS:
        raise KeyError(f"count label '{raw}' is not one of {sorted(COUNT_LABELS)}")
    return COUNT_LABELS[raw]


# ---------------------------------------------------------------------------
# Splits (per recording, before windowing)
# ---------------------------------------------------------------------------

def assign_split(source: str, recording: str, role: str = "") -> str:
    """Protocol split of one recording: train / val / test / test_dev / test_external /
    test_extra_empty / unused / count (counting sources are split later into folds)."""
    if source == "espectre":
        return {"train": "train", "selection": "val", "holdout": "test", "exclude": "unused"}[role]
    if source == "pipeline":
        return "train" if recording.startswith("room1_quiet") else "test"
    if source == "homeoccupancy":
        index = recording.split("_")[1]
        return {"1": "train", "2": "val", "3": "test_dev"}[index]
    if source == "officelocalization":
        if recording in ("empty_4", "empty_5"):
            return "test_extra_empty"
        if recording == "empty_3":
            return "val"
        return {"1": "train", "2": "test_external"}[recording.split("_")[1]]
    if source in COUNT_SOURCES:
        return "count"
    raise KeyError(source)


# ---------------------------------------------------------------------------
# Timestamp cleaning
# ---------------------------------------------------------------------------

def clean_timestamps(ts_us: np.ndarray):
    """Validate a 32-bit microsecond ESP32 clock.

    Returns (keep mask over the input rows, time in seconds for the kept rows, stats).
    Order of operations matters: out-of-range values are removed first, then isolated
    spikes, and only then is the clock unwrapped. Unwrapping corrupted values first is
    the bug that broke the preliminary V4 HomeOccupancy run.
    """
    ts = np.asarray(ts_us, dtype=np.float64).copy()
    stats = {"rows": int(len(ts))}

    # some loggers print the same 32-bit clock as a SIGNED int32 (OfficeLocalization empty_4/5)
    signed = np.isfinite(ts) & (ts < 0) & (ts >= -CLOCK / 2)
    ts[signed] += CLOCK
    stats["signed_int32_converted"] = int(signed.sum())

    keep = np.isfinite(ts) & (ts >= 0) & (ts < CLOCK)
    stats["dropped_out_of_range"] = int((~keep).sum())

    # isolated spikes: far from the median of their neighbours
    idx = np.flatnonzero(keep)
    unwrapped = np.unwrap(ts[idx], period=CLOCK)
    med = pd.Series(unwrapped).rolling(5, center=True, min_periods=1).median().to_numpy()
    spike = np.abs(unwrapped - med) > SPIKE_S * 1e6
    keep[idx[spike]] = False
    stats["dropped_spikes"] = int(spike.sum())

    # unwrap the clean clock, then sort (packets can arrive slightly out of order)
    t = np.unwrap(ts[keep], period=CLOCK) / 1e6
    d = np.diff(t)
    stats["wraps"] = int((np.diff(ts[keep]) < -CLOCK / 2).sum())
    stats["out_of_order"] = int((d < 0).sum())
    stats["duplicates"] = int((d == 0).sum())
    return keep, t, stats


# ---------------------------------------------------------------------------
# Raw readers
# ---------------------------------------------------------------------------

@dataclass
class Recording:
    meta: dict
    amp: np.ndarray            # [T, S] L-LTF amplitude, time-sorted
    t: np.ndarray              # [T] seconds from the first packet
    rssi: np.ndarray           # [T]
    stats: dict = field(default_factory=dict)


def parse_iq(series, n_values: int):
    """'[1, 2, ...]' strings -> (values [N, n_values], ok mask [len(series)])."""
    def one(text):
        try:
            return np.fromstring(str(text).strip().strip('"').strip("[]"), sep=",")
        except ValueError:                       # garbled serial line -> rejected below
            return np.zeros(0)

    rows = [one(x) for x in series]
    ok = np.array([len(r) == n_values for r in rows], dtype=bool)
    values = np.vstack([r for r in rows if len(r) == n_values]) if ok.any() else np.zeros((0, n_values))
    return values.astype(np.float32), ok


def finish(values, ts_us, rssi, stats):
    """Shared tail of every raw reader: timestamp cleaning, sorting, amplitude."""
    keep, t, ts_stats = clean_timestamps(ts_us)
    stats.update(ts_stats)
    order = np.argsort(t, kind="stable")
    amp = esp32_io.lltf_amplitude(values[keep])[order]
    t = t[order] - t[order[0]]
    stats["packets_kept"] = int(len(t))
    stats["duration_s"] = float(t[-1]) if len(t) else 0.0
    stats["rate_hz"] = float(len(t) / max(t[-1], 1e-9)) if len(t) > 1 else 0.0
    return amp, t, rssi[keep][order].astype(np.float32), stats


def read_esp_csv(path: Path, n_values: int):
    """esp-csi CSV with a header row (HomeOccupancy, OfficeLocalization, WiCount)."""
    df = pd.read_csv(path, dtype=str, on_bad_lines="skip")
    stats = {"rows_read": int(len(df))}

    ok = (df["type"] == "CSI_DATA").to_numpy().copy()
    ok &= pd.to_numeric(df["len"], errors="coerce").to_numpy() == n_values
    rssi = pd.to_numeric(df["rssi"], errors="coerce").to_numpy()
    ok &= np.isfinite(rssi) & (rssi >= -110) & (rssi <= 0)
    stats["dropped_bad_fields"] = int((~ok).sum())
    df, rssi = df[ok], rssi[ok]

    values, ok_iq = parse_iq(df["data"], n_values)
    stats["dropped_bad_iq"] = int((~ok_iq).sum())
    ts = pd.to_numeric(df["local_timestamp"], errors="coerce").to_numpy()[ok_iq]
    return finish(values, ts, rssi[ok_iq], stats)


def read_oth(path: Path):
    """OTH logger CSV wrapping raw serial lines (V3 line parser, V4 timestamp cleaning)."""
    with open(path, newline="") as stream:
        lines = [row["raw_line"] for row in csv.DictReader(stream)]
    df = esp32_io.parse_raw_lines(lines)
    n_values = int(df["len"].mode()[0])
    stats = {"rows_read": len(lines), "dropped_bad_fields": int(len(lines) - (df["len"] == n_values).sum())}
    df = df[df["len"] == n_values]
    values, ok_iq = parse_iq(df["data"], n_values)
    stats["dropped_bad_iq"] = int((~ok_iq).sum())
    ts = df["local_timestamp"].to_numpy(np.float64)[ok_iq]
    return finish(values, ts, df["rssi"].to_numpy(np.float32)[ok_iq], stats)


# ---------------------------------------------------------------------------
# Recordings of one source
# ---------------------------------------------------------------------------

def base_meta(source, recording, label_raw, site, chip="", room=""):
    meta = {"source": source, "recording": recording, "label_raw": label_raw, "site": site,
            "chip": chip, "room": room, "count": -1}
    if source in COUNT_SOURCES:
        meta["count"] = map_count(label_raw)
        meta.update(presence=int(meta["count"] > 0), motion="unknown", activity="")
    else:
        presence, motion, activity = map_label(source, label_raw)
        meta.update(presence=presence, motion=motion, activity=activity)
    return meta


def recordings(source: str):
    """Yield Recording objects of one source, one at a time (keeps memory low)."""
    if source in ("homeoccupancy", "officelocalization"):
        folder = DATA / source
        for path in sorted(folder.glob("*.csv")):
            name = path.stem                                    # e.g. sleep_3
            label = name.split("_")[0]
            amp, t, rssi, stats = read_esp_csv(path, 128)
            # site for calibration: recording index (home) or split (office)
            split = assign_split(source, name)
            site = f"{source}|{name.split('_')[1]}" if source == "homeoccupancy" else f"{source}|{split}"
            meta = base_meta(source, name, label, site, chip="esp32c6", room=source)
            meta["split"] = split
            yield Recording(meta, amp, t, rssi, stats)

    elif source == "wicount":
        for path in sorted((DATA / "wicount" / "extracted" / "data").glob("*/*.csv")):
            amp, t, rssi, stats = read_esp_csv(path, 104)
            meta = base_meta(source, path.stem, path.parent.name, "wicount", chip="esp32s3", room="sribd_meeting")
            meta["split"] = "count"
            yield Recording(meta, amp, t, rssi, stats)

    elif source == "oth":
        for path in sorted(OTH_DIR.glob("*label*.csv")):
            amp, t, rssi, stats = read_oth(path)
            label = re.search(r"label(\d+)", path.name).group(1)
            meta = base_meta(source, path.stem, label, "oth", chip="esp32", room="oth_room2")
            meta["split"] = "count"
            yield Recording(meta, amp, t, rssi, stats)

    elif source in ("espectre", "pipeline"):
        loader = esp32_io.load_espectre if source == "espectre" else esp32_io.load_pipeline
        for s in loader():
            m = s.meta
            if source == "espectre":
                raw = {"empty": "empty", "static": "static_presence", "moving": "motion"}[m["motion"]]
                device = re.search(r"dev([0-9a-f]+)", m["recording"]).group(1)
                site = f"espectre|{m['room']}|{m['chip']}|{device}"
                split = assign_split(source, m["recording"], m["role"])
            else:
                raw = {"empty": "empty", "static": "stationary", "moving": "moving"}[m["motion"]]
                site = f"pipeline|{m['room']}|{m['channel']}"
                split = assign_split(source, m["recording"])
            meta = base_meta(source, m["recording"], raw, site, chip=m.get("chip", ""), room=m["room"])
            meta["split"] = split
            stats = {"packets_kept": int(len(s.t)), "duration_s": float(s.duration), "rate_hz": float(s.rate),
                     "out_of_order": int((np.diff(s.t) < 0).sum()), "duplicates": int((np.diff(s.t) == 0).sum())}
            yield Recording(meta, s.amp[:, 0, :], s.t, s.rssi, stats)
    else:
        raise KeyError(source)


# ---------------------------------------------------------------------------
# Robust preprocessing (V4 improvement candidate)
# ---------------------------------------------------------------------------

def robust_prep(amp: np.ndarray, hampel_k: int = 5, hampel_mad: float = 3.0) -> np.ndarray:
    """Packet-level clean-up before gridding.

    1. per-packet gain normalisation: divide every packet by its median amplitude over
       subcarriers (removes AGC / gain steps that hit all subcarriers at once)
    2. Hampel filter per subcarrier: values more than `hampel_mad` robust deviations away
       from the median of +-hampel_k neighbouring packets are replaced by that median
       (removes impulse interference)
    """
    gain = np.median(amp, axis=1, keepdims=True)
    x = amp / np.maximum(gain, 1e-6)

    frame = pd.DataFrame(x)
    med = frame.rolling(2 * hampel_k + 1, center=True, min_periods=1).median()
    mad = (frame - med).abs().rolling(2 * hampel_k + 1, center=True, min_periods=1).median() * 1.4826
    out = frame.where((frame - med).abs() <= hampel_mad * mad + 1e-6, med)
    return out.to_numpy(np.float32)


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

def longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of True values."""
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def make_windows(rec: Recording, robust: bool = False, features: bool = True):
    """(rows, seq float16 [n, 32, W], gap bool [n, W]) for one recording."""
    amp = robust_prep(rec.amp) if robust else rec.amp
    a3 = amp[:, None, :]
    mask = valid_subcarriers(a3)
    x, missing = to_uniform_grid(a3[:, :, mask], rec.t, FS)
    r, _ = to_uniform_grid(rec.rssi[:, None, None].astype(np.float32), rec.t, FS)
    r = r[:, 0, 0]

    rows, seqs, gaps = [], [], []
    for k, start in enumerate(range(0, len(x) - W + 1, H)):
        win, miss = x[start: start + W], missing[start: start + W]
        gap_run = longest_run(miss)
        row = {key: rec.meta[key] for key in ("source", "recording", "split", "site", "label_raw", "presence",
                                              "motion", "activity", "count", "chip", "room")}
        row.update(window_id=f"{rec.meta['source']}/{rec.meta['recording']}/{k}", win=k, t0=start / FS,
                   missing=float(miss.mean()), max_gap_steps=gap_run,
                   valid=bool(gap_run <= MAX_GAP_STEPS and miss.mean() <= MAX_MISSING))
        if features:
            row.update(window_features(win, FS))
            row.update(rssi_features(r[start: start + W]))
        rows.append(row)
        seqs.append(cnn_input(win)[0].astype(np.float16))
        gaps.append(miss)
    if not rows:
        return [], np.zeros((0, 32, W), np.float16), np.zeros((0, W), bool)
    return rows, np.stack(seqs), np.stack(gaps)


def feature_cols(df: pd.DataFrame) -> list[str]:
    """Model features only (never identifiers or labels)."""
    return [c for c in df.columns if c not in META]


# ---------------------------------------------------------------------------
# Cached builds
# ---------------------------------------------------------------------------

def cache_paths(source: str, prep: str):
    stem = CACHE / f"{source}_{prep}"
    return stem.with_suffix(".pkl"), Path(str(stem) + "_seq.npz"), Path(str(stem) + "_parse.json")


def build(source: str, prep: str = "standard", rebuild: bool = False):
    """(table, seq, gap) of one source, row-aligned; cached under V4/cache/."""
    tab_path, seq_path, parse_path = cache_paths(source, prep)
    if tab_path.exists() and seq_path.exists() and not rebuild:
        z = np.load(seq_path)
        return pickle.loads(tab_path.read_bytes()), z["seq"], z["gap"]

    CACHE.mkdir(parents=True, exist_ok=True)
    rows, seqs, gaps, parse = [], [], [], {}
    for rec in recordings(source):
        r, q, g = make_windows(rec, robust=(prep == "robust"))
        rows += r
        seqs.append(q)
        gaps.append(g)
        parse[rec.meta["recording"]] = rec.stats | {"windows": len(r), "valid_windows": int(sum(x["valid"] for x in r))}
        log(f"  {source}/{rec.meta['recording']}: {len(r)} windows, {parse[rec.meta['recording']]['valid_windows']} valid")

    table = pd.DataFrame(rows)
    seq, gap = np.concatenate(seqs), np.concatenate(gaps)
    assert len(table) == len(seq) == len(gap)
    assert table["window_id"].is_unique
    tab_path.write_bytes(pickle.dumps(table, protocol=pickle.HIGHEST_PROTOCOL))
    np.savez_compressed(seq_path, seq=seq, gap=gap)
    parse_path.write_text(json.dumps(parse, indent=2))
    return table, seq, gap


def load(sources, prep: str = "standard"):
    """Concatenate several sources; index is reset so row i <-> seq[i] <-> gap[i]."""
    tables, seqs, gaps = [], [], []
    for source in sources:
        t, s, g = build(source, prep)
        tables.append(t)
        seqs.append(s)
        gaps.append(g)
    table = pd.concat(tables, ignore_index=True)
    return table, np.concatenate(seqs), np.concatenate(gaps)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", nargs="+", default=["all"], help=f"sources to build: {SOURCES} or all")
    parser.add_argument("--prep", choices=["standard", "robust"], default="standard")
    args = parser.parse_args()
    for source in (SOURCES if args.build == ["all"] else args.build):
        log(f"building {source} ({args.prep})")
        build(source, args.prep, rebuild=True)


if __name__ == "__main__":
    main()

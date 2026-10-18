"""ESP32 CSI input: one parser for every esp-csi `CSI_DATA` variant we meet.

All our data and the user's future hardware print the Espressif esp-csi line

    CSI_DATA,<id>,<mac>,<rssi>,...,<len>,<first_word_invalid>,"[i0,q0,i1,q1,...]"

with two header layouts (older chips vs. C5/C6) and three buffer sizes:

    104 values   52 subcarriers, L-LTF without null tones   (router TX, HT20)
    128 values   64 subcarriers, L-LTF incl. null tones     (ESP-NOW, HT20)
    384 values   L-LTF 64 + HT-LTF 128 subcarriers          (ESP-NOW, HT40)

To make every setup comparable we keep the **L-LTF part only** (64 slots or the
52 already-cleaned ones). It exists in every packet type, so a model trained on
one setup sees the same kind of input on another.

Every loader yields `Stream` objects (same idea as V1):

    amp     float32 [T, 1, S]  amplitude per packet / subcarrier (one link)
    t       float64 [T]        seconds from stream start (ESP32 clock)
    rssi    float32 [T]        RSSI per packet
    count   int                people in the room (-1 if unknown)
    meta    dict               dataset / room / day / recording / activity ...
"""
from __future__ import annotations

import csv
import glob
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT = Path(__file__).resolve().parents[2]                  # D:/WIFI Based CSI Project
DATA = PROJECT / "datasets"

WICOUNT_DIR = DATA / "wicount" / "extracted" / "data"
WIGESTURE_DIR = DATA / "wigesture" / "dataset"
WIFALL_DIR = DATA / "wifall" / "WiFall"
THESIS_DIR = PROJECT / "archive" / "datasets" / "esp32_reference"


# ---------------------------------------------------------------------------
# Common record
# ---------------------------------------------------------------------------

@dataclass
class Stream:
    amp: np.ndarray
    t: np.ndarray
    rssi: np.ndarray
    count: int
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0])

    @property
    def rate(self) -> float:
        return len(self.t) / max(self.duration, 1e-9)


# ---------------------------------------------------------------------------
# CSI buffer -> L-LTF amplitude
# ---------------------------------------------------------------------------

def lltf_amplitude(values: np.ndarray) -> np.ndarray:
    """Interleaved int buffer [N, L] -> L-LTF amplitude [N, S].

    104 -> 52 subcarriers (already without nulls)
    128 / 384 -> first 64 subcarriers (L-LTF), re-ordered from esp-csi's
                 [0..31, -32..-1] to frequency order [-32..31]
    """
    n_vals = values.shape[1]
    if n_vals == 104:
        iq = values
    elif n_vals in (128, 384):
        iq = values[:, :128]
    else:
        raise ValueError(f"unsupported CSI buffer length {n_vals}")

    # esp-csi stores (imag, real) pairs; amplitude does not care about the order
    amp = np.hypot(iq[:, 0::2], iq[:, 1::2]).astype(np.float32)

    if amp.shape[1] == 64:
        amp = np.concatenate([amp[:, 32:], amp[:, :32]], axis=1)    # fftshift
    return amp


def parse_data_field(series) -> np.ndarray:
    """Column of '[1, 2, ...]' strings -> int array [N, L] (fast path)."""
    txt = "\n".join(s.strip().strip('"').strip("[]") for s in series)
    return np.loadtxt(io.StringIO(txt), delimiter=",", dtype=np.float32, ndmin=2)


# ---------------------------------------------------------------------------
# Format A: esp-csi CSV with a header row (WiCount / WiGesture / WiFall)
# ---------------------------------------------------------------------------

def read_espcsi_csv(path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (amp [T, S], t [T] seconds, rssi [T])."""
    df = pd.read_csv(path, usecols=["local_timestamp", "rssi", "len", "data"]).dropna()

    # keep the most common buffer length (drops the odd malformed line)
    mode_len = df["len"].mode()[0]
    df = df[df["len"] == mode_len]

    vals = parse_data_field(df["data"])
    ts = np.unwrap(df["local_timestamp"].to_numpy(np.float64), period=2 ** 32) / 1e6
    order = np.argsort(ts, kind="stable")

    return lltf_amplitude(vals)[order], (ts - ts[order[0]])[order], df["rssi"].to_numpy(np.float32)[order]


# ---------------------------------------------------------------------------
# Format B: logger CSV wrapping raw serial lines (OTH thesis, our own logger)
# ---------------------------------------------------------------------------

# header of the older chips (ESP32 / S2 / S3 / C3); the C5/C6 header is shorter
RAW_COLS_OLD = ["type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth", "smoothing",
                "not_sounding", "aggregation", "stbc", "fec_coding", "sgi", "noise_floor", "ampdu_cnt",
                "channel", "secondary_channel", "local_timestamp", "ant", "sig_len", "rx_format",
                "len", "first_word", "data"]
RAW_COLS_NEW = ["type", "seq", "mac", "rssi", "rate", "noise_floor", "fft_gain", "agc_gain", "channel",
                "local_timestamp", "sig_len", "rx_format", "len", "first_word", "data"]


def parse_raw_lines(lines) -> pd.DataFrame:
    """Raw 'CSI_DATA,...' serial lines -> DataFrame (both header layouts)."""
    rows = []
    for line in lines:
        if not line.startswith("CSI_DATA"):
            continue
        head, _, data = line.partition('"[')
        fields = head.rstrip(",").split(",")
        cols = RAW_COLS_OLD if len(fields) == len(RAW_COLS_OLD) - 1 else RAW_COLS_NEW
        if len(fields) != len(cols) - 1:
            continue                                              # truncated line
        rows.append(fields + [data.rstrip('"]')])
    df = pd.DataFrame(rows, columns=RAW_COLS_OLD if rows and len(rows[0]) == len(RAW_COLS_OLD) else RAW_COLS_NEW)
    for c in ["rssi", "local_timestamp", "len", "noise_floor"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["rssi", "local_timestamp", "len"])


def read_logger_csv(path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """csi_logger.py output: columns pc_timestamp,label,tag,note,raw_line."""
    with open(path, newline="") as f:
        lines = [r["raw_line"] for r in csv.DictReader(f)]
    df = parse_raw_lines(lines)

    mode_len = df["len"].mode()[0]
    df = df[df["len"] == mode_len]
    vals = parse_data_field(df["data"])

    # keep only rows whose buffer has the expected length (serial glitches)
    ok = vals.shape[1] == mode_len
    ts = np.unwrap(df["local_timestamp"].to_numpy(np.float64), period=2 ** 32) / 1e6
    order = np.argsort(ts, kind="stable")
    return lltf_amplitude(vals)[order], (ts - ts[order[0]])[order], df["rssi"].to_numpy(np.float32)[order]


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _stream(amp, t, rssi, count, **meta):
    return Stream(amp=amp[:, None, :], t=t, rssi=rssi, count=count, meta=meta)


def load_wicount():
    """SRIBD meeting room, Dec 2023, 0-3 people, 15 x 60 s."""
    for path in sorted(glob.glob(str(WICOUNT_DIR / "*" / "*.csv"))):
        amp, t, rssi = read_espcsi_csv(path)
        yield _stream(amp, t, rssi, int(Path(path).parent.name),
                      dataset="wicount", room="sribd_meeting", day="2023-12-21",
                      recording=Path(path).stem, activity="mixed", setup="router_ht20")


def load_wigesture():
    """SRIBD meeting room, Oct 2023, exactly 1 person doing gestures, 60 s each."""
    for path in sorted(glob.glob(str(WIGESTURE_DIR / "*" / "ID*" / "*.csv"))):
        p = Path(path)
        amp, t, rssi = read_espcsi_csv(path)
        yield _stream(amp, t, rssi, 1,
                      dataset="wigesture", room="sribd_meeting", day="2023-10",
                      recording=f"{p.parent.parent.name}_{p.parent.name}_{p.stem}",
                      activity=f"gesture_{p.parent.parent.name}", subject=p.parent.name,
                      setup="router_ht20")


def load_wifall():
    """SRIBD meeting room, Nov 2023, exactly 1 person: walk / sit / stand / jump / fall."""
    for path in sorted(glob.glob(str(WIFALL_DIR / "ID*" / "*" / "*.csv"))):
        p = Path(path)
        amp, t, rssi = read_espcsi_csv(path)
        if len(t) < 20:
            continue
        yield _stream(amp, t, rssi, 1,
                      dataset="wifall", room="sribd_meeting", day="2023-11",
                      recording=f"{p.parent.parent.name}_{p.parent.name}_{p.stem}",
                      activity=p.parent.name.lower(), subject=p.parent.parent.name,
                      setup="router_ht20")


def load_thesis_samples():
    """OTH thesis sample files: own room 'room2', ESP-NOW HT40, 0-3 people, ~60 s each."""
    for path in sorted(THESIS_DIR.glob("*.csv")):
        label = int(re.search(r"label(\d+)", path.name).group(1))
        amp, t, rssi = read_logger_csv(path)
        yield _stream(amp, t, rssi, label,
                      dataset="thesis", room="oth_room2", day=path.name.split("_")[2],
                      recording=path.stem, activity="mixed", setup="espnow_ht40")


# ---------------------------------------------------------------------------
# V3: real-world-noise presence datasets (no exact head-count -> count = -1)
#
# meta["presence"]  0 = empty room, 1 = someone in the room
# meta["motion"]    "empty" / "static" / "moving"
# ---------------------------------------------------------------------------

ESPECTRE_DIR = DATA / "espectre" / "data"
PIPELINE_DIR = DATA / "esp32_csi_pipeline" / "dataset"

STATE = {
    "empty": (0, "empty"),
    "static_presence": (1, "static"), "stationary": (1, "static"),
    "motion": (1, "moving"), "moving": (1, "moving"),
}


def load_espectre():
    """ESPectre (GPL-3.0): real homes (bedroom, living room, hobby room, holiday home),
    5 chip types (ESP32, S3, C3, C5, C6), long recordings. Raw int8 I/Q, 64 subcarriers."""
    import json

    info = json.loads((ESPECTRE_DIR / "dataset_info.json").read_text())
    for label, files in info["files"].items():
        presence, motion = STATE[label]
        for f in files:
            z = np.load(ESPECTRE_DIR / label / f["filename"], allow_pickle=False)
            iq = z["csi_data"].astype(np.float32)
            if iq.shape[1] not in (104, 128, 384):
                continue
            ts = z["device_ticks_us"].astype(np.float64) / 1e6
            order = np.argsort(ts, kind="stable")
            amp = lltf_amplitude(iq)[order]
            yield _stream(amp, ts[order] - ts[order[0]], z["rssi_dbm"].astype(np.float32)[order], -1,
                          dataset="espectre", room=f["environment"], day=f["collected_at"][:10],
                          recording=Path(f["filename"]).stem, activity=motion, setup=f"espectre_{f['chip']}",
                          chip=f["chip"], role=f.get("dataset_role", ""), presence=presence, motion=motion)


def load_pipeline():
    """esp32-csi-pipeline (CC BY 4.0): one subject, room 1 on the QUIET channel vs the
    MOST CONGESTED channel (real crowded-WiFi noise), plus an unseen room 2.
    Amplitude already given (64 subcarriers); ~100 Hz, packet counter as time."""
    for path in sorted(PIPELINE_DIR.glob("*/*.csv")):
        cond = path.parent.name                                    # room1_quiet / room1_noisy / room2_quiet
        state = path.stem.split("_")[0]
        presence, motion = STATE[state]

        df = pd.read_csv(path)
        df = df.apply(pd.to_numeric, errors="coerce").dropna()    # drop corrupted rows
        sub = [c for c in df.columns if c.startswith("Sub_")]
        amp = df[sub].to_numpy(np.float32)
        amp = np.concatenate([amp[:, 32:], amp[:, :32]], axis=1)  # esp-csi order -> frequency order
        t = (df["Packet_Count"].to_numpy(np.float64) - df["Packet_Count"].iloc[0]) / 100.0
        order = np.argsort(t, kind="stable")

        yield _stream(amp[order], t[order], np.zeros(len(t), np.float32), 1 if presence else 0,
                      dataset="pipeline", room=cond.split("_")[0], day="2026",
                      recording=f"{cond}_{path.stem}", activity=motion, setup="pipeline_ap",
                      channel=cond.split("_")[1], presence=presence, motion=motion)


OPENCSI_DIR = DATA / "opencsi" / "opencsi-dataset-release" / "trials"
OPENCSI_LINKS_PER_TRIAL = 12             # each link = one single-link recording (our setup)
OPENCSI_LABEL = {"empty_room": "empty", "occupied_static": "static", "occupied_moving": "moving"}


def load_opencsi():
    """OpenCSI (CC BY 4.0): 3 real rooms, ESP32-S3 / S3' / C3 meshes, 802.11n HT20, ~20 Hz.
    Protocol: 5 min empty -> 3 min seated -> 2 min walking -> 2 min empty, one subject.
    Every directed link is treated as an independent single-link recording; a fixed random
    subset of links per trial keeps memory bounded. The C6 trial (802.11ax, 246 sc) is skipped."""
    import json
    import time

    import zstandard

    for trial in sorted(OPENCSI_DIR.iterdir()):
        if trial.name.endswith("_C6"):
            continue
        ann = [json.loads(l) for l in (trial / "annotations.jsonl").read_text().splitlines() if l.strip()]

        # read all packets once: link -> list of (t, rssi, amplitude)
        links = {}
        with open(trial / "raw" / "csi_raw.jsonl.zst", "rb") as f:
            reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f), encoding="utf-8")
            bad = 0
            for line in reader:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:                  # truncated line in the real log
                    bad += 1
                    continue
                if d.get("num_subcarriers") != 52:
                    continue
                iq = np.asarray(d["iq_data"], dtype=np.float32)
                links.setdefault((d["tx_mac"], d["rx_mac"]), []).append(
                    (d["wall_ts"], d["rssi"], np.hypot(iq[:, 0], iq[:, 1])))

        if bad:
            print(f"  opencsi {trial.name}: skipped {bad} malformed packet lines")
        rng = np.random.default_rng(0)
        keys = sorted(links)
        keys = [keys[i] for i in sorted(rng.choice(len(keys), min(OPENCSI_LINKS_PER_TRIAL, len(keys)),
                                                   replace=False))]
        env, chip = trial.name.split("_", 1)

        for tx, rx in keys:
            rows = sorted(links[(tx, rx)], key=lambda r: r[0])
            t_all = np.array([r[0] for r in rows])
            rssi_all = np.array([r[1] for r in rows], dtype=np.float32)
            amp_all = np.stack([r[2] for r in rows])

            for a in ann:
                state = OPENCSI_LABEL.get(a["fine_label"])
                if state is None:
                    continue
                sel = (t_all >= a["start_ts"]) & (t_all < a["end_ts"])
                if sel.sum() < 100:
                    continue
                t = t_all[sel] - t_all[sel][0]
                presence = 0 if state == "empty" else 1
                yield _stream(amp_all[sel], t, rssi_all[sel], presence,
                              dataset="opencsi", room=env, day=time.strftime("%Y-%m-%d", time.gmtime(a["start_ts"])),
                              recording=f"{trial.name}_{tx[-5:]}_{rx[-5:]}_{a['fine_label']}_{int(a['start_ts'])}",
                              activity=state, setup=f"opencsi_{chip}", chip=chip,
                              channel=f"{tx[-5:]}>{rx[-5:]}", presence=presence, motion=state)
        del links


LOADERS = {
    "wicount": load_wicount,
    "wigesture": load_wigesture,
    "wifall": load_wifall,
    "thesis": load_thesis_samples,
    "espectre": load_espectre,
    "pipeline": load_pipeline,
    "opencsi": load_opencsi,
}

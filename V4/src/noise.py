"""Synthetic 'crazy noise' stress suite, applied to the RAW packet stream.

Each function takes a Recording and returns a new, noisy Recording. The noise is
added BEFORE preprocessing, so the whole pipeline (grid, gap mask, quality gate,
features, models) sees it the way it would see a bad real stream.

    packet_loss       bursty loss (Gilbert-Elliott)            Wi-Fi congestion, weak link
    agc_jumps         random gain steps on all subcarriers     ESP32 AGC re-tuning
    impulses          short spikes on some subcarriers         microwave / Bluetooth bursts
    white_noise       additive Gaussian noise at a given SNR   weak signal, far boards
    fan               periodic ripple 1-15 Hz                  fans, AC units, machines
    jitter            timestamp jitter                         USB / serial timing
    dead_subcarriers  some subcarriers stuck at a constant     firmware / hardware faults
    crazy             everything at level 3 at once

These are stress tests. They are NOT a substitute for real noisy recordings.
"""
from __future__ import annotations

import numpy as np

from data import Recording

KINDS = ["packet_loss", "agc_jumps", "impulses", "white_noise", "fan", "jitter", "dead_subcarriers", "crazy"]

# strength per level 1 / 2 / 3 (documented in configs/v4_protocol.json)
PARAMS = {
    "packet_loss": {"loss": [0.10, 0.30, 0.50], "burst_s": [0.05, 0.20, 0.50]},
    "agc_jumps": {"log_std": [0.10, 0.30, 0.60], "every_s": 5.0},
    "impulses": {"rate": [0.01, 0.05, 0.10], "subcarriers": 0.30},
    "white_noise": {"snr_db": [25.0, 15.0, 8.0]},
    "fan": {"depth": [0.02, 0.05, 0.10]},
    "jitter": {"std_s": [0.002, 0.005, 0.010]},
    "dead_subcarriers": {"share": [0.10, 0.25, 0.40]},
}


def _copy(rec: Recording, amp=None, t=None, rssi=None, keep=None) -> Recording:
    amp = rec.amp if amp is None else amp
    t = rec.t if t is None else t
    rssi = rec.rssi if rssi is None else rssi
    if keep is not None:
        amp, t, rssi = amp[keep], t[keep], rssi[keep]
    return Recording(dict(rec.meta), amp.astype(np.float32), t, rssi, dict(rec.stats))


# ---------------------------------------------------------------------------
# Noise types
# ---------------------------------------------------------------------------

def packet_loss(rec, level, rng):
    """Two-state Markov chain: 'good' keeps packets, 'bad' drops them."""
    p = PARAMS["packet_loss"]
    loss, burst = p["loss"][level - 1], p["burst_s"][level - 1]
    rate = max(rec.stats.get("rate_hz", len(rec.t) / max(rec.t[-1], 1e-9)), 1.0)
    mean_bad = max(burst * rate, 1.0)                     # packets per bad burst
    p_bad_to_good = 1.0 / mean_bad
    p_good_to_bad = p_bad_to_good * loss / (1.0 - loss)

    keep = np.ones(len(rec.t), dtype=bool)
    bad = False
    draws = rng.random(len(rec.t))
    for i, u in enumerate(draws):
        bad = (u < p_good_to_bad) if not bad else (u >= p_bad_to_good)
        keep[i] = not bad
    keep[0] = keep[-1] = True
    return _copy(rec, keep=keep)


def agc_jumps(rec, level, rng):
    p = PARAMS["agc_jumps"]
    n_steps = max(int(rec.t[-1] / p["every_s"]), 1)
    edges = np.sort(rng.uniform(0, rec.t[-1], n_steps))
    gains = np.exp(rng.normal(0, p["log_std"][level - 1], n_steps + 1))
    gain = gains[np.searchsorted(edges, rec.t)]
    return _copy(rec, amp=rec.amp * gain[:, None])


def impulses(rec, level, rng):
    p = PARAMS["impulses"]
    amp = rec.amp.copy()
    hit = rng.random(len(amp)) < p["rate"][level - 1]
    for i in np.flatnonzero(hit):
        cols = rng.random(amp.shape[1]) < p["subcarriers"]
        amp[i, cols] *= rng.uniform(3, 10)
    return _copy(rec, amp=amp)


def white_noise(rec, level, rng):
    snr = 10 ** (PARAMS["white_noise"]["snr_db"][level - 1] / 20)
    sigma = np.mean(rec.amp) / snr
    return _copy(rec, amp=np.abs(rec.amp + rng.normal(0, sigma, rec.amp.shape)))


def fan(rec, level, rng):
    """1 + depth * sin(2 pi f t + phase_s): a ripple with a per-subcarrier phase."""
    depth = PARAMS["fan"]["depth"][level - 1]
    f = rng.uniform(1.0, 15.0)
    phase = rng.uniform(0, 2 * np.pi, rec.amp.shape[1])
    ripple = 1 + depth * np.sin(2 * np.pi * f * rec.t[:, None] + phase[None, :])
    return _copy(rec, amp=rec.amp * ripple)


def jitter(rec, level, rng):
    t = rec.t + rng.normal(0, PARAMS["jitter"]["std_s"][level - 1], len(rec.t))
    t = np.clip(t, 0, None)
    order = np.argsort(t, kind="stable")
    return Recording(dict(rec.meta), rec.amp[order], t[order] - t[order][0], rec.rssi[order], dict(rec.stats))


def dead_subcarriers(rec, level, rng):
    amp = rec.amp.copy()
    n_dead = int(round(PARAMS["dead_subcarriers"]["share"][level - 1] * amp.shape[1]))
    cols = rng.choice(amp.shape[1], n_dead, replace=False)
    amp[:, cols] = np.median(amp[:, cols], axis=0, keepdims=True)
    return _copy(rec, amp=amp)


FUNCS = {"packet_loss": packet_loss, "agc_jumps": agc_jumps, "impulses": impulses, "white_noise": white_noise,
         "fan": fan, "jitter": jitter, "dead_subcarriers": dead_subcarriers}


def apply(rec: Recording, kind: str, level: int, seed: int) -> Recording:
    """Noisy copy of `rec`. 'crazy' applies every noise type at level 3."""
    rng = np.random.default_rng(seed)
    if kind == "clean":
        return rec
    if kind == "crazy":
        out = rec
        for name in ["fan", "agc_jumps", "impulses", "white_noise", "dead_subcarriers", "jitter", "packet_loss"]:
            out = FUNCS[name](out, 3, rng)
        return out
    return FUNCS[kind](rec, level, rng)


def crop(rec: Recording, seconds: float) -> Recording:
    """First `seconds` of a recording (keeps the noise suite affordable)."""
    keep = rec.t <= seconds
    return _copy(rec, keep=keep)


# ---------------------------------------------------------------------------
# Training augmentation (seeds 0-999 only; evaluation uses 1000+ / 2000+)
# ---------------------------------------------------------------------------

AUG_CROP_S = 300.0


def augmented(sources, prep: str):
    """One noisy copy (random type and level) of the first 300 s of every TRAINING recording.

    Cached per source as V4/cache/aug_<source>_<prep>.pkl/.npz. Window IDs end in '#aug'.
    """
    import pickle

    import pandas as pd

    from common import CACHE, log
    from data import make_windows, recordings

    tables, seqs, gaps = [], [], []
    for source in sources:
        tab_path, seq_path = CACHE / f"aug_{source}_{prep}.pkl", CACHE / f"aug_{source}_{prep}.npz"
        if not tab_path.exists():
            rows, sq, gp = [], [], []
            rng = np.random.default_rng(sum(source.encode()) % 1000)       # stable across runs
            for i, rec in enumerate(r for r in recordings(source) if r.meta["split"] == "train"):
                kind = KINDS[rng.integers(len(KINDS))]
                level = int(rng.integers(1, 4))
                noisy = apply(crop(rec, AUG_CROP_S), kind, level, seed=i % 1000)
                r, q, g = make_windows(noisy, robust=(prep == "robust"))
                for row in r:
                    row["window_id"] += f"#aug:{kind}{level}"
                rows += r
                sq.append(q)
                gp.append(g)
                log(f"  aug {source}/{rec.meta['recording']}: {kind} level {level}, {len(r)} windows")
            tab_path.write_bytes(pickle.dumps(pd.DataFrame(rows)))
            np.savez_compressed(seq_path, seq=np.concatenate(sq), gap=np.concatenate(gp))
        z = np.load(seq_path)
        tables.append(pickle.loads(tab_path.read_bytes()))
        seqs.append(z["seq"])
        gaps.append(z["gap"])
    return pd.concat(tables, ignore_index=True), np.concatenate(seqs), np.concatenate(gaps)

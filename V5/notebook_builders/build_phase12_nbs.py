"""Generate the Phase 1 + Phase 2 Colab notebooks of the people-counting plan.

  phase1_shared_format_v2.ipynb -> one shared input format for the local datasets
                                   (80 MHz set, CountFi, WiCount; uploaded to Drive as 3 zips)
  phase2_baselines_v2.ipynb     -> honest held-out baselines per dataset
                                 (LightGBM on count cues + a tiny 1D-CNN)

Both notebooks run on Colab only; nothing here touches the laptop's data.
Run:  py -3 colab_tools/build_phase12_nbs.py
"""

import ast
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# ===============================================================
# PHASE 1 - shared format
# ===============================================================
P1_INTRO = r'''# Phase 1 - one shared input format (local datasets)

Turns the three people-count datasets into the format an **ESP32-S3** can also produce:

| Step | What happens |
|---|---|
| 20 MHz slice | 80 MHz data is cut into its four 20 MHz sub-bands, **52 tones each** (WiCount already has 52) |
| One link | Every antenna x sub-band is stored as a separate **view** (one view = what one ESP32 would see) |
| Amplitude only | Phase is dropped |
| Gain normalisation | Each packet is divided by its median over tones (removes automatic gain jumps) |
| 50 Hz grid | Packets are averaged into 20 ms bins; gaps up to 60 ms are interpolated, longer gaps stay **missing** |
| Recording normalisation | Each tone is divided by its mean over the recording (per receiver) -> relative fluctuation |
| Windows | 5 s windows, 2.5 s hop; windows with > 20 % missing bins are dropped |
| Count cues | 10 features per window and view (fluctuation, motion energy, number of independent patterns, ...) |

**Data: three zips of the laptop folder `D:/wifi/Datasets`, uploaded to Google Drive.**
Colab runs on a Google machine and cannot open `D:` directly, so upload the data once:

1. On the laptop, open `D:/wifi/Datasets` and zip each folder with **7-Zip**
   (right-click the folder -> 7-Zip -> Add to archive -> format `zip`, method `Deflate`):

   | Folder | Zip name | Size |
   |---|---|---|
   | `Dataset` (PC-1a ... PC-4a, `.mat`) | `Dataset.zip` | 5.6 GB |
   | `CountFi` (`48/.../*.pcap`) | `CountFi.zip` | 4.9 GB |
   | `WiCount` (`data/<count>/*.csv`) | `WiCount.zip` | 41 MB |

2. On drive.google.com create the folder **`My Drive/wifi_csi/data/`** and drag the three zips into it.
   Wait until the upload has fully finished.
3. Run this notebook top to bottom. Cell 1 checks the zips, cell 2 unzips them onto the Colab disk
   (fast local reads). Both stop with a clear message if something is missing.

Output: `MyDrive/wifi_csi/shared_format/<dataset>/<recording>.npz` + `features.parquet`.
The run is resumable: recordings already saved are skipped. Runs fine on a CPU runtime.'''


P1_SETUP = r'''# 1. Mount Drive, paths, settings
from google.colab import drive
drive.mount("/content/drive")

import glob
import json
import os
import shutil
import struct
import subprocess
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

# All paths are Colab / Google Drive paths - never a laptop path like D:\...
DRIVE_ROOT = "/content/drive/MyDrive/wifi_csi"
ZIP_DIR = f"{DRIVE_ROOT}/data"                  # Dataset.zip, CountFi.zip, WiCount.zip go here
OUT_ROOT = f"{DRIVE_ROOT}/shared_format"        # Phase 1 output
VM_RAW = "/content/raw"                         # unzipped data on the Colab disk

# dataset key -> folder name inside D:/wifi/Datasets (= zip name without .zip)
SOURCES = {
    "mat80": "Dataset",
    "countfi": "CountFi",
    "wicount": "WiCount",
}

# ---- Check the three zips are on Drive ----
missing = []
for key, sub in SOURCES.items():
    zip_path = os.path.join(ZIP_DIR, sub + ".zip")
    on_vm = os.path.isdir(os.path.join(VM_RAW, sub))

    if os.path.isfile(zip_path):
        print(f"{key:8s} {zip_path}   {os.path.getsize(zip_path) / 1e9:5.2f} GB")
    elif on_vm:
        print(f"{key:8s} already unzipped on the VM: {VM_RAW}/{sub}")
    else:
        print(f"{key:8s} {zip_path}   NOT FOUND")
        missing.append(sub + ".zip")

if missing:
    print()
    print("Files in", ZIP_DIR, ":", sorted(os.listdir(ZIP_DIR)) if os.path.isdir(ZIP_DIR) else "(folder does not exist)")
    raise FileNotFoundError(
        f"Missing on Drive: {missing}. Upload them to My Drive/wifi_csi/data/ "
        "(see the steps at the top of the notebook), wait for the upload to finish, then re-run this cell."
    )

# ---- shared-format settings (keep identical on the ESP32 later) ----
FS = 50              # common sample rate (Hz)
WIN_S = 5.0          # window length (s)
HOP_S = 2.5          # window hop (s)
MAX_GAP_BINS = 3     # gaps up to 3 bins (60 ms) are interpolated
MAX_MISSING = 0.20   # drop windows with more missing bins than this
N_TONES = 52

os.makedirs(OUT_ROOT, exist_ok=True)
print("all zips found - output goes to", OUT_ROOT)'''


P1_COPY = r'''# 2. Unzip the three zips onto the VM disk (one read from Drive, then fast local reads)
# 7-Zip is used because it opens both normal (Deflate) and Windows big-file (Deflate64) zips.
if shutil.which("7z") is None:
    subprocess.run(["apt-get", "install", "-y", "-q", "p7zip-full"], check=True,
                   stdout=subprocess.DEVNULL)

VM_ZIPS = "/content/zips"
os.makedirs(VM_ZIPS, exist_ok=True)
os.makedirs(VM_RAW, exist_ok=True)


def unzip_source(sub):
    """Drive zip -> VM copy -> extracted folder VM_RAW/<sub>."""
    dst = os.path.join(VM_RAW, sub)
    if os.path.isdir(dst):
        print(f"{sub:8s} already unzipped: {dst}")
        return

    # copy the zip to the VM first (reading one big file from Drive is fast, many small reads are not)
    t0 = time.time()
    local_zip = os.path.join(VM_ZIPS, sub + ".zip")
    shutil.copy(os.path.join(ZIP_DIR, sub + ".zip"), local_zip)
    print(f"{sub:8s} copied zip to VM in {time.time() - t0:.0f} s")

    # extract into a temp folder
    tmp = os.path.join(VM_RAW, "_tmp_" + sub)
    shutil.rmtree(tmp, ignore_errors=True)
    result = subprocess.run(["7z", "x", "-y", f"-o{tmp}", local_zip],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"7z could not extract {sub}.zip (is the upload complete?):\n{result.stderr}")

    # the zip may hold the folder itself (Dataset/PC-1a/...) or only its contents (PC-1a/...)
    entries = [e for e in os.listdir(tmp) if not e.startswith(("__MACOSX", "."))]
    if entries == [sub]:
        shutil.move(os.path.join(tmp, sub), dst)
        shutil.rmtree(tmp)
    else:
        os.rename(tmp, dst)

    os.remove(local_zip)                         # free the VM disk
    print(f"{sub:8s} unzipped to {dst} in {time.time() - t0:.0f} s")


for sub in SOURCES.values():
    unzip_source(sub)

RAW = VM_RAW

# ---- Check the expected files are really there before processing ----
counts = {
    "mat80": len(glob.glob(os.path.join(RAW, "Dataset", "PC-*", "*.mat"))),
    "countfi": len(glob.glob(os.path.join(RAW, "CountFi", "**", "*.pcap"), recursive=True)),
    "wicount": len(glob.glob(os.path.join(RAW, "WiCount", "data", "*", "*.csv"))),
}
for key, n in counts.items():
    print(f"{key:8s} {n:6d} files")

empty = [key for key, n in counts.items() if n == 0]
if empty:
    for sub in SOURCES.values():
        print(sub, "->", sorted(os.listdir(os.path.join(RAW, sub)))[:10])
    raise RuntimeError(f"No data files found for {empty}. Check the folder layout inside the zip (printed above).")

free_gb = shutil.disk_usage("/content").free / 1e9
print(f"raw data read from: {RAW}   VM free: {free_gb:.1f} GB")'''


P1_HELPERS = r'''# 3. Shared-format helpers: sub-bands, 50 Hz grid, normalisation, count cues

# ---------------------------------------------------------------
# 20 MHz sub-bands of an 80 MHz channel (256-point FFT, natural order)
# ---------------------------------------------------------------
# Sub-band s covers tone indices -128+64s ... -65+64s. We keep 52 tones,
# skipping 6 at each edge. This automatically avoids DC (bins 0, 1, 255)
# and the guard band (bins 123-133) that carry no channel data.
def subband_bins(s):
    start = -128 + 64 * s
    tones = np.arange(start + 6, start + 6 + N_TONES)
    return np.mod(tones, 256)


SUBBANDS = [subband_bins(s) for s in range(4)]


# ---------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------
def gain_normalise(A):
    """Per packet: divide by the median over tones (removes AGC jumps)."""
    return A / (np.median(A, axis=1, keepdims=True) + 1e-6)


def recording_normalise(X):
    """Per recording + receiver: each tone relative to its mean -> fluctuation around 0."""
    mu = np.nanmean(X, axis=0, keepdims=True)
    dead = mu < 1e-3 * np.nanmedian(mu)                    # null tones, if any
    Y = X / (mu + 1e-6) - 1.0
    Y[:, dead[0]] = 0.0
    return Y


# ---------------------------------------------------------------
# Resampling onto the 50 Hz grid
# ---------------------------------------------------------------
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


def to_shared_series(t, A):
    """Raw amplitude of ONE view -> normalised 50 Hz series (bins, 52) + valid mask."""
    A = gain_normalise(A.astype(np.float32))
    X, valid = resample_to_grid(t, A)
    X, valid = fill_short_gaps(X, valid)
    return recording_normalise(X), valid


# ---------------------------------------------------------------
# Count cues per window
# ---------------------------------------------------------------
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


# ---------------------------------------------------------------
# Windowing + saving one recording
# ---------------------------------------------------------------
def save_recording(rec):
    """rec: dict with series (views, bins, 52), valid (bins,), view_names, metadata."""
    S, valid = rec["series"], rec["valid"]
    win, hop = int(WIN_S * FS), int(HOP_S * FS)

    starts = [a for a in range(0, S.shape[1] - win + 1, hop)
              if valid[a:a + win].mean() >= 1 - MAX_MISSING]

    F = np.zeros((S.shape[0], len(starts), len(FEATURE_NAMES)), dtype=np.float32)
    for v in range(S.shape[0]):
        for j, a in enumerate(starts):
            F[v, j] = window_features(S[v, a:a + win])

    meta = {k: rec[k] for k in ["dataset", "recording", "group", "setup", "receiver", "count"]}
    meta["seconds"] = round(S.shape[1] / FS, 1)
    meta["missing_frac"] = round(float(1 - valid.mean()), 4)
    meta["n_windows"] = len(starts)

    out_dir = os.path.join(OUT_ROOT, rec["dataset"])
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, rec["recording"] + ".npz"),
                        series=S.astype(np.float16), valid=valid,
                        starts=np.array(starts, dtype=np.int32), feats=F,
                        view_names=np.array(rec["view_names"]), meta=json.dumps(meta))
    return meta


def already_done(dataset, recording):
    return os.path.exists(os.path.join(OUT_ROOT, dataset, recording + ".npz"))


print("sub-band bins:", [(int(b[0]), int(b[-1])) for b in SUBBANDS])'''


P1_MAT80 = r'''# 4. 80 MHz set (Dataset/PC-*a/*.mat): 4 antennas x 4 sub-bands = 16 views per recording
MAT80_RATE = 173          # packets/s per antenna (published; the files have no timestamps)
MI_TYPES = {1: "i1", 2: "u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4", 7: "<f4", 9: "<f8"}


# ---------------------------------------------------------------
# Streaming MAT-v5 reader (column by column, low RAM)
# ---------------------------------------------------------------
class ZStream:
    def __init__(self, f, comp_len):
        self.f, self.left = f, comp_len
        self.d, self.buf = zlib.decompressobj(), bytearray()

    def read(self, n):
        while len(self.buf) < n:
            if self.left <= 0:
                self.buf += self.d.flush()
                break
            chunk = self.f.read(min(4 << 20, self.left))
            self.left -= len(chunk)
            self.buf += self.d.decompress(chunk)
        out = bytes(self.buf[:n])
        del self.buf[:n]
        return out


def read_tag(src):
    raw = src.read(8)
    t, n = struct.unpack("<II", raw)
    if t >> 16:                                   # small data element
        return t & 0xFFFF, t >> 16, raw[4:4 + (t >> 16)], True
    return t, n, None, False


def read_element(src):
    t, n, small, is_small = read_tag(src)
    if is_small:
        return t, small
    data = src.read(n)
    if n % 8:
        src.read(8 - n % 8)
    return t, data


def load_amplitude(path):
    """Return amplitude (rows, 256) float16, or None if the file holds no data."""
    with open(path, "rb") as f:
        f.read(128)
        _, n = struct.unpack("<II", f.read(8))
        src = ZStream(f, n)
        read_tag(src)                             # miMATRIX
        read_element(src)                         # flags
        _, dims = read_element(src)
        nrows, ncols = struct.unpack("<ii", dims[:8])
        read_element(src)                         # variable name
        if nrows == 0:
            return None

        t, nbytes, _, _ = read_tag(src)           # real part
        dt = np.dtype(MI_TYPES[t])
        R = np.empty((nrows, ncols), dtype=np.int16)
        for c in range(ncols):
            R[:, c] = np.frombuffer(src.read(nrows * dt.itemsize), dtype=dt)
        if nbytes % 8:
            src.read(8 - nbytes % 8)

        t, nbytes, _, _ = read_tag(src)           # imaginary part -> amplitude
        dt = np.dtype(MI_TYPES[t])
        A = np.empty((nrows, ncols), dtype=np.float16)
        for c in range(ncols):
            im = np.frombuffer(src.read(nrows * dt.itemsize), dtype=dt).astype(np.float32)
            A[:, c] = np.hypot(R[:, c].astype(np.float32), im)
    return A


def process_mat80(path):
    name = Path(path).stem                         # PC1a_n02
    setup = name.split("_")[0]
    count = int(name.split("_n")[1])
    try:
        A = load_amplitude(path)
    except struct.error:                           # header only, no data stream
        A = None
    if A is None:
        return None

    n = (len(A) // 4) * 4                          # rows cycle over the 4 antennas
    series, views, valid_all = [], [], None
    for ant in range(4):
        A_ant = A[ant:n:4]
        t = np.arange(len(A_ant)) / MAT80_RATE
        for s, bins in enumerate(SUBBANDS):
            X, valid = to_shared_series(t, A_ant[:, bins])
            series.append(X)
            views.append(f"ant{ant}_sb{s}")
            valid_all = valid if valid_all is None else (valid_all & valid)

    return dict(dataset="mat80", recording=name, group=setup, setup=setup, receiver="rx",
                count=count, series=np.stack(series), valid=valid_all, view_names=views)


mat_files = sorted(glob.glob(os.path.join(RAW, "Dataset", "PC-*", "*.mat")))
print(len(mat_files), "MAT files")
t0 = time.time()
for path in mat_files:
    name = Path(path).stem
    if already_done("mat80", name):
        continue
    rec = process_mat80(path)
    if rec is None:
        print(f"{name:12s} EMPTY - skipped")
        continue
    meta = save_recording(rec)
    print(f"{name:12s} count={meta['count']:2d}  {meta['seconds']:6.1f} s  "
          f"windows={meta['n_windows']:3d}  missing={meta['missing_frac']:.3f}  "
          f"[{time.time() - t0:.0f} s]")'''


P1_COUNTFI = r'''# 5. CountFi (CountFi/48/48_<rx>_<scenario>/<count>/*.pcap): QoS-data frames only, 4 sub-bands
# One recording = one (scenario, receiver, count) block.
# Each .pcap is a ~1 s burst (1000 packets) followed by ~1.2 s of nothing before the next file,
# so on real time only ~40 % of the 50 Hz bins would have data and every 5 s window would be dropped.
# The bursts are therefore STITCHED back to back (dead time between files removed).
# Split unit later = scenario (head and end of the same scenario always stay together).
ETH, IP, UDP, NEX = 14, 20, 8, 18


def read_qos(path):
    """Return (times s, amplitude (packets, 256)) of the QoS-data frames, or None."""
    data = open(path, "rb").read()
    off, ts, payloads = 24, [], []
    while off + 16 <= len(data):
        sec, usec, incl, _ = struct.unpack_from("<IIII", data, off)
        off += 16
        pkt = data[off:off + incl]
        off += incl
        p = pkt[ETH + IP + UDP:]
        if p[:2] != b"\x11\x11" or len(p) != NEX + 1024 or p[3] != 0x88:
            continue                               # not a Nexmon QoS-data CSI frame
        ts.append(sec + usec * 1e-6)
        payloads.append(p[NEX:])
    if not payloads:
        return None
    raw = np.frombuffer(b"".join(payloads), dtype="<i2").reshape(len(payloads), 512)
    A = np.hypot(raw[:, 0::2].astype(np.float32), raw[:, 1::2].astype(np.float32))
    return np.array(ts), A


def countfi_blocks(root):
    blocks = {}
    for path in glob.glob(os.path.join(root, "**", "*.pcap"), recursive=True):
        p = Path(path)
        parts = p.parent.parent.name.split("_")    # 48_end_mm_v2 -> ['48','end','mm','v2']
        receiver, scenario = parts[1], "_".join(parts[2:])
        count = int(p.parent.name)
        blocks.setdefault((scenario, receiver, count), []).append(path)
    return blocks


def process_countfi(key, paths):
    scenario, receiver, count = key

    # read every 1 s burst, each sorted by its own timestamps
    bursts = []
    for path in paths:
        r = read_qos(path)
        if r is not None:
            order = np.argsort(r[0], kind="stable")
            bursts.append((r[0][order], r[1][order]))
    if not bursts:
        return None

    # stitch: bursts in time order, each one starting one bin after the previous one ends
    bursts.sort(key=lambda b: b[0][0])
    times, amps, offset = [], [], 0.0
    for t_burst, A_burst in bursts:
        t_local = t_burst - t_burst[0]
        times.append(offset + t_local)
        amps.append(A_burst)
        offset += t_local[-1] + 1.0 / FS

    t = np.concatenate(times)
    A = np.concatenate(amps)

    series, views, valid_all = [], [], None
    for s, bins in enumerate(SUBBANDS):
        X, valid = to_shared_series(t, A[:, bins])
        series.append(X)
        views.append(f"sb{s}")
        valid_all = valid if valid_all is None else (valid_all & valid)

    return dict(dataset="countfi", recording=f"{scenario}_{receiver}_n{count:02d}",
                group=scenario, setup=scenario, receiver=receiver, count=count,
                series=np.stack(series), valid=valid_all, view_names=views)


blocks = countfi_blocks(os.path.join(RAW, "CountFi"))
print(len(blocks), "CountFi blocks,", sum(len(v) for v in blocks.values()), "pcap files")
t0 = time.time()
for key in sorted(blocks):
    scenario, receiver, count = key
    name = f"{scenario}_{receiver}_n{count:02d}"
    if already_done("countfi", name):
        with np.load(os.path.join(OUT_ROOT, "countfi", name + ".npz")) as old:
            n_old = json.loads(str(old["meta"]))["n_windows"]
        if n_old > 0:
            continue                               # done; redo only files saved before stitching (0 windows)
    rec = process_countfi(key, blocks[key])
    if rec is None:
        print(f"{name:18s} no QoS frames - skipped")
        continue
    meta = save_recording(rec)
    print(f"{name:18s} {meta['seconds']:6.1f} s  windows={meta['n_windows']:3d}  "
          f"missing={meta['missing_frac']:.3f}  [{time.time() - t0:.0f} s]")'''


P1_WICOUNT = r'''# 6. WiCount (WiCount/data/<count>/*.csv): ESP32-S3, 52 tones already, 1 view
# Split unit later = recording (each file is its own 1-minute session).
WRAP = 2 ** 32                                     # ESP32 local_timestamp is a 32-bit microsecond counter


def process_wicount(path):
    df = pd.read_csv(path)
    rows, keep_t = [], []
    for ts, txt in zip(df["local_timestamp"], df["data"]):
        try:
            vals = json.loads(txt)
        except (TypeError, ValueError):
            continue
        if len(vals) == 2 * N_TONES:
            rows.append(vals)
            keep_t.append(ts)
    if len(rows) < FS:
        return None

    iq = np.array(rows, dtype=np.float32)
    A = np.hypot(iq[:, 0::2], iq[:, 1::2])         # amplitude is the same for I/Q or Q/I order

    dt = np.diff(np.array(keep_t, dtype=np.int64))
    dt[dt < 0] += WRAP                             # counter wrap-around
    t = np.concatenate([[0], np.cumsum(dt)]) * 1e-6
    order = np.argsort(t, kind="stable")

    X, valid = to_shared_series(t[order], A[order])
    name = Path(path).stem
    return dict(dataset="wicount", recording=name, group=name, setup="room", receiver="esp32s3",
                count=int(Path(path).parent.name), series=X[None], valid=valid,
                view_names=["esp32"])


csv_files = sorted(glob.glob(os.path.join(RAW, "WiCount", "data", "*", "*.csv")))
print(len(csv_files), "WiCount files")
for path in csv_files:
    name = Path(path).stem
    if already_done("wicount", name):
        continue
    rec = process_wicount(path)
    if rec is None:
        print(f"{name} too few valid rows - skipped")
        continue
    meta = save_recording(rec)
    print(f"{name}  count={meta['count']}  {meta['seconds']:5.1f} s  "
          f"windows={meta['n_windows']:3d}  missing={meta['missing_frac']:.3f}")'''


P1_TABLE = r'''# 7. Build the feature table (one row per dataset x recording x view x window)
rows = []
for path in sorted(glob.glob(os.path.join(OUT_ROOT, "*", "*.npz"))):
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    F, starts, views = z["feats"], z["starts"], [str(v) for v in z["view_names"]]
    for v, view in enumerate(views):
        for j, a in enumerate(starts):
            rows.append({**{k: meta[k] for k in ["dataset", "recording", "group", "setup",
                                                 "receiver", "count"]},
                         "view": view, "window": j, "t_start_s": a / FS,
                         **dict(zip(FEATURE_NAMES, F[v, j]))})

if not rows:
    raise RuntimeError(f"No processed recordings in {OUT_ROOT}. Cells 4-6 saved nothing - check their output.")

feat = pd.DataFrame(rows)
feat.to_parquet(os.path.join(OUT_ROOT, "features.parquet"), index=False)

with open(os.path.join(OUT_ROOT, "format.json"), "w") as f:
    json.dump({"fs_hz": FS, "win_s": WIN_S, "hop_s": HOP_S, "n_tones": N_TONES,
               "max_gap_bins": MAX_GAP_BINS, "max_missing": MAX_MISSING,
               "feature_names": FEATURE_NAMES,
               "subband_bins": [[int(b[0]), int(b[-1])] for b in SUBBANDS]}, f, indent=1)

print(len(feat), "rows saved")
summary = (feat[feat["view"] == feat.groupby("dataset")["view"].transform("first")]
           .groupby(["dataset", "count"]).agg(recordings=("recording", "nunique"),
                                              windows=("window", "size")))
summary'''


P1_CHECK = r'''# 8. Sanity check: does the main count cue still rise with people after the 20 MHz cut?
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (dataset, g) in zip(axes, feat.groupby("dataset")):
    per_rec = g.groupby(["setup", "receiver", "count"])["cv_median"].median().reset_index()
    for (setup, receiver), h in per_rec.groupby(["setup", "receiver"]):
        h = h.sort_values("count")
        ax.plot(h["count"], h["cv_median"], marker="o", label=f"{setup}/{receiver}")
    ax.set_title(dataset)
    ax.set_xlabel("people")
    ax.set_ylabel("cv_median (median over windows and views)")
    ax.legend(fontsize=7)
fig.tight_layout()

fig_dir = os.path.join(OUT_ROOT, "figures")
os.makedirs(fig_dir, exist_ok=True)
fig.savefig(os.path.join(fig_dir, "cv_median_vs_count.png"), dpi=120)
plt.show()

# Spearman rank correlation of cv_median with count, per setup (80 MHz audit had 0.83-1.0)
corr = (feat.groupby(["dataset", "setup", "receiver", "count"])["cv_median"].median()
        .reset_index().groupby(["dataset", "setup", "receiver"])
        .apply(lambda h: h["cv_median"].corr(h["count"], method="spearman")))
corr.rename("spearman_cv_vs_count")'''


P1_FLUSH = r'''# 9. Make sure everything is uploaded to Drive before closing the runtime
drive.flush_and_unmount()
print("flushed - safe to close the runtime")'''


# ===============================================================
# PHASE 2 - baselines
# ===============================================================
P2_INTRO = r'''# Phase 2 - honest baselines per dataset

**Data:** the laptop datasets in `D:\wifi\Datasets` (80 MHz set, CountFi, WiCount), after Phase 1 has
turned them into the shared format. Phase 1 (`phase1_shared_format_v2.ipynb`) unzips the uploaded
`MyDrive/wifi_csi/data/*.zip` and writes `MyDrive/wifi_csi/shared_format/`; this notebook reads that folder.
**Run Phase 1 first.**

Every model is tested only on **groups it never saw in training**:

| Dataset | Split unit (left out one at a time) | Notes |
|---|---|---|
| mat80 | setup PC-1a ... PC-4a | counts 0, 2-10 (no 1-person class) |
| countfi | scenario (head + end together) | count = one time block per scenario; expect weak results (stress test) |
| wicount | recording (15 files) | counts 0 and 3 have one file each, so a classifier can never be right on them |

**Models**
- **LightGBM** on the 10 count cues: classifier (count as a class) and regressor (count as a number).
  With and without **per-group calibration** (features z-scored with the unlabelled data of each setup/scenario;
  in real use this is a short calibration recording after installation, so it is slightly optimistic).
- **Tiny 1D-CNN** (~35k parameters, fits the ESP32-S3 budget) on the raw 5 s x 52-tone windows, with a class head and a regression head.
- **Mean baseline**: always predict the training-set mean.

**Levels:** `single_view` = one antenna/sub-band as an ESP32 would see it; `view_ensemble` = average over the
views of the same moment; `recording` = median over a whole recording.

**Metrics:** MAE (people), exact accuracy, within ±1 accuracy, per-count recall.
Results go to `MyDrive/wifi_csi/results/phase2/`. Use a **GPU (T4) runtime** for the CNN; CPU also works, just slower.'''


P2_SETUP = r'''# 1. Mount Drive, load Phase 1 output (made from the laptop datasets in D:\wifi\Datasets)
from google.colab import drive
drive.mount("/content/drive")

import glob
import json
import os
import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

DRIVE_ROOT = "/content/drive/MyDrive/wifi_csi"
SHARED = f"{DRIVE_ROOT}/shared_format"          # Phase 1 output
RESULTS = f"{DRIVE_ROOT}/results/phase2"
os.makedirs(RESULTS, exist_ok=True)

# ---- Phase 1 must have run first ----
for needed in ["format.json", "features.parquet"]:
    if not os.path.exists(os.path.join(SHARED, needed)):
        raise FileNotFoundError(
            f"{needed} not found in {SHARED}. Run phase1_shared_format_v2.ipynb first "
            "(it unzips Dataset.zip, CountFi.zip and WiCount.zip from My Drive/wifi_csi/data/)."
        )

FORMAT = json.load(open(os.path.join(SHARED, "format.json")))
FEATURES = FORMAT["feature_names"]
FS, WIN = FORMAT["fs_hz"], int(FORMAT["win_s"] * FORMAT["fs_hz"])

feat = pd.read_parquet(os.path.join(SHARED, "features.parquet"))
feat[FEATURES] = feat[FEATURES].fillna(0.0)

DATASETS = ["mat80", "countfi", "wicount"]
MAX_COUNT = int(feat["count"].max())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0

print("device:", DEVICE, "(switch the runtime to GPU if this says cpu)")
print(feat.groupby("dataset").agg(recordings=("recording", "nunique"), groups=("group", "nunique"),
                                  views=("view", "nunique"), rows=("window", "size")))'''


P2_METRICS = r'''# 2. Metrics and result collection
RESULT_ROWS, PER_COUNT_ROWS, RECORDING_ROWS = [], [], []


def score(y, pred):
    """pred may be continuous (regression) or integer (classification)."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    rounded = np.clip(np.rint(pred), 0, MAX_COUNT)
    return {"n": len(y),
            "mae": float(np.mean(np.abs(pred - y))),
            "exact": float(np.mean(rounded == y)),
            "within1": float(np.mean(np.abs(rounded - y) <= 1))}


def record(dataset, model, task, calib, level, y, pred):
    RESULT_ROWS.append({"dataset": dataset, "model": model, "task": task, "calib": calib,
                        "level": level, **score(y, pred)})
    if level != "view_ensemble":
        return
    rounded = np.clip(np.rint(pred), 0, MAX_COUNT)
    for c in np.unique(y):
        m = y == c
        PER_COUNT_ROWS.append({"dataset": dataset, "model": model, "task": task, "calib": calib,
                               "count": int(c), "n": int(m.sum()),
                               "recall": float(np.mean(rounded[m] == c)),
                               "mean_pred": float(np.mean(pred[m]))})


def evaluate(dataset, model, task, calib, test_df, pred):
    """test_df: one row per (recording, view, window); pred aligned with it."""
    d = test_df[["recording", "window", "count"]].copy()
    d["pred"] = pred
    record(dataset, model, task, calib, "single_view", d["count"].values, d["pred"].values)

    ens = d.groupby(["recording", "window"]).agg(count=("count", "first"), pred=("pred", "mean"))
    record(dataset, model, task, calib, "view_ensemble", ens["count"].values, ens["pred"].values)

    rec = ens.groupby("recording").agg(count=("count", "first"), pred=("pred", "median"))
    record(dataset, model, task, calib, "recording", rec["count"].values, rec["pred"].values)
    for name, r in rec.iterrows():
        RECORDING_ROWS.append({"dataset": dataset, "model": model, "task": task, "calib": calib,
                               "recording": name, "count": int(r["count"]),
                               "pred": round(float(r["pred"]), 2)})'''


P2_GBM = r'''# 3. LightGBM on the count cues (leave one group out) + mean baseline
GBM_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  random_state=SEED, verbose=-1)


def calibrate(df):
    """z-score every feature within each (group, view) using that group's own unlabelled windows."""
    out = df.copy()
    g = out.groupby(["group", "view"])[FEATURES]
    out[FEATURES] = (out[FEATURES] - g.transform("mean")) / (g.transform("std") + 1e-6)
    return out


IMPORTANCE = []
t0 = time.time()
for dataset in DATASETS:
    data = feat[feat["dataset"] == dataset].reset_index(drop=True)
    groups = sorted(data["group"].unique())

    # WiCount: one group = one recording = one count, so calibration would erase the label
    calibs = ["none"] if dataset == "wicount" else ["none", "group_z"]

    for calib in calibs:
        d = calibrate(data) if calib == "group_z" else data
        preds = {"clf": np.zeros(len(d)), "reg": np.zeros(len(d)), "mean": np.zeros(len(d))}

        for g in groups:
            test = (d["group"] == g).values
            X_tr, y_tr = d.loc[~test, FEATURES], d.loc[~test, "count"]
            X_te = d.loc[test, FEATURES]

            clf = lgb.LGBMClassifier(**GBM_PARAMS).fit(X_tr, y_tr)
            preds["clf"][test] = clf.predict(X_te)

            reg = lgb.LGBMRegressor(**GBM_PARAMS).fit(X_tr, y_tr)
            preds["reg"][test] = reg.predict(X_te)

            preds["mean"][test] = y_tr.mean()
            IMPORTANCE.append(pd.Series(reg.booster_.feature_importance("gain"), index=FEATURES,
                                        name=f"{dataset}|{calib}|{g}"))

        evaluate(dataset, "gbm", "clf", calib, d, preds["clf"])
        evaluate(dataset, "gbm", "reg", calib, d, preds["reg"])
        if calib == "none":
            evaluate(dataset, "mean_baseline", "reg", "none", d, preds["mean"])
        print(f"{dataset:8s} calib={calib:8s} done  [{time.time() - t0:.0f} s]")

gbm_table = pd.DataFrame(RESULT_ROWS)
gbm_table[gbm_table["level"] == "view_ensemble"].round(3)'''


P2_CNN = r'''# 4. Tiny 1D-CNN on the raw windows (leave one group out)
EPOCHS = 12
BATCH = 256
SAMPLES_PER_EPOCH = 20000       # random subset of training windows per epoch (keeps CPU runs short)
LR = 2e-3


class TinyCNN(nn.Module):
    """Input (batch, 52 tones, 250 time bins). About 35k parameters."""

    def __init__(self, n_tones=52, n_classes=MAX_COUNT + 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(n_tones, 32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 48, kernel_size=5, padding=4, dilation=2), nn.BatchNorm1d(48), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(48, 64, kernel_size=5, padding=4, dilation=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.2),
        )
        self.cls_head = nn.Linear(64, n_classes)
        self.reg_head = nn.Linear(64, 1)

    def forward(self, x):
        h = self.body(x)
        return self.cls_head(h), self.reg_head(h).squeeze(1)


def load_dataset_series(dataset):
    """All recordings of one dataset in RAM, plus an index of (recording, view, start, ...)."""
    series, index = {}, []
    for path in sorted(glob.glob(os.path.join(SHARED, dataset, "*.npz"))):
        z = np.load(path)
        meta = json.loads(str(z["meta"]))
        name = meta["recording"]
        series[name] = z["series"]                               # (views, bins, 52) float16
        for v in range(z["series"].shape[0]):
            for j, a in enumerate(z["starts"]):
                index.append((name, v, int(a), j, meta["group"], meta["count"]))
    index = pd.DataFrame(index, columns=["recording", "view", "start", "window", "group", "count"])
    return series, index


def make_batch(series, rows):
    x = np.stack([series[r.recording][r.view, r.start:r.start + WIN] for r in rows.itertuples()])
    x = np.nan_to_num(x.astype(np.float32)).clip(-3, 3)
    return torch.from_numpy(x).transpose(1, 2).to(DEVICE)       # (batch, 52, 250)


def train_and_predict(series, train_idx, test_idx):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = TinyCNN().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    # classes never seen in training are masked out at prediction time
    seen = torch.zeros(MAX_COUNT + 1, dtype=torch.bool, device=DEVICE)
    seen[torch.tensor(sorted(train_idx["count"].unique()), device=DEVICE)] = True

    for epoch in range(EPOCHS):
        model.train()
        n = min(SAMPLES_PER_EPOCH, len(train_idx))
        epoch_rows = train_idx.iloc[rng.permutation(len(train_idx))[:n]]
        for b in range(0, n, BATCH):
            rows = epoch_rows.iloc[b:b + BATCH]
            y = torch.tensor(rows["count"].values, device=DEVICE)
            logits, reg = model(make_batch(series, rows))
            loss = Fnn.cross_entropy(logits, y) + Fnn.huber_loss(reg, y.float())
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    p_cls, p_reg = [], []
    with torch.no_grad():
        for b in range(0, len(test_idx), 1024):
            logits, reg = model(make_batch(series, test_idx.iloc[b:b + 1024]))
            logits = logits.masked_fill(~seen, -1e9)
            p_cls.append(logits.argmax(1).cpu().numpy())
            p_reg.append(reg.cpu().numpy())
    return np.concatenate(p_cls), np.concatenate(p_reg)


t0 = time.time()
for dataset in DATASETS:
    series, index = load_dataset_series(dataset)
    pred_cls = np.zeros(len(index))
    pred_reg = np.zeros(len(index))
    for g in sorted(index["group"].unique()):
        test = (index["group"] == g).values
        pred_cls[test], pred_reg[test] = train_and_predict(series, index[~test], index[test])
        print(f"{dataset:8s} left out {g:28s} [{time.time() - t0:.0f} s]")
    evaluate(dataset, "cnn", "clf", "none", index, pred_cls)
    evaluate(dataset, "cnn", "reg", "none", index, pred_reg)
    del series

pd.DataFrame(RESULT_ROWS).query("model == 'cnn' and level == 'view_ensemble'").round(3)'''


P2_SAVE = r'''# 5. Save all results + summary tables
results = pd.DataFrame(RESULT_ROWS)
per_count = pd.DataFrame(PER_COUNT_ROWS)
recordings = pd.DataFrame(RECORDING_ROWS)
importance = pd.concat(IMPORTANCE, axis=1)

results.to_csv(os.path.join(RESULTS, "results.csv"), index=False)
per_count.to_csv(os.path.join(RESULTS, "per_count_recall.csv"), index=False)
recordings.to_csv(os.path.join(RESULTS, "recording_predictions.csv"), index=False)
importance.to_csv(os.path.join(RESULTS, "gbm_feature_importance.csv"))

summary = (results.pivot_table(index=["dataset", "model", "task", "calib"], columns="level",
                               values=["mae", "within1", "exact"])
           .round(3))
summary.to_csv(os.path.join(RESULTS, "summary.csv"))
summary'''


P2_PLOTS = r'''# 6. Plots: per-count recall and recording-level predictions
fig_dir = os.path.join(RESULTS, "figures")
os.makedirs(fig_dir, exist_ok=True)

# per-count recall (view ensemble) for every model of every dataset
fig, axes = plt.subplots(1, 3, figsize=(17, 4))
for ax, dataset in zip(axes, DATASETS):
    d = per_count[per_count["dataset"] == dataset]
    for (model, task, calib), h in d.groupby(["model", "task", "calib"]):
        ax.plot(h["count"], h["recall"], marker="o", label=f"{model}/{task}/{calib}")
    ax.set_title(f"{dataset}: per-count recall (view ensemble)")
    ax.set_xlabel("people")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6)
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, "per_count_recall.png"), dpi=120)
plt.show()

# recording-level: true vs predicted count (regression models)
fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
for ax, dataset in zip(axes, DATASETS):
    d = recordings[(recordings["dataset"] == dataset) & (recordings["task"] == "reg")]
    for (model, calib), h in d.groupby(["model", "calib"]):
        ax.scatter(h["count"], h["pred"], s=18, label=f"{model}/{calib}", alpha=0.8)
    lim = [0, MAX_COUNT]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_title(f"{dataset}: whole-recording estimate")
    ax.set_xlabel("true people")
    ax.set_ylabel("predicted (median over recording)")
    ax.legend(fontsize=6)
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, "recording_predictions.png"), dpi=120)
plt.show()

# mean feature importance per dataset
imp = importance.copy()
imp.columns = [c.split("|")[0] for c in imp.columns]
imp = imp.T.groupby(level=0).mean().T
(imp / imp.sum()).round(3)'''


P2_REPORT = r'''# 7. Short markdown report on Drive (results/phase2/PHASE2_RESULTS.md)
ens = results[results["level"] == "view_ensemble"].copy()
best = ens.sort_values("mae").groupby("dataset").head(1)

lines = ["# Phase 2 results - honest held-out baselines", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} from shared_format ({FORMAT['win_s']} s windows, "
         f"{FORMAT['fs_hz']} Hz, {FORMAT['n_tones']} tones).", "",
         "## Best model per dataset (view ensemble)", "",
         best[["dataset", "model", "task", "calib", "mae", "exact", "within1", "n"]]
         .round(3).to_markdown(index=False), "",
         "## All models (view ensemble)", "",
         ens[["dataset", "model", "task", "calib", "mae", "exact", "within1"]]
         .round(3).to_markdown(index=False), "",
         "## Single view (what one ESP32 link sees)", "",
         results[results["level"] == "single_view"][["dataset", "model", "task", "calib", "mae", "within1"]]
         .round(3).to_markdown(index=False), "",
         "## Caveats", "",
         "- mat80 has no 1-person recordings; PC4a_n00 looked occupied in the deep audit.",
         "- countfi: each count is one time block per scenario, so results measure scenario transfer only.",
         "- wicount: counts 0 and 3 have one file each; a left-out file of those counts was never seen in training.",
         "- group_z calibration uses all unlabelled windows of the test group (slightly optimistic)."]

with open(os.path.join(RESULTS, "PHASE2_RESULTS.md"), "w") as f:
    f.write(chr(10).join(lines))
print("saved", os.path.join(RESULTS, "PHASE2_RESULTS.md"))
best'''


P2_FLUSH = r'''# 8. Make sure results are uploaded to Drive before closing the runtime
drive.flush_and_unmount()
print("flushed - safe to close the runtime")'''


# ===============================================================
# Notebook writer
# ===============================================================
NOTEBOOKS = {
    "phase1_shared_format_v2.ipynb": [
        ("md", P1_INTRO), ("code", P1_SETUP), ("code", P1_COPY), ("code", P1_HELPERS),
        ("code", P1_MAT80), ("code", P1_COUNTFI), ("code", P1_WICOUNT),
        ("code", P1_TABLE), ("code", P1_CHECK), ("code", P1_FLUSH),
    ],
    "phase2_baselines_v2.ipynb": [
        ("md", P2_INTRO), ("code", P2_SETUP), ("code", P2_METRICS), ("code", P2_GBM),
        ("code", P2_CNN), ("code", P2_SAVE), ("code", P2_PLOTS), ("code", P2_REPORT),
        ("code", P2_FLUSH),
    ],
}


def to_cell(kind, text):
    lines = text.strip("\n").split("\n")
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": source}
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": source}


def main():
    for filename, cells in NOTEBOOKS.items():
        for kind, text in cells:
            if kind == "code":
                ast.parse(text)                      # syntax check every code cell

        nb = {"cells": [to_cell(k, t) for k, t in cells],
              "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                          "name": "python3"},
                           "language_info": {"name": "python"},
                           "colab": {"provenance": []}},
              "nbformat": 4, "nbformat_minor": 5}

        out = os.path.join(ROOT, filename)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print("wrote", out, f"({len(cells)} cells)")


if __name__ == "__main__":
    main()

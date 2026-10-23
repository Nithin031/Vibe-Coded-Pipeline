"""Generate LAPTOP versions of the Phase 1 + Phase 2 notebooks (no Colab, no Google Drive).

  phase1_shared_format_local.ipynb -> reads D:/wifi/Datasets directly, writes D:/wifi/shared_format
  phase2_baselines_local.ipynb     -> reads D:/wifi/shared_format, writes D:/wifi/results/phase2_baselines__mat80_countfi_wicount
                                      (CNN uses the laptop NVIDIA GPU if PyTorch sees it, else the CPU)

The processing code is shared with build_phase12_nbs.py, so results match the Colab versions.
Run:  py -3 colab_tools/build_local_nbs.py
"""

import ast
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from build_phase12_nbs import (  # noqa: E402  (same cells as the Colab notebooks)
    P1_CHECK, P1_COUNTFI, P1_HELPERS, P1_MAT80, P1_TABLE, P1_WICOUNT,
    P2_CNN, P2_GBM, P2_METRICS, P2_PLOTS, P2_REPORT, P2_SAVE,
    to_cell,
)


# ===============================================================
# PHASE 1 - shared format (laptop)
# ===============================================================
L1_INTRO = r'''# Phase 1 - one shared input format (laptop version)

Runs on the laptop, straight from `D:/wifi/Datasets` - no Colab, no Google Drive, no uploads.

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

```
D:/wifi/Datasets/Dataset/PC-1a ... PC-4a/*.mat             (80 MHz set, 5.6 GB)
D:/wifi/Datasets/CountFi/48/48_<rx>_<scenario>/<count>/*.pcap   (4.9 GB)
D:/wifi/Datasets/WiCount/data/<count>/*.csv                (41 MB)
```

**Needs:** `numpy pandas pyarrow matplotlib` (CPU only - this phase does not use the GPU).

Output: `D:/wifi/shared_format/<dataset>/<recording>.npz` + `features.parquet`.
The run is resumable: recordings already saved are skipped, so you can stop and restart any time.'''


L1_SETUP = r'''# 1. Paths and settings (laptop)
import glob
import json
import os
import shutil
import struct
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

WIFI_ROOT = "D:/wifi"
RAW = f"{WIFI_ROOT}/Datasets"                   # the three dataset folders
OUT_ROOT = f"{WIFI_ROOT}/shared_format"         # Phase 1 output

SOURCES = {
    "mat80": "Dataset",
    "countfi": "CountFi",
    "wicount": "WiCount",
}

# ---- shared-format settings (keep identical on the ESP32 later) ----
FS = 50              # common sample rate (Hz)
WIN_S = 5.0          # window length (s)
HOP_S = 2.5          # window hop (s)
MAX_GAP_BINS = 3     # gaps up to 3 bins (60 ms) are interpolated
MAX_MISSING = 0.20   # drop windows with more missing bins than this
N_TONES = 52

# ---- Check the data is really there before processing ----
counts = {
    "mat80": len(glob.glob(os.path.join(RAW, "Dataset", "PC-*", "*.mat"))),
    "countfi": len(glob.glob(os.path.join(RAW, "CountFi", "**", "*.pcap"), recursive=True)),
    "wicount": len(glob.glob(os.path.join(RAW, "WiCount", "data", "*", "*.csv"))),
}
for key, n in counts.items():
    print(f"{key:8s} {os.path.join(RAW, SOURCES[key]):35s} {n:6d} files")

empty = [key for key, n in counts.items() if n == 0]
if empty:
    raise FileNotFoundError(f"No data files found for {empty} under {RAW}. Check the folder names.")

os.makedirs(OUT_ROOT, exist_ok=True)
free_gb = shutil.disk_usage(WIFI_ROOT).free / 1e9
print(f"output: {OUT_ROOT}   free on D: {free_gb:.1f} GB")'''


L1_DONE = r'''# 9. Done
print("Phase 1 finished. Output in", OUT_ROOT)
print("Next: open phase2_baselines_local.ipynb")'''


# ===============================================================
# PHASE 2 - baselines (laptop GPU / CPU)
# ===============================================================
L2_INTRO = r'''# Phase 2 - honest baselines per dataset (laptop version)

Reads the Phase 1 output in `D:/wifi/shared_format` (**run `phase1_shared_format_local.ipynb` first**).
The tiny CNN uses the laptop **NVIDIA GPU** if PyTorch can see it, otherwise it runs on the CPU (slower, same results).

Every model is tested only on **groups it never saw in training**:

| Dataset | Split unit (left out one at a time) | Notes |
|---|---|---|
| mat80 | setup PC-1a ... PC-4a | counts 0, 2-10 (no 1-person class) |
| countfi | scenario (head + end together) | count = one time block per scenario; expect weak results (stress test) |
| wicount | recording (15 files) | counts 0 and 3 have one file each, so a classifier can never be right on them |

**Models**
- **LightGBM** on the 10 count cues: classifier and regressor, with and without **per-group calibration**.
- **Tiny 1D-CNN** (~35k parameters, fits the ESP32-S3 budget) on the raw 5 s x 52-tone windows.
- **Mean baseline**: always predict the training-set mean.

**Levels:** `single_view` = one antenna/sub-band; `view_ensemble` = average over the views of the same moment;
`recording` = median over a whole recording.

**Needs:** `numpy pandas pyarrow matplotlib lightgbm tabulate torch`.
For the GPU, install the CUDA build of PyTorch (pytorch.org -> Get Started -> Windows / Pip / CUDA).
Cell 1 prints whether the GPU is used.

Results go to `D:/wifi/results/phase2_baselines__mat80_countfi_wicount/`.'''


L2_SETUP = r'''# 1. Paths, device, load Phase 1 output (laptop)
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

WIFI_ROOT = "D:/wifi"
SHARED = f"{WIFI_ROOT}/shared_format"           # Phase 1 output
RESULTS = f"{WIFI_ROOT}/results/phase2_baselines__mat80_countfi_wicount"
os.makedirs(RESULTS, exist_ok=True)

# ---- Phase 1 must have run first ----
for needed in ["format.json", "features.parquet"]:
    if not os.path.exists(os.path.join(SHARED, needed)):
        raise FileNotFoundError(
            f"{needed} not found in {SHARED}. Run phase1_shared_format_local.ipynb first."
        )

FORMAT = json.load(open(os.path.join(SHARED, "format.json")))
FEATURES = FORMAT["feature_names"]
FS, WIN = FORMAT["fs_hz"], int(FORMAT["win_s"] * FORMAT["fs_hz"])

feat = pd.read_parquet(os.path.join(SHARED, "features.parquet"))
feat[FEATURES] = feat[FEATURES].fillna(0.0)

DATASETS = ["mat80", "countfi", "wicount"]
MAX_COUNT = int(feat["count"].max())
SEED = 0

# ---- GPU if available, else CPU ----
if torch.cuda.is_available():
    DEVICE = "cuda"
    torch.backends.cudnn.benchmark = True
    print("device: GPU -", torch.cuda.get_device_name(0))
else:
    DEVICE = "cpu"
    torch.set_num_threads(os.cpu_count() or 4)
    print("device: CPU (no CUDA GPU visible to PyTorch - install the CUDA build of torch to use the GPU)")

print(feat.groupby("dataset").agg(recordings=("recording", "nunique"), groups=("group", "nunique"),
                                  views=("view", "nunique"), rows=("window", "size")))'''


L2_DONE = r'''# 8. Done
print("Phase 2 finished. Results in", RESULTS)'''


# ===============================================================
# Notebook writer
# ===============================================================
NOTEBOOKS = {
    "phase1_shared_format_local.ipynb": [
        ("md", L1_INTRO), ("code", L1_SETUP), ("code", P1_HELPERS),
        ("code", P1_MAT80), ("code", P1_COUNTFI), ("code", P1_WICOUNT),
        ("code", P1_TABLE), ("code", P1_CHECK), ("code", L1_DONE),
    ],
    "phase2_baselines_local.ipynb": [
        ("md", L2_INTRO), ("code", L2_SETUP), ("code", P2_METRICS), ("code", P2_GBM),
        ("code", P2_CNN), ("code", P2_SAVE), ("code", P2_PLOTS), ("code", P2_REPORT),
        ("code", L2_DONE),
    ],
}


def main():
    for filename, cells in NOTEBOOKS.items():
        for kind, text in cells:
            if kind == "code":
                ast.parse(text)                      # syntax check every code cell
                assert "google.colab" not in text and "/content" not in text, filename

        nb = {"cells": [to_cell(k, t) for k, t in cells],
              "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                          "name": "python3"},
                           "language_info": {"name": "python"}},
              "nbformat": 4, "nbformat_minor": 5}

        out = os.path.join(ROOT, filename)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print("wrote", out, f"({len(cells)} cells)")


if __name__ == "__main__":
    main()

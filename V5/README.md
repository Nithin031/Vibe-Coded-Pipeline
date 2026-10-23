# Wi-Fi CSI people counting on the ESP32-S3 (V5)

Count the people in a room from the Wi-Fi channel state information (CSI) of **one ESP32-S3 link**,
with a model small enough to run next to the radio.
V5 is the snapshot of Phases 1-5 (2026-10-23): the code, every notebook that was run, all results, the final model,
and a benchmark against V4 ([results](results/README.md#v4-vs-v5-benchmark-kaggle-2026-10-23)).

## Final model in one look

| | |
|---|---|
| Model | LightGBM regression, 300 trees, on 10 window cues - ~77 KB flash, ~75 KB RAM, ~3 M operations every 2.5 s |
| Trained on | WiCount (ESP32-S3) + 80 MHz people-counting set (router) + WiMANS 5 GHz (Intel 5300) |
| Install | 20 s empty-room calibration (required) |
| Honest output | **empty / 1 / 2 / 3+**, updated every 2.5 s (median of the last 30 s) |
| Held-out error | ESP32 (WiCount) MAE 0.29, 100 % within ±1 · WiMANS unseen room MAE 1.01 · empty found 93-100 % |
| Files | [`deploy/`](deploy/) - the only copy of the model |
| How to deploy | [deployment guide](deploy/README.md) - 2 or 3 ESP32-S3, input format, calibration, Python + C |

Above 2-3 people the estimates bunch together (one link measures total motion, which stops growing).
Full numbers: [`results/README.md`](results/README.md) and
[`PHASE4_RESULTS.md`](results/phase4_final__mat80_wicount_wimans_csibench/PHASE4_RESULTS.md).

## Repository layout

```
V5/
├── README.md                 <- this file
├── deploy/                   THE MODEL (V5 final) + everything to run it - self-contained
│   ├── infer.py              run on a recorded ESP32 CSV (calibrate, count, score)
│   ├── src/                  data.py, calibration.py, metrics.py, models.py
│   ├── tests/                test_deploy.py (src == esp32_counter, files consistent, infer end to end)
│   └── trees.json, counter_model.h, model.txt, model_card.json, README.md (deployment guide)
├── esp32_counter/            the counter: preprocessing, 10 cues, tree export (Python + C header), streaming API
│   ├── esp32_counter.py
│   └── test_esp32_counter.py stream == batch, no future leakage, C header == Python, budget
├── notebooks/
│   ├── local/                run on the laptop (outputs saved inside)
│   │   ├── phase1_shared_format_local.ipynb   raw data -> shared 52-tone, 50 Hz format
│   │   ├── phase2_baselines_local.ipynb       first held-out baselines
│   │   ├── phase3_esp32_final_local.ipynb     device pipeline, one link, Phase 3 model
│   │   └── phase4_final_local.ipynb           all sources pooled, final model
│   └── kaggle/               run as Kaggle kernels
│       ├── kaggle_wimans.ipynb                WiMANS audit + counting (v3)
│       ├── kaggle_csibench.ipynb              CSI-Bench audit + Localization counting (v3)
│       ├── kaggle_phase4_features.ipynb       WiMANS + CSI-Bench -> cue tables for Phase 4
│       ├── kaggle_phase5_features.ipynb       all 5 datasets -> base + extra cue tables (~3 h)
│       ├── kaggle_phase5_experiments.ipynb    Phase 5 upgrade tests
│       ├── kaggle_v4_vs_v5_benchmark.ipynb    V4 (its own GitHub code) vs V5
│       └── kaggle_ehunam_explore.ipynb        EHUNAM discovery (not used for counting yet)
├── notebook_builders/        Python scripts that generate the notebooks (edit these, not the .ipynb)
├── kaggle_kernels/           kernel-metadata.json per Kaggle kernel (for `kaggle kernels push`)
├── results/                  all results: phase2-5, phase4 = how the final model was chosen, benchmark_v4_vs_v5, kaggle/
└── reports/                  dataset audit, deep audit (80 MHz + CountFi), audit plan, dataset survey
```

## Phases

| Phase | What | Where it ran | Result |
|---|---|---|---|
| 0 | Dataset audit: which public sets really have people counts | laptop | `reports/` |
| 1 | One shared format: 52 tones, 50 Hz, amplitude, one link = one "view" | laptop | intermediate data (not in repo) |
| 2 | First held-out baselines (GBM, tiny CNN) | laptop | GBM wins; CNN dropped |
| 2k | WiMANS + CSI-Bench counting | Kaggle | 5 GHz only; 3+ people blur; CSI-Bench 1 vs 2 not separable |
| 3 | Device-ready pipeline (causal, one link, 30 s median), C export, replay | laptop | ESP32 MAE 0.31 |
| 4 | Pool every labelled source, leave one room out, pick + export final model | Kaggle (features) + laptop | **final model**, ESP32 MAE 0.29 |
| 5 | Extra cues, 4-class output, 30 s memory, two-point calibration, 2-3 link averaging | Kaggle | none adopted; Phase 4 model stays |
| - | V4 vs V5 benchmark: V4's own code, same packets, splits and noise suite | Kaggle | V5 best on clean new sessions (0.85 vs 0.78); V4 gap-CNN best under noise (0.73 vs 0.50) |

## Data (not in the repo)

| Name | Source | Hardware | People |
|---|---|---|---|
| mat80 | 80 MHz people-counting set (Meneghello et al.), folders PC-1a ... PC-4a | router + Nexmon, 4 antennas | 0, 2-10 |
| WiCount | [CSI-BERT2 / WiCount](https://github.com/RS2002/CSI-BERT2/tree/main/WiCount) | ESP32-S3 | 0-3 |
| CountFi | April-2025 subset | Nexmon, 80 MHz | 1-10 (stress test only) |
| WiMANS | Kaggle `shuokanghuang/wimans` | Intel 5300 | 0-5 |
| CSI-Bench | Kaggle `guozhenjennzhu/csi-bench` | several devices | Localization: 1-2 |

The laptop notebooks read `D:/wifi/Datasets` and `D:/wifi/shared_format` (paths are set in each notebook's first cell).
mat80, CountFi and WiCount are also in a private Kaggle dataset (`wizard1803/wifi-csi`) for cloud runs.
Check each dataset's licence before redistributing it.

## Reproduce

```bash
pip install numpy pandas pyarrow scipy matplotlib lightgbm tabulate
python -m unittest esp32_counter/test_esp32_counter.py      # run from the repo root
python notebook_builders/build_phase4_nbs.py                 # regenerate notebooks (written next to the builders' parent)
```

Run order: Phase 1 -> Phase 3 (builds the causal feature cache) -> Kaggle `kaggle_phase4_features` -> Phase 4.

## Known limits

- Only one room of real ESP32-S3 data (WiCount, one evening, 0-3 people).
- Counts above 3 come from router / Intel 5300 data and saturate.
- WiMANS / CSI-Bench clips (3 s / 5 s) use clip-mean normalisation instead of the device's 30 s causal baseline.
- The preprocessing and the 10 cues are in Python only; the C header has the trees and input preparation.
- Not robust to noise: white noise, impulses, fans and dead subcarriers break it (V4's gap-CNN copes much better).
- Next: robust preprocessing + noise-augmented training, then recordings on our own ESP32-S3 boards.

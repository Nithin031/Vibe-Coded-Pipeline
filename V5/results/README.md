# Wi-Fi CSI people counting on ESP32-S3: results

Goal: count the people in a room from the Wi-Fi channel state information (CSI) of **one ESP32-S3 link**.
Last updated 2026-10-23 (Phase 4 model; Phase 5 experiments and V4 vs V5 benchmark).

```
results/
├── README.md                                     <- this summary
├── phase2_baselines__mat80_countfi_wicount/      Phase 2: first held-out baselines (GBM + tiny CNN)
├── kaggle/                                       Kaggle runs: WiMANS v3, CSI-Bench v3 (+ audit), Phase 4a log
├── phase3_esp32_final__mat80_wicount_countfi/    Phase 3: device-ready pipeline (superseded)
└── phase4_final__mat80_wicount_wimans_csibench/  Phase 4: all labelled sources pooled -> FINAL MODEL
    ├── PHASE4_RESULTS.md                         full tables
    ├── model_esp32_counter/                      <- the model to deploy
    ├── figures/
    └── *.csv
├── phase5_experiments/                           Phase 5 (Kaggle): upgrades tested, none adopted
└── benchmark_v4_vs_v5/                           V4 (GitHub v4 branch, its own code) vs V5, same data + noise
```

## Datasets

| Short name | Dataset | Hardware | People | Recordings | Used as |
|---|---|---|---|---|---|
| `mat80` | 80 MHz people-counting set (Meneghello et al.), `Datasets/Dataset/PC-1a ... PC-4a` | Router + Nexmon, 4 antennas, 802.11ac 80 MHz, 173 packets/s | 0, 2-10 (no 1) | 4 setups (PC1a-PC4a), 39 recordings | training + test on unseen setups |
| `wicount` | WiCount, `Datasets/WiCount` | **ESP32-S3**, 1 antenna, 20 MHz, ~87 packets/s | 0-3 | 15 files, ~1 min each, one room, 2023-12-21 | the only ESP32 data: training + test |
| `wimans` | WiMANS, 5 GHz only (Kaggle) | Intel 5300, 3x3 links, ~1 kHz | 0-5 | 3 rooms, 5,643 clips of 3 s | training + test on unseen rooms |
| `csibench` | CSI-Bench Localization (Kaggle) | 3 devices, 100 Hz | 1-2 | 4 rooms, 6,731 clips of 5 s | test (no empty recording) |
| `countfi` | CountFi (Apr-2025 subset), `Datasets/CountFi` | Nexmon, 80 MHz | 1-10 | 6 scenarios x 2 receivers | stress test only (each count = one time block) |

To match the ESP32, the 80 MHz data is cut into four 20 MHz sub-bands with 52 tones each. Every antenna x sub-band
is treated as one "view", which stands in for one ESP32 link. Only amplitude is used.
Intermediate data (`shared_format/`, Phase 1 output + cached Phase 3 features) is not in the repo - regenerate it with Phase 1.

## Phase 2: first baselines (16 views averaged, look-ahead normalisation)

| Dataset | Best model | MAE | Within ±1 |
|---|---|---|---|
| mat80 | GBM + per-setup calibration | 1.11 (whole recording 0.71) | 0.70 (0.87) |
| mat80, one view | GBM regression + calibration | 1.42 | 0.60 |
| wicount | GBM | 0.32 | 1.00 |
| countfi | any | ~2.2 (= mean guess) | ~0.4 |

The CNN was worse than GBM on one view and collapsed on WiCount, so it was dropped.

## Phase 3: device-ready, one link, no look-ahead

Pipeline: per-packet gain normalisation -> 50 Hz bins -> causal 30 s baseline per tone -> 5 s windows every 2.5 s ->
10 features -> GBM regression -> output = median of the last 30 s.

| Test | Train | MAE | Exact | Within ±1 |
|---|---|---|---|---|
| WiCount (ESP32) held-out files | mat80 + other WiCount files, empty-room cal. (**final**) | **0.31** | 0.81 | 1.00 |
| WiCount | mat80 + other WiCount files, no calibration | 0.36 | 0.70 | 0.99 |
| WiCount | WiCount only | 0.34 | 0.82 | 1.00 |
| WiCount | mat80 only (router -> ESP32) | 0.77 | 0.37 | 0.89 |
| mat80, unseen setup | other mat80 setups, no calibration | 1.74 | 0.21 | 0.50 |
| mat80, unseen setup | other mat80 setups, empty-room cal. | 2.85 | 0.19 | 0.35 |
| CountFi | mat80 | 4.65 | 0.04 | 0.15 |

**Findings**
- On ESP32 data (0-3 people, one room) the counter is good: about 0.3 MAE and always within ±1.
- On unseen mat80 rooms, the estimates rise in the right order with the count (rank correlation 0.92) but the scale is squeezed:
  empty ≈ 0, 2 people -> about 4, 10 people -> about 6.5.
- The router data barely transfers to the ESP32 (rank correlation 0.34) and adds only 0.03 MAE gain on WiCount.
- Empty-room calibration helps a little on WiCount but hurts badly on mat80. **Recommended: switch the final model
  to no calibration** (rerun cells 7-11 of `phase3_esp32_final_local.ipynb` with that candidate).
- Device check: streaming the raw WiCount CSVs packet by packet gives the same numbers as training (largest difference 0.03 people).

## Phase 4: final model, all labelled sources (2026-10-23)

Same device pipeline as Phase 3. WiMANS + CSI-Bench features come from the Kaggle kernel `wifi-phase4-features`
(clip-mean normalisation, because their clips are too short for the 30 s causal baseline).
Leave one room out for 5 training pools x 2 modes; scores are held-out, smoothed MAE.

| Model | WiCount (ESP32) | WiMANS unseen room | mat80 unseen setup | Install step |
|---|---|---|---|---|
| **Final: ESP32 + mat80 + WiMANS, empty-room cal.** | **0.29** | **1.01** | 2.67 | 20 s empty room |
| Phase 3 model (ESP32 + mat80, empty cal.) | 0.31 | 1.45 | 2.76 | 20 s empty room |
| ESP32 + mat80, no calibration | 0.36 | 1.62 | 1.65 | none |
| ESP32 + mat80 + WiMANS, no calibration | 0.92 | 1.06 | 1.70 | none |

**Findings**
- WiMANS helps: unseen-room WiMANS error drops from 1.45 to 1.01, and WiCount is as good as before.
- Without calibration, adding WiMANS breaks the ESP32 scale (WiCount 0.36 -> 0.92). With the 20 s empty-room step it is fine.
- CSI-Bench adds nothing: no model separates 1 from 2 people there, and it has no empty recording, so the calibrated model cannot train on it.
- Strong saturation. Mean estimate per true count, final model: WiCount 3 -> 2.0; WiMANS 1/2/3/4/5 -> 2.0/2.6/2.8/3.0/3.1;
  mat80 2...10 -> 4.1...5.1. Empty is found 93-100 % of the time everywhere.
- Clip check: 5-s clip cues ≈ device cues (0.2 people difference), 3-s WiMANS clips 0.5 people.
- Replay of the raw WiCount CSVs through the bundle = batch numbers exactly; ESP32 budget unchanged (77 KB flash, 75 KB RAM).
- Honest output: **empty / 1 / 2 / 3+**, after a 20 s empty-room calibration, in rooms like the training rooms.

## Phase 5: upgrades tested (Kaggle, 2026-10-23) - none adopted

Leave one site out on all 5 datasets, same rule for every upgrade: it helps if MAE drops >= 0.05 on >= 2 datasets
and WiCount gets no worse than +0.05. With the install calibration (the deployed mode) **no upgrade passes**.

| Upgrade | Effect with empty-room calibration | Verdict |
|---|---|---|
| 9 extra cues | MAE mat80 -0.25, WiCount -0.03, WiMANS -0.01; empty found less often (mat80 94 -> 86 %) | no |
| 4-class output (0/1/2/3+) | WiCount -0.13, empty found more often (WiMANS 93 -> 98 %); mat80 +0.08 | no, closest - retest on own data |
| 30 s memory cues | mat80 -0.50, WiCount +0.07 | no |
| Two-point calibration (empty + 2 people) | worse on mat80, WiCount, WiMANS | no |
| 2 / 3 links averaged | -0.03 to -0.06 (links = antennas of one device, a lower bound) | small gain; retest with 2 boards on different walls |

Full tables: [`phase5_experiments/PHASE5_RESULTS.md`](phase5_experiments/PHASE5_RESULTS.md).

## V4 vs V5 benchmark (Kaggle, 2026-10-23)

V4 = GitHub `Nithin031/Vibe-Coded-Pipeline` branch `v4`, run with its own code (harness check: V4 LightGBM exact 0.843,
published 0.84). Same WiCount packets, same splits, same noisy packets. Each system keeps its own smoothing.

| Benchmark | V4 LightGBM | V4 gap-CNN | V5 WiCount-only | **V5 final** |
|---|---|---|---|---|
| B1 V4's 3 folds - exact | 0.849 | 0.850 | 0.885 | 0.881 |
| B1 - F1 on 0/1/2/3+ | 0.720 | 0.685 | 0.461 | **0.736** |
| B2 new session (leave one file out) - exact | 0.755 | 0.783 | 0.825 | **0.854** |
| B2 - empty room recognised | 0 % | 0 % | 0 % | **100 %** |
| B2 - 3 people recognised | 0 % | 0 % | 0 % | 0 % |
| B3 noise, 22 conditions - exact (unanswered = wrong) | 0.624 | **0.727** | 0.620 | 0.504 |
| Counting model size | ~1 MB arrays | ~191 KB | 77 KB | 77 KB |
| First answer | ~3 s | ~3 s | 15 s | 15 s |

- **Clean signal, new session: V5 final wins** (exact 0.85 vs 0.76-0.78) and is the only system that still finds the
  empty room when the empty file was never trained on - the empty-room calibration does that.
- **Noise: V4 gap-CNN wins clearly.** V5 final breaks on white noise (exact 0.04 at level 1), impulses (0.09-0.33),
  fans (0.48 at level 3) and dead subcarriers (~0.5). Unchanged by: AGC jumps, jitter, packet loss.
- Nobody counts 3 people on an unseen file. V4 also has a presence model and an 'unknown' answer; V5 does not.

Full tables: [`benchmark_v4_vs_v5/BENCHMARK.md`](benchmark_v4_vs_v5/BENCHMARK.md).

## Final model: `phase4_final__mat80_wicount_wimans_csibench/model_esp32_counter/`

| File | What |
|---|---|
| `counter_model.h` | for the ESP32 firmware: settings + tree arrays + `cc_prepare()` + `cc_predict()` |
| `trees.json` | same model for Python (no LightGBM needed) |
| `model.txt` | LightGBM native model |
| `model_card.json` | settings, calibration steps, held-out results, caveats |
| `README.md` | **deployment guide**: hardware setups (2 or 3 ESP32-S3), input format, calibration, Python + C use |

300 trees, 8,700 nodes, ~77 KB flash, ~75 KB RAM, ~3 M flops per 2.5 s window. Counts 0-10, trained on mat80 + WiCount + WiMANS 5 GHz. **Needs the 20 s empty-room calibration at install.**

```python
import sys; sys.path.insert(0, "esp32_counter")
import esp32_counter as ec
model, card = ec.load_bundle("results/phase4_final__mat80_wicount_wimans_csibench/model_esp32_counter")
counter = ec.StreamingCounter(model)
counter.begin_calibration()                 # required: room empty for the first 30 s
for t, amp52 in packets:                    # amp52 = |CSI| of the 52 data tones
    for r in counter.push(t, amp52):
        if r["status"] == "ok":
            print(r["count"])
```

## Code and notebooks

| Path | What |
|---|---|
| `esp32_counter/esp32_counter.py` | the counter (shared by training and device), model export |
| `esp32_counter/test_esp32_counter.py` | tests: stream == batch, no future leakage, export parity, C compile |
| `notebooks/local/phase1_shared_format_local.ipynb` | Phase 1: raw datasets -> shared 52-tone, 50 Hz format |
| `notebooks/local/phase2_baselines_local.ipynb` | Phase 2 |
| `notebooks/local/phase3_esp32_final_local.ipynb` | Phase 3: tests, checks, experiments, model, export, replay |
| `notebooks/kaggle/kaggle_phase4_features.ipynb` | Phase 4a (Kaggle): WiMANS + CSI-Bench -> feature tables |
| `notebooks/local/phase4_final_local.ipynb` | Phase 4: pool all sources, leave one room out, final model, export |
| `notebooks/kaggle/kaggle_phase5_features.ipynb`, `kaggle_phase5_experiments.ipynb` | Phase 5 (Kaggle): all cue tables, then the upgrade tests |
| `notebooks/kaggle/kaggle_v4_vs_v5_benchmark.ipynb` | V4 vs V5 benchmark (Kaggle, clones V4 from GitHub) |
| `notebook_builders/build_*.py` | generate the notebooks |

## Limits and next steps

- WiCount is one room on one evening; counts 0 and 3 have one file each, and the final model was chosen on these same files.
- No ESP32 data above 3 people. Counts 4-10 are learned from 4-antenna router data only.
- Next: record your own ESP32-S3 data in several rooms and on several days, 0-5+ people, at >= 100 packets/s, and use it as the test set.
- Noise is the main weakness (benchmark B3): white noise, impulses, fans and dead subcarriers. Next fix: V4-style robust
  preprocessing (impulse / outlier clipping, dead-tone masking) and noise-augmented training, then rerun the benchmark.
- Two-point calibration was tested in Phase 5 and does not help.

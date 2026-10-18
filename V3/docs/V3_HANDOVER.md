# V3 project handover

V3 is an ESP32 WiFi CSI research pipeline for detecting room presence and
motion state under room, hardware, and measured WiFi-channel changes.

## Project overview

WiFi Channel State Information (CSI) describes how each wireless subcarrier is
changed by the path between a transmitter and receiver. A person changes the
multipath pattern. Walking usually creates faster temporal changes; a
stationary person may create subtle changes. An empty room is not necessarily
quiet because fans, appliances, access-point traffic, AGC, multipath, and
receiver noise can also move CSI. This is why cross-room false alarms matter.

V3 tasks:

* **Presence:** `empty` versus `occupied`.
* **Motion:** `empty`, `static`, or `moving`.

## Repository map

| Path | Role |
| --- | --- |
| `src/esp32_io.py` | CSI parsers and dataset loaders |
| `src/preprocess.py` | Subcarrier filtering and irregular-to-20-Hz resampling |
| `src/features.py` | LightGBM features and sequence preparation |
| `src/build.py` | Caching, 3-second windows, 1-second hop |
| `src/calibrate_v3.py` | 30-second empty-room calibration |
| `src/experiments_noise.py` | N1-N4 and N1c-N3c evaluation |
| `src/deep.py` | CNN, transformer, training, prediction |
| `src/train_final_v3.py` | Full-pool final model training |
| `src/plots.py` | Original five figures |
| `src/dashboard_plots.py` | Additional dashboard figures |
| `src/build_registry.py` | Metrics, model, checksum, and figure registries |
| `models/` | Checkpoints and model card |
| `results/` | JSON results, figures, metrics, logs |
| `manifests/` | Canonical registries and checksums |
| `docs/` | Reports and handover material |

Start with [PROTOCOL.md](PROTOCOL.md), [FEATURE_INVENTORY.md](FEATURE_INVENTORY.md),
and [protocol_feature_audit.md](protocol_feature_audit.md).

## End-to-end implementation

1. A transmitter ESP32 sends regular WiFi packets.
2. A receiver ESP32 measures CSI and streams records over USB serial.
3. The parser reads timestamp, RSSI, CSI length, metadata, and interleaved I/Q.
4. I/Q becomes amplitude; L-LTF bins are retained and unstable bins removed.
5. Irregular packets are binned at 20 Hz. Collisions are averaged; empty bins
   are linearly interpolated and marked in a gap mask.
6. Three-second windows are emitted every one second.
7. Surviving subcarriers are represented as 32 bins.
8. LightGBM receives engineered features; CNN/transformer receive `[32, 60]`.
9. Optional site calibration uses the first 30 seconds of an empty recording.
10. Smoothed reported predictions use a five-window median.

The deep input is `float16[32, 60]` plus `bool[60]`. The transformer consumes
the gap mask as padding information; the current CNN ignores it. LightGBM
does not receive the sequence or gap mask.

## Hardware and acquisition

The firmware documentation specifies two ESP32 DevKit boards, one transmitter
and one receiver, 802.11n HT20, an approximately 100 Hz transmitter beacon,
and a 460800-baud receiver serial link. The receiver can select a quiet or
congested channel. Firmware and capture documentation are under
`datasets/esp32_csi_pipeline/firmware/` and
`datasets/esp32_csi_pipeline/capture/`.

The full parser supports 104, 128, or 384 interleaved I/Q values and keeps the
L-LTF portion. Preserve raw records. The shorter amplitude-only capture format
documented by the firmware is not sufficient for the full V3 timestamp/RSSI/IQ
contract without an adapter. Live capture has not been validated end to end
with V3 inference.

## Datasets

| Dataset | Current use |
| --- | --- |
| ESPectre | Real homes, held-out-room N1/N1c, chip and room variation |
| `esp32-csi-pipeline` | N2 quiet/congested channel and N3 room transfer |
| WiCount | ESP32-S3 labelled counting reference and N4 source |
| WiGesture/WiFall | Transformer masked pretraining and activity data |
| OTH thesis | ESP-NOW HT40 labelled reference and N4 source |
| OpenCSI | Excluded from V3 scores because links are bursty at about 1 Hz |
| HomeOccupancy | Separate V4 home-domain fine-tuning, not V3 evidence |

See [datasets.json](../manifests/datasets.json).

## Models

V3 contains LightGBM, CNN, and transformer files for both tasks. The six
classifier artifacts were load-tested and hash-verified in
[models.json](../manifests/models.json). The `.pt` files are PyTorch state
dictionaries, not complete model objects. Reconstruct the architecture from
`src/deep.py` and call `load_state_dict()`.

Use only these six V3 runtime classifiers:
`presence_lgbm.txt`, `presence_cnn.pt`, `presence_tf.pt`, `motion_lgbm.txt`,
`motion_cnn.pt`, and `motion_tf.pt`. The additional
`transformer_pretrained.pt` file is initialization-only and has no occupancy or
motion prediction head. V2 and V4 checkpoint files belong to other versions.

Presence uses a two-class head; motion uses a three-class head. LightGBM files
load with `lightgbm.Booster(model_file=...)`. The final files were trained on
the full ESPectre + pipeline pool after held-out N1-N4 evaluation, so they are
deployment artifacts rather than the exact historical evaluation checkpoints.

## Protocols and verified results

* **N1:** train on other ESPectre rooms, test one held-out room.
* **N2:** train on quiet pipeline data, test the same room on a measured
  congested channel.
* **N3:** train on pipeline room 1, test unseen room 2.
* **N4:** train on other ESP32 recordings, test the pipeline setup.
* **N1c/N2c/N3c:** corresponding 30-second calibrated evaluations; reference
  windows are excluded from scoring.

The source JSON files are under `V3/results/`; the canonical flattened table is
[v3_metrics.csv](../results/metrics/v3_metrics.csv). It separates window and
smoothed metrics and derives per-class metrics only from saved confusion
matrices. The verified N3 presence result is approximately 0.47 balanced
accuracy without calibration versus 0.72 with calibration for the ensemble;
empty-room false alarms fall from 100% to about 10%. These are held-out
protocol results, not claims about every final checkpoint.

The exact mapping is: `presence_lgbm.txt` -> `lgbm` presence rows,
`presence_cnn.pt` -> `cnn` presence rows, `presence_tf.pt` -> `tf` presence
rows, and the corresponding `motion_*` files -> motion rows. `ensemble` rows
are probability averages from the evaluation process, not a separate saved
checkpoint. See [MODEL_AND_RESULTS_INDEX.md](MODEL_AND_RESULTS_INDEX.md).

The figure descriptions and checksums are in
[figure_index.json](../results/reports/figure_index.json). The figures show
raw/calibrated presence, false alarms, N2 confusion matrices, N1 room transfer,
fluctuation distributions, N3, N4, model-family, and per-class views. They do
not establish universal accuracy or event-level latency.

## Which files should I use?

Inspect results without training:

```powershell
Get-Content V3/results/metrics/v3_metrics.csv
Get-Content V3/manifests/models.json
Get-Content V3/manifests/experiment_index.json
```

Regenerate figures and indexes:

```powershell
py -3.12 V3/src/plots.py
py -3.12 V3/src/dashboard_plots.py
py -3.12 V3/src/build_registry.py
```

Load a transformer from the repository root:

```python
import sys
import torch
sys.path.insert(0, "V3/src")
import deep

state = torch.load("V3/models/presence_tf.pt", map_location="cpu")
model = deep.make_model("tf", n_classes=2)
model.load_state_dict(state)
```

For motion use `motion_tf.pt` and `n_classes=3`. CNN loading uses
`deep.make_model("cnn", n_classes=2 or 3)`. LightGBM uses
`lgb.Booster(model_file="V3/models/presence_lgbm.txt")`.

Offline smoke prediction, using cached pipeline data:

```powershell
py -3.12 V3/src/show_transformer.py --task presence
py -3.12 V3/src/show_transformer.py --task motion
```

From `V3/src`, the historical evaluation commands are:

```powershell
py -3.12 experiments_noise.py N1 N2 N3 N4
py -3.12 experiments_noise.py N1c N2c N3c
```

They retrain/evaluate and write JSON results and logs. Final model retraining
is `py -3.12 V3/src/train_final_v3.py --epochs 30`; it can overwrite the final
V3 filenames and should only be run intentionally.

ESP32 capture requires ESP-IDF, the two firmware boards, `pyserial`, a
configured receiver COM port, and the capture script. It produces raw capture
data, but a complete live serial-to-model application is not implemented or
validated.

## Environment and limitations

Python 3.12 was used for current validation on Windows. Root requirements
include NumPy, pandas, SciPy, scikit-learn, LightGBM, Matplotlib, ReportLab,
and PyTorch. Capture requires `pyserial>=3.5`. GPU is optional for loading and
inference; ESP-IDF is only needed for hardware flashing.

Known limitations are the CNN's ignored gap mask, interpolated missing values,
an index-alignment risk after filtered calibrated rows, overlapping-window
leakage risk, unmeasured smoothing delay, unvalidated phase/features, and
unvalidated live receiver/Raspberry Pi inference.
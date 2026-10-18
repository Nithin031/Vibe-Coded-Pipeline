# V3 audit and V4 development completion report

## 1. Starting state

The repository already contained V3 source, real-noise N1-N4/N1c-N3c JSON
results, five figures, a V3 model card, and six final model artifacts. The
working tree also contained unrelated user/system changes outside the CSI
project. Git reported `main` behind `origin/main` by five commits. No unrelated
file was reverted, deleted, or reorganized.

## 2. Preserved and organized

V3 artifacts were left in their original paths to avoid breaking imports and
historical references. Additive organization now exists through:

* `V3/README.md`
* `V3/docs/V3_REPORT.md`
* `V3/docs/PROTOCOL.md`
* `V3/docs/FEATURE_INVENTORY.md`
* `V3/docs/LIMITATIONS.md`
* `V3/docs/protocol_feature_audit.md`
* `V3/manifests/models.json`
* `V3/manifests/experiment_index.json`
* `V3/manifests/datasets.json`
* `V3/manifests/checksums.json`
* `V3/results/metrics/v3_metrics.csv`
* `V3/results/reports/figure_index.json`
* `V3/results/reports/cleanup_manifest.json`

All six V3 models were loaded successfully during registry generation. No
checkpoint was overwritten. The cleanup manifest records zero removals and
zero archives because no CSI duplicate was removed without a verified
replacement.

## 3. Figures and metrics

The original five figures were regenerated, and four additional figures were
created from the saved JSON metrics:

* N3 unseen-room comparison
* N4 new-setup transfer
* N2 model-family comparison
* Presence per-class metrics

The canonical CSV contains 168 flattened metric rows. It retains window-level
and smoothed values separately and derives per-class precision, recall, and F1
only from saved confusion matrices. Missing event-delay and confidence-
interval values remain unavailable rather than being estimated.

## 4. Technical issues found

* The ESP32 packet stream is irregular; 20 Hz is a resampling grid, not the
  native packet rate.
* Empty-bin values are interpolated, while a separate gap mask is preserved.
  The transformer consumes the mask; the CNN currently does not.
* Full raw I/Q records are required for defensible parsing and future phase
  work; amplitude-only lines are insufficient for the complete V3 contract.
* Calibrated paths require an explicit DataFrame/NumPy index-alignment test
  after filtering excluded recordings before new ablations are trusted.
* Five-window smoothing changes response delay, which historical JSON files do
  not measure as an event-level metric.

## 5. V4 implemented

V4 is separate and does not replace V3. It contains:

* `V4/configs/homeoccupancy.json`
* `V4/manifests/datasets.json`
* `V4/manifests/experiment_index.json`
* `V4/docs/V4_PLAN.md`, `DATASETS.md`, and `EXPERIMENTS.md`
* `V4/src/finetune_homeoccupancy.py`
* `V4/src/build_manifest.py`
* `V4/tests/test_protocol.py`

HomeOccupancy is downloaded on demand, parsed from raw ESP32-C6 I/Q CSV,
cleaned for malformed/incomplete rows, timestamp-unwrapped, converted to the
V3 representation, and split by its documented temporal file holdout.

## 6. Experiments actually run

The one-epoch V4 smoke runs completed end to end:

| Run | Test windows | Accuracy | Interpretation |
| --- | ---: | ---: | --- |
| V4 transformer smoke | 12,072 | 0.497 | Pipeline validation only |
| V4 CNN smoke | 12,072 | 0.525 | Pipeline validation only |

These are not final V4 performance claims. Multi-epoch training remains
pending and is intentionally not represented as completed.

## 7. Not available or not claimed

WiFi-Wild access was not available in the local repository and was not claimed
as downloaded or integrated. HomeOccupancy does not establish a controlled
congested-channel experiment, so it is not substituted for V3 N2 evidence.

## 8. Continue from here

```powershell
py -3.12 V3/src/plots.py
py -3.12 V3/src/dashboard_plots.py
py -3.12 V3/src/build_registry.py
py -3.12 V3/src/audit_artifacts.py
py -3.12 -m unittest discover -s V4/tests -v
py -3.12 V4/src/finetune_homeoccupancy.py --models tf cnn --epochs 10
py -3.12 V4/src/build_manifest.py
```

Before any larger V4 claim, run the grouped feature ablations in the protocol
audit, add latency/detection-delay measurement, and validate the live receiver
record schema against the no-compromise protocol.
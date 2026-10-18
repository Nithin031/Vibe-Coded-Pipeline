# V3 research report

## Objective

V3 evaluates ESP32 CSI occupancy sensing under real room, hardware, and radio
conditions. It separates presence (empty/occupied) from motion state
(empty/static/moving), because an occupied but stationary person is not
equivalent to a strongly moving person.

## Data and protocols

The primary real-noise sources are ESPectre and the esp32-csi-pipeline data.
N1 is leave-one-room-out ESPectre evaluation. N2 trains on a quiet pipeline
recording and tests the same room on a measured congested channel. N3 trains
on pipeline room 1 and tests unseen room 2. N4 transfers from other ESP32 data
to the pipeline setup. N1c-N3c repeat the relevant protocols with a 30-second
empty-room calibration.

The common representation is a 20 Hz grid, 3-second causal window, 1-second
hop, 32 relative-amplitude bins, and a packet-gap mask. Results are reported
both before and after five-window median smoothing. The operational priority is
empty-room false alarms, not training accuracy.

## Models

* LightGBM uses engineered relative-CSI, temporal, spectral, periodicity,
  cross-subcarrier, and RSSI-dynamics features.
* CNN uses the `[32, 60]` relative-amplitude sequence.
* Transformer uses the sequence and gap mask, with masked-reconstruction
  initialization from WiGesture/WiFall.

The final raw-input presence and motion files are in `V3/models/`. Their exact
paths, sizes, checksums, class mappings, and load-test status are in
[models.json](../manifests/models.json). Historical evaluation-to-model mapping
is in [experiment_index.json](../manifests/experiment_index.json).

## Verified results

The source of truth is [v3_metrics.csv](../results/metrics/v3_metrics.csv),
generated directly from the saved JSON result files. The headline N3 presence
comparison is approximately 0.47 balanced accuracy without calibration versus
0.72 with calibration for the saved ensemble scores; empty-room false alarms
fall from 100% to approximately 10%. These values are held-out N3 protocol
results, not scores of the final full-pool deployment checkpoints.

The figures are indexed in [figure_index.json](../results/reports/figure_index.json):

* `fig1_presence_raw_vs_cal.png`: presence balanced accuracy.
* `fig2_false_alarms.png`: empty-room false-alarm rate.
* `fig3_congested_channel.png`: N2 confusion matrices.
* `fig4_espectre_rooms.png`: held-out-room comparison.
* `fig5_empty_vs_occupied.png`: fluctuation distributions.

The implementation cannot currently recover every requested metric from the
historical JSONs, such as event-level detection delay or independent session
confidence intervals. The canonical table records only metrics supported by
the saved confusion matrices and score objects rather than estimating missing
values.

## Reproduction

```powershell
py -3.12 V3/src/plots.py
py -3.12 V3/src/build_registry.py
py -3.12 V3/src/audit_artifacts.py
```

To retrain final raw-input artifacts, use
`py -3.12 V3/src/train_final_v3.py --epochs 30`. This writes the final V3
files and model card; do not use it as a replacement for the historical
held-out evaluation results.

## Limitations

The number of independent rooms and hardware environments is small. Windows
overlap, labels are recording-level, OpenCSI has incompatible burst timing,
and the current CNN ignores the gap mask. Phase/complex CSI has not been
validated. See [LIMITATIONS.md](LIMITATIONS.md) and the full
[protocol/feature audit](protocol_feature_audit.md).
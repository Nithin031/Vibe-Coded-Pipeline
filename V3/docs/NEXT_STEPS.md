# V3/V4 next steps

## Preserve V3

Do not overwrite V3 JSON results, figures, or model files. Use
[models.json](../manifests/models.json) and
[checksums.json](../manifests/checksums.json) as the baseline record.

## Correctness checks first

1. Test raw `CSI_DATA` parsing, timestamp wrap, malformed packets, stable CSI
   lengths, and gap-mask shapes.
2. Assert DataFrame/NumPy positional alignment after excluded-row filtering in
   calibrated experiment paths.
3. Add event-level smoothing delay and packet-quality metrics.
4. Confirm the actual receiver output matches the full V3 parser contract.

## Feature ablations

Run recording-grouped comparisons for sequence only; sequence plus gap mask;
robust temporal statistics; PCA/correlation shape; spectrum/periodicity; RSSI;
and phase only after phase calibration. Select on training groups and evaluate
once on locked N1-N4 protocols. Prioritize empty-room false alarms and
stationary-person recall.

## V4 and additional data

HomeOccupancy is available through the V4 on-demand loader for empty/sleep/work
temporal adaptation. It is not a measured congestion dataset. WiFi-Wild is not
available locally and must not be claimed as integrated without verified access
and format.

Run controlled V4 fine-tuning only after reviewing the temporal split:

```powershell
py -3.12 V4/src/finetune_homeoccupancy.py --models tf cnn --epochs 10
py -3.12 V4/src/build_manifest.py
```

## Hardware validation

Build and flash the two-board transmitter/receiver setup, record one raw line,
verify timestamps/RSSI/IQ/length, perform the stream quality check, collect a
30-second empty reference, then measure one-second prediction cadence and
end-to-end latency. Do not claim live deployment until the serial adapter,
calibration state, and degraded-stream behavior are tested.
# V3 limitations and known issues

* The final deployment checkpoints were trained on the full ESPectre/pipeline
  pool after the held-out evaluation runs. They are not the exact checkpoints
  that generated N1-N4 scores.
* The CNN receives interpolated values but does not consume the gap mask;
  transformer gap handling is therefore the more faithful missing-packet path.
* The resampler assumes non-negative elapsed timestamps. Timestamp unwrap and
  reset tests are required for live hardware.
* Calibrated experiment paths should assert DataFrame/NumPy positional index
  alignment after excluded-row filtering before new ablations are trusted.
* Five-window smoothing changes response latency; historical JSONs do not store
  event-level detection delay.
* Empty-room background variation overlaps stationary-person variation, so
  total fluctuation magnitude is not a universal occupancy discriminator.
* The available data is not sufficient to claim universal cross-building,
  cross-chip, or high-load interference robustness.
* Phase, complex CSI, synthetic noise, and HomeOccupancy adaptation require
  separate experiments and must not be merged into V3 claims.
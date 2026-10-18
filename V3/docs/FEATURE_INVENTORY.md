# V3 feature inventory and ablation design

| Group | Existing implementation | Task value | Decision |
| --- | --- | --- | --- |
| Relative amplitude | Per-bin window-median normalization; 32-bin sequence | Baseline for both tasks | Required baseline |
| Robust temporal motion | Std, robust std, first-difference energy, range, quantiles | Strong motion and presence candidate | First ablation |
| PCA/cross-subcarrier shape | Eigen-spectrum, effective rank, participation, adjacent/cross-bin correlation | Candidate for room-transfer robustness | First ablation |
| Temporal spectrum | Band powers, centroid, entropy, ACF | Candidate for slow/fast and periodic background | Second ablation |
| Periodicity | Spectral peak share, flatness, periodic ACF | Candidate for fans/AC/non-human motion | Optional ablation |
| RSSI dynamics | Std, range, absolute first difference | Secondary radio cue; not absolute RSSI | Isolate and test for shortcuts |
| Gap metadata | Missing mask and missing ratio | Essential interpretation under packet loss | Keep; CNN integration remains open |
| Channel/chip/session IDs | Metadata | Distribution-shift diagnostics, shortcut risk | Do not train by default |
| Raw I/Q/phase | Available only when full records are retained | Potential fine-motion cue, hardware-sensitive | Defer until phase calibration |

Recommended locked comparisons are: sequence only; sequence plus mask;
motion-only; motion plus PCA/correlation; motion plus spectrum/periodicity;
each of those plus RSSI; and phase only after a phase-quality test. Select on
recording-grouped training validation, then evaluate once on N1-N4/N1c-N3c.

See [protocol_feature_audit.md](protocol_feature_audit.md) for the full
feature interpretation, runtime cost, noise sensitivity, leakage controls, and
acceptance criteria.
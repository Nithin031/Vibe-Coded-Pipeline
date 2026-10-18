# V3 real-noise occupancy baseline

V3 is the frozen baseline for real ESP32 CSI occupancy and motion-state
experiments. Existing models and historical JSON results remain in their
original paths; additive manifests make them discoverable without breaking
code references.

## Start here

* [V3_HANDOVER.md](docs/V3_HANDOVER.md): complete collaborator onboarding guide.
* [V3_HANDOVER.pdf](results/reports/V3_HANDOVER.pdf): printable project summary.
* [QUICKSTART.md](docs/QUICKSTART.md): shortest working command guide.
* [MODEL_AND_RESULTS_INDEX.md](docs/MODEL_AND_RESULTS_INDEX.md): checkpoint and result map.
* [NEXT_STEPS.md](docs/NEXT_STEPS.md): prioritized continuation plan.
* [V3_REPORT.md](docs/V3_REPORT.md): research report and verified results.
* [PROTOCOL.md](docs/PROTOCOL.md): acquisition and preprocessing contract.
* [FEATURE_INVENTORY.md](docs/FEATURE_INVENTORY.md): feature groups and ablation plan.
* [LIMITATIONS.md](docs/LIMITATIONS.md): known issues and claims not supported.
* [protocol_feature_audit.md](docs/protocol_feature_audit.md): implementation audit.
* [models.json](manifests/models.json): checkpoint registry and load tests.
* [experiment_index.json](manifests/experiment_index.json): result registry.
* [v3_metrics.csv](results/metrics/v3_metrics.csv): canonical flattened metrics.

## Reproduce or inspect

```powershell
py -3.12 V3/src/plots.py
py -3.12 V3/src/build_registry.py
py -3.12 V3/src/audit_artifacts.py
py -3.12 V3/src/show_transformer.py --task presence
```

The final deployment artifacts are under `V3/models/`. The held-out N1-N4 and
N1c-N3c result files are under `V3/results/`; they are historical evidence and
are not overwritten by V4.
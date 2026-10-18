# V3 model and results index

## Checkpoints

| Task | Family | Artifact | Input/classes | Load instruction | Status |
| --- | --- | --- | --- | --- | --- |
| Presence | LightGBM | `models/presence_lgbm.txt` | engineered features / 2 | `lgb.Booster(model_file=...)` | load-tested; final full-pool artifact |
| Presence | CNN | `models/presence_cnn.pt` | `[32,60]` / 2 | `deep.make_model("cnn", n_classes=2)` | load-tested; final full-pool artifact |
| Presence | Transformer | `models/presence_tf.pt` | `[32,60]` + `[60]` / 2 | `deep.make_model("tf", n_classes=2)` | load-tested; final full-pool artifact |
| Motion | LightGBM | `models/motion_lgbm.txt` | engineered features / 3 | `lgb.Booster(model_file=...)` | load-tested; final full-pool artifact |
| Motion | CNN | `models/motion_cnn.pt` | `[32,60]` / 3 | `deep.make_model("cnn", n_classes=3)` | load-tested; final full-pool artifact |
| Motion | Transformer | `models/motion_tf.pt` | `[32,60]` + `[60]` / 3 | `deep.make_model("tf", n_classes=3)` | load-tested; final full-pool artifact |

Exact sizes and SHA-256 values are in [manifests/models.json](../manifests/models.json).
The `.pt` files are PyTorch state dictionaries. Final files were trained after
the historical held-out runs and are not the exact evaluation checkpoints.

## Approved runtime files

For running a V3 occupancy/motion model, use only these six classifier files:

```text
models/presence_cnn.pt
models/presence_tf.pt
models/motion_cnn.pt
models/motion_tf.pt
models/presence_lgbm.txt
models/motion_lgbm.txt
```

`models/transformer_pretrained.pt` is also present, but it is a
self-supervised transformer initialization backbone only. It has no final
presence or motion head and must not be used as a standalone prediction model.
V2 checkpoints under `V2/models/` and V4 HomeOccupancy checkpoints under
`V4/models/` are separate-version artifacts and are not V3 runtime models.

## Result mapping

The historical JSONs evaluate model variants using these names:

| Runtime artifact | JSON variant | Task | Relevant result files |
| --- | --- | --- | --- |
| `presence_lgbm.txt` | `lgbm` | Presence | N1, N2, N3, N4 and calibrated counterparts where present |
| `presence_cnn.pt` | `cnn` | Presence | N1, N2, N3, N4 and calibrated counterparts where present |
| `presence_tf.pt` | `tf` | Presence | N1, N2, N3, N4 and calibrated counterparts where present |
| `motion_lgbm.txt` | `lgbm` | Motion | N1, N2, N3 and calibrated counterparts where present |
| `motion_cnn.pt` | `cnn` | Motion | N1, N2, N3 and calibrated counterparts where present |
| `motion_tf.pt` | `tf` | Motion | N1, N2, N3 and calibrated counterparts where present |

The `ensemble` rows in the JSONs are probability averages produced during the
historical evaluation run; they do not correspond to a separate `.pt` file.
Calibration is an evaluation/runtime transformation, not a second set of
classifier weights.

## Historical result files

| Protocol | File | Evaluation |
| --- | --- | --- |
| N1 | `results/N1_espectre_leave_room_out.json` | Held-out ESPectre room |
| N1c | `results/N1c_espectre_leave_room_out_calibrated.json` | N1 plus 30-second calibration |
| N2 | `results/N2_pipeline_quiet_to_congested.json` | Quiet training to measured congested channel |
| N2c | `results/N2c_pipeline_quiet_to_congested_calibrated.json` | N2 plus calibration |
| N3 | `results/N3_pipeline_room1_to_room2.json` | Room 1 to unseen room 2 |
| N3c | `results/N3c_pipeline_room1_to_room2_calibrated.json` | N3 plus calibration |
| N4 | `results/N4_all_to_pipeline.json` | Other ESP32 data to pipeline setup |

Each score contains `window` and `smoothed` results. The canonical
[v3_metrics.csv](../results/metrics/v3_metrics.csv) derives per-class precision,
recall, and F1 from saved confusion matrices where possible.

## Figures

See [figure_index.json](../results/reports/figure_index.json) for checksums and
descriptions. Figures 1-5 are the original V3 views; figures 6-9 add N3, N4,
model-family, and per-class views.
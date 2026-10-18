# V4: robust ESP32 Wi-Fi presence detection and people counting

V4 builds on the frozen V3 baseline (`../V3`, never modified) and adds:

- a **fixed data pipeline**: corrupted timestamps removed before the clock is
  rebuilt, signed-clock files repaired, long packet gaps marked "unknown"
  instead of being filled with fake values;
- **presence models** trained and selected without touching test data, compared
  fairly with V3;
- **people counting** (0–5 output; real ESP32 data only exists for 0–3);
- a **"crazy noise" stress test** (8 synthetic noise types x 3 levels) and a
  robustness upgrade chosen on validation data;
- tests, manifests, figures and a plain-English report.

**Read first: [docs/REPORT.md](docs/REPORT.md).**

## Folders

| Folder | What is inside |
|---|---|
| `configs/` | `v4_protocol.json`: splits, quality rules, noise and targets, all fixed before training. `recording_manifest_schema.json`: what to write down for every new recording. |
| `src/` | all code (each script has `--help`) |
| `tests/` | `test_v4.py`: 26 tests |
| `deploy/` | **the models to use** (`.pt` + LightGBM), `infer.py`, and what the ESP32 must send: see `deploy/README.md` |
| `models/` | every trained checkpoint: `presence/<variant>/`, `counting/<variant>/` |
| `results/metrics/<run_id>/` | one folder per run: `config.json` (exact settings) + `result.json` (numbers) |
| `results/figures/` | report figures + `captions.json` |
| `results/logs/v4.log` | log of every run |
| `manifests/` | `experiment_registry.json` (every run + checkpoint SHA-256), `dataset_search.json`, `removed_files.json`, `snapshot/` (V3 hashes and Git status at the start) |
| `docs/` | `REPORT.md`, `DATASET_AUDIT.md`, `COLLECTION_PROTOCOL.md` |
| `cache/` | processed windows (rebuildable, not source data) |

Raw datasets stay in `../datasets/` (see `docs/DATASET_AUDIT.md`).

## Code

| Script | Job |
|---|---|
| `src/data.py` | raw files -> clean timestamps -> labels -> splits -> 3 s windows with quality flags (cached) |
| `src/noise.py` | the 8 synthetic noise types and the training augmentation |
| `src/calibration.py` | empty-room calibration (reference = first 30 s of an empty recording of the same site and split) |
| `src/models.py` | LightGBM, plain CNN, gap-aware CNN, frozen V3 loaders |
| `src/metrics.py` | presence and counting metrics |
| `src/run_presence.py` | presence training + test (frozen V3 vs V4, raw vs calibrated) |
| `src/run_loso.py` | leave-one-source-out: the fair V3 vs V4 comparison |
| `src/run_noise.py` | noise stress test of saved presence models |
| `src/run_counting.py` | people counting (folds, cross-room, noise, latency) |
| `src/report.py` | all figures + `results/metrics/summary.json` |
| `src/validate_recording.py` | checks a new ESP32 recording and its manifest |
| `src/snapshot_repo.py` | V3 hash snapshot / check |

## Running V4 from the Git branch

Branch `v4` of the GitHub repo has the same layout as the local project: `V3/` (frozen
baseline, imported read-only) and `V4/` side by side. Raw datasets are not in Git: download
them (see `docs/DATASET_AUDIT.md`) into `datasets/` next to `V3/` and `V4/`, or point V4 at them:

```powershell
$env:CSI_DATASETS = "D:/path/to/datasets"            # folders: espectre, esp32_csi_pipeline, homeoccupancy, ...
$env:CSI_OTH_DIR  = "D:/path/to/esp32_reference"     # OTH counting samples (optional)
```

`V4/cache/` is rebuilt by `python V4/src/data.py --build all`.

## Reproduce everything

Python 3.12 with numpy, pandas, scipy, scikit-learn, lightgbm, torch, matplotlib,
huggingface_hub (versions are saved in every `config.json`). From the project root:

```powershell
$py = "python"          # Python 3.12
& $py V4/src/data.py --build all                          # windows for every source
& $py -m unittest discover -s V4/tests -v                  # tests
& $py V4/src/run_presence.py                               # presence, standard pipeline
& $py V4/src/run_presence.py --prep robust --augment       # presence, robust pipeline
& $py V4/src/run_noise.py --variant standard               # noise stress test
& $py V4/src/run_noise.py --variant robust_aug
& $py V4/src/run_loso.py                                   # fair V3 comparison
& $py V4/src/run_loso.py --prep robust --augment
& $py V4/src/run_counting.py                               # people counting
& $py V4/src/run_counting.py --prep robust --augment
& $py V4/src/report.py                                     # figures + summary
```

Robust-pipeline runs need `& $py V4/src/data.py --build all --prep robust` first.

## New recordings (needed for 4–5 people)

Follow [docs/COLLECTION_PROTOCOL.md](docs/COLLECTION_PROTOCOL.md), then:

```powershell
& $py V4/src/validate_recording.py path/to/*.csv
```

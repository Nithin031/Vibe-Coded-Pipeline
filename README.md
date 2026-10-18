# Wi-Fi CSI presence detection and people counting (ESP32): V4 branch

This branch holds the full V4 research code, next to the frozen V3 baseline it builds on.
The `main` branch holds only the ready-to-use models (`presence/`, `counting/`, `infer.py`).

| Folder | What it is |
|---|---|
| `V4/` | **V4**: fixed data pipeline, presence + people counting, noise stress tests, figures, tests. Start with [`V4/docs/REPORT.md`](V4/docs/REPORT.md) and [`V4/README.md`](V4/README.md). The models to use are in [`V4/deploy/`](V4/deploy/README.md). |
| `V3/` | the previous version, frozen (V4 imports its code read-only and checks its hashes) |

**Status:** V4 is experimental. It has been tested on public ESP32 recordings only, not on a live
ESP32 in the target room. Raw datasets are not included; see `V4/docs/DATASET_AUDIT.md` for
where to get them.

Quick check after cloning (Python 3.12, `pip install numpy pandas scipy scikit-learn lightgbm torch matplotlib huggingface_hub zstandard`):

```
python -m unittest discover -s V4/tests -v
```

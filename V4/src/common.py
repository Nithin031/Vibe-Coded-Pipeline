"""Shared V4 paths, protocol config, run IDs and result writing.

Every experiment writes to its own immutable folder:

    V4/results/metrics/<run_id>/result.json      numbers
    V4/results/metrics/<run_id>/config.json      protocol + arguments used
and appends one line to V4/manifests/experiment_registry.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Two layouts are supported:
#   project folder   ROOT/V3, ROOT/V4, ROOT/datasets
#   V3 Git repo      ROOT is the V3 repo itself, V4 lives in ROOT/V4 (branch 'v4')
V3 = ROOT / "V3" if (ROOT / "V3" / "src" / "deep.py").exists() else ROOT
V4 = ROOT / "V4"

# raw datasets are never in Git; point CSI_DATASETS at the folder that holds them
DATA = Path(os.environ.get("CSI_DATASETS", ROOT / "datasets"))
OTH_DIR = Path(os.environ.get("CSI_OTH_DIR", ROOT / "archive" / "datasets" / "esp32_reference"))
CACHE = V4 / "cache"
MODELS = V4 / "models"
METRICS = V4 / "results" / "metrics"
FIGURES = V4 / "results" / "figures"
LOGS = V4 / "results" / "logs"
REGISTRY = V4 / "manifests" / "experiment_registry.json"

# V3 is imported read-only; nothing in V3 is written
sys.path.insert(0, str(V3 / "src"))

PROTOCOL = json.loads((V4 / "configs" / "v4_protocol.json").read_text())


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def log(msg: str):
    """Print with a timestamp and append to V4/results/logs/v4.log."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / "v4.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def package_versions() -> dict:
    import lightgbm, numpy, pandas, sklearn, torch
    return {"python": platform.python_version(), "numpy": numpy.__version__, "pandas": pandas.__version__,
            "sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__, "torch": torch.__version__}


def to_jsonable(obj):
    """Convert numpy scalars / arrays inside nested dicts for json.dumps."""
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and obj != obj:          # NaN -> null
        return None
    return obj


# ---------------------------------------------------------------------------
# Immutable result folders + registry
# ---------------------------------------------------------------------------

def new_run(name: str, args: dict) -> Path:
    """Create V4/results/metrics/<name>_<timestamp>/ with config.json; refuse to reuse a folder."""
    run_id = f"{name}_{time.strftime('%Y%m%d-%H%M%S')}"
    folder = METRICS / run_id
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "config.json").write_text(json.dumps(
        {"run_id": run_id, "args": to_jsonable(args), "protocol": PROTOCOL,
         "packages": package_versions(), "command": " ".join(sys.argv)}, indent=2))
    return folder


def save_result(folder: Path, result: dict, models: list[Path] = (), status: str = "experimental"):
    """Write result.json and register the run (with checkpoint hashes)."""
    (folder / "result.json").write_text(json.dumps(to_jsonable(result), indent=2))

    entries = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else []
    entries.append({
        "run_id": folder.name,
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "status": status,
        "result": (folder / "result.json").relative_to(V4).as_posix(),
        "config": (folder / "config.json").relative_to(V4).as_posix(),
        "checkpoints": [{"file": Path(m).relative_to(V4).as_posix(), "sha256": sha256(m)} for m in models],
    })
    REGISTRY.write_text(json.dumps(entries, indent=2))
    log(f"saved {folder.relative_to(V4)}")

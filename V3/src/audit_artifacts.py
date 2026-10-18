"""Verify V3 final model files and map them to the reported evaluations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RESULTS = ROOT / "results"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    card = json.loads((MODELS / "model_card_v3.json").read_text())
    artifacts = {}
    for task, kinds in card["artifacts"].items():
        artifacts[task] = {}
        for kind, meta in kinds.items():
            path = MODELS / meta["file"]
            artifacts[task][kind] = {
                "file": meta["file"],
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
                "sha256_matches": path.exists() and sha256(path) == meta["sha256"],
            }

    evaluated = sorted(p.name for p in RESULTS.glob("N*.json"))
    manifest = {
        "version": "V3",
        "final_model_card": "models/model_card_v3.json",
        "final_artifacts": artifacts,
        "evaluation_result_files": evaluated,
        "mapping_note": (
            "N1-N4 and N1c-N3c JSON files contain held-out protocol evaluations. "
            "The final *_tf.pt, *_cnn.pt, and *_lgbm.txt files were trained after "
            "those evaluations on the full ESPectre + pipeline training pool; "
            "they are deployment artifacts, not the checkpoints used to produce "
            "the held-out scores."
        ),
        "figures": sorted(p.name for p in (RESULTS / "figures").glob("*.png")),
    }
    out = RESULTS / "v3_artifact_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
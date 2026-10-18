"""Build canonical V3 manifests from existing artifacts and result JSON files.

This script is intentionally additive: it does not move, overwrite, or delete
models, historical results, caches, or datasets.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import lightgbm as lgb
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
MANIFESTS = ROOT / "manifests"
METRICS = RESULTS / "metrics"
REPORTS = RESULTS / "reports"
FIGURES = RESULTS / "figures"
sys.path.insert(0, str(ROOT / "src"))
import deep  # noqa: E402


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def protocol(name: str) -> str:
    return name.split("_", 1)[0]


def flatten_results(obj, prefix=""):
    rows = []
    if isinstance(obj, dict) and "window" in obj and "smoothed" in obj:
        rows.append((prefix or "root", obj))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            rows.extend(flatten_results(value, f"{prefix}|{key}" if prefix else key))
    return rows


def class_metrics(confusion):
    cm = confusion
    out = {}
    total = sum(sum(row) for row in cm)
    for i, row in enumerate(cm):
        tp = row[i]
        support = sum(row)
        predicted = sum(cm[j][i] for j in range(len(cm)))
        out[f"class_{i}_precision"] = tp / predicted if predicted else None
        out[f"class_{i}_recall"] = tp / support if support else None
        p, r = out[f"class_{i}_precision"], out[f"class_{i}_recall"]
        out[f"class_{i}_f1"] = 2 * p * r / (p + r) if p is not None and r and p + r else None
        out[f"class_{i}_support"] = support
    out["n"] = total
    return out


def score_row(experiment, split, variant, smoothing, score):
    row = {"experiment": experiment, "split": split, "variant": variant,
           "smoothing": smoothing}
    row.update({k: v for k, v in score.items() if k != "confusion"})
    row.update(class_metrics(score["confusion"]))
    cm = score["confusion"]
    if len(cm) == 2:
        row["specificity"] = class_metrics(cm)["class_0_recall"]
        row["occupied_sensitivity"] = class_metrics(cm)["class_1_recall"]
    if len(cm) == 3:
        row["empty_specificity"] = class_metrics(cm)["class_0_recall"]
        row["stationary_recall"] = class_metrics(cm)["class_1_recall"]
        row["moving_recall"] = class_metrics(cm)["class_2_recall"]
    row["confusion"] = json.dumps(cm, separators=(",", ":"))
    return row


def model_registry(card):
    entries = []
    for task, kinds in card["artifacts"].items():
        n_classes = len(card["tasks"][task]["classes"])
        for family, meta in kinds.items():
            path = MODELS / meta["file"]
            loaded = False
            load_error = None
            try:
                if family == "lgbm":
                    lgb.Booster(model_file=str(path))
                else:
                    kind = "tf" if family == "tf" else "cnn"
                    state = torch.load(path, map_location="cpu")
                    model = deep.make_model(kind, n_classes=n_classes)
                    model.load_state_dict(state)
                loaded = True
            except Exception as exc:  # report, do not hide broken artifacts
                load_error = repr(exc)
            entries.append({
                "artifact": f"models/{path.name}", "family": family,
                "task": task, "classes": card["tasks"][task]["classes"],
                "input": card["input"], "training": card["training"],
                "calibration": "runtime/evaluation-time option; weights are raw",
                "smoothing": "5-window median for reported V3 noise scores",
                "bytes": path.stat().st_size if path.exists() else 0,
                "sha256": digest(path) if path.exists() else None,
                "exists": path.exists(), "load_tested": loaded,
                "load_error": load_error,
                "evaluation_variants": [family, "ensemble"],
            })
    pretrain = MODELS / "transformer_pretrained.pt"
    pretrain_loaded = False
    pretrain_error = None
    try:
        state = torch.load(pretrain, map_location="cpu")
        model = deep.CSITransformer()
        model.load_state_dict(state)
        pretrain_loaded = True
    except Exception as exc:
        pretrain_error = repr(exc)
    entries.append({
        "artifact": "models/transformer_pretrained.pt",
        "family": "transformer_backbone", "task": "pretraining_only",
        "classes": None, "input": card["input"],
        "training": {"source": "WiGesture + WiFall masked reconstruction"},
        "calibration": "not applicable", "smoothing": "not applicable",
        "bytes": pretrain.stat().st_size if pretrain.exists() else 0,
        "sha256": digest(pretrain) if pretrain.exists() else None,
        "exists": pretrain.exists(), "load_tested": pretrain_loaded,
        "load_error": pretrain_error,
        "runtime_approved": False,
        "note": "Initialization backbone only; it has no occupancy/motion classifier head.",
    })
    return entries


def main():
    MANIFESTS.mkdir(exist_ok=True)
    METRICS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    card = json.loads((MODELS / "model_card_v3.json").read_text())

    rows = []
    experiments = []
    result_files = sorted(RESULTS.glob("N*.json"))
    for path in result_files:
        data = json.loads(path.read_text())
        exp = path.stem
        experiments.append({
            "id": exp, "protocol": protocol(exp),
            "result_file": f"results/{path.name}",
            "calibrated": "calibrated" in exp.lower(),
            "smoothing": "5-window median in saved smoothed scores",
            "reproducibility": "saved result and log; final full-pool artifacts are separate",
            "log": "results/logs/run_noise_cal.out" if "c_" in exp else "results/logs/run_noise.out",
        })
        for split, result in flatten_results(data):
            variant = split.rsplit("|", 1)[-1] if "|" in split else split
            base = split.rsplit("|", 1)[0] if "|" in split else "root"
            for smooth, score in (("window", result["window"]), ("smoothed", result["smoothed"])):
                rows.append(score_row(exp, base, variant, smooth, score))

    metric_path = METRICS / "v3_metrics.csv"
    fields = sorted({key for row in rows for key in row})
    with metric_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    figure_index = []
    descriptions = {
        "fig1_presence_raw_vs_cal.png": "Presence balanced accuracy, raw versus 30 s calibration.",
        "fig2_false_alarms.png": "Empty-room false-alarm rate, raw versus calibration.",
        "fig3_congested_channel.png": "N2 same-room quiet-to-congested confusion matrices.",
        "fig4_espectre_rooms.png": "N1 held-out ESPectre-room presence comparison.",
        "fig5_empty_vs_occupied.png": "Raw versus calibrated fluctuation distributions.",
        "fig6_n3_cross_room.png": "N3 unseen-room raw versus calibrated presence comparison.",
        "fig7_n4_cross_dataset.png": "N4 new-setup model-family transfer comparison.",
        "fig8_model_family_n2.png": "N2 model-family comparison on congested-channel presence.",
        "fig9_presence_per_class.png": "Mean smoothed empty specificity and occupied sensitivity across N1-N3.",
    }
    for path in sorted(FIGURES.glob("*.png")):
        figure_index.append({"file": f"results/figures/{path.name}",
                             "description": descriptions.get(path.name, "Generated V3 figure."),
                             "bytes": path.stat().st_size, "sha256": digest(path)})

    model_entries = model_registry(card)
    checksums = {}
    for path in [*(MODELS / card["artifacts"][task][kind]["file"]
                    for task in card["artifacts"] for kind in card["artifacts"][task]),
                 MODELS / "transformer_pretrained.pt",
                 *result_files, *FIGURES.glob("*.png")]:
        if path.exists():
            checksums[str(path.relative_to(ROOT))] = digest(path)

    datasets = [
        {"name": "espectre", "use": "N1/N1c and V3 final training", "condition": "real homes, rooms, chip variation"},
        {"name": "pipeline", "use": "N2/N2c/N3/N3c and V3 final training", "condition": "quiet and measured congested channel"},
        {"name": "wicount", "use": "N4 source/reference", "condition": "ESP32-S3 labelled counting"},
        {"name": "thesis", "use": "N4 source/reference", "condition": "ESP-NOW HT40 labelled counting"},
        {"name": "wigesture/wifall", "use": "transformer self-supervised pretraining", "condition": "unlabelled/labelled activity recordings"},
        {"name": "homeoccupancy", "use": "V4 separate fine-tuning", "condition": "ESP32-C6 home empty/sleep/work; no controlled congestion claim"},
    ]
    (MANIFESTS / "models.json").write_text(json.dumps({"version": "V3", "models": model_entries}, indent=2))
    (MANIFESTS / "experiment_index.json").write_text(json.dumps({"version": "V3", "experiments": experiments}, indent=2))
    (MANIFESTS / "datasets.json").write_text(json.dumps({"version": "V3", "datasets": datasets}, indent=2))
    (MANIFESTS / "checksums.json").write_text(json.dumps(checksums, indent=2))
    (REPORTS / "figure_index.json").write_text(json.dumps({"figures": figure_index}, indent=2))
    (REPORTS / "cleanup_manifest.json").write_text(json.dumps({
        "removed": [], "archived": [],
        "decision": "No files removed: the worktree contains unrelated user changes, and no CSI duplicate was removed without a verified replacement.",
        "preserved": ["raw datasets", "V3 checkpoints", "historical N*.json results", "generated figures"],
    }, indent=2))
    print(f"wrote {metric_path} ({len(rows)} metric rows)")
    print(f"verified {sum(entry['load_tested'] for entry in model_entries)}/{len(model_entries)} model artifacts")


if __name__ == "__main__":
    main()
"""Train and save deployable V3 presence and motion models.

The final models use labelled ESP32 recordings with measured background
conditions. Synthetic noise is intentionally not mixed into this training run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch

import deep
from build import feature_cols
from experiments_noise import LGB_PARAMS, MODELS, load, pretrained_state

ROOT = Path(__file__).resolve().parents[1]
SEED = 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_deep(kind, seq, gap, y, name, epochs, init_state=None, n_classes=4):
    net = deep.fit(kind, seq, gap, y, seed=SEED, epochs=epochs,
                   init_state=init_state, tag=f"FINAL {name}", n_classes=n_classes)
    path = MODELS / f"{name}_{kind}.pt"
    torch.save({key: value.detach().cpu() for key, value in net.state_dict().items()}, path)
    return path


def train_task(df, seq, gap, task, cols, epochs, init_state):
    y = df[task].to_numpy(dtype=int)
    n_classes = int(y.max()) + 1
    files = {}
    booster = lgb.LGBMClassifier(random_state=SEED, **LGB_PARAMS).fit(df[cols], y)
    path = MODELS / f"{task}_lgbm.txt"
    booster.booster_.save_model(str(path))
    files["lgbm"] = path
    files["cnn"] = save_deep("cnn", seq, gap, y, task, epochs, n_classes=n_classes)
    files["tf"] = save_deep("tf", seq, gap, y, task, epochs, init_state, n_classes=n_classes)
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["espectre", "pipeline"],
                        choices=["espectre", "pipeline", "rf_esp32"])
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    df, seq, gap = load(args.datasets)
    keep = df["role"] != "exclude"
    df, seq, gap = df.loc[keep].reset_index(drop=True), seq[keep], gap[keep]
    df["motion"] = df["motion_id"]
    cols = feature_cols(df)
    MODELS.mkdir(parents=True, exist_ok=True)
    init_state = pretrained_state()

    artifacts = {}
    for task in ["presence", "motion"]:
        artifacts[task] = train_task(df, seq, gap, task, cols, args.epochs, init_state)

    card = {
        "version": "V3",
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "datasets": args.datasets,
        "n_windows": int(len(df)),
        "tasks": {
            "presence": {"classes": ["empty", "occupied"]},
            "motion": {"classes": ["empty", "static", "moving"]},
        },
        "input": {"sample_rate_hz": 20, "window_s": 3, "hop_s": 1,
                  "subcarrier_bins": 32, "calibration": "optional runtime step"},
        "training": {"seed": SEED, "epochs": args.epochs,
                      "deep_augmentation": "gain, circular shift, packet dropout",
                      "transformer_initialization": "WiGesture + WiFall masked reconstruction"},
        "artifacts": {},
    }
    for task, files in artifacts.items():
        card["artifacts"][task] = {}
        for kind, path in files.items():
            card["artifacts"][task][kind] = {"file": path.name, "sha256": sha256(path)}
    (MODELS / "model_card_v3.json").write_text(json.dumps(card, indent=2))
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
"""Small offline smoke demo for a saved V3 transformer checkpoint.

Usage from the repository root:
    py -3.12 V3/src/show_transformer.py --task presence
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

import deep
from experiments_noise import load

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["presence", "motion"], default="presence")
    args = parser.parse_args()
    n_classes = 2 if args.task == "presence" else 3
    df, seq, gap = load(["pipeline"])
    state = torch.load(ROOT / "models" / f"{args.task}_tf.pt", map_location="cpu")
    model = deep.make_model("tf", n_classes=n_classes)
    model.load_state_dict(state)
    proba = deep.predict_proba(model, seq[:1], gap[:1])
    print(f"checkpoint: {args.task}_tf.pt")
    print(f"input: sequence={tuple(seq[:1].shape)} gap={tuple(gap[:1].shape)}")
    print(f"recording: {df.iloc[0]['recording']}")
    print(f"probabilities: {proba[0].round(4).tolist()}")
    print(f"predicted class: {int(proba[0].argmax())}")


if __name__ == "__main__":
    main()
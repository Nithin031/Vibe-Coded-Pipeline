"""Live training log: every line goes to the console AND to results/logs/train_live.log.

Follow it while training runs (PowerShell):
    Get-Content "D:\\WIFI Based CSI Project\\V2\\results\\logs\\train_live.log" -Wait -Tail 30
"""
from __future__ import annotations

import time
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[1] / "results" / "logs"
LIVE = LOG_DIR / "train_live.log"


def log(msg: str = "") -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LIVE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
